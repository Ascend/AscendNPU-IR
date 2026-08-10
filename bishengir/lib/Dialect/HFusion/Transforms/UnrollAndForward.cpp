//===--------------- UnrollAndForward.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unroll sibling scf.for chains sharing constant bounds, then forward
// transfer_write vectors to transfer_read consumers through the collapsed
// slice chain, eliminating the intermediate-tensor UB round-trip.
//
// Pseudocode:
//   before (3 sibling loops, shared bounds, producer→consumer on T):
//     for iv in 0..N step VL { write exp → insert T[iv] } yield T      // producer
//     for iv in 0..N step VL { s += read T[iv] } yield s               // consumer (reduction)
//     for iv in 0..N step VL { o = read T[iv] / s; store o }           // consumer
//   after:
//     // fully unrolled → straight-line, T collapsed:
//     %e0 = exp x[0:VL]; %e1 = exp x[VL:2VL]; %e2 = exp x[2VL:tail]
//     %s  = %e0 + %e1 + %e2                                           // no UB read
//     store x[0:VL] / %s; store x[VL:2VL] / %s; store x[2VL:tail] / %s

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_UNROLLANDFORWARD
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-unroll-and-forward"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

/// Resolve constant offsets from mixed OpFoldResult; nullopt if non-constant.
static std::optional<SmallVector<int64_t>>
getConstOffsets(ArrayRef<OpFoldResult> mixed) {
  SmallVector<int64_t> result;
  for (OpFoldResult ofr : mixed) {
    auto c = getConstantIntValue(ofr);
    if (!c)
      return std::nullopt;
    result.push_back(*c);
  }
  return result;
}

/// True when `forOp` contains no nested scf.for in its body.
static bool isInnermost(scf::ForOp forOp) {
  auto result = forOp.getBody()->walk(
      [](scf::ForOp) { return WalkResult::interrupt(); });
  return !result.wasInterrupted();
}

static std::optional<int64_t> getConstantTripCount(scf::ForOp forOp) {
  auto lb = getConstantIntValue(forOp.getLowerBound());
  auto ub = getConstantIntValue(forOp.getUpperBound());
  auto step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb)
    return std::nullopt;
  return (*ub - *lb + *step - 1) / *step;
}

static unsigned countBodyOps(scf::ForOp forOp) {
  unsigned count = 0;
  forOp.getBody()->walk([&](Operation *op) {
    if (!isa<scf::YieldOp>(op))
      count++;
  });
  return count;
}

static bool masksAreEqual(Value a, Value b) {
  return a && b && a == b && a.getType() == b.getType();
}

static bool isSplatZero(Value v) {
  return matchPattern(v, m_Zero()) || matchPattern(v, m_AnyZeroFloat());
}

/// Key for grouping sibling loops by (parentBlock, lb, ub, step).
/// Loops in different blocks are never siblings.
struct BoundsKey {
  Block *parentBlock;
  int64_t lb, ub, step;
  bool operator==(const BoundsKey &o) const {
    return parentBlock == o.parentBlock && lb == o.lb && ub == o.ub &&
           step == o.step;
  }
  bool operator<(const BoundsKey &o) const {
    return std::tie(parentBlock, lb, ub, step) <
           std::tie(o.parentBlock, o.lb, o.ub, o.step);
  }
};

/// A validated group of sibling scf.for loops that can be unrolled together,
/// with a producer→consumer relationship on an intermediate tensor.
struct UnrollableChain {
  SmallVector<scf::ForOp> loops;
  Value intermedTensor;
  scf::ForOp producer;
  SmallVector<scf::ForOp> consumers;

  size_t size() const { return loops.size(); }
};

/// True if `op` is an elementwise arithmetic op that can relay a masked write
/// (used by isMaskSafe to trace through elementwise chains).
static bool isRelayElementwiseOp(Operation *op) {
  return isa<arith::DivFOp, arith::MulFOp, arith::SubFOp, arith::AddFOp,
             arith::MaximumFOp, arith::MinimumFOp>(op);
}

/// Trace from `start` through single-use elementwise ops to a transfer_write
/// with the same mask as `rMask`. Returns true if every path ends in such a
/// write. `visited` prevents re-traversal of diamond-shaped elementwise DAGs.
static bool endsInSameMaskWrite(Operation *start, Value rMask,
                               DenseSet<Operation *> &visited) {
  if (!visited.insert(start).second)
    return true;
  if (auto w = dyn_cast<vector::TransferWriteOp>(start))
    return masksAreEqual(w.getMask(), rMask);
  if (!isRelayElementwiseOp(start))
    return false;
  for (Value res : start->getResults()) {
    SmallVector<Operation *, 4> users(res.getUsers().begin(),
                                     res.getUsers().end());
    if (users.empty() || users.size() > 1)
      return false;
    if (!endsInSameMaskWrite(users.front(), rMask, visited))
      return false;
  }
  return true;
}

/// Check that forwarding `write` to `read` is mask-safe: either both are
/// unmasked, or they share the same mask and the read's users either select
/// with a splat-zero fallback or relay through elementwise ops to a
/// same-mask write.
static bool isMaskSafe(vector::TransferWriteOp write,
                      vector::TransferReadOp read) {
  if (write.hasOutOfBoundsDim())
    return false;
  // Forwarding replaces the read with write.getVector(); on OOB/masked lanes
  // the read yields its padding while the write vector carries the producer's
  // elementwise(padding). That changes OOB lane values whenever the producer
  // is non-identity on the padding. Require splat-zero padding so the OOB
  // semantics survive forwarding.
  if ((read.getMask() || read.hasOutOfBoundsDim()) &&
      !isSplatZero(read.getPadding()))
    return false;
  Value wMask = write.getMask();
  Value rMask = read.getMask();
  if (!wMask && !rMask)
    return true;
  if (!masksAreEqual(wMask, rMask))
    return false;
  for (Operation *u : read->getUsers()) {
    if (auto sel = dyn_cast<arith::SelectOp>(u)) {
      if (sel.getCondition() == rMask &&
          sel.getTrueValue() == read.getResult() &&
          isSplatZero(sel.getFalseValue()))
        continue;
      return false;
    }
    if (!isRelayElementwiseOp(u))
      return false;
    DenseSet<Operation *> visited;
    if (!endsInSameMaskWrite(u, rMask, visited))
      return false;
  }
  return true;
}

/// Collect innermost scf.for loops with constant, bounded trip counts within
/// [2, maxUnroll] and body size ≤ maxBodyOps. These are candidates for
/// unrollable sibling chains.
static SmallVector<scf::ForOp>
collectInnermostLoops(func::FuncOp vfFunc, unsigned maxUnroll,
                      unsigned maxBodyOps) {
  SmallVector<scf::ForOp> innermost;
  vfFunc.getBody().walk([&](scf::ForOp forOp) {
    if (!isInnermost(forOp))
      return;
    auto trip = getConstantTripCount(forOp);
    if (!trip || countBodyOps(forOp) > maxBodyOps)
      return;
    uint64_t t = static_cast<uint64_t>(*trip);
    if (t < 2 || t > maxUnroll)
      return;
    innermost.push_back(forOp);
  });
  return innermost;
}

/// Group loops by (parentBlock, lb, ub, step).
static std::map<BoundsKey, SmallVector<scf::ForOp>>
groupByBounds(ArrayRef<scf::ForOp> loops) {
  std::map<BoundsKey, SmallVector<scf::ForOp>> groups;
  for (auto forOp : loops) {
    auto lb = *getConstantIntValue(forOp.getLowerBound());
    auto ub = *getConstantIntValue(forOp.getUpperBound());
    auto step = *getConstantIntValue(forOp.getStep());
    Block *block = forOp->getBlock();
    groups[{block, lb, ub, step}].push_back(forOp);
  }
  return groups;
}

/// Analyze a group of sibling scf.for loops sharing constant bounds to
/// determine if they form unrollable chain(s). Structural checks populate
/// UnrollableChain; the transform step consumes the result without re-walking.
struct ChainAnalyzer {
public:
  ChainAnalyzer(ArrayRef<scf::ForOp> group) : group(group) {}

  /// Analyze the group. Returns one UnrollableChain per valid (producer, T)
  /// pair, or empty if the group cannot form a chain.
  SmallVector<UnrollableChain> analyze() const;

private:
  ArrayRef<scf::ForOp> group;

  /// True if loops are contiguous siblings (no intervening scf.for between
  /// consecutive loops in the same block).
  bool isContiguous() const;

  /// True if InsertSlice offsets across loops are pairwise disjoint (no two
  /// loops write to the same tensor region).
  bool hasDisjointWrites() const;

  /// Analyze one (producer, T) pair in a single pass over T.getUsers():
  ///   1. Purity: every user of T must be inside a chain loop, and T must
  ///      not be returned by the enclosing func.
  ///   2. Read path: each extract_slice of T must have constant-or-block-arg
  ///      offsets, unit strides, and feed a transfer_read whose users are
  ///      arith/vector ops or masked selects with constant false-value.
  ///   3. Consumer collection: the enclosing loop of each validated read is
  ///      recorded as a consumer (must be after producer in the same block).
  /// Returns the populated UnrollableChain on success, nullopt on any failure.
  std::optional<UnrollableChain> analyzeChain(scf::ForOp producer,
                                              Value T) const;

  /// Validate the producer's write path to T: each transfer_write in the
  /// producer must flow through a unit-stride insert_slice to the yield,
  /// with no call/if interference, and the written vector must not be a
  /// raw load.
  bool hasSimpleWritePath(scf::ForOp producer, Value T) const;
};

bool ChainAnalyzer::isContiguous() const {
  for (size_t i = 0; i + 1 < group.size(); ++i)
    for (Operation *op = group[i]->getNextNode(); op && op != group[i + 1];
         op = op->getNextNode())
      if (isa<scf::ForOp, scf::IfOp, func::CallOp>(op))
        return false;
  return true;
}

bool ChainAnalyzer::hasDisjointWrites() const {
  // Key by full (offsets, sizes) tuple: two inserts overlap only if they
  // target the same tensor region. Using offsets->back() alone misses
  // multi-dim conflicts.
  struct RegionKey {
    SmallVector<int64_t> offsets, sizes;
    bool operator==(const RegionKey &o) const {
      return offsets == o.offsets && sizes == o.sizes;
    }
  };
  struct RegionKeyInfo {
    static RegionKey getEmptyKey() {
      return {{}, {}};
    }
    static RegionKey getTombstoneKey() {
      return {{-1}, {}};
    }
    static unsigned getHashValue(const RegionKey &k) {
      return llvm::hash_combine(llvm::hash_combine_range(k.offsets.begin(),
                                                         k.offsets.end()),
                                llvm::hash_combine_range(k.sizes.begin(),
                                                         k.sizes.end()));
    }
    static bool isEqual(const RegionKey &a, const RegionKey &b) {
      return a == b;
    }
  };
  DenseMap<RegionKey, scf::ForOp, RegionKeyInfo> sliceWriters;
  bool overlap = false;
  for (auto loop : group)
    loop.getBody()->walk([&](tensor::InsertSliceOp ins) {
      auto offsets = getConstOffsets(ins.getMixedOffsets());
      auto sizes = getConstOffsets(ins.getMixedSizes());
      if (!offsets || !sizes || offsets->empty())
        return;
      RegionKey key{*offsets, *sizes};
      auto it = sliceWriters.find(key);
      if (it != sliceWriters.end() && it->second != loop) {
        overlap = true;
        return;
      }
      sliceWriters[key] = loop;
    });
  return !overlap;
}

bool ChainAnalyzer::hasSimpleWritePath(scf::ForOp producer, Value T) const {
  bool ok = true;
  producer.getBody()->walk([&](vector::TransferWriteOp write) {
    if (!ok)
      return WalkResult::interrupt();
    Value written = write.getResult();
    bool toYield = false;
    bool hasCallOrIf = false;
    for (Operation *user : written.getUsers()) {
      if (isa<func::CallOp, scf::IfOp>(user)) {
        hasCallOrIf = true;
        continue;
      }
      if (auto ins = dyn_cast<tensor::InsertSliceOp>(user)) {
        if (!ins.hasUnitStride()) {
          ok = false;
          return WalkResult::interrupt();
        }
        for (Operation *u2 : user->getResults()[0].getUsers())
          if (isa<scf::YieldOp>(u2)) {
            toYield = true;
            break;
          }
      }
    }
    if (hasCallOrIf || !toYield) {
      ok = false;
      return WalkResult::interrupt();
    }
    auto vt = write.getVectorType();
    unsigned elemBits = vt.getElementType().getIntOrFloatBitWidth();
    if (vt.getNumElements() * elemBits != util::VL * 8) {
      ok = false;
      return WalkResult::interrupt();
    }
    Value vec = write.getVector();
    if (!vec.getDefiningOp() || isa<memref::LoadOp>(vec.getDefiningOp())) {
      ok = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return ok;
}

std::optional<UnrollableChain>
ChainAnalyzer::analyzeChain(scf::ForOp producer, Value T) const {
  if (!hasSimpleWritePath(producer, T))
    return std::nullopt;

  auto collectConsumers =
      [&]() -> std::optional<SmallVector<scf::ForOp>> {
    SmallVector<scf::ForOp> consumers;
    DenseSet<scf::ForOp> consumerSet;
    for (Operation *user : T.getUsers()) {
      Operation *ancestor = user->getParentOp();
      while (ancestor && !isa<scf::ForOp>(ancestor))
        ancestor = ancestor->getParentOp();
      if (!ancestor || !llvm::is_contained(group, cast<scf::ForOp>(ancestor)))
        return std::nullopt;
      auto enclosingLoop = cast<scf::ForOp>(ancestor);

      auto extract = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!extract)
        continue;
      for (OpFoldResult ofr : extract.getMixedOffsets())
        if (auto val = dyn_cast<Value>(ofr))
          if (!getConstantIntValue(val) && !isa<BlockArgument>(val))
            return std::nullopt;
      if (!extract.hasUnitStride())
        return std::nullopt;
      for (Operation *u2 : extract.getResult().getUsers()) {
        if (!isa<vector::TransferReadOp>(u2))
          return std::nullopt;
        // Reject chains whose consumer read uses non-zero padding: forwarding
        // rewrites the read to the producer's write vector, whose OOB lanes
        // carry elementwise(padding) rather than padding itself.
        auto readOp = cast<vector::TransferReadOp>(u2);
        if ((readOp.getMask() || readOp.hasOutOfBoundsDim()) &&
            !isSplatZero(readOp.getPadding()))
          return std::nullopt;
        for (Operation *u3 : u2->getUsers()) {
          if (auto sel = dyn_cast<arith::SelectOp>(u3)) {
            if (!isSplatZero(sel.getFalseValue()))
              return std::nullopt;
            continue;
          }
          if (!isa<arith::ArithDialect>(u3->getDialect()) &&
              !isa<vector::VectorDialect>(u3->getDialect()))
            return std::nullopt;
        }
      }

      if (enclosingLoop != producer &&
          producer->getBlock() == enclosingLoop->getBlock() &&
          producer->isBeforeInBlock(enclosingLoop) &&
          consumerSet.insert(enclosingLoop).second)
        consumers.push_back(enclosingLoop);
    }
    return consumers;
  };

  auto consumers = collectConsumers();
  if (!consumers || consumers->empty())
    return std::nullopt;

  auto escapesViaReturn = [&]() -> bool {
    auto funcOp = group.front()->getParentOfType<func::FuncOp>();
    if (!funcOp)
      return false;
    bool returned = false;
    funcOp.walk([&](func::ReturnOp ret) {
      for (Value retVal : ret.getOperands())
        if (retVal == T) {
          returned = true;
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    });
    return returned;
  };
  if (escapesViaReturn())
    return std::nullopt;

  return UnrollableChain{SmallVector<scf::ForOp>(group), T, producer,
                         std::move(*consumers)};
}

SmallVector<UnrollableChain> ChainAnalyzer::analyze() const {
  SmallVector<UnrollableChain> chains;
  if (group.size() < 2 || !isContiguous() || !hasDisjointWrites())
    return chains;
  for (auto P : group)
    for (Value T : P.getResults()) {
      auto chain = analyzeChain(P, T);
      if (chain)
        chains.push_back(*chain);
    }
  return chains;
}

static SmallVector<UnrollableChain>
matchUnrollableChains(func::FuncOp vfFunc, unsigned maxUnroll,
                      unsigned maxBodyOps) {
  SmallVector<UnrollableChain> chains;
  auto innermost = collectInnermostLoops(vfFunc, maxUnroll, maxBodyOps);
  if (innermost.size() < 2)
    return {};
  auto groups = groupByBounds(innermost);
  for (auto &[key, group] : groups) {
    if (group.size() < 2)
      continue;
    ChainAnalyzer analyzer(group);
    auto found = analyzer.analyze();
    chains.append(found);
  }
  return chains;
}

/// Forward a transfer_read through a collapsed extract_slice/insert_slice
/// chain to the originating transfer_write vector.
///
/// Before:
///   %w   = vector.transfer_write %v, %sub[%c0] : vector<4xf32>, tensor<4xf32>
///   %ins = tensor.insert_slice %w into %T[%off] : tensor<4xf32> into tensor<16xf32>
///   %ext = tensor.extract_slice %ins[%off]      : tensor<16xf32> to tensor<4xf32>
///   %r   = vector.transfer_read %ext[%c0]       : tensor<4xf32>, vector<4xf32>
/// After (read replaced by write vector; slice chain left for DCE):
///   %r = %v
struct ForwardTransferReadThroughSliceChain
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (readOp.hasOutOfBoundsDim() ||
        !isa<RankedTensorType>(readOp.getShapedType()))
      return failure();
    // vector.transfer aligns vector shape to trailing tensor dims via the
    // permutation map. When vector rank != tensor rank, the leading tensor
    // dims are collapsed and readVecSize indexing is ambiguous — bail.
    if (readOp.getVectorType().getRank() !=
        readOp.getShapedType().getRank())
      return failure();

    SmallVector<int64_t> readIdx;
    for (Value idx : readOp.getIndices()) {
      auto c = getConstantIntValue(idx);
      if (!c)
        return failure();
      readIdx.push_back(*c);
    }
    SmallVector<int64_t> readVecSize(
        readOp.getVectorType().getShape().begin(),
        readOp.getVectorType().getShape().end());

    // Walk the extract_slice/insert_slice chain to find the root source
    // and accumulate the absolute offset of the read within it.
    // Each lambda returns optional<bool>: nullopt=stop, true=advance, false=fail.
    Value curSource = readOp.getSource();
    SmallVector<int64_t> absOffset = readIdx;

    auto walkExtract = [&](Value &src,
                           SmallVector<int64_t> &off) -> std::optional<bool> {
      auto extract = src.getDefiningOp<tensor::ExtractSliceOp>();
      if (!extract)
        return std::nullopt;
      auto offsets = getConstOffsets(extract.getMixedOffsets());
      if (!offsets || !extract.hasUnitStride())
        return std::nullopt;
      if (offsets->size() != off.size())
        return false;
      for (size_t i = 0; i < off.size(); ++i)
        off[i] += (*offsets)[i];
      src = extract.getSource();
      return true;
    };
    auto walkInsert = [&](Value &src,
                          SmallVector<int64_t> &off) -> std::optional<bool> {
      auto insert = src.getDefiningOp<tensor::InsertSliceOp>();
      if (!insert)
        return std::nullopt;
      auto offsets = getConstOffsets(insert.getMixedOffsets());
      auto sizes = getConstOffsets(insert.getMixedSizes());
      if (!offsets || !sizes || !insert.hasUnitStride())
        return std::nullopt;
      if (offsets->size() != off.size())
        return false;
      auto [covers, overlaps] =
          checkCoverage(off, readVecSize, *offsets, *sizes);
      if (overlaps)
        return false;
      if (covers) {
        for (size_t i = 0; i < off.size(); ++i)
          off[i] -= (*offsets)[i];
        src = insert.getSource();
      } else {
        src = insert.getDest();
      }
      return true;
    };

    while (true) {
      auto step = walkExtract(curSource, absOffset);
      if (!step)
        step = walkInsert(curSource, absOffset);
      if (!step)
        break;
      if (!*step)
        return failure();
    }

    // Validate the resolved write: index match, dominance, permutation, mask.
    auto validateWrite = [&](vector::TransferWriteOp w) -> bool {
      if (w.getIndices().size() != absOffset.size())
        return false;
      for (auto [idx, want] : llvm::zip(w.getIndices(), absOffset)) {
        auto c = getConstantIntValue(idx);
        if (!c || *c != want)
          return false;
      }
      if (w->getBlock() != readOp->getBlock() ||
          !w->isBeforeInBlock(readOp))
        return false;
      if (w.getPermutationMap() != readOp.getPermutationMap())
        return false;
      if (w.getVectorType() != readOp.getVectorType())
        return false;
      return isMaskSafe(w, readOp);
    };
    // Replace the read with the write vector. validateWrite already
    // guarantees vector types match, so no shape_cast is needed.
    auto replaceWithWrite = [&](vector::TransferWriteOp w) {
      rewriter.replaceOp(readOp, w.getVector());
    };

    auto writeOp = resolveWriteOp(curSource);
    if (!writeOp || !validateWrite(writeOp))
      return failure();
    replaceWithWrite(writeOp);
    return success();
  }

private:
  /// Resolve the transfer_write that produced `source`.
  static vector::TransferWriteOp resolveWriteOp(Value source) {
    return source.getDefiningOp<vector::TransferWriteOp>();
  }

  /// Check whether a read at [readOffset, readOffset+readSize) is covered by
  /// or overlaps with an insert at [insertOffset, insertOffset+insertSize).
  /// Returns {covers, overlaps}.
  static std::pair<bool, bool>
  checkCoverage(ArrayRef<int64_t> readOffset, ArrayRef<int64_t> readSize,
                ArrayRef<int64_t> insertOffset,
                ArrayRef<int64_t> insertSize) {
    bool covers = true, overlaps = false;
    for (size_t i = 0; i < readOffset.size(); ++i) {
      int64_t lo = insertOffset[i], hi = lo + insertSize[i];
      int64_t rs = i < readSize.size() ? readSize[i] : 1;
      int64_t rlo = readOffset[i], rhi = rlo + rs;
      if (rlo < lo || rhi > hi)
        covers = false;
      if (rlo < hi && rhi > lo && !(rlo >= lo && rhi <= hi))
        overlaps = true;
    }
    return {covers, overlaps};
  }
};

struct UnrollAndForwardPass
    : public impl::UnrollAndForwardBase<UnrollAndForwardPass> {
  using Base::Base;
  void runOnOperation() override;

private:
  /// Unroll a single chain: fully unroll all sibling loops, run a local
  /// canonicalizer+CSE cleanup to simplify the unrolled IR, then apply the
  /// ForwardTransferReadThroughSliceChain pattern to forward writes to reads.
  void unrollChain(func::FuncOp func, const UnrollableChain &chain);
};

} // namespace

void UnrollAndForwardPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(hivm::VectorFunctionAttr::name))
    return;

  auto chains = matchUnrollableChains(func, maxUnroll, maxBodyOps);
  LLVM_DEBUG(DBGS() << "chains=" << chains.size() << " for "
                    << func.getName() << "\n");
  // Conservative: only process one chain per VF. Multiple chains risk
  // excessive register pressure and complex cross-chain dependencies.
  if (chains.empty() || chains.size() > 1)
    return;
  // Limit chain size: softmax has 4 sibling loops (max/sub+exp/sum/div).
  // Larger chains explode IR after unroll.
  if (chains.front().size() > 4)
    return;

  unrollChain(func, chains.front());
}

void UnrollAndForwardPass::unrollChain(func::FuncOp func,
                                       const UnrollableChain &chain) {
  auto trip = *getConstantTripCount(chain.loops[0]);
  for (auto loop : chain.loops)
    if (failed(loopUnrollByFactor(loop, static_cast<uint64_t>(trip),
                                  nullptr))) {
      signalPassFailure();
      return;
    }

  // Canonicalize+CSE so ForwardTransferReadThroughSliceChain can match the
  // unrolled IR.
  {
    OpPassManager pm(func::FuncOp::getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(runPipeline(pm, func)))
      signalPassFailure();
  }

  RewritePatternSet patterns(func.getContext());
  patterns.add<ForwardTransferReadThroughSliceChain>(func.getContext());
  (void)applyPatternsGreedily(func, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hfusion::createUnrollAndForwardPass() {
  return std::make_unique<UnrollAndForwardPass>();
}

//===------------- LoopInvariantPromotion.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <functional>
#include <memory>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_LOOPINVARIANTPROMOTION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

struct SubsetInfo {
  SmallVector<SmallVector<OpFoldResult>> offsets, sizes, strides;
  SmallVector<Value> indices;
  VectorType vecTy;
  AffineMap permMap;
  Value mask, pad;
  ArrayAttr inBounds;

  bool operator!=(const SubsetInfo &rhs) {
    if (offsets.size() != rhs.offsets.size())
      return true;
    assert(sizes.size() == rhs.sizes.size() &&
           strides.size() == rhs.strides.size());
    auto isSameOFRs = [](ArrayRef<OpFoldResult> lhs,
                         ArrayRef<OpFoldResult> rhs) {
      return lhs.size() == rhs.size() && llvm::equal(lhs, rhs);
    };
    for (unsigned i = 0, e = offsets.size(); i != e; ++i) {
      if (!isSameOFRs(offsets[i], rhs.offsets[i]) ||
          !isSameOFRs(sizes[i], rhs.sizes[i]) ||
          !isSameOFRs(strides[i], rhs.strides[i]))
        return true;
    }
    // pad is not check here.
    return !llvm::equal(indices, rhs.indices) || vecTy != rhs.vecTy ||
           permMap != rhs.permMap || mask != rhs.mask ||
           inBounds != rhs.inBounds;
  }
};

struct Candidate {
  unsigned idx;
  SmallVector<vector::TransferReadOp> reads;
  SmallVector<Operation *> threadOps;
  Value yield;
  bool endedInWrite;
  SubsetInfo subset;
};

struct LoopInvariantPromotion
    : public impl::LoopInvariantPromotionBase<LoopInvariantPromotion> {
  void runOnOperation() override;
};

static bool isDefinedOutsideLoop(LoopLikeOpInterface loop, Value v) {
  return !v || loop.isDefinedOutsideOfLoop(v);
}

static SmallVector<tensor::ExtractSliceOp> extractChainOf(Value v) {
  SmallVector<tensor::ExtractSliceOp> chain;
  while (auto es = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    chain.push_back(es);
    v = es.getSource();
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

static std::optional<Candidate> analyze(scf::ForOp forOp, unsigned idx) {
  BlockArgument arg = forOp.getRegionIterArg(idx);
  if (!isa<RankedTensorType>(arg.getType())) // Only solve tensor type
    return std::nullopt;
  Candidate cand;
  cand.idx = idx;
  SubsetInfo &s = cand.subset;
  bool haveSubset = false;

  auto isInvariant = [&](ArrayRef<OpFoldResult> array) {
    for (auto ofr : array)
      if (auto v = dyn_cast<Value>(ofr))
        if (!isDefinedOutsideLoop(forOp, v))
          return false;
    return true;
  };
  auto match = [&](Value src, ValueRange indices, VectorType vecTy,
                   AffineMap map, Value mask, ArrayAttr inBounds, bool isRead,
                   Value pad) {
    for (Value i : indices)
      if (!isDefinedOutsideLoop(forOp, i))
        return false;
    if (!isDefinedOutsideLoop(forOp, mask) ||
        // For transfer_read we need to check padding.
        (isRead && !isDefinedOutsideLoop(forOp, pad)))
      return false;
    SmallVector<SmallVector<OpFoldResult>> offsets, sizes, strides;
    for (auto es : extractChainOf(src)) {
      auto offs = es.getMixedOffsets(), szs = es.getMixedSizes(),
           strs = es.getMixedStrides();
      if (!isInvariant(offs) || !isInvariant(szs) || !isInvariant(strs))
        return false;
      offsets.push_back(std::move(offs));
      sizes.push_back(std::move(szs));
      strides.push_back(std::move(strs));
    }
    SubsetInfo temp = {offsets, sizes, strides, indices, vecTy,
                       map,     mask,  pad,     inBounds};
    if (!haveSubset)
      s = temp, haveSubset = true;
    else if (s != temp)
      return false;
    if (isRead) {
      if (!s.pad)
        s.pad = pad;
      else if (s.pad != pad)
        return false;
    }
    return true;
  };

  DenseSet<Operation *> handled;
  DenseSet<Value> visited;
  SmallVector<Value> worklist{arg};
  auto record = [&](Operation *op) {
    if (handled.insert(op).second)
      cand.threadOps.push_back(op);
  };
  // 1. Check the legality of promotion and collect optimization instructions.
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    visited.insert(v);
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == forOp.getBody()->getTerminator()) {
        if (use.getOperandNumber() != idx)
          return std::nullopt;
        continue;
      }
      // A thread value used outside the loop's own body (e.g. captured into a
      // nested scf.for / scf.if) is not a straight-line access we can promote.
      if (owner->getBlock() != forOp.getBody())
        return std::nullopt;
      if (auto rd = dyn_cast<vector::TransferReadOp>(owner)) {
        assert(rd.getSource() == v);
        if (!match(v, rd.getIndices(), rd.getVectorType(),
                   rd.getPermutationMap(), rd.getMask(), rd.getInBoundsAttr(),
                   true, rd.getPadding()))
          return std::nullopt;
        cand.reads.push_back(rd);
        record(rd);
        // As the endpoint.
        continue;
      }
      if (auto wr = dyn_cast<vector::TransferWriteOp>(owner)) {
        assert(wr.getSource() == v && wr.getResult() != nullptr);
        if (!match(v, wr.getIndices(), wr.getVectorType(),
                   wr.getPermutationMap(), wr.getMask(), wr.getInBoundsAttr(),
                   false, {}))
          return std::nullopt;
        worklist.push_back(wr.getResult());
        record(wr);
        continue;
      }
      if (auto es = dyn_cast<tensor::ExtractSliceOp>(owner)) {
        assert(es.getSource() == v);
        worklist.push_back(es.getResult());
        record(es);
        continue;
      }
      if (auto is = dyn_cast<tensor::InsertSliceOp>(owner)) {
        if (is.getSource() == v) {
          if (!isInvariant(is.getMixedOffsets()) ||
              !isInvariant(is.getMixedSizes()) ||
              !isInvariant(is.getMixedStrides()))
            return std::nullopt;
          worklist.push_back(is.getResult());
          record(is);
          continue;
        }
        if (is.getDest() == v)
          continue;
        return std::nullopt; // ????
      }
      // Default.
      return std::nullopt;
    }
  }

  // 2. Verify the closure value.
  for (auto v : visited) {
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == forOp.getBody()->getTerminator() &&
          use.getOperandNumber() == idx)
        continue;
      // For example, %1 = insert_slice %2 into %3[0,0][1,8], and %2 is not in
      // the closure. We should conservatively exit optimization.
      if (!handled.count(owner))
        return std::nullopt;
    }
  }
  // Illiagle instruction arith.select(mask: vector<4x8xi1>, value:
  // vector<8x4xf32>, ...).
  if (cand.reads.empty() ||
      (s.mask &&
       cast<VectorType>(s.mask.getType()).getShape() != s.vecTy.getShape()))
    return std::nullopt;

  // 3. Classify the yield value.
  Value yield = forOp.getYieldedValues()[idx];
  cand.yield = yield;
  cand.endedInWrite = yield != arg;
  if (cand.endedInWrite) {
    if (!visited.count(yield))
      return std::nullopt;
    Value u = yield;
    while (auto ins = u.getDefiningOp<tensor::InsertSliceOp>())
      u = ins.getSource();
    if (!u.getDefiningOp<vector::TransferWriteOp>())
      return std::nullopt;
  }
  return cand;
}

static LogicalResult promote(IRRewriter &rewriter, scf::ForOp forOp,
                             Candidate cand) {
  auto &s = cand.subset;
  auto loc = forOp.getLoc();

  auto rootOf = [](Value v) {
    while (auto es = v.getDefiningOp<tensor::ExtractSliceOp>())
      v = es.getSource();
    return v;
  };
  auto cloneRead = [&](vector::TransferReadOp r, Value base) {
    IRMapping map;
    map.map(rootOf(r.getSource()), base);
    for (tensor::ExtractSliceOp es : extractChainOf(r.getSource()))
      rewriter.clone(*es.getOperation(), map);
    return rewriter.clone(*r.getOperation(), map);
  };

  // 1. Create hoist transfer_read.
  assert(!cand.reads.empty());
  rewriter.setInsertionPoint(forOp);
  auto hoistedRead = cast<vector::TransferReadOp>(
      cloneRead(cand.reads.front(), forOp.getInitArgs()[cand.idx]));
  Value padBrc;
  if (s.mask)
    padBrc = rewriter.create<vector::BroadcastOp>(loc, s.vecTy, s.pad);

  // For reads with padding, we need to legalize them.
  auto legalize = [&](Value r) -> Value {
    if (!s.mask)
      return r;
    return rewriter.create<arith::SelectOp>(loc, s.mask, r, padBrc);
  };
  auto removeDeadOp = [&rewriter](ArrayRef<Operation *> ops) {
    for (auto *op : reverse(ops))
      if (op->use_empty())
        rewriter.eraseOp(op);
  };
  auto mem2ssa = [&](Value regBase, Value regVal) {
    DenseMap<Value, Value> avail;
    avail[regBase] = regVal;
    std::function<Value(Value)> help = [&](Value v) -> Value {
      if (auto it = avail.find(v); it != avail.end())
        return it->second;
      Operation *d = v.getDefiningOp();
      Value r;
      if (auto es = dyn_cast_or_null<tensor::ExtractSliceOp>(d))
        r = help(es.getSource());
      else if (auto w = dyn_cast_or_null<vector::TransferWriteOp>(d))
        r = w.getVector();
      else if (auto ins = dyn_cast_or_null<tensor::InsertSliceOp>(d))
        r = help(ins.getSource());
      else
        r = nullptr;
      avail[v] = r;
      return r;
    };
    for (auto r : cand.reads) {
      Value v = help(r.getSource());
      assert(v != nullptr);
      rewriter.setInsertionPoint(r);
      rewriter.replaceAllUsesWith(r.getResult(), legalize(v));
    }
  };

  // 2. Rewrite if there is no write on the def-use chain.
  if (!cand.endedInWrite) {
    mem2ssa(forOp.getRegionIterArg(cand.idx), hoistedRead.getResult());
    removeDeadOp(cand.threadOps);
    rewriter.replaceAllUsesWith(forOp.getResult(cand.idx),
                                forOp.getInitArgs()[cand.idx]);
    return success();
  }

  // 3. Rewrite if there have write on the def-use chain.
  Value y = cand.yield;
  while (auto ins = y.getDefiningOp<tensor::InsertSliceOp>())
    y = ins.getSource();
  auto result = forOp.replaceWithAdditionalYields(
      rewriter, {hoistedRead.getResult()}, false,
      [&](OpBuilder &, Location,
          ArrayRef<BlockArgument>) -> SmallVector<Value> {
        return {cast<vector::TransferWriteOp>(y.getDefiningOp()).getVector()};
      });
  if (failed(result))
    return failure();
  auto newFor = cast<scf::ForOp>(result->getOperation());
  BlockArgument newArg = newFor.getRegionIterArgs().back(),
                idxArg = newFor.getRegionIterArg(cand.idx);
  mem2ssa(idxArg, newArg);
  Operation *yieldOp = cast<scf::YieldOp>(newFor.getBody()->getTerminator());
  // Canonicalize will remove it.
  rewriter.modifyOpInPlace(yieldOp,
                           [&] { yieldOp->setOperand(cand.idx, idxArg); });

  // Reconstruct the tensor outside the loop: write the final register value
  // (the added vector result) back into the init, and redirect the loop's
  // tensor result to it.
  auto cloneWrite = [&](Value yield, Value base, Value vec) {
    SmallVector<tensor::InsertSliceOp> inserts;
    Value u = yield;
    while (auto ins = u.getDefiningOp<tensor::InsertSliceOp>()) {
      inserts.push_back(ins);
      u = ins.getSource();
    }
    auto w = u.getDefiningOp<vector::TransferWriteOp>();
    assert(w != nullptr);
    IRMapping map;
    map.map(rootOf(w.getSource()), base);
    map.map(w.getVector(), vec);
    for (auto es : extractChainOf(w.getSource()))
      rewriter.clone(*es.getOperation(), map);
    Value res = rewriter.clone(*w.getOperation(), map)->getResult(0);
    for (auto ins : reverse(inserts))
      res = rewriter.clone(*ins.getOperation(), map)->getResult(0);
    return res;
  };
  rewriter.setInsertionPointAfter(newFor);
  Value vecRes = newFor->getResults().back();
  Value res = cloneWrite(cand.yield, newFor.getInitArgs()[cand.idx], vecRes);
  rewriter.replaceAllUsesWith(newFor.getResult(cand.idx), res);
  removeDeadOp(cand.threadOps);
  return success();
}

static void hoistLoopInvariant(LoopLikeOpInterface loop) {
  moveLoopInvariantCode(
      loop.getLoopRegions(),
      [&](Value v, Region *) { return loop.isDefinedOutsideOfLoop(v); },
      [&](Operation *op, Region *) {
        if (auto rd = dyn_cast<vector::TransferReadOp>(op))
          return isa<RankedTensorType>(rd.getShapedType());
        return isMemoryEffectFree(op) && isSpeculatable(op);
      },
      [&](Operation *op, Region *) { loop.moveOutOfLoop(op); });
}

} // namespace

//===----------------------------------------------------------------------===//
// Constant fill forwarding
//===----------------------------------------------------------------------===//
//
// AutoVectorizeV2 never fuses a `linalg.fill` into its consumer (see
// findBestFusedNodeForProducer), so OutlineVectorFunction gives the fill a
// vector function of its own, tagged `kFillVFAttrName` when the loop
// stamps nothing but constants. Its result then has to travel through UB to
// whoever reads it -- a splat of constants nobody needs to load:
//
//   %f = call @fill_vf(%empty)              // stamps `c` over %empty
//   %r = call @reader_vf(.., %f)
//   ...
//   %v = vector.transfer_read %arg[..]      // inside the reader, %arg <- %f
//
// A read rooted at a function argument observes exactly what the caller passed
// in, so `%v` can be built out of `c` instead. That is the whole condition:
// nothing about the filled tensor changes, so no coverage or use analysis is
// needed, and a read observing a running accumulator rather than the fill is
// rooted at the enclosing loop's iter_arg rather than at the argument, so it
// is left alone by construction.

static TypedAttr getSplatValue(Value v) {
  DenseElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)) || !attr.isSplat())
    return {};
  return dyn_cast<TypedAttr>(attr.getSplatValue<Attribute>());
}

/// Walks the tensor a fill vector function returns back to the argument it was
/// stamped into, picking up the constant on the way. The function is known to
/// be a fill, so this only decodes which argument carries which constant; the
/// splat check guards against the marker and the body disagreeing.
static bool traceConstantFill(Value v, TypedAttr &value, BlockArgument &root,
                              DenseSet<Value> &visited) {
  if (!visited.insert(v).second)
    return true;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *parent = arg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      OpOperand *init = forOp.getTiedLoopInit(arg);
      return init && traceConstantFill(init->get(), value, root, visited);
    }
    if (!isa<func::FuncOp>(parent) || (root && root != arg))
      return false;
    root = arg;
    return true;
  }

  Operation *def = v.getDefiningOp();
  if (isa<tensor::EmptyOp>(def))
    return true; // Undefined elements may be assumed to hold the constant.
  if (auto write = dyn_cast<vector::TransferWriteOp>(def)) {
    TypedAttr splat = getSplatValue(write.getVector());
    if (!splat || (value && value != splat))
      return false;
    value = splat;
    return traceConstantFill(write.getSource(), value, root, visited);
  }
  if (auto insert = dyn_cast<tensor::InsertSliceOp>(def))
    return traceConstantFill(insert.getSource(), value, root, visited) &&
           traceConstantFill(insert.getDest(), value, root, visited);
  if (auto extract = dyn_cast<tensor::ExtractSliceOp>(def))
    return traceConstantFill(extract.getSource(), value, root, visited);
  if (auto forOp = dyn_cast<scf::ForOp>(def))
    return traceConstantFill(
        forOp.getYieldedValues()[cast<OpResult>(v).getResultNumber()], value,
        root, visited);
  return false;
}

static BlockArgument getPristineArgument(Value v) {
  while (auto extract = v.getDefiningOp<tensor::ExtractSliceOp>())
    v = extract.getSource();
  auto arg = dyn_cast<BlockArgument>(v);
  if (arg && isa<func::FuncOp>(arg.getOwner()->getParentOp()))
    return arg;
  return {};
}

static TypedAttr getFilledConstant(Value v,
                                   SymbolTableCollection &symbolTables) {
  // Consumers beyond the first are fed a copy of the filled tensor -- the copy
  // that later shows up as a UB-to-UB `hivm.hir.copy`. A copy of a uniform
  // tensor is just as uniform.
  while (auto copy = v.getDefiningOp<linalg::CopyOp>()) {
    if (copy.getInputs().size() != 1)
      return {};
    v = copy.getInputs().front();
  }
  auto call = v.getDefiningOp<func::CallOp>();
  if (!call)
    return {};
  auto callee = dyn_cast_or_null<func::FuncOp>(
      symbolTables.lookupNearestSymbolFrom(call, call.getCalleeAttr()));
  if (!callee || callee.isExternal() ||
      !callee->hasAttr(kFillVFAttrName))
    return {};
  auto retOp = dyn_cast<func::ReturnOp>(callee.getBody().front().getTerminator());
  if (!retOp)
    return {};

  TypedAttr value;
  BlockArgument root;
  DenseSet<Value> visited;
  if (!traceConstantFill(retOp.getOperand(cast<OpResult>(v).getResultNumber()),
                         value, root, visited) ||
      !value || !root)
    return {};
  // We need to start with an empty tensor.
  if (!call.getOperand(root.getArgNumber()).getDefiningOp<tensor::EmptyOp>())
    return {};
  return value;
}

static bool isFoldableRead(vector::TransferReadOp read) {
  if (auto inBounds = read.getInBoundsAttr())
    for (Attribute dim : inBounds)
      if (!cast<BoolAttr>(dim).getValue())
        return false;
  return true;
}

static Value materializeFilledVector(OpBuilder &builder,
                                     vector::TransferReadOp read,
                                     TypedAttr value) {
  Location loc = read.getLoc();
  VectorType vecTy = read.getVectorType();
  Value filled = builder.create<arith::ConstantOp>(
      loc, vecTy, DenseElementsAttr::get(vecTy, value));
  Value mask = read.getMask();
  if (!mask)
    return filled;
  Value padding = read.getPadding();
  Attribute paddingAttr;
  if (matchPattern(padding, m_Constant(&paddingAttr)) && paddingAttr == value)
    return filled;
  Value padded = builder.create<vector::BroadcastOp>(loc, vecTy, padding);
  return builder.create<arith::SelectOp>(loc, mask, filled, padded);
}

// The folded case (the reader has exactly one call site):
//   fill_vf(%empty):
//     return transfer_write(splat(c), %empty, ...)
//   reader_vf(..., %input):
//     vector = transfer_read %input[...]       // all dimensions in bounds
//   caller:
//     %filled = call @fill_vf(%empty)
//     call @reader_vf(..., %filled)
// becomes:
//   reader_vf(..., %input):
//     vector = splat(c)                         // preserve mask padding
static void foldConstantFills(ModuleOp module, IRRewriter &rewriter) {
  SymbolTableCollection symbolTables;
  DenseMap<Operation *, func::CallOp> soleCaller;
  DenseSet<Operation *> shared;
  module.walk([&](func::CallOp call) {
    auto callee = dyn_cast_or_null<func::FuncOp>(
        symbolTables.lookupNearestSymbolFrom(call, call.getCalleeAttr()));
    if (callee && !soleCaller.insert({callee, call}).second)
      shared.insert(callee);  // multi call sites, bail out.
  });

  SmallVector<std::pair<vector::TransferReadOp, TypedAttr>> folds;
  module.walk([&](vector::TransferReadOp read) {
    BlockArgument arg = getPristineArgument(read.getSource());
    if (!arg || !isFoldableRead(read))
      return;
    Operation *func = arg.getOwner()->getParentOp();
    if (shared.contains(func))
      return;
    auto caller = soleCaller.find(func);
    if (caller == soleCaller.end())
      return;
    if (TypedAttr value = getFilledConstant(
            caller->second.getOperand(arg.getArgNumber()), symbolTables))
      folds.emplace_back(read, value);
  });

  for (auto [read, value] : folds) {
    rewriter.setInsertionPoint(read);
    rewriter.replaceOp(read, materializeFilledVector(rewriter, read, value));
  }
}

/// Promotes a loop-carried tensor subset accessed through vector transfers to
/// a loop-carried vector value. Unlike MLIR's LoopInvariantSubsetHoisting,
/// which recognizes and moves matching tensor.extract_slice/insert_slice
/// pairs, this handles vector.transfer_read/write (and optional surrounding
/// slice chains) whose accessed subset is invariant. The library pass cannot
/// prove the accessed subset of the vector transfers, whereas this pass
/// explicitly requires every transfer in the def-use chain to use the same
/// invariant subset.
///
/// Pseudocode for the read-modify-write case:
///   before:
///     for (...; tensor = init) {
///       vector = transfer_read tensor[invariant_indices]
///       updated = compute(vector)
///       tensor = transfer_write updated, tensor[invariant_indices]
///       yield tensor
///     }
///   after:
///     vector = transfer_read init[invariant_indices]
///     for (...; tensor = init, vector) {
///       vector = compute(vector)
///       yield tensor, vector
///     }
///     result = transfer_write vector, init[invariant_indices]
///
/// TODO
/// 1. The current equivalent dependency on multiple extract_Slice/insert_stice
///    is that the extract_Slice/insert_stice chains are completely equivalent,
///    which can be optimized for offset and strips calculations in the future.
/// 2. If there are only invariant writes in the loop, a sink optimization may
///    be needed.
/// 3. Currently relying on canonicalize to remove dead iter_args, if
///    compilation time is long, consider rewriting scf.for internally in Pass.
/// 4. There is still room for optimization in scenarios where iter_arg index
///    and yield index are different.
/// 5. Constant fill forwarding follows a load's source through extract_slice
///    only, so threading the filled tensor through an scf.for iter_arg stops
///    it. For an accumulator that is the point -- past the first iteration the
///    iter_arg holds the running value rather than the fill -- but it also
///    gives up on an iter_arg the loop merely forwards to its yield. Those are
///    what canonicalization drops, so the residual case is narrow.
static void promoteInFunc(func::FuncOp func, MLIRContext *context) {
  // Only optimize the vf function.
  if (!func->hasAttr(hivm::VectorFunctionAttr::name))
    return;

  FrozenRewritePatternSet cleanup = [&] {
    RewritePatternSet patterns(context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    return FrozenRewritePatternSet(std::move(patterns));
  }();

  IRRewriter rewriter(context);
  bool changed = true;
  while (changed) {
    func.walk([](LoopLikeOpInterface loop) { hoistLoopInvariant(loop); });
    bool nested = false;
    WalkResult res = func.walk([&](scf::ForOp forOp) {
      for (unsigned i = 0, e = forOp.getNumRegionIterArgs(); i != e; ++i) {
        std::optional<Candidate> cand = analyze(forOp, i);
        if (!cand)
          continue;
        bool isNested = forOp->getParentOfType<scf::ForOp>() != nullptr;
        if (succeeded(promote(rewriter, forOp, *cand))) {
          nested = isNested;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    changed = res.wasInterrupted();
    if (changed && nested)
      (void)applyPatternsGreedily(func, cleanup);
  }
}

void LoopInvariantPromotion::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();
  for (auto func : module.getOps<func::FuncOp>())
    promoteInFunc(func, context);

  IRRewriter rewriter(context);
  foldConstantFills(module, rewriter);
}

std::unique_ptr<Pass> mlir::hfusion::createLoopInvariantPromotionPass() {
  return std::make_unique<LoopInvariantPromotion>();
}

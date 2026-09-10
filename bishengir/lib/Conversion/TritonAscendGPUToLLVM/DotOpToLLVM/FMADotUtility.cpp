//===-- FMADotUtility.cpp - K-outer FMA microkernel with lazy extraction --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/FMADotUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <numeric>

using namespace mlir;

namespace mlir::triton::ascend {

static constexpr llvm::StringLiteral kCGroupedAttr =
    "bishengir.dot.c_grouped_for_overlap";
static constexpr llvm::StringLiteral kGroupIdAttr = "bishengir.dot.group_id";
static constexpr llvm::StringLiteral kFmaSrcDotAttr = "fma.src_dot";

static std::optional<int64_t> getGroupId(DotOp op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(kGroupIdAttr))
    return attr.getInt();
  return std::nullopt;
}

static bool isGroupedDot(DotOp op) {
  return op && op->hasAttr(kCGroupedAttr) && getGroupId(op).has_value();
}

static SmallVector<DotOp> getGroupedDotsInBlock(DotOp anchor) {
  SmallVector<DotOp> groupedDots;
  auto groupId = getGroupId(anchor);
  if (!groupId)
    return groupedDots;

  Block *block = anchor->getBlock();
  if (!block)
    return groupedDots;

  for (Operation &candidate : *block) {
    auto dot = dyn_cast<DotOp>(&candidate);
    if (!isGroupedDot(dot))
      continue;
    auto candidateGroupId = getGroupId(dot);
    if (candidateGroupId && *candidateGroupId == *groupId)
      groupedDots.push_back(dot);
  }
  return groupedDots;
}

static SmallVector<DotOp> topoOrderGroupedDots(ArrayRef<DotOp> dots) {
  DenseSet<Operation *> inGroup;
  for (DotOp dot : dots)
    inGroup.insert(dot.getOperation());

  DenseMap<Operation *, unsigned> indegree;
  DenseMap<Operation *, SmallVector<DotOp>> successors;
  for (DotOp dot : dots)
    indegree.try_emplace(dot.getOperation(), 0);

  for (DotOp dot : dots) {
    DenseSet<Operation *> seenProducers;
    auto addProducer = [&](Value operand) {
      while (auto cvt = operand.getDefiningOp<triton::gpu::ConvertLayoutOp>())
        operand = cvt.getSrc();
      if (auto producer = operand.getDefiningOp<DotOp>()) {
        Operation *producerOp = producer.getOperation();
        if (!inGroup.count(producerOp) || !seenProducers.insert(producerOp).second)
          return;
        successors[producerOp].push_back(dot);
        ++indegree[dot.getOperation()];
      }
    };
    addProducer(dot.getA());
    addProducer(dot.getB());
    addProducer(dot.getC());
  }

  SmallVector<DotOp> remaining(dots.begin(), dots.end());
  SmallVector<DotOp> ordered;
  ordered.reserve(dots.size());

  while (!remaining.empty()) {
    auto it = llvm::find_if(remaining, [&](DotOp dot) {
      auto indegreeIt = indegree.find(dot.getOperation());
      return indegreeIt != indegree.end() && indegreeIt->second == 0;
    });
    if (it == remaining.end()) {
      ordered.append(remaining.begin(), remaining.end());
      break;
    }

    DotOp current = *it;
    ordered.push_back(current);
    remaining.erase(it);

    for (DotOp succ : successors[current.getOperation()])
      --indegree[succ.getOperation()];
  }

  return ordered;
}

static DotOp getGroupedProducer(Value operand) {
  while (auto cvt = operand.getDefiningOp<triton::gpu::ConvertLayoutOp>())
    operand = cvt.getSrc();
  return operand.getDefiningOp<DotOp>();
}

static FailureOr<SmallVector<Value>>
getRemappedElements(Value operand, Location loc,
                    ConversionPatternRewriter &rewriter) {
  SmallVector<Value> remapped;
  if (failed(rewriter.getRemappedValues(ValueRange{operand}, remapped)) ||
      remapped.empty())
    return failure();
  Type ty = remapped.front().getType();
  if (!(ty.isIntOrIndexOrFloat() || isa<triton::PointerType>(ty) ||
        isa<LLVM::LLVMPointerType>(ty) || isa<LLVM::LLVMStructType>(ty)))
    return failure();
  return unpackLLElements(loc, remapped.front(), rewriter);
}

enum class SourceKind { External, ProducerAcc };

struct ResolvedSource {
  SourceKind kind;
  SmallVector<Value> values;
  Operation *producer = nullptr;
  RankedTensorType tensorTy;
};

struct LoweredAccumulator {
  SmallVector<Value> values;
  RankedTensorType tensorTy;
};

static bool lessOffset(ArrayRef<unsigned> lhs, ArrayRef<unsigned> rhs) {
  for (auto [l, r] : llvm::zip(lhs, rhs)) {
    if (l != r)
      return l < r;
  }
  return lhs.size() < rhs.size();
}

static FailureOr<SmallVector<Value>> canonicalizeElements(
    ArrayRef<Value> values, RankedTensorType tensorTy) {
  auto offsets = emitOffsetForLayout(tensorTy.getEncoding(), tensorTy);
  if (offsets.size() != values.size())
    return failure();

  SmallVector<unsigned> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
    return lessOffset(offsets[lhs], offsets[rhs]);
  });

  SmallVector<Value> result;
  result.reserve(values.size());
  for (unsigned index : order)
    result.push_back(values[index]);
  return result;
}

static FailureOr<SmallVector<Value>> remapCanonicalElements(
    ArrayRef<Value> canonicalValues, RankedTensorType srcTy,
    RankedTensorType dstTy) {
  auto srcOffsets = emitOffsetForLayout(srcTy.getEncoding(), srcTy);
  auto dstOffsets = emitOffsetForLayout(dstTy.getEncoding(), dstTy);
  if (srcOffsets.size() != canonicalValues.size())
    return failure();

  SmallVector<unsigned> srcOrder(canonicalValues.size());
  std::iota(srcOrder.begin(), srcOrder.end(), 0);
  llvm::sort(srcOrder, [&](unsigned lhs, unsigned rhs) {
    return lessOffset(srcOffsets[lhs], srcOffsets[rhs]);
  });

  SmallVector<Value> result;
  result.reserve(dstOffsets.size());
  for (const auto &dstOffset : dstOffsets) {
    auto it = llvm::find_if(srcOrder, [&](unsigned index) {
      return srcOffsets[index] == dstOffset;
    });
    if (it == srcOrder.end())
      return failure();
    result.push_back(canonicalValues[it - srcOrder.begin()]);
  }
  return result;
}

struct FmaTileSizes {
  unsigned mTile;
  unsigned nTile;
};

static FmaTileSizes computeFmaTileSizes(DotOp op, unsigned totalM,
                                        unsigned totalN) {
  constexpr unsigned kTotalCTARegs = 32768;
  unsigned nTile = totalN;
  unsigned mTile = totalM;
  if (auto mod = op->getParentOfType<ModuleOp>()) {
    int numWarps = gpu::lookupNumWarps(op.getOperation());
    int tpw = gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (numWarps > 0 && tpw > 0) {
      unsigned threadsPerCTA =
          static_cast<unsigned>(numWarps) * static_cast<unsigned>(tpw);
      unsigned regsPerThread = kTotalCTARegs / threadsPerCTA;

      unsigned sharedMem = 0;
      if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
              "bishengir.shared-mem-dynamic-size"))
        sharedMem = static_cast<unsigned>(attr.getInt());

      unsigned dcacheKB;
      if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
              "bishengir.fma-dcache-budget-kb")) {
        dcacheKB = static_cast<unsigned>(attr.getInt());
      } else {
        constexpr unsigned kTotalKB = 256, kOsKB = 8, kMinSharedKB = 128;
        constexpr unsigned kDCacheMinKB = 32, kDCacheMaxKB = 120;
        unsigned sharedKB = (sharedMem + 1023u) / 1024u;
        dcacheKB = kTotalKB - kOsKB - std::max(kMinSharedKB, sharedKB);
        dcacheKB = std::max(kDCacheMinKB, std::min(kDCacheMaxKB, dcacheKB));
      }
      unsigned dcacheF32PerThread = (dcacheKB * 1024u) / (4u * threadsPerCTA);

      unsigned rawBudget = regsPerThread + dcacheF32PerThread;
      unsigned headroomPct = 100u;
      if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
              "bishengir.fma-budget-headroom-pct"))
        headroomPct = static_cast<unsigned>(attr.getInt());
      unsigned budget = (rawBudget * headroomPct) / 100u;

      if (budget > totalM && totalM * totalN + totalM + totalN > budget) {
        unsigned maxN = (budget - totalM) / (totalM + 1);
        unsigned floorP2 = (maxN > 0) ? (1u << llvm::Log2_32(maxN)) : 1u;
        nTile = std::min(totalN, std::max(1u, floorP2));
      }
      if (budget > nTile) {
        unsigned maxM = (budget - nTile) / (nTile + 1);
        unsigned floorP2M = (maxM > 0) ? (1u << llvm::Log2_32(maxM)) : 1u;
        mTile = std::min(totalM, std::max(1u, floorP2M));
      }
    }
  }
  return {mTile, nTile};
}

static FailureOr<ResolvedSource>
resolveSource(Value operand, DenseMap<Operation *, LoweredAccumulator> &lowered,
              Location loc, ConversionPatternRewriter &rewriter) {
  if (auto producer = getGroupedProducer(operand)) {
    if (auto it = lowered.find(producer.getOperation()); it != lowered.end()) {
      auto values = remapCanonicalElements(
          it->second.values, it->second.tensorTy,
          cast<RankedTensorType>(operand.getType()));
      if (succeeded(values))
        return ResolvedSource{SourceKind::ProducerAcc, *values,
                              producer.getOperation(), it->second.tensorTy};
    }
    return failure();
  }

  auto elems = getRemappedElements(operand, loc, rewriter);
  if (failed(elems))
    return failure();
  auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorTy)
    return failure();
  auto canonical = canonicalizeElements(*elems, tensorTy);
  if (failed(canonical))
    return failure();
  return ResolvedSource{SourceKind::External, *canonical, nullptr, tensorTy};
}

static FailureOr<ResolvedSource>
resolveWavefrontBaseC(Value operand, const DenseSet<Operation *> &wavefront,
                      Location loc, ConversionPatternRewriter &rewriter) {
  while (auto producer = getGroupedProducer(operand)) {
    if (!wavefront.count(producer.getOperation()))
      break;
    operand = producer.getC();
  }
  auto elems = getRemappedElements(operand, loc, rewriter);
  if (failed(elems))
    return failure();
  auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorTy)
    return failure();
  auto canonical = canonicalizeElements(*elems, tensorTy);
  if (failed(canonical))
    return failure();
  return ResolvedSource{SourceKind::External, *canonical, nullptr, tensorTy};
}

static StringAttr getFmaSrcDotAttr(DotOp source,
                                   ConversionPatternRewriter &rewriter) {
  std::string locStr;
  llvm::raw_string_ostream os(locStr);
  source.getLoc().print(os);
  return rewriter.getStringAttr(os.str());
}

static void annotateFmaSource(Value v, DotOp source,
                              ConversionPatternRewriter &rewriter) {
  auto attr = getFmaSrcDotAttr(source, rewriter);
  if (auto op = v.getDefiningOp()) {
    op->setAttr(kFmaSrcDotAttr, attr);
    if (auto add = dyn_cast<LLVM::AddOp>(op)) {
      if (auto mul = add.getLhs().getDefiningOp<LLVM::MulOp>())
        mul->setAttr(kFmaSrcDotAttr, attr);
    } else if (auto fadd = dyn_cast<LLVM::FAddOp>(op)) {
      if (auto mul = fadd.getLhs().getDefiningOp<LLVM::FMulOp>())
        mul->setAttr(kFmaSrcDotAttr, attr);
    }
  }
}

class GroupedFMAVectorMultiplier : public FMAVectorMultiplier {
  ConversionPatternRewriter &rewriter;
  Location loc;

public:
  GroupedFMAVectorMultiplier(ConversionPatternRewriter &rewriter, Location loc)
      : rewriter(rewriter), loc(loc) {}

  Value promoteToAccType(Value v, Type accTy) override {
    Type srcTy = v.getType();
    if (srcTy == accTy)
      return v;
    if (isa<FloatType>(srcTy) && isa<FloatType>(accTy))
      return rewriter.create<LLVM::FPExtOp>(loc, accTy, v);
    if (isa<IntegerType>(srcTy) && isa<IntegerType>(accTy))
      return rewriter.create<LLVM::SExtOp>(loc, accTy, v);
    return v;
  }

  Value emitSingleFMA(Value a, Value b, Value acc) override {
    Type accTy = acc.getType();
    if (isa<FloatType>(accTy))
      return rewriter.create<LLVM::FMulAddOp>(loc, a, b, acc);
    return rewriter.create<LLVM::AddOp>(loc,
                                         rewriter.create<LLVM::MulOp>(loc, a, b),
                                         acc);
  }

  Value multiplyVectors(const Value *, const Value *, Value, unsigned) override {
    llvm_unreachable("grouped dot lowering does not use multiplyVectors");
  }
};

LogicalResult parametricConvertFMADot(DotOp op, DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier) {
  auto loc = op.getLoc();

  auto A = op.getA();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  SmallVector<int64_t> aShapePerCTA =
      gpu::expandMatrixShapeWithBatch(ArrayRef(gpu::getShapePerCTA(aTensorTy)));

  // Accumulator: fully unpacked upfront — must remain live across all k steps.
  SmallVector<Value> acc = unpackLLElements(loc, adaptor.getC(), rewriter);

  // A and B kept as raw LLVM struct values. Elements are extracted lazily
  // inside the k-outer loop so each extracted SSA value is live for only one
  // k-step, keeping register pressure proportional to mTile + nTile + acc-tile
  // rather than totalM*K + totalN*K.
  Value aStruct = adaptor.getA();
  Value bStruct = adaptor.getB();

  // Determine per-thread element counts from the LLVM struct sizes.
  // (Same ordinal count as unpackLLElements would return.)
  unsigned totalAElems = 1, totalBElems = 1;
  if (auto sTy = dyn_cast<LLVM::LLVMStructType>(aStruct.getType()))
    totalAElems = static_cast<unsigned>(sTy.getBody().size());
  if (auto sTy = dyn_cast<LLVM::LLVMStructType>(bStruct.getType()))
    totalBElems = static_cast<unsigned>(sTy.getBody().size());

  const unsigned K = static_cast<unsigned>(aShapePerCTA[2]);

  assert(K > 0 && totalAElems % K == 0 &&
         "A element count must be a multiple of K");
  assert(K > 0 && totalBElems % K == 0 &&
         "B element count must be a multiple of K");

  const unsigned totalM = totalAElems / K;
  const unsigned totalN = totalBElems / K;
  const unsigned totalB =
      (totalM * totalN > 0)
          ? static_cast<unsigned>(acc.size()) / (totalM * totalN)
          : 1;

  assert(
      acc.size() == totalB * totalM * totalN &&
      "FMA layout mismatch: acc.size() != totalB * totalM * totalN. "
      "The linear layouts for A/B are inconsistent with D's blocked layout.");

  Type accTy = acc[0].getType();
  FmaTileSizes tileSizes = computeFmaTileSizes(op, totalM, totalN);
  unsigned mTile = tileSizes.mTile;
  unsigned nTile = tileSizes.nTile;

  // ── K-outer microkernel with lazy element extraction ──────────────────────
  //
  // Loop order: batch → k → M-tile → N-tile → m → n
  //
  // Ordinal conventions (must match layouts from ConvertDotInputToLinearLayout):
  //   A is K-innermost: ordinal(bi, mi, k) = (bi*totalM + mi)*K + k
  //   B is N-innermost: ordinal(bi, k, ni) = (bi*K + k)*totalN + ni
  //
  // A ordinal has no niBase dependence, so A is extracted once per (bi, k,
  // miBase) and reused across all N-tile iterations — eliminating redundant
  // ExtractValue+FPExt pairs. B ordinal has no miBase dependence, so B is
  // re-extracted per (k, miBase) iteration (same trade-off N-tiling makes).
  //
  for (unsigned bi = 0; bi < totalB; ++bi) {
    for (unsigned k = 0; k < K; ++k) {
      for (unsigned miBase = 0; miBase < totalM; miBase += mTile) {
        const unsigned miEnd = std::min(miBase + mTile, totalM);

        // Lazily extract and promote A[bi, mi, k] for mi in [miBase, miEnd).
        // Live across all N-tile iterations at this (bi, k, miBase).
        SmallVector<Value, 4> aSlice;
        aSlice.reserve(miEnd - miBase);
        for (unsigned mi = miBase; mi < miEnd; ++mi) {
          unsigned aOrdinal = (bi * totalM + mi) * K + k;
          Value raw =
              rewriter.create<LLVM::ExtractValueOp>(loc, aStruct, aOrdinal);
          aSlice.push_back(multiplier.promoteToAccType(raw, accTy));
        }

        for (unsigned niBase = 0; niBase < totalN; niBase += nTile) {
          const unsigned niEnd = std::min(niBase + nTile, totalN);

          // Lazily extract and promote B[bi, k, ni] for ni in [niBase, niEnd).
          SmallVector<Value, 16> bSlice;
          bSlice.reserve(niEnd - niBase);
          for (unsigned ni = niBase; ni < niEnd; ++ni) {
            unsigned bOrdinal = (bi * K + k) * totalN + ni;
            Value raw =
                rewriter.create<LLVM::ExtractValueOp>(loc, bStruct, bOrdinal);
            bSlice.push_back(multiplier.promoteToAccType(raw, accTy));
          }

          // Emit mTile * nTile FMAs.  Only aSlice + bSlice + acc-tile live.
          for (unsigned mi = miBase; mi < miEnd; ++mi) {
            for (unsigned ni = niBase; ni < niEnd; ++ni) {
              unsigned accIdx = bi * totalM * totalN + mi * totalN + ni;
              acc[accIdx] = multiplier.emitSingleFMA(aSlice[mi - miBase],
                                                     bSlice[ni - niBase],
                                                     acc[accIdx]);
            }
          }
          // bSlice is dead here — LLVM recycles its registers.
        }
        // aSlice is dead here — LLVM recycles its registers.
      }
    }
  }

  auto res = packLLElements(loc, typeConverter, acc, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);
  return success();
}

bool isGroupedDotAnchor(DotOp op) {
  if (!isGroupedDot(op))
    return false;

  auto groupId = getGroupId(op);
  if (!groupId)
    return false;

  for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode()) {
    auto prevDot = dyn_cast<DotOp>(prev);
    if (!isGroupedDot(prevDot))
      continue;
    auto prevGroupId = getGroupId(prevDot);
    if (prevGroupId && *prevGroupId == *groupId)
      return false;
  }
  return true;
}

/// Lowers the complete C-group containing `op` as one accumulator schedule.
///
/// The anchor owns the group and collects all dots with the same group ID,
/// orders them by their intra-group dependencies, and lowers their FMAs while
/// forwarding producer accumulators directly to consumers.  This keeps the
/// chain's accumulator flow intact, shortens intermediate value lifetimes,
/// and avoids packing and unpacking each intermediate dot result.  Results
/// are remapped and replaced only after the whole group has been lowered to
/// avoid SSA dominance issues.
LogicalResult convertGroupedFMADots(DotOp op,
                                    DotOp::Adaptor adaptor,
                                    const LLVMTypeConverter *typeConverter,
                                    ConversionPatternRewriter &rewriter) {
  if (!isGroupedDotAnchor(op))
    return failure();

  SmallVector<DotOp> groupedDots = getGroupedDotsInBlock(op);
  if (groupedDots.empty())
    return failure();

  groupedDots = topoOrderGroupedDots(groupedDots);

  auto aTensorTy = cast<RankedTensorType>(op.getA().getType());

  const auto aShape = aTensorTy.getShape();
  assert(!aShape.empty() && "grouped dot A must be rank >= 1");
  const unsigned K = static_cast<unsigned>(aShape.back());

  auto getStructElems = [](Value v) -> unsigned {
    auto sTy = dyn_cast<LLVM::LLVMStructType>(v.getType());
    if (!sTy)
      return 1;
    return static_cast<unsigned>(sTy.getBody().size());
  };

  const unsigned totalAElems = getStructElems(adaptor.getA());
  const unsigned totalBElems = getStructElems(adaptor.getB());
  const unsigned totalCElems = unpackLLElements(op.getLoc(), adaptor.getC(),
                                                rewriter)
                                   .size();

  assert(K > 0 && totalAElems % K == 0 && totalBElems % K == 0 &&
         "grouped dot element counts must be multiples of K");

  const unsigned totalM = totalAElems / K;
  const unsigned totalN = totalBElems / K;
  const unsigned totalB =
      (totalM * totalN > 0) ? totalCElems / (totalM * totalN) : 1;

  assert(totalB * totalM * totalN == totalCElems &&
         "grouped dot layout mismatch");

  DenseSet<Operation *> inGroup;
  for (DotOp dot : groupedDots)
    inGroup.insert(dot.getOperation());

  auto getAccTy = [&](DotOp dot) -> Type {
    return cast<RankedTensorType>(dot.getResult().getType()).getElementType();
  };

  FmaTileSizes tileSizes = computeFmaTileSizes(op, totalM, totalN);
  unsigned mTile = tileSizes.mTile;
  unsigned nTile = tileSizes.nTile;

  auto lowerSequentialDot =
      [&](DotOp groupedDot,
          DenseMap<Operation *, LoweredAccumulator> &loweredAccs)
      -> LogicalResult {
    rewriter.setInsertionPoint(groupedDot);
    auto loc = groupedDot.getLoc();
    Type accTy = getAccTy(groupedDot);
    GroupedFMAVectorMultiplier multiplier(rewriter, loc);

    auto aSource =
        resolveSource(groupedDot.getA(), loweredAccs, loc, rewriter);
    auto bSource =
        resolveSource(groupedDot.getB(), loweredAccs, loc, rewriter);
    auto cSource =
        resolveSource(groupedDot.getC(), loweredAccs, loc, rewriter);
    if (failed(aSource) || failed(bSource) || failed(cSource))
      return failure();

    SmallVector<Value> acc = cSource->values;
    if (acc.empty())
      return failure();

    for (unsigned bi = 0; bi < totalB; ++bi) {
      for (unsigned k = 0; k < K; ++k) {
        for (unsigned miBase = 0; miBase < totalM; miBase += mTile) {
          const unsigned miEnd = std::min(miBase + mTile, totalM);
          SmallVector<Value, 4> aSlice;
          aSlice.reserve(miEnd - miBase);
          for (unsigned mi = miBase; mi < miEnd; ++mi) {
            unsigned aIdx = aSource->kind == SourceKind::External
                                ? ((bi * totalM) + mi) * K + k
                                : mi * totalN + k;
            if (aIdx >= aSource->values.size())
              return failure();
            Value a = multiplier.promoteToAccType(aSource->values[aIdx], accTy);
            aSlice.push_back(a);
          }

          for (unsigned niBase = 0; niBase < totalN; niBase += nTile) {
            const unsigned niEnd = std::min(niBase + nTile, totalN);
            SmallVector<Value, 16> bSlice;
            bSlice.reserve(niEnd - niBase);
          for (unsigned ni = niBase; ni < niEnd; ++ni) {
            unsigned bIdx = bSource->kind == SourceKind::External
                                ? ((bi * K) + k) * totalN + ni
                                : k * totalN + ni;
            if (bIdx >= bSource->values.size())
              return failure();
            Value b = multiplier.promoteToAccType(bSource->values[bIdx], accTy);
            bSlice.push_back(b);
          }

          for (unsigned mi = miBase; mi < miEnd; ++mi) {
            for (unsigned ni = niBase; ni < niEnd; ++ni) {
              unsigned accIdx = bi * totalM * totalN + mi * totalN + ni;
              if (accIdx >= acc.size())
                return failure();
              acc[accIdx] = multiplier.emitSingleFMA(
                  aSlice[mi - miBase], bSlice[ni - niBase], acc[accIdx]);
              annotateFmaSource(acc[accIdx], groupedDot, rewriter);
            }
          }
          }
        }
      }
    }

    auto resultTy = cast<RankedTensorType>(groupedDot.getResult().getType());
    auto canonical = canonicalizeElements(acc, resultTy);
    if (failed(canonical))
      return failure();
    loweredAccs[groupedDot.getOperation()] =
        LoweredAccumulator{*canonical, resultTy};
    return success();
  };

  DenseMap<Operation *, LoweredAccumulator> loweredAccs;
  DenseSet<Operation *> alreadyLowered;

  SmallVector<DotOp> activeWavefrontDots;
  SmallVector<DotOp> sequentialDotsFiltered;
  DenseSet<Operation *> wavefrontSet;
  for (DotOp dot : groupedDots) {
    bool cReady = true;
    if (auto producer = getGroupedProducer(dot.getC())) {
      if (inGroup.count(producer.getOperation()))
        cReady = false;
      else
        cReady = wavefrontSet.count(producer.getOperation());
    }
    if (cReady) {
      activeWavefrontDots.push_back(dot);
      wavefrontSet.insert(dot.getOperation());
    } else {
      sequentialDotsFiltered.push_back(dot);
    }
  }

  if (!activeWavefrontDots.empty()) {
    struct WavefrontDotState {
      SmallVector<Value> acc;
      SmallVector<Value> aVals;
      SmallVector<Value> bVals;
    };

    DenseMap<Operation *, WavefrontDotState> states;
    for (DotOp dot : activeWavefrontDots) {
      rewriter.setInsertionPoint(dot);
      auto loc = dot.getLoc();
      WavefrontDotState state;
      auto aSource = resolveSource(dot.getA(), loweredAccs, loc, rewriter);
      auto bSource = resolveSource(dot.getB(), loweredAccs, loc, rewriter);
      auto cSource = resolveWavefrontBaseC(dot.getC(), wavefrontSet, loc,
                                           rewriter);
      if (failed(aSource) || failed(bSource) || failed(cSource))
        return failure();
      state.acc = cSource->values;
      state.aVals = aSource->values;
      state.bVals = bSource->values;
      if (state.acc.empty() || state.aVals.empty() || state.bVals.empty())
        return failure();
      states[dot.getOperation()] = std::move(state);
    }

    rewriter.setInsertionPoint(op);
    for (unsigned bi = 0; bi < totalB; ++bi) {
      for (unsigned k = 0; k < K; ++k) {
        for (DotOp groupedDot : activeWavefrontDots) {
          rewriter.setInsertionPoint(groupedDot);
          auto stateIt = states.find(groupedDot.getOperation());
          if (stateIt == states.end())
            return failure();
          auto &state = stateIt->second;
          auto loc = groupedDot.getLoc();
          Type accTy = getAccTy(groupedDot);
          GroupedFMAVectorMultiplier multiplier(rewriter, loc);
          for (unsigned miBase = 0; miBase < totalM; miBase += mTile) {
            const unsigned miEnd = std::min(miBase + mTile, totalM);
            SmallVector<Value, 4> aSlice;
            aSlice.reserve(miEnd - miBase);
            for (unsigned mi = miBase; mi < miEnd; ++mi) {
              unsigned aIdx = ((bi * totalM) + mi) * K + k;
              if (aIdx >= state.aVals.size())
                return failure();
              Value a = multiplier.promoteToAccType(state.aVals[aIdx], accTy);
              aSlice.push_back(a);
            }

            for (unsigned niBase = 0; niBase < totalN; niBase += nTile) {
              const unsigned niEnd = std::min(niBase + nTile, totalN);
              SmallVector<Value, 16> bSlice;
              bSlice.reserve(niEnd - niBase);
              for (unsigned ni = niBase; ni < niEnd; ++ni) {
                unsigned bIdx = ((bi * K) + k) * totalN + ni;
                if (bIdx >= state.bVals.size())
                  return failure();
                Value b = multiplier.promoteToAccType(state.bVals[bIdx], accTy);
                bSlice.push_back(b);
              }

              for (unsigned mi = miBase; mi < miEnd; ++mi) {
                for (unsigned ni = niBase; ni < niEnd; ++ni) {
                  unsigned accIdx = bi * totalM * totalN + mi * totalN + ni;
                  if (accIdx >= state.acc.size())
                    return failure();
                  state.acc[accIdx] = multiplier.emitSingleFMA(
                      aSlice[mi - miBase], bSlice[ni - niBase],
                      state.acc[accIdx]);
                  annotateFmaSource(state.acc[accIdx], groupedDot, rewriter);
                }
              }
            }
          }
        }
      }
    }

    for (DotOp groupedDot : activeWavefrontDots) {
      rewriter.setInsertionPoint(groupedDot);
      auto stateIt = states.find(groupedDot.getOperation());
      if (stateIt == states.end())
        return failure();
      auto resultTy = cast<RankedTensorType>(groupedDot.getResult().getType());
      auto canonical = canonicalizeElements(stateIt->second.acc, resultTy);
      if (failed(canonical))
        return failure();
      loweredAccs[groupedDot.getOperation()] =
          LoweredAccumulator{*canonical, resultTy};
      alreadyLowered.insert(groupedDot.getOperation());
    }
  }

  for (DotOp groupedDot : sequentialDotsFiltered) {
    if (alreadyLowered.count(groupedDot.getOperation()))
      continue;
    if (failed(lowerSequentialDot(groupedDot, loweredAccs)))
      return failure();
    alreadyLowered.insert(groupedDot.getOperation());
  }

  // Replace all ops at the end, after every accumulator vector has been
  // fully computed. This avoids any SSA dominance issues.
  for (DotOp groupedDot : groupedDots) {
    rewriter.setInsertionPoint(groupedDot);
    auto accIt = loweredAccs.find(groupedDot.getOperation());
    if (accIt == loweredAccs.end())
      continue;
    auto loc = groupedDot.getLoc();
    auto resultTy = cast<RankedTensorType>(groupedDot.getResult().getType());
    auto physical = remapCanonicalElements(
        accIt->second.values, accIt->second.tensorTy, resultTy);
    if (failed(physical))
      return failure();
    auto res = packLLElements(loc, typeConverter, *physical, rewriter,
                              resultTy);
      rewriter.replaceOp(groupedDot, res);
  }

  return success();
}

} // namespace mlir::triton::ascend

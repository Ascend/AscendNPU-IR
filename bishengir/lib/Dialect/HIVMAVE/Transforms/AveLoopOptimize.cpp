//===------------- AveLoopOptimize.cpp - optimize AVE loops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass merges under-filled small-width vector work across loop
// iterations. The supported conversion widths and merge factors are:
//
//   factor 2: f16/bf16 <-> f32, i16 <-> i32, i8 <-> i16
//   factor 4: f8E4M3FN/f8E5M2 <-> f32, i8 <-> i32
//
//   16 <-> 32: 64 lanes per iteration, 128 narrow lanes after packing
//    8 <-> 16: 128 lanes per iteration, 256 narrow lanes after packing
//    8 <-> 32: 64 lanes per iteration, 256 narrow lanes after packing
//
// Widening example (f16 -> f32, factor 2):
//
//   Before:
//     n0 = vload 64xf16
//     w0 = vextf n0 : 64xf16 -> 64xf32
//     wide_op w0
//     n1 = vload 64xf16
//     w1 = vextf n1 : 64xf16 -> 64xf32
//     wide_op w1
//
//   After:
//     packed = vload 128xf16
//     n0, n1 = vintlv <SPARSE> packed, packed
//     w0 = vextf n0 : 64xf16 -> 64xf32
//     w1 = vextf n1 : 64xf16 -> 64xf32
//     wide_op w0
//     wide_op w1
//
// The load is merged, but each full-width conversion and computation remains
// independent. Factor-4 widening uses a vintlv tree to recover four inputs.
//
// Narrowing example (f32 -> f16, factor 2):
//
//   Before:
//     n0 = vtruncf w0 : 64xf32 -> 64xf16
//     r0 = narrow_op n0
//     masked_store r0
//     n1 = vtruncf w1 : 64xf32 -> 64xf16
//     r1 = narrow_op n1
//     masked_store r1
//
//   After:
//     n0 = vtruncf w0 : 64xf32 -> 64xf16
//     n1 = vtruncf w1 : 64xf32 -> 64xf16
//     packed = vdintlv <DENSE> n0, n1
//     result = narrow_op packed
//     masked_store result : 128xf16
//
// The conversions remain independent because every wide value already fills
// one vector register. The narrow elementwise chain and masked store are
// merged. Factor-4 narrowing uses a vdintlv tree to pack four results.
//
// Integer conversions use vextsi/vextui/vtrunci in the same shapes. Rewriting
// still requires adjacent accesses, equivalent masks/scalars and a supported
// one-to-one narrow elementwise chain.

#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_AVELOOPOPTIMIZE
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivmave;

namespace {

constexpr llvm::StringLiteral kGroupAttr = "__ave_small_width_group_id";
constexpr llvm::StringLiteral kNodeAttr = "__ave_small_width_node_id";
constexpr llvm::StringLiteral kCloneAttr = "__ave_small_width_clone_index";
constexpr llvm::StringLiteral kContinuousAttr = "hivm.is_continuous";

enum class GroupKind {
  Load,
  NarrowStore,
};

struct ConversionInfo {
  Value src;
  Value result;
  VectorType dstType;
  unsigned factor;
  bool widening;
};

struct SmallWidthGroup {
  int64_t id;
  unsigned factor;
  GroupKind kind;
  SmallVector<Operation *, 8> nodes;
};

struct SmallWidthMergePlan {
  unsigned unrollFactor;
  SmallVector<SmallWidthGroup, 8> groups;
};

struct PackedAccess {
  Value base;
  SmallVector<Value, 1> indices;
};

using GroupInstances = SmallVector<SmallVector<Operation *, 4>, 8>;

// Return the coefficient of the loop IV in an affine expression. A missing
// value means the expression is not linear enough to prove an address stride.
static std::optional<int64_t>
getAffineExprCoefficient(AffineExpr expr,
                         ArrayRef<std::optional<int64_t>> dimCoefficients,
                         ArrayRef<std::optional<int64_t>> symbolCoefficients) {
  if (isa<AffineConstantExpr>(expr))
    return 0;
  if (auto dim = dyn_cast<AffineDimExpr>(expr))
    return dimCoefficients[dim.getPosition()];
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expr))
    return symbolCoefficients[symbol.getPosition()];

  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary)
    return std::nullopt;
  std::optional<int64_t> lhs = getAffineExprCoefficient(
      binary.getLHS(), dimCoefficients, symbolCoefficients);
  std::optional<int64_t> rhs = getAffineExprCoefficient(
      binary.getRHS(), dimCoefficients, symbolCoefficients);
  if (!lhs || !rhs)
    return std::nullopt;

  switch (expr.getKind()) {
  case AffineExprKind::Add:
    return *lhs + *rhs;
  case AffineExprKind::Mul:
    if (auto lhsConstant = dyn_cast<AffineConstantExpr>(binary.getLHS()))
      return lhsConstant.getValue() * *rhs;
    if (auto rhsConstant = dyn_cast<AffineConstantExpr>(binary.getRHS()))
      return rhsConstant.getValue() * *lhs;
    return std::nullopt;
  default:
    // Floor/mod/ceil of an IV-dependent expression is not globally linear.
    return *lhs == 0 && *rhs == 0 ? std::optional<int64_t>(0) : std::nullopt;
  }
}

static std::optional<int64_t> getValueCoefficient(Value value, Value iv,
                                                  scf::ForOp forOp) {
  if (value == iv)
    return 1;
  if (hivmave::getConstantIntValue(value))
    return 0;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == forOp.getBody())
      return std::nullopt;
    return 0;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp || !forOp->isAncestor(defOp))
    return 0;

  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return getValueCoefficient(castOp.getIn(), iv, forOp);
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    auto lhs = getValueCoefficient(addOp.getLhs(), iv, forOp);
    auto rhs = getValueCoefficient(addOp.getRhs(), iv, forOp);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs + *rhs;
  }
  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    auto lhs = getValueCoefficient(subOp.getLhs(), iv, forOp);
    auto rhs = getValueCoefficient(subOp.getRhs(), iv, forOp);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs - *rhs;
  }
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    if (auto lhsConstant = hivmave::getConstantIntValue(mulOp.getLhs())) {
      auto rhs = getValueCoefficient(mulOp.getRhs(), iv, forOp);
      return rhs ? std::optional<int64_t>(*lhsConstant * *rhs) : std::nullopt;
    }
    if (auto rhsConstant = hivmave::getConstantIntValue(mulOp.getRhs())) {
      auto lhs = getValueCoefficient(mulOp.getLhs(), iv, forOp);
      return lhs ? std::optional<int64_t>(*rhsConstant * *lhs) : std::nullopt;
    }
    return std::nullopt;
  }
  if (auto applyOp = dyn_cast<affine::AffineApplyOp>(defOp)) {
    AffineMap map = applyOp.getAffineMap();
    if (map.getNumResults() != 1)
      return std::nullopt;

    SmallVector<std::optional<int64_t>> coefficients;
    coefficients.reserve(applyOp.getMapOperands().size());
    for (Value operand : applyOp.getMapOperands())
      coefficients.push_back(getValueCoefficient(operand, iv, forOp));
    ArrayRef<std::optional<int64_t>> coefficientRef(coefficients);
    return getAffineExprCoefficient(
        map.getResult(0), coefficientRef.take_front(map.getNumDims()),
        coefficientRef.drop_front(map.getNumDims()));
  }
  return std::nullopt;
}

// Flatten the explicit indices and nested subview offsets into one coefficient:
// increasing the IV by one moves the accessed address by this many elements.
static std::optional<int64_t> getLinearizedIvStride(Value base,
                                                    ValueRange indices,
                                                    MemRefType memrefType,
                                                    scf::ForOp forOp) {
  Value iv = forOp.getInductionVar();
  int64_t linearCoefficient = 0;

  auto accumulateOffsets = [&linearCoefficient, iv,
                            forOp](MemRefType type,
                                   ArrayRef<OpFoldResult> offsets) -> bool {
    SmallVector<int64_t> strides;
    int64_t baseOffset = 0;
    if (failed(getStridesAndOffset(type, strides, baseOffset)) ||
        strides.size() != offsets.size())
      return false;
    for (auto [offset, stride] : llvm::zip(offsets, strides)) {
      if (ShapedType::isDynamic(stride))
        return false;
      auto offsetValue = dyn_cast<Value>(offset);
      if (!offsetValue)
        continue;
      auto coefficient = getValueCoefficient(offsetValue, iv, forOp);
      if (!coefficient)
        return false;
      linearCoefficient += *coefficient * stride;
    }
    return true;
  };

  SmallVector<OpFoldResult> accessIndices;
  accessIndices.reserve(indices.size());
  for (Value index : indices)
    accessIndices.push_back(index);
  if (!accumulateOffsets(memrefType, accessIndices))
    return std::nullopt;

  Value currentMemref = base;
  while (auto subview = currentMemref.getDefiningOp<memref::SubViewOp>()) {
    auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
    if (!sourceType ||
        !accumulateOffsets(sourceType, subview.getMixedOffsets()))
      return std::nullopt;
    currentMemref = subview.getSource();
  }

  if (auto blockArg = dyn_cast<BlockArgument>(currentMemref)) {
    if (blockArg.getOwner() == forOp.getBody())
      return std::nullopt;
  } else if (Operation *defOp = currentMemref.getDefiningOp();
             defOp && forOp->isAncestor(defOp)) {
    return std::nullopt;
  }
  return linearCoefficient;
}

static bool hasContiguousIterationAccess(Value base, ValueRange indices,
                                         MemRefType memrefType,
                                         VectorType vectorType,
                                         scf::ForOp forOp) {
  std::optional<int64_t> ivStride =
      getLinearizedIvStride(base, indices, memrefType, forOp);
  std::optional<int64_t> loopStep =
      hivmave::getConstantIntValue(forOp.getStep());
  // Strides are measured in elements, not bytes. Adjacent iterations are
  // contiguous only when their address delta equals one original vector. For
  // index = IV, step = 64 and vector<64xT>, the proof is 1 * 64 == 64.
  return ivStride && loopStep &&
         *ivStride * *loopStep == vectorType.getNumElements();
}

static bool isFloat8(Type type) {
  return isa<Float8E4M3FNType, Float8E5M2Type>(type);
}

static bool isSupportedConversion(Operation &op, Type srcElementType,
                                  Type dstElementType) {
  if (isa<VFExtFOp>(&op)) {
    return ((srcElementType.isF16() || srcElementType.isBF16()) &&
            dstElementType.isF32()) ||
           (isFloat8(srcElementType) && dstElementType.isF32());
  }
  if (isa<VFTruncFOp>(&op)) {
    return (srcElementType.isF32() &&
            (dstElementType.isF16() || dstElementType.isBF16())) ||
           (srcElementType.isF32() && isFloat8(dstElementType));
  }
  if (isa<VFExtSIOp, VFExtUIOp>(&op)) {
    return (srcElementType.isSignlessInteger(8) &&
            (dstElementType.isSignlessInteger(16) ||
             dstElementType.isSignlessInteger(32))) ||
           (srcElementType.isSignlessInteger(16) &&
            dstElementType.isSignlessInteger(32));
  }
  if (isa<VFTruncIOp>(&op)) {
    return ((srcElementType.isSignlessInteger(16) ||
             srcElementType.isSignlessInteger(32)) &&
            dstElementType.isSignlessInteger(8)) ||
           (srcElementType.isSignlessInteger(32) &&
            dstElementType.isSignlessInteger(16));
  }
  return false;
}

static std::optional<ConversionInfo>
getConversionInfo(Operation &op, unsigned maxMergeFactor) {
  Value src;
  Value result;
  if (auto castOp = dyn_cast<VFExtFOp>(&op)) {
    src = castOp.getSrc();
    result = castOp.getRes();
  } else if (auto castOp = dyn_cast<VFTruncFOp>(&op)) {
    src = castOp.getSrc();
    result = castOp.getRes();
  } else if (auto castOp = dyn_cast<VFExtSIOp>(&op)) {
    src = castOp.getSrc();
    result = castOp.getRes();
  } else if (auto castOp = dyn_cast<VFExtUIOp>(&op)) {
    src = castOp.getSrc();
    result = castOp.getRes();
  } else if (auto castOp = dyn_cast<VFTruncIOp>(&op)) {
    src = castOp.getSrc();
    result = castOp.getRes();
  } else {
    return std::nullopt;
  }

  auto srcType = dyn_cast<VectorType>(src.getType());
  auto dstType = dyn_cast<VectorType>(result.getType());
  if (!srcType || !dstType || srcType.getRank() != 1 ||
      dstType.getRank() != 1 ||
      srcType.getNumElements() != dstType.getNumElements())
    return std::nullopt;

  Type srcElementType = srcType.getElementType();
  Type dstElementType = dstType.getElementType();
  if (!isSupportedConversion(op, srcElementType, dstElementType))
    return std::nullopt;

  unsigned srcBits = srcElementType.getIntOrFloatBitWidth();
  unsigned dstBits = dstElementType.getIntOrFloatBitWidth();
  unsigned smallBits = std::min(srcBits, dstBits);
  unsigned wideBits = std::max(srcBits, dstBits);
  if (smallBits == 0 || wideBits % smallBits != 0)
    return std::nullopt;

  unsigned factor = wideBits / smallBits;
  if ((factor != 2 && factor != 4) || factor > maxMergeFactor)
    return std::nullopt;

  // Both sides keep the same lane count, so using wideBits here proves that
  // the wide side of each conversion occupies exactly one physical vector.
  if (dstType.getNumElements() * static_cast<int64_t>(wideBits) !=
      static_cast<int64_t>(util::VL_BITS))
    return std::nullopt;

  return ConversionInfo{src, result, dstType, factor, srcBits < dstBits};
}

static bool isSupportedNarrowElementwise(Operation &op, VectorType narrowType) {
  if (op.getNumRegions() != 0 || op.getNumResults() != 1 ||
      op.getResult(0).getType() != narrowType)
    return false;

  return isa<AVEElementwiseOp>(&op);
}

static bool isPgeMask(Value value, int64_t expectedLanes) {
  auto maskType = dyn_cast<VectorType>(value.getType());
  auto pgeOp = value.getDefiningOp<VFPgeOp>();
  return maskType && maskType.getRank() == 1 &&
         maskType.getElementType().isInteger(1) &&
         maskType.getNumElements() == expectedLanes && pgeOp;
}

static bool hasSupportedOperands(Operation &op, VectorType narrowType) {
  for (Value operand : op.getOperands()) {
    auto vectorType = dyn_cast<VectorType>(operand.getType());
    if (!vectorType)
      continue;
    if (vectorType == narrowType)
      continue;
    if (vectorType.getElementType().isInteger(1) &&
        vectorType.getNumElements() == narrowType.getNumElements() &&
        isPgeMask(operand, narrowType.getNumElements()))
      continue;
    return false;
  }
  return true;
}

static std::optional<SmallVector<Operation *, 8>>
findNarrowStoreChain(const ConversionInfo &conversion) {
  if (conversion.widening)
    return std::nullopt;

  Operation *conversionOp = conversion.result.getDefiningOp();
  if (!conversionOp)
    return std::nullopt;

  VectorType narrowType = conversion.dstType;
  Value current = conversion.result;
  SmallVector<Operation *, 8> nodes{conversionOp};
  // Only a single-use, lane-wise chain can become one packed computation.
  // Reductions, gathers, branches and side users stop the search.
  while (current.hasOneUse()) {
    Operation *user = *current.getUsers().begin();
    if (auto store = dyn_cast<VFMaskedStoreOp>(user);
        store && store.getVal() == current) {
      Value mask = store.getMask();
      if (!isPgeMask(mask, narrowType.getNumElements()))
        return std::nullopt;
      nodes.push_back(store);
      return nodes;
    }
    if (!isSupportedNarrowElementwise(*user, narrowType) ||
        !hasSupportedOperands(*user, narrowType))
      return std::nullopt;
    nodes.push_back(user);
    current = user->getResult(0);
  }
  return std::nullopt;
}

static bool isInnermostLoop(scf::ForOp forOp) {
  bool hasInnerLoop = false;
  forOp.walk([&](scf::ForOp inner) {
    if (inner == forOp)
      return WalkResult::advance();
    hasInnerLoop = true;
    return WalkResult::interrupt();
  });
  return !hasInnerLoop;
}

static bool isOnePointStore(StoreDist pattern) {
  switch (pattern) {
  case StoreDist::ONEPT_B8:
  case StoreDist::ONEPT_B16:
  case StoreDist::ONEPT_B32:
  case StoreDist::ONEPT_B64:
    return true;
  default:
    return false;
  }
}

static bool hasContinuousOnePointStore(scf::ForOp forOp) {
  bool found = false;
  forOp.walk([&](VFMaskedStoreOp store) {
    if (store->hasAttr(kContinuousAttr) &&
        isOnePointStore(store.getPattern())) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool canUnrollLoop(scf::ForOp forOp, unsigned factor) {
  auto lowerBound = hivmave::getConstantIntValue(forOp.getLowerBound());
  auto upperBound = hivmave::getConstantIntValue(forOp.getUpperBound());
  auto step = hivmave::getConstantIntValue(forOp.getStep());
  if (!lowerBound || !upperBound || !step)
    return false;
  if (*step <= 0 || *upperBound <= *lowerBound)
    return false;
  int64_t range = *upperBound - *lowerBound;
  if (range % *step != 0)
    return false;
  int64_t tripCount = range / *step;
  const int64_t factorAsInt64 = static_cast<int64_t>(factor);
  // loopUnrollByFactor keeps the remaining full iterations in a cleanup loop.
  return tripCount >= factorAsInt64;
}

static bool isTemporaryAttr(StringRef name) {
  return name == kGroupAttr || name == kNodeAttr || name == kCloneAttr;
}

static void copyNonTemporaryAttrs(Operation &from, Operation &to) {
  for (NamedAttribute attr : from.getAttrs()) {
    if (!isTemporaryAttr(attr.getName().strref()))
      to.setAttr(attr.getName(), attr.getValue());
  }
}

static void removeTemporaryAttrs(Operation &root) {
  root.removeAttr(kGroupAttr);
  root.removeAttr(kNodeAttr);
  root.removeAttr(kCloneAttr);
  for (Region &region : root.getRegions())
    for (Block &block : region)
      for (Operation &operation : block)
        removeTemporaryAttrs(operation);
}

static SmallVector<Value, 4> splitPackedValue(Value packed, unsigned factor,
                                              Location loc,
                                              PatternRewriter &rewriter) {
  if (factor == 1)
    return {packed};

  auto packedType = cast<VectorType>(packed.getType());
  VectorType resultType = VectorType::get({packedType.getNumElements() / 2},
                                          packedType.getElementType());
  auto split = rewriter.create<VFInterleaveOp>(
      loc, resultType, resultType, packed, packed, Layout_Change::SPARSE);
  // Each vintlv level divides one dense value into its even/odd sparse halves.
  // Recursing builds the two- or four-leaf layout required by the conversions.
  SmallVector<Value, 4> leaves;
  SmallVector<Value, 4> left =
      splitPackedValue(split.getRes1(), factor / 2, loc, rewriter);
  SmallVector<Value, 4> right =
      splitPackedValue(split.getRes2(), factor / 2, loc, rewriter);
  leaves.append(left);
  leaves.append(right);
  return leaves;
}

static Value packValues(ArrayRef<Value> values, VectorType leafType,
                        Location loc, PatternRewriter &rewriter) {
  SmallVector<Value, 4> level(values);
  int64_t lanes = leafType.getNumElements();
  // Pairwise vdintlv keeps clone order while building one dense 2VL/4VL value.
  while (level.size() > 1) {
    SmallVector<Value, 4> next;
    lanes *= 2;
    VectorType resultType = VectorType::get({lanes}, leafType.getElementType());
    for (size_t i = 0; i < level.size(); i += 2) {
      auto packed = rewriter.create<VFDeInterleaveOp>(
          loc, resultType, resultType, level[i], level[i + 1],
          Layout_Change::DENSE);
      next.push_back(packed.getRes1());
    }
    level = std::move(next);
  }
  return level.front();
}

static bool areEquivalentScalarValues(ArrayRef<Value> values) {
  if (values.empty())
    return false;
  if (llvm::all_of(values,
                   [&](Value value) { return value == values.front(); }))
    return true;

  Attribute firstConstant;
  if (!matchPattern(values.front(), m_Constant(&firstConstant)))
    return false;
  return llvm::all_of(values.drop_front(), [&](Value value) {
    Attribute constant;
    return matchPattern(value, m_Constant(&constant)) &&
           constant == firstConstant;
  });
}

static bool areEquivalentPgeMasks(ArrayRef<Value> values,
                                  int64_t expectedLanes) {
  if (values.empty())
    return false;
  auto first = values.front().getDefiningOp<VFPgeOp>();
  if (!first || !isPgeMask(values.front(), expectedLanes))
    return false;
  return llvm::all_of(values.drop_front(), [&](Value value) {
    auto pge = value.getDefiningOp<VFPgeOp>();
    return pge && isPgeMask(value, expectedLanes) &&
           pge.getPatternAttr() == first.getPatternAttr();
  });
}

static Value buildMergedMask(ArrayRef<Value> masks, unsigned factor,
                             VectorType narrowType, Location loc,
                             PatternRewriter &rewriter) {
  auto pge = masks.front().getDefiningOp<VFPgeOp>();
  // Validation requires equivalent PGE masks, so concatenating F clones simply
  // multiplies the active-lane count and the mask width by F.
  uint32_t trueShape = getNumfromPgePattern(pge) * factor;
  VectorType maskType = VectorType::get({narrowType.getNumElements() * factor},
                                        rewriter.getI1Type());
  auto pattern =
      hivmave::getPgePatternAttr(
          rewriter, trueShape, static_cast<uint32_t>(maskType.getNumElements()))
          .value();
  return rewriter.create<VFPgeOp>(loc, maskType, pattern);
}

static bool canBuildPackedAccess(MemRefType memrefType, ValueRange indices) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  return succeeded(getStridesAndOffset(memrefType, strides, offset)) &&
         strides.size() == indices.size();
}

static PackedAccess buildPackedAccess(Value base, ValueRange indices,
                                      MemRefType sourceType,
                                      int64_t packedElements, Location loc,
                                      PatternRewriter &rewriter) {
  // Rebuild a contiguous view from the first narrow access's physical offset.
  auto metadata = rewriter.create<memref::ExtractStridedMetadataOp>(loc, base);
  Value linearOffset = metadata.getOffset();
  for (auto [index, stride] : llvm::zip(indices, metadata.getStrides())) {
    Value scaledIndex = rewriter.create<arith::MulIOp>(loc, index, stride);
    linearOffset =
        rewriter.create<arith::AddIOp>(loc, linearOffset, scaledIndex);
  }

  auto packedType = MemRefType::get(
      {packedElements}, sourceType.getElementType(),
      StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic, {1}),
      sourceType.getMemorySpace());
  auto packedView = rewriter.create<memref::ReinterpretCastOp>(
      loc, packedType, metadata.getBaseBuffer(),
      getAsOpFoldResult(linearOffset),
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(packedElements)},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)});
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  return {packedView.getResult(), {zero}};
}

static bool validateLoadBatch(ArrayRef<Operation *> operations,
                              unsigned factor) {
  if (operations.size() != factor)
    return false;
  auto first = dyn_cast_or_null<VFLoadOp>(operations.front());
  if (!first || first.getPattern() != LoadDist::NORM || first.getRes1())
    return false;
  VectorType type = first.getVectorType();
  for (Operation *operation : operations) {
    auto load = dyn_cast_or_null<VFLoadOp>(operation);
    if (!load || load.getPattern() != LoadDist::NORM || load.getRes1() ||
        load.getVectorType() != type || !load->hasOneUse())
      return false;
  }
  return true;
}

static bool rewriteLoadBatch(ArrayRef<Operation *> operations, unsigned factor,
                             PatternRewriter &rewriter) {
  if (!validateLoadBatch(operations, factor))
    return false;

  auto first = cast<VFLoadOp>(operations.front());
  VectorType narrowType = first.getVectorType();
  VectorType packedType = VectorType::get(
      {narrowType.getNumElements() * factor}, narrowType.getElementType());

  rewriter.setInsertionPoint(first);
  PackedAccess access = buildPackedAccess(
      first.getBase(), first.getIndices(), first.getMemRefType(),
      packedType.getNumElements(), first.getLoc(), rewriter);
  auto packedLoad = rewriter.create<VFLoadOp>(first.getLoc(), packedType,
                                              access.base, access.indices);
  copyNonTemporaryAttrs(*first.getOperation(), *packedLoad.getOperation());
  SmallVector<Value, 4> leaves =
      splitPackedValue(packedLoad.getRes(), factor, first.getLoc(), rewriter);
  if (leaves.size() != operations.size())
    return false;

  for (auto [operation, leaf] : llvm::zip(operations, leaves))
    cast<VFLoadOp>(operation).getRes().replaceAllUsesWith(leaf);
  for (size_t index = operations.size(); index > 0; --index)
    rewriter.eraseOp(operations[index - 1]);
  return true;
}

static bool
validateNarrowBatch(ArrayRef<SmallVector<Operation *, 4>> nodesByClone,
                    unsigned factor, VectorType narrowType) {
  if (nodesByClone.size() != factor || nodesByClone.empty())
    return false;
  size_t nodeCount = nodesByClone.front().size();
  if (nodeCount < 2)
    return false;
  for (ArrayRef<Operation *> nodes : nodesByClone) {
    if (nodes.size() != nodeCount)
      return false;
  }

  // One packed op can replace the clones only when their op sequence and
  // operand roles match at every position.
  for (size_t node = 0; node < nodeCount; ++node) {
    Operation *first = nodesByClone.front()[node];
    if (!first)
      return false;
    for (unsigned clone = 1; clone < factor; ++clone) {
      Operation *other = nodesByClone[clone][node];
      if (!other || other->getName() != first->getName() ||
          other->getNumOperands() != first->getNumOperands() ||
          other->getNumResults() != first->getNumResults())
        return false;
    }
  }

  for (size_t node = 1; node + 1 < nodeCount; ++node) {
    Operation *first = nodesByClone.front()[node];
    if (!isSupportedNarrowElementwise(*first, narrowType))
      return false;
    for (unsigned operandIndex = 0; operandIndex < first->getNumOperands();
         ++operandIndex) {
      SmallVector<Value, 4> values;
      for (unsigned clone = 0; clone < factor; ++clone)
        values.push_back(nodesByClone[clone][node]->getOperand(operandIndex));
      auto vectorType = dyn_cast<VectorType>(values.front().getType());
      if (!vectorType) {
        if (!areEquivalentScalarValues(values))
          return false;
        continue;
      }
      if (vectorType == narrowType)
        continue;
      if (vectorType.getElementType().isInteger(1) &&
          vectorType.getNumElements() == narrowType.getNumElements() &&
          areEquivalentPgeMasks(values, narrowType.getNumElements()))
        continue;
      return false;
    }
  }

  Operation *firstStore = nodesByClone.front().back();
  if (!isa<VFMaskedStoreOp>(firstStore))
    return false;
  SmallVector<Value, 4> storeMasks;
  for (unsigned clone = 0; clone < factor; ++clone) {
    auto store = cast<VFMaskedStoreOp>(nodesByClone[clone].back());
    storeMasks.push_back(store.getMask());
  }
  return areEquivalentPgeMasks(storeMasks, narrowType.getNumElements());
}

static bool
rewriteNarrowBatch(ArrayRef<SmallVector<Operation *, 4>> nodesByClone,
                   unsigned factor, PatternRewriter &rewriter) {
  Operation *firstConversion = nodesByClone.front().front();
  auto conversionResultType =
      dyn_cast<VectorType>(firstConversion->getResult(0).getType());
  if (!conversionResultType ||
      !validateNarrowBatch(nodesByClone, factor, conversionResultType))
    return false;

  VectorType narrowType = conversionResultType;
  VectorType packedType = VectorType::get(
      {narrowType.getNumElements() * factor}, narrowType.getElementType());
  Operation *insertionAnchor = nodesByClone.back().back();
  rewriter.setInsertionPoint(insertionAnchor);

  SmallVector<Value, 4> conversionResults;
  for (unsigned clone = 0; clone < factor; ++clone)
    conversionResults.push_back(nodesByClone[clone].front()->getResult(0));
  Value mergedValue = packValues(conversionResults, narrowType,
                                 firstConversion->getLoc(), rewriter);

  DenseMap<Value, Value> oldToMerged;
  for (Value result : conversionResults)
    oldToMerged[result] = mergedValue;

  size_t nodeCount = nodesByClone.front().size();
  for (size_t node = 1; node + 1 < nodeCount; ++node) {
    Operation *first = nodesByClone.front()[node];
    IRMapping mapping;
    for (unsigned operandIndex = 0; operandIndex < first->getNumOperands();
         ++operandIndex) {
      SmallVector<Value, 4> values;
      for (unsigned clone = 0; clone < factor; ++clone)
        values.push_back(nodesByClone[clone][node]->getOperand(operandIndex));

      Value replacement;
      auto vectorType = dyn_cast<VectorType>(values.front().getType());
      if (!vectorType) {
        replacement = values.front();
      } else if (vectorType == narrowType) {
        auto mapped = oldToMerged.find(values.front());
        bool allMapped = mapped != oldToMerged.end();
        if (allMapped) {
          replacement = mapped->second;
          for (Value value : ArrayRef<Value>(values).drop_front()) {
            auto other = oldToMerged.find(value);
            if (other == oldToMerged.end() || other->second != replacement) {
              allMapped = false;
              break;
            }
          }
        }
        if (!allMapped)
          replacement =
              packValues(values, narrowType, first->getLoc(), rewriter);
      } else {
        replacement = buildMergedMask(values, factor, narrowType,
                                      first->getLoc(), rewriter);
      }
      mapping.map(first->getOperand(operandIndex), replacement);
    }

    Operation *mergedOp = rewriter.clone(*first, mapping);
    mergedOp->getResult(0).setType(packedType);
    removeTemporaryAttrs(*mergedOp);
    mergedValue = mergedOp->getResult(0);
    for (unsigned clone = 0; clone < factor; ++clone)
      oldToMerged[nodesByClone[clone][node]->getResult(0)] = mergedValue;
  }

  SmallVector<Value, 4> storeMasks;
  for (unsigned clone = 0; clone < factor; ++clone) {
    auto store = cast<VFMaskedStoreOp>(nodesByClone[clone].back());
    storeMasks.push_back(store.getMask());
  }
  Value mergedMask =
      buildMergedMask(storeMasks, factor, narrowType,
                      nodesByClone.front().back()->getLoc(), rewriter);

  Operation *firstStore = nodesByClone.front().back();
  auto maskedStore = cast<VFMaskedStoreOp>(firstStore);
  PackedAccess access = buildPackedAccess(
      maskedStore.getBase(), maskedStore.getIndices(),
      maskedStore.getMemRefType(), packedType.getNumElements(),
      maskedStore.getLoc(), rewriter);
  Operation *newStore =
      rewriter.create<VFMaskedStoreOp>(maskedStore.getLoc(), access.base,
                                       access.indices, mergedMask, mergedValue);
  copyNonTemporaryAttrs(*firstStore, *newStore);

  for (unsigned clone = 0; clone < factor; ++clone)
    rewriter.eraseOp(nodesByClone[clone].back());
  for (size_t node = nodeCount - 1; node > 1; --node) {
    for (unsigned clone = 0; clone < factor; ++clone)
      rewriter.eraseOp(nodesByClone[clone][node - 1]);
  }
  return true;
}

static void collectGroupInstance(Operation &op, const SmallWidthGroup &group,
                                 unsigned unrollFactor, GroupInstances &byNode,
                                 bool &duplicate) {
  auto groupAttr = op.getAttrOfType<IntegerAttr>(kGroupAttr);
  if (groupAttr && groupAttr.getInt() == group.id) {
    auto nodeAttr = op.getAttrOfType<IntegerAttr>(kNodeAttr);
    auto cloneAttr = op.getAttrOfType<IntegerAttr>(kCloneAttr);
    if (nodeAttr && cloneAttr) {
      int64_t node = nodeAttr.getInt();
      int64_t clone = cloneAttr.getInt();
      if (node >= 0 && node < static_cast<int64_t>(byNode.size()) &&
          clone >= 0 && clone < static_cast<int64_t>(unrollFactor)) {
        if (byNode[node][clone])
          duplicate = true;
        byNode[node][clone] = &op;
      }
    }
  }

  for (Region &region : op.getRegions())
    for (Block &block : region)
      for (Operation &nestedOp : block)
        collectGroupInstance(nestedOp, group, unrollFactor, byNode, duplicate);
}

static std::optional<GroupInstances>
collectGroupInstances(Block &block, const SmallWidthGroup &group,
                      unsigned unrollFactor) {
  GroupInstances byNode(group.nodes.size(),
                        SmallVector<Operation *, 4>(unrollFactor, nullptr));
  bool duplicate = false;
  for (Operation &op : block)
    collectGroupInstance(op, group, unrollFactor, byNode, duplicate);
  if (duplicate)
    return std::nullopt;
  for (ArrayRef<Operation *> clones : byNode) {
    for (Operation *op : clones)
      if (!op)
        return std::nullopt;
  }
  return byNode;
}

static SmallVector<SmallWidthGroup, 8>
buildGroups(scf::ForOp forOp, unsigned maxMergeFactor, int64_t &nextGroupId) {
  SmallVector<SmallWidthGroup, 8> groups;
  // Widening conversions contribute adjacent narrow loads. Narrowing
  // conversions contribute a conversion -> narrow elementwise -> store chain.
  for (Operation &operation : forOp.getBody()->without_terminator()) {
    auto conversion = getConversionInfo(operation, maxMergeFactor);
    if (!conversion)
      continue;

    if (conversion->widening) {
      auto load = conversion->src.getDefiningOp<VFLoadOp>();
      if (!load || load->getBlock() != forOp.getBody() ||
          load.getPattern() != LoadDist::NORM || load.getRes1() ||
          !load->hasOneUse() ||
          !canBuildPackedAccess(load.getMemRefType(), load.getIndices()) ||
          !hasContiguousIterationAccess(load.getBase(), load.getIndices(),
                                        load.getMemRefType(),
                                        load.getVectorType(), forOp))
        continue;
      groups.push_back(
          {nextGroupId++, conversion->factor, GroupKind::Load, {load}});
      continue;
    }

    auto nodes = findNarrowStoreChain(*conversion);
    if (!nodes)
      continue;
    auto store = cast<VFMaskedStoreOp>(nodes->back());
    if (!canBuildPackedAccess(store.getMemRefType(), store.getIndices()) ||
        !hasContiguousIterationAccess(
            store.getBase(), store.getIndices(), store.getMemRefType(),
            cast<VectorType>(store.getVal().getType()), forOp))
      continue;
    groups.push_back({nextGroupId++, conversion->factor, GroupKind::NarrowStore,
                      std::move(*nodes)});
  }
  return groups;
}

static bool hasCloneEquivalentScalars(const SmallWidthGroup &group,
                                      scf::ForOp forOp) {
  if (group.kind != GroupKind::NarrowStore)
    return true;

  // One packed op has only one scalar operand. A scalar computed from the IV
  // would differ between clones, so only loop-invariant values or constants
  // that can be proven equivalent after unrolling are accepted.
  for (Operation *operation :
       ArrayRef<Operation *>(group.nodes).drop_front().drop_back()) {
    for (Value operand : operation->getOperands()) {
      if (isa<VectorType>(operand.getType()) ||
          forOp.isDefinedOutsideOfLoop(operand))
        continue;
      Attribute constant;
      if (!matchPattern(operand, m_Constant(&constant)))
        return false;
    }
  }
  return true;
}

static std::optional<SmallWidthMergePlan>
buildMergePlan(scf::ForOp forOp, unsigned maxMergeFactor,
               int64_t &nextGroupId) {
  SmallVector<SmallWidthGroup, 8> candidates =
      buildGroups(forOp, maxMergeFactor, nextGroupId);
  SmallWidthMergePlan plan{0, {}};
  for (SmallWidthGroup &group : candidates) {
    if (!hasCloneEquivalentScalars(group, forOp))
      continue;
    plan.unrollFactor = std::max(plan.unrollFactor, group.factor);
    plan.groups.push_back(std::move(group));
  }

  if (plan.groups.empty() || !canUnrollLoop(forOp, plan.unrollFactor))
    return std::nullopt;
  return plan;
}

static void annotateGroups(ArrayRef<SmallWidthGroup> groups,
                           OpBuilder &builder) {
  // group/node identify the original chain. The unroll callback adds clone,
  // allowing collection by (group, node, clone) instead of operation order.
  for (const SmallWidthGroup &group : groups) {
    for (auto [node, op] : llvm::enumerate(group.nodes)) {
      op->setAttr(kGroupAttr, builder.getI64IntegerAttr(group.id));
      op->setAttr(kNodeAttr, builder.getI64IntegerAttr(node));
    }
  }
}

static LogicalResult optimizeLoop(scf::ForOp forOp, unsigned maxMergeFactor,
                                  int64_t &nextGroupId,
                                  PatternRewriter &rewriter) {
  // HoistVstas cannot yet carry one continuous ONEPT stream across unrolled
  // clones. Preserve the original loop until that state can be shared.
  if (hasContinuousOnePointStore(forOp))
    return failure();

  std::optional<SmallWidthMergePlan> plan =
      buildMergePlan(forOp, maxMergeFactor, nextGroupId);
  if (!plan)
    return failure();

  unsigned unrollFactor = plan->unrollFactor;
  annotateGroups(plan->groups, rewriter);
  Block *originalParentBlock = forOp->getBlock();
  auto annotateClone = [](unsigned cloneIndex, Operation *const &operation,
                          OpBuilder builder) {
    Operation &op = *operation;
    if (op.hasAttr(kGroupAttr))
      op.setAttr(kCloneAttr, builder.getI64IntegerAttr(cloneIndex));
  };
  auto unrollResult = loopUnrollByFactor(forOp, unrollFactor, annotateClone);
  if (failed(unrollResult))
    return failure();

  Block *unrolledBody = originalParentBlock;
  if (unrollResult->mainLoopOp)
    unrolledBody = unrollResult->mainLoopOp->getBody();

  // Load groups are rewritten first so their users retain the original
  // conversion and wide-compute structure.
  unsigned successfulMerges = 0;
  for (GroupKind kind : {GroupKind::Load, GroupKind::NarrowStore}) {
    for (const SmallWidthGroup &group : plan->groups) {
      if (group.kind != kind)
        continue;
      auto instances =
          collectGroupInstances(*unrolledBody, group, unrollFactor);
      if (!instances)
        continue;
      // A factor-4 unroll can contain factor-2 groups. Process those as clone
      // pairs [0, 1] and [2, 3], preserving each group's natural width.
      for (unsigned batchStart = 0; batchStart < unrollFactor;
           batchStart += group.factor) {
        if (group.kind == GroupKind::Load) {
          SmallVector<Operation *, 4> loads;
          for (unsigned clone = batchStart; clone < batchStart + group.factor;
               ++clone)
            loads.push_back((*instances)[0][clone]);
          bool rewritten = rewriteLoadBatch(loads, group.factor, rewriter);
          assert(rewritten && "prevalidated load merge must succeed");
          successfulMerges += rewritten;
          continue;
        }

        SmallVector<SmallVector<Operation *, 4>, 4> nodesByClone;
        for (unsigned clone = batchStart; clone < batchStart + group.factor;
             ++clone) {
          SmallVector<Operation *, 4> nodes;
          for (ArrayRef<Operation *> nodeInstances : *instances)
            nodes.push_back(nodeInstances[clone]);
          nodesByClone.push_back(std::move(nodes));
        }
        bool rewritten =
            rewriteNarrowBatch(nodesByClone, group.factor, rewriter);
        assert(rewritten && "prevalidated narrow merge must succeed");
        successfulMerges += rewritten;
      }
    }
  }
  assert(successfulMerges != 0 && "merge plan must produce at least one merge");
  return success();
}

// Peel a partial data tail first, for example the final 8 elements of
// `for 0 to 200 step 64`. loopUnrollByFactor later owns any remaining complete
// iteration that cannot fill a factor-2/factor-4 group.
struct PeelEpiloguePattern : public OpRewritePattern<scf::ForOp> {
  explicit PeelEpiloguePattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOp->hasAttr("__peeled_loop__"))
      return failure();
    if (!forOp.getLowerBound().getType().isIndex() ||
        !forOp.getUpperBound().getType().isIndex() ||
        !forOp.getStep().getType().isIndex())
      return failure();
    scf::ForOp partialIteration;
    if (failed(scf::peelForLoopAndSimplifyBounds(rewriter, forOp,
                                                 partialIteration)))
      return failure();
    partialIteration->setAttr("__peeled_loop__", rewriter.getUnitAttr());
    return success();
  }
};

struct aveLoopOptimizePass
    : public impl::AveLoopOptimizeBase<aveLoopOptimizePass> {
  using Base::Base;

  void runOnOperation() override {
    // Supported shapes:
    //   narrow load -> widening conversion
    //   narrowing conversion -> supported narrow elementwise ops -> masked_store
    // The first path merges loads and keeps wide computation independent; the
    // second packs the narrow computation and its final store.
    // The pass only rewrites innermost VF loops with provably adjacent accesses
    // and complete clone groups. Dynamic/nonlinear addresses, cross-lane chains,
    // unsupported masks/types and continuous ONEPT stores remain unchanged.
    if (maxSmallWidthMergeFactor != 2 && maxSmallWidthMergeFactor != 4) {
      getOperation().emitError()
          << "max-small-width-merge-factor must be 2 or 4";
      return signalPassFailure();
    }

    auto funcOp = getOperation();
    if (!hivm::isVF(funcOp))
      return;

    RewritePatternSet peelPatterns(&getContext());
    peelPatterns.add<PeelEpiloguePattern>(&getContext());
    if (failed(applyPatternsGreedily(funcOp, std::move(peelPatterns))))
      return signalPassFailure();

    SmallVector<scf::ForOp, 8> loops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (isInnermostLoop(forOp))
        loops.push_back(forOp);
    });

    PatternRewriter rewriter(&getContext());
    int64_t nextGroupId = 0;
    for (scf::ForOp forOp : loops) {
      if (!forOp || !forOp->getBlock())
        continue;
      if (failed(optimizeLoop(forOp, maxSmallWidthMergeFactor, nextGroupId,
                              rewriter)))
        removeTemporaryAttrs(*forOp.getOperation());
    }
    removeTemporaryAttrs(*funcOp.getOperation());

    RewritePatternSet canonicalizationPatterns(&getContext());
    scf::ForOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                            &getContext());
    affine::AffineApplyOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                       &getContext());
    if (failed(
            applyPatternsGreedily(funcOp, std::move(canonicalizationPatterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
hivmave::createAveLoopOptimizePass(const AveLoopOptimizeOptions &options) {
  return std::make_unique<aveLoopOptimizePass>(options);
}

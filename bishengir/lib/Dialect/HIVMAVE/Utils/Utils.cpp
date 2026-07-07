//===- Utils.cpp - Utilities to support the AVE dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the AVE dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cassert>
#include <climits>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::hivmave;
using namespace mlir::hivm::util;

Value hivmave::castToDstVectorType(Value src, VectorType dstTy, OpBuilder &b) {
  VectorType srcTy = dyn_cast<VectorType>(src.getType());
  assert(srcTy);
  if (srcTy == dstTy)
    return src;
  if (trimNonScalableUnitDims(srcTy) == trimNonScalableUnitDims(dstTy)) {
    return b.create<vector::ShapeCastOp>(src.getLoc(), dstTy, src);
  }
  return b.create<UnrealizedConversionCastOp>(src.getLoc(), dstTy, src)
      ->getResult(0);
}

ValueRange hivmave::getIndices(mlir::Operation *op) {
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto vectorMaskStoreOp = dyn_cast<vector::MaskedStoreOp>(op))
    return vectorMaskStoreOp.getIndices();
  if (auto vectorMaskLoadOp = dyn_cast<vector::MaskedLoadOp>(op))
    return vectorMaskLoadOp.getIndices();
  llvm_unreachable("unsupported op type");
}

Value hivmave::computeLinearMemRefOffset(OpBuilder &builder, Location loc,
                                         Value memref, ValueRange indices,
                                         Type resultType) {
  assert((resultType.isIndex() || isa<IntegerType>(resultType)) &&
         "result type must be index or integer");

  auto memrefType = cast<MemRefType>(memref.getType());
  int64_t rank = memrefType.getRank();

  auto makeConst = [&](int64_t value) -> Value {
    if (resultType.isIndex())
      return builder.create<arith::ConstantIndexOp>(loc, value);
    return builder.create<arith::ConstantOp>(
        loc, IntegerAttr::get(resultType, value));
  };
  auto castToResultType = [&](Value value) -> Value {
    if (value.getType() == resultType)
      return value;
    return builder.create<arith::IndexCastOp>(loc, resultType, value);
  };

  SmallVector<Value, 4> strides(rank);
  if (auto stridedLayout =
          dyn_cast<StridedLayoutAttr>(memrefType.getLayout())) {
    ArrayRef<int64_t> layoutStrides = stridedLayout.getStrides();
    std::optional<memref::ExtractStridedMetadataOp> metadata;
    for (int64_t i = 0; i < rank; ++i) {
      if (!ShapedType::isDynamic(layoutStrides[i])) {
        strides[i] = makeConst(layoutStrides[i]);
        continue;
      }

      if (!metadata)
        metadata =
            builder.create<memref::ExtractStridedMetadataOp>(loc, memref);
      strides[i] = castToResultType(metadata->getStrides()[i]);
    }
  } else {
    Value strideValue = makeConst(1);
    for (int64_t i = rank - 1; i >= 0; --i) {
      strides[i] = strideValue;
      if (i == 0)
        continue;

      Value dimSize;
      if (memrefType.isDynamicDim(i)) {
        dimSize = builder.create<memref::DimOp>(loc, memref, i);
        dimSize = castToResultType(dimSize);
      } else {
        dimSize = makeConst(memrefType.getDimSize(i));
      }
      strideValue = builder.create<arith::MulIOp>(loc, strideValue, dimSize);
    }
  }

  Value linearOffset = makeConst(0);
  for (auto [idx, index] : llvm::enumerate(indices)) {
    Value indexValue = castToResultType(index);
    Value product =
        builder.create<arith::MulIOp>(loc, indexValue, strides[idx]);
    linearOffset = builder.create<arith::AddIOp>(loc, linearOffset, product);
  }
  return linearOffset;
}

static bool isIntegerMultipleOf(int64_t value, int64_t factor) {
  return value % factor == 0;
}

static bool isValueKnownMultipleOf(Value value, int64_t factor);

// For an affine expression to be known as a multiple of factor, its result must
// always be a multiple of factor.
static bool isAffineExprKnownMultipleOf(AffineExpr expr, ValueRange operands,
                                        unsigned numDims, int64_t factor) {
  assert(factor > 0 && "factor must be positive");
  if (factor == 1)
    return true;

  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
    return isIntegerMultipleOf(constExpr.getValue(), factor);

  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    unsigned pos = dimExpr.getPosition();
    if (pos >= operands.size())
      return false;
    return isValueKnownMultipleOf(operands[pos], factor);
  }

  if (auto symbolExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    unsigned pos = numDims + symbolExpr.getPosition();
    if (pos >= operands.size())
      return false;
    return isValueKnownMultipleOf(operands[pos], factor);
  }

  auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binaryExpr)
    return false;

  switch (expr.getKind()) {
  case AffineExprKind::Add:
    // For lhs + rhs to stay a multiple, both lhs and rhs must already be
    // multiples.
    return isAffineExprKnownMultipleOf(binaryExpr.getLHS(), operands, numDims,
                                       factor) &&
           isAffineExprKnownMultipleOf(binaryExpr.getRHS(), operands, numDims,
                                       factor);
  case AffineExprKind::Mul: {
    AffineExpr lhs = binaryExpr.getLHS();
    AffineExpr rhs = binaryExpr.getRHS();
    // For lhs * rhs to be a multiple, a constant factor can satisfy all or part
    // of the requested factor; the other side must satisfy the rest.
    auto proveConstMul =
        [operands, numDims, factor](AffineExpr maybeConst,
                                    AffineExpr other) -> std::optional<bool> {
      auto constExpr = dyn_cast<AffineConstantExpr>(maybeConst);
      if (!constExpr)
        return std::nullopt;

      int64_t constantFactor = constExpr.getValue();
      if (constantFactor < 0) {
        if (constantFactor == std::numeric_limits<int64_t>::min())
          return false;
        constantFactor = -constantFactor;
      }
      if (constantFactor == 0 ||
          isIntegerMultipleOf(constantFactor, factor))
        return true;
      if (factor % constantFactor != 0)
        return false;
      return isAffineExprKnownMultipleOf(other, operands, numDims,
                                         factor / constantFactor);
    };

    if (std::optional<bool> result = proveConstMul(lhs, rhs))
      return *result;
    if (std::optional<bool> result = proveConstMul(rhs, lhs))
      return *result;

    return isAffineExprKnownMultipleOf(lhs, operands, numDims, factor) ||
           isAffineExprKnownMultipleOf(rhs, operands, numDims, factor);
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    // For lhs floordiv divisor or lhs ceildiv divisor to stay a multiple, lhs
    // must be a multiple of factor * divisor.
    auto rhsConst = dyn_cast<AffineConstantExpr>(binaryExpr.getRHS());
    if (!rhsConst || rhsConst.getValue() <= 0)
      return false;
    int64_t divisor = rhsConst.getValue();
    if (factor > std::numeric_limits<int64_t>::max() / divisor)
      return false;
    return isAffineExprKnownMultipleOf(binaryExpr.getLHS(), operands, numDims,
                                       factor * divisor);
  }
  case AffineExprKind::Mod: {
    // For lhs mod modulus to be a multiple, prove it is zero or prove both lhs
    // and modulus are already multiples.
    auto rhsConst = dyn_cast<AffineConstantExpr>(binaryExpr.getRHS());
    if (!rhsConst || rhsConst.getValue() <= 0)
      return false;
    int64_t modulus = rhsConst.getValue();
    AffineExpr lhs = binaryExpr.getLHS();
    return isAffineExprKnownMultipleOf(lhs, operands, numDims, modulus) ||
           (isIntegerMultipleOf(modulus, factor) &&
            isAffineExprKnownMultipleOf(lhs, operands, numDims, factor));
  }
  default:
    return false;
  }
}

// For a value to be known as a multiple of factor, follow only simple SSA
// producers whose multiple-of property can be proven locally.
static bool isValueKnownMultipleOf(Value value, int64_t factor) {
  assert(factor > 0 && "factor must be positive");
  if (factor == 1)
    return true;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    // Only the induction variable follows lowerBound + n * step. Loop-carried
    // iter_args need separate yield analysis, so keep them conservatively
    // unproven.
    if (!forOp || blockArg != forOp.getInductionVar())
      return false;
    // For lowerBound + n * step to stay a multiple, both lowerBound and step
    // must already be multiples.
    Value start = forOp.getLowerBound();
    Value step = forOp.getStep();
    return isValueKnownMultipleOf(start, factor) &&
           isValueKnownMultipleOf(step, factor);
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  } else if (arith::ConstantOp constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    // For a constant value to be known as a multiple, it must be a multiple of
    // the requested factor.
    auto intVal = getConstantIntValue(constOp.getResult());
    if (intVal)
      return isIntegerMultipleOf(*intVal, factor);
  } else if (auto affineApplyOp = dyn_cast<affine::AffineApplyOp>(defOp)) {
    AffineMap map = affineApplyOp.getAffineMap();
    if (map.getNumResults() == 1)
      return isAffineExprKnownMultipleOf(map.getResult(0),
                                         affineApplyOp.getMapOperands(),
                                         map.getNumDims(), factor);
  }
  // Unknown producers are treated conservatively as unproven multiples.
  return false;
}

bool static isOffsetAligned(Value memrefVal,
                            llvm::SmallVector<mlir::OpFoldResult, 4u> indices,
                            int64_t hwAlignBits, int64_t elemBits) {
  auto srcMemRefType = mlir::cast<MemRefType>(memrefVal.getType());
  auto [srcStrides, srcOffset] = getStridesAndOffset(srcMemRefType);
  for (size_t i = 0; i < indices.size(); i++) {
    bool isOffsetAlign = false;
    if (auto v = dyn_cast<Value>(indices[i])) {
      int64_t requiredElementOffsetAlignment = hwAlignBits / elemBits;
      isOffsetAlign = isValueKnownMultipleOf(v, requiredElementOffsetAlignment);
    } else if (Attribute attr = dyn_cast<Attribute>(indices[i])) {
      int64_t staticVal = dyn_cast<IntegerAttr>(attr).getValue().getSExtValue();
      isOffsetAlign = (staticVal * elemBits) % hwAlignBits == 0;
    }
    int64_t stride = srcStrides[i];
    bool isStrideAlign = (stride * elemBits) % hwAlignBits == 0;
    if (!isOffsetAlign && !isStrideAlign) {
      return false;
    }
  }
  return true;
}

// Analyze whether the starting address of the memref object meets the alignment
// requirements.
static bool isMemrefAligned(Value memrefVal, int64_t hwAlignBits,
                            int64_t elemBits) {
  Operation *defOp = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
    defOp = blockArg.getOwner()->getParentOp();
  } else {
    defOp = memrefVal.getDefiningOp();
  }
  if (dyn_cast<func::FuncOp>(defOp)) {
    // Check whether the memref object is a function parameter.
    // The memref object address in the parameters is aligned.
    return true;
  } else if (auto subViewOp = dyn_cast<memref::SubViewOp>(defOp)) {
    // If the memref object comefrom SubViewOp. Compute the offset as follow:
    // result_offset = src_offset + dot_product(offset_operands, src_strides)
    auto srcValue = subViewOp.getSource();
    if (!isMemrefAligned(srcValue, hwAlignBits, elemBits)) {
      return false;
    }
    auto offsetOperands = subViewOp.getMixedOffsets();
    return isOffsetAligned(srcValue, offsetOperands, hwAlignBits, elemBits);
  } else if (auto viewOp = dyn_cast<memref::ViewOp>(defOp)) {
    auto srcValue = viewOp.getSource();
    if (!isMemrefAligned(srcValue, hwAlignBits, elemBits)) {
      return false;
    }
    bool isByteShiftAlign = false;
    if (auto v = dyn_cast<Value>(viewOp.getByteShift())) {
      // The source memref alignment is checked above; byte_shift is measured in
      // bytes, so prove that the byte displacement preserves HW alignment.
      int64_t requiredByteOffsetAlignment = hwAlignBits / 8;
      isByteShiftAlign = isValueKnownMultipleOf(v, requiredByteOffsetAlignment);
    }
    return isByteShiftAlign;
  } else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(defOp)) {
    auto srcValue = collapseOp.getSrc();
    llvm::errs() << "collapseOp src: " << srcValue << "\n";
    return isMemrefAligned(srcValue, hwAlignBits, elemBits);
  } else if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(defOp)) {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        getGlobal, getGlobal.getNameAttr());

    if (!global)
      return false;

    if (auto alignAttr = global.getAlignmentAttr()) {
      int64_t alignBytes = alignAttr.getInt();
      if (hwAlignBits % 8 != 0)
        return false;
      int64_t hwAlignBytes = hwAlignBits / 8;
      return alignBytes >= hwAlignBytes;
    }
  }
  return false;
}

bool hivmave::isLoadStoreIndexAligned(Value memrefVal,
                                      mlir::Operation::operand_range indices) {
  auto srcMemRefType = mlir::cast<MemRefType>(memrefVal.getType());
  int64_t elemBits = static_cast<int64_t>(
      getElementTypeOrSelf(srcMemRefType).getIntOrFloatBitWidth());
  int64_t hwAlignBits = static_cast<int64_t>(
      hivm::getHWAlignBytes(srcMemRefType.getMemorySpace()) * 8);
  if (!isMemrefAligned(memrefVal, hwAlignBits, elemBits)) {
    return false;
  }
  return isOffsetAligned(memrefVal, indices, hwAlignBits, elemBits);
}

uint32_t hivmave::getNumfromPgePattern(VFPgeOp pge) {
  uint32_t res = -1;
  PgePattern pgePattern = pge.getPattern();
  switch (pgePattern) {
  case PgePattern::VL1:
    res = 1;
    break;
  case PgePattern::VL2:
    res = 2;
    break;
  case PgePattern::VL3:
    res = 3;
    break;
  case PgePattern::VL4:
    res = 4;
    break;
  case PgePattern::VL8:
    res = 8;
    break;
  case PgePattern::VL16:
    res = 16;
    break;
  case PgePattern::VL32:
    res = 32;
    break;
  case PgePattern::VL64:
    res = 64;
    break;
  case PgePattern::VL128:
    res = 128;
    break;
  case PgePattern::ALL: {
    res =
        static_cast<uint32_t>(cast<VectorType>(pge.getType()).getNumElements());
    break;
  }
  case PgePattern::ALLF:
    res = 0;
    break;
  default:
    break;
  }
  return res;
}

FailureOr<PgePatternAttr> hivmave::getPgePatternAttr(PatternRewriter &rewriter,
                                                     int64_t trueShape,
                                                     int64_t resultShape) {
  // TODO: cover the scenarios of pge pattern.
  if (resultShape == trueShape) {
    return PgePatternAttr::get(rewriter.getContext(), PgePattern::ALL);
  }
  PgePattern pat = PgePattern::ALLF;
  switch (trueShape) {
  case 0:
    pat = PgePattern::ALLF;
    break;
  case 1:
    pat = PgePattern::VL1;
    break;
  case 2:
    pat = PgePattern::VL2;
    break;
  case 3:
    pat = PgePattern::VL3;
    break;
  case 4:
    pat = PgePattern::VL4;
    break;
  case 8:
    pat = PgePattern::VL8;
    break;
  case 16:
    pat = PgePattern::VL16;
    break;
  case 32:
    pat = PgePattern::VL32;
    break;
  case 64:
    pat = PgePattern::VL64;
    break;
  case 128:
    pat = PgePattern::VL128;
    break;
  default:
    return failure();
  }

  return PgePatternAttr::get(rewriter.getContext(), pat);
}

Value hivmave::getElemSizeByStoreMask(Value mask, Type dElemType, Location loc,
                                      PatternRewriter &rewriter, bool getCnt) {
  // vstus need an argument element size. Get it from mask.
  Operation *defOp = mask.getDefiningOp();
  Value elemSize;
  uint32_t typeSizeBit = dElemType.getIntOrFloatBitWidth();
  uint32_t typeSizeByte = typeSizeBit / 8;
  while (defOp) {
    if (auto plt = dyn_cast<VFPltOp>(defOp)) {
      auto trueShape = plt.getTrueShape();
      if (auto constantOp =
              dyn_cast<arith::ConstantOp>(trueShape.getDefiningOp())) {
        IntegerAttr attr = dyn_cast<IntegerAttr>(constantOp.getValue());
        uint32_t elemCnt = (uint32_t)(attr.getValue().getZExtValue());
        elemSize =
            getCnt
                ? rewriter.create<arith::ConstantOp>(
                      loc, rewriter.getI32IntegerAttr(elemCnt))
                : rewriter.create<arith::ConstantOp>(
                      loc, rewriter.getI32IntegerAttr(elemCnt * typeSizeByte));
      } else {
        auto typeSizeVal = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(typeSizeByte));
        Value elemCnt;
        if (trueShape.getType().isIndex()) {
          elemCnt = rewriter
                        .create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                    trueShape)
                        .getResult();
        } else {
          elemCnt = rewriter
                        .create<arith::TruncIOp>(loc, rewriter.getI32Type(),
                                                 trueShape)
                        .getResult();
        }
        elemSize =
            getCnt ? elemCnt
                   : rewriter.create<arith::MulIOp>(loc, rewriter.getI32Type(),
                                                    elemCnt, typeSizeVal);
      }
      break;
    } else if (auto pge = dyn_cast<VFPgeOp>(defOp)) {
      uint32_t elemCnt = getNumfromPgePattern(pge);
      elemSize =
          getCnt ? rewriter.create<arith::ConstantOp>(
                       loc, rewriter.getI32IntegerAttr(elemCnt))
                 : rewriter.create<arith::ConstantOp>(
                       loc, rewriter.getI32IntegerAttr(elemCnt * typeSizeByte));
      break;
    } else if (auto cast = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
      defOp = cast->getOperand(0).getDefiningOp();
    } else if (auto extract = dyn_cast<LLVM::ExtractValueOp>(defOp)) {
      defOp = extract->getOperand(0).getDefiningOp();
    } else {
      llvm::errs() << "not process yet " << *defOp << "\n";
      break;
    }
  };
  if (!elemSize) {
    uint32_t constVal = getCnt ? hivm::util::VL / typeSizeByte : hivm::util::VL;
    elemSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(constVal));
  }
  return elemSize;
}

Value hivmave::createMaskByPGE(VectorType vecTy, PatternRewriter &rewriter,
                               Location loc, bool allTrue) {
  Value mask;
  auto maskType = VectorType::get(SmallVector<int64_t>{vecTy.getNumElements()},
                                  rewriter.getI1Type());
  auto pgePattern =
      allTrue ? hivmave::PgePattern::ALL : hivmave::PgePattern::ALLF;
  auto maskOp = rewriter.create<hivmave::VFPgeOp>(
      loc, maskType,
      hivmave::PgePatternAttr::get(rewriter.getContext(), pgePattern));
  mask = maskOp->getResult(0);
  if (vecTy.getElementTypeBitWidth() == 1 && vecTy != maskType)
    mask = rewriter
               .create<UnrealizedConversionCastOp>(rewriter.getUnknownLoc(),
                                                   vecTy, maskOp.getResult())
               ->getResult(0);
  return mask;
}

Value hivmave::findReuseableMask(Operation *maskedOp,
                                 PatternRewriter &rewriter) {
  Value mask;
  if (utils::getAnnotateOpWithAttr(maskedOp->getResults()[0],
                                   utils::reachedMaskOpsIdx)) {
    annotation::MarkOp mark = dyn_cast<annotation::MarkOp>(
        utils::getAnnotateOpWithAttr(maskedOp->getResults()[0],
                                     utils::reachedMaskOpsIdx)
            .value());
    int reachedMaskOpIdx = static_cast<int>(
        mark->template getAttrOfType<IntegerAttr>(utils::reachedMaskOpsIdx)
            .getValue()
            .getZExtValue());
    auto funcOp = maskedOp->getParentOfType<func::FuncOp>();
    funcOp->walk([&](Operation *op) {
      if (op->getNumResults() > 0 &&
          utils::getAnnotateOpWithAttr(op->getResults()[0], utils::maskOpIdx)) {
        annotation::MarkOp candidateMaskOpMark = dyn_cast<annotation::MarkOp>(
            utils::getAnnotateOpWithAttr(op->getResults()[0], utils::maskOpIdx)
                .value());
        int candidateMaskOpIdx =
            candidateMaskOpMark
                ->template getAttrOfType<IntegerAttr>(utils::maskOpIdx)
                .getValue()
                .getZExtValue();
        if (reachedMaskOpIdx == candidateMaskOpIdx) {
          DominanceInfo domInfo(op);
          if (domInfo.dominates(op, maskedOp)) {
            mask = op->getResults()[0];
          } else if (op->getBlock() == maskedOp->getBlock()) {
            Operation *cloneOp = rewriter.clone(*op);
            mask = cloneOp->getResults()[0];
          }
        }
      }
    });
  }
  return mask;
}

Value hivmave::findReuseableMaskOrCreateOne(Operation *maskedOp,
                                            VectorType vecTy,
                                            PatternRewriter &rewriter) {
  Value mask = findReuseableMask(maskedOp, rewriter);
  if (!mask)
    mask = hivmave::createMaskByPGE(vecTy, rewriter, maskedOp->getLoc());
  return mask;
}

template <bool DropUnitDimOnly>
static VectorType getLegalizedVectorType(VectorType source) {
  Type elemTy = source.getElementType();
  if constexpr (DropUnitDimOnly) {
    return trimNonScalableUnitDims(source);
  } else {
    return VectorType::get(
        SmallVector<int64_t>{
            hivm_regbaseintrins::getVectorSizeByElementType(elemTy)},
        elemTy);
  }
}

template <bool DropUnitDimOnly>
static Value adjustVectorType(PatternRewriter &rewriter, VectorType resultTy,
                              Value src) {
  if constexpr (DropUnitDimOnly)
    // Use shape cast to drop unit dims to exploit the vector dialect fold
    // patterns
    return rewriter.create<vector::ShapeCastOp>(src.getLoc(), resultTy, src);

  // shape_cast cannot cast something like <1xf32> to <64xf32>
  return rewriter
      .create<UnrealizedConversionCastOp>(src.getLoc(), resultTy, src)
      .getResult(0);
}

template <bool DropUnitDimOnly>
LogicalResult hivmave::ForOpLegalization<DropUnitDimOnly>::matchAndRewrite(
    scf::ForOp op, PatternRewriter &rewriter) const {
  // if the for op has a vector type iter_arg and the shape is not supported
  // by the hardware, we rewrite the shape
  OperandRange iterArgs = op.getInitArgs();
  SmallVector<Value> newIterArgs, newYields;
  SmallVector<unsigned> modified;
  for (unsigned i = 0; i < iterArgs.size(); i++) {
    if (op.getRegionIterArg(i).use_empty())
      continue;
    if (VectorType vecTy = dyn_cast<VectorType>(iterArgs[i].getType())) {
      VectorType adjustedType = getLegalizedVectorType<DropUnitDimOnly>(vecTy);

      if (vecTy.getShape().size() > 1 ||
          adjustedType.getNumElements() != vecTy.getNumElements()) {
        // need to adjust the iter arg
        // do this by making a new iter_arg of the supported type and replace
        // all use of the old iter arg with this new one. Leave the old one
        // for the canonicalizer to clean up.
        modified.push_back(i);

        rewriter.setInsertionPoint(op);
        Value adjustedIterArg = adjustVectorType<DropUnitDimOnly>(
            rewriter, adjustedType, iterArgs[i]);
        newIterArgs.push_back(adjustedIterArg);

        rewriter.setInsertionPoint(op.getBody()->getTerminator());
        Value adjustedYieldedValue = adjustVectorType<DropUnitDimOnly>(
            rewriter, adjustedType, op.getYieldedValues()[i]);
        newYields.push_back(adjustedYieldedValue);
      }
    }
  }

  if (newIterArgs.empty())
    return failure();

  rewriter.setInsertionPointAfter(op);
  NewYieldValuesFn fn =
      [&](OpBuilder &innerBuilder, Location loc,
          ArrayRef<BlockArgument> innerNewBBArgs) -> SmallVector<Value> {
    return newYields;
  };
  scf::ForOp newForOp = cast<scf::ForOp>(
      *op.replaceWithAdditionalYields(rewriter, newIterArgs, false, fn));

  int idx = 0;
  for (unsigned i = 0; i < iterArgs.size(); i++) {
    if (std::find(modified.begin(), modified.end(), i) != modified.end()) {
      rewriter.setInsertionPointAfter(newForOp);
      Value adjustedResult = adjustVectorType<DropUnitDimOnly>(
          rewriter, cast<VectorType>(newForOp.getResult(i).getType()),
          newForOp.getResult(iterArgs.size() + idx));
      rewriter.replaceAllUsesWith(newForOp.getResult(i), adjustedResult);
      rewriter.setInsertionPointToStart(newForOp.getBody());
      Value adjustedArg = adjustVectorType<DropUnitDimOnly>(
          rewriter, cast<VectorType>(newForOp.getRegionIterArg(i).getType()),
          newForOp.getRegionIterArg(iterArgs.size() + idx));
      rewriter.replaceAllUsesWith(newForOp.getRegionIterArg(i), adjustedArg);
      idx++;
    }
  }

  return success();
}

template struct hivmave::ForOpLegalization<true>;
template struct hivmave::ForOpLegalization<false>;

std::optional<int64_t> hivmave::getConstantIntValue(Value val) {
  if (!val)
    return std::nullopt;

  // 1. Handle arith.constant
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }

  // 2. Handle llvm.constant
  if (auto constOp = val.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }

  // 3. Recursively handle IndexCast
  if (auto castOp = val.getDefiningOp<arith::IndexCastOp>()) {
    return getConstantIntValue(castOp.getIn());
  }

  // 4. Handle UnrealizedConversionCast
  if (auto castOp = val.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() == 1) {
      return getConstantIntValue(castOp.getInputs()[0]);
    }
  }

  // 5. Handle Function Arguments (via "hivm.constant_value" attribute)
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(val)) {
    Block *ownerBlock = blockArg.getOwner();
    Operation *parentOp = ownerBlock->getParentOp();

    if (auto funcOp = mlir::dyn_cast<func::FuncOp>(parentOp)) {
      if (&funcOp.getBody().front() == ownerBlock) {
        unsigned argIdx = blockArg.getArgNumber();
        if (auto attr = funcOp.getArgAttrOfType<IntegerAttr>(
                argIdx, "hivm.constant_value")) {
          return attr.getInt();
        }
      }
    }
  }

  return std::nullopt;
}

void hivmave::tagConstantArguments(ModuleOp module) {
  // Local debug logger helper using the passed tag
  SymbolTable symbolTable(module);
  module.walk([&](func::CallOp callOp) {
    auto calleeFunc = symbolTable.lookup<func::FuncOp>(callOp.getCallee());
    if (!calleeFunc)
      return;

    for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
      if (i >= calleeFunc.getNumArguments())
        break;

      std::optional<int64_t> constVal =
          getConstantIntValue(callOp.getOperand(i));
      if (constVal.has_value()) {
        OpBuilder builder(calleeFunc.getContext());
        calleeFunc.setArgAttr(i, "hivm.constant_value",
                              builder.getI64IntegerAttr(*constVal));
      }
    }
  });
}

Operation *hivmave::getBroadcastOp(Value scalar, VectorType tileType,
                                   PatternRewriter &rewriter,
                                   const Location &loc) {
  auto maskType = VectorType::get(
      SmallVector<int64_t>{tileType.getNumElements()}, rewriter.getI1Type());
  auto mask = rewriter.create<hivmave::VFPgeOp>(
      loc, maskType,
      hivmave::PgePatternAttr::get(rewriter.getContext(),
                                   hivmave::PgePattern::ALL));
  auto broadcastmaskOp = rewriter.create<hivmave::VFBroadcastScalarMaskOp>(
      loc, tileType, scalar, mask);
  return broadcastmaskOp;
}

Value hivmave::sparseByIntlv(Value src, RewriterBase &rewriter,
                             const Location &loc, Attribute attr) {
  hivmave::VFInterleaveOp interOp = rewriter.create<hivmave::VFInterleaveOp>(
      loc, ArrayRef<Type>({src.getType(), src.getType()}),
      ValueRange{src, src});
  if (attr)
    interOp->setAttr(utils::elementAlignmentBitWidth, attr);
  return interOp.getResult(0);
}

Value hivmave::denseByDIntlv(Value src, RewriterBase &rewriter,
                             const Location &loc, Attribute attr) {
  hivmave::VFDeInterleaveOp deionOp = rewriter.create<hivmave::VFDeInterleaveOp>(
      loc, ArrayRef<Type>({src.getType(), src.getType()}),
      ValueRange{src, src});
  if (attr)
    deionOp->setAttr(utils::elementAlignmentBitWidth, attr);
  return deionOp.getResult(0);
}

int hivmave::getBitWidthFromAttr(Operation *Op) {
  auto funcDistAttr = Op->getAttrOfType<FunctionDistTypeAttr>("functionType");
  int elementAlignment = -1;
  if (funcDistAttr) {
    switch (funcDistAttr.getValue()) {
    case FunctionDistType::PB8:
      elementAlignment = 8;
      break;
    case FunctionDistType::PB16:
      elementAlignment = 16;
      break;
    case FunctionDistType::PB32:
      elementAlignment = 32;
      break;
    default:
      break;
    }
  }
  return elementAlignment;
}

Value hivmave::constrainVectorLayout(Value src, VecMemType targetLayout,
                                     OpBuilder &builder) {
  // Verify the input is a VectorType
  auto vecType = dyn_cast<VectorType>(src.getType());
  assert(vecType && "input value must be of VectorType");

  // Create the target layout attribute
  auto layoutAttr = VectorLayoutAttr::get(
      builder.getContext(),
      VecMemTypeAttr::get(builder.getContext(), targetLayout));

  // First cast: None -> targetLayout
  auto castToLayout = builder.create<VectorLayoutCastOp>(
      src.getLoc(), vecType.cloneWith(layoutAttr), src);

  // Second cast: targetLayout -> None
  auto castToNone = builder.create<VectorLayoutCastOp>(
      src.getLoc(), vecType.cloneWith({}), castToLayout.getResult());

  return castToNone.getResult();
}

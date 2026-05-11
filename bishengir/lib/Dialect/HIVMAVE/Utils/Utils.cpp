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
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cassert>

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

static bool checkValueAligned(Value v, int64_t hwAlignBits, int64_t elemBits) {
  Operation *defOp = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    defOp = blockArg.getOwner()->getParentOp();
  } else {
    defOp = v.getDefiningOp();
  }
  if (!defOp) {
    return false;
  } else if (scf::ForOp forOp = dyn_cast<scf::ForOp>(defOp)) {
    // In for op, if the start value and step value all aligned both,
    // the iterative variables can be considered aligned.
    Value start = forOp.getLowerBound();
    Value step = forOp.getStep();
    return checkValueAligned(start, hwAlignBits, elemBits) &&
           checkValueAligned(step, hwAlignBits, elemBits);
  } else if (arith::ConstantOp constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    // Check constant value is aligned.
    auto intVal = getConstantIntValue(constOp.getResult());
    if (intVal) {
      return (*intVal * elemBits) % hwAlignBits == 0;
    } else {
      return false;
    }
  }
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
      isOffsetAlign = checkValueAligned(v, hwAlignBits, elemBits);
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
      isByteShiftAlign = checkValueAligned(v, hwAlignBits, elemBits);
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
    res = static_cast<uint32_t>(cast<VectorType>(pge.getType()).getNumElements());
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
  return mask;
}

Value hivmave::findReuseableMask(Operation *maskedOp, PatternRewriter &rewriter) {
  Value mask;
  if (utils::getAnnotateOpWithAttr(maskedOp->getResults()[0], utils::reachedMaskOpsIdx)) {
    annotation::MarkOp mark = dyn_cast<annotation::MarkOp>(
        utils::getAnnotateOpWithAttr(maskedOp->getResults()[0], utils::reachedMaskOpsIdx)
            .value());
    int reachedMaskOpIdx =
         static_cast<int>(mark->template getAttrOfType<IntegerAttr>(utils::reachedMaskOpsIdx)
            .getValue().getZExtValue());
    auto funcOp = maskedOp->getParentOfType<func::FuncOp>();
    funcOp->walk([&](Operation *op) {
      if (op->getNumResults() > 0 &&
          utils::getAnnotateOpWithAttr(op->getResults()[0], utils::maskOpIdx)) {
        annotation::MarkOp candidateMaskOpMark = dyn_cast<annotation::MarkOp>(
            utils::getAnnotateOpWithAttr(op->getResults()[0], utils::maskOpIdx).value());
        int candidateMaskOpIdx =
            candidateMaskOpMark->template getAttrOfType<IntegerAttr>(utils::maskOpIdx)
                .getValue().getZExtValue();
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

Value hivmave::findReuseableMaskOrCreateOne(
    Operation *maskedOp, VectorType vecTy, PatternRewriter &rewriter) {
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
  if (!val) return std::nullopt;

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
        if (auto attr = funcOp.getArgAttrOfType<IntegerAttr>(argIdx, "hivm.constant_value")) {
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
    if (!calleeFunc) return;

    for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
      if (i >= calleeFunc.getNumArguments()) break;

      std::optional<int64_t> constVal = getConstantIntValue(callOp.getOperand(i));
      if (constVal.has_value()) {
        OpBuilder builder(calleeFunc.getContext());
        calleeFunc.setArgAttr(i, "hivm.constant_value", builder.getI64IntegerAttr(*constVal));
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

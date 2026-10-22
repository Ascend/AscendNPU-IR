//===---------------------- NormalizeVector.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "bishengir/Dialect/Vector/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <limits>
#define DEBUG_TYPE "normalize-vector"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZEVECTOR
#include "bishengir/Dialect/Vector/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::hivm;
using namespace mlir::hivm::util;

namespace {

struct NormalizeVectorPass
    : public impl::NormalizeVectorBase<NormalizeVectorPass> {
  using NormalizeVectorBase<NormalizeVectorPass>::NormalizeVectorBase;

  void runOnOperation() override;
};

} // namespace

// Generate dense attribute with init value
static DenseElementsAttr
getDenseAttrForSelect(VectorType srcType, PatternRewriter &rewriter,
                      vector::CombiningKind reduceKind) {
  // For max related reduction kind, only use the minimize value as
  // 'false' element indicated by outerMask could ensure correctness,
  // For min related reduction kind, use the maximum value.
  // For add/or, use 0, and mul/and, use 1.
  Type elemTy = srcType.getElementType();
  DenseElementsAttr denseAttr;
  if (elemTy.isF32() || elemTy.isF16()) {
    const llvm::fltSemantics &semantics = mlir::cast<FloatType>(elemTy).getFloatSemantics();
    llvm::APFloat initVal(semantics);
    switch (reduceKind) {
    case vector::CombiningKind::MAXNUMF:
    case vector::CombiningKind::MAXIMUMF:
      initVal = llvm::APFloat::getInf(semantics, /*Negative*/true);
      break;
    case vector::CombiningKind::MINNUMF:
    case vector::CombiningKind::MINIMUMF:
      initVal = llvm::APFloat::getInf(semantics);
      break;
    case vector::CombiningKind::ADD:
      initVal = llvm::APFloat::getZero(semantics);
      break;
    case vector::CombiningKind::MUL:
      initVal = llvm::APFloat::getOne(semantics);
      break;
    default:
      llvm::report_fatal_error("Unsupport reduce kind.");
      break;
    }
    denseAttr =
        DenseElementsAttr::get(srcType, rewriter.getFloatAttr(elemTy, initVal));
  } else if (elemTy.isUnsignedInteger() || elemTy.isInteger()) {
    llvm::APInt initVal;
    unsigned width = elemTy.getIntOrFloatBitWidth();
    switch (reduceKind) {
    case vector::CombiningKind::ADD:
    case vector::CombiningKind::OR:
    case vector::CombiningKind::XOR:
      initVal = llvm::APInt::getZero(width);
      break;
    case vector::CombiningKind::MUL:
      initVal = llvm::APInt(width, 1);
      break;
    case vector::CombiningKind::AND:
      initVal = llvm::APInt::getAllOnes(width);
      break;
    case vector::CombiningKind::MAXSI:
      initVal = llvm::APInt::getSignedMinValue(width);
      break;
    case vector::CombiningKind::MINSI:
      initVal = llvm::APInt::getSignedMaxValue(width);
      break;
    case vector::CombiningKind::MAXUI:
      initVal = llvm::APInt::getMinValue(width);
      break;
    case vector::CombiningKind::MINUI:
      initVal = llvm::APInt::getMaxValue(width);
      break;
    default:
      llvm::report_fatal_error("Unsupport reduce kind.");
      break;
    }
    denseAttr = DenseElementsAttr::get(
        srcType, rewriter.getIntegerAttr(elemTy, initVal));
  } else {
    llvm::report_fatal_error("Unsupport element data type.");
  }

  return denseAttr;
}

/// Rewrites vector.create_mask 'op' to drop non-scalable one dimensions.
static FailureOr<Value>
createMaskDropNonScalableUnitDims(PatternRewriter &rewriter, Location loc,
                                  vector::CreateMaskOp op) {
  auto type = op.getType();
  VectorType reducedType = trimNonScalableUnitDims(type);
  if (reducedType.getRank() == type.getRank())
    return failure();

  SmallVector<Value> reducedOperands;
  for (auto [dim, dimIsScalable, operand] : llvm::zip_equal(
           type.getShape(), type.getScalableDims(), op.getOperands())) {
    if (dim == 1 && !dimIsScalable) {
      // If the mask for the unit dim is not a constant of 1, do nothing.
      auto constant = operand.getDefiningOp<arith::ConstantIndexOp>();
      if (!constant || (constant.value() != 1))
        return failure();
      continue;
    }
    reducedOperands.push_back(operand);
  }
  return rewriter
      .create<vector::CreateMaskOp>(loc, reducedType, reducedOperands)
      .getResult();
}

static SmallVector<int64_t>
getNonOneReductionShape(ArrayAttr reductionDimsAttr, VectorType vecType) {
  auto vecShape = vecType.getShape();
  auto values = reductionDimsAttr.getAsRange<IntegerAttr>();
  SmallVector<int64_t> reductionDims;

  for (auto intAttr : values)
    reductionDims.push_back(intAttr.getInt());

  SmallVector<int64_t> nonOneDim;
  for (int64_t dim : reductionDims) {
    if (vecShape[dim] != 1) {
      // We found a non-unit reduction dimension
      nonOneDim.push_back(vecShape[dim]);
    }
  }
  return nonOneDim;
}

static FailureOr<Value> maskDropNonScalableUnitDims(PatternRewriter &rewriter,
                                                    Location loc, Value mask) {
  VectorType maskType = cast<VectorType>(mask.getType());
  if (!isOneDimLikeVecType(maskType))
    return failure();

  if (auto createMaskOp = mask.getDefiningOp<vector::CreateMaskOp>()) {
    FailureOr<Value> rankReducedCreateMask =
        createMaskDropNonScalableUnitDims(rewriter, loc, createMaskOp);
    if (failed(rankReducedCreateMask))
      return failure();
    return *rankReducedCreateMask;
  } else if (mask.getDefiningOp<vector::ConstantMaskOp>()) {
    VectorType castedMaskType =
        VectorType::get(ArrayRef<int64_t>{maskType.getShape().back()},
                        maskType.getElementType());
    return rewriter.create<vector::ShapeCastOp>(loc, castedMaskType, mask)
        .getResult();
  }

  return failure();
}

/// Converts multi-dim vector.multi_reduction into vector.reduction
/// clang-format off
/// Before conversion:
/// ```mlir
///   %reduction = vector.mask %mask {
///                vector.multi_reduction <add>, %v, %acc[1] : vector<1x64xf32>
///                to vector<1xf32> } : vector<1x64xi1> -> vector<1xf32>
/// ```
/// After conversion:
/// ```mlir
///   %acc_scalar = builtin.unrealized_conversion_cast %acc : vector<1xf32> to
///   f32
///   %reduction = vector.mask %mask {
///                vector.reduction <add>, %v, %acc_scalar : vector<64xf32> into
///                f32 } : vector<64xi1> -> f32
///   %cast = builtin.unrealized_conversion_cast %reduction : f32 to
///   vector<1xf32>
/// ```
/// clang-format on
struct OneDimAccMultiReductionToReduction
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {

    auto dst = multiReductionOp.getDest();
    // unit dim reduction is handle in another pattern
    if (dst.getType().isIntOrFloat())
      return failure();

    auto reductionDims = multiReductionOp.getReductionDims();

    Value src = multiReductionOp.getSource();
    VectorType srcType = cast<VectorType>(src.getType());

    auto nonOneDim = getNonOneReductionShape(reductionDims, srcType);
    assert(nonOneDim.size() > 0);

    Value acc = multiReductionOp.getAcc();
    VectorType accType = cast<VectorType>(acc.getType());
    if (accType.getRank() != 1 && accType.getShape()[0] != 1)
      return failure();

    // Vector mask setup.
    OpBuilder::InsertionGuard guard(rewriter);
    auto maskableOp =
        cast<vector::MaskableOpInterface>(multiReductionOp.getOperation());
    Operation *rootOp;
    if (maskableOp.isMasked()) {
      rewriter.setInsertionPoint(maskableOp.getMaskingOp());
      rootOp = maskableOp.getMaskingOp();
    } else
      rootOp = multiReductionOp;

    Location loc = multiReductionOp.getLoc();
    Type elemType = srcType.getElementType();

    VectorType castedSrcType = VectorType::get({srcType.getNumElements()}, elemType);
    Value castedSrc =
        rewriter.create<vector::ShapeCastOp>(loc, castedSrcType, src);
    Value castedAcc =
        rewriter.create<UnrealizedConversionCastOp>(loc, elemType, acc)
            .getResult(0);
    Operation *reductionOp = rewriter.create<vector::ReductionOp>(
        loc, multiReductionOp.getKind(), castedSrc, castedAcc);
    if (multiReductionOp->hasAttr("withoutInitMergeOp"))
      reductionOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());

    // If masked, slice the mask and mask the new reduction operation.
    if (maskableOp.isMasked()) {
      Value mask = maskableOp.getMaskingOp().getMask();
      FailureOr<Value> maskDropDims =
          maskDropNonScalableUnitDims(rewriter, loc, mask);
      if (failed(maskDropDims))
        return failure();
      mask = *maskDropDims;
      reductionOp = mlir::vector::maskOperation(rewriter, reductionOp, mask);
    }

    Value result = rewriter
                       .create<UnrealizedConversionCastOp>(
                           loc, dst.getType(), reductionOp->getResult(0))
                       .getResult(0);

    rewriter.replaceOp(rootOp, result);
    return success();
  }
};

/// Add `withoutInitMergeOp` attribute to `vector.multi_reduction` if the init
/// value comes from function argument.
/// Then VectorToHIVMAVE will apply to reduction op with `withoutInitMergeOp`
/// attribute to remove unnecessary operations for reduce init value.
///
/// before:
///   %1 = vector.broadcast %arg0 : f32 to vector<1xf32>
///   vector.multi_reduction %0, %1 [1] : vector<1x64xf32> to vector<1xf32>
///
/// after `TagMultiReductionInit` and `OneDimAccMultiReductionToReduction`:
///   %1 = vector.broadcast %arg0 : f32 to vector<1xf32>
///   %2 = builtin.unrealized_conversion_cast %1 : vector<1xf32> to f32
///   vector.reduction %2, %3 {withoutInitMergeOp} : vector<64xf32> into f32
///
/// after ConvertVectorToHIVMAVE:
///   %1 = ave.hir.reduction <max>, ...: vector<64xf32>, vector<64xi1>
///   this will not be generated: arith.maximumf %..., %1 : vector<64xf32>
struct TagMultiReductionInit
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Check if the attribute already exists
    StringRef attrName = "withoutInitMergeOp";
    if (op->hasAttr(attrName)) {
      return failure();
    }

    // 2. Get the accumulator/init operand.
    Value acc = op.getAcc();

    // 3. Look through vector.broadcast if present.
    // We check if the value is defined by a BroadcastOp. If so, we swap 'acc'
    // to look at the source of the broadcast instead.
    if (auto broadcastOp = acc.getDefiningOp<vector::BroadcastOp>()) {
      acc = broadcastOp.getSource();
    }

    // 4. Mark Attribute if the value comes from Arith Scalar Constant.
    auto constAcc = acc.getDefiningOp<arith::ConstantOp>();
    if (constAcc && constAcc.getType().isIntOrIndexOrFloat()) {
      markAttr(op, attrName, rewriter);
      return success();
    }

    // 5. Verify the value is a BlockArgument.
    auto blockArg = dyn_cast<BlockArgument>(acc);
    if (!blockArg) {
      return failure();
    }

    // 6. Verify the BlockArgument belongs to a Function (and is not an iter
    // arg).
    // - Get the block owning the argument.
    // - Check if the parent operation is a func::FuncOp.
    // - Check if this block is the entry block of that function.
    // Note: Loop iter args (scf.for) belong to blocks nested inside the
    // function, not the entry block.
    Block *ownerBlock = blockArg.getOwner();
    auto funcOp = dyn_cast_or_null<func::FuncOp>(ownerBlock->getParentOp());
    if (!funcOp || !ownerBlock->isEntryBlock()) {
      return failure();
    }

    markAttr(op, attrName, rewriter);
    return success();
  }

private:
  void markAttr(Operation *op, StringRef attrName,
                PatternRewriter &rewriter) const {
    rewriter.modifyOpInPlace(
        op, [&] { op->setAttr(attrName, rewriter.getUnitAttr()); });
  }
};

/// Converts one-dim vector.multi_reduction into vector.reduction
/// clang-format off
/// Before conversion:
/// ```mlir
///  %acc_scalar = vector.extractelement %acc[] : vector<f32>
///  %reduction = vector.multi_reduction <add>, %v, %acc_scalar[0] :
///  vector<64xf32> to f32
///  %brc = vector.broadcast %reduction : f32 to vector<64xf32>
/// ```
/// After conversion:
/// ```mlir
///  %acc_scalar = builtin.unrealized_conversion_cast %acc : vector<f32> to f32
///  %reduction = vector.reduction <add>, %v, %acc_scalar : vector<64xf32> into
///  f32
///  %cast = builtin.unrealized_conversion_cast %reduction : f32 to vector<f32>
///  %brc = vector.broadcast %cast : vector<f32> to vector<64xf32>
/// ```
/// clang-format on
struct UnitDimMultiReductionToReduction
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    Value src = multiReductionOp.getSource();
    Value dst = multiReductionOp.getDest();
    Value acc = multiReductionOp.getAcc();
    VectorType srcType = cast<VectorType>(src.getType());
    if (!isOneDimLikeVecType(srcType) || !dst.getType().isIntOrFloat() ||
        !acc.getType().isIntOrFloat()) {
      return failure();
    }
    auto maskableOp =
        cast<vector::MaskableOpInterface>(multiReductionOp.getOperation());
    if (maskableOp.isMasked()) {
      return rewriteReductionWithMask(multiReductionOp, rewriter);
    }
    return rewriteReductionWithoutMask(multiReductionOp, rewriter);
  }

  Value castVectorShapeIfNeedConvert(VectorType vecType, Value vec,
                                     bool needConvert,
                                     PatternRewriter &rewriter,
                                     Location &loc) const {
    if (needConvert) {
      auto vecCastedType =
          VectorType::get(vecType.getShape().back(), vecType.getElementType());
      return rewriter.create<vector::ShapeCastOp>(loc, vecCastedType, vec);
    }
    return vec;
  }

  LogicalResult
  rewriteReductionWithoutMask(vector::MultiDimReductionOp multiReductionOp,
                              PatternRewriter &rewriter) const {
    Value src = multiReductionOp.getSource();
    Value dst = multiReductionOp.getDest();
    Value acc = multiReductionOp.getAcc();
    VectorType srcType = cast<VectorType>(src.getType());
    Operation *rootOp = multiReductionOp;
    Operation *user = *dst.getUsers().begin();
    Location loc = rootOp->getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    auto accDefOp = acc.getDefiningOp<vector::ExtractElementOp>();
    if (!accDefOp || dst.use_empty()) {
      return failure();
    }
    Value curSrc = src;
    bool needConvert = srcType.getRank() > 1;
    curSrc =
        castVectorShapeIfNeedConvert(srcType, src, needConvert, rewriter, loc);
    auto dstBrc = dyn_cast<vector::BroadcastOp>(user);
    if (!dstBrc) {
      return failure();
    }
    auto accUcc = rewriter.create<UnrealizedConversionCastOp>(
        loc, acc.getType(), accDefOp.getVector());
    auto reductionOp = rewriter.create<vector::ReductionOp>(
        loc, multiReductionOp.getKind(), curSrc, accUcc->getResult(0));
    if (multiReductionOp->hasAttr("withoutInitMergeOp"))
      reductionOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
    if (utils::isScalarLike(dstBrc)) {
      auto dstUcc = rewriter.create<UnrealizedConversionCastOp>(
          loc, dstBrc.getResultVectorType(), reductionOp->getResult(0));
      rewriter.replaceAllUsesWith(dstBrc.getResult(), dstUcc->getResult(0));
    } else {
      auto dstUcc = rewriter.create<UnrealizedConversionCastOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>(),
                          dstBrc.getResultVectorType().getElementType()),
          reductionOp->getResult(0));
      rewriter.modifyOpInPlace(dstBrc, [&]() {
        dstBrc.getSourceMutable().set(dstUcc->getResult(0));
      });
    }
    rewriter.replaceOp(rootOp, reductionOp);
    return success();
  }

  LogicalResult
  rewriteReductionWithMask(vector::MultiDimReductionOp multiReductionOp,
                           PatternRewriter &rewriter) const {
    Value src = multiReductionOp.getSource();
    Value dst = multiReductionOp.getDest();
    Value acc = multiReductionOp.getAcc();
    VectorType srcType = cast<VectorType>(src.getType());
    auto maskableOp =
        cast<vector::MaskableOpInterface>(multiReductionOp.getOperation());
    Operation *rootOp = maskableOp.getMaskingOp();
    Operation *user = *rootOp->getOpResult(0).getUsers().begin();
    Location loc = rootOp->getLoc();
    rewriter.setInsertionPoint(maskableOp.getMaskingOp());
    OpBuilder::InsertionGuard guard(rewriter);
    auto accDefOp = acc.getDefiningOp<vector::ExtractElementOp>();
    if (!accDefOp || dst.use_empty()) {
      return failure();
    }
    Value curSrc = src;
    bool needConvert = srcType.getRank() > 1;
    curSrc =
        castVectorShapeIfNeedConvert(srcType, src, needConvert, rewriter, loc);
    auto dstBrc = dyn_cast<vector::BroadcastOp>(user);
    if (!dstBrc) {
      return failure();
    }
    auto accUcc = rewriter.create<UnrealizedConversionCastOp>(
        loc, acc.getType(), accDefOp.getVector());
    vector::CombiningKind reduceKind = multiReductionOp.getKind();
    bool maskBySelect = false;
    Value outerMask = maskableOp.getMaskingOp().getMask();

    Value srcMask;
    Operation *srcDefOp = src.getDefiningOp();
    if (isa<vector::TransferReadOp>(srcDefOp)) {
      srcMask = dyn_cast<vector::TransferReadOp>(srcDefOp).getMask();
    } else if (isa<vector::MaskedLoadOp>(srcDefOp)) {
      srcMask = dyn_cast<vector::MaskedLoadOp>(srcDefOp).getMask();
    }
    // For vector.mask %mask {vector.multi_reduction <kind> %vec, %scalar}
    // If %mask is different with mask used in %vec defining, we should select
    // elements in %vec first using %mask, then generate a new
    // vector.reduction <kind> %new_vec %scalar to replace the vector.mask,
    // this could ensure every element masked by %mask in %vec participate
    // in reduction.
    if (outerMask != srcMask) {
      // Select src vector using outerMask
      auto vectorMask = castVectorShapeIfNeedConvert(
          llvm::cast<VectorType>(outerMask.getType()), outerMask, needConvert,
          rewriter, loc);
      DenseElementsAttr denseAttr = getDenseAttrForSelect(
          cast<VectorType>(curSrc.getType()), rewriter, reduceKind);
      arith::ConstantOp cstOpForSel =
          rewriter.create<arith::ConstantOp>(loc, denseAttr);
      arith::SelectOp selOp = rewriter.create<arith::SelectOp>(
          loc, vectorMask, curSrc, cstOpForSel);
      curSrc = selOp->getResult(0);
      maskBySelect = true;
    }
    auto reductionOp = rewriter.create<vector::ReductionOp>(
        loc, multiReductionOp.getKind(), curSrc, accUcc->getResult(0));
    Operation *res = reductionOp;
    if (!maskBySelect) {
      Value vectorMask = maskableOp.getMaskingOp().getMask();
      vectorMask = castVectorShapeIfNeedConvert(
          llvm::cast<VectorType>(vectorMask.getType()), vectorMask, needConvert,
          rewriter, loc);
      res = mlir::vector::maskOperation(rewriter, reductionOp, vectorMask);
    }
    if(utils::isScalarLike(dstBrc)){
      auto dstUcc = rewriter.create<UnrealizedConversionCastOp>(
        loc, dstBrc.getResultVectorType(), res->getResult(0));
      rewriter.replaceAllUsesWith(dstBrc.getResult(), dstUcc->getResult(0));
    } else {
      auto dstUcc = rewriter.create<UnrealizedConversionCastOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>(),
                          dstBrc.getResultVectorType().getElementType()),
          res->getResult(0));
      rewriter.modifyOpInPlace(dstBrc, [&]() {
        dstBrc.getSourceMutable().set(dstUcc->getResult(0));
      });
    }
    rewriter.replaceOp(rootOp, res);
    return success();
  }
};

/// Convert multi-dim vector::broadcast
/// clang-format off
/// Before conversion:
/// ```mlir
/// %6 = vector.xxx : vector<1x64xf32>
/// %7 = vector.broadcast %6 : vector<1x64xf32> to vector<1x1x64xf32>
/// ```
/// After Conversion:
/// ```mlir
/// %6 = vector.xxx : vector<1x64xf32>
/// %7 = vector.shape_cast %6 : vector<1x64xf32> to vector<1x1x64xf32>
/// ```
/// This pattern will also take effect on other broadcast form such as
/// vector<1x64xf32> to vector<64x1x1xf32>
/// clang-format on
struct ShapeCastUnitDimsForBrc : public OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<VectorType>(op.getSourceType())) {
      return failure();
    }
    VectorType srcType = cast<VectorType>(op.getSourceType());
    VectorType dstType = op.getResultVectorType();
    if (srcType.getRank() != dstType.getRank() &&
        srcType.getNumElements() == dstType.getNumElements()) {
      Value dupSrc = op.getSource();
      Location loc = op->getLoc();
      auto findValidDim = [=](int64_t size) { return size != 1; };
      if (llvm::count_if(srcType.getShape(), findValidDim) == 1 &&
          llvm::count_if(dstType.getShape(), findValidDim) == 1) {
        auto reshapeOp =
            rewriter.create<vector::ShapeCastOp>(loc, dstType, dupSrc);
        rewriter.replaceOp(op, reshapeOp);
        return success();
      }
    }
    return failure();
  }
};

/// Handle specific multi-dim vector::transpose pattern
/// clang-format off
/// Before conversion:
/// ```mlir
/// %8 = vector.xxx %7 : vector<1x1x64xf32>
/// %9 = vector.transpose %8, [1, 0, 2]: vector<1x1x64xf32> to
///                                      vector<1x1x64xf32>
/// %10 = vector.xxxx %9 : vector<1x1x64xf32>
/// ```
/// The size of reshaped dims are all one, so the use of vector.transpose result
/// could be replaced by its operand vector
/// After Conversion:
/// ```mlir
/// %8 = vector.xxx %7 : vector<1x1x64xf32>
/// %10 = vector.xxxx %8 : vector<1x1x64xf32>
/// ```
/// clang-format on
struct UnitDimsVecTransposeNormalize
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    VectorType srcTy = op.getSourceVectorType();
    if (srcTy == op.getResultVectorType()) {
      auto vec = op.getVector();
      auto permArr = op.getPermutation();
      auto shapes = srcTy.getShape();
      for (size_t i = 0; i < permArr.size(); i++) {
        if (permArr[i] != ssize_t(i) && shapes[i] != 1) {
          return failure();
        }
      }
      op.getResult().replaceAllUsesWith(vec);
      return success();
    }
    return failure();
  }
};

namespace {

/// copy from VectorTransferOpTransforms.cpp
/// Returns the number of dims that aren't unit dims.
static int getReducedRank(ArrayRef<int64_t> shape) {
  return llvm::count_if(shape, [](int64_t dimSize) { return dimSize != 1; });
}

/// This is a copy with the same function in VectorTransferOpTransforms.cpp
/// Rewrites vector.constant_mask 'op' to drop non-scalable one dimensions.
/// before conversion:
/// %4 = vector.constant_mask [1, 1, 1, 1, 6] : vector<1x1x1x1x64xi1>
/// after conversion:
/// %4 = vector.constant_mask [6] : vector<64xi1>
static FailureOr<Value>
constMaskDropNonScalableUnitDims(PatternRewriter &rewriter, Location loc,
                                 vector::ConstantMaskOp constMaskOP) {

  ArrayRef<Attribute> maskDimSizes = constMaskOP.getMaskDimSizes().getValue();
  size_t numMaskOperands = maskDimSizes.size();
  if (numMaskOperands == 1)
    return failure();
  VectorType reducedType = trimNonScalableUnitDims(
      cast<VectorType>(constMaskOP->getResultTypes()[0]));
  auto origIndex =
      cast<IntegerAttr>(maskDimSizes[numMaskOperands - 1]).getInt();
  IntegerAttr maskIndexAttr = rewriter.getI64IntegerAttr(origIndex);
  SmallVector<Attribute> newMaskDimSizes;
  newMaskDimSizes.push_back(maskIndexAttr);
  return rewriter
      .create<vector::ConstantMaskOp>(loc, reducedType,
                                      rewriter.getArrayAttr(newMaskDimSizes))
      .getResult();
}

/// This is a copy with the same function in VectorTransferOpTransforms.cpp
static FailureOr<Value>
dropNonScalableUnitDimsForMask(PatternRewriter &rewriter, Location loc,
                               Value maskOp) {
  auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
  auto constMaskOP = maskOp.getDefiningOp<vector::ConstantMaskOp>();
  if (createMaskOp) {
    FailureOr<Value> rankReducedCreateMask =
        createMaskDropNonScalableUnitDims(rewriter, loc, createMaskOp);
    if (failed(rankReducedCreateMask))
      return failure();
    return rankReducedCreateMask;
  } else if (constMaskOP) {
    FailureOr<Value> rankReducedConstMask =
        constMaskDropNonScalableUnitDims(rewriter, loc, constMaskOP);
    if (failed(rankReducedConstMask))
      return failure();
    return rankReducedConstMask;
  } else {
    return failure();
  }
}

/// Drop unit dims for shape_cast op, but if the mask has only one dim, do
/// nothing :
/// %0 = vector.constant_mask [1, 1, 1, 1, 6] : vector<1x1x1x1x64xi1>
/// %1 = vector.shape_cast %0 : vector<1x1x1x1x64xi1> to vector<64xi1>
/// %2 = op .. %1

/// after conversion:

/// %3 = vector.constant_mask [6] : vector<64xi1>
/// %4 = op .. %3
class ShapeCastDropUnitDimsForMultiDimVectorPattern
    : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {
    auto loc = shapeCastOp.getLoc();
    Value dstVec = shapeCastOp.getResult();
    VectorType dstType = dyn_cast<VectorType>(dstVec.getType());
    Value srcVec = shapeCastOp.getSource();
    VectorType srcType = dyn_cast<VectorType>(srcVec.getType());
    if (!dstType || !srcType)
      return failure();
    if (dstType.getRank() != 1)
      return failure();
    if (getReducedRank(srcType.getShape()) > 1)
      return failure();

    auto reducedMask = dropNonScalableUnitDimsForMask(rewriter, loc, srcVec);
    if (failed(reducedMask))
      return failure();
    dstVec.replaceAllUsesWith(reducedMask.value());
    return success();
  }
};

/// ```mlir
/// %low, %high = arith.mulsi_extended %arg0, %zero: vector<1x64xi32>
///  ==>
/// %0 = vector.shape_cast %arg0 : vector<1x64xi32> to vector<64xi32>
/// %1 = vector.shape_cast %zero : vector<1x64xi32> to vector<64xi32>
/// %low, %high = arith.mulsi_extended %arg0, %zero: vector<64xi32>
/// %2 = vector.shape_cast %high : vector<64xi32> to vector<1x64xi32>
/// ```
template <typename MulExtendedOp>
struct DropUnitDimFromMulExtendedOp : public OpRewritePattern<MulExtendedOp> {
  using OpRewritePattern<MulExtendedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulExtendedOp op,
                                PatternRewriter &rewriter) const override {

    auto resultLowType = dyn_cast<VectorType>(op.getLow().getType());
    auto resultHighType = dyn_cast<VectorType>(op.getHigh().getType());
    if (!resultLowType || !resultHighType || resultLowType.getRank() < 2)
      return failure();

    auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
    auto rhsType = dyn_cast<VectorType>(op.getRhs().getType());
    if (!lhsType || !rhsType || lhsType.getRank() < 2)
      return failure();

    bool hasUnitDim =
        llvm::any_of(lhsType.getShape(), [](int64_t dim) { return dim == 1; });
    if (!hasUnitDim)
      return rewriter.notifyMatchFailure(op, "No unit dimension to remove.");

    auto loc = op.getLoc();
    auto flattenType = [&](VectorType type) -> VectorType {
      SmallVector<int64_t> newShape;
      for (int64_t dim : type.getShape()) {
        if (dim != 1)
          newShape.push_back(dim);
      }
      return VectorType::get(newShape, type.getElementType());
    };

    Value flatLhs = rewriter.create<vector::ShapeCastOp>(
        loc, flattenType(lhsType), op.getLhs());
    Value flatRhs = rewriter.create<vector::ShapeCastOp>(
        loc, flattenType(rhsType), op.getRhs());

    auto newResultType = flattenType(resultLowType);
    auto newOp = rewriter.create<MulExtendedOp>(loc,
                                                newResultType, // low
                                                newResultType, // high
                                                flatLhs, flatRhs);

    Value newLow = rewriter.create<vector::ShapeCastOp>(loc, resultLowType,
                                                        newOp.getLow());
    Value newHigh = rewriter.create<vector::ShapeCastOp>(loc, resultHighType,
                                                         newOp.getHigh());

    rewriter.replaceOp(op, {newLow, newHigh});
    return success();
  }
};

/// Pattern to convert multi-dimensional create_mask operations to 1D and cast
/// back

/// Before:

///  %6 = vector.create_mask %c1, %0 : vector<1x64xi1>
///  %7 = vector.transfer_read %subview_1[%c0, %c0], %cst, %6
// {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1) -> (0,
// d0, d1)>} : memref<1x?xf32, strided<[255, 1], offset: ?>,
// #hivm.address_space<ub>>, vector<1x1x64xf32>

/// After conversion:

///  %6 = vector.create_mask %0 : vector<64xi1>
///  %7 = vector.shape_cast %6 : vector<64xi1> to vector<1x64xi1>
///  %8 = vector.transfer_read %subview_1[%c0, %c0], %cst, %7 {in_bounds =
///  [true, true, true], permutation_map = affine_map<(d0, d1) -> (0, d0, d1)>}
///  : memref<1x?xf32, strided<[255, 1], offset: ?>, #hivm.address_space<ub>>,
///  vector<1x1x64xf32>
/// Pattern to convert multi-dimensional create_mask operations to 1D and cast
/// back
struct ConvertCreateMaskTo1D : public OpRewritePattern<vector::CreateMaskOp> {
  using OpRewritePattern<vector::CreateMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::CreateMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto maskType = dyn_cast<VectorType>(op.getType());
    if (!maskType || maskType.getRank() <= 1)
      return failure();
    Location loc = op.getLoc();
    auto shape = maskType.getShape();
    if (op->getNumOperands() != shape.size())
      return failure();

    // Multiplying the per-dim bounds only yields the right *positions* when
    // the active region is a row-major prefix.  With dynamic bounds we cannot
    // prove that in general, but we can prove it for the shape that actually
    // occurs after unit-dim folding: every dim except the innermost is a unit
    // dim, so the outer coordinates are always 0 and the innermost bound alone
    // decides the prefix length.
    //
    // For a genuinely multi-dim dynamic mask (e.g. <2x32xi1> with bounds
    // (%r, %c)) the product %r * %c would move active lanes, exactly like the
    // constant-mask case, so bail out rather than emit a wrong mask.
    //
    // TODO: this rejects every non-unit outer dim, including bounds that are
    // a valid row-major prefix at runtime (<2x32xi1> with (%rows, %c32) is one
    // whenever %rows is in [0, 2]).  Clamp each bound to [0, shape[d]] and
    // prove the active region is a prefix before flattening, so dynamic
    // prefixes are supported too.  Refusing to rewrite is correct but
    // conservative; restoring the unconditional product would not be.
    for (unsigned i = 0; i + 1 < shape.size(); ++i) {
      if (shape[i] != 1)
        return rewriter.notifyMatchFailure(
            op, "dynamic multi-dim mask active region may not be a prefix");
    }

    // All outer dims are unit dims: their bound must be exactly 1 for the
    // lanes to start at 0.  A bound of 0 makes the whole mask empty, which the
    // product below would also express, but a non-constant bound could be
    // either, so require a constant 1.
    for (unsigned i = 0; i + 1 < shape.size(); ++i) {
      APInt boundVal;
      if (!matchPattern(op->getOperand(i), m_ConstantInt(&boundVal)) ||
          !boundVal.isOne())
        return rewriter.notifyMatchFailure(
            op, "unit-dim mask bound is not a constant 1");
    }

    // Only the innermost bound is meaningful now.
    Value totalSize = op->getOperand(shape.size() - 1);

    int64_t returnTotalElements = maskType.getNumElements();
    VectorType flatMaskType =
        VectorType::get({returnTotalElements}, rewriter.getI1Type());
    // create the 1D mask with a single operand
    Value flatMask = rewriter.create<vector::CreateMaskOp>(
        loc, flatMaskType, ValueRange{totalSize});
    // cast back to original shape so consumers get the right shape
    Value originalShapeMask =
        rewriter.create<vector::ShapeCastOp>(loc, maskType, flatMask);

    rewriter.replaceOp(op, originalShapeMask);
    return success();
  }
};
struct ConvertConstantMaskTo1D
    : public OpRewritePattern<vector::ConstantMaskOp> {
  using OpRewritePattern<vector::ConstantMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ConstantMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto maskType = dyn_cast<VectorType>(op.getType());
    if (!maskType || maskType.getRank() <= 1)
      return failure();
    Location loc = op.getLoc();
    // get the mask dimension sizes attribute
    auto maskDimSizesAttr = op.getMaskDimSizes();
    if (!maskDimSizesAttr)
      return failure();
    auto shape = maskType.getShape();
    if (maskDimSizesAttr.size() != shape.size())
      return failure();

    SmallVector<int64_t> maskBounds;
    for (Attribute attr : maskDimSizesAttr.getValue()) {
      auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
      if (!intAttr)
        return failure();
      maskBounds.push_back(
          static_cast<int64_t>(intAttr.getValue().getZExtValue()));
    }

    // Flattening a multi-dim mask to `constant_mask [N]` (a prefix of N true
    // lanes) is only valid when the active region really is a prefix of the
    // row-major lane order.  That holds iff every dim that keeps more than
    // one active slice has all of its inner dims fully active:
    //
    //   mask [2, 32] on <2x32>  -> lanes 0..63   true   (prefix, OK)
    //   mask [1, 16] on <2x32>  -> lanes 0..15   true   (prefix, OK)
    //   mask [2, 16] on <2x32>  -> [T16, F16, T16, F16] (NOT a prefix)
    //
    // Multiplying the bounds of a non-prefix mask (2 * 16 = 32 ->
    // `constant_mask [32]`) keeps the active lane *count* but moves the
    // active lane *positions*: lanes 16..31 flipped to true and 32..47 to
    // false.  Bail out instead of silently changing semantics; the mask then
    // stays multi-dimensional for the consumer to handle.
    for (int64_t d = 0; d < static_cast<int64_t>(shape.size()); ++d) {
      if (maskBounds[d] <= 1)
        continue;
      for (int64_t inner = d + 1; inner < static_cast<int64_t>(shape.size());
           ++inner) {
        if (maskBounds[inner] != shape[inner])
          return rewriter.notifyMatchFailure(
              op, "multi-dim mask active region is not a row-major prefix");
      }
    }

    // The active region is a prefix, so its length is simply the product of
    // the per-dim bounds.
    int64_t returnTotalElements = maskType.getNumElements();
    int64_t activeLanes = 1;
    for (int64_t d = 0; d < static_cast<int64_t>(shape.size()); ++d)
      activeLanes *= maskBounds[d];

    // create 1D mask type (flat)
    VectorType flatMaskType =
        VectorType::get({returnTotalElements}, rewriter.getI1Type());
    auto flatMaskDimSizesAttr = rewriter.getI64ArrayAttr({activeLanes});
    // create the 1D constant_mask and cast back to original shape
    Value flatMask = rewriter.create<vector::ConstantMaskOp>(
        loc, flatMaskType, flatMaskDimSizesAttr);
    Value originalShapeMask =
        rewriter.create<vector::ShapeCastOp>(loc, maskType, flatMask);
    rewriter.replaceOp(op, originalShapeMask);
    return success();
  }
};

struct FixBitTypeBroadcastTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  // Rewrites:
  //   %r = vector.transfer_read %src[...], %pad
  //       {permutation_map = <with broadcast>}
  //       : tensor<...xi1>, vector<...xi1>
  // into:
  //   %read   = vector.transfer_read %src[...], %pad
  //             {in_bounds = <preserved from source>}
  //             : tensor<...xi1>, vector<...xi1>  // broadcast dims -> 1
  //   %one    = arith.constant dense<1.0> : vector<...xf16>
  //   %zero0  = arith.constant dense<0.0> : vector<...xf16>
  //   %sel    = arith.select %read, %one, %zero0 : vector<...xf16>
  //   %bcast  = vector.broadcast %sel : vector<...xf16> to vector<...xf16>
  //   %zero   = arith.constant dense<0.0> : vector<...xf16>
  //   %r      = arith.cmpf une %bcast, %zero : vector<...xf16> -> vector<...xi1>
  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    // Skip transfer_reads inside vector.mask 鈥?replacing a single op with
    // a chain would break the mask (which requires exactly one body op).
    if (isa<vector::MaskOp>(op->getParentOp()))
      return failure();
    auto resultVecType = op.getVectorType();
    auto elemType = resultVecType.getElementType();
    auto intTy = dyn_cast<IntegerType>(elemType);
    if (!intTy || intTy.getIntOrFloatBitWidth() > 1)
      return failure();

    auto map = op.getPermutationMap();
    if (map.isEmpty())
      return failure();

    auto rank = resultVecType.getRank();

    SmallVector<bool> isNonBroadcastDim(rank, false);
    for (auto [idx, expr] : llvm::enumerate(map.getResults()))
      if (isa<AffineDimExpr>(expr))
        isNonBroadcastDim[idx] = true;

    // Only match if a broadcast dim has actual size > 1.
    auto hasActualBroadcast = llvm::any_of(
        llvm::seq<int>(0, rank), [&isNonBroadcastDim, &resultVecType](auto i) {
          return !isNonBroadcastDim[i] && resultVecType.getDimSize(i) > 1;
        });
    if (!hasActualBroadcast)
      return failure();

    // Don't transform if the target f16 vector exceeds the hardware
    // broadcast limit (VL_BITS / 16 bits per element).
    if (!ShapedType::isDynamic(resultVecType.getNumElements()) &&
        resultVecType.getNumElements() * 16 > hivm::util::VL_BITS)
      return failure();

    SmallVector<int64_t> intermediateShape(rank, 1);
    for (int64_t i = 0; i < rank; i++)
      if (isNonBroadcastDim[i])
        intermediateShape[i] = resultVecType.getDimSize(i);

    auto intermediateVecType =
        VectorType::get(intermediateShape, elemType);
    auto intermediateVecTypeF16 =
        VectorType::get(intermediateShape, rewriter.getF16Type());
    auto targetVecTypeF16 =
        VectorType::get(resultVecType.getShape(), rewriter.getF16Type());

    auto read = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), intermediateVecType, op.getSource(),
        op.getIndices(), map, op.getPadding(),
        /*mask=*/Value(), op.getInBoundsAttr());

    auto oneVec = rewriter.create<arith::ConstantOp>(
        op.getLoc(), intermediateVecTypeF16,
        DenseElementsAttr::get(intermediateVecTypeF16,
                               rewriter.getFloatAttr(rewriter.getF16Type(), 1.0)));
    auto zeroVec = rewriter.create<arith::ConstantOp>(
        op.getLoc(), intermediateVecTypeF16,
        rewriter.getZeroAttr(intermediateVecTypeF16));

    auto sel = rewriter.create<arith::SelectOp>(op.getLoc(), read, oneVec, zeroVec);

    auto bcast = rewriter.create<vector::BroadcastOp>(
        op.getLoc(), targetVecTypeF16, sel);

    auto zero = rewriter.create<arith::ConstantOp>(
        op.getLoc(), targetVecTypeF16,
        rewriter.getZeroAttr(targetVecTypeF16));

    auto cmp = rewriter.create<arith::CmpFOp>(
        op.getLoc(), arith::CmpFPredicate::UNE, bcast, zero);

    rewriter.replaceOp(op, cmp);
    return success();
  }
};

class TransferReadToGatheringLoadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  /// Decide whether `readop` can be rewritten as a gather and, if so, collect
  /// what the rewrite needs: the source and destination types, the physical
  /// `strides`, and the mask description (`isFullMask`, `constantMaskBounds`).
  ///
  /// None of the `analyze*` helpers creates IR, so each one is free to bail
  /// out -- see `matchAndRewrite` for why every rejecting check has to run
  /// before the first `rewriter.create`.
  LogicalResult analyzeTransferRead(vector::TransferReadOp readop,
                                    PatternRewriter &rewriter,
                                    AffineMap &permMap, MemRefType &memrefType,
                                    VectorType &destType, bool &isFullMask,
                                    SmallVector<int64_t, 4> &constantMaskBounds,
                                    SmallVector<int64_t, 4> &strides) const {
    // check the permutation map
    if (!permMap.isPermutation())
      return rewriter.notifyMatchFailure(readop, "unsupported permutation map");

    // Broadcasts are handled by the existing broadcast lowering.
    if (permMap.isConstant())
      return failure();

    memrefType = dyn_cast<MemRefType>(readop.getSource().getType());
    if (!memrefType || !memrefType.hasRank())
      return failure();
    destType = dyn_cast<VectorType>(readop.getResult().getType());
    if (!destType || !destType.hasStaticShape())
      return failure();

    if (permMap.isIdentity()) {
      if (failed(analyzeIdentityLayout(readop, memrefType, destType, strides)))
        return failure();
      return analyzeIdentityMask(readop, memrefType, isFullMask,
                                 constantMaskBounds);
    }

    return analyzePermutedRead(readop, permMap, memrefType, isFullMask,
                               constantMaskBounds, strides);
  }

  /// Identity path, layout half: accept only a read whose tile is laid out
  /// with a gap, which is what makes a contiguous load wrong, and fill in
  /// `strides`.
  LogicalResult analyzeIdentityLayout(vector::TransferReadOp readop,
                                      MemRefType memrefType,
                                      VectorType destType,
                                      SmallVector<int64_t, 4> &strides) const {
    // Out-of-bounds reads need the transfer_read padding semantics, which
    // the gather path does not reproduce for partially in-bounds dims.
    if (readop.hasOutOfBoundsDim())
      return failure();
    int64_t offset;
    if (failed(getStridesAndOffset(memrefType, strides, offset)))
      return failure();
    // A negative or dynamic stride cannot be turned into a static index
    // vector.
    if (llvm::any_of(strides, [](int64_t stride) {
          return ShapedType::isDynamic(stride) || stride < 1;
        }))
      return failure();

    int64_t rank = memrefType.getRank();
    if (rank == 1) {
      // For a 1-D view, unit stride is contiguous even with a dynamic
      // extent, so only a real stride needs the gather path.
      if (memrefType.getDimSize(0) == 1 || strides.front() <= 1)
        return failure();
      return success();
    }

    // For rank > 1 the index vector is materialized from the static
    // extents, so every dim must be static.
    if (!memrefType.hasStaticShape())
      return failure();
    if (memrefType.getNumElements() == 1)
      return failure();
    // Detect whether the *tile being read* is densely packed.  Contiguity
    // follows the tile extents, not the source extents: reading
    // vector<2x32> out of memref<2x64, strided<[32, 1]>> touches lanes
    // 0..63 and is contiguous, while the same tile out of a packed
    // memref<2x64> (strides [64, 1]) touches 0..31 and 64..95 and is not.
    // Note: identity permutation alone does not imply contiguity, and
    // checking only the innermost stride is not enough either.
    auto tileShape = destType.getShape();
    bool isPacked = true;
    int64_t expectedStride = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
      if (strides[d] != expectedStride) {
        isPacked = false;
        break;
      }
      expectedStride *= tileShape[d];
    }
    if (isPacked)
      return failure();

    return success();
  }

  /// Identity path, mask half: classify the mask as full, or record its
  /// per-dim bounds in `constantMaskBounds`.
  LogicalResult
  analyzeIdentityMask(vector::TransferReadOp readop, MemRefType memrefType,
                      bool &isFullMask,
                      SmallVector<int64_t, 4> &constantMaskBounds) const {
    isFullMask = true;
    if (auto mask = readop.getMask()) {
      auto constantMask = mask.getDefiningOp<vector::ConstantMaskOp>();
      if (!constantMask) {
        // A non-constant mask cannot be flattened per lane at compile time.
        // A rank-1 mask, however, already matches the flattened gather lane
        // order and is reused as-is by the rewrite, so it needs no
        // flattening (dynamic tail masks from loop peeling hit this path).
        //
        // FIXME: a multi-dim runtime mask on a strided source is not
        // supported.  Rejecting it here falls back to the contiguous
        // VFLoad, which loses the stride semantics -- a match failure is
        // not a correctness guarantee.  A stride-aware runtime mask path is
        // needed; this is a pre-existing gap, not a regression.
        if (cast<VectorType>(mask.getType()).getRank() != 1)
          return failure();
      } else if (!constantMask.isAllOnesMask()) {
        // Keep inactive indices zero, as on the existing constant-mask path,
        // so that a large stride does not overflow unused gather indices.
        for (auto v : constantMask.getMaskDimSizes())
          constantMaskBounds.push_back(
              cast<IntegerAttr>(v).getValue().getSExtValue());
        // `computeGatherIndicesAndMasks` indexes `constantMaskBounds` with
        // the source dim, so a mask that does not cover every dim would read
        // out of bounds.
        if (constantMaskBounds.size() !=
            static_cast<size_t>(memrefType.getRank()))
          return failure();
        isFullMask = false;
      }
    }
    return success();
  }

  /// Permuted (transposing) path: check the shape and the mask, fill in
  /// `strides`, and decline a transpose whose permuted dims are masked down
  /// to a single element, which needs no gather.
  LogicalResult analyzePermutedRead(vector::TransferReadOp readop,
                                    AffineMap permMap, MemRefType memrefType,
                                    bool &isFullMask,
                                    SmallVector<int64_t, 4> &constantMaskBounds,
                                    SmallVector<int64_t, 4> &strides) const {
    // must have static shape
    // dynamic shape size is represented by a large negative number
    auto shape = memrefType.getShape();
    for (auto dim : shape) {
      if (dim < 1) {
        return failure();
      }
      // make sure the mask is a vector.constant_mask or a full mask
      // (we currently cannot handle non-constant mask because we cannot
      // apply inverse affine transform to them for now.)
      auto originalMask = readop.getMask();
      if (!originalMask) {
        isFullMask = true;
      } else {
        auto constmask =
            dyn_cast<vector::ConstantMaskOp>(originalMask.getDefiningOp());
        if (!constmask)
          return failure();
        if (constmask.isAllOnesMask()) {
          isFullMask = true;
        } else {
          for (auto v : constmask.getMaskDimSizes()) {
            constantMaskBounds.push_back(
                cast<IntegerAttr>(v).getValue().getSExtValue());
          }
        }
      }
    }
    // Extract physical strides from both strided and affine layouts.
    int64_t offset;
    if (failed(getStridesAndOffset(memrefType, strides, offset)) ||
        llvm::any_of(strides, [](int64_t stride) { return stride < 1; }))
      return failure();

    // if transpose dim with 1 mask value
    // no need to change transfer_read to gather
    bool permDimWithOneMask = true;
    if (!constantMaskBounds.empty()) {
      for (unsigned int i = 0; i < permMap.getNumResults(); i++) {
        unsigned int targetDim = permMap.getDimPosition(i);
        if (i != targetDim) {
          if (constantMaskBounds[i] != 1 ||
              constantMaskBounds[targetDim] != 1) {
            permDimWithOneMask = false;
            break;
          }
        }
      }
      if (permDimWithOneMask)
        return failure();
    }

    return success();
  }

  void computeGatherIndicesAndMasks(
      vector::TransferReadOp readop, AffineMap permMap, MemRefType memrefType,
      VectorType destType, SmallVector<int64_t, 4> &strides, bool &isFullMask,
      SmallVector<int64_t, 4> &constantMaskBounds,
      SmallVector<APInt, 0> &gatherIndices, SmallVector<bool, 0> &gatherMasks,
      int64_t &firstEnabledBit, int64_t &lastEnabledBit,
      int64_t &firstDisabledBit, int64_t &lastDisabledBit,
      int64_t &totalDestSize, bool &indexOverflow) const {
    // try to find all gather indices
    // we need to look at the result vector type to find the range
    // for all the source dimensions
    auto destShape = destType.getShape();
    for (unsigned i = 0; i < destShape.size(); ++i)
      totalDestSize *= destShape[i];

    // permMap maps the memref index space onto the result vector index space,
    // so recovering the memref indices of a result element requires the
    // *inverse* map. For a 2-D transpose they're the same, but not for a
    // 3-D one, like (d0, d1, d2) -> (d1, d2, d0)
    AffineMap invPermMap = inversePermutation(permMap);
    if (!invPermMap)
      llvm::report_fatal_error("expected an invertible permutation map");

    unsigned elementBitWidth = memrefType.getElementTypeBitWidth();
    SmallVector<int64_t, 0> composeIndices(destShape.size(), 0);
    gatherIndices.reserve(totalDestSize);

    // also prepare to convert the masks
    gatherMasks.reserve(totalDestSize);

    composeIndices.back() = -1; // cancel out the first +1
    for (int64_t i = 0; i < totalDestSize; ++i) {
      composeIndices.back() += 1;
      for (unsigned dim = destShape.size() - 1; dim >= 1; dim--) {
        if (composeIndices[dim] < destShape[dim])
          break;
        composeIndices[dim] = 0;
        composeIndices[dim - 1] += 1;
      }

      auto composeResult = invPermMap.compose(composeIndices);
      if (composeResult.size() != strides.size())
        llvm::report_fatal_error("Unexpected dimension mismatch");

      bool isEnabled = true;
      // check if this index is masked off
      if (!isFullMask) {
        for (unsigned dim = 0; dim < composeResult.size(); ++dim) {
          if (composeResult[dim] >= constantMaskBounds[dim]) {
            isEnabled = false;
            break;
          }
        }
        gatherMasks.push_back(isEnabled);
        if (isEnabled) {
          lastEnabledBit = i;
          if (firstEnabledBit == -1)
            firstEnabledBit = i;
        } else {
          lastDisabledBit = i;
          if (firstDisabledBit == -1)
            firstDisabledBit = i;
        }
      }
      // compute the gather index
      // we only compute it when the mask enables this element
      // because large strides can make the index overflow
      //
      // The zeroed tail must stay: a masked-off lane often maps to a
      // coordinate outside the source (that is what the mask is for -- see
      // @transpose_2d_outlined_vf_0 in normalize_vector.mlir, where 112 of 128
      // lanes are out of a 16x1 source), so computing its "real" index would
      // address past the end of the memref.
      //
      // The flat tail this leaves is not a staircase, which used to make the
      // index constant unlowerable.  `ConstantOpToHivmSegmentedVCILowering`
      // now fits its closed form to the active prefix and writes the tail back
      // with a prefix predicate, so the convention is honoured on the consumer
      // side (see `emitActivePrefixSelect` in ArithToHIVMAVE.cpp).
      int64_t index = 0;
      if (isFullMask || isEnabled) {
        for (unsigned dim = 0; dim < strides.size(); ++dim)
          index += composeResult[dim] * strides[dim];
      }
      if (elementBitWidth == 8) {
        gatherIndices.push_back(APInt(16, index));
      } else {
        gatherIndices.push_back(APInt(elementBitWidth, index));
      }
      if (elementBitWidth == 8 || elementBitWidth == 16) {
        if (index > (int64_t)APInt::getMaxValue(16).getZExtValue())
          indexOverflow = true;
      }
    }
    if (!isFullMask && lastDisabledBit == -1)
      isFullMask = true;
  }

  SmallVector<Value, 2> createIndexVector(vector::TransferReadOp readop,
                                          PatternRewriter &rewriter,
                                          MemRefType memrefType,
                                          SmallVector<APInt, 0> &gatherIndices,
                                          int64_t totalDestSize) const {
    SmallVector<Value, 2> indexVecValues;

    // B8 gather indices are computed as 16-bit values because an 8-bit
    // element type cannot represent the full range of memref-relative byte
    // offsets a gather may need to index. A single HIVM AVE tile register
    // can hold at most 128 16-bit indices so only a B8 gather wider than
    // that needs its index vector split across two registers. A narrower B8
    // gather fits a single native gather directly.
    bool isB8 = memrefType.getElementTypeBitWidth() == 8;
    unsigned indexBitWidth = isB8 ? 16 : memrefType.getElementTypeBitWidth();
    IntegerType indexVecElementType =
        IntegerType::get(rewriter.getContext(), indexBitWidth);
    auto indexVecType = VectorType::get(totalDestSize, indexVecElementType);

    int64_t subIndexLanes =
        mlir::utils::getHIVMTileSliceMinNumElts(indexVecElementType);
    bool isDualGather = isB8 && totalDestSize > subIndexLanes;

    auto checkIndex = [](const SmallVector<APInt, 0> &src,
                         const DenseElementsAttr &attr) {
      auto values = llvm::to_vector<0>(attr.getValues<APInt>());
      for (unsigned i = 0; i < values.size(); ++i) {
        if (values[i].getSExtValue() != src[i].getSExtValue()) {
          llvm::report_fatal_error("Sub index value error");
        }
      }
    };

    if (isDualGather) {
      SmallVector<APInt, 0> gatherIndices1, gatherIndices2;
      for (unsigned i = 0; i < gatherIndices.size(); ++i) {
        (i % 2 == 0) ? gatherIndices1.push_back(gatherIndices[i])
                     : gatherIndices2.push_back(gatherIndices[i]);
      }
      // Below the full VL width each half holds fewer than subIndexLanes
      // real indices. Fill the rest with offset 0
      gatherIndices1.resize(subIndexLanes, APInt(indexBitWidth, 0));
      gatherIndices2.resize(subIndexLanes, APInt(indexBitWidth, 0));

      IntegerType i16Type = IntegerType::get(
          rewriter.getContext(), 16, indexVecElementType.getSignedness());
      VectorType subIndexVecType = VectorType::get(subIndexLanes, i16Type);

      auto processIndexVec =
          [&indexVecElementType, &i16Type, &subIndexVecType, &rewriter,
           &readop](const SmallVector<APInt, 0> &srcI8Indices)
          -> std::tuple<SmallVector<APInt, 0>, DenseElementsAttr, Value> {
        SmallVector<APInt, 0> dstI16Indices;
        dstI16Indices.reserve(srcI8Indices.size());
        for (const auto &idx : srcI8Indices) {
          if (indexVecElementType.isSigned()) {
            dstI16Indices.push_back(idx.sext(16));
          } else {
            dstI16Indices.push_back(idx.zext(16));
          }
        }

        SmallVector<Attribute, 128> indexAttrs;
        indexAttrs.reserve(dstI16Indices.size());
        for (const auto &idx : dstI16Indices) {
          indexAttrs.push_back(IntegerAttr::get(i16Type, idx));
        }
        DenseElementsAttr indexVecAttr =
            DenseElementsAttr::get(subIndexVecType, indexAttrs);

        Value indexVecValue =
            rewriter.create<arith::ConstantOp>(readop.getLoc(), indexVecAttr);

        return {dstI16Indices, indexVecAttr, indexVecValue};
      };

      auto [indices1_i16, attr1, value1] = processIndexVec(gatherIndices1);
      auto [indices2_i16, attr2, value2] = processIndexVec(gatherIndices2);

      checkIndex(indices1_i16, attr1);
      checkIndex(indices2_i16, attr2);

      indexVecValues.push_back(value1);
      indexVecValues.push_back(value2);
    } else {
      // other type
      DenseElementsAttr indexVecAttr = DenseElementsAttr::get(
          indexVecType, llvm::ArrayRef<APInt>(gatherIndices));
      auto indexVecValue =
          rewriter.create<arith::ConstantOp>(readop.getLoc(), indexVecAttr);

      checkIndex(gatherIndices, indexVecAttr);
      indexVecValues.push_back(indexVecValue);
    }

    return indexVecValues;
  }

  Value createPaddingValue(vector::TransferReadOp readop,
                           PatternRewriter &rewriter, VectorType destType,
                           int64_t totalDestSize) const {

    // convert the padding value from scalar to vector
    auto padding = readop.getPadding();
    if (auto constantPadding =
            dyn_cast<arith::ConstantOp>(padding.getDefiningOp())) {
      // create a vector padding
      VectorType resultType =
          VectorType::get({totalDestSize}, destType.getElementType());
      auto passthru =
          DenseElementsAttr::get(resultType, constantPadding.getValueAttr());
      padding = rewriter.create<arith::ConstantOp>(padding.getLoc(), resultType,
                                                   passthru);
    } else {
      llvm::report_fatal_error("TODO support non-constant padding value for "
                       "vector.transfer_read -> vector.gather transform");
    }
    return padding;
  }

  LogicalResult matchAndRewrite(vector::TransferReadOp readop,
                                PatternRewriter &rewriter) const override {
    // Convert permuted or 1-D identity-strided reads using the existing
    // physical offset computation and gather lowering.
    auto permMap = readop.getPermutationMap();
    MemRefType memrefType;
    VectorType destType;
    bool isFullMask = false;
    SmallVector<int64_t, 4> constantMaskBounds, strides;
    if (failed(analyzeTransferRead(readop, rewriter, permMap, memrefType,
                                   destType, isFullMask, constantMaskBounds,
                                   strides)))
      return failure();

    SmallVector<APInt, 0> gatherIndices;
    SmallVector<bool, 0> gatherMasks;
    int64_t firstEnabledBit{-1}, lastEnabledBit{-1}, firstDisabledBit{-1},
        lastDisabledBit{-1};
    int64_t totalDestSize = 1;
    bool indexOverflow = false;
    computeGatherIndicesAndMasks(readop, permMap, memrefType, destType, strides,
                                 isFullMask, constantMaskBounds, gatherIndices,
                                 gatherMasks, firstEnabledBit, lastEnabledBit,
                                 firstDisabledBit, lastDisabledBit,
                                 totalDestSize, indexOverflow);
    // Not support gather index overflow.
    // Such as, gather B8/B16 type date with 65537 index.
    // Todo: Use template lib to extend scenarios.
    if (indexOverflow)
      return failure();
    // A B8 gather wider than one native gather (128 16-bit indices) is split
    // into two sub-gathers by index parity. Two sub-gathers cover at most VL
    // elements; nothing wider is supported.
    if (memrefType.getElementTypeBitWidth() == hivm::util::BITS_PER_BYTE &&
        totalDestSize > hivm::util::VL) {
      return readop.emitError()
             << "unsupported B8 transpose gather width " << totalDestSize
             << ": expected <= " << hivm::util::VL;
    }

    // A rank-1 identity mask already matches the flattened gather lane order,
    // so it can be reused as is.
    bool reuseOriginalMask =
        permMap.isIdentity() && readop.getMask() &&
        cast<VectorType>(readop.getMask().getType()).getRank() == 1;
    // Currently the only partial mask that can be rebuilt on the flattened
    // lane order is a prefix: gatherMasks[0..N-1] true, gatherMasks[N..] false.
    // This must be decided before any IR is created -- a `return failure()`
    // after `createIndexVector` leaves a dead index constant behind, and the
    // greedy driver then re-matches this op until it hits the iteration limit.
    // TODO: handle non-prefix masks instead of falling back to the contiguous
    // load, which silently ignores the stride.
    if (!reuseOriginalMask && !isFullMask &&
        !(firstEnabledBit < lastEnabledBit &&
          lastEnabledBit + 1 == firstDisabledBit &&
          firstDisabledBit <= lastDisabledBit))
      return failure();

    auto indexVecValues = createIndexVector(readop, rewriter, memrefType,
                                            gatherIndices, totalDestSize);
    Value finalResult;
    decltype(readop.getMask()) newMask;
    VectorType maskType =
        VectorType::get({totalDestSize}, rewriter.getI1Type());
    if (reuseOriginalMask) {
      newMask = readop.getMask();
    } else {
      // Checked above: a partial mask covers the first `firstDisabledBit`
      // lanes, a full mask covers all of them.
      ArrayAttr maskDimSizes = rewriter.getI64ArrayAttr(
          {isFullMask ? totalDestSize : firstDisabledBit});
      auto newConstMask = rewriter.create<vector::ConstantMaskOp>(
          readop.getLoc(), maskType, maskDimSizes);
      newMask = newConstMask;
    }

    if (indexVecValues.size() == 2) {
      auto eltType = destType.getElementType();
      VectorType subDestType = VectorType::get({totalDestSize}, eltType);
      Value padding =
          createPaddingValue(readop, rewriter, subDestType, totalDestSize);

      auto gather = rewriter.create<vector::GatherOp>(
          readop.getLoc(), subDestType, readop.getSource(), readop.getIndices(),
          indexVecValues[0], newMask, padding);
      auto subIndex2ConstantOp =
          indexVecValues[1].getDefiningOp<arith::ConstantOp>();
      auto subIndex2Attr =
          mlir::dyn_cast<DenseElementsAttr>(subIndex2ConstantOp.getValueAttr());
      gather->setAttr("secondary_index", subIndex2Attr);
      finalResult = gather.getResult();
    } else {
      VectorType resultType =
          VectorType::get({totalDestSize}, destType.getElementType());
      Value padding =
          createPaddingValue(readop, rewriter, destType, totalDestSize);
      auto gather = rewriter.create<vector::GatherOp>(
          readop.getLoc(), resultType, readop.getSource(), readop.getIndices(),
          indexVecValues[0], newMask, padding);
      finalResult = gather.getResult();
    }

    if (destType.getRank() > 1) {
      finalResult =
          rewriter.create<vector::ShapeCastOp>(readop.getLoc(), destType,
                                               finalResult);
    }
    rewriter.replaceOp(readop, finalResult);
    return success();
  }
};

struct TruncfScalarToVector : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const final {
    Value truncResult = op.getResult();
    if (!truncResult.hasOneUse())
      return failure();
    auto *user = *truncResult.user_begin();
    auto broadcastOp = dyn_cast<vector::BroadcastOp>(user);
    if(!broadcastOp)
      return failure();
    Value truncInput = op.getOperand();
    Type truncInputType = truncInput.getType();
    if (isa<ShapedType>(truncInputType))
      return failure();
    VectorType broadcastResultType = broadcastOp.getResultVectorType();
    if(!broadcastResultType.getElementType().isBF16())
      return failure();
    Location loc = op.getLoc();
    VectorType intermediateType = VectorType::get(
      broadcastResultType.getShape(),truncInputType);
    Value broadcastScalar = rewriter.create<vector::BroadcastOp>(
      loc, intermediateType, truncInput);
    Value newTrunc = rewriter.create<arith::TruncFOp>(
      loc, broadcastResultType, broadcastScalar);
    rewriter.replaceOp(broadcastOp, newTrunc);
    if (op->use_empty()) {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

/// Before conversion:
/// ```mlir
///    %17 = arith.cmpf oeq, %arg3, %arg10 : f32
/// ```
/// After conversion:
/// ```mlir
///    %19 = vector.broadcast %arg3 : f32 to vector<64xf32>
///    %20 = vector.broadcast %arg10 : f32 to vector<64xf32>
///    %21 = arith.cmpf oeq, %19, %a20 : vector<64xf32>
/// ```
template <typename BinaryOpType>
struct BinaryScalarOpToVectorPattern : public OpRewritePattern<BinaryOpType> {
  using OpRewritePattern<BinaryOpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(BinaryOpType op, PatternRewriter &rewriter) const override {
    if (isa<VectorType>(op.getResult().getType()))
      return failure();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    Type lhsType = lhs.getType();
    Type rhsType = rhs.getType();
    Type resultType = op.getType();
    if (!((lhsType.isF32() && rhsType.isF32()) || (lhsType.isInteger(1) && rhsType.isInteger(1))))
      return failure();
    int numOperands = static_cast<int>(op.getNumOperands());
    if (numOperands != 2)
      return failure();
    Value leftOperand = op.getOperand(0);
    Value rightOperand = op.getOperand(1);

    auto leftBlockArg = dyn_cast<BlockArgument>(leftOperand);
    auto rightBlockArg = dyn_cast<BlockArgument>(rightOperand);
    VectorType vecType = VectorType::get({64}, resultType);
    Location loc = op.getLoc();
    if (!leftBlockArg && rightBlockArg) {
      VectorType cmpVecType = VectorType::get({64}, rewriter.getF32Type());
      if constexpr (std::is_same_v<BinaryOpType, arith::CmpFOp>) {
        arith::CmpFPredicate predicate = op.getPredicate();
        auto lhsVec =
            rewriter.create<UnrealizedConversionCastOp>(loc, cmpVecType, lhs);
        auto rhsVec =
            rewriter.create<vector::BroadcastOp>(loc, cmpVecType, rhs);
        auto vectorCmp = rewriter.create<BinaryOpType>(
            loc, cmpVecType.clone(rewriter.getI1Type()), predicate,
            lhsVec.getResult(0), rhsVec.getResult());
        rewriter.replaceOp(op, vectorCmp);
        return success();
      } else if constexpr (std::is_same_v<BinaryOpType, mathExt::DivFHPOp> ||
                           std::is_same_v<BinaryOpType, arith::AndIOp> ||
                           std::is_same_v<BinaryOpType, arith::XOrIOp> ||
                           std::is_same_v<BinaryOpType, arith::MulFOp> ||
                           std::is_same_v<BinaryOpType, arith::SubFOp>) {
        auto lhsVec =
          rewriter.create<UnrealizedConversionCastOp>(loc, vecType, lhs);
        auto rhsVec = rewriter.create<vector::BroadcastOp>(loc, vecType, rhs);
        auto vecResult = rewriter.create<BinaryOpType>(
            loc, vecType, lhsVec.getResult(0), rhsVec.getResult());
        rewriter.replaceOp(op, vecResult);
        return success();
      }
    } else if (leftBlockArg && !rightBlockArg) {
      if constexpr (std::is_same_v<BinaryOpType, arith::CmpFOp>) {
        VectorType cmpVecType = VectorType::get({64}, rewriter.getF32Type());
        arith::CmpFPredicate predicate = op.getPredicate();
        auto rhsVec =
            rewriter.create<UnrealizedConversionCastOp>(loc, cmpVecType, rhs);
        auto lhsVec =
            rewriter.create<vector::BroadcastOp>(loc, cmpVecType, lhs);
        auto vectorCmp = rewriter.create<BinaryOpType>(
            loc, cmpVecType.clone(rewriter.getI1Type()), predicate,
            lhsVec.getResult(), rhsVec.getResult(0));
        rewriter.replaceOp(op, vectorCmp);
        return success();
      } else if constexpr (std::is_same_v<BinaryOpType, mathExt::DivFHPOp> ||
                           std::is_same_v<BinaryOpType, arith::AndIOp> ||
                           std::is_same_v<BinaryOpType, arith::XOrIOp> ||
                           std::is_same_v<BinaryOpType, arith::MulFOp> ||
                           std::is_same_v<BinaryOpType, arith::SubFOp>) {
        auto rhsVec =
            rewriter.create<UnrealizedConversionCastOp>(loc, vecType, rhs);
        auto lhsVec = rewriter.create<vector::BroadcastOp>(loc, vecType, lhs);
        auto vecResult = rewriter.create<BinaryOpType>(
            loc, vecType, lhsVec.getResult(), rhsVec.getResult(0));
        rewriter.replaceOp(op, vecResult);
        return success();
    }
    } else if (!leftBlockArg && !rightBlockArg) {
      if constexpr (std::is_same_v<BinaryOpType, arith::CmpFOp>) {
        VectorType cmpVecType = VectorType::get({64}, rewriter.getF32Type());
        arith::CmpFPredicate predicate = op.getPredicate();
        auto lhsVec =
              rewriter.create<UnrealizedConversionCastOp>(loc, cmpVecType, lhs);
        auto rhsVec =
              rewriter.create<UnrealizedConversionCastOp>(loc, cmpVecType, rhs);
        auto vectorCmp = rewriter.create<BinaryOpType>(
              loc, cmpVecType.clone(rewriter.getI1Type()), predicate,
              lhsVec.getResult(0), rhsVec.getResult(0));
        rewriter.replaceOp(op, vectorCmp);
        return success();
      } else if constexpr (std::is_same_v<BinaryOpType, mathExt::DivFHPOp> ||
                            std::is_same_v<BinaryOpType, arith::AndIOp> ||
                            std::is_same_v<BinaryOpType, arith::XOrIOp> ||
                            std::is_same_v<BinaryOpType, arith::MulFOp> ||
                            std::is_same_v<BinaryOpType, arith::SubFOp>) {
      auto lhsVec =
          rewriter.create<UnrealizedConversionCastOp>(loc, vecType, lhs);
      auto rhsVec =
            rewriter.create<UnrealizedConversionCastOp>(loc, vecType, rhs);
      auto vecResult = rewriter.create<BinaryOpType>(
            loc, vecType, lhsVec.getResult(0), rhsVec.getResult(0));
      rewriter.replaceOp(op, vecResult);
      return success();
      }
    } else {
      if constexpr (std::is_same_v<BinaryOpType, arith::CmpFOp>) {
        VectorType cmpVecType = VectorType::get({64}, rewriter.getF32Type());
        arith::CmpFPredicate predicate = op.getPredicate();
        auto lhsVec =
            rewriter.create<vector::BroadcastOp>(loc, cmpVecType, lhs);
        auto rhsVec =
            rewriter.create<vector::BroadcastOp>(loc, cmpVecType, rhs);
        auto vectorCmp = rewriter.create<BinaryOpType>(
            loc, cmpVecType.clone(rewriter.getI1Type()), predicate,
            lhsVec.getResult(), rhsVec.getResult());
        rewriter.replaceOp(op, vectorCmp);
        return success();
      } else if constexpr (std::is_same_v<BinaryOpType, mathExt::DivFHPOp> ||
                           std::is_same_v<BinaryOpType, arith::AndIOp> ||
                           std::is_same_v<BinaryOpType, arith::XOrIOp> ||
                           std::is_same_v<BinaryOpType, arith::MulFOp> ||
                           std::is_same_v<BinaryOpType, arith::SubFOp>) {
        auto lhsVec = rewriter.create<vector::BroadcastOp>(loc, vecType, lhs);
        auto rhsVec = rewriter.create<vector::BroadcastOp>(loc, vecType, rhs);
        auto vecResult = rewriter.create<BinaryOpType>(
            loc, vecType, lhsVec.getResult(), rhsVec.getResult());
        rewriter.replaceOp(op, vecResult);
        return success();
       }
    }
    return failure();
  }
};

/// Before conversion:
/// ```mlir
///    %1 = math.absf %arg3 : f32
/// ```
/// After conversion:
/// ```mlir
///    %1 = vector.broadcast %arg3 : f32 to vector<64xf32>
///    %2 = math.absf %1 : vector<64xf32>
/// ```
template <typename UnaryOpType>
struct UnaryScalarOpToVectorPattern : public OpRewritePattern<UnaryOpType> {
  using OpRewritePattern<UnaryOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryOpType op,
                                PatternRewriter &rewriter) const override {
    if (isa<VectorType>(op.getResult().getType()))
      return failure();
    Value operand = op.getOperand();
    Type operandType = operand.getType();
    if (!operandType.isF32()) {
      return failure();
    }
    VectorType vectorType = VectorType::get({64}, operandType);
    Location loc = op.getLoc();
    auto blockArg = dyn_cast<BlockArgument>(operand);

    // Reject if any user is neither a broadcast, a conversion cast inserted
    // by sibling patterns, nor an op this pass will vectorize: the cast-back
    // we'd emit cannot be folded (its producer is a vector op) and leaks to
    // LLVM lowering, where the AIC vec backend cannot select it.
    SmallVector<vector::BroadcastOp, 4> broadcastUsers;
    for (auto user : op->getUsers()) {
      if (auto broadcastUser = dyn_cast<vector::BroadcastOp>(user)) {
        broadcastUsers.push_back(broadcastUser);
        continue;
      }
      if (isa<UnrealizedConversionCastOp>(user))
        continue;
      if (!isVectorizableByThisPass(user))
        return failure();
    }

    // Create vector operand
    Value vecOperand;
    if (blockArg) {
      vecOperand =
          rewriter.create<vector::BroadcastOp>(loc, vectorType, operand);
    } else {
      vecOperand =
          rewriter.create<UnrealizedConversionCastOp>(loc, vectorType, operand)
              .getResult(0);
    }

    // Create vector unary op
    auto vecOp = rewriter.create<UnaryOpType>(loc, vecOperand);
    copyRoundAttributes(op, vecOp);

    // Replace broadcast users
    for (auto broadcastUser : broadcastUsers) {
      rewriter.replaceOp(broadcastUser, vecOp.getResult());
    }

    // Handle remaining uses
    if (!op->use_empty()) {
      auto castOp = rewriter.create<UnrealizedConversionCastOp>(
          loc, operandType, vecOp.getResult());
      rewriter.replaceOp(op, castOp.getResult(0));
    }
    return success();
  }

  // Whether `user` is rewritten to vector form by this pass, so the cast-back
  // feeding it folds. Mirrors the op list in NormalizeVectorPass::runOnOperation.
  static bool isVectorizableByThisPass(Operation *user) {
    if (isa<math::AbsFOp, math::LogOp, math::RoundOp>(user))
      return true;
    if (isa<mathExt::DivFHPOp, arith::AndIOp, arith::XOrIOp, arith::MulFOp,
            arith::SubFOp, arith::CmpFOp>(user))
      return true;
    if (auto fma = dyn_cast<math::FmaOp>(user)) {
      auto types = fma.getOperandTypes();
      return std::all_of(types.begin(), types.end(),
                         [](Type t) { return t.isF32(); });
    }
    return false;
  }

  void copyRoundAttributes(Operation *src, Operation *dst) const {
    if constexpr (std::is_same_v<UnaryOpType, math::RoundOp>) {
      if (auto enableSaturate =
              src->template getAttrOfType<BoolAttr>("enable_saturate"))
        dst->setAttr("enable_saturate", enableSaturate);
      if (auto roundMode = src->template getAttrOfType<Attribute>("round_mode"))
        dst->setAttr("round_mode", roundMode);
      if (auto unsignedMode =
              src->template getAttrOfType<Attribute>("unsigned_mode"))
        dst->setAttr("unsigned_mode", unsignedMode);
    }
  }
};

struct FmaOpScalarToVectorPattern : public OpRewritePattern<math::FmaOp> {
  using OpRewritePattern<math::FmaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const override {
    auto opTypes = op.getOperandTypes();

    bool nonVecArgsAreF32 = std::all_of(opTypes.begin(), opTypes.end(), [](Type t) {
        return t.isF32();
    });

    if (!nonVecArgsAreF32) {
      return failure();
    }

    auto loc = op.getLoc();
    auto operandType = op.getC().getType();
    auto vectorType = VectorType::get({64}, operandType);

    SmallVector<vector::BroadcastOp, 4> broadCastUsers;
    for (auto user: op->getUsers()) {
      if (auto bcUser = dyn_cast<vector::BroadcastOp>(user)) {
        broadCastUsers.push_back(bcUser);
      }
    }

    SmallVector<Value, 3> vOperands;
    for (auto operand: op->getOperands()) {
      Value vOperand;
      if (isa<BlockArgument>(operand)) {
        vOperand = rewriter.create<vector::BroadcastOp>(loc, vectorType, operand);
      } else {
        vOperand = rewriter.create<UnrealizedConversionCastOp>(
            loc, vectorType, operand).getResult(0);
      }
      vOperands.push_back(vOperand);
    }

    auto vecFmaOp = rewriter.create<math::FmaOp>(loc, vectorType, vOperands);

    for (auto bcUser : broadCastUsers) {
      rewriter.replaceOp(bcUser, vecFmaOp.getResult());
    }

    if (!op->use_empty()) {
      auto castOp = rewriter.create<UnrealizedConversionCastOp>(
          loc, operandType, vecFmaOp.getResult());
      rewriter.replaceOp(op, castOp.getResult(0));
    }

    return success();
  }
};

// Vectorize a single-use scalar chain ending in `vector.broadcast %scalar :
// T to vector<64xT>` (T = f32 or i32). Each link is a bit-exact op (bitcast
// / andi / addi / subi / muli / minsi / maxsi / math.absf). Vectorizing the
// whole chain at once avoids leaving a scalar `arith.bitcast f32<->i32`,
// whose LLVM lowering the AIC vec backend cannot select.
struct ScalarChainBroadcastToVectorPattern
    : public OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp brcOp,
                                PatternRewriter &rewriter) const override {
    Value src = brcOp.getOperand();
    Type scalarType = src.getType();
    VectorType vecType =
        mlir::dyn_cast<VectorType>(brcOp.getResult().getType());
    if (!vecType || vecType.getRank() != 1 || vecType.getNumElements() != 64)
      return failure();
    if (!scalarType.isF32() && !scalarType.isInteger(32))
      return failure();

    SmallVector<Operation *, 8> chain;
    Value cur = src;
    while (auto defOp = cur.getDefiningOp()) {
      if (!isScalarChainOp(defOp) || !defOp->hasOneUse())
        break;
      chain.push_back(defOp);
      cur = defOp->getOperand(0);
    }
    if (chain.empty())
      return failure();

    // Only fire when the chain actually contains a bitcast. Vectorizing a
    // chain of, say, pure arithmetic ops that happen to end in a broadcast is
    // not necessary (the scalar form lowers fine to LLVM) and could mask a
    // genuine remaining scalar bitcast elsewhere.
    if (!llvm::any_of(chain,
                      [](Operation *op) { return isa<arith::BitcastOp>(op); }))
      return failure();

    Location loc = brcOp.getLoc();
    VectorType rootVecType = VectorType::get({64}, cur.getType());
    Value curVec = isa<BlockArgument>(cur)
                       ? (Value)rewriter.create<vector::BroadcastOp>(
                             loc, rootVecType, cur)
                       : rewriter
                             .create<UnrealizedConversionCastOp>(
                                 loc, rootVecType, cur)
                             .getResult(0);

    for (auto it = chain.rbegin(), e = chain.rend(); it != e; ++it) {
      Operation *op = *it;
      VectorType opVecType =
          VectorType::get({64}, op->getResult(0).getType());
      if (auto bc = dyn_cast<arith::BitcastOp>(op)) {
        curVec = rewriter.create<arith::BitcastOp>(loc, opVecType, curVec);
      } else if (auto andi = dyn_cast<arith::AndIOp>(op)) {
        curVec = rewriter.create<arith::AndIOp>(
            loc, opVecType, curVec,
            broadcastOrCast(rewriter, loc, andi.getRhs(), opVecType));
      } else if (auto addi = dyn_cast<arith::AddIOp>(op)) {
        curVec = rewriter.create<arith::AddIOp>(
            loc, opVecType, curVec,
            broadcastOrCast(rewriter, loc, addi.getRhs(), opVecType));
      } else if (auto subi = dyn_cast<arith::SubIOp>(op)) {
        curVec = rewriter.create<arith::SubIOp>(
            loc, opVecType, curVec,
            broadcastOrCast(rewriter, loc, subi.getRhs(), opVecType));
      } else if (auto muli = dyn_cast<arith::MulIOp>(op)) {
        curVec = rewriter.create<arith::MulIOp>(
            loc, opVecType, curVec,
            broadcastOrCast(rewriter, loc, muli.getRhs(), opVecType));
      } else if (auto minsi = dyn_cast<arith::MinSIOp>(op)) {
        curVec = rewriter.create<arith::MinSIOp>(
            loc, opVecType, curVec,
            broadcastOrCast(rewriter, loc, minsi.getRhs(), opVecType));
      } else if (auto maxsi = dyn_cast<arith::MaxSIOp>(op)) {
        curVec = rewriter.create<arith::MaxSIOp>(
            loc, opVecType, curVec,
            broadcastOrCast(rewriter, loc, maxsi.getRhs(), opVecType));
      } else if (auto cmpi = dyn_cast<arith::CmpIOp>(op)) {
        // CmpIOp result is i1; operands carry the comparison value type.
        // curVec and the rhs operand are vectorized at the operand's element
        // type, not the i1 result type.
        VectorType cmpOperandVecType =
            VectorType::get({64}, cmpi.getLhs().getType());
        curVec = rewriter.create<arith::CmpIOp>(
            loc, opVecType, cmpi.getPredicate(),
            curVec, broadcastOrCast(rewriter, loc, cmpi.getRhs(),
                                     cmpOperandVecType));
      } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
        // Select is a fork: spine continues on the condition (operand 0);
        // true_val/false_val are side inputs broadcast/cast to the result
        // element type.
        curVec = rewriter.create<arith::SelectOp>(
            loc, opVecType, curVec,
            broadcastOrCast(rewriter, loc, select.getTrueValue(), opVecType),
            broadcastOrCast(rewriter, loc, select.getFalseValue(), opVecType));
      } else if (isa<math::AbsFOp>(op)) {
        curVec = rewriter.create<math::AbsFOp>(loc, opVecType, curVec);
      } else {
        return failure();
      }
    }

    rewriter.replaceOp(brcOp, curVec);
    for (auto *op : chain)
      rewriter.eraseOp(op);
    return success();
  }

private:
  static bool isScalarChainOp(Operation *op) {
    return isa<arith::BitcastOp, arith::AndIOp, arith::AddIOp, arith::MinSIOp,
               arith::MaxSIOp, arith::SubIOp, arith::MulIOp, math::AbsFOp,
               arith::SelectOp, arith::CmpIOp>(op);
  }

  static Value broadcastOrCast(PatternRewriter &rewriter, Location loc,
                               Value scalar, VectorType vecType) {
    if (isa<BlockArgument>(scalar))
      return rewriter.create<vector::BroadcastOp>(loc, vecType, scalar);
    return rewriter.create<UnrealizedConversionCastOp>(loc, vecType, scalar)
        .getResult(0);
  }
};

} // namespace
/// This pass normalizes vector ops to meet HIVM requirements:
///  1. drop unit dims for elementwise and TransferRead/Write ops
///  2. convert vector.multi_reduction to vector.reduction
void NormalizeVectorPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  // To avoid non-deterministic optimization problems, some patterns should only
  // be applied after the dependent patterns finished.
  RewritePatternSet initPatterns(ctx);
  // `TagMultiReductionInit` happens before `OneDimAccMultiReductionToReduction`
  initPatterns.add<TagMultiReductionInit>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(initPatterns)))) {
    signalPassFailure();
  }

  RewritePatternSet patterns(ctx);
  vector::populateDropUnitDimWithShapeCastPatterns(patterns);
  vector::populateVectorTransferDropUnitDimsPatterns(patterns);
  vector::populateFlattenVectorOpsPattern(patterns);

  patterns
      .add<OneDimAccMultiReductionToReduction,
           DropUnitDimFromMulExtendedOp<arith::MulSIExtendedOp>,
           DropUnitDimFromMulExtendedOp<arith::MulUIExtendedOp>,
           ShapeCastDropUnitDimsForMultiDimVectorPattern,
           UnitDimMultiReductionToReduction, utils::ForOpLegalization<true>,
           ShapeCastUnitDimsForBrc, UnitDimsVecTransposeNormalize,
           ConvertCreateMaskTo1D, ConvertConstantMaskTo1D, TruncfScalarToVector,
           FixBitTypeBroadcastTransferReadPattern>(
          patterns.getContext());
  if (funcOp->hasAttr(hivm::VectorFunctionAttr::name))
    patterns.add<UnaryScalarOpToVectorPattern<math::AbsFOp>,
                 UnaryScalarOpToVectorPattern<math::LogOp>,
                 UnaryScalarOpToVectorPattern<math::RoundOp>,
                 BinaryScalarOpToVectorPattern<mathExt::DivFHPOp>,
                 BinaryScalarOpToVectorPattern<arith::AndIOp>,
                 BinaryScalarOpToVectorPattern<arith::XOrIOp>,
                 BinaryScalarOpToVectorPattern<arith::MulFOp>,
                 BinaryScalarOpToVectorPattern<arith::SubFOp>,
                 BinaryScalarOpToVectorPattern<arith::CmpFOp>,
                 FmaOpScalarToVectorPattern,
                 ScalarChainBroadcastToVectorPattern>(
        patterns.getContext());
  patterns.add<TransferReadToGatheringLoadPattern>(patterns.getContext());
  vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
  vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> vector::createNormalizeVectorPass() {
  return std::make_unique<NormalizeVectorPass>();
}

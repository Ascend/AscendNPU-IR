//===---------------------- NormalizeVector.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "bishengir/Dialect/Vector/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Builders.h"
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
    const llvm::fltSemantics &semantics = elemTy.cast<FloatType>().getFloatSemantics();
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
      llvm_unreachable("Unsupport reduce kind.");
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
      llvm_unreachable("Unsupport reduce kind.");
      break;
    }
    denseAttr = DenseElementsAttr::get(
        srcType, rewriter.getIntegerAttr(elemTy, initVal));
  } else {
    llvm_unreachable("Unsupport element data type.");
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
          VectorType::get(ArrayRef<long>(),
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
          VectorType::get(ArrayRef<long>(),
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
    // calculate the product of all dimensions for the 1D mask size
    Value totalSize = nullptr;
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      Value dimSize = op->getOperand(i);
      if (!totalSize) {
        totalSize = dimSize;
      } else {
        totalSize = rewriter.create<arith::MulIOp>(loc, totalSize, dimSize);
      }
    }
    // create 1D mask type
    int64_t returnTotalElements = 1;
    for (int64_t dim : maskType.getShape()) {
      returnTotalElements *= dim;
    }
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
    // calculate return shape
    int64_t returnTotalElements = 1;
    for (int64_t dim : maskType.getShape())
      returnTotalElements *= dim;
    // calculate the product of all dimensions for the 1D mask size active
    int64_t totalActiveElements = 1;
    for (Attribute attr : maskDimSizesAttr.getValue()) {
      auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
      if (!intAttr)
        return failure();
      int64_t numElements = static_cast<int64_t>(intAttr.getValue().getZExtValue());
      totalActiveElements *= numElements;
    }
    // create 1D mask type (flat)
    VectorType flatMaskType =
        VectorType::get({returnTotalElements}, rewriter.getI1Type());
    // new 1D mask_dim_sizes attribute
    SmallVector<int64_t, 1> flatMaskDims;
    flatMaskDims.push_back(totalActiveElements);
    auto flatMaskDimSizesAttr = rewriter.getI64ArrayAttr(flatMaskDims);
    // create the 1D constant_mask and cast back to original shape
    Value flatMask = rewriter.create<vector::ConstantMaskOp>(
        loc, flatMaskType, flatMaskDimSizesAttr);
    Value originalShapeMask =
        rewriter.create<vector::ShapeCastOp>(loc, maskType, flatMask);
    rewriter.replaceOp(op, originalShapeMask);
    return success();
  }
};

class TransferReadToGatheringLoadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult analyzeTransferRead(vector::TransferReadOp readop,
                                    PatternRewriter &rewriter,
                                    AffineMap &permMap, MemRefType &memrefType,
                                    VectorType &destType, bool &isFullMask,
                                    SmallVector<int64_t, 4> &constantMaskBounds,
                                    SmallVector<int64_t, 4> &strides) const {
    // check the permutation map
    if (!permMap.isPermutation())
      return rewriter.notifyMatchFailure(readop, "unsupported permutation map");

    // we don't touch the ones with (1) identity, (2) broadcast (constant)
    if (permMap.isIdentity() || permMap.isConstant())
      return failure();

    memrefType = dyn_cast<MemRefType>(readop.getSource().getType());
    if (!memrefType || !memrefType.hasRank())
      return failure();
    destType = dyn_cast<VectorType>(readop.getResult().getType());
    if (!destType || !destType.hasStaticShape())
      return failure();

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
    // must have static stride (if strided)
    auto memrefLayout = memrefType.getLayout();
    if (auto strided = dyn_cast<StridedLayoutAttr>(memrefLayout)) {
      for (auto v : strided.getStrides()) {
        if (v < 1)
          return failure();
        strides.push_back(v);
      }

    } else {
      // not strided; compute the strides from the dimensions
      strides.resize(shape.size(), 1);
      if (shape.size() > 1) {
        for (unsigned i = shape.size() - 1; i >= 1; --i) {
          strides[i - 1] = strides[i] * shape[i];
        }
      }
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
      int64_t &totalDestSize) const {
    // try to find all gather indices
    // we need to look at the result vector type to find the range
    // for all the source dimensions
    auto destShape = destType.getShape();
    for (unsigned i = 0; i < destShape.size(); ++i)
      totalDestSize *= destShape[i];

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

      auto composeResult = permMap.compose(composeIndices);
      if (composeResult.size() != strides.size())
        llvm_unreachable("Unexpected dimension mismatch");

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

    IntegerType indexVecElementType = IntegerType::get(
        rewriter.getContext(), memrefType.getElementTypeBitWidth());
    auto indexVecType = VectorType::get(totalDestSize, indexVecElementType);

    bool is256xi8 = (indexVecType.getNumElements() == 256) &&
                    (indexVecElementType.getWidth() == 8);

    auto checkIndex = [](const SmallVector<APInt, 0> &src,
                         const DenseElementsAttr &attr) {
      auto values = llvm::to_vector<0>(attr.getValues<APInt>());
      for (unsigned i = 0; i < values.size(); ++i) {
        if (values[i].getSExtValue() != src[i].getSExtValue()) {
          llvm_unreachable("Sub index value error");
        }
      }
    };

    if (is256xi8) {
      SmallVector<APInt, 0> gatherIndices1, gatherIndices2;
      for (unsigned i = 0; i < gatherIndices.size(); ++i) {
        (i % 2 == 0) ? gatherIndices1.push_back(gatherIndices[i])
                     : gatherIndices2.push_back(gatherIndices[i]);
      }

      IntegerType i16Type = IntegerType::get(
          rewriter.getContext(), 16, indexVecElementType.getSignedness());
      VectorType subIndexVecType = VectorType::get(128, i16Type);

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
      llvm_unreachable("TODO support non-constant padding value for "
                       "vector.transfer_read -> vector.gather transform");
    }
    return padding;
  }

  LogicalResult matchAndRewrite(vector::TransferReadOp readop,
                                PatternRewriter &rewriter) const override {
    // convert vector.transfer_read with non-identity permutation map
    // to vector.gather and index computations doing the same operations
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
    computeGatherIndicesAndMasks(
        readop, permMap, memrefType, destType, strides, isFullMask,
        constantMaskBounds, gatherIndices, gatherMasks, firstEnabledBit,
        lastEnabledBit, firstDisabledBit, lastDisabledBit, totalDestSize);

    auto indexVecValues = createIndexVector(readop, rewriter, memrefType,
                                            gatherIndices, totalDestSize);
    Value finalResult;
    decltype(readop.getMask()) newMask;
    VectorType maskType =
        VectorType::get({totalDestSize}, rewriter.getI1Type());
    if (!isFullMask) {
      // check if we can convert the constant mask to something simple
      // currently we only try to create another constant mask
      // if there are more patterns we can support, update them here
      // for 1D vector.constant_mask, check if there exists N
      // such that gatherMasks[0...N-1]==true && gatherMasks[N...]==false
      if (firstEnabledBit < lastEnabledBit &&
          lastEnabledBit + 1 == firstDisabledBit &&
          firstDisabledBit <= lastDisabledBit) {
        ArrayAttr maskDimSizes = rewriter.getI64ArrayAttr({firstDisabledBit});
        auto newConstMask = rewriter.create<vector::ConstantMaskOp>(
            readop.getLoc(), maskType, maskDimSizes);
        newMask = newConstMask;
      } else
        return failure();
    } else {
      // create a mask for full vector size
      ArrayAttr maskDimSizes = rewriter.getI64ArrayAttr({totalDestSize});
      auto newConstMask = rewriter.create<vector::ConstantMaskOp>(
          readop.getLoc(), maskType, maskDimSizes);
      newMask = newConstMask;
    }

    if (indexVecValues.size() == 2) {
      auto eltType = destType.getElementType();
      VectorType subDestType =
        VectorType::get({totalDestSize}, eltType);
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

    rewriter.replaceOp(readop, finalResult);
    return success();
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
           ShapeCastDropUnitDimsForMultiDimVectorPattern,
           TransferReadToGatheringLoadPattern,
           UnitDimMultiReductionToReduction, utils::ForOpLegalization<true>,
           ShapeCastUnitDimsForBrc, UnitDimsVecTransposeNormalize,
           ConvertCreateMaskTo1D, ConvertConstantMaskTo1D>(
          patterns.getContext());
  vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
  vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> vector::createNormalizeVectorPass() {
  return std::make_unique<NormalizeVectorPass>();
}
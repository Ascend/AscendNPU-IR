//===- NormalizeSliceOps.cpp ------- Normalize Slice Ops Pass -------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to normalize slice operations.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace hfusion {

#define GEN_PASS_DEF_NORMALIZESLICEOPS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

static thread_local bool archIsRegbased{false};
static thread_local bool archisAscend950{false};

/// Compute the dropped dimensions of a rank-reducing tensor.extract_slice op or
/// rank-extending tensor.insert_slice op.
static llvm::SmallBitVector
getDroppedDimsForInterleave(ArrayRef<int64_t> reducedShape,
                            ArrayRef<OpFoldResult> mixedSizes) {
  // TODO this is old community function, delete it afterwards
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  int64_t shapePos = 0;

  for (const auto &size : enumerate(mixedSizes)) {
    // Rank-reduced dims must have a static unit dimension.
    bool isStaticUnitSize =
        size.value().is<Attribute>() &&
        llvm::cast<IntegerAttr>(size.value().get<Attribute>()).getInt() == 1;

    if (shapePos == static_cast<int64_t>(reducedShape.size())) {
      // There are no more dims in the reduced shape. All remaining sizes must
      // be rank-reduced dims.
      assert(isStaticUnitSize && "expected unit dim");
      droppedDims.set(size.index());
      continue;
    }

    // Dim is preserved if the size is not a static 1.
    if (!isStaticUnitSize) {
      ++shapePos;
      continue;
    }

    // Dim is preserved if the reduced shape dim is also 1.
    if (reducedShape[shapePos] == 1) {
      ++shapePos;
      continue;
    }

    // Otherwise: Dim is dropped.
    droppedDims.set(size.index());
  }

  return droppedDims;
}

// ExtractSliceOp with the restrictions can be transformed to deinterleave op:
// 1. input_shape[i] == output_shape[i],               i != last_dim
// 2. input_shape[i] == output_shape[i] * channel_num, i != last_dim
// To make sure stride is valid, we enforce additional restriction:
//    `last_dim_stride == channel_num` or `last_dim_size == 1`
//
// This pattern also supports rank-reduced ExtractSliceOp, for example:
// e.g.
//   %1 = extract_slice %0 ... tensor<4x2x2xf16> to tensor<4x2xf16>
//   %2 = use(%1)
// will be normalized to
//   %1 = deinterleave %0 channel<1> : tensor<4x2x2xf16> -> tensor<4x2x1xf16>
//   %collapsed = collapse_shape %1 ... : tensor<4x2x1xf16> into tensor<4x2xf16>
//   %2 = use(%collapsed)
class NormalizeExtractSliceToDeinterleaveOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    // for rank reduce slice like `<Ax2> -> <A>`, refine the slice output type
    // to `<Ax1>`, to fit in the deinterleave pattern
    llvm::SmallBitVector reducedRankRecord = getDroppedDimsForInterleave(
        sliceOp.getResultType().getShape(), sliceOp.getMixedSizes());
    auto refinedOutputType =
        refineOutputIfReduceRank(sliceOp, reducedRankRecord, rewriter);
    if (failed(refinedOutputType)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "sliceOp output cannot be rank-reduced refined");
    }

    // TODO: support getDeInterLeaveChannelNum() != 2
    int64_t channelNum = hfusion::DeinterleaveOp::getDeInterLeaveChannelNum();
    ShapedType outputType = refinedOutputType.value();
    if (!isDeinterleavePattern(sliceOp, outputType, channelNum))
      return rewriter.notifyMatchFailure(
          sliceOp, "slice layout doen't satisfy condition for deinterleave");

    auto loc = sliceOp.getLoc();
    Value inputVal = sliceOp.getSource();
    auto mode = sliceOp.getStaticOffsets().back();
    auto indexAttr = rewriter.getI64IntegerAttr(mode);
    auto deinterleaveOp = rewriter.create<hfusion::DeinterleaveOp>(
        loc, TypeRange{outputType}, inputVal, indexAttr);
    assert(deinterleaveOp.getOutput().size() == 1 &&
           "Initially created deinterleave op just has 1 output");

    if (!reducedRankRecord.any()) {
      rewriter.replaceOp(sliceOp, deinterleaveOp.getOutput());
      return success();
    }
    // for rank reduce slice like `<Ax2> -> <A>`, after we create deinterleave
    // of `<Ax2> -> <Ax1>`, add `<Ax1> -> <A>` collapse for compatibility
    auto reassociation = getReassociationIndicesForCollapse(
        outputType.getShape(), sliceOp.getResultType().getShape());
    assert(reassociation.has_value());
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, deinterleaveOp.getOutput()[0], reassociation.value());
    rewriter.replaceOp(sliceOp, collapseOp);
    return success();
  }

private:
  FailureOr<ShapedType>
  refineOutputIfReduceRank(tensor::ExtractSliceOp sliceOp,
                           const llvm::SmallBitVector &reducedRankRecord,
                           PatternRewriter &rewriter) const {
    ShapedType outputType = sliceOp.getResultType();
    if (!reducedRankRecord.any()) {
      return outputType;
    }

    int srcRank = sliceOp.getSourceType().getRank();
    if (reducedRankRecord.find_first() != srcRank - 1)
      return rewriter.notifyMatchFailure(
          sliceOp, "reduced rank could only be last dimension");

    // Reduced rank size of source must be deinterLeaveChannelNum
    int64_t channelNum = hfusion::DeinterleaveOp::getDeInterLeaveChannelNum();
    if (sliceOp.getSourceType().getShape().back() != channelNum)
      return rewriter.notifyMatchFailure(
          sliceOp, "size of reduced rank axis should equal channel num");

    SmallVector<int64_t> refinedShape(outputType.getShape());
    refinedShape.push_back(1);
    ShapedType refinedType =
        RankedTensorType::get(refinedShape, outputType.getElementType());
    return refinedType;
  }

  bool isDeinterleavePattern(tensor::ExtractSliceOp extractSliceOp,
                             ShapedType refinedOutputType,
                             int64_t channelNum) const {
    ArrayRef<int64_t> srcShape = extractSliceOp.getSourceType().getShape();
    ArrayRef<int64_t> dstShape = refinedOutputType.getShape();

    if (srcShape.size() != dstShape.size())
      return false;

    if (!llvm::equal(llvm::drop_end(srcShape), llvm::drop_end(dstShape))) {
      // rule 1: src/dst non-last dims should equal
      return false;
    }
    if (ShapedType::isDynamic(dstShape.back()) ||
        ShapedType::isDynamic(srcShape.back()) ||
        srcShape.back() != dstShape.back() * channelNum) {
      // rule 2.1: src/dst last dim should be static
      // rule 2.2: src.last_dim = dst.last_dim * channel_num
      return false;
    }

    int64_t lastDimOffset = extractSliceOp.getStaticOffsets().back();
    if (ShapedType::isDynamic(lastDimOffset) || lastDimOffset >= channelNum) {
      // rule 3: last dim offset is static constant and less than channel_num
      return false;
    }

    int64_t lastDimSliceSize = extractSliceOp.getStaticSizes().back();
    int64_t lastDimStride = extractSliceOp.getStaticStrides().back();
    if (lastDimStride == channelNum) {
      // rule 4.1: if last dim stride equal channel_num, can deinterleave it
      return true;
    }
    // rule 4.2: if last dim stride not equal channel_num, but only has one elem
    // sliced, can also deinterleave it
    return lastDimSliceSize == 1;
  }
};

// When ExtractSlice Op is rank reduce, like `<Ax2> -> <A>`, refine the slice
// output to `<Ax1>`, then add `<Ax1> -> <A>` collapse for compatibility.
class NormalizeExtractSliceOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    // If not rank reduce, skip
    llvm::SmallBitVector reducedRankRecord = getDroppedDimsForInterleave(
        sliceOp.getResultType().getShape(), sliceOp.getMixedSizes());
    if (!reducedRankRecord.any()) {
      return rewriter.notifyMatchFailure(
          sliceOp,
          "slice layout doen't satisfy condition for NormalizeExtractSliceOp.");
    }
    // Refine Output Shape
    SmallVector<int64_t> refinedShape(sliceOp.getResultType().getShape());
    for (size_t i = 0; i < reducedRankRecord.size(); ++i) {
      if (reducedRankRecord[i]) {
        refinedShape.insert(refinedShape.begin() + i, {1});
      }
    }
    ShapedType refinedType = RankedTensorType::get(
        refinedShape, sliceOp.getResultType().getElementType());
    auto newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(),
        RankedTensorType::get(refinedShape,
                              sliceOp.getResultType().getElementType()),
        sliceOp.getSource(), sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());
    // collapse from `<Ax1> -> <A>`
    auto reassociation = getReassociationIndicesForCollapse(
        refinedType.getShape(), sliceOp.getResultType().getShape());
    assert(reassociation.has_value());
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        newExtractSliceOp.getLoc(), newExtractSliceOp.getResult(),
        reassociation.value());
    rewriter.replaceOp(sliceOp, collapseOp);
    return success();
  }
};

// %tmp = tensor.insert_slice src0 into %tmpbuffer[0] [64] [2] :
// tensor<64xf16> into tensor<128xf16> %dst = tensor.insert_slice %src1 into
// %tmp[1] [64] [2] : tensor<64xf16> into tensor<128xf16>
struct NormalizeInsertSliceOpToInterleaveOp
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  std::optional<int> getInterLeaveChannelIdx(Operation *op) const {
    if (auto insertSliceOp = llvm::dyn_cast<tensor::InsertSliceOp>(op)) {
      if (insertSliceOp.getStaticOffsets().empty())
        return std::nullopt;
      // dynamic offset return INT64_MIN
      if (insertSliceOp.getStaticOffsets().back() == INT64_MIN)
        return std::nullopt;
      return insertSliceOp.getStaticOffsets().back();
    }
    return std::nullopt;
  }

  bool isInterLeavePartialPattern(tensor::InsertSliceOp insertSliceOp,
                                  Value curSource,
                                  int64_t interLeaveChannelNums) const {
    SmallVector<int64_t> srcShape(
        dyn_cast<ShapedType>(curSource.getType()).getShape());
    SmallVector<int64_t> dstShape(
        dyn_cast<ShapedType>(insertSliceOp.getDest().getType()).getShape());

    if (dstShape.size() != srcShape.size())
      return false;

    // Correspond to rule 1&2
    if (!std::equal(srcShape.begin(), srcShape.end() - 1, dstShape.begin()) ||
        ShapedType::isDynamic(dstShape.back()) ||
        ShapedType::isDynamic(srcShape.back()) ||
        dstShape.back() != srcShape.back() * interLeaveChannelNums) {
      return false;
    }

    // Collect layout offset
    std::optional<int> interLeaveChannelIdx =
        getInterLeaveChannelIdx(insertSliceOp);
    if (!interLeaveChannelIdx.has_value()) {
      return false;
    }
    if (interLeaveChannelIdx > interLeaveChannelNums) {
      return false;
    }

    // Correspond to rule 3
    if (insertSliceOp.getStaticStrides().back() != interLeaveChannelNums) {
      return false;
    }
    return true;
  }

  // Actually, tensor::InsertSliceOp could express any dimension insertion
  // flexibly with layout info.
  //
  // While current hfusion::InterleaveOp just wanna match pattern where
  // 1. non-last diemsnion shape of input and output should be equal
  // 2. tail shape scale between dst and src equals channelNum, which also
  // means
  //    last dimension of all types could only be static
  // 3. In layout, last dimension stride equals channelNum
  // 4. In layout, last dimension offsets of all candidate InsertSliceOp
  // should
  //    make a range of [0, channelNum)
  //
  // And for rank-extended state of tensor::InsertSliceOp, where source rank
  // may less than destination, InterleaveOp just support extended rank is
  // last dimension and size equals channelNum, then explicitly expand shape
  // on src before create InterleaveOp
  LogicalResult traceInterLeavePattern(tensor::InsertSliceOp insertSliceOp,
                                       llvm::BitVector &findChannels,
                                       SmallVector<Value> &inputs,
                                       int64_t interLeaveChannelNums,
                                       PatternRewriter &rewriter) const {
    Value curSrc = insertSliceOp.getSource();

    llvm::SmallBitVector extendedRankRecord =
        getDroppedDimsForInterleave(insertSliceOp.getSourceType().getShape(),
                                    insertSliceOp.getMixedSizes());
    tensor::ExpandShapeOp expandOp = nullptr;
    if (extendedRankRecord.any()) {
      if (extendedRankRecord.find_first() !=
          static_cast<int>(extendedRankRecord.size()) - 1)
        return rewriter.notifyMatchFailure(
            insertSliceOp, "extended rank could only be last dimension");

      // Extented rank size of destination must be interLeaveChannelNums
      if (dyn_cast<ShapedType>(insertSliceOp.getDest().getType())
              .getShape()
              .back() != interLeaveChannelNums)
        return rewriter.notifyMatchFailure(
            insertSliceOp,
            "size of extended rank axis should equal channel num");

      auto originType = llvm::dyn_cast<RankedTensorType>(curSrc.getType());
      if (!originType) {
        llvm::errs() << "Error: Expected RankedTensorType, got: "
                     << curSrc.getType() << "\n";
        return failure();
      }
      SmallVector<int64_t> shape(originType.getShape());

      // Here represents last dimension which is only extended rank
      shape.push_back(1);
      RankedTensorType newType =
          RankedTensorType::get(shape, originType.getElementType());
      std::optional<SmallVector<ReassociationIndices>> reassociation =
          getReassociationIndicesForReshape(originType, newType);
      assert(reassociation.has_value());

      expandOp = rewriter.create<tensor::ExpandShapeOp>(
          insertSliceOp.getLoc(), newType, curSrc, reassociation.value());
      curSrc = expandOp.getResult();
    }

    if (!isInterLeavePartialPattern(insertSliceOp, curSrc,
                                    interLeaveChannelNums)) {
      if (expandOp) {
        rewriter.eraseOp(expandOp);
      }
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "current tensor::InsertSliceOp layout doen't satisfy condition "
          "to be converted to hfusion::InterleaveOp");
    }

    auto channelIdxMaybe = getInterLeaveChannelIdx(insertSliceOp);
    if (!channelIdxMaybe.has_value()) {
      if (expandOp) {
        rewriter.eraseOp(expandOp);
      }
      return failure();
    }
    int channelIdx = channelIdxMaybe.value();
    // set channelIdx-bit to findChannels and push corresponding input
    findChannels[channelIdx] = true;
    inputs[channelIdx] = curSrc;

    // findChannels all true
    // Correspond to rule 4
    if (findChannels == llvm::BitVector(findChannels.size(), true)) {
      return success();
    }

    // trace further
    auto dstDefiningOp =
        insertSliceOp->getOperand(1).getDefiningOp<tensor::InsertSliceOp>();
    if (!dstDefiningOp) {
      if (expandOp) {
        rewriter.eraseOp(expandOp);
      }
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "tensor::InsertSliceOp chain from current op can't reach "
          "interLeave channel num");
    }
    if (failed(traceInterLeavePattern(dstDefiningOp, findChannels, inputs,
                                      interLeaveChannelNums, rewriter))) {
      if (expandOp) {
        rewriter.eraseOp(expandOp);
      }
      return failure();
    }
    return success();
  }

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!insertSliceOp.hasPureTensorSemantics()) {
      return failure();
    }
    // TODO: find interLeaveChannelNums greedily.
    const int64_t interLeaveChannelNums =
        hfusion::InterleaveOp::getInterLeaveChannelNums();
    llvm::BitVector findChannels(interLeaveChannelNums, false);
    SmallVector<Value> inputs(interLeaveChannelNums);
    if (traceInterLeavePattern(insertSliceOp, findChannels, inputs,
                               interLeaveChannelNums, rewriter)
            .failed()) {
      return failure();
    }

    auto loc = insertSliceOp.getLoc();
    auto interleaveop = rewriter.create<hfusion::InterleaveOp>(
        loc, ValueRange(insertSliceOp.getDest()), ValueRange(inputs));
    rewriter.replaceOp(insertSliceOp, interleaveop);
    return success();
  }
};

namespace {
struct NormalizeHFusionSliceOpsPass
    : public impl::NormalizeSliceOpsBase<NormalizeHFusionSliceOpsPass> {
public:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
    archIsRegbased = hacc::utils::isRegBasedArch(moduleOp);
    archisAscend950 = hacc::utils::isAscend950(moduleOp);
    RewritePatternSet patterns(&getContext());
    patterns.add<NormalizeExtractSliceToDeinterleaveOp>(patterns.getContext());
    patterns.add<NormalizeInsertSliceOpToInterleaveOp>(patterns.getContext());
    if (!archIsRegbased) {
      patterns.add<NormalizeExtractSliceOp>(patterns.getContext());
    }
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace hfusion
} // namespace mlir

std::unique_ptr<mlir::Pass>
mlir::hfusion::createHFusionNormalizeSliceOpsPass() {
  return std::make_unique<NormalizeHFusionSliceOpsPass>();
}

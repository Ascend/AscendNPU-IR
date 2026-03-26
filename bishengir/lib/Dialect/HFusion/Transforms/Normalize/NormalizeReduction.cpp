//===- NormalizeReduction.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"

namespace mlir::hfusion {

// Normalize argmax and argmin
// hfusion.reduce_with_index <max>(src)
// is normalized to
// src_nan_mask = hfusion.isnan(src)
// new_src = hfusion.select(src_nan_mask, -inf, src)
// hfusion.reduce_with_index <max> (new_src)
struct NormalizeArgMinMaxOp
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    SmallVector<Value> inputs = op.getInputs();
    assert(inputs.size() == 2);
    Value src = inputs[0];
    Value src1 = inputs[1];

    static constexpr llvm::StringLiteral kAlreadyInitalizeInit =
        "already_initialize_init";
    if (op->hasAttr(kAlreadyInitalizeInit)) {
      return failure();
    }

    Type elemType = getElementTypeOrSelf(src);
    if (elemType.isInteger()) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    Location loc = op.getLoc();
    auto kind = op.getReduceKind();
    auto tieBreakLeft = op.getTieBreakLeftAttr();
    auto dims = op.getDimensions();
    auto unsignedSource = op.getUnsignedSrcAttr();
    bool isMin = kind.getReduceWithIndexKind() == ReduceWithIndexKind::MIN;

    auto infSign = isMin ? 1 : -1;
    double signedInf = infSign * std::numeric_limits<double>::infinity();

    auto infValue =
        utils::createConstantOp<double>(rewriter, loc, elemType, signedInf);
    utils::createConstantOp<double>(rewriter, loc, elemType, 0.);

    auto srcMask = utils::createEmptyOpWithTargetElemType(rewriter, loc, src,
                                                          rewriter.getI1Type());
    auto srcNanMask =
        rewriter.create<hfusion::IsNanOp>(loc, srcMask.getType(), src)
            .getResult();

    auto srcNanMasked = utils::createEmptyOp(rewriter, loc, src);
    srcNanMasked =
        rewriter
            .create<hfusion::SelectOp>(loc, TypeRange(srcNanMasked),
                                       ValueRange({srcNanMask, infValue, src}),
                                       ValueRange(srcNanMasked))
            .getResults()[0];

    auto srcNanVals = utils::createEmptyOp(rewriter, loc, op.getResults()[0]);
    auto srcNanIdxs = utils::createEmptyOp(rewriter, loc, op.getResults()[1]);
    auto srcNanReduceOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        loc, TypeRange{srcNanVals.getType(), srcNanIdxs.getType()},
        /*input*/ ValueRange{srcNanMasked, src1},
        /*outputValue&Index*/
        ValueRange{srcNanVals, srcNanIdxs}, kind, unsignedSource, tieBreakLeft,
        dims);
    rewriter.modifyOpInPlace(srcNanReduceOp, [&]() {
      srcNanReduceOp->setAttr(kAlreadyInitalizeInit,
                              UnitAttr::get(op->getContext()));
    });
    rewriter.replaceOp(op, srcNanReduceOp);
    return success();
  }
};

static void replaceF16ResultsWithF32(const SmallVector<Value> &oldResults,
                                     const SmallVector<Value> &newResults,
                                     PatternRewriter &rewriter) {
  assert(oldResults.size() == newResults.size() &&
         "result sizes mismatch when replace op results");
  for (const auto [idx, oldResult] : llvm::enumerate(oldResults)) {
    Value newResult = newResults[idx];
    if (!isF16ElemType(oldResult.getType())) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
      continue;
    }

    Value castResult = castTo(rewriter, newResult, rewriter.getF16Type());
    rewriter.replaceAllUsesWith(oldResult, castResult);
  }
}

/// normalize f16 reduce_sum as bellow for high precision
/// eg.
///    reduce_sum f16
/// is normalized to
///    cast f16 to f32
///    reduce_sum f32
///    cast f32 to f16
struct NormalizeF16ReduceSum : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();

    if (!hasF16ElemType(inputs) && !hasF16ElemType(inits)) {
      return failure();
    }

    if (!shouldComputeF16ToF32(op)) {
      return failure();
    }

    SmallVector<Value> newInputs =
        normalizeSrcToTargetType<float, Float32Type>(rewriter, inputs);
    SmallVector<Value> newInits =
        normalizeSrcToTargetType<float, Float32Type>(rewriter, inits);
    Operation *newOp =
        createNewReduceOp(op, rewriter, rewriter.getF16Type(),
                          rewriter.getF32Type(), newInputs, newInits);
    replaceF16ResultsWithF32(op->getResults(), newOp->getResults(), rewriter);

    return success();
  }

private:
  bool shouldComputeF16ToF32(linalg::ReduceOp op) const {
    Block *block = &op.getRegion().front();
    for (Operation &bodyOp : *block) {
      if (dyn_cast_or_null<arith::AddFOp>(bodyOp)) {
        return true;
      }
    }
    return false;
  }
};

struct NormalizeI8Transpose : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {

    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    Value input = op.getInput();
    Value init = op.getInit();
    Location loc = op.getLoc();
    if (!isI8ElemType(input.getType()) && !isI8ElemType(init.getType())) {
      return failure();
    }
    Value newInput = hfusion::castTo(rewriter, input, rewriter.getI16Type(),
                                     hfusion::RoundMode::TRUNC);
    Value newInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, init, rewriter.getI16Type());
    Value newTransOp = rewriter
                           .create<linalg::TransposeOp>(loc, newInput, newInit,
                                                        op.getPermutation())
                           ->getResult(0);
    Value newResult =
        hfusion::castTo(rewriter, newTransOp, rewriter.getI8Type(),
                        hfusion::RoundMode::TRUNC, init,
                        /* enableOverflow = */ false);
    rewriter.replaceAllUsesWith(op->getResult(0), newResult);
    rewriter.eraseOp(op);
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// VReduceOp RA [b, r, a]-> transpose [b, a, r] + AR reduce [b, a]
// ===----------------------------------------------------------------------===//
/// Normalize reduceRa_with_index to transpose + reduceAR_with_index +
/// reshape so its performance will be better in some cases
///
/// e.g.
/// %reduced:2 = hfusion.reduce_with_index
///               ins(%0, %1 : tensor<64x32xf32>, tensor<64x32xi32>)
///               outs(%25, %26 : tensor<32xf32>, tensor<32xi32>)
///               dimensions = [0]
///
/// will be normalized to
///
/// %empty_0 = tensor.empty() : tensor<32x64xf32>
/// %transposed_0 = linalg.transpose ins(%0 : tensor<64x32xf32>)
///                   outs(%empty_0 : tensor<32x64xf32>)
///                   permutation = [1, 0]
/// %empty_1 = tensor.empty() : tensor<32x64xi32>
/// %transposed_1 = linalg.transpose ins(%0 : tensor<64x32xi32>)
///                   outs(%empty_1 : tensor<32x64xi32>) permutation = [1,
///                   0]
/// %reduced:2 = hfusion.reduce_with_index
///     ins(%transposed_0, %transposed_1 : tensor<32x64xf32>,
///     tensor<32x64xi32>) outs(%25, %26 : tensor<32xf32>, tensor<32xi32>)
///     dimensions = [1]
struct ReduceWithIndexRAHighPerformance
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  static Value getTransposedValue(Value source, const Location loc,
                                  PatternRewriter &rewriter,
                                  llvm::ArrayRef<int> order) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto sourceRank = sourceType.getRank();

    SmallVector<int64_t> perm(order);
    SmallVector<int64_t> originalShape(sourceType.getShape());
    SmallVector<int64_t> transposedShape(sourceRank);
    for (int64_t i = 0; i < sourceRank; i++) {
      transposedShape[i] = originalShape[perm[i]];
    }

    Value transposeInit = rewriter.create<tensor::EmptyOp>(
        loc, transposedShape, sourceType.getElementType());

    Value transpose =
        rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
            .getResults()[0];

    return transpose;
  }

  // limitation of memref'shape from hivm::transposeOp
  // if we have a tensor like [b, r, a]
  // if eleType is float16
  // The strides of both r, a need to be divisible by 16.
  // if eleType is float32
  // The stride of a or r needs to be divisible by 16,
  // and the other's needs to be divisible by 8.
  // reducedim must be a single one
  static bool
  isSizeCompatibleForTransposeForReduceOp(PatternRewriter &rewriter, Value src,
                                          SmallVector<int64_t> srcShape,
                                          int reduceDim) {
    auto floatEleType =
        dyn_cast<FloatType>(getElementTypeOrSelf(src.getType()));
    // at this level
    // reduce int have been transformed into reduce float for now
    if (!floatEleType) {
      return false;
    }
    const unsigned num_per_block =
        utils::INTR_BYTES_PER_BLOCK /
        (floatEleType.getWidth() / utils::INTR_BITS_PER_BYTE);

    // get total A axis size
    int totalRShape = srcShape[reduceDim];
    int totalAShape = 1;
    for (size_t i = static_cast<size_t>(reduceDim) + 1lu; i < srcShape.size();
         i++) {
      totalAShape *= srcShape[i];
    }

    // refer to the num of the registers
    // used in transpose operation
    const int registerCount = 16;

    if ((totalRShape % num_per_block == 0 &&
         totalAShape % registerCount == 0) ||
        (totalAShape % num_per_block == 0 && totalRShape % registerCount == 0))
      return true;

    return false;
  }

  Value reshapeOpRewriterHelper(Value input, ArrayRef<int64_t> reshape,
                                PatternRewriter &rewriter, Location loc) const {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    // Prepare reshaped tensor type
    auto reshapeType =
        RankedTensorType::get(reshape, inputType.getElementType());
    // Prepare reshape info value
    auto reshapeInfo = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64TensorAttr(reshape));
    return rewriter.create<tensor::ReshapeOp>(loc, reshapeType, input,
                                              reshapeInfo);
  }

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    // reduceOp only handles tensors
    auto loc = op.getLoc();
    auto src = op.getInputs()[0];
    ShapedType srcShapeType = cast<ShapedType>(src.getType());
    ArrayRef<int64_t> srcShape = srcShapeType.getShape();

    auto srcShapeRank = srcShapeType.getRank();

    // only support one axis reduce
    // only handle transpose of ra
    auto reduceDims = op.getDimensions();
    auto reduceDim = reduceDims[0];
    if (reduceDims.size() > 1 || reduceDim == srcShapeRank - 1) {
      return failure();
    }

    SmallVector<Value> newInputs;
    newInputs.insert(newInputs.end(), op.getInputs().begin(),
                     op.getInputs().end());

    if (!isSizeCompatibleForTransposeForReduceOp(
            rewriter, src, SmallVector<int64_t>{srcShape}, reduceDim)) {
      return failure();
    }

    // knowing that we are processing with reduce ra with index
    // then we transpose the tensor
    // create transposeOp
    SmallVector<int32_t> transposePerm;
    for (int i = 0; i < srcShapeRank; i++) {
      if (i != reduceDim)
        transposePerm.push_back(i);
    }
    transposePerm.push_back(reduceDim);

    // create mapper to map the inputs to the new reduce op
    IRMapping mapper;
    for (const auto &[idx, operand] : llvm::enumerate(op.getInputs())) {
      newInputs[idx] = getTransposedValue(newInputs[idx], loc, rewriter,
                                          ArrayRef<int32_t>(transposePerm));
      mapper.map(operand, newInputs[idx]);
    }

    // clone & replace the reduceOp
    SmallVector<int64_t> newReduceDim{srcShapeRank - 1};
    auto newReduceOp = rewriter.clone(*op, mapper);
    dyn_cast<hfusion::ReduceWithIndexOp>(newReduceOp)
        .setDimensions(ArrayRef<int64_t>(newReduceDim));

    rewriter.replaceOp(op, newReduceOp);
    return success();
  }
};

/// remove linalg.fill Ops that are fed into reduce_with_index as inits
/// becuase everything is handle in template functions for regbase, and
/// every valid reduce_with_index gets lowered to template functions, unlike
/// membase that some lowers to scalar loops
/// For example:
/// %0 = linalg.fill -> tensor<37x3xf32>
/// %1 = linalg.fill -> tensor<37x3xi32>
/// %idx = linalg.fill -> tensor<37x5x3xi32>
/// %2:2 = hfusion.reduce_with_index <max> ins(%data,%idx)
/// outs(%0, %1 :tensor<37x3xf32>, tensor<37x3xi32>) dimensions = [1]
///
/// becomes
///
/// %0 = tensor.empty() -> tensor<37x3xf32>
/// %1 = tensor.empty() -> tensor<37x3xi32>
/// %idx = tensor.empty() -> tensor<37x5x3xi32>
/// %2:2 = hfusion.reduce_with_index <max> ins(%data,%idx)
/// outs(%0, %1 :tensor<37x3xf32>, tensor<37x3xi32>) dimensions = [1]
struct NormalizeReduceWithIndexInitsAndInputs
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool changed = false;

    SmallVector<Value> newInits;
    newInits.reserve(op.getNumDpsInits());
    for (Value init : op.getInits()) {
      if (init.getDefiningOp<tensor::EmptyOp>()) {
        newInits.push_back(init);
        continue;
      }

      auto initTy = dyn_cast<RankedTensorType>(init.getType());
      if (!initTy || !initTy.hasStaticShape())
        return failure();

      // Replace with tensor.empty of the same static shape
      Value empty = rewriter.create<tensor::EmptyOp>(loc, initTy.getShape(),
                                                     initTy.getElementType());
      newInits.push_back(empty);
      changed = true;
    }

    SmallVector<Value> newInputs(op.getInputs().begin(), op.getInputs().end());
    Value indexInput = newInputs[1];

    if (!indexInput.getDefiningOp<tensor::EmptyOp>() &&
        !isa<BlockArgument>(indexInput)) {
      auto ty = dyn_cast<RankedTensorType>(indexInput.getType());
      if (!ty || !ty.hasStaticShape())
        return failure();

      Value empty = rewriter.create<tensor::EmptyOp>(loc, ty.getShape(),
                                                     ty.getElementType());
      newInputs[1] = empty;
      changed = true;
    }

    if (!changed)
      return failure();

    auto newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        loc, op->getResultTypes(), newInputs, newInits, op.getReduceKindAttr(),
        op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getDimensionsAttr());

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// helper function to cast i64 index to i32 which we support
struct NormalizeReduceIndexToI32
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    // Basic checks
    if (op->getNumResults() < 2)
      return rewriter.notifyMatchFailure(op, "expects at least two results");

    Location loc = op->getLoc();
    MLIRContext *ctx = op->getContext();
    Value oldIndexVal = op->getResult(1);
    auto oldIndexTT = mlir::dyn_cast<RankedTensorType>(oldIndexVal.getType());
    if (!oldIndexTT)
      return rewriter.notifyMatchFailure(
          op, "index result is not RankedTensorType");
    auto oldElemTy = mlir::dyn_cast<IntegerType>(oldIndexTT.getElementType());
    if (!oldElemTy)
      return rewriter.notifyMatchFailure(op, "index element is not integer");

    // if already i32, nothing to do
    if (oldElemTy.getWidth() == 32)
      return rewriter.notifyMatchFailure(op, "index already i32");

    // target i32 element type and new index RankedTensorType
    IntegerType i32Ty = IntegerType::get(ctx, 32);
    RankedTensorType newIndexTT =
        RankedTensorType::get(oldIndexTT.getShape(), i32Ty);

    // build new inputs with i32 type for index
    SmallVector<Value, 8> newInputs;
    newInputs.reserve(op.getInputs().size());
    for (Value in : op.getInputs()) {
      newInputs.push_back(makeI32ValueFor(rewriter, loc, in, oldElemTy, i32Ty));
    }

    // build new inputs with i32 type for index
    SmallVector<Value, 4> newInits;
    newInits.reserve(op.getInits().size());
    for (Value init : op.getInits()) {
      newInits.push_back(
          makeI32ValueFor(rewriter, loc, init, oldElemTy, i32Ty));
    }

    // new result types: keep first (reduced value) unchanged, second becomes
    // tensor<...xi32>
    Type valueResultTy = op->getResult(0).getType();
    SmallVector<Type, 2> newResultTypes = {valueResultTy, newIndexTT};

    // create the new reduce_with_index op with i32 index results and i32
    // inputs/inits
    auto newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        loc, ArrayRef<Type>(newResultTypes), ArrayRef<Value>(newInputs),
        ArrayRef<Value>(newInits), op.getReduceKindAttr(),
        op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getDimensionsAttr());

    Value newIndexVal = newOp->getResult(1);
    Value replacedIndexVal;
    if (oldElemTy.getWidth() != 32) {
      // cast back to original index element type for consumers
      replacedIndexVal = hfusion::castTo(rewriter, newIndexVal, oldElemTy);
    } else {
      replacedIndexVal = newIndexVal;
    }

    op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
    op->getResult(1).replaceAllUsesWith(replacedIndexVal);
    rewriter.eraseOp(op);
    return success();
  }

private:
  // helper function to produce an i32-typed Value for a tensor Value that
  // currently has oldElemTy. If the defining op is tensor.empty, synthesize a
  // new tensor.empty with i32 element type preserving dynamic sizes. Otherwise,
  // use hfusion::castTo to cast the tensor's element type.
  static Value makeI32ValueFor(RewriterBase &rewriter, Location loc, Value val,
                               Type oldElemTy, Type i32Ty) {
    // If not a ranked tensor or element doesn't match oldElemTy, just return
    // val
    auto rt = mlir::dyn_cast<RankedTensorType>(val.getType());
    if (!rt)
      return val;
    auto elem = mlir::dyn_cast<IntegerType>(rt.getElementType());
    if (!elem || elem != oldElemTy)
      return val;

    // create a new tensor.empty with the same
    // shape but i32 element type
    if (Operation *def = val.getDefiningOp()) {
      if (auto emptyOp = dyn_cast<tensor::EmptyOp>(def)) {
        RankedTensorType newRT = RankedTensorType::get(rt.getShape(), i32Ty);
        // collect dynamic sizes (works if emptyOp has them)
        SmallVector<Value, 4> dynSizes;
        for (Value ds : emptyOp.getDynamicSizes())
          dynSizes.push_back(ds);
        return rewriter.create<tensor::EmptyOp>(loc, newRT, dynSizes);
      }
    }

    return hfusion::castTo(rewriter, val, i32Ty);
  }
};

void populateNormalizePreReductionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (!archIsRegbased) {
    patterns.add<ReduceWithIndexRAHighPerformance>(ctx);
  }
  if (archIsRegbased) {
    patterns.add<NormalizeReduceWithIndexInitsAndInputs>(ctx);
    patterns.add<NormalizeReduceIndexToI32>(ctx);
  }
}

void populateNormalizeReductionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeF16ReduceSum>(ctx);
  patterns.add<ReduceWithIndexRAHighPerformance>(ctx);
}

void populateNormalizeFinalReductionPatterns(RewritePatternSet &patterns) {
  if (archIsRegbased)
    patterns.add<NormalizeArgMinMaxOp>(patterns.getContext());
}
} // namespace mlir::hfusion

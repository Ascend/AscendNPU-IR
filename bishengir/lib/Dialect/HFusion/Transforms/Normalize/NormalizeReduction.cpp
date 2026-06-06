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
#include "bishengir/Dialect/HFusion/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"
#include "bishengir/Transforms/Normalize/NormalizeReductionTemplate.h"

namespace mlir {

/// HFusion hooks for the shared argmin/argmax normalization template.
struct HFusionNormalizeArgMinMaxTraits : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeArgMinMax(hfusion::ReduceWithIndexOp op) {
    if (!op.hasPureTensorSemantics() || isReductionInitAlreadyNormalized(*op))
      return false;
    auto inputs = op.getInputs();
    return inputs.size() == 2 &&
           !getElementTypeOrSelf(inputs[0].getType()).isInteger();
  }

  static bool isMinReduction(hfusion::ReduceWithIndexOp op) {
    return op.getReduceKind().getReduceWithIndexKind() ==
           hfusion::ReduceWithIndexKind::MIN;
  }

  static Value createIsNanMask(PatternRewriter &rewriter, Location loc,
                               Value src) {
    Value srcMask = utils::createEmptyOpWithTargetElemType(rewriter, loc, src,
                                                           rewriter.getI1Type());
    return rewriter.create<hfusion::IsNanOp>(loc, srcMask.getType(), src)
        .getResult();
  }
};

/// HFusion hooks for promoting f16 reduce-sum to f32 accumulation.
struct HFusionNormalizeF16ReduceSumTraits
    : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeF16ReduceSum(linalg::ReduceOp op) {
    if (!op.hasPureTensorSemantics())
      return false;

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hfusion::hasF16ElemType(inputs) && !hfusion::hasF16ElemType(inits))
      return false;

    Block *block = &op.getRegion().front();
    return llvm::any_of(*block, [](Operation &bodyOp) {
      return isa<arith::AddFOp>(bodyOp);
    });
  }

  static Operation *createPromotedReduceOp(linalg::ReduceOp op,
                                           PatternRewriter &rewriter,
                                           SmallVector<Value> &newInputs,
                                           SmallVector<Value> &newInits) {
    return hfusion::createNewReduceOp(op, rewriter, rewriter.getF16Type(),
                                      rewriter.getF32Type(), newInputs,
                                      newInits);
  }
};

/// HFusion hooks for replacing reduce-with-index shape-only operands with
/// `tensor.empty`.
struct HFusionNormalizeReduceWithIndexInitsAndInputsTraits
    : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeReduceWithIndexInitsAndInputs(
      hfusion::ReduceWithIndexOp op) {
    return true;
  }
};

/// HFusion hooks for routing index tensors through an i32 reduction kernel.
struct HFusionNormalizeReduceIndexToI32Traits
    : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeReduceIndexToI32(hfusion::ReduceWithIndexOp op) {
    return hasNonI32ReductionIndexResult(op);
  }
};

using NormalizeArgMinMaxOp = NormalizeArgMinMaxOpTemplate<
    hfusion::ReduceWithIndexOp, HFusionNormalizeArgMinMaxTraits>;
using NormalizeF16ReduceSum =
    NormalizeF16ReduceSumTemplate<linalg::ReduceOp,
                                  HFusionNormalizeF16ReduceSumTraits>;
using NormalizeReduceWithIndexInitsAndInputs =
    NormalizeReduceWithIndexInitsAndInputsTemplate<
        hfusion::ReduceWithIndexOp,
        HFusionNormalizeReduceWithIndexInitsAndInputsTraits>;
using NormalizeReduceIndexToI32 =
    NormalizeReduceIndexToI32Template<hfusion::ReduceWithIndexOp,
                                      HFusionNormalizeReduceIndexToI32Traits>;

} // namespace mlir

namespace mlir::hfusion {
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

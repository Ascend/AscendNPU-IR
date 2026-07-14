//===-------- NormalizeReductionTemplate.h --------------------------------===//
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

#ifndef BISHENGIR_TRANSFORMS_REGBASE_NORMALIZE_NORMALIZEREDUCTIONTEMPLATE_H
#define BISHENGIR_TRANSFORMS_REGBASE_NORMALIZE_NORMALIZEREDUCTIONTEMPLATE_H

#include "bishengir/Transforms/regbase/Normalize/Utils/ReductionTemplateHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include <limits>

namespace mlir {

/// Normalizes floating-point argmax/argmin by making NaN handling explicit
/// before rebuilding the reduction
///
/// argmax(x) -> argmax(select(isnan(x), -inf, x))
/// argmin(x) -> argmin(select(isnan(x), +inf, x))
///
/// NaNs are replaced with the least favorable sentinel so they never win the
/// comparison.
template <typename ReduceWithIndexOpType, typename Traits>
struct NormalizeArgMinMaxOpTemplate
    : public OpRewritePattern<ReduceWithIndexOpType> {
public:
  using OpRewritePattern<ReduceWithIndexOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceWithIndexOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeArgMinMax(op))
      return failure();

    Value src = op.getDpsInputs()[0];
    Type elemType = getElementTypeOrSelf(src.getType());
    Location loc = op.getLoc();

    double signedInf = (Traits::isMinReduction(op) ? 1.0 : -1.0) *
                       std::numeric_limits<double>::infinity();
    Value infValue = rewriter.create<arith::ConstantOp>(
                loc, elemType, rewriter.getFloatAttr(elemType, signedInf))
            .getResult();

    Value srcNanMask = Traits::createIsNanMask(rewriter, loc, src);
    Value srcNanMaskDst = utils::createEmptyOp(rewriter, loc, src);
    Value srcNanMasked = Traits::createSelectOp(rewriter, loc, srcNanMask,
                                                infValue, src, srcNanMaskDst);
    SmallVector<Value> newInputs(op.getDpsInputs());
    newInputs.front() = srcNanMasked;
    SmallVector<Value> newInits =
        createReductionResultEmpties(rewriter, loc, op);
    if (newInits.size() != op->getNumResults())
      return failure();

    Operation *newReduceOp =
        Traits::createReduceWithIndexOp(rewriter, loc, op, newInputs, newInits);
    markReductionInitNormalized(*newReduceOp);
    rewriter.replaceOp(op, newReduceOp->getResults());
    return success();
  }
};

/// Normalizes f16 reduce-sum into f32 accumulation:
///   reduce_sum<f16>(x, init)
///     => cast to f32, reduce_sum<f32>, cast back
///
/// The key change is the accumulator type. Computing
///   s = sum_i x_i
/// in f32 avoids repeated f16 rounding during the reduction.
template <typename ReduceOpType, typename Traits>
struct NormalizeF16ReduceSumTemplate : public OpRewritePattern<ReduceOpType> {
public:
  using OpRewritePattern<ReduceOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeF16ReduceSum(op))
      return failure();

    SmallVector<Value> newInputs =
        promoteF16ReductionValuesToF32<Traits>(rewriter, op.getDpsInputs());
    SmallVector<Value> newInits =
        promoteF16ReductionValuesToF32<Traits>(rewriter, op.getDpsInits());
    Operation *newOp =
        Traits::createPromotedReduceOp(op, rewriter, newInputs, newInits);
    if (failed(replacePromotedReductionResults<Traits>(op, *newOp, rewriter)))
      return failure();
    return success();
  }
};

/// Moves a non-innermost reduce-with-index axis to the innermost position by
/// inserting transpose(s) before rebuilding the reduction.
///
/// For a tensor
///   [prefix, r, suffix]
/// reduced along `r`, the rewrite builds
///   transpose([prefix, r, suffix]) -> [prefix, suffix, r]
/// and then reduces the last axis. In flattened form this changes
///   [B, R, A] reduce R
/// into
///   [B, A, R] reduce R
/// so the reduced dimension becomes contiguous.
///
/// Dialect traits decide whether the results are already in the desired layout
/// or must be transposed back.
template <typename ReduceWithIndexOpType, typename Traits>
struct ReduceWithIndexRAHighPerformanceTemplate
    : public OpRewritePattern<ReduceWithIndexOpType> {
public:
  using OpRewritePattern<ReduceWithIndexOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceWithIndexOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeRAHighPerformance(op))
      return failure();

    Value src = op.getDpsInputs().front();
    auto srcType = dyn_cast<RankedTensorType>(src.getType());
    if (!srcType || !srcType.hasStaticShape())
      return failure();

    ArrayRef<int64_t> reduceDims = Traits::getRAHighPerformanceReduceDims(op);
    int64_t srcRank = srcType.getRank();
    if (reduceDims.size() != 1)
      return failure();

    int64_t reduceDim = reduceDims.front();
    if (reduceDim == srcRank - 1)
      return failure();

    if (!isSizeCompatibleForTransposeForReduceOp(src, srcType.getShape(),
                                                 reduceDim))
      return failure();

    Location loc = op.getLoc();
    SmallVector<int64_t> transposePerm;
    transposePerm.reserve(srcRank);
    for (int64_t dim = 0; dim < srcRank; ++dim) {
      if (dim != reduceDim)
        transposePerm.push_back(dim);
    }
    transposePerm.push_back(reduceDim);

    SmallVector<Value> newInputs = transposeReductionValueRange<Traits>(
        rewriter, loc, op.getDpsInputs(), transposePerm);
    if (newInputs.size() != op.getDpsInputs().size())
      return failure();

    SmallVector<Value> newInits(op.getDpsInits());

    // After `transposePerm = [prefix, suffix, reduceDim]`, the reduction is
    // always rebuilt on the last axis:
    //   src[B, R, A] -> transposed[B, A, R] -> reduce dim 2.
    SmallVector<int64_t> newReduceDims{srcRank - 1};
    Operation *newOp = Traits::createRAHighPerformanceReduceOp(
        rewriter, loc, op, newInputs, newInits, newReduceDims, transposePerm);
    if (!newOp)
      return failure();

    if (isReductionInitAlreadyNormalized(*op))
      markReductionInitNormalized(*newOp);

    SmallVector<Value> replacements;
    if (reductionResultsNeedLayoutRestore(*op, *newOp)) {
      SmallVector<int64_t> inversePerm =
          invertReductionPermutation(transposePerm);
      replacements = transposeReductionValueRange<Traits>(
          rewriter, loc, newOp->getResults(), inversePerm);
    } else {
      replacements = collectReductionResults(*newOp);
    }
    if (replacements.size() != op->getNumResults())
      return failure();
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Replaces shape-only reduce-with-index init tensors with `tensor.empty` and
/// lets dialect traits clean up any additional shape-only operands.
template <typename ReduceWithIndexOpType, typename Traits>
struct NormalizeReduceWithIndexInitsAndInputsTemplate
    : public OpRewritePattern<ReduceWithIndexOpType> {
public:
  using OpRewritePattern<ReduceWithIndexOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceWithIndexOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeReduceWithIndexInitsAndInputs(op))
      return failure();

    Location loc = op.getLoc();
    bool changed = false;

    SmallVector<Value> oldInits(op.getDpsInits());
    SmallVector<Value> newInits;
    newInits.reserve(oldInits.size());
    for (Value init : oldInits) {
      if (isa_and_nonnull<tensor::EmptyOp>(init.getDefiningOp())) {
        newInits.push_back(init);
        continue;
      }

      Value empty = createStaticReductionEmptyLike(rewriter, loc, init);
      if (!empty)
        return failure();

      newInits.push_back(empty);
      changed = true;
    }

    SmallVector<Value> newInputs(op.getDpsInputs());
    if (failed(
            normalizeReduceWithIndexInputs(rewriter, loc, op, newInputs, changed)))
      return failure();

    if (!changed)
      return failure();

    Operation *newOp = Traits::createReduceWithIndexOp(rewriter, loc, op,
                                                       newInputs, newInits);
    if (isReductionInitAlreadyNormalized(*op))
      markReductionInitNormalized(*newOp);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

/// Normalizes non-i32 reduce-with-index ops by routing the internal reduction
/// through i32 indices:
///   reduce_with_index<idx_ty>(src, idx)
///     =>
///   reduce_with_index<i32>(src, cast_i32(idx))
///   cast result index back to `idx_ty`
///
/// This keeps backend index-width requirements inside the rewritten kernel.
template <typename ReduceWithIndexOpType, typename Traits>
struct NormalizeReduceIndexToI32Template
    : public OpRewritePattern<ReduceWithIndexOpType> {
public:
  using OpRewritePattern<ReduceWithIndexOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceWithIndexOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeReduceIndexToI32(op))
      return failure();

    Location loc = op.getLoc();
    IntegerType oldIndexElemType = getReductionIndexResultElementType(op);
    IntegerType i32Type = IntegerType::get(op->getContext(), 32);

    SmallVector<Value> newInputs = castReductionIndexOperandsToType<Traits>(
        rewriter, loc, op.getDpsInputs(), oldIndexElemType, i32Type);
    SmallVector<Value> newInits = castReductionIndexOperandsToType<Traits>(
        rewriter, loc, op.getDpsInits(), oldIndexElemType, i32Type);

    Operation *newOp = Traits::createReduceWithIndexOp(rewriter, loc, op,
                                                       newInputs, newInits);
    if (isReductionInitAlreadyNormalized(*op))
      markReductionInitNormalized(*newOp);
    SmallVector<Value> replacements = buildReduceIndexReplacementValues<Traits>(
        rewriter, loc, op, *newOp, oldIndexElemType);
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_REGBASE_NORMALIZE_NORMALIZEREDUCTIONTEMPLATE_H

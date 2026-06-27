//===-------- NormalizeTypeConversionTemplate.h ---------------------------===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZETYPECONVERSIONTEMPLATE_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZETYPECONVERSIONTEMPLATE_H

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/Utils/Kinds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include <cstdint>
#include <optional>

namespace mlir {

/// Casts every f16-shaped operand/result in `values` into f32 and leaves the
/// rest untouched. Shared f16 type-conversion templates use this helper to
/// build the temporary f32 compute domain before rebuilding the original op.
template <typename Traits>
static SmallVector<Value> normalizeF16ValuesToF32(PatternRewriter &rewriter,
                                                   Location loc,
                                                   ValueRange values) {
  SmallVector<Value> result;
  result.reserve(values.size());
  for (Value value : values) {
    Type elemType = getElementTypeOrSelf(value.getType());
    if (!elemType.isF16()) {
      result.push_back(value);
      continue;
    }

    result.push_back(Traits::createCastOp(rewriter, loc, value,
                                           rewriter.getF32Type(),
                                           CastRoundKind::Default));
  }
  return result;
}

/// Normalizes the bool-select special case where the selected values are `i1`.
///
/// Example:
///   %one = const dense<1> : tensor<...xi16>
///   %zero = const dense<0> : tensor<...xi16>
///   %tmp = select %cond, %one, %zero : tensor<...xi16>
///   %res = cmp ne %tmp, %zero : tensor<...xi1>
///
template <typename SelectOpType, typename Traits>
struct NormalizeToTargetTypeI1SelectTemplate
    : public OpRewritePattern<SelectOpType>, private Traits {
public:
  using OpRewritePattern<SelectOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: match the select contract and collect semantic operands/results.
    if (!Traits::shouldNormalize(op))
      return failure();

    Location loc = op.getLoc();
    SmallVector<Value> inputs = Traits::getInputs(op);
    SmallVector<Value> outputs = Traits::getOutputs(op);
    if (inputs.size() != 3 || outputs.size() != 1)
      return failure();

    Value cond = inputs[0];
    Value trueValue = inputs[1];
    Value falseValue = inputs[2];
    Value output = outputs[0];

    // Step 2: turn each selected i1 tensor into an i16 tensor with the same
    // shape, where false becomes 0 and true becomes 1. The real select
    // condition itself still stays in i1.
    auto convertI1ValueToI16 = [&](Value value) -> Value {
      Type elemType = getElementTypeOrSelf(value.getType());
      if (!elemType.isInteger(1))
        return value;

      Value one = createI16DenseTensorLike(rewriter, loc, value, 1);
      Value zero = createI16DenseTensorLike(rewriter, loc, value, 0);
      Value empty = utils::createEmptyOpWithTargetElemType(
          rewriter, loc, value, rewriter.getI16Type());
      return Traits::createSelectOp(rewriter, loc, value, one, zero, empty);
    };

    Value newTrueValue = convertI1ValueToI16(trueValue);
    Value newFalseValue = convertI1ValueToI16(falseValue);
    Value newOutput = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, output, rewriter.getI16Type());
    Value selectResult = Traits::createSelectOp(rewriter, loc, cond,
                                                newTrueValue, newFalseValue,
                                                newOutput);

    // Step 3: convert the widened select result back to bool through `!= 0`.
    Value zero = createI16DenseTensorLike(rewriter, loc, output, 0);
    Value cmpResult = Traits::createCmpOp(rewriter, loc, selectResult, zero,
                                          CompareKind::NE);
    rewriter.replaceOp(op, cmpResult);
    return success();
  }

private:
  /// Creates an i16 dense tensor constant that matches `likeValue`'s shape.
  static Value createI16DenseTensorLike(PatternRewriter &rewriter, Location loc,
                                        Value likeValue, int64_t fillValue) {
    auto shapedType =
        dyn_cast<RankedTensorType>(cast<ShapedType>(likeValue.getType()));
    auto targetType = RankedTensorType::get(shapedType.getShape(),
                                            rewriter.getI16Type());
    auto attr = DenseElementsAttr::get(targetType,
                                       rewriter.getI16IntegerAttr(fillValue));
    return rewriter.create<arith::ConstantOp>(loc, targetType, attr);
  }
};

/// Normalizes DPS ops that compute in f16 by rebuilding them in f32 and then
/// casting the final result back to the original destination element type.
///
/// Example:
///   %in_f32 = cast %in_f16 : tensor<...xf16> -> tensor<...xf32>
///   %out_f32 = cast %out_f16 : tensor<...xf16> -> tensor<...xf32>
///   %tmp = same-op(%in_f32, %out_f32)
///   %res = cast %tmp : tensor<...xf32> -> tensor<...xf16>
template <typename OpType, typename Traits>
struct NormalizeF16ToF32Type : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: match a DPS op whose inputs/results still contain f16 lanes.
    if (!Traits::shouldNormalize(op))
      return failure();

    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp)
      return failure();

    auto inputs = dpsOp.getDpsInputs();
    if (!llvm::any_of(inputs, [](Value input) {
          return getElementTypeOrSelf(input.getType()).isF16();
        })) {
      return failure();
    }

    // Step 2: cast every f16 input/init to f32 and rebuild the same op in f32.
    Location loc = op.getLoc();
    SmallVector<Value> newInputs =
        mlir::normalizeF16ValuesToF32<Traits>(rewriter, loc, inputs);
    auto originalInits = dpsOp.getDpsInits();
    SmallVector<Value> newInits =
        mlir::normalizeF16ValuesToF32<Traits>(rewriter, loc, originalInits);
    Value result = Traits::rebuildOpInF32(rewriter, loc, op, newInputs,
                                          newInits);

    // Step 3: cast the rebuilt result back to the original destination type.
    Value castBack = Traits::createCastOp(
        rewriter, loc, result,
        getElementTypeOrSelf(originalInits[0].getType()),
        CastRoundKind::Default);
    rewriter.replaceOp(op, castBack);
    return success();
  }
};

/// Normalizes bool/i8 ops that share the generic
/// cast-up / rebuild-same-op / cast-back skeleton.
///
/// The exact widened element type is decided by `Traits::getTargetType`.
/// Typical cases are:
/// - `i1 -> f16`, rebuild, then compare/cast back to `i1`
/// - `i8 -> f16/f32`, rebuild, then cast back to `i8`
template <typename ElemType, typename OpType, typename Traits>
struct NormalizeToTargetTemplate : public OpRewritePattern<OpType>,
                                   private Traits {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: match the op and collect only the semantic inputs/outputs that
    // participate in the type-conversion rewrite.
    if (!Traits::shouldNormalize(op))
      return failure();

    SmallVector<Value> inputs = Traits::getInputs(op);
    SmallVector<Value> outputs = Traits::getOutputs(op);
    if (!hasAnyNormalizedElemType(inputs) &&
        !hasAnyNormalizedElemType(outputs)) {
      return failure();
    }

    Type targetType = Traits::getTargetType(rewriter, op);
    if (!targetType)
      return failure();

    // Step 2: widen every normalized operand/result slot to the chosen target
    // type and rebuild the original op in that compute domain.
    CastSignKind intKind = Traits::getCastSignKind(op);
    Location loc = op.getLoc();
    SmallVector<Value> newInputs =
        normalizeValuesToTargetType(rewriter, loc, inputs, targetType, intKind);
    SmallVector<Value> newOutputs = normalizeValuesToTargetType(
        rewriter, loc, outputs, targetType, intKind);
    Operation *newOp = Traits::rebuildOpInTargetType(rewriter, loc, op,
                                                     newInputs, newOutputs);

    // Step 3: restore the original result contract through the dialect-local
    // cast-back policy.
    Traits::replaceResults(rewriter, op, newOp->getResults(), intKind);
    return success();
  }

private:
  static bool isNormalizedElemType(Type elemType) {
    if constexpr (std::is_same_v<ElemType, bool>)
      return elemType.isInteger(1);
    if constexpr (std::is_same_v<ElemType, int8_t>)
      return elemType.isInteger(8);
    return false;
  }

  /// Returns whether any semantic operand/result still uses the source element
  /// family handled by this template (`i1` for `bool`, `i8` for `int8_t`).
  static bool hasAnyNormalizedElemType(ValueRange values) {
    return llvm::any_of(values, [](Value value) {
      return isNormalizedElemType(getElementTypeOrSelf(value.getType()));
    });
  }

  /// Casts only the normalized operand/result slots to `targetType` and leaves
  /// all other semantic operands untouched.
  static SmallVector<Value> normalizeValuesToTargetType(
      PatternRewriter &rewriter, Location loc, ValueRange values,
      Type targetType, CastSignKind intKind = CastSignKind::Signed) {
    SmallVector<Value> result;
    result.reserve(values.size());
    for (Value value : values) {
      Type elemType = getElementTypeOrSelf(value.getType());
      if (!isNormalizedElemType(elemType)) {
        result.push_back(value);
        continue;
      }

      result.push_back(Traits::createCastOp(rewriter, loc, value, targetType,
                                            CastRoundKind::Default, Value(),
                                            intKind));
    }
    return result;
  }
};

/// Normalizes cumulative ops that still compute on f16 tensors by temporarily
/// rebuilding the cumulative op in f32 and then casting the result back.
template <typename CumOpType, typename Traits>
struct NormalizeCumOpF16ToF32Type : public OpRewritePattern<CumOpType>,
                                    private Traits {
public:
  using OpRewritePattern<CumOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CumOpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: match a cumulative op whose semantic input/output still contain
    // f16 lanes.
    if (!Traits::shouldNormalize(op))
      return failure();

    SmallVector<Value> inputs = {Traits::getInput(op)};
    SmallVector<Value> outputs = {Traits::getOutput(op)};
    if (!llvm::any_of(inputs, [](Value input) {
          return getElementTypeOrSelf(input.getType()).isF16();
        }) &&
        !llvm::any_of(outputs, [](Value output) {
          return getElementTypeOrSelf(output.getType()).isF16();
        })) {
      return failure();
    }

    // Step 2: cast the semantic input/output pair to f32 and rebuild the same
    // cumulative op in f32.
    Location loc = op.getLoc();
    auto newInputs =
        mlir::normalizeF16ValuesToF32<Traits>(rewriter, loc, inputs);
    auto newOutputs =
        mlir::normalizeF16ValuesToF32<Traits>(rewriter, loc, outputs);
    Value result = Traits::rebuildOpInF32(rewriter, loc, op, newInputs[0],
                                          newOutputs[0]);

    // Step 3: cast the cumulative result back to the original output type.
    Value castBack = Traits::createCastOp(
        rewriter, loc, result, getElementTypeOrSelf(outputs[0].getType()),
        CastRoundKind::Default);
    rewriter.replaceOp(op, castBack);
    return success();
  }
};

/// Normalizes cumulative ops whose semantic input/output are `i8` by promoting
/// them into an f32 compute domain and then casting the cumulative result back
/// to the original output type.
template <typename CumOpType, typename Traits>
struct NormalizeCumOpI8ToTargetType : public OpRewritePattern<CumOpType>,
                                      private Traits {
public:
  using OpRewritePattern<CumOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CumOpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: match a cumulative op whose semantic input/output still contain
    // i8 lanes.
    if (!Traits::shouldNormalize(op))
      return failure();

    SmallVector<Value> inputs = {Traits::getInput(op)};
    SmallVector<Value> outputs = {Traits::getOutput(op)};
    if (!llvm::any_of(inputs, [](Value input) {
          return getElementTypeOrSelf(input.getType()).isInteger(8);
        }) &&
        !llvm::any_of(outputs, [](Value output) {
          return getElementTypeOrSelf(output.getType()).isInteger(8);
        })) {
      return failure();
    }

    // Step 2: cast the semantic input/output pair into f32 and rebuild the
    // same cumulative op in that widened domain.
    Location loc = op.getLoc();
    auto newInputs = normalizeI8ValuesToF32(rewriter, loc, inputs);
    auto newOutputs =
        normalizeI8ValuesToF32(rewriter, loc, outputs);
    Value result = Traits::rebuildOpInF32(rewriter, loc, op, newInputs[0],
                                          newOutputs[0]);

    // Step 3: cast the cumulative result back to the original output type.
    Value castBack = Traits::createCastOp(
        rewriter, loc, result, getElementTypeOrSelf(outputs[0].getType()),
        CastRoundKind::Default);
    rewriter.replaceOp(op, castBack);
    return success();
  }

private:
  /// Casts only the semantic `i8` input/output slots to f32 and leaves all
  /// other semantic values untouched.
  static SmallVector<Value> normalizeI8ValuesToF32(PatternRewriter &rewriter,
                                                   Location loc,
                                                   ValueRange values) {
    SmallVector<Value> result;
    result.reserve(values.size());
    for (Value value : values) {
      Type elemType = getElementTypeOrSelf(value.getType());
      if (!elemType.isInteger(8)) {
        result.push_back(value);
        continue;
      }

      result.push_back(Traits::createCastOp(rewriter, loc, value,
                                            rewriter.getF32Type(),
                                            CastRoundKind::Default));
    }
    return result;
  }
};

/// Canonicalizes bool reductions that already stay in bool element type by
/// rewriting only the reduce combiner into the equivalent logical reduction
/// kind. No cast-up / cast-back is inserted on this path.
///
/// Example:
///   reduce(add i1) -> reduce(or)
///   reduce(mul i1) -> reduce(and)
enum class ReduceBoolLogicalKind { Or, And };

template <typename ReduceOpType, typename Traits>
struct NormalizeReduceBoolToLogicalTemplate
    : public OpRewritePattern<ReduceOpType>, private Traits {
public:
  using OpRewritePattern<ReduceOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: match a bool reduction whose combiner can be expressed as a
    // logical reduction.
    if (!Traits::shouldNormalize(op))
      return failure();

    // Step 2: map the source reduction semantics to `or` / `and`.
    std::optional<ReduceBoolLogicalKind> kind =
        Traits::getLogicalReduceKind(op);
    if (!kind)
      return failure();

    // Step 3: rebuild or mutate the reduction into the logical form.
    return Traits::rewriteToLogicalReduce(rewriter, op, *kind);
  }
};

/// Normalizes the special bool-reduce-add case into:
/// 1. select `{1, 0}` in i16,
/// 2. reduce with `max`,
/// 3. compare the reduced value against zero.
///
/// Example:
///   reduce(add i1) -> select(i16 1, i16 0) -> reduce(max i16) -> cmp ne 0
template <typename ReduceOpType, typename Traits>
struct ReduceI1AddToSelectMaxCompareTemplate
    : public OpRewritePattern<ReduceOpType> {
public:
  using OpRewritePattern<ReduceOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: match the special bool-reduce-add shape.
    if (!Traits::shouldNormalize(op))
      return failure();

    // Step 2: materialize the bool input as an i16 {1, 0} select and reduce
    // the selected values with `max`.
    Location loc = op.getLoc();
    Type i16Type = rewriter.getI16Type();
    Value input = Traits::getInput(op);
    Value oneTensor =
        Traits::createPredicateTensor(rewriter, loc, input, i16Type, 1);
    Value zeroTensor =
        Traits::createPredicateTensor(rewriter, loc, input, i16Type, 0);
    Value selectDst =
        utils::createEmptyOpWithTargetElemType(rewriter, loc, input, i16Type);
    Value selected = Traits::createSelectOp(rewriter, loc, input, oneTensor,
                                            zeroTensor, selectDst);
    Value reduceInit = Traits::createReduceInit(rewriter, loc, op, i16Type);
    Value reduced =
        Traits::createMaxReduce(rewriter, loc, op, selected, reduceInit);

    // Step 3: recover the bool result by comparing the reduced i16 value
    // against zero.
    Value cmp = Traits::createCmpNeZero(rewriter, loc, op, reduced, i16Type);
    Traits::replaceResults(rewriter, op, cmp);
    return success();
  }
};

/// Normalizes bool reductions expressed as `and` / `or` into i16 reductions so
/// the combiner can be represented with integer min/max semantics.
///
/// Example:
///   reduce(and i1) -> cast input to i16 -> reduce(min i16) -> cast back to i1
///   reduce(or  i1) -> cast input to i16 -> reduce(max i16) -> cast back to i1
template <typename ReduceOpType, typename Traits>
struct ReduceI1AndOrToI16Template : public OpRewritePattern<ReduceOpType>,
                                    private Traits {
public:
  using OpRewritePattern<ReduceOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: match a bool reduction whose combiner is `and` / `or`.
    if (!Traits::shouldNormalize(op))
      return failure();

    // Step 2: cast the bool input to i16, rebuild the reduction with min/max,
    // and keep the original reduce dimensions.
    Location loc = op.getLoc();
    Type i16Type = rewriter.getI16Type();
    Value input = Traits::getInput(op);
    Value inputI16 = Traits::castInputToI16(rewriter, loc, input, i16Type);
    bool isAndReduce = Traits::isAndReduce(op);
    Value reduceInit = Traits::createReduceInit(rewriter, loc, op, i16Type,
                                                isAndReduce ? -1 : 0);
    Value reduced = Traits::createReducedOp(rewriter, loc, op, inputI16,
                                            reduceInit, isAndReduce);
    if (!reduced)
      return failure();

    // Step 3: cast the reduced i16 result back to i1.
    Value castBack = Traits::castResultToI1(rewriter, loc, reduced);
    Traits::replaceResults(rewriter, op, castBack);
    return success();
  }
};

/// Normalizes the Ascend 910/95 reduce special case by widening semantic `i8`
/// lanes to `i16`, rebuilding the same reduce, and then casting the result
/// back with the original signed/unsigned policy.
template <typename ReduceOpType, typename Traits>
struct ReduceNormalize910_95Template : public OpRewritePattern<ReduceOpType>,
                                       private Traits {
public:
  using OpRewritePattern<ReduceOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: match the target-specific i8-reduce case.
    if (!Traits::shouldNormalize(op))
      return failure();

    // Step 2: widen every semantic i8 input/output slot to i16 and rebuild
    // the same reduction in the widened domain.
    Location loc = op.getLoc();
    CastSignKind intKind = Traits::getCastSignKind(op);
    SmallVector<Value> newInputs =
        normalizeI8ValuesToI16(rewriter, loc, Traits::getInputs(op), intKind);
    SmallVector<Value> newOutputs =
        normalizeI8ValuesToI16(rewriter, loc, Traits::getOutputs(op), intKind);
    Operation *newOp = Traits::rebuildOpInI16(rewriter, loc, op, newInputs,
                                              newOutputs);

    // Step 3: restore the original result contract with the dialect-local
    // cast-back policy.
    Traits::replaceResults(rewriter, op, newOp->getResults(), intKind);
    return success();
  }

private:
  /// Casts only the semantic `i8` lanes to i16 and leaves all other semantic
  /// values untouched.
  static SmallVector<Value> normalizeI8ValuesToI16(PatternRewriter &rewriter,
                                                   Location loc,
                                                   ValueRange values,
                                                   CastSignKind intKind) {
    SmallVector<Value> result;
    result.reserve(values.size());
    for (Value value : values) {
      Type elemType = getElementTypeOrSelf(value.getType());
      if (!elemType.isInteger(8)) {
        result.push_back(value);
        continue;
      }

      result.push_back(Traits::createI8ToI16Cast(rewriter, loc, value, intKind));
    }
    return result;
  }
};

/// Normalizes gather-like ops whose index operand is a non-i32 integer type
/// by casting the index tensor to i32.
///
/// Example:
///   gather(src, %index_i16, init) → cast %index_i16 → i32 →
///   gather(src, %index_i32, init)
template <typename GatherOpType, typename Traits>
struct NormalizeGatherIndexToI32Template
    : public OpRewritePattern<GatherOpType> {
public:
  using OpRewritePattern<GatherOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOpType op,
                                PatternRewriter &rewriter) const override {
    // Step 1: guard on pure tensor semantics (common to both dialects) and let
    // the local trait apply dialect-specific constraints.
    if (!op.hasPureTensorSemantics())
      return failure();
    if (!Traits::shouldNormalize(op))
      return failure();

    // Step 2: match a non-i32 integer index operand.
    Value index = Traits::getIndex(op);
    Type indexElemType = getElementTypeOrSelf(index.getType());
    if (!indexElemType.isInteger() || indexElemType.isInteger(32))
      return failure();

    // Step 3: cast the index tensor to i32 and rebuild the gather op.
    Value castedIndex =
        Traits::castIndexToI32(rewriter, op.getLoc(), index);
    Traits::rebuildOp(rewriter, op, castedIndex);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZETYPECONVERSIONTEMPLATE_H

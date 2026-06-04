//===-------- ReductionTemplateHelpers.h ----------------------------------===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_REDUCTIONTEMPLATEHELPERS_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_REDUCTIONTEMPLATEHELPERS_H

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/Utils/Kinds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringRef.h"

#include <type_traits>
#include <utility>

namespace mlir {

inline constexpr llvm::StringLiteral kAlreadyInitializeReductionInitAttr =
    "already_initialize_init";

inline bool isReductionInitAlreadyNormalized(Operation &op) {
  if (op.getDiscardableAttr(kAlreadyInitializeReductionInitAttr))
    return true;
  return static_cast<bool>(
      op.getAttrDictionary().get(kAlreadyInitializeReductionInitAttr));
}

inline void markReductionInitNormalized(Operation &op) {
  op.setDiscardableAttr(kAlreadyInitializeReductionInitAttr,
                        UnitAttr::get(op.getContext()));
}

template <typename Traits>
SmallVector<Value> promoteF16ReductionValuesToF32(PatternRewriter &rewriter,
                                                  ValueRange values) {
  SmallVector<Value> promoted;
  promoted.reserve(values.size());
  for (Value value : values) {
    if (getElementTypeOrSelf(value.getType()).isF16()) {
      promoted.push_back(Traits::createCastOp(rewriter, value.getLoc(), value,
                                              rewriter.getF32Type(),
                                              CastRoundKind::Default));
      continue;
    }
    promoted.push_back(value);
  }
  return promoted;
}

inline Value createStaticReductionEmptyLike(PatternRewriter &rewriter,
                                            Location loc, Value value) {
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!valueType || !valueType.hasStaticShape())
    return Value();

  return rewriter.create<tensor::EmptyOp>(loc, valueType.getShape(),
                                          valueType.getElementType());
}

template <typename ReductionOpType>
SmallVector<Value> createReductionResultEmpties(PatternRewriter &rewriter,
                                                Location loc,
                                                ReductionOpType op) {
  SmallVector<Value> newInits;
  newInits.reserve(op->getNumResults());
  for (Value result : op->getResults())
    newInits.push_back(utils::createEmptyOp(rewriter, loc, result));
  return newInits;
}

/// Casts index tensors that currently use `sourceElemType` to
/// `targetElemType`.
///
/// `tensor.empty` is rebuilt directly because only shape and element type
/// matter there.
template <typename Traits>
Value castReductionIndexTensorToType(PatternRewriter &rewriter, Location loc,
                                     Value value, IntegerType sourceElemType,
                                     IntegerType targetElemType) {
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!valueType)
    return value;

  auto elemType = dyn_cast<IntegerType>(valueType.getElementType());
  if (!elemType || elemType != sourceElemType)
    return value;

  if (auto emptyOp = value.getDefiningOp<tensor::EmptyOp>()) {
    RankedTensorType newType =
        RankedTensorType::get(valueType.getShape(), targetElemType);
    SmallVector<Value> dynamicSizes(emptyOp.getDynamicSizes().begin(),
                                    emptyOp.getDynamicSizes().end());
    return rewriter.create<tensor::EmptyOp>(loc, newType, dynamicSizes);
  }

  return Traits::castReduceIndexTensor(rewriter, loc, value, targetElemType,
                                       value);
}

template <typename Traits>
SmallVector<Value>
castReductionIndexOperandsToType(PatternRewriter &rewriter, Location loc,
                                 ValueRange values,
                                 IntegerType sourceElemType,
                                 IntegerType targetElemType) {
  SmallVector<Value> converted;
  converted.reserve(values.size());
  for (Value value : values) {
    converted.push_back(castReductionIndexTensorToType<Traits>(
        rewriter, loc, value, sourceElemType, targetElemType));
  }
  return converted;
}

template <typename ReduceWithIndexOpType>
bool hasNonI32ReductionIndexResult(ReduceWithIndexOpType op) {
  if (op->getNumResults() < 2)
    return false;

  auto indexType = dyn_cast<RankedTensorType>(op->getResult(1).getType());
  if (!indexType)
    return false;

  auto indexElemType = dyn_cast<IntegerType>(indexType.getElementType());
  return indexElemType && indexElemType.getWidth() != 32;
}

template <typename ReduceWithIndexOpType>
IntegerType getReductionIndexResultElementType(ReduceWithIndexOpType op) {
  auto indexType = cast<RankedTensorType>(op->getResult(1).getType());
  return cast<IntegerType>(indexType.getElementType());
}

/// Normalizes non-source reduce-with-index tensor inputs that only carry shape.
///
/// HFusion exposes the explicit index tensor as an input while HIVM does not.
/// Replacing every extra tensor input after the source with `tensor.empty`
/// keeps both dialects on the same normalization path without changing the
/// public reduction contract.
template <typename ReduceWithIndexOpType>
LogicalResult normalizeReduceWithIndexInputs(PatternRewriter &rewriter,
                                             Location loc,
                                             ReduceWithIndexOpType op,
                                             SmallVector<Value> &newInputs,
                                             bool &changed) {
  if (newInputs.size() != op.getDpsInputs().size())
    return failure();

  for (auto [idx, input] : llvm::enumerate(newInputs)) {
    if (idx == 0)
      continue;

    if (isa_and_nonnull<tensor::EmptyOp>(input.getDefiningOp()) ||
        isa<BlockArgument>(input))
      continue;

    Value empty = createStaticReductionEmptyLike(rewriter, loc, input);
    if (!empty)
      return failure();

    newInputs[idx] = empty;
    changed = true;
  }
  return success();
}

/// Restores the public result type after the internal index reduction ran on
/// i32:
///   (value, index:i32) -> (value, cast(index, old_index_type))
template <typename Traits, typename ReduceWithIndexOpType>
SmallVector<Value> buildReduceIndexReplacementValues(
    PatternRewriter &rewriter, Location loc, ReduceWithIndexOpType op,
    Operation &newOp, IntegerType oldIndexElemType) {
  SmallVector<Value> replacements;
  replacements.reserve(newOp.getNumResults());
  replacements.push_back(newOp.getResult(0));
  replacements.push_back(Traits::castReduceIndexTensor(
      rewriter, loc, newOp.getResult(1), oldIndexElemType, op->getResult(1)));
  return replacements;
}

template <typename Traits, typename ReductionOpType>
inline LogicalResult replacePromotedReductionResults(ReductionOpType oldOp,
                                                      Operation &newOp,
                                                      PatternRewriter &rewriter) {
  auto oldResults = oldOp->getResults();
  auto newResults = newOp.getResults();
  if (oldResults.size() != newResults.size())
    return oldOp->emitError(
        "result sizes mismatch when replacing promoted reduction results");

  SmallVector<Value> replacements;
  replacements.reserve(oldResults.size());
  for (const auto [idx, oldResult] : llvm::enumerate(oldResults)) {
    Value newResult = newResults[idx];
    Type oldElemType = getElementTypeOrSelf(oldResult.getType());
    Type newElemType = getElementTypeOrSelf(newResult.getType());
    if (oldElemType == newElemType) {
      replacements.push_back(newResult);
      continue;
    }

    replacements.push_back(Traits::createCastOp(rewriter, oldResult.getLoc(),
                                                newResult, oldElemType,
                                                CastRoundKind::Default));
  }

  rewriter.replaceOp(oldOp, replacements);
  return success();
}

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_REDUCTIONTEMPLATEHELPERS_H

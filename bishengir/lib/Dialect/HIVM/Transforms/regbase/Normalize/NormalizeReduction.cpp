//===-------- NormalizeReduction.cpp ----------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeReductionTemplate.h"

#include <numeric>

namespace mlir {

static bool isReduceWithIndexOp(hivm::VReduceOp op) {
  auto reduceOp = op.getArith().getReduceOp();
  return reduceOp == hivm::ReduceOperation::max_with_index_left ||
         reduceOp == hivm::ReduceOperation::max_with_index_right ||
         reduceOp == hivm::ReduceOperation::min_with_index_left ||
         reduceOp == hivm::ReduceOperation::min_with_index_right;
}

static Value createHIVMTransposeByAdjacentSwaps(PatternRewriter &rewriter,
                                                Location loc, Value input,
                                                Value finalInit,
                                                ArrayRef<int64_t> finalPerm) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasStaticShape() || finalPerm.empty())
    return Value();

  SmallVector<int64_t> currentOrder(inputType.getRank());
  std::iota(currentOrder.begin(), currentOrder.end(), int64_t{0});

  Value current = input;
  for (int64_t targetPos = 0, rank = inputType.getRank(); targetPos < rank;
       ++targetPos) {
    auto it = llvm::find(currentOrder, finalPerm[targetPos]);
    if (it == currentOrder.end())
      return Value();

    int64_t pos = static_cast<int64_t>(std::distance(currentOrder.begin(), it));
    while (pos > targetPos) {
      SmallVector<int64_t> swapPerm(rank);
      std::iota(swapPerm.begin(), swapPerm.end(), int64_t{0});
      std::swap(swapPerm[pos - 1], swapPerm[pos]);

      SmallVector<int64_t> nextOrder = currentOrder;
      std::swap(nextOrder[pos - 1], nextOrder[pos]);
      bool isFinalSwap = llvm::equal(nextOrder, finalPerm);
      Value stepInit =
          isFinalSwap ? finalInit
                      : createReductionTransposeInit(rewriter, loc, current,
                                                     swapPerm);
      if (!stepInit)
        return Value();

      current = rewriter
                    .create<hivm::VTransposeOp>(
                        loc, TypeRange{stepInit.getType()}, current, stepInit,
                        rewriter.getDenseI64ArrayAttr(swapPerm))
                    .getResult()
                    .front();
      std::swap(currentOrder[pos - 1], currentOrder[pos]);
      --pos;
    }
  }
  return current;
}

/// HIVM hooks for the shared argmin/argmax normalization template.
struct HIVMNormalizeArgMinMaxTraits : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeArgMinMax(hivm::VReduceOp op) {
    auto reduceKind = op.getArith().getReduceOp();
    return op.hasPureTensorSemantics() &&
           !isReductionInitAlreadyNormalized(*op) &&
           (reduceKind == hivm::ReduceOperation::max_with_index_left ||
            reduceKind == hivm::ReduceOperation::max_with_index_right ||
            reduceKind == hivm::ReduceOperation::min_with_index_left ||
            reduceKind == hivm::ReduceOperation::min_with_index_right) &&
           isa<FloatType>(getElementTypeOrSelf(op.getSrc().getType()));
  }

  static bool isMinReduction(hivm::VReduceOp op) {
    auto reduceOp = op.getArith().getReduceOp();
    return reduceOp == hivm::ReduceOperation::min_with_index_left ||
           reduceOp == hivm::ReduceOperation::min_with_index_right;
  }

  /// Builds an NaN mask with HIVM's native `visnan` op so the reduction
  /// normalization stays aligned with HFusion's explicit `isnan` path.
  static Value createIsNanMask(PatternRewriter &rewriter, Location loc,
                               Value src) {
    Value srcMask = utils::createEmptyOpWithTargetElemType(rewriter, loc, src,
                                                           rewriter.getI1Type());
    return rewriter
        .create<hivm::VIsNanOp>(loc, TypeRange{srcMask.getType()},
                                ValueRange{src}, ValueRange{srcMask},
                                /*transpose=*/ArrayRef<int64_t>{},
                                /*broadcast=*/ArrayRef<int64_t>{})
        .getResult()
        .front();
  }
};

/// HIVM hooks for promoting f16 reduce-sum to f32 accumulation.
struct HIVMNormalizeF16ReduceSumTraits : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeF16ReduceSum(hivm::VReduceOp op) {
    return op.hasPureTensorSemantics() &&
           op.getArith().getReduceOp() == hivm::ReduceOperation::sum &&
           hasF16ReduceSrcOrInit(op);
  }

  static Operation *createPromotedReduceOp(hivm::VReduceOp op,
                                           PatternRewriter &rewriter,
                                           SmallVector<Value> &newInputs,
                                           SmallVector<Value> &newInits) {
    return NormalizeTraitsBase::createReduceWithIndexOp(rewriter, op.getLoc(),
                                                        op, newInputs,
                                                        newInits);
  }

private:
  static bool hasF16ReduceSrcOrInit(hivm::VReduceOp op) {
    return getElementTypeOrSelf(op.getSrc().getType()).isF16() ||
          llvm::any_of(op.getDst(), [](Value dst) {
            return getElementTypeOrSelf(dst.getType()).isF16();
          });
  }
};

/// HIVM-specific hooks for the transpose-based RA high-performance rewrite.
struct HIVMReduceWithIndexRAHighPerformanceTraits
    : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeRAHighPerformance(hivm::VReduceOp op) {
    return op.hasPureTensorSemantics() && isReduceWithIndexOp(op);
  }

  static ArrayRef<int64_t> getRAHighPerformanceReduceDims(hivm::VReduceOp op) {
    return op.getReduceDims();
  }

  static Value createRAHighPerformanceTransposeOp(PatternRewriter &rewriter,
                                                  Location loc, Value input,
                                                  Value init,
                                                  ArrayRef<int64_t> perm) {
    return createHIVMTransposeByAdjacentSwaps(rewriter, loc, input, init, perm);
  }

  static Operation *
  createRAHighPerformanceReduceOp(PatternRewriter &rewriter, Location loc,
                                  hivm::VReduceOp op, ArrayRef<Value> newInputs,
                                  ArrayRef<Value> newInits,
                                  ArrayRef<int64_t> newReduceDims,
                                  ArrayRef<int64_t> transposePerm) {
    SmallVector<Value> transposedInits =
        transposeReductionValueRange<HIVMReduceWithIndexRAHighPerformanceTraits>(
            rewriter, loc, newInits, transposePerm);
    if (transposedInits.size() != newInits.size())
      return nullptr;

    return NormalizeTraitsBase::createReduceOp(rewriter, loc, op,
                                               newInputs.front(), transposedInits,
                                               rewriter.getDenseI64ArrayAttr(
                                                   newReduceDims));
  }

};

struct HIVMNormalizeReduceWithIndexInitsAndInputsTraits
    : public hivm::NormalizeTraitsBase {
public:
  /// For HIVM reduce-with-index, destination tensors are shape carriers. This
  /// matches the HFusion pre-conversion normalization that drops filled init
  /// tensors in favor of `tensor.empty`.
  static bool shouldNormalizeReduceWithIndexInitsAndInputs(hivm::VReduceOp op) {
    return op.hasPureTensorSemantics() && isReduceWithIndexOp(op);
  }

};

struct HIVMNormalizeReduceIndexToI32Traits : public hivm::NormalizeTraitsBase {
public:
  /// HIVM reduce-with-index verifies only i32 destination indices, so rebuild
  /// the reduction on i32 and cast the public result index back afterward.
  static bool shouldNormalizeReduceIndexToI32(hivm::VReduceOp op) {
    return op.hasPureTensorSemantics() && isReduceWithIndexOp(op) &&
           hasNonI32ReductionIndexResult(op);
  }
};

using NormalizeArgMinMaxOp =
    NormalizeArgMinMaxOpTemplate<hivm::VReduceOp, HIVMNormalizeArgMinMaxTraits>;
using NormalizeF16ReduceSum =
    NormalizeF16ReduceSumTemplate<hivm::VReduceOp,
                                  HIVMNormalizeF16ReduceSumTraits>;
using ReduceWithIndexRAHighPerformance =
    ReduceWithIndexRAHighPerformanceTemplate<
        hivm::VReduceOp, HIVMReduceWithIndexRAHighPerformanceTraits>;
using NormalizeReduceWithIndexInitsAndInputs =
    NormalizeReduceWithIndexInitsAndInputsTemplate<
        hivm::VReduceOp, HIVMNormalizeReduceWithIndexInitsAndInputsTraits>;
using NormalizeReduceIndexToI32 =
    NormalizeReduceIndexToI32Template<hivm::VReduceOp,
                                      HIVMNormalizeReduceIndexToI32Traits>;

} // namespace mlir

void mlir::hivm::populateNormalizePreReductionPatterns(
    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  if (!NormalizeTraitsBase::archIsRegbased()) {
    patterns.add<ReduceWithIndexRAHighPerformance>(ctx);
  }
  if (NormalizeTraitsBase::archIsRegbased()) {
    patterns.add<NormalizeReduceWithIndexInitsAndInputs>(ctx);
    patterns.add<NormalizeReduceIndexToI32>(ctx);
  }
}

void mlir::hivm::populateNormalizeReductionPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<NormalizeF16ReduceSum>(ctx);
  patterns.add<ReduceWithIndexRAHighPerformance>(ctx);
}

void mlir::hivm::populateNormalizeFinalReductionPatterns(
    RewritePatternSet &patterns) {
  if (NormalizeTraitsBase::archIsRegbased())
    patterns.add<NormalizeArgMinMaxOp>(patterns.getContext());
}

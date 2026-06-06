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
#include "bishengir/Transforms/Normalize/NormalizeReductionTemplate.h"

namespace mlir {

static bool isReduceWithIndexOp(hivm::VReduceOp op) {
  return utils::isReduceWithIndex(op.getArith().getReduceOp());
}

/// HIVM hooks for the shared argmin/argmax normalization template.
struct HIVMNormalizeArgMinMaxTraits : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeArgMinMax(hivm::VReduceOp op) {
    auto reduceKind = op.getArith().getReduceOp();
    return op.hasPureTensorSemantics() &&
           !isReductionInitAlreadyNormalized(*op) &&
           (reduceKind == hivm::ReduceOperation::max_with_index ||
            reduceKind == hivm::ReduceOperation::min_with_index) &&
           isa<FloatType>(getElementTypeOrSelf(op.getSrc().getType()));
  }

  static bool isMinReduction(hivm::VReduceOp op) {
    return op.getArith().getReduceOp() == hivm::ReduceOperation::min_with_index;
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
using NormalizeReduceWithIndexInitsAndInputs =
    NormalizeReduceWithIndexInitsAndInputsTemplate<
        hivm::VReduceOp, HIVMNormalizeReduceWithIndexInitsAndInputsTraits>;
using NormalizeReduceIndexToI32 =
    NormalizeReduceIndexToI32Template<hivm::VReduceOp,
                                      HIVMNormalizeReduceIndexToI32Traits>;

} // namespace mlir

void mlir::hivm::populateNormalizeReductionPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<NormalizeArgMinMaxOp>(ctx);
  patterns.add<NormalizeF16ReduceSum>(ctx);
}

void mlir::hivm::populateNormalizeFinalReductionPatterns(
    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<NormalizeReduceWithIndexInitsAndInputs>(ctx);
  patterns.add<NormalizeReduceIndexToI32>(ctx);
}

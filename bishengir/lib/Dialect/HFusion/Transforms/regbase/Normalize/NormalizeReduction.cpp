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

#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeUtils.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeReductionTemplate.h"

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

/// HFusion-specific hooks for the transpose-based RA high-performance rewrite.
struct HFusionReduceWithIndexRAHighPerformanceTraits
    : public hfusion::NormalizeTraitsBase {
public:
  static bool shouldNormalizeRAHighPerformance(hfusion::ReduceWithIndexOp op) {
    return true;
  }

  static ArrayRef<int64_t>
  getRAHighPerformanceReduceDims(hfusion::ReduceWithIndexOp op) {
    return op.getDimensions();
  }

  static Value createRAHighPerformanceTransposeOp(PatternRewriter &rewriter,
                                                  Location loc, Value input,
                                                  Value init,
                                                  ArrayRef<int64_t> perm) {
    return rewriter.create<linalg::TransposeOp>(loc, input, init, perm)
        ->getResult(0);
  }

  static Operation *
  createRAHighPerformanceReduceOp(PatternRewriter &rewriter, Location loc,
                                  hfusion::ReduceWithIndexOp op,
                                  ArrayRef<Value> newInputs,
                                  ArrayRef<Value> newInits,
                                  ArrayRef<int64_t> newReduceDims,
                                  ArrayRef<int64_t>) {
    return rewriter.create<hfusion::ReduceWithIndexOp>(
        loc, op->getResultTypes(), newInputs, newInits, op.getReduceKindAttr(),
        op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(), newReduceDims);
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

using NormalizeArgMinMaxOpRegBase = NormalizeArgMinMaxOpTemplate<
    hfusion::ReduceWithIndexOp, HFusionNormalizeArgMinMaxTraits>;
using NormalizeF16ReduceSumRegBase =
    NormalizeF16ReduceSumTemplate<linalg::ReduceOp,
                                  HFusionNormalizeF16ReduceSumTraits>;
using ReduceWithIndexRAHighPerformance = ReduceWithIndexRAHighPerformanceTemplate<
    hfusion::ReduceWithIndexOp, HFusionReduceWithIndexRAHighPerformanceTraits>;
using NormalizeReduceWithIndexInitsAndInputsRegBase =
    NormalizeReduceWithIndexInitsAndInputsTemplate<
        hfusion::ReduceWithIndexOp,
        HFusionNormalizeReduceWithIndexInitsAndInputsTraits>;
using NormalizeReduceIndexToI32RegBase =
    NormalizeReduceIndexToI32Template<hfusion::ReduceWithIndexOp,
                                      HFusionNormalizeReduceIndexToI32Traits>;

} // namespace mlir

namespace mlir::hfusion {
struct NormalizeI8TransposeRegBase : public OpRewritePattern<linalg::TransposeOp> {
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

void populateNormalizePreReductionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (!archIsRegbased) {
    patterns.add<ReduceWithIndexRAHighPerformance>(ctx);
  }
  if (archIsRegbased) {
    patterns.add<NormalizeReduceWithIndexInitsAndInputsRegBase>(ctx);
    patterns.add<NormalizeReduceIndexToI32RegBase>(ctx);
  }
}

void populateNormalizeReductionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeF16ReduceSumRegBase>(ctx);
  patterns.add<ReduceWithIndexRAHighPerformance>(ctx);
}

void populateNormalizeFinalReductionPatterns(RewritePatternSet &patterns) {
  if (archIsRegbased)
    patterns.add<NormalizeArgMinMaxOpRegBase>(patterns.getContext());
}
} // namespace mlir::hfusion

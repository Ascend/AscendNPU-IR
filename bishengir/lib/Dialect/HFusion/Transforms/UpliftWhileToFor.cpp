//===- UpliftWhileToFor.cpp - Uplift scf.while to scf.for ----------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
//
// Thin downstream wrapper around upstream
// `mlir::scf::populateUpliftWhileToForPatterns`, exposed as the
// `hfusion-uplift-while-to-for` pass and wired into the head of HFusion
// preProcess (HFusionPipelines.cpp).
//
// The upstream pattern only fires when the `scf.while` matches a
// canonical for-shape:
//   * `before` block: a single `arith.cmpi` feeding `scf.condition`
//   * `after`  block: a linear `arith.addi` on the induction variable
// All other while-loops (e.g. data-driven exit conditions) are left
// untouched. Downstream HIVM multi-buffer can take the simpler scf.for
// counter path whenever this pass succeeds; structurally-while loops
// still fall back to the alloca-based counter machinery in
// MultiBufferLoopAdapter.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/ShapeRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_UPLIFTWHILETOFOR
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

/// Rewrite the generated `sext(iv) < min(wide, sext(cap))` condition to the
/// induction variable's type so the upstream while-to-for pattern can match.
/// The minimum proves the upper edge is representable; saturating the lower
/// edge preserves the empty-loop result for bounds below the narrow range.
struct NormalizeSignExtendedWhileCondition
    : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp loop,
                                PatternRewriter &rewriter) const override {
    Block *before = loop.getBeforeBody();
    if (!llvm::hasNItems(before->without_terminator(), 2))
      return failure();

    auto condition = loop.getConditionOp();
    auto cmp = condition.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
        !cmp->hasOneUse() || cmp->getBlock() != before)
      return failure();

    auto ivExt = cmp.getLhs().getDefiningOp<arith::ExtSIOp>();
    auto iv = ivExt ? dyn_cast<BlockArgument>(ivExt.getIn()) : BlockArgument();
    auto boundMin = cmp.getRhs().getDefiningOp<arith::MinSIOp>();
    if (!ivExt || !ivExt->hasOneUse() || ivExt->getBlock() != before || !iv ||
        iv.getOwner() != before || !boundMin)
      return failure();

    auto narrowType = dyn_cast<IntegerType>(iv.getType());
    auto wideType = dyn_cast<IntegerType>(ivExt.getType());
    if (!narrowType || !wideType ||
        narrowType.getWidth() >= wideType.getWidth())
      return failure();

    auto isNarrowCap = [&](Value value) {
      auto capExt = value.getDefiningOp<arith::ExtSIOp>();
      return capExt && capExt.getIn().getType() == narrowType;
    };
    if (!isNarrowCap(boundMin.getLhs()) && !isNarrowCap(boundMin.getRhs()))
      return failure();

    rewriter.setInsertionPoint(loop);
    llvm::APInt signedMin =
        llvm::APInt::getSignedMinValue(narrowType.getWidth())
            .sext(wideType.getWidth());
    Value min = rewriter.create<arith::ConstantOp>(
        cmp.getLoc(), rewriter.getIntegerAttr(wideType, signedMin));
    Value saturated =
        rewriter.create<arith::MaxSIOp>(cmp.getLoc(), cmp.getRhs(), min);
    Value narrowBound =
        rewriter.create<arith::TruncIOp>(cmp.getLoc(), narrowType, saturated);

    rewriter.setInsertionPoint(cmp);
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(cmp, arith::CmpIPredicate::slt,
                                               iv, narrowBound);
    rewriter.eraseOp(ivExt);
    return success();
  }
};

struct UpliftWhileToForPass
    : public impl::UpliftWhileToForBase<UpliftWhileToForPass> {
  using Base = impl::UpliftWhileToForBase<UpliftWhileToForPass>;
  using Base::Base;
  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // 3160 sign-extend normalize: whitelist / bypass only. HFusion runs
    // before HIVM compile flags are threaded, so a registered name is
    // enough at compile time.
    if (this->bypassShapeRegistry ||
        hivm::isLoopShapeRegistered(funcOp.getName()))
      patterns.add<NormalizeSignExtendedWhileCondition>(ctx);
    scf::populateUpliftWhileToForPatterns(patterns);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createUpliftWhileToForPass(
    const UpliftWhileToForOptions &options) {
  return std::make_unique<UpliftWhileToForPass>(options);
}

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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
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

/// Peel scalar constructors so filled tensors and dense constants can be
/// matched as the same induction variable or step.
static Value unwrapSingleton(Value value) {
  if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    Value scalar = fill.getInputs().front();
    if (scalar.getType() ==
        cast<RankedTensorType>(value.getType()).getElementType())
      return scalar;
  }
  if (auto elements = value.getDefiningOp<tensor::FromElementsOp>())
    return elements.getElements().front();
  return value;
}

/// Keep a unit-step loop counter in scalar SSA. A tensor-backed predicate
/// otherwise makes Cube read the vector counter from L1 after core splitting.
struct ScalarizeWhileInductionVariable : OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto condition = loop.getConditionOp();
    auto cmp = condition.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
        condition.getArgs() != ValueRange(loop.getBeforeArguments()))
      return failure();

    auto extract = cmp.getLhs().getDefiningOp<tensor::ExtractOp>();
    if (!extract || !extract.getType().isIntOrIndex() ||
        extract.getType().isInteger(1))
      return failure();
    auto iv = dyn_cast<BlockArgument>(extract.getTensor());
    auto type = extract.getTensor().getType();
    if (!iv || iv.getOwner() != loop.getBeforeBody() || type.getEncoding() ||
        !type.hasStaticShape() || type.getNumElements() != 1 ||
        !llvm::hasNItems(iv.getUses(), 2) ||
        !llvm::all_of(extract.getIndices(),
                      [](Value i) { return matchPattern(i, m_Zero()); }))
      return failure();

    unsigned index = iv.getArgNumber();
    BlockArgument bodyIV = loop.getAfterArguments()[index];
    auto add =
        loop.getYieldOp().getOperand(index).getDefiningOp<arith::AddIOp>();
    if (!add || !bodyIV.hasOneUse() || *bodyIV.user_begin() != add ||
        !add->hasOneUse() || !loop.getResult(index).use_empty())
      return failure();
    Value step = add.getLhs() == bodyIV ? add.getRhs() : add.getLhs();
    // A signed IV below its bound can always advance by one without overflow.
    if (!matchPattern(unwrapSingleton(step), m_One()))
      return failure();

    Value initial = unwrapSingleton(loop.getInits()[index]);
    Type scalarType = extract.getType();
    if (isa<ShapedType>(initial.getType())) {
      DenseIntElementsAttr constant;
      if (!matchPattern(initial, m_Constant(&constant)))
        return failure();
      initial = rewriter.create<arith::ConstantOp>(
          loop.getLoc(), constant.getSplatValue<IntegerAttr>());
    }
    Value one = rewriter.create<arith::ConstantOp>(
        add.getLoc(), rewriter.getIntegerAttr(scalarType, 1));
    rewriter.modifyOpInPlace(loop, [&] {
      loop.getInitsMutable()[index].set(initial);
      iv.setType(scalarType);
      bodyIV.setType(scalarType);
      loop.getResult(index).setType(scalarType);
    });
    rewriter.replaceOp(extract, iv);
    rewriter.modifyOpInPlace(add, [&] {
      add->setOperands({bodyIV, one});
      add.getResult().setType(scalarType);
    });
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
    patterns.add<ScalarizeWhileInductionVariable>(ctx);
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

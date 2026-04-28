//===------------------------- PLTToPLTM.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_PLTTOPLTM
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "ave-plt-to-pltm"

using namespace mlir;

static std::optional<int64_t> getConstantUB(scf::ForOp loop) {
  IntegerAttr ub;
  if (matchPattern(loop.getUpperBound(), m_Constant(&ub)))
    return ub.getInt();
  return {};
}

// check if the given map is in the form (-d0 + <loop_ub>, <loop_step>)
// If the loop has dynamic ub, the constantUB should be nullopt
static bool isAffineMapComputingTailBlock(AffineMap map,
                                          std::optional<int64_t> constantUB,
                                          int64_t step) {
  if (map.getNumResults() != 2)
    return false;
  if (AffineConstantExpr cst = dyn_cast<AffineConstantExpr>(map.getResult(1)))
    if (step != cst.getValue())
      return false;
  if (AffineBinaryOpExpr binOp =
          dyn_cast<AffineBinaryOpExpr>(map.getResult(0))) {
    // match for -d0 + s0/constant
    AffineConstantExpr ub = dyn_cast<AffineConstantExpr>(binOp.getRHS());

    if (!binOp.getRHS().isSymbolicOrConstant() ||
        (ub && !constantUB.has_value()) ||
        (ub && constantUB.has_value() && ub.getValue() != constantUB.value()))
      return false;
    if (AffineBinaryOpExpr negOp =
            dyn_cast<AffineBinaryOpExpr>(binOp.getLHS())) {
      if (!isa<AffineDimExpr>(negOp.getLHS()))
        return false;
      if (AffineConstantExpr negOne =
              dyn_cast<AffineConstantExpr>(negOp.getRHS()))
        return negOne.getValue() == -1;
    }
  }

  return false;
}

namespace {

struct PLTToPLTMPass : public impl::PLTToPLTMBase<PLTToPLTMPass> {
  using PLTToPLTMBase<PLTToPLTMPass>::PLTToPLTMBase;

public:
  void runOnOperation() override;
};

struct PLTToPLTMPattern : public OpRewritePattern<hivmave::VFPltOp> {
public:
  using OpRewritePattern<hivmave::VFPltOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivmave::VFPltOp op,
                                PatternRewriter &rewriter) const override {
    // don't do anything if something is using the post_update'ed value
    if (!op.getNewTrueShape().use_empty())
      return failure();
    for (auto u : op.getRes().getUsers()) {
      if (auto storeOp = dyn_cast<hivmave::VFMaskedStoreOp>(u)) {
        if (storeOp->hasAttr(hivmave::UnalignedAttr::name)) {
          return failure();
        }
      }
    }
    affine::AffineMinOp min = dyn_cast_if_present<affine::AffineMinOp>(
        op.getTrueShape().getDefiningOp());
    if (min && min.getNumOperands() >= 1) {
      scf::ForOp forOp = scf::getForInductionVarOwner(min->getOperand(0));
      if (forOp && isAffineMapComputingTailBlock(
                       min.getAffineMap(), getConstantUB(forOp),
                       forOp.getConstantStep()->getSExtValue())) {
        if (min->getNumOperands() == 2 &&
            min->getOperand(1) != forOp.getUpperBound())
          return failure();

        // replace the PLT with PLTM
        hivmave::VFPltMOp pltm = rewriter.create<hivmave::VFPltMOp>(
            op->getLoc(), op.getRes().getType(), forOp.getInductionVar(),
            forOp.getUpperBound());
        rewriter.replaceAllUsesWith(op.getRes(), pltm.getRes());
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};
} // namespace

void PLTToPLTMPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<PLTToPLTMPattern>(ctx);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> hivmave::createPLTToPLTMPass() {
  return std::make_unique<PLTToPLTMPass>();
}

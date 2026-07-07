//===------------- ProcessWideVextf.cpp - process vextf op
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "ave-process-wide-vextf"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_PROCESSWIDEVEXTF
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::scf;
using namespace mlir::vector;
using namespace mlir::hivmave;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace mlir::annotation;

// Unroll and interleave for loops with widening casts
struct WideVextfUnrollInterleavePattern : public OpRewritePattern<scf::ForOp> {
  WideVextfUnrollInterleavePattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // check if loop should be unrolled
    if (!shouldUnroll(forOp))
      return failure();

    // perform loop unrolling by factor 2
    auto unrollResult =
        loopUnrollByFactor(forOp, /*unrollFactor=*/2, /*annotateFn=*/nullptr);
    if (failed(unrollResult)) {
      return failure();
    }

    // get the unrolled loop body
    Block *unrolledBody = nullptr;
    if (unrollResult->mainLoopOp) {
      unrolledBody = unrollResult->mainLoopOp->getBody();
    } else if (unrollResult->epilogueLoopOp) {
      unrolledBody = unrollResult->epilogueLoopOp->getBody();
    } else {
      unrolledBody = forOp->getBlock();
    }

    if (!unrolledBody)
      return failure();

    return success();
  }

private:
  bool isInnermostLoop(scf::ForOp forOp) const {
    bool hasInnerLoop = false;
    forOp.walk([&](scf::ForOp inner) {
      if (inner != forOp) {
        hasInnerLoop = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return !hasInnerLoop;
  }

  bool shouldUnroll(scf::ForOp forOp) const {
    if (!isInnermostLoop(forOp))
      return false;

    auto step = forOp.getStep();
    IntegerAttr stepAttr;
    if (!matchPattern(step, m_Constant(&stepAttr))) {
      return false;
    }

    int64_t stepVal = stepAttr.getInt();
    if (stepVal != 64) {
      return false;
    }

    auto lb = forOp.getLowerBound();
    auto ub = forOp.getUpperBound();
    IntegerAttr lbAttr, ubAttr;
    if (!matchPattern(lb, m_Constant(&lbAttr)) ||
        !matchPattern(ub, m_Constant(&ubAttr))) {
      return false;
    }

    int64_t lbVal = lbAttr.getInt();
    int64_t ubVal = ubAttr.getInt();
    int64_t range = ubVal - lbVal;

    // TODO: Currently, only support even-numbered iterations of the 64-step
    // unroll. Support process tail blocks and odd-numbered iteration in future.
    if (range % 128 != 0 || range <= 128) {
      return false;
    }

    // loop must contain widening cast(vextf)
    if (!hasWideningCast(forOp))
      return false;

    return true;
  }

  bool hasWideningCast(scf::ForOp forOp) const {
    bool hasCast = false;
    forOp.walk([&](Operation *op) {
      if (isWideningCast(*op)) {
        hasCast = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return hasCast;
  }

  bool isWideningCast(Operation &op) const {
    // check for vextf widening operations
    if (auto vextfOp = dyn_cast<VFExtFOp>(op)) {
      VectorType srcType = cast<VectorType>(vextfOp.getSrc().getType());
      VectorType dstType = cast<VectorType>(vextfOp.getRes().getType());
      // TODO: Currently, only supporte conversion from 16-bit to 32-bit
      return srcType.getElementTypeBitWidth() == 16 &&
             dstType.getElementTypeBitWidth() == 32;
    }
    return false;
  }
};

namespace {
struct processWideVextfPass
    : public impl::ProcessWideVextfBase<processWideVextfPass> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<WideVextfUnrollInterleavePattern>(context);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> hivmave::createProcessWideVextfPass() {
  return std::make_unique<processWideVextfPass>();
}

//===------------------------- MoveUpArith.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "bishengir/Dialect/Utils/Util.h"

namespace mlir::arith {
#define GEN_PASS_DEF_MOVEUPARITH
#include "bishengir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir::arith

namespace {

using namespace mlir;
using namespace mlir::arith;

template <typename ArithOp>
struct MoveUpPattern : public OpRewritePattern<ArithOp> {
  using OpRewritePattern<ArithOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ArithOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->getPrevNode() || utils::isArithOp(op->getPrevNode())) {
      return rewriter.notifyMatchFailure(op, "do not need move up");
    }

    bool allArgsBlockArguments =
        llvm::all_of(op->getOperands(),
                     [](const Value opr) { return isa<BlockArgument>(opr); });
    if (allArgsBlockArguments) {
      auto applyParentBlock = op->getBlock();
      rewriter.moveOpBefore(op, applyParentBlock, applyParentBlock->begin());
      auto *newOp = rewriter.clone(*op);
      rewriter.replaceOp(op, newOp);
      return success();
    }

    if (!op->getOperands().empty()) {
      Operation *latestDef = op->getOperands()[0].getDefiningOp();
      if (latestDef && !(op->getPrevNode() == latestDef)) {
        rewriter.modifyOpInPlace(op, [&]() { op->moveAfter(latestDef); });
        return success();
      }
    }

    return rewriter.notifyMatchFailure(op, "do not need move up arith op.");
  }
};

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<MoveUpPattern<OpType>>(patterns.getContext());
}

template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populateMoveUpArithPatterns(RewritePatternSet &patterns) {
  registerAll<
#define GET_OP_LIST
#include "mlir/Dialect/Arith/IR/ArithOps.cpp.inc"
      >(patterns);
}

struct MoveUpArith
    : public arith::impl::MoveUpArithBase<MoveUpArith> {
  using MoveUpArithBase::MoveUpArithBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet loweringPatterns(context);
    populateMoveUpArithPatterns(loweringPatterns);

    if (failed(applyPatternsGreedily(op, std::move(loweringPatterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::arith::createMoveUpArithPass() {
  return std::make_unique<MoveUpArith>();
}
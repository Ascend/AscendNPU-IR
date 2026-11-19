//===------ LegalizeScalarFP8.cpp - Legalize scalar f8 conditionals -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_LEGALIZESCALARFP8PASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

/// Carry scalar f8 results through scf.if as i8, preserving all bits and
/// leaving the original consumers unchanged. Other result types are not
/// modified.
///
/// Before (f8 denotes either supported f8 format):
///   %v = scf.if %cond -> (f8) {
///     scf.yield %x : f8
///   } else {
///     scf.yield %y : f8
///   }
///   use(%v)
///
/// After:
///   %bits = scf.if %cond -> (i8) {
///     %xb = arith.bitcast %x : f8 to i8
///     scf.yield %xb : i8
///   } else {
///     %yb = arith.bitcast %y : f8 to i8
///     scf.yield %yb : i8
///   }
///   %v = arith.bitcast %bits : i8 to f8
///   use(%v)
struct LegalizeFP8IfOp : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> indices;
    SmallVector<Type> originalTypes;
    for (auto result : op.getResults()) {
      if (isa<Float8E4M3FNType, Float8E5M2Type>(result.getType())) {
        indices.push_back(result.getResultNumber());
        originalTypes.push_back(result.getType());
      }
    }
    if (indices.empty()) {
      return failure();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    for (Block *block : {op.thenBlock(), op.elseBlock()}) {
      auto yieldOp = cast<scf::YieldOp>(block->getTerminator());
      rewriter.setInsertionPoint(yieldOp);
      SmallVector<Value> values(yieldOp.getOperands());
      for (unsigned index : indices) {
        values[index] = rewriter.createOrFold<arith::BitcastOp>(
            yieldOp.getLoc(), rewriter.getI8Type(), values[index]);
      }
      rewriter.modifyOpInPlace(yieldOp, [&] { yieldOp->setOperands(values); });
    }
    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned index : indices) {
        op.getResult(index).setType(rewriter.getI8Type());
      }
    });
    rewriter.setInsertionPointAfter(op);
    for (auto [index, type] : llvm::zip(indices, originalTypes)) {
      auto restored = rewriter.create<arith::BitcastOp>(op.getLoc(), type,
                                                        op.getResult(index));
      rewriter.replaceAllUsesExcept(op.getResult(index), restored.getResult(),
                                    restored);
    }
    return success();
  }
};

/// Temporary workaround for backend parsing of f8 phi constants. Integer phi
/// nodes carry the exact f8 representation without numerical conversions.
struct LegalizeScalarFP8Pass
    : impl::LegalizeScalarFP8PassBase<LegalizeScalarFP8Pass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LegalizeFP8IfOp>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createLegalizeScalarFP8Pass() {
  return std::make_unique<LegalizeScalarFP8Pass>();
}

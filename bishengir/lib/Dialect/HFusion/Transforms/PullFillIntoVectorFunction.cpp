//===- PullFillIntoVectorFunction.cpp - Pull fill ops into VF -------------===//
//
// Part of the BiShengIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass rematerializes constant 0-rank tensor linalg.fill operations that
// haven't been outlined inside SIMD Vector Functions (VFs) and are passed as
// call arguments.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_PULLFILLINTOVECTORFUNCTION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Reconstruct an incoming fill-initialized 0-rank tensor at VF entry, while
/// preserving its caller-provided destination:
///
///   %filled = linalg.fill ins(%seed) outs(%dst)
///   %result = call @vf(%filled)
///
/// becomes call @vf(%dst), with a constant vector and tensor transfer_write at
/// VF entry. LoopInvariantPromotion and canonicalization can then fold the
/// accumulator's initial read to the constant and remove the initialization
/// write, leaving only the final write to the original destination.
struct PullConstantFillIntoVectorFunction
    : public OpRewritePattern<func::CallOp> {
  PullConstantFillIntoVectorFunction(MLIRContext *ctx, ModuleOp module)
      : OpRewritePattern<func::CallOp>(ctx), module(module) {}

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (!hfusion::util::isSIMDVectorFunction(call))
      return failure();
    auto callee = utils::getCalledFunction<func::FuncOp, func::CallOp>(call);
    if (!callee || callee.isExternal() ||
        !hfusion::util::isSIMDVectorFunction(callee) ||
        callee->isAncestor(call))
      return failure();

    /// TODO: skip VFs containing any masked
    /// transfer reads through tensor reshapes. Current MLIR canonicalizer can
    /// incorrectly replace such a read with a shape-cast result and lose the
    /// transfer-read padding/masked semantics.
    if (hasMaskedReshapedTransferRead(callee))
      return failure();

    auto uses = callee.getSymbolUses(module);
    // Outlined VFs are compiler-internal, even when their symbol visibility
    // is public. Specializing a shared/address-taken VF would change other
    // invocations. Count all symbol uses, not just marked VF calls.
    if (!uses.has_value() || !llvm::hasSingleElement(*uses) ||
        uses->begin()->getUser() != call.getOperation())
      return failure();

    for (auto [argIdx, operand] : llvm::enumerate(call.getOperands())) {
      auto fill = operand.getDefiningOp<linalg::FillOp>();
      if (!fill)
        continue;
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || !isSupportedFillType(type))
        continue;
      Attribute seed;

      /// linalg.fill may cast its scalar input to the destination element type.
      /// Only rematerialize seeds that do not require such a conversion.
      if (!matchFillSeed(fill, type.getElementType(), seed))
        continue;

      const auto arg = callee.getArgument(argIdx);
      if (arg.use_empty())
        continue;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&callee.getBody().front());
      auto initialized =
          buildVectorInitializer(rewriter, fill.getLoc(), arg, type, seed);
      rewriter.replaceAllUsesExcept(arg, initialized.getResult(), initialized);
      rewriter.modifyOpInPlace(call, [&, argIdx = argIdx]() {
        call.setOperand(argIdx, fill.getDpsInits().front());
      });
      if (fill->use_empty())
        rewriter.eraseOp(fill);
      return success();
    }
    return failure();
  }

private:
  static bool isSupportedFillType(RankedTensorType type) {
    if (type.getRank() != 0 || type.getEncoding())
      return false;

    auto elementType = type.getElementType();
    return elementType.isF16() || elementType.isBF16() || elementType.isF32();
  }

  static bool matchFillSeed(linalg::FillOp fill, Type elementType,
                            Attribute &seed) {
    if (!matchPattern(fill.getInputs().front(), m_Constant(&seed)))
      return false;
    auto typedSeed = dyn_cast<TypedAttr>(seed);
    if (!typedSeed || typedSeed.getType() != elementType)
      return false;
    return isa<FloatAttr>(seed);
  }

  static vector::TransferWriteOp
  buildVectorInitializer(PatternRewriter &rewriter, Location loc,
                         Value destination, RankedTensorType tensorType,
                         Attribute seed) {
    auto vectorType = VectorType::get({}, tensorType.getElementType());
    auto constant = rewriter.create<arith::ConstantOp>(
        loc, vectorType, DenseElementsAttr::get(vectorType, seed));
    return rewriter.create<vector::TransferWriteOp>(loc, constant, destination,
                                                    ValueRange{});
  }

  static bool hasReshapeOrigin(Value value) {
    if (value.getDefiningOp<tensor::ExpandShapeOp>() ||
        value.getDefiningOp<tensor::CollapseShapeOp>())
      return true;

    auto arg = dyn_cast<BlockArgument>(value);
    auto forOp = arg ? dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())
                     : value.getDefiningOp<scf::ForOp>();
    if (!forOp || (arg && arg.getArgNumber() == 0))
      return false;

    auto tripCount = constantTripCount(forOp.getLowerBound(),
                                       forOp.getUpperBound(), forOp.getStep());
    if (!tripCount || *tripCount != 1)
      return false;

    if (arg)
      return hasReshapeOrigin(forOp.getInitArgs()[arg.getArgNumber() - 1]);
    return hasReshapeOrigin(
        forOp.getYieldedValues()[cast<OpResult>(value).getResultNumber()]);
  }

  static bool hasMaskedReshapedTransferRead(func::FuncOp func) {
    bool affected = false;
    func.walk([&](vector::TransferReadOp read) {
      if (affected || !read.getMask())
        return;
      affected = hasReshapeOrigin(read.getSource());
    });
    return affected;
  }

  ModuleOp module;
};

struct PullFillIntoVectorFunction
    : public impl::PullFillIntoVectorFunctionBase<PullFillIntoVectorFunction> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PullConstantFillIntoVectorFunction>(patterns.getContext(),
                                                     getOperation());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createPullFillIntoVectorFunctionPass() {
  return std::make_unique<PullFillIntoVectorFunction>();
}

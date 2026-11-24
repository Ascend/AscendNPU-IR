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

#include <cstdint>

namespace mlir {
#define GEN_PASS_DEF_PULLFILLINTOVECTORFUNCTION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

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

static bool callPullQualified(func::CallOp call, func::FuncOp callee,
                              ModuleOp module) {
  if (!hfusion::util::isSIMDVectorFunction(call))
    return false;
  if (!callee || callee.isExternal() ||
      !hfusion::util::isSIMDVectorFunction(callee) || callee->isAncestor(call))
    return false;

  /// TODO: this skips VFs containing any masked
  /// transfer reads through tensor reshapes. Current MLIR canonicalizer can
  /// incorrectly replace such a read with a shape-cast result and lose the
  /// transfer-read padding/masked semantics.
  /// Should be lifted after canonicalizer is fixed.
  if (hasMaskedReshapedTransferRead(callee))
    return false;

  auto uses = callee.getSymbolUses(module);
  // Outlined VFs are compiler-internal, even when their symbol visibility
  // is public. Specializing a shared/address-taken VF would change other
  // invocations. Count all symbol uses, not just marked VF calls.
  if (!uses.has_value() || !llvm::hasSingleElement(*uses) ||
      uses->begin()->getUser() != call.getOperation())
    return false;

  return true;
}

static bool typePullQualified(RankedTensorType type, int64_t rank) {
  // singleton && required rank
  return type.hasStaticShape() && type.getNumElements() == 1 &&
         type.getRank() == rank && !type.getEncoding();
}

static bool isBroadcastable(Type type) {
  // lowering of broadcast for most types is currently either unsupported or
  // buggy
  // TODO: add more supported types when lowering path is fixed
  return type.isF16() || type.isBF16() || type.isF32();
}

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
    auto callee = utils::getCalledFunction<func::FuncOp, func::CallOp>(call);
    if (!callPullQualified(call, callee, module))
      return failure();

    for (auto [argIdx, operand] : llvm::enumerate(call.getOperands())) {
      auto fill = operand.getDefiningOp<linalg::FillOp>();
      if (!fill)
        continue;
      auto type = dyn_cast<RankedTensorType>(operand.getType());

      if (!type || !typePullQualified(type, /*rank=*/0) ||
          !isBroadcastable(type.getElementType()))
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

  ModuleOp module;
};

//===----------------------------------------------------------------------===//
// PullScalarIntoVectorFunction: singleton tensor operand handling
//===----------------------------------------------------------------------===//
//
// Replace caller-side singleton tensor construction with a scalar VF argument:
//
//   %tensor = tensor.from_elements %scalar : tensor<1xf32>
//   call @vf(%tensor)
//
// becomes call @vf(%scalar), with a vector.broadcast and transfer_write at VF
// entry. A tensor.insert at index zero into tensor<1xT> is equivalent and is
// handled by the same rewrite.
struct PullScalarIntoVectorFunction : public OpRewritePattern<func::CallOp> {
  PullScalarIntoVectorFunction(MLIRContext *ctx, ModuleOp module)
      : OpRewritePattern<func::CallOp>(ctx), module(module) {}

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto callee = utils::getCalledFunction<func::FuncOp, func::CallOp>(call);
    if (!callPullQualified(call, callee, module))
      return failure();

    SmallVector<ScalarOperandMatch> matches;
    for (auto [argIdx, operand] : llvm::enumerate(call.getOperands())) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());

      if (!type || !typePullQualified(type, /*rank=*/1) ||
          !isBroadcastable(type.getElementType()))
        continue;

      if (callee.getArgument(argIdx).use_empty())
        continue;

      auto scalar = matchScalarOperand(operand);
      if (!scalar)
        continue;
      matches.push_back(ScalarOperandMatch{argIdx, scalar, type,
                                           operand.getDefiningOp()->getLoc()});
    }

    if (matches.empty())
      return failure();

    SmallVector<Type> newInputTypes(callee.getFunctionType().getInputs());
    for (const ScalarOperandMatch &match : matches)
      newInputTypes[match.argIdx] = match.scalar.getType();

    Block &entryBlock = callee.getBody().front();
    rewriter.modifyOpInPlace(callee, [&]() {
      callee.setFunctionType(rewriter.getFunctionType(
          newInputTypes, callee.getFunctionType().getResults()));
      for (const ScalarOperandMatch &match : matches)
        entryBlock.getArgument(match.argIdx).setType(match.scalar.getType());
    });

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&entryBlock);
      for (const ScalarOperandMatch &match : matches) {
        BlockArgument scalarArg = entryBlock.getArgument(match.argIdx);
        Location loc = match.materializationLoc;
        auto empty = rewriter.create<tensor::EmptyOp>(loc, match.tensorType,
                                                      ValueRange{});
        auto vectorType =
            VectorType::get({1}, match.tensorType.getElementType());
        auto broadcast =
            rewriter.create<vector::BroadcastOp>(loc, vectorType, scalarArg);
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto initialized = rewriter.create<vector::TransferWriteOp>(
            loc, broadcast, empty, ValueRange{zero});
        rewriter.replaceAllUsesExcept(scalarArg, initialized.getResult(),
                                      broadcast);
      }
    }

    rewriter.modifyOpInPlace(call, [&]() {
      for (const ScalarOperandMatch &match : matches)
        call.setOperand(match.argIdx, match.scalar);
    });
    return success();
  }

private:
  struct ScalarOperandMatch {
    size_t argIdx;
    Value scalar;
    RankedTensorType tensorType;
    Location materializationLoc;
  };

  static Value matchScalarOperand(Value operand) {
    if (auto fromElements = operand.getDefiningOp<tensor::FromElementsOp>()) {
      if (fromElements.getElements().size() == 1)
        return fromElements.getElements().front();
      return {};
    }

    auto insert = operand.getDefiningOp<tensor::InsertOp>();
    if (!insert || insert.getIndices().size() != 1 ||
        !matchPattern(insert.getIndices().front(), m_Zero()))
      return {};
    return insert.getScalar();
  }

  ModuleOp module;
};

struct PullFillIntoVectorFunction
    : public impl::PullFillIntoVectorFunctionBase<PullFillIntoVectorFunction> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<PullConstantFillIntoVectorFunction, PullScalarIntoVectorFunction>(
            patterns.getContext(), getOperation());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createPullFillIntoVectorFunctionPass() {
  return std::make_unique<PullFillIntoVectorFunction>();
}

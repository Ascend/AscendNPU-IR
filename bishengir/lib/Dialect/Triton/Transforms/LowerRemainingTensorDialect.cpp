//===- LowerRemainingTensorDialect.cpp ----------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers remaining tensor dialect ops before the TritonToTritonGPU pass to
// ensure layouts are correctly given
// Signals a pass failure if a tensor dialect op is not lowered
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

namespace bishengir::triton {
#define GEN_PASS_DEF_LOWERREMAININGTENSORDIALECT
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

Value createZeroTensor(OpBuilder &builder, Location loc,
                       RankedTensorType tensorType) {
  Value tensor;
  Type elementType = tensorType.getElementType();
  if (auto ptrType = dyn_cast<triton::PointerType>(elementType)) {
    Value scalarZero =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
    Value tritonNullptr =
        builder.create<triton::IntToPtrOp>(loc, elementType, scalarZero);
    tensor = builder.create<triton::SplatOp>(loc, tensorType, tritonNullptr);
  } else {
    DenseElementsAttr constAttr =
        DenseElementsAttr::get(tensorType, builder.getZeroAttr(elementType));
    tensor = builder.create<arith::ConstantOp>(loc, constAttr);
  }

  return tensor;
}

// Returns the error message for the given op if it cannot be lowered, otherwise
// returns std::nullopt if supported
std::optional<Twine> getError(Operation *op) {
  if (!isa<tensor::TensorDialect>(op->getDialect())) {
    return std::nullopt;
  }
  return llvm::TypeSwitch<Operation *, std::optional<Twine>>(op)
      .Case<tensor::EmptyOp>([](tensor::EmptyOp op) -> std::optional<Twine> {
        if (op.getDynamicSizes().size() > 0) {
          return "tensor::EmptyOp's with dynamic sizes are not supported "
                 "in SIMT mode";
        }
        return std::nullopt;
      })
      .Case<tensor::FromElementsOp>(
          [](tensor::FromElementsOp op) -> std::optional<Twine> {
            if (op->getNumOperands() > 1) {
              return "tensor::FromElementsOp's with more than 1 scalar operand "
                     "are not supported in SIMT mode";
            }
            return std::nullopt;
          })
      .Default([](Operation *op) -> std::optional<Twine> {
        return op->getName().getStringRef() +
               " is an unsupported tensor dialect operation";
      });
}

struct EmptyOpToConstantOp : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    if (getError(op)) {
      return failure();
    }

    Location loc = op->getLoc();
    RankedTensorType type = op.getType();
    Value constTensor = createZeroTensor(rewriter, loc, type);
    rewriter.replaceOp(op, constTensor);
    return success();
  }
};

struct FromElementsToSplatOp : public OpRewritePattern<tensor::FromElementsOp> {
  using OpRewritePattern<tensor::FromElementsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    if (getError(op)) {
      return failure();
    }
    Location loc = op->getLoc();
    RankedTensorType resultType = op.getType();
    Value scalar = op->getOperand(0);
    Value replacement =
        rewriter.create<triton::SplatOp>(loc, resultType, scalar);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class LowerRemainingTensorDialectPass
    : public impl::LowerRemainingTensorDialectBase<
          LowerRemainingTensorDialectPass> {
public:
  using LowerRemainingTensorDialectBase::LowerRemainingTensorDialectBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<EmptyOpToConstantOp, FromElementsToSplatOp>(ctx);

    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
      mod.emitError(
          "Unsupported tensor dialect operations found in the SIMT kernel");
      signalPassFailure();
      return;
    }

    // Anything left over is an unsupported tensor dialect op
    bool hasRemainingTensorOps = false;
    mod.walk([&](Operation *op) {
      std::optional<Twine> errorMsg = getError(op);
      if (errorMsg) {
        hasRemainingTensorOps = true;
        op->emitError(*errorMsg);
      }
    });
    if (hasRemainingTensorOps) {
      mod.emitError(
          "Unsupported tensor dialect operations found in the SIMT kernel");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerRemainingTensorDialectPass() {
  return std::make_unique<LowerRemainingTensorDialectPass>();
}

} // namespace bishengir::triton

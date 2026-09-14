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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

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

struct EmptyOpToConstantOp : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    // Return failure if tensor shape is not known at compile time
    if (op.getDynamicSizes().size() > 0) {
      return failure();
    }
    Location loc = op->getLoc();
    RankedTensorType type = op.getType();
    Value constTensor = createZeroTensor(rewriter, loc, type);
    rewriter.replaceOp(op, constTensor);
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
    patterns.add<EmptyOpToConstantOp>(ctx);

    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
      mod.emitError(
          "Unsupported tensor dialect operations found in the SIMT kernel");
      signalPassFailure();
      return;
    }

    // Anything left over is an unsupported tensor dialect op
    bool hasRemainingTensorOps = false;
    mod.walk([&](Operation *op) {
      if (isa<tensor::TensorDialect>(op->getDialect())) {
        op->emitError(op->getName().getStringRef() +
                      " is an unsupported tensor dialect operation");
        hasRemainingTensorOps = true;
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

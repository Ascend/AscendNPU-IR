//===- Lowering.cpp - Annotation lowering pass -------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/Transforms/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "annotation-lowering"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_ANNOTATIONLOWERING
#include "bishengir/Dialect/Annotation/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::annotation;

namespace {
class MarkOpLowering : public ConversionPattern {
public:
  explicit MarkOpLowering(MLIRContext *context)
      : ConversionPattern(MarkOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct AnnotationLoweringPass
    : public impl::AnnotationLoweringBase<AnnotationLoweringPass> {
  void runOnOperation() override {
    auto *func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addIllegalDialect<annotation::AnnotationDialect>();
    patterns.add<MarkOpLowering>(context);

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<Pass> mlir::annotation::createAnnotationLoweringPass() {
  return std::make_unique<AnnotationLoweringPass>();
}

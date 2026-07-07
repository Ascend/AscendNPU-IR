//===- Lowering.cpp - Annotation lowering pass -------------------*- C++-*-===//
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

#include "bishengir/Dialect/Annotation/Transforms/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Transforms/DialectConversion.h"
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

//===- EraseSymbol.cpp ------------- Erase Symbol Pass --------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to erase symbols
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace symbol {
#define GEN_PASS_DEF_ERASESYMBOL
#include "bishengir/Dialect/Symbol/Transforms/Passes.h.inc"

namespace {

template <typename OpType>
class EraseSymbol : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

class EraseSymbolPass : public impl::EraseSymbolBase<EraseSymbolPass> {
public:
  explicit EraseSymbolPass() : EraseSymbolBase() {}
  void runOnOperation() final;
};

void EraseSymbolPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  patterns.add<EraseSymbol<symbol::BindSymbolicShapeOp>>(ctx);
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass> createEraseSymbolPass() {
  return std::make_unique<EraseSymbolPass>();
}

} // namespace symbol
} // namespace mlir
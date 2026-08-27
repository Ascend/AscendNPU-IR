//===- EraseSymbol.cpp ------------- Erase Symbol Pass --------------------===//
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

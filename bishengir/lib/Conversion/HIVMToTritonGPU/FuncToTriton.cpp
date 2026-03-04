//===---- FuncToTriton.cpp - conversion from Func to Triton dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::triton;

namespace {
class FuncOpPattern : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto oldFuncTy = op.getFunctionType();
    // Convert function type which contains parameter types
    // Note: The function can only be simt vf, and no return value.
    SmallVector<Type> newInputTypes;
    for (auto inputTy : oldFuncTy.getInputs()) {
      newInputTypes.push_back(HIVMToTritonTypeConvert(inputTy));
    }

    auto funcType = FunctionType::get(op.getContext(), newInputTypes,
                                      oldFuncTy.getResults());
    auto newFunc =
        rewriter.create<triton::FuncOp>(op.getLoc(), op.getName(), funcType);
    auto *newEntryBlock = newFunc.addEntryBlock();

    IRMapping mapping;

    // Update block argument types in new tt.func and build the map from old
    // block argument to new block argument
    auto &oldEntryBlock = op.getBody().front();
    for (auto [idx, oldArg] : llvm::enumerate(oldEntryBlock.getArguments())) {
      auto newArg = newEntryBlock->getArgument(idx);
      newArg.setType(funcType.getInput(idx));
      mapping.map(oldArg, newArg);
    }

    // Clone all of operators in entry block recursively.
    // Note: There is only one top block named entry block in ttir
    rewriter.setInsertionPointToStart(newEntryBlock);
    assert(op.getBody().getBlocks().size() == 1 &&
           "Multi blocks are not supported");
    for (auto &oldOp : oldEntryBlock.getOperations()) {
      // Replace the func.return with tt.return
      if (isa<func::ReturnOp>(oldOp)) {
        rewriter.create<triton::ReturnOp>(op.getLoc());
        continue;
      }
      rewriter.clone(oldOp, mapping);
    }
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::hivm::populateFuncToTritonPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<FuncOpPattern>(context);
}

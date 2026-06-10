//===- Context.h - Auto vectorization context helpers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_CONTEXT_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_CONTEXT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include <string>

namespace mlir {
namespace hfusion {

struct VectorizeContext {
  func::FuncOp func;
  unsigned maxFusedOps;
  bool enableMultipleConsumerFusion;
  unsigned loopCount = 0;

  void resetLoopCount() { loopCount = 0; }

  std::string nextLoopLabel() {
    return "outlined-loop-target-" + std::to_string(++loopCount);
  }

  func::FuncOp cloneFunc(OpBuilder &builder) {
    MLIRContext *mlirContext = func->getContext();
    std::string funcName = func.getSymName().str();

    builder.setInsertionPointAfter(func);
    func::FuncOp clonedFunc = cast<func::FuncOp>(builder.clone(*func));
    SymbolTable::setSymbolName(
        clonedFunc, StringAttr::get(mlirContext, "cloned_" + funcName));
    return clonedFunc;
  }

  void restoreFunc(func::FuncOp clonedFunc, IRRewriter &rewriter) {
    MLIRContext *mlirContext = func->getContext();
    std::string funcName = func.getSymName().str();

    rewriter.eraseOp(func);
    SymbolTable::setSymbolName(clonedFunc,
                               StringAttr::get(mlirContext, funcName));
    func = clonedFunc;
  }
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_CONTEXT_H

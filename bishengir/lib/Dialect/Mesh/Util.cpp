//===--------- Util.cpp - Utility functions for Mesh dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the lowering logic for host code to clean up remaining
// bufferization operations and allocs that deal with memrefs in different
// address spaces.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/Mesh/Util.h"

using namespace mlir;
/// Helper to create a function symbol in the module.
func::FuncOp bishengir::getCustomFunction(StringRef name, ModuleOp parent,
                                          Location loc, OpBuilder &builder,
                                          TypeRange funcArgs,
                                          TypeRange results) {
  SymbolTable st(parent);
  auto func = st.lookup<func::FuncOp>(name);
  if (func) {
    return func;
  }

  auto *block = builder.getInsertionBlock();
  auto insertionPoint = builder.getInsertionPoint();

  builder.setInsertionPointToStart(parent.getBody());
  auto funcTy = FunctionType::get(builder.getContext(), funcArgs, results);
  auto newFunc = builder.create<func::FuncOp>(loc, name, funcTy);
  newFunc.setVisibility(SymbolTable::Visibility::Private);
  // Set host attribute for hccl operationinns
  hacc::utils::setHost(newFunc);

  builder.setInsertionPoint(block, insertionPoint);
  return newFunc;
}

//===- Utils.cpp - Symbol Dialect Utilities ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/Utils/Utils.h"

using namespace mlir;
using namespace mlir::symbol;

std::optional<symbol::BindSymbolicShapeOp>
utils::getBindSymbolUser(Value value) {
  for (Operation *userOp : value.getUsers()) {
    if (auto target = dyn_cast<symbol::BindSymbolicShapeOp>(userOp)) {
      return target;
    }
  }
  return std::nullopt;
}

Location utils::getValueLocation(Value val) {
  if (Operation *op = val.getDefiningOp()) {
    return op->getLoc();
  }
  auto blockArg = llvm::cast<BlockArgument>(val);
  return blockArg.getOwner()->front().getLoc();
}
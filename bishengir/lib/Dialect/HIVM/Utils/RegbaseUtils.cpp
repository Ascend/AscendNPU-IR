//===- RegbaseUtils.cpp - Utilities to support the HIVM dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the HIVM dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"

namespace mlir {
namespace hivm {
bool isVFCall(Operation *op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    if (callOp->hasAttr(hivm::VectorFunctionAttr::name))
      return true;
  }
  return false;
}

bool isVF(func::FuncOp funcOp) {
  return funcOp->hasAttr(hivm::VectorFunctionAttr::name);
}
} // namespace hivm
} // namespace mlir

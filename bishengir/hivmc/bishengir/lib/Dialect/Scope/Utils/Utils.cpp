//===- Utils.cpp - Scope Dialect Utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Scope/Utils/Utils.h"

namespace mlir {
namespace scope {

// check if funcOp is manual vf scope
bool utils::isManualVFScope(func::FuncOp funcOp) {
  return funcOp->hasAttr("vector_mode");
}

} // namespace scope
} // namespace mlir

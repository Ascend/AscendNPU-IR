//===- Utils.cpp - Scope Dialect Utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Scope/Utils/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

namespace mlir {
namespace scope {

// check if funcOp is manual vf scope
bool utils::isManualVFScope(func::FuncOp funcOp) {
  return funcOp->hasAttr("vector_mode");
}

bool utils::isInCubeScope(Operation *op) {
  auto scopeOp = op->getParentOfType<ScopeOp>();
  if (!scopeOp)
    return false;

  auto attr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!attr)
    return false;

  return attr.getTcoretype() == mlir::hivm::TCoreType::CUBE;
}

} // namespace scope
} // namespace mlir

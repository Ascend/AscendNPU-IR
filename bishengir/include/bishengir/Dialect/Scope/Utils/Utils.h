//===- Utils.h - Scope Dialect Utilities ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_SCOPE_DIALECT_UTILS_H
#define BISHENGIR_SCOPE_DIALECT_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace scope {
namespace utils {

// check if funcOp is manual vf scope
bool isManualVFScope(func::FuncOp funcOp);

// Check if operations is in cube scope
bool isInCubeScope(Operation *op);

} // namespace utils
} // namespace scope
} // namespace mlir

#endif // BISHENGIR_SCOPE_DIALECT_UTILS_H

//===- RegbaseUtils.h - Utilities to support the HIVM dialect ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_HIVM_UTILS_REGBASEUTILS_H
#define MLIR_DIALECT_HIVM_UTILS_REGBASEUTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace hivm {
bool isVFCall(Operation *op);

bool isVF(func::FuncOp funcOp);

} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_REGBASEUTILS_H

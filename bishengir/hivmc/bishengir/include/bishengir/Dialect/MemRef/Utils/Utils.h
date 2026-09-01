//===- Utils.h - Utilities to support the Memref dialect ----------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_UTILS_UTILS_H
#define MLIR_DIALECT_MEMREF_UTILS_UTILS_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::memref {
BitVector getContiguousAxes(ArrayRef<Type> shapedTypes);

LogicalResult foldMemRefSpaceCast(Operation *op, Value inner = nullptr);

} // namespace mlir::memref

#endif // MLIR_DIALECT_MEMREF_UTILS_UTILS_H

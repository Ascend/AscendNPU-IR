//===- MemRefExtImpl.h - MemRefExt dialect Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_MEMREF_IR_MEMREFEXTIMPL_H
#define BISHENGIR_DIALECT_MEMREF_IR_MEMREFEXTIMPL_H

#include "mlir/IR/Value.h"

namespace mlir {
namespace memref_ext {

/// Determine whether the current buffer is defined by one of the following:
///   - memref.alloc
///   - memref_ext.alloc_workspace
bool isDefiningOpAllocLike(Value operand);

} // namespace memref_ext
} // namespace mlir

#endif // BISHENGIR_DIALECT_MEMREF_IR_MEMREFEXTIMPL_H

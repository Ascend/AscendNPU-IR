//===- MemRefImpl.h - MemRef Dialect Implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_MEMREF_IR_MEMREFIMPL_H
#define BISHENGIR_DIALECT_MEMREF_IR_MEMREFIMPL_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace memref {

constexpr llvm::StringLiteral kBindBufferAttrName = "bind_buffer";

/// Create memref.alloc op with the same type as source
Value createMemRefAllocOp(OpBuilder &builder, Location loc, Value source);

/// Create memref.alloc op with the same shape as source
/// but with element type targetElemType
Value createMemRefAllocOpWithTargetElemType(OpBuilder &builder, Location loc,
                                            Value source, Type targetElemType);

} // namespace memref
} // namespace mlir

#endif // BISHENGIR_DIALECT_MEMREF_IR_MEMREFIMPL_H

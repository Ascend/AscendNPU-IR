//===- TensorImpl.h - Tensor Dialect Implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TENSOR_IR_TENSORIMPL_H
#define BISHENGIR_DIALECT_TENSOR_IR_TENSORIMPL_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace tensor {

/// Create tensor.empty op with the same type as source
Value createTensorEmptyOp(OpBuilder &builder, Location loc, Value source);

/// Create tensor.empty op with the same shape as source
/// but with element type targetElemType
Value createTensorEmptyOpWithTargetElemType(OpBuilder &builder, Location loc,
                                            Value source, Type targetElemType);

} // namespace tensor
} // namespace mlir

#endif // BISHENGIR_DIALECT_TENSOR_IR_TENSORIMPL_H

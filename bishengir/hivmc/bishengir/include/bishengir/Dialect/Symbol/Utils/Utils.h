//===- Utils.h - Symbol Dialect Utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_SYMBOL_DIALECT_UTILS_H
#define BISHENGIR_SYMBOL_DIALECT_UTILS_H

#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace mlir {
namespace symbol {
namespace utils {

std::optional<symbol::BindSymbolicShapeOp> getBindSymbolUser(Value value);

// get defining operation location of a value
Location getValueLocation(Value val);

} // namespace utils
} // namespace symbol
} // namespace mlir

#endif // BISHENGIR_SYMBOL_DIALECT_UTILS_H

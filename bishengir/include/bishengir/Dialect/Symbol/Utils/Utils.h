//===- Utils.h - Symbol Dialect Utilities -----------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

template <typename Type>
std::optional<Type> getAnyUserOfType(Value value) {
  for (Operation *userOp : value.getUsers()) {
    if (auto target = dyn_cast<Type>(userOp)) {
      return target;
    }
  }
  return std::nullopt;
}


} // namespace utils
} // namespace symbol
} // namespace mlir

#endif // BISHENGIR_SYMBOL_DIALECT_UTILS_H

//===- OpLayoutInterface.h ------------------------------------------------===//
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

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_H
#define BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace hivm {
/// Forward declaration.
class DataLayoutAttr;
struct FractalOperandLayouts {
  Attribute a;
  Attribute b;
  Attribute c;
  // Null attribute indicates the op does not have per-channel bias.
  Attribute bias;
};
} // namespace hivm
} // namespace mlir

// Include the generated interface declarations.
#include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_H

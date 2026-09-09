//===- VectorizableOpInterface.h ------------------------------------------===//
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

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_H
#define BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::OpTrait {
/// Marker trait for ops that are not vectorizable
template <typename ConcreteType>
class NotVectorizableTrait
    : public TraitBase<ConcreteType, NotVectorizableTrait> {};
} // namespace mlir::OpTrait

// Include the generated interface declarations.
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_VECTORIZABLEOPINTERFACE_H

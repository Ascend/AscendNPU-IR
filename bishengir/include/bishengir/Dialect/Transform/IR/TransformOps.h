//===- TransformOps.h - Bishengir transform ops -----------------*- C++ -*-===//
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
//
// Dialect-neutral transform ops shared by the BishengIR schedule
// infrastructure. These ops only depend on generic dialects (func, tensor,
// linalg, ...), not on any BishengIR hardware dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H
#define BISHENGIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformTypes.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// Bishengir Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/Transform/IR/TransformOps.h.inc"

namespace bishengir {
namespace transform {
void registerTransformDialectExtension(mlir::DialectRegistry &registry);
} // namespace transform
} // namespace bishengir

#endif // BISHENGIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H

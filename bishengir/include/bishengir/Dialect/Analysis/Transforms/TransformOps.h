//===- TransformOps.h - Analysis transform ops ------------------*- C++ -*-===//
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
// Dialect-neutral transform ops used by the Analysis-level schedule
// infrastructure. These ops only depend on generic dialects (func, tensor,
// linalg, ...), not on any BishengIR hardware dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_TRANSFORMS_TRANSFORMOPS_H
#define BISHENGIR_DIALECT_ANALYSIS_TRANSFORMS_TRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformTypes.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// Analysis Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/Analysis/Transforms/TransformOps.h.inc"

namespace mlir {
namespace analysis {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace analysis
} // namespace mlir

#endif // BISHENGIR_DIALECT_ANALYSIS_TRANSFORMS_TRANSFORMOPS_H

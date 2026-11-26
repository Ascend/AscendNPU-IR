//===- HFusionTransformOps.h - HFusion transform ops -------------*- C++-*-===//
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

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// HFusion Transform Operations
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h.inc"

namespace mlir {
namespace hfusion {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS_H

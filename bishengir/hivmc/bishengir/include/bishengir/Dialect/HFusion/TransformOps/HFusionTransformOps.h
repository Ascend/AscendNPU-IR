//===- HFusionTransformOps.h - HFusion transform ops -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS_H

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

//===- SCFTransformOps.h - BiShengIR SCF transform ops ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H
#define BISHENGIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// BiShengIR SCF Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.h.inc"

namespace mlir {
class DialectRegistry;
}

namespace bishengir {
namespace scf {
void registerTransformDialectExtension(::mlir::DialectRegistry &registry);
} // namespace scf
} // namespace bishengir

#endif // BISHENGIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H

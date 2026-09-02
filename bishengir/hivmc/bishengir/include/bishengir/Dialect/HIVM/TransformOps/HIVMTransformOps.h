//===- HIVMTransformOps.h - HIVM transform ops -------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_HIVMTRANSFORMOPS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_HIVMTRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// HIVM Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.h.inc"

namespace mlir {
namespace hivm {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_HIVMTRANSFORMOPS_H

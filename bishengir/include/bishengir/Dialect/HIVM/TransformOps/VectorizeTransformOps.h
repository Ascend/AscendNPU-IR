//===- VectorizeTransformOps.h - Vectorizer transforms ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_HIVM_VECTORIZE_TRANSFORM_OPS_H
#define BISHENGIR_HIVM_VECTORIZE_TRANSFORM_OPS_H

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeAttrs.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h.inc"

namespace mlir::hivm {
void registerVectorizeTransformDialectExtension(DialectRegistry &registry);
}

#endif

//===----- CacheFuncIO.h - cache func input and output args -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_CACHEFUNCIO_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_CACHEFUNCIO_H
namespace mlir {
namespace hfusion {
/// Apply caching to the input and output of the target function.
/// When \p annotate is true, the caching op will be annotated.
void cacheFuncIO(func::FuncOp funcOp, bool annotate = false,
                 bool writeUnique = false);
} // namespace hfusion
} // namespace mlir
#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_CACHEFUNCIO_H

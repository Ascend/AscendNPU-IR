//===----- ReorderOpsByBFS.h - reorder hfusion ops by bfs--------*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_REORDEROPSBYBFS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_REORDEROPSBYBFS_H
namespace mlir {
namespace hfusion {
void reorderOpsByBFS(func::FuncOp funcOp);
}
} // namespace mlir
#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_REORDEROPSBYBFS_H

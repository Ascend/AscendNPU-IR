//===-- Transform.h. - Pass Entrypoints -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_SCF_TRANSFORMS_TRANSFORM_H
#define BISHENGIR_DIALECT_SCF_TRANSFORMS_TRANSFORM_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace bishengir {
namespace scf {
/// normalize scf::for with integer type into an index type. And replace the
/// scf::for with new one. And it only can be used for scf::for without yield
/// part
mlir::scf::ForOp normalizeToIndex(mlir::PatternRewriter &rewriter,
                                  mlir::scf::ForOp op);
} // namespace scf
} // namespace bishengir

#endif // BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_H

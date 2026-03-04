//===- Transforms.h - Tensor Transformation Patterns ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H
#define BISHENGIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace bishengir {
namespace tensor {

/// Populates `patterns` with patterns that...
void populateOptimizeDpsOpWithYieldedInsertSlicePattern(
    mlir::RewritePatternSet &patterns);

} // namespace tensor
} // namespace bishengir

#endif // BISHENGIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H

//===- Transforms.h - Vector Dialect Transformation Entrypoints *- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_VECTOR_TRANSFORMS_H
#define BISHENGIR_DIALECT_VECTOR_TRANSFORMS_H

namespace mlir {

class RewritePatternSet;

namespace vector {

/// Collect a set of patterns to Flatten Vector ops
void populateFlattenVectorOpsPattern(RewritePatternSet &patterns);

} // namespace vector
} // namespace mlir

#endif // BISHENGIR_DIALECT_VECTOR_TRANSFORMS_H

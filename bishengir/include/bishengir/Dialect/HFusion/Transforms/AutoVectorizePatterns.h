//===- AutoVectorizePatterns.h - Auto-vectorization cleanup patterns -------===//
//
// Part of the BiShengIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares cleanup rewrite patterns used by the auto-vectorization
// passes. These patterns are shared by AutoVectorize and AutoVectorize2 and are
// responsible for canonicalizing and simplifying vector-related IR after
// vectorization.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZEPATTERNS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZEPATTERNS_H
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hfusion {

/// Populate CleanUp patterns shared by AutoVectorize and AutoVectorize2.
void populateAutoVectorizeCleanUpPatterns(RewritePatternSet &patterns);

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZEPATTERNS_H
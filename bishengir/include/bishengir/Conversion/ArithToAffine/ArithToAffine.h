//===- ArithToAffine.h - Arith to Affine conversion -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_ARITHTOAFFINE_ARITHTOAFFINE_H
#define BISHENGIR_CONVERSION_ARITHTOAFFINE_ARITHTOAFFINE_H

#include <memory>

namespace mlir {
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOAFFINE
#include "bishengir/Conversion/Passes.h.inc"

namespace arith {
void populateArithToAffineConversionPatterns(RewritePatternSet &patterns);
} // namespace arith

/// Creates a pass to convert the Arith dialect to the Affine dialect.
std::unique_ptr<Pass> createArithToAffineConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_ARITHTOAFFINE_ARITHTOAFFINE_H
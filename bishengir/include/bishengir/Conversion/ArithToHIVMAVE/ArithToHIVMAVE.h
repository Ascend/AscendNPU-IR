
//===- ArithToHIVMAVE.h - Arith To HIVMAVE ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the Arith dialect to the AVE dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_ARITHTOHIVMAVE_ARITHTOHIVMAVE_H
#define BISHENGIR_CONVERSION_ARITHTOHIVMAVE_ARITHTOHIVMAVE_H

#include <memory>

namespace mlir {
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOHIVMAVE
#include "bishengir/Conversion/Passes.h.inc"

namespace hivmave {
/// Conllect the patterns to convert from the arith dialect to ave dialect.
void populateArithToHIVMAVEConversionPatterns(RewritePatternSet &patterns);
} // namespace hivmave

/// Creates a pass to convert the Arith dialect to the AVE dialect.
std::unique_ptr<Pass> createArithToHIVMAVEConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_ARITHTOHIVMAVE_ARITHTOHIVMAVE_H

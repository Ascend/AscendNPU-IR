
//===- VectorToHIVMAVE.h - Vector To HIVMAVE --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the Vector dialect to the AVE dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_VECTORTOHIVMAVE_VECTORTOHIVMAVE_H
#define BISHENGIR_CONVERSION_VECTORTOHIVMAVE_VECTORTOHIVMAVE_H

#include <memory>

namespace mlir {
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTVECTORTOHIVMAVE
#include "bishengir/Conversion/Passes.h.inc"

namespace hivmave {
/// Conllect the patterns to convert from the vector dialect to ave dialect.
void populateVectorToHIVMAVEConversionPatterns(RewritePatternSet &patterns);
} // namespace hivmave

/// Creates a pass to convert the Vector dialect to the AVE dialect.
std::unique_ptr<Pass> createVectorToHIVMAVEConversionPass(
    const ConvertVectorToHIVMAVEOptions &options = {});

} // namespace mlir

#endif // BISHENGIR_CONVERSION_VECTORTOHIVMAVE_VECTORTOHIVMAVE_H

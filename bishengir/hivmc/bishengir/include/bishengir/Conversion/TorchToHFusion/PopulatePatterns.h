//===- PopulatePatterns.h -- Populate Torch to HFusion patterns -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_CONVERSION_TORCHTOHFUSION_POPULATEPATTERNS_H
#define BISHENGIR_CONVERSION_TORCHTOHFUSION_POPULATEPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
struct ConvertTorchToHFusionOptions;

// -----------------------------------------------------------------------------
// TorchToNamedOp Conversion Patterns
// -----------------------------------------------------------------------------
void populateElementWisePatternsAndLegality(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            ConversionTarget &target);

void populateReductionPatternsAndLegality(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          ConversionTarget &target);

void populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const ConvertTorchToHFusionOptions &options);

void populateUncategorizedPatternsAndLegality(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns,
                                              ConversionTarget &target);

void populateTensorConstructorsPatternsAndLegality(TypeConverter &typeConverter,
                                                   RewritePatternSet &patterns,
                                                   ConversionTarget &target);
} // namespace mlir

#endif // BISHENGIR_CONVERSION_TORCHTOHFUSION_POPULATEPATTERNS_H

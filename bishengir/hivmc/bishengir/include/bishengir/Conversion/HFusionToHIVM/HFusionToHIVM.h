//===- HFusionToHIVM.h - HFusion to HIVM Conversion Patterns ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert HFusion dialect to HIVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVM_H
#define BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVM_H

#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVMPass.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
class RewritePatternSet;

void populateReductionPatternsAndLegality(RewritePatternSet &patterns,
                                          ConversionTarget &target);

void populateMatmulPatternsAndLegality(
    RewritePatternSet &patterns, ConversionTarget &target,
    const ConvertHFusionToHIVMOptions &options);

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVM_H

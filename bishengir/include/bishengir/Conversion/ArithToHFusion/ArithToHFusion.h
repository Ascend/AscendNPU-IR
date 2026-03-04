//===- ArithToHFusion.h - Arith to HFusion conversion -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_ARITHTOHFUSION_ARITHTOHFUSION_H
#define BISHENGIR_CONVERSION_ARITHTOHFUSION_ARITHTOHFUSION_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"

namespace hfusion {
// Collect a set of patterns to convert Arith dialect ops to Linalg dialect ops.
void populateArithToLinalgConversionPatterns(RewritePatternSet &patterns);
// Collect a set of patterns to convert Arith dialect ops to HFusion dialect
// ops.
void populateArithToHFusionConversionPatterns(RewritePatternSet &patterns);
} // namespace hfusion

/// Creates a pass to convert the HFusion dialect to the HIVM dialect.
std::unique_ptr<Pass> createArithToHFusionConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_ARITHTOHFUSION_ARITHTOHFUSION_H
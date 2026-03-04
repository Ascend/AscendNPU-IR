//===- LinalgToHFusion.h - Linalg to HFusion conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_LINALGTOHFUSION_LINALGTOHFUSION_H
#define BISHENGIR_CONVERSION_LINALGTOHFUSION_LINALGTOHFUSION_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTLINALGTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"

namespace hfusion {
void populateLinalgToHFusionConversionPatterns(RewritePatternSet &patterns);
} // namespace hfusion

/// Creates a pass to convert the HFusion dialect to the HIVM dialect.
std::unique_ptr<Pass> createLinalgToHFusionConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_LINALGTOHFUSION_LINALGTOHFUSION_H
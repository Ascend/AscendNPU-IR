//===- TensorToHFusion.h - Tensor to HFusion conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_TENSORTOHFUSION_TENSORTOHFUSION_H
#define BISHENGIR_CONVERSION_TENSORTOHFUSION_TENSORTOHFUSION_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTTENSORTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"

namespace hfusion {
void populateTensorToHFusionConversionPatterns(RewritePatternSet &patterns);
} // namespace hfusion

/// Creates a pass to convert certain tensor ops to linalg/hfusion ops
std::unique_ptr<Pass> createTensorToHFusionConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_TENSORTOHFUSION_TENSORTOHFUSION_H
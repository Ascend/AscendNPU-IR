//===------- TensorToHIVM.h - Tensor to HIVM conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_TENSORTOHIVM_TENSORTOHIVM_H
#define BISHENGIR_CONVERSION_TENSORTOHIVM_TENSORTOHIVM_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTTENSORTOHIVM
#include "bishengir/Conversion/Passes.h.inc"

namespace hivm {
void populateTensorToHIVMConversionPatterns(RewritePatternSet &patterns);
} // namespace hivm

/// Creates a pass to convert certain tensor ops to hivm ops
std::unique_ptr<Pass> createTensorToHIVMConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_TENSORTOHIVM_TENSORTOHIVM_H
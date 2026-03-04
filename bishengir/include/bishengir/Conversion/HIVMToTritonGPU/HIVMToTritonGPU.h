//===- HIVMToTritonGPU.h - HIVM to TritonGPU conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HIVMTOTRITONGPU_H
#define BISHENGIR_CONVERSION_HIVMTOTRITONGPU_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;
class Type;

#define GEN_PASS_DECL_CONVERTHIVMTOTRITONGPU
#include "bishengir/Conversion/Passes.h.inc"

namespace hivm {
Type HIVMToTritonTypeConvert(Type ty);
void populateHIVMToTritonPatterns(RewritePatternSet &patterns);
void populateFuncToTritonPatterns(RewritePatternSet &patterns);
void populateBufferizationToTritonPatterns(RewritePatternSet &patterns);
void populateHIVMToTensorPatterns(RewritePatternSet &patterns);
void populateReinterpretCastToUnrealizedCastPatterns(RewritePatternSet &patterns);
} // namespace hivm

/// Creates a pass to convert the hivm dialect to the triton dialect.
std::unique_ptr<Pass> createHIVMToTritonGPUConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMTOTRITONGPU_H

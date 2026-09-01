//===- HIVMToLLVM.h - HIVM to LLVM ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the HIVM dialect to the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HIVMTOLLVM_HIVMTOLLVM_H
#define BISHENGIR_CONVERSION_HIVMTOLLVM_HIVMTOLLVM_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTHIVMTOLLVM
#include "bishengir/Conversion/Passes.h.inc"

namespace hivm {
/// Collect the patterns to convert from the HIVM dialect to LLVM.
void populateHIVMToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          bool isRegBased = false);
} // namespace hivm

/// Creates a pass to convert the HIVM dialect into the LLVMIR dialect.
std::unique_ptr<Pass> createConvertHIVMToLLVMPass();

std::unique_ptr<Pass>
createConvertHIVMToLLVMPass(const ConvertHIVMToLLVMOptions &options);

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMTOLLVM_HIVMTOLLVM_H

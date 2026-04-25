//===- HACCToLLVM.h - HACC to LLVM ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the HACC dialect to the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HACCTOLLVM_HACCTOLLVM_H
#define BISHENGIR_CONVERSION_HACCTOLLVM_HACCTOLLVM_H

#include <memory>
#include <string>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTHACCTOLLVM
#include "bishengir/Conversion/Passes.h.inc"

namespace hacc {
/// Collect the patterns to convert from the HACC dialect to LLVM.
void populateHACCToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          const std::string &deviceFilePath);
} // namespace hacc

/// Creates a pass to convert the HACC dialect into the LLVMIR dialect.
std::unique_ptr<Pass> createConvertHACCToLLVMPass();

std::unique_ptr<Pass>
createConvertHACCToLLVMPass(const ConvertHACCToLLVMOptions &option);

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HACCTOLLVM_HACCTOLLVM_H

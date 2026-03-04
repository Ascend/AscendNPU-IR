//===- ArithToHIVMLLVM.h - Arith to HIVM LLVM conversion --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_ARITHTOHIVMLLVM_ARITHTOHIVMLLVM_H
#define BISHENGIR_CONVERSION_ARITHTOHIVMLLVM_ARITHTOHIVMLLVM_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_ARITHTOHIVMLLVMCONVERSIONPASS
#include "bishengir/Conversion/Passes.h.inc"

namespace arith {
void populateArithToHIVMLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);
} // namespace arith

/// Creates a pass to convert the Arith dialect into the LLVMIR dialect with
/// HIVM Target enhancements.
std::unique_ptr<Pass> createArithToHIVMLLVMConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_ARITHTOHIVMLLVM_ARITHTOHIVMLLVM_H

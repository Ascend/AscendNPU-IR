//===- HIVMToMath.h - HIVM to Math ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the HIVM dialect to the Math IR dialect.
//
//===----------------------------------------------------------------------===//
 
#ifndef BISHENGIR_CONVERSION_HIVMTOMATH_HIVMTOMATH_H
#define BISHENGIR_CONVERSION_HIVMTOMATH_HIVMTOMATH_H
 
#include <memory>
 
namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;
 
namespace hivm {
/// Collect the patterns to convert from the HIVM dialect to LLVM.
void populateHIVMToMathConversionPatterns(RewritePatternSet &patterns);
} // namespace hivm
} // namespace mlir
 
#endif // BISHENGIR_CONVERSION_HIVMTOMATH_HIVMTOMATH_H
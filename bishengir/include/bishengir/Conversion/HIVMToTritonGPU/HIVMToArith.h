//===- HIVMToArith.h - HIVM to Arith ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the HIVM dialect to the Arith IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HIVMTOARITH_HIVMTOARITH_H
#define BISHENGIR_CONVERSION_HIVMTOARITH_HIVMTOARITH_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace hivm {
/// Collect the patterns to convert from the HIVM dialect to Arith.
void populateHIVMToArithConversionPatterns(RewritePatternSet &patterns);
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMTOARITH_HIVMTOARITH_H

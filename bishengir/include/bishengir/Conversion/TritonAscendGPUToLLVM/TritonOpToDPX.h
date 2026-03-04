//===--TritonOpToDPX.h - Triton Op to DPX Conversion ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONASCENDGPUTOLLVM_TRITONTODPX_H
#define TRITON_CONVERSION_TRITONASCENDGPUTOLLVM_TRITONTODPX_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
namespace triton {
namespace ascend {

void populateTritonOpToDPXPatterns(LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);

} // namespace ascend
} // namespace triton
} // namespace mlir

#endif
//===--LoadStoreOpToLLVM.h - Load/Store Op to LLVM Conversion ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONASCENDGPUTOLLVM_LOADSTOREOPTOLLVM_H
#define TRITON_CONVERSION_TRITONASCENDGPUTOLLVM_LOADSTOREOPTOLLVM_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
namespace triton {
class ModuleAxisInfoAnalysis;

namespace ascend {
class TargetInfo;

void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);
} // namespace ascend
} // namespace triton
} // namespace mlir

#endif
//===- AppendSanitizerData.cpp - Append and use args ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to append allocated space for sanitizer.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_CONVERSION_HIVMTOLLVM_APPENDSANITIZERDATA_H
#define BISHENGIR_CONVERSION_HIVMTOLLVM_APPENDSANITIZERDATA_H

#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_APPENDSANITIZERDATA
#include "bishengir/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createAppendSanitizerDataPass();
} // namespace mlir

#endif // ISHENGIR_CONVERSION_HIVMTOLLVM_APPENDSANITIZERDATA_H

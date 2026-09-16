//===- AppendUsePrintDebugData.cpp - Append and use args -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to append and use print debug data.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_CONVERSION_HIVMTOLLVM_APPENDUSEPRINTDEBUGDATA_H
#define BISHENGIR_CONVERSION_HIVMTOLLVM_APPENDUSEPRINTDEBUGDATA_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_APPENDUSEPRINTDEBUGDATA
#include "bishengir/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createAppendUsePrintDebugDataPass();
} // namespace mlir

#endif // ISHENGIR_CONVERSION_HIVMTOLLVM_APPENDUSEPRINTDEBUGDATA_H

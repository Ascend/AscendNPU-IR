//===- TritonGlobalKernelArgsToLLVM.h -Replace and eliminate args-*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to replace and eliminate args in LLVM IR dialect.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_CONVERSION_HIVMTOLLVM_TRITONGLOBALKERNELARGSTOLLVM_H
#define BISHENGIR_CONVERSION_HIVMTOLLVM_TRITONGLOBALKERNELARGSTOLLVM_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_TRITONGLOBALKERNELARGSTOLLVM
#include "bishengir/Conversion/Passes.h.inc"
#include "bishengir/Conversion/HIVMToLLVM/MemRefAndTritonGlobalConstants.h"

std::unique_ptr<Pass> createTritonGlobalKernelArgsToLLVMPass();
} // namespace mlir

#endif // ISHENGIR_CONVERSION_HIVMTOLLVM_TRITONGLOBALKERNELARGSTOLLVM_H

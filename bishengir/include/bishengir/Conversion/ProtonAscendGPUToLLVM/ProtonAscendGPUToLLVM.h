//===--ProtonnAscendGPUToLLVM.h - ProtonAscendGPU to LLVM Conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PROTON_CONVERSION_PROTONASCENDGPUTOLLVM_PASSES_H
#define PROTON_CONVERSION_PROTONASCENDGPUTOLLVM_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton::proton::gpu {

std::unique_ptr<Pass> createConvertProtonAscendGPUToLLVMPass();

#define GEN_PASS_DECL_CONVERTPROTONASCENDGPUTOLLVM
#include "bishengir/Conversion/Passes.h.inc"

} // namespace mlir::triton::proton::gpu

#endif

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

namespace bishengir::triton::proton::gpu {

std::unique_ptr<mlir::Pass> createConvertProtonAscendGPUToLLVMPass();
std::unique_ptr<mlir::Pass>
createAllocateProtonAscendGlobalScratchBufferPass();

#define GEN_PASS_DECL_CONVERTPROTONASCENDGPUTOLLVM
#define GEN_PASS_DECL_ALLOCATEPROTONASCENDGLOBALSCRATCHBUFFERPASS
#include "bishengir/Conversion/Passes.h.inc"

} // namespace bishengir::triton::proton::gpu

#endif

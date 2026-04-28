//===- Passes.h - LLVM Pass Construction and Registration -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/LLVMIR/Transforms/Passes.h.inc"

namespace LLVM {
std::unique_ptr<Pass> createParameterPackingPass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/LLVMIR/Transforms/Passes.h.inc"

} // namespace LLVM
} // namespace mlir

#endif // BISHENGIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_H

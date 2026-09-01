//===- All.h - BiShengIR To LLVM IR Translation Registration ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to register the translations of all suitable
// bisheng dialects to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRBISHENGIR_TARGET_LLVMIR_DIALECT_ALL_H
#define MLIRBISHENGIR_TARGET_LLVMIR_DIALECT_ALL_H

#include "bishengir/Target/LLVMIR/Dialect/HACC/HACCToLLVMIRTranslation.h"
#include "bishengir/Target/LLVMIR/Dialect/HIVM/HIVMToLLVMIRTranslation.h"
#include "bishengir/Target/LLVMIR/Dialect/HIVMRegbaseIntrins/HIVMRegbaseIntrinsToLLVMIRTranslation.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace bishengir {
/// Registers all hacc-specific dialects that can be translated to LLVM IR
/// and the corresponding translation interfaces.
static inline void
registerAllToLLVMIRTranslations(mlir::DialectRegistry &registry) {
  mlir::registerHACCDialectTranslation(registry);
  mlir::registerHIVMDialectTranslation(registry);
  mlir::registerHIVMRegbaseIntrinsDialectTranslation(registry);
}

} // namespace bishengir

#endif // MLIRHACC_TARGET_LLVMIR_DIALECT_ALL_H

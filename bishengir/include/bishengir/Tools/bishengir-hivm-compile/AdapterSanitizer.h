//===- AdapterSanitizer.h - Mssanitizer enabling ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines some funcs used to enable mssanitizer
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_MSSANITIZER_H
#define BISHENGIR_TOOLS_MSSANITIZER_H

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

namespace bishengir {
using namespace mlir;
// Enabled when we need sanitizer
// The name of the arg need to be .arg_address_sanitizer_gm_ptr
LogicalResult
setSanitizerAddrArgName(mlir::ModuleOp module,
                        const std::unique_ptr<llvm::Module> &llvmModule);
} // namespace bishengir

#endif // BISHENGIR_TOOLS_MSSANITIZER_H
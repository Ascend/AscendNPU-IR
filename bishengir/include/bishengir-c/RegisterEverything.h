//===-------- RegisterEverything.h - Registration functions -------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_C_REGISTEREVERYTHING_H
#define BISHENGIR_C_REGISTEREVERYTHING_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers all dialects with a context.
/// This is needed before creating IR for these Dialects.
MLIR_CAPI_EXPORTED void bishengirRegisterAllDialects(MlirContext context);

/// Registers all passes for symbolic access with the global registry.
MLIR_CAPI_EXPORTED void bishengirRegisterAllPasses(void);

MLIR_CAPI_EXPORTED void bishengirRegisterAllTranslations(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // BISHENGIR_C_REGISTEREVERYTHING_H

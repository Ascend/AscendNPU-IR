//===--------- HIVMAVE.h - C API for HIVMAVE dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_C_DIALECT_HIVMAVE_H
#define BISHENGIR_C_DIALECT_HIVMAVE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HIVMAVE, hivmave);

#ifdef __cplusplus
}
#endif

#endif // BISHENGIR_C_DIALECT_HIVMAVE_H

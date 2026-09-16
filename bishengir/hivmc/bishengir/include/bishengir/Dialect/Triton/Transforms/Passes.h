//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_TRITON_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_TRITON_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace bishengir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace triton {

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace triton

} // namespace bishengir

#endif // BISHENGIR_DIALECT_TRITON_TRANSFORMS_PASSES_H

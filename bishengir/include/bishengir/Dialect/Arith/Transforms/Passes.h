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
#ifndef BISHENGIR_DIALECT_ARITH_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_ARITH_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Arith/Transforms/Passes.h.inc"

namespace arith {

/// Pass to normalizes arith ops to meet HIVM requirements.
std::unique_ptr<Pass> createNormalizeArithPass();

/// Pass to lift arith.index_cast to simplify vector instr lowering
std::unique_ptr<Pass> createLiftArithIndexCastPass();

/// Move up arith op to prevent fusable operations from being blocked by arith op.
std::unique_ptr<Pass> createMoveUpArithPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace arith
} // namespace mlir

#endif // BISHENGIR_DIALECT_ARITH_TRANSFORMS_PASSES_H
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
#ifndef BISHENGIR_DIALECT_VECTOR_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_VECTOR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Vector/Transforms/Passes.h.inc"

namespace vector {

/// Creates an operation pass to convert `transfer.read` and `transfer.write`
/// to vector.load and vector.write.
std::unique_ptr<Pass> createVectorTransferLoweringPass();

/// Pass to normalizes vector ops to meet HIVM requirements.
std::unique_ptr<Pass> createNormalizeVectorPass();

/// Pass to peel loops which contains `vector.transfer_read` which is transpose
/// load.
std::unique_ptr<Pass> createPeelLoopsContainingTransposePass();

// Create a pass to materialize transfer_write to destination.
std::unique_ptr<Pass> createMaterializeVectorWriteToDestinationPass();

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace vector
} // namespace mlir

#endif // BISHENGIR_DIALECT_VECTOR_TRANSFORMS_PASSES_H
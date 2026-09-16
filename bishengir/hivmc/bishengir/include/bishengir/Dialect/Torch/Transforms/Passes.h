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
#ifndef BISHENGIR_DIALECT_TORCH_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_TORCH_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Torch/Transforms/Passes.h.inc"

namespace torch {

/// Create a pass to canonicalize tensor reshape.
std::unique_ptr<Pass> createLiteralDataTypeCastPass();

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Torch/Transforms/Passes.h.inc"

void registerLiteralDataTypeCast();

} // namespace torch

} // namespace mlir

#endif // BISHENGIR_DIALECT_TORCH_TRANSFORMS_PASSES_H

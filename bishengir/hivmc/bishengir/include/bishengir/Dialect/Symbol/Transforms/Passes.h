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
#ifndef BISHENGIR_DIALECT_SYMBOL_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_SYMBOL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Symbol/Transforms/Passes.h.inc"
namespace symbol {

/// Create a pass to propagate symbols
std::unique_ptr<Pass> createPropagateSymbolPass();

/// Create a pass to erase symbols
std::unique_ptr<Pass> createEraseSymbolPass();

/// Create a pass to convert bind_symbolic_shape to tensor encoding
std::unique_ptr<Pass> createSymbolToEncodingPass();

/// Create a pass to convert tensor encoding to bind_symbolic_shape
std::unique_ptr<Pass> createEncodingToSymbolPass();

/// Create a pass to replace symbol.symbolic_int with tensor.dim
std::unique_ptr<mlir::Pass> createUnfoldSymbolicIntPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Symbol/Transforms/Passes.h.inc"
} // namespace symbol
} // namespace mlir

#endif // BISHENGIR_DIALECT_SYMBOL_TRANSFORMS_PASSES_H

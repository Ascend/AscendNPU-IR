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
#ifndef BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/MemRef/Transforms/Passes.h.inc"

namespace memref {
std::unique_ptr<Pass> createFoldAllocReshapePass();
std::unique_ptr<Pass> createDeadStoreEliminationPass();
std::unique_ptr<Pass> createRemoveRedundantCopyPass();

/// Create a pass to bind buffer according to annotation.
std::unique_ptr<Pass> createBindBufferPass();

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

#endif // BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H

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
#ifndef BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/SCF/Transforms/Passes.h.inc"

namespace scf {

std::unique_ptr<Pass>
createMapForToForallPass(const MapForToForallOptions &options = {});

std::unique_ptr<Pass> createRemoveRedundantLoopInitPass();

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace scf
} // namespace mlir

#endif // BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_H

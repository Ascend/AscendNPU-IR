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
#ifndef BISHENGIR_DIALECT_ASCENDDPX_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_ASCENDDPX_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/AscendDPX/Transforms/Passes.h.inc"

namespace ascend_dpx {

/// Creates wrappers and attributes for SIMT functions
std::unique_ptr<mlir::Pass> createDPXDivOptimizationPass(bishengir::TritonRemapOptions options = {});

/// Hoist ascend_dpx.call_scalar ops from callees to their call sites.
std::unique_ptr<mlir::Pass> createHoistCallScalarToCallerPass();

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/AscendDPX/Transforms/Passes.h.inc"

} // namespace ascend_dpx

} // namespace mlir

#endif // BISHENGIR_DIALECT_ASCENDDPX_TRANSFORMS_PASSES_H

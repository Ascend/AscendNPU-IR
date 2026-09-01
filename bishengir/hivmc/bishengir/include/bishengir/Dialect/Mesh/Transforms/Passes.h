//===- Passes.h - Transform Pass Construction and Registration ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_MESH_TRANSFORMS_PASSES_H
#define BISHENGIR_MESH_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_DECL
#include "bishengir/Dialect/Mesh/Transforms/Passes.h.inc"

namespace mesh {
std::unique_ptr<Pass> createLowerMeshHostPass();

#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Mesh/Transforms/Passes.h.inc"
} // namespace mesh

} // namespace mlir

#endif // BISHENGIR_MESH_TRANSFORMS_PASSES_H

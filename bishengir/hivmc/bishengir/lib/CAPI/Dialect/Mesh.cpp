//===-- Mesh.cpp - C Interface for Mesh dialect -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Pass/Pass.h"

#include "bishengir-c/Dialect/Mesh.h"
#include "bishengir/Dialect/Mesh/Transforms/Passes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Mesh, mesh, mlir::mesh::MeshDialect)

#include "bishengir/Dialect/Mesh/Transforms/Passes.capi.h.inc"

using namespace mlir;
using namespace mlir::mesh;

#ifdef __cplusplus
extern "C" {
#endif

#include "bishengir/Dialect/Mesh/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif

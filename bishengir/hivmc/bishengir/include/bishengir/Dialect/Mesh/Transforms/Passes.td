//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_MESH_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_MESH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerMeshHostPass : Pass<"lower-mesh-host", "ModuleOp"> {
  let summary = "Lower mesh and related ops to calls to proper host functions";
  let constructor = "mlir::mesh::createLowerMeshHostPass()";
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect",
                           "arith::ArithDialect", "LLVM::LLVMDialect"];
}

#endif // BISHENGIR_DIALECT_MESH_TRANSFORMS_PASSES_TD

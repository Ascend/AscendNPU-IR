//===- Passes.td - Vector pass definitions -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_VECTOR_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_VECTOR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def MaterializeVectorWriteToDestination
    : Pass<"materialize-vector-write-to-destination", "func::FuncOp"> {
  let summary =
      "materialize vector transfer-write destination to avoid write to empty tensor";
  let constructor =
      "::mlir::vector::createMaterializeVectorWriteToDestinationPass()";
  let dependentDialects = ["arith::ArithDialect", "scf::SCFDialect",
                           "tensor::TensorDialect", "vector::VectorDialect"];
}

#endif // BISHENGIR_DIALECT_VECTOR_TRANSFORMS_PASSES_TD

//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TORCH_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_TORCH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LiteralDataTypeCast
   : Pass<"literal-data-type-cast", "func::FuncOp"> {
  let summary = "Cast Literal from f64 to f32";
  let constructor = "mlir::torch::createLiteralDataTypeCastPass()";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

#endif

//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ARITH_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_ARITH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def NormalizeArith
    : Pass<"normalize-arith", "func::FuncOp"> {
  let summary = "Normalize arith simd op";
  let constructor = "mlir::arith::createNormalizeArithPass()";
  let dependentDialects = ["arith::ArithDialect", "vector::VectorDialect"];
}

def LiftArithIndexCast : Pass<"lift-arith-indexcast",
                                       "func::FuncOp"> {
  let summary = "Bubble-up arith.index_cast index to i32/...";
  let constructor = "mlir::arith::createLiftArithIndexCastPass()";
  let dependentDialects = ["arith::ArithDialect", "vector::VectorDialect"];
}

def MoveUpArith : Pass<"move-up-arith"> {
  let summary = "Move up arith op";
  let constructor = "mlir::arith::createMoveUpArithPass()";
  let dependentDialects = ["vector::VectorDialect"];
}

#endif // BISHENGIR_DIALECT_ARITH_TRANSFORMS_PASSES_TD

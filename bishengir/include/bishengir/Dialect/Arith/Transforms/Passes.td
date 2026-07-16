//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

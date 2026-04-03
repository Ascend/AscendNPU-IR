//===- Passes.td - Scope pass definition file --------------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_PASSES_TD
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def VFFusion : Pass<"vf-fusion", "mlir::ModuleOp"> {
  let summary = "VFusion Fuse operations";
  let description = [{
    This pass fuses operations based on the fusion kind strategies.

    Example:

    ```mlir
    module {
      func.func @test() {
        <fusible_operations>
        <infusible_operations>
        return
      }
    }
    ```

    will be transformed into:

    ```mlir
    module {
      func.func @test_fused_0() {
        <fusible_operations>
        return
      }
      func.func @test() {
        call @test_fused_0() : () -> ()
        <infusible_operations>
        return
      }
    }
    ```
  }];
  let constructor = "mlir::analysis::createVFFusionPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
  let options =
      [Option<"fusionMode", "fusion-mode", "::mlir::analysis::FusionMode",
              "analysis::FusionMode::MaxParallel",
              "Strategy of fusing operations",
              [{::llvm::cl::values(
               clEnumValN(analysis::FusionMode::AllOp, "all-op",
                          "Fuse all vector operations"),
               clEnumValN(analysis::FusionMode::NMostOp, "n-most-op",
                          "Fuse at most N number of operations"), 
               clEnumValN(analysis::FusionMode::MaxParallel, "max-parallel",
                          "Fuse operations for max parallel"))}]>,
      Option<"enableOutlineCF", "enable-outline-cf", "bool", /*default=*/"false",
              "Enable outlining control-flow operations">,
      Option<"enableOutlineMemref", "enable-outline-memref", "bool", /*default=*/"false",
              "Enable outlining memref operands">,
      Option<"enableOutlineArith", "enable-outline-arith", "bool", /*default=*/"false",
              "Enable outlining arith operations">,
      Option<"enableOutlineCube", "enable-outline-cube", "bool", /*default=*/"false",
              "Enable outlining cube operations">,
      Option<"enableReshapeTiling", "enable-reshape-tiling", "bool", /*default=*/"false",
              "Enable reshape operation fused">,
      ];
}

#endif // BISHENGIR_DIALECT_ANALYSIS_VFFUSION_PASSES_TD

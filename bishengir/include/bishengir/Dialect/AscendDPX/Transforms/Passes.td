//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef BISHENGIR_DIALECT_ASCENDDPX_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_ASCENDDPX_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def DPXDivOptimization : Pass<"dpx-div-optimization", "ModuleOp"> {
    let summary = "Optimize some DPX Divisions and Remainders";
    let constructor = "mlir::ascend_dpx::createDPXDivOptimizationPass()";
    let dependentDialects = [
        "ascend_dpx::AscendDPXDialect", "LLVM::LLVMDialect"
    ];
}
 	 
def HoistCallScalarToCaller
    : Pass<"hoist-call-scalar-to-caller", "ModuleOp"> {
    let summary = "Hoist ascend_dpx.call_scalar ops from callees to call sites";
    let constructor = "mlir::ascend_dpx::createHoistCallScalarToCallerPass()";
    let dependentDialects = ["ascend_dpx::AscendDPXDialect",
                            "LLVM::LLVMDialect"];
}

#endif

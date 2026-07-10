//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

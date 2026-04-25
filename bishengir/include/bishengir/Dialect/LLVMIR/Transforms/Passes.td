//===-- Passes.td - LLVM pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ParameterPacking : Pass<"parameter-packing", "ModuleOp"> {
  let summary = "Pack function parameters into a single pointer.";
  let constructor = "mlir::LLVM::createParameterPackingPass()";
  let dependentDialects = ["LLVM::LLVMDialect",];
}


#endif // BISHENGIR_DIALECT_LLVMIR_TRANSFORMS_PASSES_TD

//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LegalizeLoopIterArgs : Pass<"hivm-legalize-loop-iter-arg"> {
  let summary = "Legalize vector typed iter args.";
  let constructor = "mlir::scf::createLegalizeLoopIterArgsPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "scf::SCFDialect"];
}

def MapForToForall
    : Pass<"map-for-to-forall", "func::FuncOp"> {
  let constructor = "mlir::scf::createMapForToForallPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
  let options = [
    Option<"simtMode", "simt-mode", "bool", "false",
      "Activate for SIMT mode compilation">,];
}

def RemoveRedundantLoopInit
    : Pass<"scf-remove-redundant-loop-init"> {
  let constructor = "mlir::scf::createRemoveRedundantLoopInitPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
}

def CanonicalizeIterArg : Pass<"scf-canonicalize-iter-arg", "func::FuncOp"> {
  let constructor = "mlir::scf::createCanonicalizeIterArgPass()";
  let options = [Option<"enableRemoveDeadIterArgBackwardPatterns",
                        "enable-remove-dead-iter-arg-bwd-patterns", "bool",
                        "true", "Enable RemoveDeadIterArgBackward patterns">];
  let dependentDialects = ["scf::SCFDialect"];
}

#endif // BISHENGIR_DIALECT_SCF_TRANSFORMS_PASSES_TD

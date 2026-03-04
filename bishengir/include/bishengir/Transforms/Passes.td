//===-- Passes.td - transforms pass definition file -----------*-tablegen-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRANSFORMS_PASSES_H
#define BISHENGIR_TRANSFORMS_PASSES_H

include "mlir/Pass/PassBase.td"

def CanonicalizeModule : Pass<"canonicalize-module", "mlir::ModuleOp"> {
  let summary = "CanonicalizeModule operations";
  let description = [{This pass performs canonicalizations on ModuleOp}];
  let constructor = "bishengir::createCanonicalizeModulePass()";
}

def LowerToCPUBackend : Pass<"hivm-lower-to-cpu-backend", "mlir::ModuleOp"> {
  let summary = "lower current context to trace the intrinsics in cpu backend, "
                "which helps to verify correctness of intrinsics.";
  let description = [{
    After hivm-to-std in current pass pipeline, most of module contents are
    general operations which belong to standard dialect, while few are still
    carring hivm special trait, like intrinsic and memory space attribute of
    memref type.

    To enable module could be accepted by cpu backend in next compilation, here
    just eliminate these hivm special traits.
  }];

  let constructor = "bishengir::createLowerToCPUBackendPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "mlir::hivm::HIVMDialect", "mlir::func::FuncDialect",
                           "mlir::LLVM::LLVMDialect"];

  let options = [Option<"enableTritonKernelCompile",
                        "enable-triton-kernel-compile", "bool", "false",
                        "Help to determine source info of launch block dim">];
}

def DeadFunctionElimination : Pass<"dead-func-elimination", "mlir::ModuleOp"> {
  let summary = "Eliminate functions that are dead.";
  let description = [{
    Remove a function-like op that is known to have no use in the module.

    Note: in comparison to the SymbolDCE pass, this pass:
      1. Only targets function-like ops;
      2. Is more agressive. It will remove function's that are public;
      3. Provids a mechanism to specify which functions to remove.
  }];
  let constructor = "bishengir::createDeadFunctionEliminationPass()";
}

#endif //BISHENGIR_TRANSFORMS_PASSES_H

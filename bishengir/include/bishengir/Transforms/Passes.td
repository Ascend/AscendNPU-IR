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

def InjectIR : Pass<"inject-ir", "mlir::ModuleOp"> {
  let summary = "Load IR from file and replace matching functions for debug.";
  let description = [{
    When a file path is set via option (e.g. inject-ir{file-path=...} or
    compile option --inject-ir-from-file), this pass loads the module from
    that file and replaces the body of each function in the current module
    that has the same name as a function in the loaded module. Used for
    debugging the pass pipeline by injecting IR from a previously dumped file.
  }];
  let constructor = "bishengir::createInjectIRPass()";
  let options = [Option<"filePath", "file-path", "std::string", "\"\"",
                        "Path to IR file; functions from this file replace "
                        "matching functions in the current module">];
}


def FuseReductionIntoLoop : Pass<"fuse-reduction-into-loop",
                                  "mlir::func::FuncOp"> {
  let summary = "Fuse post-loop reductions into the loop body";
  let description = [{
    This pass identifies linalg.reduce operations that consume scf.for loop
    results and fuses them into the loop body.  When a loop accumulates full
    (e.g. 2-D) tensors via arith.addf and a subsequent linalg.reduce collapses
    the accumulated result to a lower-rank tensor, this pass moves the
    reduction inside the loop so that the loop carries the smaller accumulator
    directly.  This reduces on-chip buffer pressure and improves data locality.
  }];
  let constructor = "bishengir::createFuseReductionIntoLoopPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def EliminateSingleIterationScfFor
    : Pass<"eliminate-single-iteration-scf-for", "mlir::ModuleOp"> {
  let summary = "Eliminate scf.for loops that iterate exactly once";
  let description = [{
    Removes each `scf.for` that is proven to run exactly one iteration.

    Proof strategies (see `AffineMinMaxValueBoundsCollector`):
    - Trip count from constant or ValueBounds-collapsed lower/upper/step, or
      a constant induction variable (same as the former standalone pass).
    - Header bounds refined using intervals recorded for `affine.min` /
      `affine.max` results (including discrete enumeration over a constant
      `scf.for` IV when needed).

    The loop body is cloned once with the induction variable and `iter_args`
    mapped to header values; the `scf.for` is replaced by the body's yield.
  }];
  let constructor = "bishengir::createEliminateSingleIterationScfForPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "scf::SCFDialect",
  ];
  let options = [
    Option<"dumpAffineMinmaxBounds",
           "dump-affine-minmax-bounds", "bool", "false",
           "If true, dump AffineMinMaxValueBoundsCollector value2bounds to "
           "stderr in a stable format (for lit/FileCheck).">
  ];
}

#endif //BISHENGIR_TRANSFORMS_PASSES_H

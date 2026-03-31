//===------------- Passes.h - Pass ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the
// bishengir transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRANSFORMS_PASSES_H
#define BISHENGIR_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace bishengir {
#define GEN_PASS_DECL
#include "bishengir/Transforms/Passes.h.inc"

/// Create a pass to canonicalize modules.
std::unique_ptr<mlir::Pass> createCanonicalizeModulePass();

/// Create a pass to lower bishengir to cpu backend.
std::unique_ptr<mlir::Pass>
createLowerToCPUBackendPass(const LowerToCPUBackendOptions &options = {});

// Options struct for DeadEmptyFunctionElimination pass.
// Note: defined only here, not in tablegen.
struct DeadFunctionEliminationOptions {
  // Filter function; returns true if the function should be considered for
  // removal. Defaults to true, i.e. all applicable functions are removed.
  llvm::function_ref<bool(mlir::FunctionOpInterface)> filterFn =
      [](mlir::FunctionOpInterface func) { return true; };
};

/// Create a pass to eliminate dead function.
std::unique_ptr<mlir::Pass> createDeadFunctionEliminationPass(
    const DeadFunctionEliminationOptions &options = {});

/// Eliminate functions that are known to be dead.
void eliminateDeadFunctions(mlir::ModuleOp module,
                            const DeadFunctionEliminationOptions &options);

/// Create InjectIR pass (load IR from file and replace matching functions).
std::unique_ptr<mlir::Pass> createInjectIRPass(llvm::StringRef filePath = "");

/// Create a pass to fuse post-loop reductions into the loop body.
std::unique_ptr<mlir::Pass> createFuseReductionIntoLoopPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.

#define GEN_PASS_REGISTRATION
#include "bishengir/Transforms/Passes.h.inc"

} // namespace bishengir

#endif // BISHENGIR_TRANSFORMS_PASSES_H

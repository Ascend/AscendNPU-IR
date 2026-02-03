//===------------- Passes.h - Pass ------------------------------*- C++ -*-===//
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
#include "mlir/Transforms/Passes.h"

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

/// Create a pass for external canonicalization
std::unique_ptr<mlir::Pass>
createExtendedCanonicalizerPass(const mlir::CanonicalizerOptions &options = {});

/// Create InjectIR pass (load IR from file and replace matching functions).
std::unique_ptr<mlir::Pass> createInjectIRPass(llvm::StringRef filePath = "");

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.

#define GEN_PASS_REGISTRATION
#include "bishengir/Transforms/Passes.h.inc"

} // namespace bishengir

#endif // BISHENGIR_TRANSFORMS_PASSES_H

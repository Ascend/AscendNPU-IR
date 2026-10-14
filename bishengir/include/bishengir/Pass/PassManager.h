//===- PassManager.h - Pass Management Interface ----------------*- C++ -*-===//
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

#ifndef BISHENGIR_PASS_PASSMANAGER_H
#define BISHENGIR_PASS_PASSMANAGER_H

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Pass/PassExecutionPolicy.h"
#include "bishengir/Tools/BiShengIRConfigBase/Config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>
#include <memory>

namespace bishengir {
/// Register a set of useful command-line options that can be used to configure
/// a pass manager. The values of these options can be applied via the
/// 'applyPassManagerCLOptions' method below.
void registerPassManagerCLOptions();

/// Apply any values provided to the pass manager options that were registered
/// with 'registerPassManagerOptions'.
llvm::LogicalResult applyPassManagerCLOptions(mlir::PassManager &pm);

// A pass manager that allows filtering the passes before running. It's more
// expensive to use with compared to mlir::PassManager.
//
// Both constructors register the context action handler that implements the
// filtering, so all pipeline entry points (bishengir-compile regbase,
// RetriablePassManager, hivmc runPipeline, unit tests) get the same behavior.
// The base-class constructors are intentionally not inherited: doing so
// provided a construction path that bypassed the handler and silently ran
// unfiltered passes.
class BiShengIRPassManager : public mlir::PassManager {
public:
  BiShengIRCompileConfigBase config;

  BiShengIRPassManager(const BiShengIRCompileConfigBase &config,
                       mlir::MLIRContext *ctx, llvm::StringRef operationName,
                       Nesting nesting)
      : PassManager(ctx, operationName, nesting), config(config) {
    initializeActionHandler(ctx);
  }

  // Constructor for the config-less construction paths (e.g. the hivmc
  // runPipeline entry point and the unit tests). It registers the action
  // handler exactly like the config-carrying constructor above, so those paths
  // do not silently run unfiltered passes.
  BiShengIRPassManager(
      mlir::MLIRContext *ctx,
      llvm::StringRef operationName = mlir::PassManager::getAnyOpAnchorName(),
      Nesting nesting = Nesting::Explicit)
      : PassManager(ctx, operationName, nesting) {
    initializeActionHandler(ctx);
  }

  ~BiShengIRPassManager();

#if MLIR_ENABLE_EXECUTION_ENGINE
  mlir::LogicalResult run(mlir::Operation *op);

private:
  void filterCPURunnerPasses(mlir::OpPassManager &originalPM);
#endif // MLIR_ENABLE_EXECUTION_ENGINE

private:
  /// State shared between this manager and the context action handler it
  /// registers. The MLIRContext keeps only the last registered handler and
  /// that handler can outlive the manager, so the state is reference-counted
  /// by the handler, and `managerAlive` is cleared in the destructor: a stale
  /// handler then skips the (per-run) Mix/Aiv filtering instead of reading
  /// destroyed state, while the context-global per-op FilterPassesAttr
  /// filtering keeps behaving exactly as before.
  struct PolicyState {
    PassExecutionPolicy policy;
    std::atomic<bool> managerAlive{true};
  };
  std::shared_ptr<PolicyState> policyState = std::make_shared<PolicyState>();

  /// Registers the context action handler implementing the pass filtering for
  /// this manager. It must not capture `this`, only `policyState`.
  ///
  /// Order of checks for a PassExecutionAction:
  ///   1. Aiv-mode global denylist (PassExecutionPolicy::shouldSkipGlobally)
  ///   2. Per-op annotation.filter_passes (FilterPassesAttr) whitelist
  /// Both are bypassed for adaptor passes (empty pass argument).
  void initializeActionHandler(mlir::MLIRContext *ctx);
};

} // namespace bishengir

#endif // BISHENGIR_PASS_PASSMANAGER_H

//===- RetriablePassManager.h - Retriable pass manager ----------*- C++ -*-===//
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

#ifndef BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_RETRIABLEPASSMANAGER_H
#define BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_RETRIABLEPASSMANAGER_H

#include "bishengir/Tools/RetriablePassManager/RetryPolicy.h"
#include "bishengir/Tools/bishengir-compile/Config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bishengir {

/// Option changed during a retriable pipeline retry, with the policy's
/// user-visible reason (from \c RetryPolicy::userVisibleRetryCause()).
struct AppliedCompileFallback {
  std::string retryOption;
  std::string retryValue;
  std::string retryCause;
};

/// Runs a BiShengIR pass pipeline with optional retry policies (e.g. UB overflow
/// fallback). Rebuilds the pass manager on each attempt and clones the module
/// before every run.
class RetriablePassManager {
public:
  using BuildPipelineFn = std::function<void(mlir::PassManager &)>;

  RetriablePassManager(BiShengIRCompileMainConfig &config, mlir::MLIRContext *ctx);

  /// Register a retry policy. Policies are tried in registration order on
  /// failure; the first policy that returns a recovery option triggers a retry.
  void addPolicy(std::unique_ptr<RetryPolicy> policy);

  /// Build a fresh BiShengIRPassManager, run buildPipeline, and execute it on
  /// mod in place. Does not clone mod and does not invoke retry policies.
  mlir::LogicalResult runOnce(mlir::ModuleOp mod, const BuildPipelineFn &buildPipeline,
                              llvm::StringRef pipelineName);

  /// On each attempt, clone mod and run the pipeline via runOnce. If a
  /// registered policy handles the failure, update config and retry. On success,
  /// replace mod with the cloned result. Appends to appliedFallbacks and
  /// records new diagnostics in collectedDiagnostics (diagnostics from
  /// superseded failed attempts may be removed before retry).
  mlir::LogicalResult
  runWithRetry(mlir::ModuleOp &mod, const BuildPipelineFn &buildPipeline,
               llvm::StringRef pipelineName,
               std::vector<std::unique_ptr<mlir::Diagnostic>> &collectedDiagnostics,
               std::vector<AppliedCompileFallback> &appliedFallbacks);

  /// Print a user-visible note for a pipeline retry (e.g. "set X to Y").
  static void emitFallbackNote(llvm::StringRef retryCause,
                               llvm::StringRef retryOption,
                               llvm::StringRef retryValue);

  /// Print a summary of all options changed during this compile after retry
  /// fallbacks. No output if appliedFallbacks is empty.
  static void emitFallbackSummary(llvm::ArrayRef<AppliedCompileFallback> appliedFallbacks,
                                  bool compilationSucceeded);

private:
  BiShengIRCompileMainConfig &config;
  mlir::MLIRContext *context;
  llvm::SmallVector<std::unique_ptr<RetryPolicy>> policies;
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_RETRIABLEPASSMANAGER_H

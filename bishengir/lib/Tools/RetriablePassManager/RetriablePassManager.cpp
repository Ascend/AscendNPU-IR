//===- RetriablePassManager.cpp - Retriable pass manager --------*- C++ -*-===//
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

#include "bishengir/Tools/RetriablePassManager/RetriablePassManager.h"

#include "bishengir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bishengir-retriable-pass-manager"

namespace {
void printSetOptionAction(llvm::raw_ostream &os, llvm::StringRef option,
                          llvm::StringRef value) {
  os << "set " << option << " to " << value;
}
} // namespace

using namespace mlir;
using namespace bishengir;

RetriablePassManager::RetriablePassManager(BiShengIRCompileMainConfig &config,
                                           MLIRContext *ctx)
    : config(config), context(ctx) {}

void RetriablePassManager::addPolicy(std::unique_ptr<RetryPolicy> policy) {
  policies.push_back(std::move(policy));
}

void RetriablePassManager::emitFallbackNote(llvm::StringRef retryCause,
                                            llvm::StringRef retryOption,
                                            llvm::StringRef retryValue) {
  llvm::errs() << "[NOTE] " << retryCause << " detected; automatically ";
  printSetOptionAction(llvm::errs(), retryOption, retryValue);
  llvm::errs() << " and retrying compilation.\n";
}

void RetriablePassManager::emitFallbackSummary(
    llvm::ArrayRef<AppliedCompileFallback> appliedFallbacks,
    bool compilationSucceeded) {
  if (appliedFallbacks.empty())
    return;

  const AppliedCompileFallback &first = appliedFallbacks.front();
  const bool allSameCause = llvm::all_of(appliedFallbacks, [&](const AppliedCompileFallback &fb) {
    return fb.retryCause == first.retryCause;
  });

  llvm::errs() << "[NOTE] ";
  if (allSameCause) {
    llvm::errs() << "Due to " << first.retryCause << ", automatically ";
    for (size_t i = 0, e = appliedFallbacks.size(); i != e; ++i) {
      if (i != 0)
        llvm::errs() << ", ";
      printSetOptionAction(llvm::errs(), appliedFallbacks[i].retryOption,
                           appliedFallbacks[i].retryValue);
    }
  } else {
    llvm::errs() << "Automatically ";
    for (size_t i = 0, e = appliedFallbacks.size(); i != e; ++i) {
      if (i != 0)
        llvm::errs() << ", ";
      const AppliedCompileFallback &fb = appliedFallbacks[i];
      printSetOptionAction(llvm::errs(), fb.retryOption, fb.retryValue);
      llvm::errs() << " (due to " << fb.retryCause << ")";
    }
  }
  if (compilationSucceeded)
    llvm::errs() << "; compilation then succeeded.\n";
  else
    llvm::errs() << "; compilation still failed.\n";
}

LogicalResult RetriablePassManager::runOnce(ModuleOp mod,
                                            const BuildPipelineFn &buildPipeline,
                                            StringRef pipelineName) {
  BiShengIRPassManager passManager(config, context,
                                   ModuleOp::getOperationName(),
                                   OpPassManager::Nesting::Implicit);
  buildPipeline(passManager);

  (void)mlir::applyPassManagerCLOptions(passManager);
  (void)bishengir::applyPassManagerCLOptions(passManager);

  if (failed(passManager.run(mod)))
    return mod->emitError("Failed to run " + pipelineName.str() + " pipeline\n");

  return success();
}

LogicalResult RetriablePassManager::runWithRetry(
    ModuleOp &mod, const BuildPipelineFn &buildPipeline, StringRef pipelineName,
    std::vector<std::unique_ptr<Diagnostic>> &collectedDiagnostics,
    std::vector<AppliedCompileFallback> &appliedFallbacks) {
  while (true) {
    for (const std::unique_ptr<RetryPolicy> &policy : policies)
      policy->onBeforePipelineAttempt();

    const size_t pipelineAttemptDiagStart = collectedDiagnostics.size();
    OwningOpRef<ModuleOp> attempt = cast<ModuleOp>(mod->clone());
    ModuleOp attemptModule = attempt.get();

    if (succeeded(runOnce(attemptModule, buildPipeline, pipelineName))) {
      collectedDiagnostics.resize(pipelineAttemptDiagStart);
      mod.erase();
      mod = attemptModule;
      attempt.release();
      return success();
    }

    if (policies.empty())
      return failure();

    llvm::ArrayRef<std::unique_ptr<Diagnostic>> attemptDiagnostics(
        collectedDiagnostics.data() + pipelineAttemptDiagStart,
        collectedDiagnostics.size() - pipelineAttemptDiagStart);

    RetryPolicy *matchedPolicy = nullptr;
    std::optional<RetryRecoveryAction> retryAction;
    for (const std::unique_ptr<RetryPolicy> &policy : policies) {
      if (std::optional<RetryRecoveryAction> action =
              policy->onFailure(attemptDiagnostics, config)) {
        retryAction = std::move(action);
        matchedPolicy = policy.get();
        break;
      }
    }

    if (!retryAction || !matchedPolicy)
      return failure();

    LLVM_DEBUG({
      llvm::dbgs() << "[BiShengHIR] Pipeline retry ("
                   << matchedPolicy->userVisibleRetryCause() << "): ";
      printSetOptionAction(llvm::dbgs(), retryAction->option, retryAction->value);
      llvm::dbgs() << " and retrying\n";
    });
    if (matchedPolicy->recordsCompileFallback()) {
      if (llvm::none_of(appliedFallbacks, [&](const AppliedCompileFallback &a) {
            return a.retryOption == retryAction->option;
          }))
        appliedFallbacks.push_back(
            {retryAction->option, retryAction->value,
             std::string(matchedPolicy->userVisibleRetryCause())});
      emitFallbackNote(matchedPolicy->userVisibleRetryCause(), retryAction->option,
                       retryAction->value);
    }

    collectedDiagnostics.resize(pipelineAttemptDiagStart);
  }
}

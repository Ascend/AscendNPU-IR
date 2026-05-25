//===- TuningRetryPolicy.cpp - HFusion buffer tuning retry ------*- C++ -*-===//
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

#include "bishengir/Tools/RetriablePassManager/TuningRetryPolicy.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <string>

#define DEBUG_TYPE "bishengir-tuning-retry-policy"

using namespace bishengir;

namespace {
constexpr int64_t kBufferCountTuningDelta = 2;
constexpr llvm::StringLiteral kHfusionMaxBufferCountTuningOption =
    "hfusion-max-buffer-count-tuning";
} // namespace

TuningRetryPolicy::TuningRetryPolicy() {
  auto &opts = llvm::cl::getRegisteredOptions();
  if (opts.count("mlir-print-ir-after-failure") == 0)
    return;

  restorePrintIrAfterFailureOnLastAttempt_ =
      static_cast<llvm::cl::opt<bool> *>(opts["mlir-print-ir-after-failure"])
          ->getValue();
  static_cast<llvm::cl::opt<bool> *>(opts["mlir-print-ir-after-failure"])
      ->setValue(false);
}

void TuningRetryPolicy::onBeforePipelineAttempt() const {
  if (!restorePrintIrAfterFailureOnLastAttempt_)
    return;

  auto &opts = llvm::cl::getRegisteredOptions();
  if (opts.count("mlir-print-ir-after-failure") == 0)
    return;

  const bool isLastTuningAttempt =
      tuningRetriesUsed_ >= TuningRetryPolicy::kMaxAttempts - 1;
  static_cast<llvm::cl::opt<bool> *>(opts["mlir-print-ir-after-failure"])
      ->setValue(isLastTuningAttempt);
}

std::optional<RetryRecoveryAction> TuningRetryPolicy::onFailure(
    llvm::ArrayRef<std::unique_ptr<mlir::Diagnostic>> attemptDiagnostics,
    BiShengIRCompileMainConfig &config) {
  (void)attemptDiagnostics;

  if (tuningRetriesUsed_ + 1 >= kMaxAttempts)
    return std::nullopt;

  config.increaseMaxBufferCountTuning(kBufferCountTuningDelta);
  ++tuningRetriesUsed_;

  LLVM_DEBUG(llvm::dbgs()
             << "[BiShengHIR] HFusion buffer tuning retry "
             << tuningRetriesUsed_ << "/" << kMaxAttempts
             << ", set " << kHfusionMaxBufferCountTuningOption << " to "
             << config.getHfusionMaxBufferCountTuning() << "\n");

  return RetryRecoveryAction{
      std::string(kHfusionMaxBufferCountTuningOption),
      std::to_string(config.getHfusionMaxBufferCountTuning())};
}

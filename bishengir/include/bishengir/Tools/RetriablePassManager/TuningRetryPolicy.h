//===- TuningRetryPolicy.h - HFusion buffer tuning retry --------*- C++ -*-===//
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

#ifndef BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_TUNINGRETRYPOLICY_H
#define BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_TUNINGRETRYPOLICY_H

#include "bishengir/Tools/RetriablePassManager/RetryPolicy.h"

namespace bishengir {

/// Retry policy for HFusion max-buffer-count tuning (up to five attempts).
/// Register only when \c enable-tuning-mode is true and
/// \c enable-triton-kernel-compile is false.
class TuningRetryPolicy : public RetryPolicy {
public:
  static constexpr int kMaxAttempts = 5;

  TuningRetryPolicy();

  llvm::StringRef userVisibleRetryCause() const override {
    return "HFusion buffer tuning";
  }

  bool recordsCompileFallback() const override { return false; }

  void onBeforePipelineAttempt() const override;

  std::optional<RetryRecoveryAction> onFailure(
      llvm::ArrayRef<std::unique_ptr<mlir::Diagnostic>> attemptDiagnostics,
      BiShengIRCompileMainConfig &config) override;

private:
  int tuningRetriesUsed_ = 0;
  bool restorePrintIrAfterFailureOnLastAttempt_ = false;
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_TUNINGRETRYPOLICY_H

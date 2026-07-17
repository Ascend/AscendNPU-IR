//===- TuningRetryPolicy.h - Retry tuning policy ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_TUNINGRETRYPOLICY_H
#define BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_TUNINGRETRYPOLICY_H

#include "llvm/Support/CommandLine.h"

namespace bishengir {

class TuningRetryPolicy {
public:
  TuningRetryPolicy();
  ~TuningRetryPolicy();

  TuningRetryPolicy(const TuningRetryPolicy &) = delete;
  TuningRetryPolicy &operator=(const TuningRetryPolicy &) = delete;

  void onBeforePipelineAttempt(bool isLastAttempt);

private:
  llvm::cl::opt<bool> *printIrAfterFailureOption = nullptr;
  bool originalPrintIrAfterFailure = false;
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_TUNINGRETRYPOLICY_H

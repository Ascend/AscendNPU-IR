//===- TuningRetryPolicy.cpp - Retry tuning policy --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/RetriablePassManager/TuningRetryPolicy.h"

using namespace bishengir;

TuningRetryPolicy::TuningRetryPolicy() {
  auto &opts = llvm::cl::getRegisteredOptions();
  auto it = opts.find("mlir-print-ir-after-failure");
  if (it == opts.end())
    return;

  printIrAfterFailureOption = static_cast<llvm::cl::opt<bool> *>(it->second);
  originalPrintIrAfterFailure = printIrAfterFailureOption->getValue();
  if (originalPrintIrAfterFailure)
    printIrAfterFailureOption->setValue(false);
}

TuningRetryPolicy::~TuningRetryPolicy() {
  if (printIrAfterFailureOption)
    printIrAfterFailureOption->setValue(originalPrintIrAfterFailure);
}

void TuningRetryPolicy::onBeforePipelineAttempt(bool isLastAttempt) {
  if (printIrAfterFailureOption && originalPrintIrAfterFailure)
    printIrAfterFailureOption->setValue(isLastAttempt);
}

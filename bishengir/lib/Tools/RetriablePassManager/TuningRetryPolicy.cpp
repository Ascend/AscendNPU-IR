//===- TuningRetryPolicy.cpp - Retry tuning policy --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/RetriablePassManager/TuningRetryPolicy.h"

#include "llvm/Support/CommandLine.h"

using namespace bishengir;

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

TuningRetryPolicy::~TuningRetryPolicy() {
  auto &opts = llvm::cl::getRegisteredOptions();
  if (opts.count("mlir-print-ir-after-failure") == 0)
    return;

  static_cast<llvm::cl::opt<bool> *>(opts["mlir-print-ir-after-failure"])
      ->setValue(restorePrintIrAfterFailureOnLastAttempt_);
}

void TuningRetryPolicy::onBeforePipelineAttempt(bool isLastAttempt) {
  if (!restorePrintIrAfterFailureOnLastAttempt_)
    return;

  auto &opts = llvm::cl::getRegisteredOptions();
  if (opts.count("mlir-print-ir-after-failure") == 0)
    return;

  static_cast<llvm::cl::opt<bool> *>(opts["mlir-print-ir-after-failure"])
      ->setValue(isLastAttempt);
}

//===- OverFlowPolicyBase.cpp - Memory overflow retry base ------*- C++ -*-===//
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

#include "bishengir/Tools/RetriablePassManager/OverFlowPolicyBase.h"

#include "bishengir/Tools/RetriablePassManager/DiagnosticUtils.h"
#include "bishengir/Tools/bishengir-compile/Config.h"

using namespace mlir;
using namespace bishengir;

std::optional<RetryRecoveryAction> OverFlowPolicyBase::onFailure(
    llvm::ArrayRef<std::unique_ptr<Diagnostic>> attemptDiagnostics,
    BiShengIRCompileMainConfig &config) {
  if (!bishengir::diagnosticsContainOverflowMessage(attemptDiagnostics,
                                         overflowDiagnosticPattern()))
    return std::nullopt;

  if (config.getEnableCodeMotion()) {
    config.setEnableCodeMotion(false);
    return RetryRecoveryAction{"enable-code-motion", "false"};
  }

  if (config.getEnableAutoMultiBuffer()) {
    config.setEnableAutoMultiBuffer(false);
    return RetryRecoveryAction{"enable-auto-multi-buffer", "false"};
  }

  return std::nullopt;
}

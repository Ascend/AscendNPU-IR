//===- OverFlowPolicyBase.h - Memory overflow retry base --------*- C++ -*-===//
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

#ifndef BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_OVERFLOWPOLICYBASE_H
#define BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_OVERFLOWPOLICYBASE_H

#include "bishengir/Tools/RetriablePassManager/RetryPolicy.h"

namespace bishengir {

/// Base retry policy for plan-memory scope overflow failures. Sets
/// enable-code-motion to false first, then enable-auto-multi-buffer to false.
/// No retry if those options are already false.
class OverFlowPolicyBase : public RetryPolicy {
public:
  std::optional<RetryRecoveryAction> onFailure(
      llvm::ArrayRef<std::unique_ptr<mlir::Diagnostic>> attemptDiagnostics,
      BiShengIRCompileMainConfig &config) override;

protected:
  virtual llvm::StringRef overflowDiagnosticPattern() const = 0;
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_OVERFLOWPOLICYBASE_H

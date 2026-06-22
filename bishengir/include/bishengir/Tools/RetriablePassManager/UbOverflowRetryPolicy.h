//===- UbOverflowRetryPolicy.h - UB overflow retry policy -----*- C++ -*-===//
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

#ifndef BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_UBOVERFLOWRETRYPOLICY_H
#define BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_UBOVERFLOWRETRYPOLICY_H

#include "bishengir/Tools/RetriablePassManager/OverFlowPolicyBase.h"

namespace bishengir {

/// Retry policy for UB scope overflow. Register only when
/// enable-triton-kernel-compile is true.
class UbOverflowRetryPolicy : public OverFlowPolicyBase {
public:
  llvm::StringRef userVisibleRetryCause() const override { return "Ub overflow"; }

protected:
  llvm::StringRef overflowDiagnosticPattern() const override {
    return "ub overflow";
  }
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_UBOVERFLOWRETRYPOLICY_H

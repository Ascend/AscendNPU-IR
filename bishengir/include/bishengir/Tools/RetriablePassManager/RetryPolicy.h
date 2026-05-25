//===- RetryPolicy.h - Pass pipeline retry policies -------------*- C++ -*-===//
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

#ifndef BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_RETRYPOLICY_H
#define BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_RETRYPOLICY_H

#include "bishengir/Tools/bishengir-compile/Config.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
#include <string>

namespace bishengir {

/// Config change applied by a retry policy after a failed pipeline attempt.
struct RetryRecoveryAction {
  std::string option;
  std::string value;
};

/// Invoked when a pipeline attempt fails. Returns the config change applied for
/// retry, or std::nullopt if this policy does not apply or cannot retry further.
class RetryPolicy {
public:
  virtual ~RetryPolicy() = default;

  /// Short user-visible label for why a retry ran (e.g. \c "UB overflow").
  /// Used in messages like "<label> detected" and "Due to <label>, ...".
  virtual llvm::StringRef userVisibleRetryCause() const {
    return "Compile error";
  }

  /// If false, onFailure may still trigger a retry but no fallback note or
  /// summary entry is recorded (e.g. tuning retries).
  virtual bool recordsCompileFallback() const { return true; }

  /// Called before each pipeline clone/run attempt.
  virtual void onBeforePipelineAttempt() const {}

  virtual std::optional<RetryRecoveryAction> onFailure(
      llvm::ArrayRef<std::unique_ptr<mlir::Diagnostic>> attemptDiagnostics,
      BiShengIRCompileMainConfig &config) = 0;
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_RETRYPOLICY_H

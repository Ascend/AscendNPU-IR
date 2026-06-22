//===- DiagnosticUtils.h - Diagnostic helpers -------------------*- C++ -*-===//
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

#ifndef BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_DIAGNOSTICUTILS_H
#define BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_DIAGNOSTICUTILS_H

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>

namespace bishengir {

/// Returns true if diag or any attached note contains pattern
/// (case-insensitive) in its message text.
bool diagnosticTreeContainsOverflowMessage(const mlir::Diagnostic &diag,
                                           llvm::StringRef pattern);

/// Returns true if any diagnostic in diagnostics matches
/// diagnosticTreeContainsOverflowMessage.
bool diagnosticsContainOverflowMessage(
    llvm::ArrayRef<std::unique_ptr<mlir::Diagnostic>> diagnostics,
    llvm::StringRef pattern);

/// Returns true if any diagnostic in diagnostics at or after startIndex
/// matches diagnosticTreeContainsOverflowMessage. Used to scope checks to a
/// single pipeline attempt when diagnostics are accumulated across retries.
bool diagnosticsSinceContainOverflowMessage(
    const std::vector<std::unique_ptr<mlir::Diagnostic>> &diagnostics,
    size_t startIndex, llvm::StringRef pattern);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_RETRIABLEPASSMANAGER_DIAGNOSTICUTILS_H

//===- DiagnosticUtils.cpp - Diagnostic helpers -----------------*- C++ -*-===//
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

#include "bishengir/Tools/RetriablePassManager/DiagnosticUtils.h"

#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace bishengir;

namespace {

static bool containsOverflowMessage(llvm::StringRef text,
                                    llvm::StringRef pattern) {
  std::string lowerCopy = text.lower();
  return llvm::StringRef(lowerCopy).contains(pattern.lower());
}

} // namespace

bool bishengir::diagnosticTreeContainsOverflowMessage(const Diagnostic &diag,
                                                      llvm::StringRef pattern) {
  if (containsOverflowMessage(diag.str(), pattern))
    return true;
  for (const Diagnostic &note : diag.getNotes()) {
    if (diagnosticTreeContainsOverflowMessage(note, pattern))
      return true;
  }
  return false;
}

bool bishengir::diagnosticsContainOverflowMessage(
    llvm::ArrayRef<std::unique_ptr<Diagnostic>> diagnostics,
    llvm::StringRef pattern) {
  for (const std::unique_ptr<Diagnostic> &diag : diagnostics) {
    if (diagnosticTreeContainsOverflowMessage(*diag, pattern))
      return true;
  }
  return false;
}

bool bishengir::diagnosticsSinceContainOverflowMessage(
    const std::vector<std::unique_ptr<Diagnostic>> &diagnostics,
    size_t startIndex, llvm::StringRef pattern) {
  for (size_t i = startIndex, e = diagnostics.size(); i != e; ++i) {
    if (diagnosticTreeContainsOverflowMessage(*diagnostics[i], pattern))
      return true;
  }
  return false;
}

//===-- Version.h - BiShengIR Version Number --------------------*- C++ -*-===//
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
//
// This file defines version macros and version-related utility functions
// for BiShengIR.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_VERSION_VERSION_H
#define BISHENGIR_VERSION_VERSION_H

#include "llvm/ADT/StringRef.h"

namespace bishengir {

/// Retrieves the repository path (e.g., Subversion path) that
/// identifies the particular BiShengIR branch, tag, or trunk from which this
/// BiShengIR was built.
std::string getBiShengIRRepositoryPath();

/// Retrieves the repository path from which LLVM was built.
///
/// This supports LLVM residing in a separate repository from BiShengIR.
std::string getLLVMRepositoryPath();

/// Retrieves the repository revision number (or identifier) from which
/// this BiShengIR was built.
std::string getBiShengIRRevision();

/// Retrieves the repository revision number (or identifier) from which
/// LLVM was built.
///
/// If BiShengIR and LLVM are in the same repository, this returns the same
/// string as getBiShengIRRevision.
std::string getLLVMRevision();

/// Retrieves the BiShengIR vendor tag.
std::string getBiShengIRVendor();

/// Retrieves a string representing the complete bishengir version,
/// which includes the bishengir version number, the repository version,
/// and the vendor tag.
std::string getBiShengIRFullVersion();

/// return a tup contains (commit_hash, date)
std::string getBiShengIRFullRepositoryVersion();

/// Like getBiShengIRFullVersion(), but with a custom tool name.
std::string getBiShengIRToolFullVersion(llvm::StringRef toolName);

} // namespace bishengir

#endif // BISHENGIR_VERSION_VERSION_H

//===- AuxBitcodePaths.h - Aux bitcode path resolution ----------*- C++ -*-===//
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
// Path-only resolution of the aicore base-aux bitcode files that live next to
// the bishengir-compile executable.
//
// The early true-async PreloadBase needs these paths before the HIR/RegBase
// lowering pipeline runs, but it must not mutate MLIR. Keeping resolution
// free of MLIR (no ModuleOp, no dialect attributes) lets PreloadBase consume
// a copied result while the historical post-HIR `addBitcodeAttrsToModule`
// call sites write the attributes authoritatively. A content SHA
// cross-check verifies the early-resolved paths match the attribute paths
// written by `addBitcodeAttrsToModule`.
//
// This header intentionally depends on LLVM Support only so a path-only harness
// can exercise it without an MLIR build.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_UTILS_AUXBITCODEPATHS_H
#define BISHENGIR_TOOLS_UTILS_AUXBITCODEPATHS_H

#include "llvm/ADT/StringRef.h"

#include <string>

namespace bishengir {
namespace auxbitcode {

/// Absolute canonical paths of the aux bitcode files present in the install
/// prefix. A field stays empty when the corresponding file does not exist, so
/// an "empty" result is the normal degenerate case (missing vendor files) and
/// never an error.
struct ResolvedAuxBitcodePaths {
  std::string aic;
  std::string aiv;
  std::string mixAic;
  std::string mixAiv;
  std::string host;

  bool empty() const {
    return aic.empty() && aiv.empty() && mixAic.empty() && mixAiv.empty() &&
           host.empty();
  }
};

/// Get the lib directory path (`../lib` relative to the executable). Returns a
/// canonical absolute path without ".." or "."; empty when it cannot be
/// derived.
std::string getLibDirFromExecutable(llvm::StringRef executablePath);

/// Resolve the existing `<libDir>/meta_op.*.<subcoreTag>.<optLevel>.bc` and
/// `<libDir>/host.<optLevel>.bc` files. \p subcoreTag selects the aicore
/// bitcode variant ("c220", "c310", "920r1", ...). An empty \p optLevel selects
/// unsuffixed files. Only files that exist are returned, canonicalized via
/// real_path. Never inspects or mutates MLIR.
ResolvedAuxBitcodePaths resolveAuxBitcodePaths(llvm::StringRef executablePath,
                                               llvm::StringRef subcoreTag,
                                               llvm::StringRef optLevel = "");

} // namespace auxbitcode
} // namespace bishengir

#endif // BISHENGIR_TOOLS_UTILS_AUXBITCODEPATHS_H

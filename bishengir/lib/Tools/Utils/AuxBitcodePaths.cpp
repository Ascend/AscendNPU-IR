//===- AuxBitcodePaths.cpp - Aux bitcode path resolution ------------------===//
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

#include "bishengir/Tools/Utils/AuxBitcodePaths.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <string>

std::string
bishengir::auxbitcode::getLibDirFromExecutable(llvm::StringRef executablePath) {
  if (executablePath.empty() ||
      (!executablePath.contains('/') && !executablePath.contains('\\')))
    return "";
  llvm::SmallString<256> absPath(executablePath);
  if (llvm::sys::fs::make_absolute(absPath))
    return "";
  llvm::SmallString<256> realPath;
  if (!llvm::sys::fs::real_path(absPath, realPath))
    absPath = realPath;
  llvm::sys::path::remove_filename(absPath);
  llvm::sys::path::append(absPath, "..", "lib");
  llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);
  return std::string(absPath.str());
}

namespace {

/// Canonicalize \p file if it exists, else return an empty string. Mirrors the
/// historical `addIfExists` predicate: only existing files become attributes.
std::string resolveExisting(const llvm::SmallString<256> &file) {
  if (!llvm::sys::fs::exists(file))
    return "";
  llvm::SmallString<256> canonicalPath;
  if (llvm::sys::fs::real_path(file, canonicalPath))
    return "";
  return canonicalPath.str().str();
}

std::string joinLibDir(llvm::StringRef libDir, const char *filename) {
  llvm::SmallString<256> path(libDir);
  llvm::sys::path::append(path, filename);
  return resolveExisting(path);
}

} // namespace

bishengir::auxbitcode::ResolvedAuxBitcodePaths
bishengir::auxbitcode::resolveAuxBitcodePaths(llvm::StringRef executablePath,
                                              llvm::StringRef subcoreTag,
                                              llvm::StringRef optLevel) {
  ResolvedAuxBitcodePaths paths;
  std::string libDir = getLibDirFromExecutable(executablePath);
  if (libDir.empty())
    return paths;

  auto resolveBitcode = [&](llvm::SmallString<64> name) {
    if (!optLevel.empty()) {
      name += ".";
      name += optLevel;
    }
    name += ".bc";
    return joinLibDir(libDir, name.c_str());
  };
  auto metaOpName = [&](llvm::StringRef core, std::string &out) {
    llvm::SmallString<64> name("meta_op.");
    name += core;
    name += ".";
    name += subcoreTag;
    out = resolveBitcode(name);
  };

  metaOpName("aic", paths.aic);
  metaOpName("aiv", paths.aiv);
  metaOpName("mix.aic", paths.mixAic);
  metaOpName("mix.aiv", paths.mixAiv);
  paths.host = resolveBitcode(llvm::SmallString<64>("host"));
  return paths;
}

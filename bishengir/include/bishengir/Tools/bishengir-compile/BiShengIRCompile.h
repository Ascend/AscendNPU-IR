//===- BiShengIRCompile.h - BiShengIR Compile Tool Support -------*- C++-*-===//
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

#ifndef BISHENGIR_TOOLS_BISHENGIRCOMPILE_BISHENGIRCOMPILE_H
#define BISHENGIR_TOOLS_BISHENGIRCOMPILE_BISHENGIRCOMPILE_H

#include "bishengir/Tools/bishengir-compile/Config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <optional>
#include <tuple>
#include <utility>

namespace bishengir {

using OwningModuleRef = mlir::OwningOpRef<mlir::ModuleOp>;

/// Detect the CANN version (major.minor.patch) by reading
/// {ASCEND_TOOLKIT_HOME}/Ascend/ascend-toolkit/latest/<arch>-linux/
/// ascend_toolkit_install.info, mirroring triton-ascend's backend/utils.py.
/// The first line containing "version" is parsed for a
/// "<major>.<minor>[.<patch>]" version string. Returns std::nullopt when the
/// variable, the file, or a parseable version is unavailable.
inline std::optional<std::tuple<unsigned, unsigned, unsigned>>
detectCannVersion() {
  std::optional<std::string> toolkitHome =
      llvm::sys::Process::GetEnv("ASCEND_TOOLKIT_HOME");
  if (!toolkitHome || toolkitHome->empty())
    return std::nullopt;
  llvm::Triple hostTriple(llvm::sys::getProcessTriple());
  llvm::SmallString<256> versionFile(*toolkitHome);
  llvm::sys::path::append(versionFile, "Ascend", "ascend-toolkit", "latest");
  llvm::sys::path::append(
      versionFile, llvm::Twine(hostTriple.getArchName()) + "-linux",
      "ascend_toolkit_install.info");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(versionFile);
  if (std::error_code ec = file.getError())
    return std::nullopt;
  llvm::Regex versionPattern("[0-9]+\\.[0-9]+(\\.[0-9]+)?");
  llvm::SmallVector<llvm::StringRef, 8> lines;
  (*file)->getBuffer().split(lines, '\n');
  for (llvm::StringRef line : lines) {
    if (!line.contains_insensitive("version"))
      continue;
    llvm::SmallVector<llvm::StringRef, 2> matches;
    if (!versionPattern.match(line, &matches) || matches.empty())
      continue;
    llvm::SmallVector<llvm::StringRef, 3> parts;
    matches[0].split(parts, '.');
    unsigned major = 0;
    unsigned minor = 0;
    unsigned patch = 0;
    if (parts.size() < 2 || parts[0].getAsInteger(10, major) ||
        parts[1].getAsInteger(10, minor))
      continue;
    if (parts.size() > 2 && parts[2].getAsInteger(10, patch))
      continue;
    return std::tuple<unsigned, unsigned, unsigned>{major, minor, patch};
  }
  return std::nullopt;
}

/// Resolve the template bitcode optimization level to use, in priority order:
/// 1. an explicit --enable-optimized-metaop flag (true=O2, false=O0);
/// 2. the CANN version detected from the environment (>= 9.2.0 uses O2,
///    older versions use O0);
/// 3. the default: O2.
inline std::string
resolveTemplateBitcodeOptLevel(const BiShengIRCompileMainConfig &config) {
  auto &registeredOptions = llvm::cl::getRegisteredOptions();
  auto optIt = registeredOptions.find("enable-optimized-metaop");
  if (optIt != registeredOptions.end() &&
      optIt->second->getNumOccurrences() > 0)
    return config.getEnableOptimizedMetaop() ? "O2" : "O0";
  if (std::optional<std::tuple<unsigned, unsigned, unsigned>> version =
          detectCannVersion())
    return *version >= std::make_tuple(9u, 2u, 0u) ? "O2" : "O0";
  return "O2";
}

/// Main entry point to run BiShengIR pipeline to compile module into binary.
llvm::FailureOr<OwningModuleRef>
runBiShengIRPipeline(mlir::ModuleOp mod, BiShengIRCompileMainConfig config);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIRCOMPILE_BISHENGIRCOMPILE_H

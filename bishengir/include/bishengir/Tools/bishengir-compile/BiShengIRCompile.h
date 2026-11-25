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
#include <string>
#include <tuple>
#include <utility>

namespace bishengir {

using OwningModuleRef = mlir::OwningOpRef<mlir::ModuleOp>;

class BishengAsyncRuntime;

struct CannVersion {
  std::tuple<unsigned, unsigned, unsigned> release;
  std::string prerelease;
};

/// Read the version field in the CANN toolkit or all-in-one install metadata.
/// Returns std::nullopt when no parseable version is available.
inline std::optional<CannVersion> detectCannVersionWithPrerelease() {
  std::optional<std::string> homePath =
      llvm::sys::Process::GetEnv("ASCEND_HOME_PATH");
  if (!homePath || homePath->empty())
    return std::nullopt;
  llvm::Triple hostTriple(llvm::sys::getProcessTriple());
  llvm::SmallString<256> infoDir(*homePath);
  llvm::sys::path::append(infoDir,
                          llvm::Twine(hostTriple.getArchName()) + "-linux");
  llvm::Regex versionPattern("^([0-9]+\\.[0-9]+(\\.[0-9]+)?)(-([[:alnum:].-]+))"
                             "?(\\+[[:alnum:].-]+)?$");

  for (llvm::StringRef name :
       {"ascend_toolkit_install.info", "ascend_all_cann_install.info"}) {
    llvm::SmallString<256> path(infoDir);
    llvm::sys::path::append(path, name);
    auto file = llvm::MemoryBuffer::getFile(path);
    if (!file)
      continue;
    llvm::SmallVector<llvm::StringRef, 8> lines;
    (*file)->getBuffer().split(lines, '\n');
    for (llvm::StringRef line : lines) {
      auto [key, value] = line.split('=');
      if (!key.trim().equals_insensitive("version"))
        continue;
      llvm::SmallVector<llvm::StringRef, 6> matches;
      if (!versionPattern.match(value.trim(), &matches))
        continue;
      llvm::SmallVector<llvm::StringRef, 3> parts;
      matches[1].split(parts, '.');
      unsigned major = 0;
      unsigned minor = 0;
      unsigned patch = 0;
      if (parts[0].getAsInteger(10, major) ||
          parts[1].getAsInteger(10, minor) ||
          (parts.size() == 3 && parts[2].getAsInteger(10, patch)))
        continue;
      return CannVersion{{major, minor, patch}, matches[4].str()};
    }
  }
  return std::nullopt;
}

inline std::optional<std::tuple<unsigned, unsigned, unsigned>>
detectCannVersion() {
  if (auto version = detectCannVersionWithPrerelease())
    return version->release;
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
/// \p runtime  compile-scope true-async runtime (nullptr in Process/CPU-Runner
///             mode; downstream compileDeviceKernel falls back to sync).
llvm::FailureOr<OwningModuleRef>
runBiShengIRPipeline(mlir::ModuleOp mod, BiShengIRCompileMainConfig config,
                     BishengAsyncRuntime *runtime = nullptr);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIRCOMPILE_BISHENGIRCOMPILE_H

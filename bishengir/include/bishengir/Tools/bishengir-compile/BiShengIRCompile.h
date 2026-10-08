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
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"

#include <optional>
#include <utility>

namespace bishengir {

using OwningModuleRef = mlir::OwningOpRef<mlir::ModuleOp>;

/// Detect the CANN version (major.minor) from the environment. The first of
/// CANN_VERSION / ASCEND_CANN_VERSION / ASCEND_TOOLKIT_VERSION that contains a
/// "<major>.<minor>" version string wins; otherwise the version is extracted
/// from the toolkit home path variable (ASCEND_TOOLKIT_HOME / TOOLCHAIN_HOME /
/// ASCEND_HOME_PATH), e.g. ".../cann-9.2.0-beta.2". Returns std::nullopt when
/// no version can be determined.
inline std::optional<std::pair<unsigned, unsigned>> detectCannMajorMinor() {
  auto parseFrom =
      [](llvm::StringRef text) -> std::optional<std::pair<unsigned, unsigned>> {
    llvm::Regex versionPattern("[0-9]+\\.[0-9]+");
    llvm::SmallVector<llvm::StringRef, 1> matches;
    if (!versionPattern.match(text, &matches) || matches.empty())
      return std::nullopt;
    llvm::SmallVector<llvm::StringRef, 2> parts;
    matches[0].split(parts, '.');
    if (parts.size() < 2)
      return std::nullopt;
    unsigned major = 0;
    unsigned minor = 0;
    if (parts[0].getAsInteger(10, major) || parts[1].getAsInteger(10, minor))
      return std::nullopt;
    return std::pair<unsigned, unsigned>{major, minor};
  };
  for (llvm::StringRef var :
       {"CANN_VERSION", "ASCEND_CANN_VERSION", "ASCEND_TOOLKIT_VERSION"}) {
    if (std::optional<std::string> value = llvm::sys::Process::GetEnv(var))
      if (std::optional<std::pair<unsigned, unsigned>> version =
              parseFrom(*value))
        return version;
  }
  for (llvm::StringRef var :
       {"ASCEND_TOOLKIT_HOME", "TOOLCHAIN_HOME", "ASCEND_HOME_PATH"}) {
    if (std::optional<std::string> value = llvm::sys::Process::GetEnv(var))
      if (std::optional<std::pair<unsigned, unsigned>> version =
              parseFrom(*value))
        return version;
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
  if (std::optional<std::pair<unsigned, unsigned>> version =
          detectCannMajorMinor()) {
    unsigned major = version->first;
    unsigned minor = version->second;
    return (major > 9 || (major == 9 && minor >= 2)) ? "O2" : "O0";
  }
  return "O2";
}

/// Main entry point to run BiShengIR pipeline to compile module into binary.
llvm::FailureOr<OwningModuleRef>
runBiShengIRPipeline(mlir::ModuleOp mod, BiShengIRCompileMainConfig config);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIRCOMPILE_BISHENGIRCOMPILE_H

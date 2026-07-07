//===- Config.h - BiShengIR Compile Tool Support -----------------*- C++-*-===//
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

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_CONFIG_H
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_CONFIG_H

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Tools/BiShengIRConfigBase/Config.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bishengir {

/// Configuration options for the bishengir-compile tool.
/// This is intended to help building tools like bishengir-compile by collecting
/// the supported options.
/// The API is fluent, and the options are ordered by functionality. The options
/// can be exposed to the LLVM command line by registering them with
/// `BiShengIRCompileMainConfig::registerCLOptions();` and creating a
/// config using
/// `auto config = BiShengIRCompileMainConfig::createFromCLOptions();`.
class BiShengIRCompileMainConfig : public BiShengIRCompileConfigBase {
public:
  BiShengIRCompileMainConfig() = default;
  ~BiShengIRCompileMainConfig() override = default;

  /// Register the options as global LLVM command line options.
  static void registerCLOptions();

  /// Create a new config with the default set from the CL options.
  static BiShengIRCompileMainConfig createFromCLOptions();

  /// Collect compile arguments that will be passed to hivmc.
  static void collectHIVMCArgs();
  static void collectHIVMCArgs(BiShengIRCompileMainConfig &config);
  static bool isSharedWithDownstreamToolchain(llvm::StringRef argName);

#include "bishengir/Tools/bishengir-compile/CompileConfigs.cpp.inc"

  bool isUBAwareVfFusion() const {
    return getVfFusionMode() == mlir::analysis::FusionMode::UBAwareOp;
  }
  BiShengIRCompileMainConfig &updateMaxInputParamsSizeInBytes(size_t size) {
    deviceMaxInputParamSizeInBytesFlag =
        std::max(size, deviceMaxInputParamSizeInBytesFlag);
    return *this;
  }

  BiShengIRCompileMainConfig &setClArgs(std::vector<std::string> args) {
    clArgsFlag = std::move(args);
    return *this;
  }
  std::vector<std::string> getClArgs() const { return clArgsFlag; }

  /// Update max buffer count tuning delta.
  BiShengIRCompileMainConfig &increaseMaxBufferCountTuning(int64_t delta) {
    hfusionMaxBufferCountTuningFlag += delta;
    return *this;
  }

  BiShengIRCompileMainConfig &increaseHfusionMaxBufferCountTuning(
      int64_t delta) {
    hfusionMaxBufferCountTuningFlag += delta;
    return *this;
  }

  bool shouldEnableLayoutOptimization() const {
    return getEnableLayoutOptimization() &&
           mlir::hacc::utils::isAscend950(getTarget());
  }

  bool shouldEnableMixedCV() const {
    return getEnableMixedCV() &&
           mlir::hacc::utils::isAscend950(getTarget());
  }

  bool hasSimtStackLimit() const { return getSimtStackLimit() >= 0; }

  std::vector<std::string> getHIVMCArgsDashDash() const {
    std::vector<std::string> args;
    for (auto &arg : getHIVMCArgs()) {
      if (llvm::StringRef(arg).starts_with("-")) {
        args.push_back(arg);
        continue;
      }
      args.push_back("--" + arg);
    }
    return args;
  }

  /// Set the path of the bishengir-compile executable (e.g. argv[0]).
  /// Used to locate the default aicore bitcode files in ../lib.
  BiShengIRCompileMainConfig &setExecutablePath(const std::string &path) {
    executablePath = path;
    return *this;
  }
  std::string getExecutablePath() const { return executablePath; }

protected:
  // Real option/config state lives in CompileConfigs.cpp.inc. Keep only
  // tool-specific runtime bookkeeping here.
  std::vector<std::string> clArgsFlag;

private:
  std::string executablePath;
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_CONFIG_H

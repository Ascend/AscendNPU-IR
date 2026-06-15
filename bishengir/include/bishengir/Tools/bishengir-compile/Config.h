//===- Config.h - BiShengIR Compile Tool Support -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

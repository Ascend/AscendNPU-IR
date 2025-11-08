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

#ifndef BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_CONFIG_H
#define BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_CONFIG_H

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Tools/BiShengIRConfigBase/Config.h"

namespace bishengir {

/// Configuration options for the bishengir-hivm-compile tool.
/// This is intended to help building tools like bishengir-hivm-compile by
/// collecting the supported options. The API is fluent, and the options are
/// ordered by functionality. The options can be exposed to the LLVM command
/// line by registering them with
/// `BiShengIRHIVMCompileMainConfig::registerCLOptions();` and creating a
/// config using
/// `auto config = BiShengIRHIVMCompileMainConfig::createFromCLOptions();`.
class BiShengIRHIVMCompileMainConfig : public BiShengIRCompileConfigBase {
public:
  BiShengIRHIVMCompileMainConfig() = default;
  ~BiShengIRHIVMCompileMainConfig() override = default;

  /// Register the options as global LLVM command line options.
  static void registerCLOptions();

  /// Create a new config with the default set from the CL options.
  static BiShengIRHIVMCompileMainConfig createFromCLOptions();

  /// Collect compile arguments that will be passed to hivmc.
  static void collectHIVMCArgs();

// Include auto-generated config
#include "bishengir/Tools/bishengir-hivm-compile/HIVMCompileConfigs.cpp.inc"

  std::vector<std::string> getHIVMCArgsDashDash() const {
    std::vector<std::string> args;
    for (auto &arg : getHIVMCArgs()) {
      args.push_back("--" + arg);
    }
    return args;
  }
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_CONFIG_H

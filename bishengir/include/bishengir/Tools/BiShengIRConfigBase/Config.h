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

#ifndef BISHENGIR_TOOLS_BISHENGIR_CONFIG_BASE_CONFIG_H
#define BISHENGIR_TOOLS_BISHENGIR_CONFIG_BASE_CONFIG_H

#include "bishengir/Config/bishengir-config.h"

#if (defined(MLIR_ENABLE_EXECUTION_ENGINE) && MLIR_ENABLE_EXECUTION_ENGINE) || \
    defined(BISHENGIR_ENABLE_EXECUTION_ENGINE)
#define BISHENGIR_HAS_EXECUTION_ENGINE 1
#include "bishengir/Pass/CPURunnerMetadata.h"
#else
#define BISHENGIR_HAS_EXECUTION_ENGINE 0
#endif

#include <string>

namespace bishengir {

class BiShengIRCompileConfigBase {
public:
  virtual ~BiShengIRCompileConfigBase() = default;

  BiShengIRCompileConfigBase &setInputFile(const std::string &file) {
    inputFileFlag = file;
    return *this;
  }
  std::string getInputFile() const { return inputFileFlag; }

  BiShengIRCompileConfigBase &setOutputFile(const std::string &file) {
    outputFileFlag = file;
    return *this;
  }
  std::string getOutputFile() const { return outputFileFlag; }

  bool shouldEnableCPURunner() const {
#if BISHENGIR_HAS_EXECUTION_ENGINE
    return enableCPURunnerFlag.numOccurrences != 0 ||
           enableCPURunnerBeforeFlag.numOccurrences != 0 ||
           enableCPURunnerAfterFlag.numOccurrences != 0;
#else
    return false;
#endif
  }

#if BISHENGIR_HAS_EXECUTION_ENGINE
  CPURunnerMetadata<false> CPURunnerOpt() const { return enableCPURunnerFlag; }

  CPURunnerMetadata<true> CPURunnerBeforeOpt() const {
    return enableCPURunnerBeforeFlag;
  }

  CPURunnerMetadata<true> CPURunnerAfterOpt() const {
    return enableCPURunnerAfterFlag;
  }
#endif

protected:
  std::string inputFileFlag{"-"};
  std::string outputFileFlag{"-"};

#if BISHENGIR_HAS_EXECUTION_ENGINE
  CPURunnerMetadata<false> enableCPURunnerFlag;
  CPURunnerMetadata<true> enableCPURunnerBeforeFlag;
  CPURunnerMetadata<true> enableCPURunnerAfterFlag;
#endif
};

} // namespace bishengir

#undef BISHENGIR_HAS_EXECUTION_ENGINE

#endif // BISHENGIR_TOOLS_BISHENGIR_CONFIG_BASE_CONFIG_H

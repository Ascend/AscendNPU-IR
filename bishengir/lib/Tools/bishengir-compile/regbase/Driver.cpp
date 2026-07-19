//===- Driver.cpp - RegBase compile driver ----------------------*- C++ -*-===//
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

#include "bishengir/Tools/bishengir-compile/regbase/Driver.h"

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/regbase/PassPipeline.h"
#include "bishengir/Tools/bishengir-compile/regbase/Utility.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace llvm;

namespace bishengir {
namespace regbase {
namespace {

bool shouldUseDirectRegBasePipeline(const BiShengIRCompileMainConfig &config) {
  if (config.getPureSimt())
    return false;
#if defined(BISHENGIR_ENABLE_TRITON_COMPILE) && BISHENGIR_ENABLE_TRITON_COMPILE
  if (config.getEnableSimdSimtMixCompile() || config.getEnableTritonIRCompile())
    return false;
#endif
  return true;
}

int runExternalRegBaseCompile(ArrayRef<std::string> originalCLArgs) {
  SmallVector<StringRef> arguments;
  arguments.push_back(""); // placeholder, replaced by execute with full path
  for (const auto &arg : originalCLArgs)
    arguments.push_back(arg);

  if (failed(bishengir::execute("bishengir-compile-a5",
                                bishengir::getBiShengInstallPath(), arguments)))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

} // namespace

int runRegBaseCompile(ModuleOp module, BiShengIRCompileMainConfig config,
                      ArrayRef<std::string> originalCLArgs) {
  if (!shouldUseDirectRegBasePipeline(config))
    return runExternalRegBaseCompile(originalCLArgs);

  if (failed(inferLayoutOptimization(module, config))) {
    llvm::errs() << "[ERROR] Failed to infer layout optimization\n";
    return EXIT_FAILURE;
  }
  if (config.getEnableTritonKernelCompile()) {
    if (failed(inferMixedCV(module, config))) {
      llvm::errs() << "[ERROR] Failed to infer mix mode\n";
      return EXIT_FAILURE;
    }
  }

  config.setClArgs(filterRegBaseForwardedHIVMCOptions(originalCLArgs));

  if (failed(runRegBasePipeline(module, config))) {
    llvm::errs() << "[ERROR] Failed to run BiShengIR regbase pipeline\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

} // namespace regbase
} // namespace bishengir

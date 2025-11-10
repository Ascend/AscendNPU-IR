//===- BiShengIRHIVMCompile.cpp - BiShengIR HIVM Compile Tool Support C++-*-==//
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

#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-hivm-compile/BiShengIRHIVMCompile.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {

/// Get the HIVMC binary name.
StringRef getHIVMCName() {
  const char *kBiShengIRHIVMBinaryName = "hivmc";
  return kBiShengIRHIVMBinaryName;
}

#ifndef BISHENGIR_PUBLISH
std::vector<std::string>
getCompatibleOptions(const std::vector<std::string> &arguments) {
  std::vector<std::string> result;
  DenseSet<std::string> skipArgs = {"debug", "debug-only",
                                    "mlir-print-ir-before-all",
                                    "mlir-print-ir-after-all"};
  for (const std::string &arg : arguments) {
    StringRef argRef = arg;
    std::string trimArg = argRef.trim().ltrim('-').str();
    if (skipArgs.contains(trimArg)) {
      continue;
    }
    result.push_back(arg);
  }
  return result;
}
#endif

LogicalResult runExternalHIVMC(ModuleOp module,
                               const BiShengIRHIVMCompileMainConfig &config) {
  TempDirectoriesStore tempDirsStore;
  std::string inputFile = "module.hivm.opt.mlir";
  std::string outputFile = config.getOutputFile();
  auto inputFileHandler = getTempFile(inputFile, tempDirsStore);
  if (!inputFileHandler) {
    llvm::dbgs()
        << "[ERROR] Failed to create temporary input file needed to run "
           "hivm compile.\n";
    return failure();
  }
  inputFile = inputFileHandler->outputFilename();
  module.print(inputFileHandler->os(),
               mlir::OpPrintingFlags().enableDebugInfo(
                   config.getEnableSanitizer() || config.getEnableDebugInfo()));
  inputFileHandler->os().flush();

  std::vector<std::string> arguments;
  arguments.emplace_back("");
  arguments.push_back(inputFile);

  auto hivmcArgs = config.getHIVMCArgsDashDash();
  arguments.insert(arguments.end(), hivmcArgs.begin(), hivmcArgs.end());

  arguments.emplace_back("-o");
  arguments.push_back(outputFile);

#ifndef BISHENGIR_PUBLISH
  arguments = getCompatibleOptions(arguments);
#endif

  SmallVector<StringRef> argumentsRef(arguments.begin(), arguments.end());
  if (failed(execute(getHIVMCName(), getBiShengInstallPath(), argumentsRef))) {
    return failure();
  }

  return success();
}

} // namespace

FailureOr<OwningModuleRef>
bishengir::runBiShengIRHIVMPipeline(ModuleOp hirCompileModule,
                                    BiShengIRHIVMCompileMainConfig config) {
  MLIRContext *ctx = hirCompileModule->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  diagEngine.registerHandler([&](Diagnostic &diag) -> mlir::LogicalResult {
    return handleDiagnostic(diag);
  });
  auto buildPipeline =
      std::bind(buildBiShengHIRHIVMPipeline, std::placeholders::_1, config);
  if (failed(runPipeline(hirCompileModule, buildPipeline, config,
                         "BiShengHIR HIVM")))
    return failure();

  // No need to lower to LLVM
  if (!config.getConvertHIRToLIR() || config.shouldEnableCPURunner()) {
    auto outputFile = config.getOutputFile();
    std::string errorMessage;
    std::unique_ptr<llvm::ToolOutputFile> fileHandle =
        mlir::openOutputFile(outputFile, &errorMessage);
    if (!fileHandle) {
      llvm::errs() << "[ERROR] Failed to open: " << outputFile
                   << " error message: " << errorMessage << "\n";
      return failure();
    }
    hirCompileModule.print(
        fileHandle->os(),
        mlir::OpPrintingFlags().enableDebugInfo(config.getEnableSanitizer() ||
                                                config.getEnableDebugInfo()));
    fileHandle->keep();

    return OwningModuleRef(hirCompileModule);
  }

  auto res = runExternalHIVMC(hirCompileModule, config);
  if (res.failed())
    hirCompileModule.emitWarning(
        "External hivmc run fails, returning module before running "
        "external compiler");

  return OwningModuleRef(hirCompileModule);
}

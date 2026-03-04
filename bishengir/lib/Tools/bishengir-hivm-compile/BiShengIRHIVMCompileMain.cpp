//===- BiShengIRHIVMCompile.cpp - BiShengIR HIVM Compile Tool Support C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/bishengir-compile/Utility.h"
#include "bishengir/Tools/bishengir-hivm-compile/AdapterSanitizer.h"
#include "bishengir/Tools/bishengir-hivm-compile/BiShengIRHIVMCompile.h"
#include "bishengir/Tools/bishengir-hivm-compile/PassPipeline.h"
#include "bishengir/Tools/bishengir-hivm-compile/Utility.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;
using namespace object;

llvm::LogicalResult
runExternalHIVMC(ModuleOp &module,
                        const bishengir::BiShengIRCompileMainConfig &config) {
  TempDirectoriesStore tempDirsStore;
  std::string inputFile = "module.hivm.opt.mlir";
  std::string outputFile = config.outputFile();

  auto inputFileHandler = getTempFile(inputFile, tempDirsStore);
  if (!inputFileHandler) {
    llvm::dbgs()
        << "[ERROR] Failed to create temporary input file needed to run "
           "hivmc compile.\n";
    return failure();
  }
  inputFile = inputFileHandler->outputFilename();

  module.print(inputFileHandler->os(), mlir::OpPrintingFlags().enableDebugInfo(
                                           config.shouldEnableSanitizer() ||
                                           config.shouldEnableDebugInfo()));
  inputFileHandler->os().flush();

  std::vector<std::string> arguments;
  arguments.push_back("");
  arguments.push_back(inputFile);

  auto clArgs = config.getClArgs();
  arguments.insert(arguments.end(), clArgs.begin(), clArgs.end());

  arguments.push_back("-o");
  arguments.push_back(outputFile);
  arguments.push_back("--only-run-hivm-pipeline=false");

  SmallVector<StringRef> argumentsRef(arguments.begin(), arguments.end());
  if (failed(execute(getHIVMCName(),
                     getBiShengIRHIVMCompileInstallPath(), argumentsRef))) {
    return failure();
  }

  return success();
}

LogicalResult
bishengir::runBiShengIRHIVMPipeline(ModuleOp hirCompileModule,
                                    BiShengIRCompileMainConfig config) {
  if (failed(checkOptionValidity(config))) {
    return failure();
  }

  MLIRContext *ctx = hirCompileModule->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<Diagnostic> collectedDiagnostics;
  // Collect diagnostics and emit them afterwards because we have tuning
  // mechanism.
  auto handlerID =
      diagEngine.registerHandler([&collectedDiagnostics](Diagnostic &diag) {
        collectedDiagnostics.emplace_back(std::move(diag));
      });
  
  LogicalResult runPipelineStatus = failure();
  if (config.shouldEnableSimdSimtMixCompile()) {
    auto [mainMod, simtMods] = getMixedModules(hirCompileModule);
    // only main module runs HIVM pipelines 
    runPipelineStatus = runPipeline(mainMod, buildBiShengHIRHIVMPipeline,
                                    config, "BiShengHIRHIVM");
  } else {
    runPipelineStatus =
        runPipeline(hirCompileModule, buildBiShengHIRHIVMPipeline, config,
                    "BiShengHIRHIVM");
  }

  // Restore to the default handler.
  diagEngine.eraseHandler(handlerID);

  if (failed(runPipelineStatus)) {
    for (auto &diag : llvm::reverse(collectedDiagnostics)) {
      diagEngine.emit(std::move(diag));
    }
    return failure();
  }

  if (failed(runExternalHIVMC(hirCompileModule, config))) {
    return failure();
  }

  return success();
}

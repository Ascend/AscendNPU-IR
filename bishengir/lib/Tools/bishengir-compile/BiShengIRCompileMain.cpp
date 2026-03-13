//===- BiShengIRCompile.cpp - BiShengIR Compile Tool Support -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/PassPipeline.h"
#include "bishengir/Tools/bishengir-compile/Utility.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {
llvm::LogicalResult
runExternalHIVMPipeline(ModuleOp &module,
                        const bishengir::BiShengIRCompileMainConfig &config) {
  TempDirectoriesStore tempDirsStore;
  std::string inputFile = "module.hivm.mlir";
  std::string outputFile = "module.hivm.opt.mlir";
  auto inputFileHandler = getTempFile(inputFile, tempDirsStore);
  auto outputFileHandler = getTempFile(outputFile, tempDirsStore);
  if (!inputFileHandler || !outputFileHandler) {
    llvm::dbgs()
        << "[ERROR] Failed to create temporary input/output files needed "
           "to run hivm pipeline.\n";
    return failure();
  }

  inputFile = inputFileHandler->outputFilename();
  outputFile = outputFileHandler->outputFilename();

  module.print(inputFileHandler->os(), mlir::OpPrintingFlags().enableDebugInfo(
                                           config.shouldEnableSanitizer() ||
                                           config.shouldEnableDebugInfo()));
  inputFileHandler->os().flush();

  std::vector<std::string> arguments;
  arguments.push_back("");
  arguments.push_back(inputFile);

  auto clArgs = config.getClArgs();
  llvm::copy_if(clArgs, std::back_inserter(arguments),
                [](const std::string &arg) {
                  return !llvm::StringRef(arg).starts_with("--proton-");
                });

  arguments.push_back("-o");
  arguments.push_back(config.outputFile());
  arguments.push_back("--only-run-hivm-pipeline=true");

  SmallVector<StringRef> argumentsRef(arguments.begin(), arguments.end());
  if (failed(execute(getBiShengIRHIVMCompilerName(),
                     getBiShengIRHIVMCompileInstallPath(), argumentsRef))) {
    return failure();
  }

  std::string errorMessage;
  auto file = mlir::openInputFile(outputFile, &errorMessage);
  if (!file) {
    llvm::errs() << "[ERROR] Failed to open: " << outputFile
                 << " error message: " << errorMessage << '\n';
    return failure();
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), mlir::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> moduleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, module->getContext());
  if (!moduleRef) {
    llvm::errs() << "[ERROR] Failed to open: " << outputFile << '\n';
    return failure();
  }

  module = moduleRef->clone();
  return success();
}

} // namespace

LogicalResult
bishengir::runBiShengIRPipeline(ModuleOp mod,
                                BiShengIRCompileMainConfig config) {
  if (failed(checkOptionValidity(config))) {
    return failure();
  }

  MLIRContext *ctx = mod->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<Diagnostic> collectedDiagnostics;
  // Collect diagnostics and emit them afterwards because we have tuning
  // mechanism.
  auto handlerID = diagEngine.registerHandler([&](Diagnostic &diag) {
    collectedDiagnostics.emplace_back(std::move(diag));
  });

  bool hlResult = false;
  bool compileSuccess = false;
  int tryTimes = config.isTuning() ? 1 : 5;
  for (int i = 0; i < tryTimes; i++) {
    LDBG("Attempt number: " << i << " with max buffer count tuning delta: "
                            << config.maxBufferCountTuning());
    ModuleOp hirCompileMode = mod.clone();
    if (config.shouldEnableSimdSimtMixCompile()) {
      // simt-simd mixed pipeline
      hlResult = succeeded(runPipeline(
          hirCompileMode, buildBiShengHIRPipeline, config, "BiShengHIR"));

      // extract main module and simt modules
      auto [mainMod, simtMods] = getMixedModules(hirCompileMode);
      // run ttir pipeline on simt modules
      for (auto simtMod : simtMods) {
        hlResult &= succeeded(runPipeline(simtMod, buildBiShengTTIRPipeline,
                                              config, "BiShengTTIR"));
      }

      // Pass top module to HIVMpipeline and hivmc
    } else if (config.shouldCompileTritonDialect()) {
      // simt-only pipeline(The input is ttir).
      hlResult = succeeded(runPipeline(
          hirCompileMode, buildBiShengTTIRPipeline, config, "BiShengTTIR"));
    } else {
      hlResult = succeeded(runPipeline(
          hirCompileMode, buildBiShengHIRPipeline, config, "BiShengHIR"));
    }

    compileSuccess =
        hlResult && succeeded(runExternalHIVMPipeline(hirCompileMode, config));
    if (compileSuccess) {
      mod = hirCompileMode.clone();
      break;
    }

    // increase max buffers by 2 in HFusion auto schedule
    config.increaseMaxBufferCountTuning(2);
  }

  // Restore to the default handler.
  diagEngine.eraseHandler(handlerID);

  if (!compileSuccess) {
    for (auto &diag : llvm::reverse(collectedDiagnostics)) {
      diagEngine.emit(std::move(diag));
    }
    return failure();
  }

  if (config.shouldEnableCPURunner()) {
    auto fileHandle = mlir::openOutputFile(config.outputFile());
    assert(fileHandle != nullptr);
    fileHandle->os() << mod << '\n';
    fileHandle->keep();
    return success();
  }

  return success();
}

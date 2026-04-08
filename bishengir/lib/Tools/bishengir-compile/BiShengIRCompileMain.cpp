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

static std::vector<std::string>
skipOptions(const std::vector<std::string> &options,
            const std::set<std::string> &skip) {
  std::vector<std::string> result;
  for (const std::string &arg : options) {
    StringRef argRef = arg;
    SmallVector<StringRef> parts;
    argRef.split(parts, '=');
    if (parts.empty()) {
      continue;
    }
    std::string trimArg = parts[0].trim().ltrim('-').str();
    if (skip.count(trimArg) != 0) {
      continue;
    }
    result.push_back(arg);
  }
  return result;
}

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
  llvm::copy_if(clArgs, std::back_inserter(arguments),
                [](const std::string &arg) {
                  return !llvm::StringRef(arg).starts_with("--proton-");
                });

  arguments.push_back("-o");
  arguments.push_back(outputFile);
  arguments.push_back("--only-run-hivm-pipeline=false");

  // TODO: Support options in hivmc
  std::set<std::string> blacklist = {"inject-ir-from-file", "print-pass-id",
                                     "inject-ir-before", "inject-ir-after",
                                     "hfusion-enable-multiple-consumer-fusion",
                                     "disable-tightly-coupled-buffer-reuse"};
  auto skippedArgs = skipOptions(arguments, blacklist);

  SmallVector<StringRef> argumentsRef(skippedArgs.begin(), skippedArgs.end());
  if (failed(execute(getHIVMCName(),
                     getBiShengIRHIVMCompileInstallPath(), argumentsRef))) {
    return failure();
  }

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

  bool hirCompileSuccess = false;
  int tryTimes = config.isTuning() ? 1 : 5;
  // triton compile has nothing to do with HFusion auto schedule, so we don't
  // need to tune for it.
  tryTimes = config.shouldCompileTriton() ? 1 : tryTimes;
  for (int i = 0; i < tryTimes; i++) {
    LDBG("Attempt number: " << i << " with max buffer count tuning delta: "
                            << config.maxBufferCountTuning());
    ModuleOp hirCompileMode = mod.clone();
    // simt-simd mixed pipeline
    bool success = true;
    if (config.shouldEnableSimdSimtMixCompile()) {
        success &= succeeded(runPipeline(
            hirCompileMode, buildBiShengHIRPipeline, config, "BiShengHIR"));
      // extract main module and simt modules
      auto [mainMod, simtMods] = getMixedModules(hirCompileMode);
      // run ttir pipeline on simt modules
      for (auto simtMod : simtMods) {
        success &= succeeded(runPipeline(simtMod, buildBiShengTTIRPipeline,
                                         config, "BiShengTTIR"));
      }
      success &= succeeded(runPipeline(hirCompileMode, buildBiShengHIRFinishPipeline,
                                         config, "BishengHIR"));
      success &= succeeded(runPipeline(mainMod, buildFinalHIVMPipelines,
                                         config, "buildFinalHIVMPipelines"));
    } else if (config.shouldCompileTritonDialect()) {
      success = succeeded(runPipeline(
          hirCompileMode, buildBiShengTTIRPipeline, config, "BiShengTTIR"));
    } else {
      success = succeeded(runPipeline(
          hirCompileMode, buildBiShengHIRPipeline, config, "BiShengHIR"));
      success &= succeeded(runPipeline(hirCompileMode, buildFinalHIVMPipelines,
                                         config, "buildFinalHIVMPipelines"));
    }

    if (success && succeeded(runExternalHIVMC(hirCompileMode, config))) {
      hirCompileSuccess = true;
      mod = hirCompileMode.clone();
      break;
    }

    // increase max buffers by 2 in HFusion auto schedule
    config.increaseMaxBufferCountTuning(2);
  }

  // Restore to the default handler.
  diagEngine.eraseHandler(handlerID);

  if (!hirCompileSuccess) {
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

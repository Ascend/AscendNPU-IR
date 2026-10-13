//===- BiShengIRCompile.cpp - BiShengIR Compile Tool Support -----*- C++-*-===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Tools/RetriablePassManager/CbufOverflowRetryPolicy.h"
#include "bishengir/Tools/RetriablePassManager/CcOverflowRetryPolicy.h"
#include "bishengir/Tools/RetriablePassManager/RetriablePassManager.h"
#include "bishengir/Tools/RetriablePassManager/TuningRetryPolicy.h"
#include "bishengir/Tools/RetriablePassManager/UbOverflowRetryPolicy.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/PassPipeline.h"
#include "bishengir/Tools/hivmc/Config.h"
#include "bishengir/Version/Version.h"
#include "bishengir/Tools/hivmc/HIVMC.h"
#include "bishengir/Tools/hivmc/HIVMCA3.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VersionTuple.h"
#include <functional>
#include <regex>
#include <set>
#include <vector>

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {

/// Get the lib directory path (../lib relative to bishengir-compile
/// executable). Returns canonical absolute path without ".." or ".".
std::string getLibDirFromExecutable(StringRef executablePath) {
  if (executablePath.empty() ||
      (!executablePath.contains('/') && !executablePath.contains('\\')))
    return "";
  llvm::SmallString<256> absPath(executablePath);
  if (llvm::sys::fs::make_absolute(absPath))
    return "";
  llvm::SmallString<256> realPath;
  if (!llvm::sys::fs::real_path(absPath, realPath))
    absPath = realPath;
  llvm::sys::path::remove_filename(absPath);
  llvm::sys::path::append(absPath, "..", "lib");
  llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);
  return std::string(absPath.str());
}

/// Add bitcode path attributes to ModuleOp from ../lib/*.bc files.
/// Paths are canonical (no ".." or ".") before being stored in attributes.
void addBitcodeAttrsToModule(ModuleOp module, StringRef executablePath,
                             const BiShengIRCompileMainConfig &config) {
  std::string libDir = getLibDirFromExecutable(executablePath);
  MLIRContext *ctx = module->getContext();

  using CreateAttrFn =
      std::function<mlir::Attribute(MLIRContext *, mlir::StringAttr)>;
  auto addIfExists = [&](const char *filename, llvm::StringRef attrName,
                         CreateAttrFn createAttr) {
    llvm::SmallString<256> bcPath(libDir);
    llvm::sys::path::append(bcPath, filename);
    if (!llvm::sys::fs::exists(bcPath))
      return;
    llvm::SmallString<256> canonicalPath;
    if (llvm::sys::fs::real_path(bcPath, canonicalPath))
      return;
    module->setAttr(
        attrName,
        createAttr(ctx, mlir::StringAttr::get(ctx, canonicalPath.str().str())));
  };

  // Template bitcode is installed with an optimization-level suffix
  // (e.g. meta_op.aic.c220.O2.bc); pick the variant with priority: explicit
  // --enable-optimized-metaop > CANN version (>= 9.2.0 uses O2) > default O2.
  std::string optLevel = resolveTemplateBitcodeOptLevel(config);
  auto bitcodeFileName = [&](StringRef base) -> std::string {
    return (base + "." + optLevel + ".bc").str();
  };

  addIfExists(bitcodeFileName("meta_op.aic.c220").c_str(),
              mlir::hivm::AIC_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::AIC_BITCODEAttr::get(c, s);
              });
  addIfExists(bitcodeFileName("meta_op.aiv.c220").c_str(),
              mlir::hivm::AIV_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::AIV_BITCODEAttr::get(c, s);
              });
  addIfExists(bitcodeFileName("meta_op.mix.aic.c220").c_str(),
              mlir::hivm::MIX_AIC_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
              });
  addIfExists(bitcodeFileName("meta_op.mix.aiv.c220").c_str(),
              mlir::hivm::MIX_AIV_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
              });
  addIfExists(bitcodeFileName("host").c_str(),
              mlir::hivm::HOST_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::HOST_BITCODEAttr::get(c, s);
              });
}

/// Get the HIVMC binary name.
StringRef getHIVMCName() {
  const char *kBiShengIRHIVMBinaryName = "hivmc";
  return kBiShengIRHIVMBinaryName;
}

LogicalResult handleSaveTemps(ModuleOp& module, BiShengIRCompileMainConfig& config) {
  TempDirectoriesStore tempDirsStore;
  std::string inputFile = "module.hivm.opt.mlir";
  std::string outputFile = config.getOutputFile();
  std::unique_ptr<llvm::ToolOutputFile> inputFileHandler;

  // Handle --save-temps=<directory> option to store module.hivm.opt.mlir
  if (!config.getSaveTemps().empty()) {
    llvm::SmallString<256> saveTempsDir(config.getSaveTemps());
    if (llvm::sys::fs::make_absolute(saveTempsDir)) {
      llvm::errs() << "[ERROR] Failed to get absolute path for save-temps.\n";
      return failure();
    }
    if (!llvm::sys::fs::exists(saveTempsDir))
      if (auto ec = llvm::sys::fs::create_directories(saveTempsDir)) {
        llvm::errs() << "[ERROR] Failed to create save-temps directory: " << saveTempsDir << "\n";
        return failure();
      }
    llvm::sys::path::append(saveTempsDir, inputFile);
    std::string errorMessage;
    inputFileHandler = mlir::openOutputFile(saveTempsDir, &errorMessage);
    if (!inputFileHandler) {
      llvm::errs() << "[ERROR] Failed to open save-temps file: " << errorMessage << "\n";
      return failure();
    }
    inputFileHandler->keep();
  } else {
    inputFileHandler = getTempFile(inputFile, tempDirsStore);
    if (!inputFileHandler) {
      llvm::dbgs() << "[ERROR] Failed to create temporary input file needed to run hivmc compile.\n";
      return failure();
    }
  }
  inputFile = inputFileHandler->outputFilename();

  module.print(inputFileHandler->os(), mlir::OpPrintingFlags().enableDebugInfo(
                                         config.getEnableSanitizer() ||
                                         config.getEnableDebugInfo()));
  inputFileHandler->os().flush();
  return success();
}

HIVMCMainConfig HIVMCFromBiShengIRConfig(BiShengIRCompileMainConfig& config) {
    HIVMCMainConfig hivmcConfig;
    /// A3-specific
    hivmcConfig.targetBackend(hacc::TargetDevice::Unknown);
    hivmcConfig.autoVectorizeV2(true);
    hivmcConfig.compileTriton(false);
    ///

    hivmcConfig.limitAutoMultiBufferForLocalBuffer(true);
    hivmcConfig.deterministicComputing(true);
    hivmcConfig.simtOptimizationMode(1900101);
    hivmcConfig.enableAutoCVBalance(true);
    hivmcConfig.setUseDPX(true);
    hivmcConfig.onlyRunHIVMPipeline(false);

    hivmcConfig.appendBishengOptions(config.getAppendBishengOptions());
    hivmcConfig.compileTriton(config.getEnableTritonKernelCompile());
    hivmcConfig.compileTritonDialect(config.getEnableTritonIRCompile());
    hivmcConfig.enableSimdSimtMixCompile(config.getEnableSimdSimtMixCompile());
    hivmcConfig.enableSIMTOnly(config.getPureSimt());
    hivmcConfig.enableSanitizer(config.getEnableSanitizer());
    hivmcConfig.enableSIMTFastDiv(config.getEnableSIMTFastDiv());
    hivmcConfig.enableDebugVariables(config.getEnableDebugVariables());
    hivmcConfig.enableDebugInfo(config.getEnableDebugInfo());
    hivmcConfig.saveTemps(config.getSaveTemps());
    hivmcConfig.injectBarrierAllSync(config.getEnableHIVMInjectBarrierAllSync());
    hivmcConfig.setExtraDeviceBCPaths(config.getLinkAicoreBitcode());
    hivmcConfig.setDisableFMA(config.getDisableFMA());
    hivmcConfig.setSaveLinkedIR(config.getSaveLinkedIR());
    hivmcConfig.setNumWarps(config.getNumWarps());
    hivmcConfig.setThreadsPerWarp(config.getThreadsPerWarp());
    hivmcConfig.setSharedDynamicSize(config.getSharedMemDynamicSize());
    hivmcConfig.tritonMetadataOutput(config.getTritonMetadataOutput());
    hivmcConfig.disableDecomposeReduction(config.getDisableDecomposeReduction());
    hivmcConfig.disableReorderInstruction(config.getDisableReorderInstruction());

    if (hivmcConfig.getTritonGridDim().size() > 3) {
        report_fatal_error(
            "Invalid --simt-triton-grid: at most 3 elements allowed x,y,z\n");
    }
    hivmcConfig.setOutputFile(config.getOutputFile());

    StringTmpPath path(hivmcConfig.outputFile());

    // TODO: investigate if this check is redundant
    llvm::cantFail(llvm::errorCodeToError(hivmcCanonicalizePath(path)));
    hivmcConfig.setOutputFile(path.str().str());
    return hivmcConfig;
}
} // namespace
FailureOr<OwningModuleRef>
bishengir::runBiShengIRPipeline(ModuleOp mod,
                                BiShengIRCompileMainConfig config) {
  MLIRContext *ctx = mod->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<std::unique_ptr<Diagnostic>> collectedDiagnostics;

  // Collect diagnostics and emit them afterwards because we have tuning
  // mechanism.
  auto handlerID = diagEngine.registerHandler([&](Diagnostic &diag) {
    collectedDiagnostics.push_back(
        std::make_unique<Diagnostic>(std::move(diag)));
  });

  RetriablePassManager retriablePm(config, ctx);
  if (config.getEnableTritonKernelCompile()) {
    retriablePm.addPolicy(std::make_unique<UbOverflowRetryPolicy>());
    retriablePm.addPolicy(std::make_unique<CbufOverflowRetryPolicy>());
    retriablePm.addPolicy(std::make_unique<CcOverflowRetryPolicy>());
  }

  if (config.getEnableTuningMode() && !config.getEnableTritonKernelCompile()) {
    retriablePm.addPolicy(std::make_unique<TuningRetryPolicy>());
  }

  std::vector<AppliedCompileFallback> retriablePipelineFallbacks;
  auto buildPipeline = std::bind(buildBiShengHIRPipeline, std::placeholders::_1,
                                 std::cref(config));
  bool hirCompileSuccess = succeeded(retriablePm.runWithRetry(
      mod, buildPipeline, "BiShengHIR", collectedDiagnostics,
      retriablePipelineFallbacks));

  // Restore to the default handler.
  diagEngine.eraseHandler(handlerID);
  for (auto &diag : llvm::reverse(collectedDiagnostics)) {
    [[maybe_unused]] auto res = handleDiagnostic(*diag);
  }

  if (!hirCompileSuccess) {
    RetriablePassManager::emitFallbackSummary(retriablePipelineFallbacks,
                                              /*compilationSucceeded=*/false);
    for (auto &diag : llvm::reverse(collectedDiagnostics)) {
      diagEngine.emit(std::move(*diag));
    }
    return failure();
  }

  RetriablePassManager::emitFallbackSummary(retriablePipelineFallbacks,
                                            /*compilationSucceeded=*/true);

  if (config.shouldEnableCPURunner()) {
    auto outputFile = config.getOutputFile();
    std::string errorMessage;
    std::unique_ptr<llvm::ToolOutputFile> fileHandle =
        mlir::openOutputFile(outputFile, &errorMessage);
    if (!fileHandle) {
      llvm::errs() << "[ERROR] Failed to open: " << outputFile
                   << " error message: " << errorMessage << "\n";
      return failure();
    }
    mod.print(fileHandle->os(),
              mlir::OpPrintingFlags().enableDebugInfo(
                  config.getEnableSanitizer() || config.getEnableDebugInfo()));
    fileHandle->keep();

    return OwningModuleRef(mod);
  }
  // Add bitcode path attributes from ../lib/*.bc to ModuleOp before hivmc.
  addBitcodeAttrsToModule(mod, config.getExecutablePath(), config);
  auto savedTemp = handleSaveTemps(mod, config);
  if (failed(savedTemp)) {
    return failure();
  }

  auto hivmcConfig = HIVMCFromBiShengIRConfig(config);
  auto res = runHIVMCCompileA3(mod, hivmcConfig);
  if (res.failed()) {
    mod.emitError("External hivmc run fails, returning module before running "
                  "external compiler");
    return failure();
  }

  return OwningModuleRef(mod);
}

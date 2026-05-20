//===- BiShengIRCompile.cpp - BiShengIR Compile Tool Support -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/PassPipeline.h"
#include "bishengir/Tools/bishengir-compile/Utility.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VersionTuple.h"
#include <functional>
#include <set>

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {

/// Get the lib directory path (../lib relative to bishengir-compile
/// executable). Returns canonical absolute path without ".." or ".".
static std::string getLibDirFromExecutable(StringRef executablePath) {
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
static void addBitcodeAttrsToModule(ModuleOp module, StringRef executablePath,
                                    const BiShengIRCompileMainConfig &config) {
  std::string libDir = getLibDirFromExecutable(executablePath);
  MLIRContext *ctx = module->getContext();
  ctx->loadDialect<mlir::hivm::HIVMDialect>();

  auto addIfExists = [&](const char *filename, llvm::StringRef attrName,
                         auto createAttr) {
    llvm::SmallString<256> bcPath(libDir);
    llvm::sys::path::append(bcPath, filename);
    if (!llvm::sys::fs::exists(bcPath))
      return;
    llvm::SmallString<256> canonicalPath;
    if (llvm::sys::fs::real_path(bcPath, canonicalPath))
      return;
    module->setAttr(
        attrName, createAttr(ctx, mlir::StringAttr::get(ctx, canonicalPath.str().str())));
  };

  if (hacc::utils::isAscend950(config.getTarget())) {
    addIfExists("meta_op.aic.c310.bc", mlir::hivm::AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.aiv.c310.bc", mlir::hivm::AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIV_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aic.c310.bc", mlir::hivm::MIX_AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aiv.c310.bc", mlir::hivm::MIX_AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
                });
  } else {
    addIfExists("meta_op.aic.c220.bc", mlir::hivm::AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.aiv.c220.bc", mlir::hivm::AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIV_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aic.c220.bc", mlir::hivm::MIX_AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aiv.c220.bc", mlir::hivm::MIX_AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
                });
  }
  addIfExists("host-a5.bc", mlir::hivm::HOST_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::HOST_BITCODEAttr::get(c, s);
              });
}

static std::vector<std::string>
filterSharedHIVMCOptions(const std::vector<std::string> &options) {
  std::vector<std::string> result;
  for (const std::string &arg : options) {
    StringRef argRef = arg;
    // Keep the fixed input/output wiring in runExternalHIVMC unchanged and
    // only filter user-facing options that may be forwarded to hivmc.
    if (!argRef.starts_with("-")) {
      continue;
    }
    SmallVector<StringRef> parts;
    argRef.split(parts, '=');
    if (parts.empty()) {
      continue;
    }
    std::string trimArg = parts[0].trim().ltrim('-').str();
    if (!BiShengIRCompileMainConfig::isSharedWithDownstreamToolchain(
            trimArg)) {
      continue;
    }
    result.push_back(arg);
  }
  return result;
}

static std::vector<std::string>
skipOptions(const std::vector<std::string> &options,
            const std::set<std::string> &skip) {
  std::vector<std::string> result;
  for (const std::string &arg : options) {
    StringRef argRef = arg;
    SmallVector<StringRef> parts;
    argRef.split(parts, '=');
    if (parts.empty())
      continue;
    std::string trimArg = parts[0].trim().ltrim('-').str();
    if (skip.count(trimArg) != 0)
      continue;
    result.push_back(arg);
  }
  return result;
}

static bool hasCLIArg(const std::vector<std::string> &arguments,
                      StringRef argName) {
  return llvm::any_of(arguments, [&](const std::string &arg) {
    StringRef argRef = arg;
    if (!argRef.starts_with("-"))
      return false;
    return argRef.ltrim('-').split('=').first == argName;
  });
}

llvm::LogicalResult
runExternalHIVMC(ModuleOp &module,
                 const bishengir::BiShengIRCompileMainConfig &config) {
  TempDirectoriesStore tempDirsStore;
  std::string inputFile = "module.hivm.opt.mlir";
  std::string outputFile = config.getOutputFile();

  auto inputFileHandler = getTempFile(inputFile, tempDirsStore);
  if (!inputFileHandler) {
    llvm::dbgs()
        << "[ERROR] Failed to create temporary input file needed to run "
           "hivmc compile.\n";
    return failure();
  }
  inputFile = inputFileHandler->outputFilename();

  module.print(inputFileHandler->os(), mlir::OpPrintingFlags().enableDebugInfo(
                                           config.getEnableSanitizer() ||
                                           config.getEnableDebugInfo()));
  inputFileHandler->os().flush();

  std::vector<std::string> arguments;
  arguments.push_back("");
  arguments.push_back(inputFile);

  auto hivmcOptions = filterSharedHIVMCOptions(config.getHIVMCArgsDashDash());
  llvm::append_range(arguments, hivmcOptions);
  for (const auto &arg : filterSharedHIVMCOptions(config.getClArgs())) {
    auto argName = StringRef(arg).ltrim('-').split('=').first;
    if (!hasCLIArg(arguments, argName))
      arguments.push_back(arg);
  }


  arguments.push_back("-o");
  arguments.push_back(outputFile);
  arguments.push_back("--only-run-hivm-pipeline=false");

  // TODO: Support options in hivmc
  std::set<std::string> blacklist = {"inject-ir-from-file", "print-pass-id",
                                     "inject-ir-before", "inject-ir-after",
                                     "hfusion-enable-multiple-consumer-fusion",
                                     "disable-tightly-coupled-buffer-reuse",
                                     "enable-hivm-cross-core-gss",
                                     "enable-tree-reduce-v2",
                                     "vf-fusion-mode",
                                     "disable-vf-reachable-check",
                                     "disable-sink-dpx-load"};
  auto skippedArgs = skipOptions(arguments, blacklist);

  SmallVector<StringRef> argumentsRef(skippedArgs.begin(), skippedArgs.end());
  if (failed(execute(getHIVMCName(), getBiShengIRHIVMCompileInstallPath(),
                     argumentsRef))) {
    return failure();
  }

  return success();
}

bool runSIMTToLLVMCompile(ArrayRef<ModuleOp> modules,
                          BiShengIRCompileMainConfig config) {
  config.setPureSimt(true);
  bool result = true;
  for (auto module : modules) {
    result &= runPipeline(module, buildSIMTPipeline, config, "BiShengSIMT")
                  .succeeded();
  }
  return result;
}

bool runSIMDToLLVMCompile(ModuleOp module, BiShengIRCompileMainConfig &config) {
  return runPipeline(module, buildBiShengHIRAVEToLLVMPipeline, config,
                     "BiShengSIMD")
      .succeeded();
}
} // namespace

LogicalResult
bishengir::runBiShengIRPipeline(ModuleOp mod,
                                BiShengIRCompileMainConfig config) {
  if (failed(checkOptionValidity(config))) {
    return failure();
  }

  bool hasUboverflow = false;
  MLIRContext *ctx = mod->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<Diagnostic> collectedDiagnostics;
  // Collect diagnostics and emit them afterwards because we have tuning
  // mechanism.
  auto handlerID = diagEngine.registerHandler([&](Diagnostic &diag) {
    // VF fusion may cause ub overflow. in this case, it will fallback to allop
    // fused to decrease ub occupation
    // Todo: use Enum to standardize the format of error message printing
    if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
      std::string errMsg;
      llvm::raw_string_ostream errStream(errMsg);
      errStream << diag;
      if (errStream.str().find("ub overflow") != std::string::npos) {
        hasUboverflow = true;
      }
    }
    collectedDiagnostics.emplace_back(std::move(diag));
  });

  bool hirCompileSuccess = false;
  int tryTimes = 5;
  // triton compile has nothing to do with HFusion auto schedule, so we don't
  // need to tune for it.
  //
  // TODO: refactor this ad-hoc retry loop into a dedicated retryPassManager
  // so each fallback policy is composable and explicit. Planned policies:
  //   - OpFusion retry policy: bump tiling-max-counter each attempt, up to 5
  //     retries (the current default branch).
  //   - AutoBlockify retry policy: progressively disable hoisting and then
  //     multi-buffer.
  //   - MultiBuffer retry policy: disable auto-multi-buffer.
  // Once the retryPassManager exists, the tryTimes / nested-if logic below
  // should be replaced by composing those policies.
  if (config.getEnableTuningMode()) {
    tryTimes = 1;
  }
  for (int i = 0; i < tryTimes; i++) {
    LDBG("Attempt number: " << i << " with max buffer count tuning delta: "
                            << config.getHfusionMaxBufferCountTuning());
    ModuleOp hirCompileMode = mod.clone();
    // simt-simd mixed pipeline
    bool success = true;
    hasUboverflow = false;
    if (config.getEnableSimdSimtMixCompile()) {
      success &= succeeded(runPipeline(hirCompileMode, buildBiShengHIRPipeline,
                                       config, "BiShengHIR"));
      // extract main module and simt modules
      auto [mainMod, simtMods] = getMixedModules(hirCompileMode);
      // run ttir pipeline on simt modules
      for (auto simtMod : simtMods) {
        success &= succeeded(runPipeline(simtMod, buildBiShengTTIRPipeline,
                                         config, "BiShengTTIR"));
      }
      success &= succeeded(runPipeline(
          hirCompileMode, buildBiShengHIRFinishPipeline, config, "BishengHIR"));
      success &= succeeded(runPipeline(mainMod, buildFinalHIVMPipelines, config,
                                       "buildFinalHIVMPipelines"));
    } else if (config.getEnableTritonIRCompile()) {
      success = succeeded(runPipeline(hirCompileMode, buildBiShengTTIRPipeline,
                                      config, "BiShengTTIR"));
    } else {
      success = succeeded(runPipeline(hirCompileMode, buildBiShengHIRPipeline,
                                      config, "BiShengHIR"));
      success &= succeeded(runPipeline(hirCompileMode, buildFinalHIVMPipelines,
                                       config, "buildFinalHIVMPipelines"));
      if (!success && hasUboverflow) {
        if (config.getEnableAutoMultiBuffer()) {
          // First-tier fallback on ub overflow: turn off
          // auto-multi-buffer and retry. Keep the diagnostic handler
          // registered so a subsequent ub overflow on the retry can
          // still be detected; clear the captured diagnostics so they
          // are not surfaced if the retry succeeds (or are replaced by
          // fresher ones before the next-tier fallback re-emits them).
          LDBG("ub overflow detected at attempt "
               << (i + 1) << "/" << tryTimes
               << ", fallback with disabled auto multi buffer");
          collectedDiagnostics.clear();
          config.setEnableAutoMultiBuffer(false);
        } else if (config.getEnableVFFusion()) {
          LDBG("ub overflow detected at attempt "
               << (i + 1) << "/" << tryTimes
               << ", fallback with disabled vffusion");
          collectedDiagnostics.clear();
          config.setEnableVFFusion(false);
        } else if (!config.getDisableVFReachableCheck()) {
          LDBG("ub overflow detected at attempt "
               << (i + 1) << "/" << tryTimes
               << ", fallback with disabled VF reachable check");
          collectedDiagnostics.clear();
          config.setDisableVFReachableCheck(true);
        } else if (!config.getDisableTightCoupledBuffer()) {
          LDBG("ub overflow detected at attempt "
               << (i + 1) << "/" << tryTimes
               << ", fallback with MixCV GM path");
          collectedDiagnostics.clear();
          config.setDisableTightCoupledBuffer(true);
        }
      }
    }

    // hivmc pipepine
    if (config.getEnableSimdSimtMixCompile()) {
      auto [mainMod, simtMods] = getMixedModules(hirCompileMode);
      // SIMT modules run triton lowering pipeline
      // Main module runs regular pipeline
      if (runSIMTToLLVMCompile(simtMods, config) &&
          runSIMDToLLVMCompile(mainMod, config)) {
        // Once both are lowered, flatten into single module
        // success &= succeeded(runPipeline(hirCompileModule,
        //                                  buildFinalMixVFCompilePipeline,
        //                                  config, "BiShengFinishLLVM"));
      }
    } else if (config.getPureSimt()) {
      success &= runSIMTToLLVMCompile(hirCompileMode, config);
    } else {
      success &= runSIMDToLLVMCompile(hirCompileMode, config);
    }
    addBitcodeAttrsToModule(hirCompileMode, config.getExecutablePath(), config);
    if (success && succeeded(runExternalHIVMC(hirCompileMode, config))) {
      hirCompileSuccess = true;
      mod = hirCompileMode.clone();
      break;
    }

    // increase max buffers by 2 in HFusion auto schedule
    config.increaseHfusionMaxBufferCountTuning(2);
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
    auto fileHandle = mlir::openOutputFile(config.getOutputFile());
    assert(fileHandle != nullptr);
    fileHandle->os() << mod << '\n';
    fileHandle->keep();
    return success();
  }

  return success();
}

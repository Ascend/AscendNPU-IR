//===- BiShengIRCompileMain.cpp - RegBase compile orchestration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RegBase (A5) pipeline orchestration with retry/fallback logic.
// Migrated from AscendNPU-IR-Dev BishengIRCompileMain.cpp.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Tools/hivmc/Config.h"
#include "bishengir/Tools/hivmc/HIVMC.h"
#include "bishengir/Tools/hivmc/HIVMCA5.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Version/Version.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/regbase/PassPipeline.h"
#include "bishengir/Tools/bishengir-compile/regbase/Utility.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VersionTuple.h"
#include <functional>
#include <set>

#define DEBUG_TYPE "bishengir-compile-regbase"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {

using PipelineBuilder = std::function<void(mlir::PassManager &,
                                           const BiShengIRCompileMainConfig &)>;

enum class CompileFlow {
  Mixed,
  PureSimt,
  Simd,
};

CompileFlow getCompileFlow(const BiShengIRCompileMainConfig &config) {
  if (config.getEnableSimdSimtMixCompile())
    return CompileFlow::Mixed;
  if (config.getPureSimt())
    return CompileFlow::PureSimt;
  return CompileFlow::Simd;
}

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
        attrName,
        createAttr(ctx, mlir::StringAttr::get(ctx, canonicalPath.str().str())));
  };

  if (mlir::hacc::utils::isAscend950(config.getTarget())) {
    addIfExists("meta_op.aic.c310.bc", mlir::hivm::AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.aiv.c310.bc", mlir::hivm::AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::AIV_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aic.c310.bc",
                mlir::hivm::MIX_AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aiv.c310.bc",
                mlir::hivm::MIX_AIV_BITCODEAttr::name,
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
    addIfExists("meta_op.mix.aic.c220.bc",
                mlir::hivm::MIX_AIC_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
                });
    addIfExists("meta_op.mix.aiv.c220.bc",
                mlir::hivm::MIX_AIV_BITCODEAttr::name,
                [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                  return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
                });
  }
  addIfExists("host.bc", mlir::hivm::HOST_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::HOST_BITCODEAttr::get(c, s);
              });
}

using MixedModules = std::pair<ModuleOp, SmallVector<ModuleOp, 2>>;

/// Run a pipeline on a module using the regbase builder and A3's runPipeline
/// utility.
static LogicalResult runPipelineRegBase(ModuleOp mod,
                                        const PipelineBuilder &buildPipeline,
                                        BiShengIRCompileMainConfig &config,
                                        const std::string &pipelineName) {
  // Bind the config parameter into the builder to match A3's runPipeline
  // signature.
  auto boundPipeline =
      std::bind(buildPipeline, std::placeholders::_1, std::cref(config));
  return bishengir::runPipeline(mod, boundPipeline, config, pipelineName);
}

//   Scope::ScopeOp SIMT detection logic needed.
static MixedModules getMixedModules(ModuleOp topMod) {
  MixedModules res;
  res.first = nullptr;
  for (auto subMod : topMod.getOps<ModuleOp>()) {
    if (subMod->hasAttr(hacc::SIMTModuleAttr::name)) {
      res.second.push_back(subMod);
    } else {
      assert(!res.first && "only one main module is allowed");
      res.first = subMod;
    }
  };
  // if no main module, return the top module
  if (!res.first)
    res.first = topMod;

  return res;
}

static bool runModulePipeline(ModuleOp module, const PipelineBuilder &builder,
                              BiShengIRCompileMainConfig &config,
                              StringRef pipelineName) {
  return succeeded(
      runPipelineRegBase(module, builder, config, pipelineName.str()));
}

static bool runModulePipelines(ArrayRef<ModuleOp> modules,
                               const PipelineBuilder &builder,
                               BiShengIRCompileMainConfig &config,
                               StringRef pipelineName) {
  bool success = true;
  for (auto module : modules) {
    // Stop this pipeline on each module after earlier pipeline fails.
    success =
        success && runModulePipeline(module, builder, config, pipelineName);
  }
  return success;
}

static bool runMixedPipelines(ModuleOp mixedModule,
                              BiShengIRCompileMainConfig &config) {
  if (!runModulePipeline(mixedModule, regbase::buildBiShengHIRPipeline, config,
                         "BiShengHIR")) {
    return false;
  }

  auto [mainMod, simtMods] = getMixedModules(mixedModule);
  if (!runModulePipelines(simtMods, regbase::buildBiShengTTIRPipeline, config,
                          "BiShengSIMT")) {
    return false;
  }

  if (!runModulePipeline(mixedModule, regbase::buildBiShengHIRFinishPipeline,
                         config, "BishengHIR")) {
    return false;
  }

  if (!runModulePipeline(mainMod, regbase::buildFinalHIVMPipelines, config,
                         "buildFinalHIVMPipelines")) {
    return false;
  }
  return runModulePipeline(mainMod, regbase::buildBiShengHIRAVEToLLVMPipeline,
                           config, "BiShengSIMD");
}

LogicalResult handleSaveTemps(ModuleOp module, BiShengIRCompileMainConfig& config) {
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

    /// TODO: add flag difference logic for hivmc and bishengir flags in Options.td
    /// These flags are set manually due to mismatch in
    /// HIVMCConfigCLOptions constructor and Options.td
    ///
    /// Options.td is used both by bishengir pipeline and hivmc pipeline,
    /// So the flags which are both in bishengir pipeline and hivmc pipeline are just
    /// copied from the BiShengIRCompileMainConfig through dispatch table
    /// (which is generated in CompileOptionsGen.cpp)
    /// P.S : the same problem is also on A3 pipeline, check the translation there

    /// A5-specific
    hivmcConfig.autoVectorizeV2(false);
    hivmcConfig.targetBackend(config.getTarget());
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

LogicalResult
bishengir::regbase::runRegBasePipeline(ModuleOp mod,
                                       BiShengIRCompileMainConfig config) {
  if (failed(checkInOutOptionsValidity(config))) {
    return failure();
  }

  // Overflow tracking: UB gets a Vector-only multi-buffer fallback; L1/L0C
  // overflow still uses the coarse enable-auto-multi-buffer switch.
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
      const std::string &msg = errStream.str();
      if (msg.find("ub overflow") != std::string::npos) {
        hasUboverflow = true;
      }
    }
    collectedDiagnostics.emplace_back(std::move(diag));
  });

  bool hirCompileSuccess = false;
  int tryTimes = 2;
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
  CompileFlow compileFlow = getCompileFlow(config);
  for (int i = 0; i < tryTimes; i++) {
    LDBG("Attempt number: " << i << " with max buffer count tuning delta: "
                            << config.getHfusionMaxBufferCountTuning());
    ModuleOp hirCompileMode = mod.clone();
    bool success = true;
    hasUboverflow = false;

    if (compileFlow == CompileFlow::Mixed) {
      success = runMixedPipelines(hirCompileMode, config);
    } else if (compileFlow == CompileFlow::PureSimt) {
      success =
          runModulePipeline(hirCompileMode, regbase::buildBiShengTTIRPipeline,
                            config, "BiShengSIMT");
    } else {
      // Standard HIR compile path (no SIMD/SIMT split)
      success =
          runModulePipeline(hirCompileMode, regbase::buildBiShengHIRPipeline,
                            config, "BiShengHIR");
      // Stop final HIVM lowering after earlier pipeline fails.
      success = success && runModulePipeline(hirCompileMode,
                                             regbase::buildFinalHIVMPipelines,
                                             config, "buildFinalHIVMPipelines");
    }
    if (!success) {
      if (hasUboverflow &&
          config.getVfFusionMode() == mlir::analysis::FusionMode::MaxParallel) {
        LDBG("ub overflow detected at attempt "
             << (i + 1) << "/" << tryTimes << ", fallback with all-op mode");
        config.setVfFusionMode(mlir::analysis::FusionMode::AllOp);
        // increase max buffers by 2 in HFusion auto schedule
        config.increaseHfusionMaxBufferCountTuning(2);
        collectedDiagnostics.clear();
        continue;
      }
      LDBG("can't retry at attempt " << (i + 1) << "/" << tryTimes
                                     << ", no fallback");
      break;
    }

    if (compileFlow == CompileFlow::Simd) {
      // Stop SIMD lowering after earlier pipeline fails.
      success = runModulePipeline(hirCompileMode,
                                  regbase::buildBiShengHIRAVEToLLVMPipeline,
                                  config, "BiShengSIMD");
    }


    addBitcodeAttrsToModule(hirCompileMode, config.getExecutablePath(), config);

    auto savedTemp = handleSaveTemps(hirCompileMode, config);
    if (failed(savedTemp)) {
      return failure();
    }

    auto hivmcConfig = HIVMCFromBiShengIRConfig(config);
    if (success && succeeded(runHIVMCCompileA5(hirCompileMode, hivmcConfig))) {
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

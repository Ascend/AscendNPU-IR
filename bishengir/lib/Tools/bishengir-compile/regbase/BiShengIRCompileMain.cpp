//===- BiShengIRCompileMain.cpp - RegBase compile orchestration ------------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/regbase/PassPipeline.h"
#include "bishengir/Tools/bishengir-compile/regbase/Utility.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Support/FileUtilities.h"
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

  if (mlir::hacc::utils::isAscend950(config.getTarget())) {
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

static StringRef getHIVMCName() {
  const char *kBiShengIRHIVMBinaryName = "hivmc-a5";
  return kBiShengIRHIVMBinaryName;
}

llvm::LogicalResult
runExternalHIVMC(ModuleOp &module,
                 const bishengir::BiShengIRCompileMainConfig &config) {
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

  std::vector<std::string> arguments;
  arguments.push_back("");
  arguments.push_back(inputFile);

  auto hivmcOptions = config.getHIVMCArgsDashDash();
  llvm::append_range(arguments, hivmcOptions);
  for (const auto &arg : regbase::filterSharedHIVMCOptions(config.getClArgs())) {
    auto argName = StringRef(arg).ltrim('-').split('=').first;
    if (!regbase::hasCLIArg(arguments, argName))
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
                                     "enable-sink-dpx-load",
                                     // hivmc-a5 does not recognize
                                     // --enable-lir-compile (it is an A3
                                     // bishengir-compile-only flag). Drop it
                                     // before forwarding to hivmc-a5 in the
                                     // RegBase path.
                                     "enable-lir-compile"};
  auto skippedArgs = skipOptions(arguments, blacklist);

  SmallVector<StringRef> argumentsRef(skippedArgs.begin(), skippedArgs.end());
  if (failed(execute(getHIVMCName(), getBiShengInstallPath(),
                     argumentsRef))) {
    return failure();
  }

  return success();
}

using MixedModules = std::pair<ModuleOp, SmallVector<ModuleOp, 2>>;

/// Run a pipeline on a module using the regbase builder and A3's runPipeline utility.
static LogicalResult
runPipelineRegBase(ModuleOp mod,
                   const std::function<void(mlir::PassManager &,
                                            const BiShengIRCompileMainConfig &)>
                       &buildPipeline,
                   BiShengIRCompileMainConfig &config,
                   const std::string &pipelineName) {
  // Bind the config parameter into the builder to match A3's runPipeline signature.
  auto boundPipeline = std::bind(buildPipeline, std::placeholders::_1,
                                 std::cref(config));
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

//   Depends on Triton and DPX lowering passes.
static bool runSIMTToLLVMCompile(ArrayRef<ModuleOp> modules,
                                 BiShengIRCompileMainConfig config) {
  config.setPureSimt(true);
  bool result = true;
  for (auto module : modules) {
    result = result && runPipelineRegBase(module, regbase::buildSIMTPipeline,
                                          config, "BiShengSIMT").succeeded();
  }
  return result;
}

static bool runSIMDToLLVMCompile(ModuleOp module,
                                 BiShengIRCompileMainConfig &config) {
  return runPipelineRegBase(module, regbase::buildBiShengHIRAVEToLLVMPipeline,
                            config, "BiShengSIMD")
      .succeeded();
}

} // namespace

LogicalResult
bishengir::regbase::runRegBasePipeline(ModuleOp mod,
                                       BiShengIRCompileMainConfig config) {
  if (failed(checkInOutOptionsValidity(config))) {
    return failure();
  }

  // Track each address-space overflow independently so fallback can disable the
  // matching multi-buffer knob before trying coarser vector-side mitigations.
  bool hasUboverflow = false;
  bool hasCcOverflow = false;
  bool hasCbufOverflow = false;
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
      if (msg.find("cc overflow") != std::string::npos) {
        hasCcOverflow = true;
      }
      if (msg.find("cbuf overflow") != std::string::npos) {
        hasCbufOverflow = true;
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
    bool success = true;
    hasUboverflow = false;
    hasCcOverflow = false;
    hasCbufOverflow = false;

    //   Depends on Triton pipeline and SIMT module split.
    if (config.getEnableSimdSimtMixCompile()) {
      // Mixed SIMD/SIMT pipeline
      success = success && succeeded(runPipelineRegBase(
                               hirCompileMode,
                               regbase::buildBiShengHIRPipeline, config,
                               "BiShengHIR"));
      auto [mainMod, simtMods] = getMixedModules(hirCompileMode);
      for (auto simtMod : simtMods) {
        success = success && succeeded(runPipelineRegBase(
                                 simtMod, regbase::buildBiShengTTIRPipeline,
                                 config, "BiShengTTIR"));
      }
      success = success && succeeded(runPipelineRegBase(
                               hirCompileMode,
                               regbase::buildBiShengHIRFinishPipeline,
                               config, "BishengHIR"));
      success = success && succeeded(runPipelineRegBase(
                               mainMod, regbase::buildFinalHIVMPipelines,
                               config, "buildFinalHIVMPipelines"));

    } else if (config.getEnableTritonIRCompile()) {
      success = succeeded(runPipelineRegBase(hirCompileMode,
                                             regbase::buildBiShengTTIRPipeline,
                                             config, "BiShengTTIR"));
    }

    // Standard HIR compile path (no SIMD/SIMT split)
    success = succeeded(runPipelineRegBase(hirCompileMode,
                                           regbase::buildBiShengHIRPipeline,
                                           config, "BiShengHIR"));
    // Stop final HIVM lowering after earlier pipeline fails.
    success = success &&
              succeeded(runPipelineRegBase(hirCompileMode,
                                           regbase::buildFinalHIVMPipelines,
                                           config, "buildFinalHIVMPipelines"));
    bool hasMemoryOverflow = hasUboverflow || hasCcOverflow || hasCbufOverflow;
    if (!success && hasMemoryOverflow) {
      bool flippedAnyMultiBufferKnob = false;
      if (config.getEnableAutoMultiBuffer()) {
        if (hasUboverflow && !config.getDisableMultiBufferOnUB()) {
          LDBG("ub overflow detected at attempt "
               << (i + 1) << "/" << tryTimes
               << ", fallback with disabled UB (Vector) multi buffer");
          config.setDisableMultiBufferOnUB(true);
          flippedAnyMultiBufferKnob = true;
        }
        if (hasCcOverflow && !config.getDisableMultiBufferOnL0C()) {
          LDBG("cc overflow detected at attempt "
               << (i + 1) << "/" << tryTimes
               << ", fallback with disabled L0C multi buffer");
          config.setDisableMultiBufferOnL0C(true);
          flippedAnyMultiBufferKnob = true;
        }
        if (hasCbufOverflow && !config.getDisableMultiBufferOnL1()) {
          LDBG("cbuf overflow detected at attempt "
               << (i + 1) << "/" << tryTimes
               << ", fallback with disabled L1 (cbuf) multi buffer");
          config.setDisableMultiBufferOnL1(true);
          flippedAnyMultiBufferKnob = true;
        }
      }
      if (flippedAnyMultiBufferKnob) {
        collectedDiagnostics.clear();
      } else if (config.getEnableAutoMultiBuffer()) {
        LDBG("memory overflow (ub/cc/cbuf) detected at attempt "
             << (i + 1) << "/" << tryTimes
             << ", per-buffer knobs exhausted, fallback with disabled auto "
                "multi buffer");
        collectedDiagnostics.clear();
        config.setEnableAutoMultiBuffer(false);
      } else if (hasUboverflow && config.getEnableVFFusion()) {
        LDBG("ub overflow detected at attempt "
             << (i + 1) << "/" << tryTimes
             << ", fallback with disabled vffusion");
        config.setEnableVFFusion(false);
        collectedDiagnostics.clear();
      } else if (hasUboverflow && !config.getDisableVFReachableCheck()) {
        LDBG("ub overflow detected at attempt "
             << (i + 1) << "/" << tryTimes
             << ", fallback with disabled VF reachable check");
        config.setDisableVFReachableCheck(true);
        collectedDiagnostics.clear();
      } else if (hasUboverflow && !config.getDisableTightCoupledBuffer()) {
        LDBG("ub overflow detected at attempt "
             << (i + 1) << "/" << tryTimes
             << ", fallback with MixCV GM path");
        config.setDisableTightCoupledBuffer(true);
        collectedDiagnostics.clear();
      }
    }

    if (!success) {
      // Stop hivmc pipeline when upstream pipeline fails
      continue;
    }

    // hivmc pipeline
    if (config.getEnableSimdSimtMixCompile()) {
      auto [mainMod, simtMods] = getMixedModules(hirCompileMode);
      if (runSIMTToLLVMCompile(simtMods, config) &&
          runSIMDToLLVMCompile(mainMod, config)) {
        // Once both are lowered, flatten into single module
      }
    } else if (config.getPureSimt()) {
      success = success && runSIMTToLLVMCompile(hirCompileMode, config);
    } else {
      success = success && runSIMDToLLVMCompile(hirCompileMode, config);
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

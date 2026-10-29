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
#include "bishengir/Tools/Utils/AuxBitcodePaths.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/PassPipeline.h"
#include "bishengir/Tools/hivmc/BishengAsyncRuntime.h"
#include "bishengir/Tools/hivmc/Config.h"
#include "bishengir/Tools/hivmc/Utility.h"
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

//===--------------------------------------------------------------------===//
// Pure-path base aux resolver (A3, fixed c220)
//===--------------------------------------------------------------------===//
//
// These helpers run *before* the first HIR PassManager, so they must not read
// Module Attrs (which `addBitcodeAttrsToModule` writes only after the first
// HIR Pass).  They depend solely on `config` (CLI), `executablePath`, and the
// filesystem.  SHA cross-check against the Attr paths happens after
// `addBitcodeAttrsToModule` runs (see verifyEarlyAuxSha call site below).

/// Resolve base aux paths via the pure path calculator (A3 always uses c220,
/// matching `addBitcodeAttrsToModule` above).  Returns a SubCoreTarget-keyed
/// map of existing canonical paths; missing files are omitted.
static std::map<SubCoreTarget, std::string>
resolveEarlyAuxBitcodePaths(const BiShengIRCompileMainConfig &config) {
  return bishengir::toSubCoreTargetPaths(
      bishengir::auxbitcode::resolveAuxBitcodePaths(
          config.getExecutablePath(), /*subcoreTag=*/"c220",
          resolveTemplateBitcodeOptLevel(config)));
}

/// Per-lane distinct base aux path list.  Base aux is the set of aux files
/// needed by EVERY scenario (PureVV / PureCube / MixCV) on that lane, so it
/// can be preloaded BEFORE the topology is known.  Conditionally-needed aux
/// (mix BC, which only MixCV uses) is deferred to Final Delta, submitted
/// after the topology is known — this prevents a PureVV compile from failing
/// PreloadBase on a corrupt/missing mix BC it never actually consumes.
///   AIC base: aic + extraDeviceBCPaths
///   AIV base: aiv + extraDeviceBCPaths
///   mixAic/mixAiv: deferred to Final Delta (MixCV-only)
/// Host BC is *not* preloaded (it never enters the device worker auxPaths;
/// `host.<opt>.bc` is only consumed by host tiling func handling).
struct BaseAuxByLane {
  std::vector<std::string> aic;
  std::vector<std::string> aiv;
};

static BaseAuxByLane
splitBaseAuxByLane(const std::map<SubCoreTarget, std::string> &paths,
                   const BiShengIRCompileMainConfig &config) {
  BaseAuxByLane out;
  auto addIf = [&](SubCoreTarget t, std::vector<std::string> &dst) {
    auto it = paths.find(t);
    if (it != paths.end() && !it->second.empty())
      dst.push_back(it->second);
  };
  // Base aux: only the per-lane metaOp (aic/aiv).  Mix BC (mixAic/mixAiv) is
  // deferred to Final Delta so PureVV/PureCube never preload (and never fail
  // on) a mix BC they do not consume.
  addIf(SubCoreTarget::AIC, out.aic);
  addIf(SubCoreTarget::AIV, out.aiv);
  // A3 does not consult appendBishengOptions for device aux (HIVMCMainA3.cpp
  // auxPaths never reads appendBishengOptions).  Extra device BC paths from
  // --cce-link-aicore-ll-module are shared by both lanes.
  for (const auto &bc : config.getLinkAicoreBitcode()) {
    out.aic.push_back(bc);
    out.aiv.push_back(bc);
  }
  // HOST intentionally excluded: not preloaded to device workers.
  // MIX_AIC / MIX_AIV intentionally excluded: deferred to Final Delta.
  return out;
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

HIVMCMainConfig HIVMCFromBiShengIRConfig(BiShengIRCompileMainConfig& config,
                                        BishengAsyncRuntime *runtime) {
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
    hivmcConfig.setRuntime(runtime);
    return hivmcConfig;
}
} // namespace
FailureOr<OwningModuleRef>
bishengir::runBiShengIRPipeline(ModuleOp mod,
                                BiShengIRCompileMainConfig config,
                                BishengAsyncRuntime *runtime) {
  MLIRContext *ctx = mod->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<std::unique_ptr<Diagnostic>> collectedDiagnostics;

  // Collect diagnostics and emit them afterwards because we have tuning
  // mechanism.
  auto handlerID = diagEngine.registerHandler([&](Diagnostic &diag) {
    collectedDiagnostics.push_back(
        std::make_unique<Diagnostic>(std::move(diag)));
  });

  // ---- LoadInit submission (before the first HIR Pass, no wait) ----
  // submitLoadInit submits the LoadInit command (dlmopen + dlsym +
  // bishengLlcInit) only; it does NOT preload any aux path.  LoadInit is
  // independent of aux paths (the SO path is frozen by the runtime at
  // construction from BISHENG_LLC_LIBRARY env).  Submit returns immediately
  // (does not block); the main thread proceeds into the first HIR
  // PassManager below so LoadInit overlaps with HIR lowering.
  // Base aux paths are resolved and PreloadBase is submitted after this;
  // PreloadFinalDelta is submitted once the final topology is known.
  if (runtime) {
    runtime->initTrace();
    runtime->emitEvent("MAIN", "compile_scope", "begin");
    runtime->emitEvent("MAIN", "early_start_submit", "point");
    // mask = Both (conservative; lane usage is inferred later from a pure
    // path calculator).  LoadInit only loads SO + init, no aux preload.
    runtime->submitLoadInit(LaneMask::Both);
    // Resolve base aux paths via the pure path calculator (no MLIR Attr
    // read; `addBitcodeAttrsToModule` has NOT run yet at this point) and
    // submit PreloadBase so base aux preload overlaps HIR lowering.
    // LaneMask::Both is conservative: lane usage is unknown before HIR.
    //     `compileDeviceKernel` only drains the actually-used lane ticket, so
    //     a preload failure on an unused lane does not kill the Compile.
    //     Base aux unavailability (empty paths) submits a no-op PreloadBase
    //     and does NOT block LoadInit; handleCompile delta preload covers it.
    auto earlyAuxPaths = resolveEarlyAuxBitcodePaths(config);
    auto baseByLane = splitBaseAuxByLane(earlyAuxPaths, config);
    runtime->submitPreloadBase(LaneMask::Both, baseByLane.aic, baseByLane.aiv);
    runtime->emitEvent("MAIN", "hir_lowering", "begin");
  }

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

  // ---- HIR lowering block end ----
  if (runtime)
    runtime->emitEvent("MAIN", "hir_lowering", "end");

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
    if (runtime)
      runtime->emitEvent("MAIN", "compile_scope", "end", "fail");
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

    if (runtime)
      runtime->emitEvent("MAIN", "compile_scope", "end");
    return OwningModuleRef(mod);
  }
  // Add bitcode path attributes from ../lib/*.bc to ModuleOp before hivmc.
  addBitcodeAttrsToModule(mod, config.getExecutablePath(), config);
  // Cross-check the early-resolved paths against the Attr paths just
  // written by `addBitcodeAttrsToModule`.  Mismatch is warning-only
  // (perf diagnostic): handleCompile delta preload will still cover the
  // authoritative Attr paths, so correctness is preserved.
  if (runtime) {
    auto earlyAuxPaths = resolveEarlyAuxBitcodePaths(config);
    verifyEarlyAuxSha(mod, earlyAuxPaths);
  }
  auto savedTemp = handleSaveTemps(mod, config);
  if (failed(savedTemp)) {
    if (runtime)
      runtime->emitEvent("MAIN", "compile_scope", "end", "fail");
    return failure();
  }

  auto hivmcConfig = HIVMCFromBiShengIRConfig(config, runtime);
  auto res = runHIVMCCompileA3(mod, hivmcConfig);
  if (res.failed()) {
    mod.emitError("External hivmc run fails, returning module before running "
                  "external compiler");
    if (runtime)
      runtime->emitEvent("MAIN", "compile_scope", "end", "fail");
    return failure();
  }

  // Runtime drains warmup tickets + joins workers in ~CompileScope at main()
  // scope exit.  Emit the compile_scope end marker here for trace clarity.
  if (runtime)
    runtime->emitEvent("MAIN", "compile_scope", "end");
  return OwningModuleRef(mod);
}

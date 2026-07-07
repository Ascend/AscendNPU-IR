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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/RetriablePassManager/RetriablePassManager.h"
#include "bishengir/Tools/RetriablePassManager/CbufOverflowRetryPolicy.h"
#include "bishengir/Tools/RetriablePassManager/CcOverflowRetryPolicy.h"
#include "bishengir/Tools/RetriablePassManager/TuningRetryPolicy.h"
#include "bishengir/Tools/RetriablePassManager/UbOverflowRetryPolicy.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/PassPipeline.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
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
#include <regex>
#include <set>
#include <vector>
#include <set>

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {

/// Get the lib directory path (../lib relative to bishengir-compile
/// executable). Returns canonical absolute path without ".." or ".".
std::string getLibDirFromExecutable(StringRef executablePath) {
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
void addBitcodeAttrsToModule(ModuleOp module, StringRef executablePath,
                             const BiShengIRCompileMainConfig &config) {
  auto version = bishengir::parseHIVMCVersion(config.getHIVMCVersion());
  if (!version.has_value() || version.value().empty() ||
      version.value().getAsString() == "0.1.0")
    return;
  std::string libDir = getLibDirFromExecutable(executablePath);
  MLIRContext *ctx = module->getContext();

  using CreateAttrFn =
      std::function<mlir::Attribute(MLIRContext *, mlir::StringAttr)>;
  auto addIfExists = [&](const char *filename, llvm::StringRef attrName,
                         CreateAttrFn createAttr) {
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

  addIfExists("meta_op.aic.bc", mlir::hivm::AIC_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::AIC_BITCODEAttr::get(c, s);
              });
  addIfExists("meta_op.aiv.bc", mlir::hivm::AIV_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::AIV_BITCODEAttr::get(c, s);
              });
  addIfExists("meta_op.mix.aic.bc", mlir::hivm::MIX_AIC_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::MIX_AIC_BITCODEAttr::get(c, s);
              });
  addIfExists("meta_op.mix.aiv.bc", mlir::hivm::MIX_AIV_BITCODEAttr::name,
              [](MLIRContext *c, mlir::StringAttr s) -> mlir::Attribute {
                return mlir::hivm::MIX_AIV_BITCODEAttr::get(c, s);
              });
  addIfExists("host.bc", mlir::hivm::HOST_BITCODEAttr::name,
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

/// Get the HIVMC binary name.
StringRef getHIVMCName() {
  const char *kBiShengIRHIVMBinaryName = "hivmc";
  return kBiShengIRHIVMBinaryName;
}

std::vector<std::string> skipOptions(const std::vector<std::string> &options,
                                     const std::set<std::string> &skip) {
  std::vector<std::string> result;
  for (const std::string &arg : options) {
    StringRef argRef = arg;
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
    if (skip.count(trimArg) != 0) {
    if (!BiShengIRCompileMainConfig::isSharedWithDownstreamToolchain(
            trimArg)) {
      continue;
    }
    result.push_back(arg);
  }
  return result;
}

std::vector<std::string>
skipDebugOptions(const std::vector<std::string> &options) {
  std::set<std::string> debugOptions = {"debug", "debug-only",
                                        "mlir-print-ir-before-all",
                                        "mlir-print-ir-after-all"};
  return skipOptions(options, debugOptions);
}

std::vector<std::string>
getCompatibleOptions(const std::vector<std::string> &arguments,
                     const BiShengIRCompileMainConfig &config) {
  std::vector<std::string> options = arguments;
  // if enabled, skip debug options for compatibility.
  options = skipDebugOptions(options);
  // TODO: support hivmc compatibility for different versions
  auto version = bishengir::parseHIVMCVersion(config.getHIVMCVersion());
  if (!version.has_value() || version.value().empty()) {
    // null or empty version means we are using unknown or legacy hivmc
    // 1. legacy hivmc does not support debug or print
    options = skipDebugOptions(options);
    // 2. legacy hivmc has to manually enable triton compile pipeline
    if (config.getEnableTritonKernelCompile()) {
      options.push_back("--enable-triton-kernel-compile=true");
    }
    // 3. legacy hivmc has some unsupported options
    std::set<std::string> unsupported = {"enable-lir-compile",
                                         "enable-cpu-trace-intrinsic",
                                         "link-aicore-bitcode"};
    options = skipOptions(options, unsupported);
  } else if (version.value().getAsString() == "0.1.0") {
    // 0.1.0 version means we are using legacy hivmc
    std::set<std::string> unsupported = {"link-aicore-bitcode"};
    options = skipOptions(options, unsupported);
  }
  return options;
}

LogicalResult runExternalHIVMC(ModuleOp module,
                               const BiShengIRCompileMainConfig &config) {
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
    // Make sure module.hivm.opt.mlir will not be deleted.
    inputFileHandler->keep();
  // If --save-temps is not set, use a temporary directory for module.hivm.opt.mlir
  } else {
    inputFileHandler = getTempFile(inputFile, tempDirsStore);
    if (!inputFileHandler) {
      llvm::dbgs() << "[ERROR] Failed to create temporary input file needed to run hivm compile.\n";
      llvm::dbgs() << "[ERROR] Failed to create temporary input file needed to run hivmc compile.\n";
      return failure();
    }
  }
  inputFile = inputFileHandler->outputFilename();

  std::string content;
  llvm::raw_string_ostream buffer(content);
  module.print(buffer,
               mlir::OpPrintingFlags().enableDebugInfo(
                   config.getEnableSanitizer() || config.getEnableDebugInfo()));

  // TODO: Once version 0.2.0 is released, warning should be added to notice the
  // user upgrade the hivmc version.
  // TODO: Once version 0.1.0 is not supported, the following regex should be
  // removed.
  std::regex re("hacc\\.(hivmc_compatible_print|hivmc_version)[^,]*,");
  std::string modified = std::regex_replace(content, re, "");

  inputFileHandler->os() << modified;
  inputFileHandler->os().flush();

  std::vector<std::string> arguments;
  arguments.emplace_back("");
  arguments.push_back(inputFile);

  auto hivmcArgs = getCompatibleOptions(config.getHIVMCArgsDashDash(), config);
  arguments.insert(arguments.end(), hivmcArgs.begin(), hivmcArgs.end());
  arguments.emplace_back("-o");
  arguments.push_back(outputFile);
  SmallVector<StringRef> argumentsRef(arguments.begin(), arguments.end());
  if (failed(execute(getHIVMCName(), getBiShengInstallPath(), argumentsRef))) {
  module.print(inputFileHandler->os(), mlir::OpPrintingFlags().enableDebugInfo(
                                           config.getEnableSanitizer() ||
                                           config.getEnableDebugInfo()));
  inputFileHandler->os().flush();

  std::vector<std::string> arguments;
  arguments.push_back("");
  arguments.push_back(inputFile);

  // HIVMCArgs is populated by collectHIVMCArgs from (a) the auto-collected
  // cl::opt sweep, which only emits options whose Options.td entry has
  // isSharedWithDownstreamToolchain = 1 (already filtered at codegen time),
  // and (b) verbatim user --hivmc-args content (intentional passthrough).
  // filterSharedHIVMCOptions would only ever strip (b), which is exactly the
  // case where users want diagnostic flags like -debug-only, -stats, or
  // -print-after-all to reach hivmc; skip the filter on this path.
  auto hivmcOptions = config.getHIVMCArgsDashDash();
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
                                     "vf-fusion-mode",
                                     "disable-vf-reachable-check",
                                     "enable-sink-dpx-load"};
  auto skippedArgs = skipOptions(arguments, blacklist);

  SmallVector<StringRef> argumentsRef(skippedArgs.begin(), skippedArgs.end());
  if (failed(execute(getHIVMCName(), getBiShengIRHIVMCompileInstallPath(),
                     argumentsRef))) {
    return failure();
  }

  return success();
}

} // namespace

FailureOr<OwningModuleRef>
bishengir::runBiShengIRPipeline(ModuleOp mod,
                                BiShengIRCompileMainConfig config) {
  MLIRContext *ctx = mod->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<std::unique_ptr<Diagnostic>> collectedDiagnostics;

  // Resolve hivmc backward compatibility
  auto versionMaybe = detectHIVMCVersion(getHIVMCName());
  if (versionMaybe.has_value()) {
    llvm::VersionTuple hivmcVersion = versionMaybe.value();
    config.setHIVMCVersion(hivmcVersion.getAsString());
  } else {
    // Not return failure directly to support run compile without hivmc.
    // Let user to specify hivmc version by commandline.
    llvm::dbgs() << "[WARNING] Failed to detect hivmc version for backward "
                    "compatibility\n";
  }

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
  bool hirCompileSuccess =
      succeeded(retriablePm.runWithRetry(mod, buildPipeline, "BiShengHIR",
                                         collectedDiagnostics,
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
bool runSIMTToLLVMCompile(ArrayRef<ModuleOp> modules,
                          BiShengIRCompileMainConfig config) {
  config.setPureSimt(true);
  bool result = true;
  for (auto module : modules) {
    // Stop SIMT lowering after earlier pipeline fails.
    result = result && runPipeline(module, buildSIMTPipeline, config,
                                   "BiShengSIMT").succeeded();
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

  // VF fusion may cause ub overflow. When that happens we fall back through
  // the vector-side tiers (VFFusion / VF-reachable-check / tight-coupled
  // buffer), all of which are only meaningful when the offender is UB, so
  // they are gated on hasUboverflow.
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
  int tryTimes = 5;
  // triton compile has nothing to do with HFusion auto schedule, so we don't
  // need to tune for it.
  //
  // TODO: refactor this ad-hoc retry loop into a dedicated retryPassManager
  // so each fallback policy is composable and explicit. Planned policies:
  //   - OpFusion retry policy: bump tiling-max-counter each attempt, up to 5
  //     retries (the current default branch).
  //   - AutoBlockify retry policy: progressively disable hoisting.
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
      // Do not use `success &= ...` here: `&=` evaluates the RHS.
      success = success && succeeded(runPipeline(
                               hirCompileMode, buildBiShengHIRPipeline, config,
                               "BiShengHIR"));
      // extract main module and simt modules
      auto [mainMod, simtMods] = getMixedModules(hirCompileMode);
      // run ttir pipeline on simt modules
      for (auto simtMod : simtMods) {
        // Stop TTIR lowering after earlier pipeline fails.
        success = success && succeeded(runPipeline(
                                 simtMod, buildBiShengTTIRPipeline, config,
                                 "BiShengTTIR"));
      }
      success = success && succeeded(runPipeline(
                               hirCompileMode, buildBiShengHIRFinishPipeline,
                               config, "BishengHIR"));
      // Stop final HIVM lowering after earlier pipeline fails.
      success = success && succeeded(runPipeline(
                               mainMod, buildFinalHIVMPipelines, config,
                               "buildFinalHIVMPipelines"));
    } else if (config.getEnableTritonIRCompile()) {
      success = succeeded(runPipeline(hirCompileMode, buildBiShengTTIRPipeline,
                                      config, "BiShengTTIR"));
    } else {
      success = succeeded(runPipeline(hirCompileMode, buildBiShengHIRPipeline,
                                      config, "BiShengHIR"));
      // Stop final HIVM lowering after earlier pipeline fails.
      success = success &&
                succeeded(runPipeline(hirCompileMode, buildFinalHIVMPipelines,
                                      config, "buildFinalHIVMPipelines"));
      if (!success && hasUboverflow) {
        // Vector-side fallback tiers for UB overflow. These do not touch
        // multi-buffer and are tried in order until one is applicable.
        bool clear = false;
        if (config.getEnableVFFusion()) {
          LDBG("ub overflow detected at attempt "
               << (i + 1) << "/" << tryTimes
               << ", fallback with disabled vffusion");
          clear = true;
          config.setEnableVFFusion(false);
        } else if (!config.getDisableVFReachableCheck()) {
          LDBG("ub overflow detected at attempt "
               << (i + 1) << "/" << tryTimes
               << ", fallback with disabled VF reachable check");
          clear = true;
          config.setDisableVFReachableCheck(true);
        }
        if (clear && i != tryTimes)
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
      // Stop SIMT lowering after earlier pipeline fails.
      success = success && runSIMTToLLVMCompile(hirCompileMode, config);
    } else {
      // Stop SIMD lowering after earlier pipeline fails.
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
  // Skip for legacy hivmc (version 0.1.0 or empty) which does not support it.
  addBitcodeAttrsToModule(mod, config.getExecutablePath(), config);

  auto res = runExternalHIVMC(mod, config);
  if (res.failed()) {
    mod.emitError("External hivmc run fails, returning module before running "
                  "external compiler");
    return failure();
  }

  return OwningModuleRef(mod);
  if (config.shouldEnableCPURunner()) {
    auto fileHandle = mlir::openOutputFile(config.getOutputFile());
    assert(fileHandle != nullptr);
    fileHandle->os() << mod << '\n';
    fileHandle->keep();
    return success();
  }

  return success();
}

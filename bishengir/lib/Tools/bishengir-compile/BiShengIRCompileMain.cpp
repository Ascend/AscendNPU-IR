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
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/PassPipeline.h"

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
  std::vector<Diagnostic> collectedDiagnostics;

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
    collectedDiagnostics.emplace_back(std::move(diag));
  });

  bool hirCompileSuccess = false;
  int tryTimes = config.getEnableTuningMode() ? 1 : 5;
  for (int i = 0; i < tryTimes; i++) {
    LDBG("Attempt number: " << i << " with max buffer count tuning delta: "
                            << config.getHfusionMaxBufferCountTuning());

    ModuleOp hirCompileModule = cast<ModuleOp>(mod->clone());
    auto buildPipeline =
        std::bind(buildBiShengHIRPipeline, std::placeholders::_1, config);
    if (succeeded(runPipeline(hirCompileModule, buildPipeline, config,
                              "BiShengHIR"))) {
      hirCompileSuccess = true;
      mod.erase();
      mod = hirCompileModule;
      break;
    }
    hirCompileModule.erase();

    // increase max buffers by 2 in HFusion auto schedule
    config.increaseMaxBufferCountTuning(2);
  }

  // Restore to the default handler.
  diagEngine.eraseHandler(handlerID);
  for (auto &diag : llvm::reverse(collectedDiagnostics)) {
    [[maybe_unused]] auto res = handleDiagnostic(diag);
  }

  if (!hirCompileSuccess) {
    for (auto &diag : llvm::reverse(collectedDiagnostics)) {
      diagEngine.emit(std::move(diag));
    }
    return failure();
  }

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
  if (res.failed())
    mod.emitWarning("External hivmc run fails, returning module before running "
                    "external compiler");

  return OwningModuleRef(mod);
}

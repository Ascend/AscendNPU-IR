//===- Utility.cpp - BiShengIR pass pipeline Utility ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/bishengir-compile/Utility.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

/// Get the HIVMC binary name.
StringRef getHIVMCName() {
  const char *kHIVMCBinaryName = "hivmc-a5";
  return kHIVMCBinaryName;
}

std::string getBiShengIRHIVMCompileInstallPath() {
  const char *kBiShengInstallPathEnv = "BISHENG_INSTALL_PATH";
  const char *kBiShengInstallPath = getenv(kBiShengInstallPathEnv);
  if (kBiShengInstallPath) {
    return kBiShengInstallPath;
  }
  return "";
}

LogicalResult execute(StringRef binName, StringRef installPath,
                      SmallVectorImpl<StringRef> &arguments) {
  std::string binPath;
  if (!installPath.empty()) {
    if (auto binPathOrErr =
            llvm::sys::findProgramByName(binName, {installPath})) {
      binPath = binPathOrErr.get();
    } else {
      llvm::errs() << "[WARNING] Cannot find " << binName << " under "
                   << installPath << "\n";
    }
  }
  if (binPath.empty()) {
    if (auto binPathOrErr = llvm::sys::findProgramByName(binName)) {
      binPath = binPathOrErr.get();
    } else {
      llvm::errs() << "[ERROR] Cannot find " << binName << " under "
                   << "$PATH \n";
      return failure();
    }
  }
  arguments[0] = binPath;

  LLVM_DEBUG({
    llvm::dbgs() << "[DEBUG] Executing: ";
    llvm::interleave(
        arguments, llvm::dbgs(),
        [](const StringRef &arg) { llvm::dbgs() << arg; }, " ");
    llvm::dbgs() << "\n";
  });

  if (llvm::sys::ExecuteAndWait(binPath, arguments) != 0) {
    llvm::errs() << "[ERROR] Executing: ";
    llvm::interleave(
        arguments, llvm::errs(),
        [](const StringRef &arg) { llvm::errs() << arg; }, " ");
    llvm::errs() << "\n";
    return failure();
  }
  return success();
}

LogicalResult checkOptionValidity(const BiShengIRCompileMainConfig &config) {
  const std::string outputFile = config.outputFile();
  if (outputFile == "-")
    return success();

  auto filename = llvm::sys::path::filename(outputFile);
  if (filename == "/" || filename == "." || filename.empty()) {
    llvm::errs() << "[ERROR] Invalid output file path: " << outputFile << "\n";
    return failure();
  }

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty() && !llvm::sys::fs::exists(parentPath)) {
    if (llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "[ERROR] Can not create parent path: " << parentPath.str()
                   << "\n";
      return failure();
    }
  }

  return success();
}

/// This is a utility function to run a pre-constructed pass pipeline on the
/// input module.
LogicalResult
runPipeline(ModuleOp mod,
            const std::function<void(mlir::PassManager &,
                                     const BiShengIRCompileMainConfig &)>
                &buildPipeline,
            const BiShengIRCompileMainConfig &config,
            const std::string &pipelineName) {
  bishengir::BiShengIRPassManager passManager(mod->getContext(),
                                              ModuleOp::getOperationName(),
                                              OpPassManager::Nesting::Implicit);
  buildPipeline(passManager, config);

  // Apply MLIR PassManager command line options.
  // Ignore the result because the invocation point of this function might not
  // necessarily be the command line, so the options might not be loaded.
  (void)mlir::applyPassManagerCLOptions(passManager);
  (void)bishengir::applyPassManagerCLOptions(passManager);

  if (failed(passManager.run(mod))) {
    return mod->emitError("Failed to run " + pipelineName + " pipeline\n");
  }

  return success();
}

void TempDirectoriesStore::assertInsideTmp(StringTmpPath path) const {
  llvm::cantFail(llvm::errorCodeToError(canonicalizePath(path)),
                 "failed to canonicalize temp path.");
  if (!path.starts_with("/tmp")) {
    llvm_unreachable("unexpected temp folder created outside of /tmp");
  }
}

std::unique_ptr<llvm::ToolOutputFile>
getTempFile(const std::string &outputFile,
            TempDirectoriesStore &tempDirsStore) {
  if (outputFile == "-") {
    return nullptr;
  }

  StringTmpPath path;
  std::error_code ec =
      llvm::sys::fs::createUniqueDirectory("bishengir-compile", path);
  if (ec) {
    llvm::errs() << "[ERROR] Failed to generate temporary directory.\n";
    return nullptr;
  }

  tempDirsStore.dirs.push_back(path);
  LLVM_DEBUG(tempDirsStore.dirs.pop_back());

  std::string errorMessage;
  llvm::sys::path::append(path, llvm::sys::path::filename(outputFile));
  auto tempFile = openOutputFile(path, &errorMessage);
  if (!tempFile) {
    llvm::errs() << "[ERROR] " << errorMessage << "\n";
    return nullptr;
  }

  LLVM_DEBUG(tempFile->keep());
  return tempFile;
}

std::error_code canonicalizePath(StringTmpPath &path) {
  if (path == "-") {
    return {};
  }
  std::error_code errorCode = llvm::sys::fs::make_absolute(path);
  if (errorCode)
    return errorCode;
  llvm::sys::path::remove_dots(path, /*removedotdot*/ true);
  return {};
}

MixedModules getMixedModules(ModuleOp topMod) {
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
 
bool hasSplitModules(ModuleOp topMod) {
  return !topMod.getOps<ModuleOp>().empty();
}

llvm::LogicalResult
inferMixedCV(ModuleOp &module, bishengir::BiShengIRCompileMainConfig &config) {
  // check scope
  if (module.walk([](scope::ScopeOp) { return mlir::WalkResult::interrupt(); })
          .wasInterrupted()) {
    return success();
  }

  auto funcs = llvm::make_filter_range(
      module.getOps<func::FuncOp>(),
      [](const func::FuncOp &func) { return func->hasAttr("mix_mode"); });
  if (funcs.empty()) {
    module.emitWarning()
        << "[WARNING] No function with attribute mix_mode found in this module";
    return success();
  }

  bool foundDotScaled = false;
  module.walk<WalkOrder::PreOrder>([&](Operation* op) -> WalkResult {
    if (isa<hfusion::MatMulMxOp>(op)) {
      foundDotScaled = true;
      return WalkResult::interrupt();
    } else if (isa<func::FuncOp>(op)) {
      const auto func = cast<func::FuncOp>(op);
      if (func->hasAttr("IsDotScaleKernel")) {
        foundDotScaled = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (foundDotScaled) {
    config.mixedCV(false);
    return success();
  }
  
  auto first =
      (*funcs.begin())->getAttrOfType<StringAttr>("mix_mode").getValue();
  if (!llvm::all_of(funcs, [&](const func::FuncOp &func) {
        return func->getAttrOfType<StringAttr>("mix_mode").getValue() == first;
      }))
    return failure();

  config.mixedCV(first != StringRef{"aiv"});
  return success();
}

llvm::LogicalResult
inferDotScale(ModuleOp &module, bishengir::BiShengIRCompileMainConfig &config) {
  module.walk([&](hfusion::MatMulMxOp op) {
    config.compileDotScaled(true);
    return WalkResult::interrupt();
  });
  return success();
}

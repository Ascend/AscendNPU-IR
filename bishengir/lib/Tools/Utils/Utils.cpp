//===- Utils.cpp - BiShengIR Tools Common Utils ------------------*- C++-*-===//
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

#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/BiShengIRConfigBase/Config.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bishengir-tools-utils"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace mlir;
using namespace llvm;
using namespace bishengir;

LogicalResult bishengir::runPipeline(
    ModuleOp mod, const std::function<void(mlir::PassManager &)> &buildPipeline,
    const BiShengIRCompileConfigBase &config, const std::string &pipelineName) {
  bishengir::BiShengIRPassManager passManager(config, mod->getContext(),
                                              ModuleOp::getOperationName(),
                                              OpPassManager::Nesting::Implicit);
  buildPipeline(passManager);

  // Apply MLIR PassManager command line options.
  // Ignore the result because the invocation point of this function might not
  // necessarily be the command line, so the options might not be loaded.
  (void)mlir::applyPassManagerCLOptions(passManager);
  (void)bishengir::applyPassManagerCLOptions(passManager);

  if (failed(passManager.run(mod)))
    return mod->emitError("Failed to run " + pipelineName + " pipeline\n");

  return success();
}

llvm::LogicalResult
bishengir::checkInOutOptionsValidity(BiShengIRCompileConfigBase &config) {
  std::string inputFile = config.getInputFile();
  StringTmpPath inputPath(config.getInputFile());
  if (llvm::errorCodeToError(canonicalizePath(inputPath))) {
    llvm::errs() << "[ERROR] Failed to canonicalize input file path: "
                 << inputFile << "\n";
    return failure();
  }
  config.setInputFile(inputPath.str().str());

  std::string outputFile = config.getOutputFile();
  StringTmpPath outputPath(outputFile);
  if (llvm::errorCodeToError(canonicalizePath(outputPath))) {
    llvm::errs() << "[ERROR] Failed to canonicalize output file path: "
                 << outputFile << "\n";
    return failure();
  }

  config.setOutputFile(outputPath.str().str());
  outputFile = config.getOutputFile();
  auto filename = llvm::sys::path::filename(outputFile);
  if (filename == "/" || filename == "." || filename.empty()) {
    llvm::errs() << "[ERROR] Invalid output file path: " << filename << "\n";
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

std::error_code bishengir::canonicalizePath(StringTmpPath &path) {
  if (path == "-") {
    return {};
  }
  std::error_code errorCode = llvm::sys::fs::make_absolute(path);
  if (errorCode)
    return errorCode;
  llvm::sys::path::remove_dots(path, /*removedotdot*/ true);
  return {};
}

void TempDirectoriesStore::assertInsideTmp(StringTmpPath path) const {
  llvm::cantFail(llvm::errorCodeToError(canonicalizePath(path)),
                 "failed to canonicalize temp path.");
  if (!path.starts_with("/tmp")) {
    llvm_unreachable("unexpected temp folder created outside of /tmp");
  }
}

TempDirectoriesStore::~TempDirectoriesStore() {
  for (auto &dir : dirs) {
    assertInsideTmp(dir);
    llvm::sys::fs::remove_directories(dir, true);
  }
}

std::unique_ptr<llvm::ToolOutputFile>
bishengir::getTempFile(const std::string &outputFile,
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

LogicalResult bishengir::execute(StringRef binName, StringRef installPath,
                                 SmallVectorImpl<StringRef> &arguments,
                                 std::optional<llvm::StringRef> outputFile,
                                 unsigned timeoutSeconds) {
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

  SmallVector<std::optional<StringRef>> redirects = {
      /*stdin(0)=*/std::nullopt, /*stdout(1)=*/outputFile,
      /*stderr(2)=*/std::nullopt};
  if (llvm::sys::ExecuteAndWait(binPath, arguments, /*Env=*/std::nullopt,
                                /*Redirects=*/redirects) != 0) {
    llvm::errs() << "[ERROR] Executing: ";
    llvm::interleave(
        arguments, llvm::errs(),
        [](const StringRef &arg) { llvm::errs() << arg; }, " ");
    llvm::errs() << "\n";
    return failure();
  }
  return success();
}

std::optional<llvm::VersionTuple>
bishengir::parseHIVMCVersion(llvm::StringRef content) {
  llvm::VersionTuple version;
  bool fail = version.tryParse(content.trim());
  if (!fail) {
    return version;
  }
  return std::nullopt;
}

std::optional<llvm::VersionTuple>
bishengir::parseHIVMCVersion(ArrayRef<StringRef> items) {
  for (StringRef item : items) {
    auto version = bishengir::parseHIVMCVersion(item);
    if (version.has_value()) {
      return version;
    }
  }
  return std::nullopt;
}

llvm::VersionTuple findHIVMCVersion(llvm::StringRef content) {
  SmallVector<StringRef> lines;
  int maxSplit = 100;
  content.split(lines, '\n', maxSplit, /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    // expected line to parse should contains "hivmc" and "version"
    if (!line.contains_insensitive("hivmc") ||
        !line.contains_insensitive("version")) {
      continue;
    }
    SmallVector<StringRef> items;
    content.split(items, ' ', maxSplit, /*KeepEmpty=*/false);
    auto version = bishengir::parseHIVMCVersion(items);
    if (version.has_value()) {
      return version.value();
    }
  }
  return llvm::VersionTuple();
}

std::optional<llvm::VersionTuple>
bishengir::detectHIVMCVersion(StringRef hivmcName) {

  TempDirectoriesStore tempDirsStore;
  auto fileHandler = getTempFile("version.out", tempDirsStore);
  if (!fileHandler) {
    llvm::dbgs() << "[ERROR] Failed to create temporary input file needed to "
                    "detect hivmc version.\n";
    return std::nullopt;
  }
  std::string fileName = fileHandler->outputFilename();
  fileHandler->os().flush();

  SmallVector<StringRef> args{"", "--version"};
  StringRef outputFile = fileHandler->getFilename();
  if (failed(execute(hivmcName, getBiShengInstallPath(), args, outputFile))) {
    llvm::dbgs() << "[ERROR] Failed to run `hivmc --version`.\n";
    return std::nullopt;
  }

  std::string errorMsg;
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      mlir::openInputFile(fileName, &errorMsg);
  if (!buffer) {
    llvm::dbgs() << "[ERROR] Failed to open file: " << fileName << "\n";
    return std::nullopt;
  }

  return findHIVMCVersion(buffer->getBuffer());
}

std::string bishengir::getBiShengInstallPath() {
  const char *kBiShengInstallPathEnv = "BISHENG_INSTALL_PATH";
  const char *kBiShengInstallPath = getenv(kBiShengInstallPathEnv);
  if (!kBiShengInstallPath) {
    llvm::dbgs() << "[DEBUG] BISHENG_INSTALL_PATH is not set.\n";
    return "";
  }

  llvm::SmallString<128> path;
  path.append(kBiShengInstallPath);
  std::error_code errorCode = llvm::sys::fs::make_absolute(path);
  if (errorCode)
    llvm::report_fatal_error("Failed to get absolute path for " + path +
                             " please verify its validity");
  return path.str().str();
}

mlir::LogicalResult bishengir::handleDiagnostic(const mlir::Diagnostic &diag) {
  auto &os = llvm::outs();
  auto loc = diag.getLocation();
  if (!llvm::isa<mlir::UnknownLoc>(loc)) {
    os << loc << ": ";
  }

  // Handle warning.
  if (diag.getSeverity() == mlir::DiagnosticSeverity::Warning) {
    os << "warning: " << diag.str() << "\n";
    return mlir::success();
  }

  // Any other severity - treat as failure.
  return mlir::failure();
}

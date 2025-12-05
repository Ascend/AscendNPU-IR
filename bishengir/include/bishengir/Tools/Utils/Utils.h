//===- Utils.h - BiShengIR Tools Common Utils --------------------*- C++-*-===//
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

#ifndef BISHENGIR_TOOLS_UTILS_UTILS_H
#define BISHENGIR_TOOLS_UTILS_UTILS_H

#include "bishengir/Tools/BiShengIRConfigBase/Config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VersionTuple.h"

namespace bishengir {

constexpr static unsigned kTmpMaxPath = 128;
using StringTmpPath = llvm::SmallString<kTmpMaxPath>;

enum class SubCoreTarget { AIC, AIV, HOST, MIX_AIC, MIX_AIV };

using IRModulePair =
    std::pair<std::unique_ptr<llvm::Module>, bishengir::SubCoreTarget>;

using IRFilePair =
    std::pair<std::unique_ptr<llvm::ToolOutputFile>, bishengir::SubCoreTarget>;

/// This is a utility function to run a pre-constructed pass pipeline on the
/// input module.
llvm::LogicalResult
runPipeline(mlir::ModuleOp mod,
            const std::function<void(mlir::PassManager &)> &buildPipeline,
            const bishengir::BiShengIRCompileConfigBase &config,
            const std::string &pipelineName);

// apply make_absolute and remove_dots on the given path.
std::error_code canonicalizePath(StringTmpPath &path);

struct TempDirectoriesStore {
  ~TempDirectoriesStore();

  void assertInsideTmp(StringTmpPath path) const;
  llvm::SmallVector<StringTmpPath> dirs;
};

std::unique_ptr<llvm::ToolOutputFile>
getTempFile(const std::string &outputFile, TempDirectoriesStore &tempDirsStore);

llvm::LogicalResult
checkInOutOptionsValidity(BiShengIRCompileConfigBase &config);

llvm::LogicalResult
execute(llvm::StringRef binName, llvm::StringRef installPath,
        llvm::SmallVectorImpl<llvm::StringRef> &arguments,
        std::optional<llvm::StringRef> outputFile = std::nullopt,
        unsigned timeoutSeconds = 15);

std::optional<llvm::VersionTuple> parseHIVMCVersion(llvm::StringRef content);

std::optional<llvm::VersionTuple>
parseHIVMCVersion(llvm::ArrayRef<llvm::StringRef> contents);

std::optional<llvm::VersionTuple> detectHIVMCVersion(llvm::StringRef hivmcName);

/// Get the path set by environment variable `BISHENG_INSTALL_PATH`.
std::string getBiShengInstallPath();

/// Prints a diagnostic to llvm::outs() and return a LogicalResult.
mlir::LogicalResult handleDiagnostic(const mlir::Diagnostic &diag);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_UTILS_UTILS_H
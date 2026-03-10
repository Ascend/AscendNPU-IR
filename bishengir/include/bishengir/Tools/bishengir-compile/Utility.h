//===- Utility.h - BiShengIR pass pipeline Utility --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_UTILITY_H
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_UTILITY_H

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/Config.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <numeric>
#include <regex>

constexpr static unsigned kTmpMaxPath = 128;

using StringTmpPath = llvm::SmallString<kTmpMaxPath>;
using MixedModules = std::pair<ModuleOp,SmallVector<ModuleOp,2>>;

/// Get the BiSheng HIVM Compiler binary name.
llvm::StringRef getBiShengIRHIVMCompilerName();

std::string getBiShengIRHIVMCompileInstallPath();

llvm::LogicalResult execute(llvm::StringRef binName,
                            llvm::StringRef installPath,
                            llvm::SmallVectorImpl<llvm::StringRef> &arguments);

llvm::LogicalResult
checkOptionValidity(const bishengir::BiShengIRCompileMainConfig &config);

/// This is a utility function to run a pre-constructed pass pipeline on the
/// input module.
llvm::LogicalResult runPipeline(
    mlir::ModuleOp mod,
    const std::function<void(mlir::PassManager &,
                             const bishengir::BiShengIRCompileMainConfig &)>
        &buildPipeline,
    const bishengir::BiShengIRCompileMainConfig &config,
    const std::string &pipelineName);

// apply make_absolute and remove_dots on the given path.
std::error_code canonicalizePath(StringTmpPath &path);

struct TempDirectoriesStore {
  llvm::SmallVector<StringTmpPath> dirs;
  ~TempDirectoriesStore() {
    for (auto &dir : dirs) {
      assertInsideTmp(dir);
      llvm::sys::fs::remove_directories(dir, true);
    }
  }
  void assertInsideTmp(StringTmpPath path) const;
};

std::unique_ptr<llvm::ToolOutputFile>
getTempFile(const std::string &outputFile, TempDirectoriesStore &tempDirsStore);

MixedModules getMixedModules(ModuleOp topMod);

bool hasSplitModules(ModuleOp topMod);

// FIXME: This will be refactored after vectorize is moved to HIVM.
llvm::LogicalResult inferMixedCV(ModuleOp &module,
                                 bishengir::BiShengIRCompileMainConfig &config);

llvm::LogicalResult inferDotScale(ModuleOp &module,
                                  bishengir::BiShengIRCompileMainConfig &config);

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_UTILITY_H

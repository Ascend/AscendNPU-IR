//===- Utility.h - BiShengIR HIVMC pass pipeline Utility --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_HIVMC_UTILITY_H
#define BISHENGIR_TOOLS_HIVMC_UTILITY_H

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Tools/hivmc/Config.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <functional>
#include <memory>
#include <numeric>
#include <regex>
#include <string>

constexpr static unsigned kTmpMaxPath = 128;
using StringTmpPath = llvm::SmallString<kTmpMaxPath>;

enum class SubCoreTarget { AIC, AIV, HOST, MIX_AIC, MIX_AIV };

template <typename KEYT, typename VALUET>
VALUET getOrDefault(const std::map<KEYT, VALUET> &map, KEYT key,
                    VALUET defaultV) {
  if (map.find(key) == map.end()) {
    return defaultV;
  }
  return map.at(key);
}

using IRModulePair = std::pair<std::unique_ptr<llvm::Module>, SubCoreTarget>;

using IRFilePair =
    std::pair<std::unique_ptr<llvm::ToolOutputFile>, SubCoreTarget>;

namespace bishengir {

struct DeviceTargetInfo {
  explicit DeviceTargetInfo(const std::string &arch, SubCoreTarget coreType)
      : coreType(coreType), arch(arch) {}
  explicit DeviceTargetInfo(SubCoreTarget coreType) : coreType(coreType) {}

  // Get the full arch string by combining arch with core type.
  std::string getFullArch() const;

  SubCoreTarget coreType{SubCoreTarget::HOST};
  std::string arch{""};
};

/// Trim prefix and postfix blank of string
std::string trimString(std::string src);

/// Split the string s with the separator string into the result string vectors
void splitString(const std::string &s, const std::string &separator,
                 std::vector<std::string> &res);

/// Given a target directory, this function will try to find a directory
/// ending with the target directory in the `$PATH` environment variable.
///
/// For example:
///   `targetDir` = "a/b"
///   `$PATH` = "a:b:c/a/b"
/// The function will return "c/a/b"
///
/// If multiple directory matches, return the first one. If none is found,
/// return NULL.
std::optional<std::string> findDirectoryInPath(llvm::StringRef targetDir);

/// Get the path to BiSheng Compiler.
/// First locate the BiSheng compiler's binary via the `BISHENG_INSTALL_PATH`
/// environment variable. If the compiler path is not present, search for a path
/// called `ccec_compiler` under PATH variable and return the full patch with
/// `bin` directory appended to it.
std::string getBiShengInstallPath();

/// Get the BiSheng Compiler binary name.
llvm::StringRef getBiShengCompilerName();

// TODO: Refactor.
llvm::StringRef getHostDebugPath();

// TODO: Refactor.
llvm::StringRef getAscendPath();

/// Modify the ir string for downgrade the llvm version
std::string modifyForVersionMismatch(std::string src);

std::string tryReplaceExtension(llvm::StringRef path,
                                llvm::StringRef newExtension);

std::string tryModifyFileName(llvm::StringRef path,
                              llvm::StringRef prepend = "",
                              llvm::StringRef append = "");

std::string tryAppendFileName(llvm::StringRef path, llvm::StringRef append);

std::optional<std::pair<mlir::ModuleOp, mlir::ModuleOp>>
splitMixModule(mlir::ModuleOp mod);

llvm::SmallVector<IRFilePair>
saveToFiles(const llvm::SmallVector<IRModulePair> &llvmModules,
            const std::string &outputFile, bool setKeepFlag,
            bool needToMangleOutputFilepath = false);

llvm::SmallVector<std::string> getLinkArgs();

bool isDebugOrShmemRelated(const ::llvm::StringRef &name);

bool isDebugOrShmemPresent(mlir::ModuleOp moduleOp /* don't need & */);

void setMetadataForIntrinsicCall(llvm::CallInst *llCall,
                                 llvm::StringRef apiName,
                                 llvm::StringRef stubManglingName);

bool getMetadataInfo(llvm::StringRef name,
                     std::pair<llvm::StringRef, llvm::StringRef> &info);

void attachMetadataToLLVMIR(llvm::Module &llMod);

llvm::LogicalResult
runPipeline(mlir::ModuleOp mod,
            const std::function<void(mlir::PassManager &,
                                     const bishengir::HIVMCMainConfig &)>
                &buildPipeline,
            const bishengir::HIVMCMainConfig &config,
            const std::string &pipelineName);

struct CompileTiming {
  mlir::DefaultTimingManager manager;
  std::unique_ptr<mlir::TimingScope> rootScope;
  CompileTiming *previousTiming = nullptr;

  CompileTiming();
  ~CompileTiming();
  mlir::TimingScope *getRootScope() const { return rootScope.get(); }
};

struct ExternalToolProfiler {
  static int run(llvm::StringRef name, std::function<int()> fn);
};

llvm::LogicalResult execute(llvm::StringRef binName,
                            llvm::StringRef installPath,
                            llvm::SmallVectorImpl<llvm::StringRef> &arguments);

llvm::LogicalResult
checkOptionValidity(const bishengir::HIVMCMainConfig &config);

/// Read bitcode path attributes from the ModuleOp.
/// Handles: aic_bitcode, aiv_bitcode, mix_aic_bitcode, mix_aiv_bitcode,
///          host_bitcode
/// Only returns entries where the attribute is set.
std::map<SubCoreTarget, std::string>
getBitcodePathsBySubCoreTarget(mlir::ModuleOp mod);

/// Get the absolute path of the current executable. Resolves symlinks and
/// handles invocation via PATH. \p argv0 is argv[0] from main, \p mainAddr
/// is reinterpret_cast<void*>(main). Returns empty string on failure.
std::string getExecutablePath(const char *argv0, void *mainAddr);

} // namespace bishengir

// apply make_absolute and remove_dots on the given path.
std::error_code hivmcCanonicalizePath(StringTmpPath &path);

#endif // BISHENGIR_TOOLS_HIVMC_UTILITY_H

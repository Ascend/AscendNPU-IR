//===- Utility.cpp - BiShengIR pass pipeline Utility ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/hivmc/Utility.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include <optional>
#include <utility>

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {
thread_local CompileTiming *currentCompileTiming = nullptr;

mlir::TimingScope *getCurrentCompileTimingScope() {
  return currentCompileTiming ? currentCompileTiming->getRootScope() : nullptr;
}
} // namespace

namespace bishengir {
std::string trimString(std::string src) {
  std::regex re("(^\\s*)|(\\s*$)");
  return std::regex_replace(src, re, "");
}

void splitString(const std::string &s, const std::string &separator,
                 std::vector<std::string> &res) {
  std::regex re(separator);
  std::regex_token_iterator<std::string::const_iterator> pos(s.begin(), s.end(),
                                                             re, -1);
  decltype(pos) end;
  for (; pos != end; pos++) {
    auto trimS = trimString(pos->str());
    res.push_back(trimS);
  }
}

std::optional<std::string> findDirectoryInPath(StringRef targetDir) {
  std::optional<std::string> pathEnv = sys::Process::GetEnv("PATH");
  if (!pathEnv.has_value())
    return nullptr;

  char separator = ':';
  SmallVector<StringRef, 8> paths;
  StringRef(*pathEnv).split(paths, separator);

  for (StringRef path : paths) {
    if (!path.ends_with(targetDir))
      continue;

    if (sys::fs::is_directory(path))
      return path.str();
  }
  return std::nullopt;
}

/// Get the path to BiSheng Compiler.
/// First locate the BiSheng compiler's binary via the `BISHENG_INSTALL_PATH`
/// environment variable. If the compiler path is not present, search for a path
/// called `ccec_compiler` under PATH variable and return the full patch with
/// `bin` directory appended to it.
std::string getBiShengInstallPath() {
  const char *kBiShengInstallPathEnv = "BISHENG_INSTALL_PATH";
  const char *kBiShengInstallPath = getenv(kBiShengInstallPathEnv);
  if (kBiShengInstallPath) {
    std::string strPath(kBiShengInstallPath);
    return strPath;
  }

  const char *kBiShengInstallPathName = "ccec_compiler";
  SmallString<128> bishengInstallPathStr(kBiShengInstallPathName);
  sys::path::append(bishengInstallPathStr, "bin");
  std::optional<std::string> maybePath =
      findDirectoryInPath(bishengInstallPathStr.str());
  if (maybePath.has_value()) {
    return *maybePath;
  }

  llvm::errs()
      << "[ERROR] Cannot find the binary path of the BiSheng compiler.\n";
  return "";
}

/// Get the BiSheng Compiler binary name.
StringRef getBiShengCompilerName() {
  const char *kBiShengBinaryName = "bisheng";
  return kBiShengBinaryName;
}

// TODO: Refactor.
#if (!BISHENGIR_PUBLISH)
StringRef getHostDebugPath() {
  const char *kHostDebugPathEnv = "HOST_DEBUG_PATH";
  return getenv(kHostDebugPathEnv);
}
#endif

// TODO: Refactor.
StringRef getAscendPath() {
  const char *kAscendHomePathEnv = "ASCEND_HOME_PATH";
  return getenv(kAscendHomePathEnv);
}

/// Modify the ir string for downgrade the llvm version
std::string modifyForVersionMismatch(std::string src) {
  std::regex downgradeMemRe("memory\\(([^()]*)\\)");
  std::map<std::string, unsigned> rwMap = {
      {"readwrite", 3}, {"write", 2}, {"read", 1}, {"none", 0}};
  std::map<std::string, unsigned> locMap = {
      {"argmem", 1}, {"inaccessiblemem", 2}, {"other", 4}};
  std::map<unsigned, std::string> revRwMap = {
      {0, "readnone"}, {1, "readonly"}, {2, "writeonly"}};
  std::map<unsigned, std::string> revLocMap = {
      {1, "argmemonly"},
      {2, "inaccessiblememonly"},
      {3, "inaccessiblemem_or_argmemonly"}};
  std::string downgradeMemStr;
  std::sregex_token_iterator tbegin(src.begin(), src.end(), downgradeMemRe,
                                    /*submatches=*/{-1, 0});
  std::sregex_token_iterator tend;
  std::for_each(tbegin, tend, [&](const std::string &token) {
    std::smatch argMatch;
    if (!std::regex_search(token, argMatch, downgradeMemRe)) {
      downgradeMemStr += token;
      return;
    }
    std::vector<std::string> splitStrs;
    splitString(argMatch[1], ",", splitStrs);
    unsigned rw_bit_flag = 0;
    unsigned loc_bit_flag = 0;
    for (auto ss : splitStrs) {
      std::vector<std::string> pairStrs;
      splitString(ss, ":", pairStrs);
      if (pairStrs.size() == 1) {
        loc_bit_flag = loc_bit_flag | locMap["other"];
        rw_bit_flag = rw_bit_flag | rwMap[pairStrs[0]];
      } else {
        assert(pairStrs.size() == 2);
        loc_bit_flag = loc_bit_flag | locMap[pairStrs[0]];
        rw_bit_flag = rw_bit_flag | rwMap[pairStrs[1]];
      }
    }
    auto newRW = getOrDefault<unsigned, std::string>(revRwMap, rw_bit_flag, "");
    auto newLoc =
        getOrDefault<unsigned, std::string>(revLocMap, loc_bit_flag, "");
    downgradeMemStr += newRW + " " + newLoc;
  });
  std::regex downgradeStackRe(
      R"((?:llvm\.stackrestore|llvm\.stacksave)(\.\w+))");
  auto downgradeStacksaveStr =
      std::regex_replace(downgradeMemStr, downgradeStackRe, "");

  std::regex dbgValueRe(
      R"(#dbg_value\(([^,]+),\s*([^,]+),\s*(!DIExpression\([^)]*\)),\s*([^)]+)\))");

  std::string dbgValuePattern = "call void @llvm.dbg.value(metadata $1, "
                                "metadata $2, metadata $3), !dbg $4";
  auto downgradeDbgValueStr =
      std::regex_replace(downgradeStacksaveStr, dbgValueRe, dbgValuePattern);

  std::regex dbgDeclareRe(
      R"(#dbg_declare\(([^,]+),\s*([^,]+),\s*(!DIExpression\([^)]*\)),\s*([^)]+)\))");

  std::string dbgDeclarePattern = "call void @llvm.dbg.declare(metadata $1, "
                                  "metadata $2, metadata $3), !dbg $4";
  auto downgradeDbgDeclareStr =
      std::regex_replace(downgradeDbgValueStr, dbgDeclareRe, dbgDeclarePattern);

  std::regex dlvRegex(
      R"((!+\d+\s*=\s*!DILocalVariable\([^)]*?)(\)))"
  );
  return std::regex_replace(downgradeDbgDeclareStr, dlvRegex, "$1, attrtype: 64$2");
}

std::string tryReplaceExtension(StringRef path, StringRef newExtension) {
  if (path == "-") {
    return std::string(path);
  }
  SmallVector<char> result(path.begin(), path.end());
  llvm::sys::path::replace_extension(result, newExtension);
  return std::string(result.data(), result.size());
}

std::string tryModifyFileName(StringRef path, StringRef prepend,
                              StringRef append) {
  if (append.empty() && prepend.empty()) {
    return path.str();
  }

  StringRef parentPath = llvm::sys::path::parent_path(path);
  StringRef stem = llvm::sys::path::stem(path);
  StringRef extension = llvm::sys::path::extension(path);

  std::string result = parentPath.str();
  if (!result.empty()) {
    result += llvm::sys::path::get_separator().str();
  }

  return result + prepend.str() + stem.str() + append.str() + extension.str();
}

std::string tryAppendFileName(StringRef path, StringRef append) {
  return tryModifyFileName(path, "", append);
}

/// Split mix module into one aic module and one aiv module
std::optional<std::pair<ModuleOp, ModuleOp>> splitMixModule(ModuleOp mod) {
  SmallVector<Operation *> aicPart;
  SmallVector<Operation *> aivPart;
  llvm::SmallSet<Operation *, 32>
      aicvSharedPart; // with linkages of private, extern_weak, etc.

  bool fail{false};
  for (Operation &op : mod) {
    auto coreTypeAttr = dyn_cast_if_present<hivm::TFuncCoreTypeAttr>(
        op.getAttr(hivm::TFuncCoreTypeAttr::name));
    // For global operations that do not have core, just place them together
    // with the parent function of their user operations.
    if (!coreTypeAttr) {
      if (auto global = dyn_cast<LLVM::GlobalOp>(op)) {
        if (auto uses = SymbolTable::getSymbolUses(global, mod)) {
          for (const SymbolTable::SymbolUse &use : *uses) {
            Operation *user = use.getUser();
            if (!user)
              continue;
            Operation *funcOp = user->getParentOfType<LLVM::LLVMFuncOp>();
            if (!funcOp)
              continue;
            auto funcTypeAttr = dyn_cast_if_present<hivm::TFuncCoreTypeAttr>(
                funcOp->getAttr(hivm::TFuncCoreTypeAttr::name));
            if (!funcTypeAttr)
              continue;

            if (!coreTypeAttr) {
              coreTypeAttr = funcTypeAttr;
            } else if (coreTypeAttr.getFuncCoreType() !=
                       funcTypeAttr.getFuncCoreType()) {
              coreTypeAttr = hivm::TFuncCoreTypeAttr::get(
                  global.getContext(), hivm::TFuncCoreType::AIC_OR_AIV);
              break;
            }
          }
        }
      }
    }
    if (!coreTypeAttr) {
      fail = true;
      llvm::errs() << "[ERROR] Unknown core type: " << op << "\n";
      continue;
    }
    // Currently mix core is not supported, but if we want to export as DAG,
    // it's ok
    if (coreTypeAttr.getFuncCoreType() == hivm::TFuncCoreType::MIX &&
        !hacc::isMixEntry(mod)) {
      fail = true;
      llvm::errs() << "[ERROR] Op still have TFuncCoreType::MIX attribute "
                      "when lowering to LLVMIR: "
                   << op << "\n";
      continue;
    }

    if (coreTypeAttr.getFuncCoreType() == hivm::TFuncCoreType::AIC) {
      aicPart.push_back(&op);
    } else if (coreTypeAttr.getFuncCoreType() == hivm::TFuncCoreType::AIV) {
      aivPart.push_back(&op);
    } else if (coreTypeAttr.getFuncCoreType() ==
               hivm::TFuncCoreType::AIC_OR_AIV) {
      aicPart.push_back(&op);
      aivPart.push_back(&op);
      aicvSharedPart.insert(&op);
    } else {
      llvm_unreachable("unsupported TFuncCoreType");
    }
  }

  if (fail) {
    return std::nullopt;
  }

  std::string moduleName = mod.getName().value_or("").str();
  ModuleOp aicModule = ModuleOp::create(mod.getLoc(), moduleName + "_mix_aic");
  ModuleOp aivModule = ModuleOp::create(mod.getLoc(), moduleName + "_mix_aiv");

  hivm::setModuleCoreTypeAttr(aicModule, hivm::TModuleCoreType::AIC);
  hivm::setModuleCoreTypeAttr(aivModule, hivm::TModuleCoreType::AIV);

  OpBuilder builder(mod.getOperation()->getContext()); // for cloning shared Ops
  for (auto *op : aicPart) {
    if (aicvSharedPart.count(op) != 0) {
      Operation *opCopy = builder.clone(*op);
      opCopy->remove();
      aicModule.insert(aicModule.end(), opCopy);
    } else {
      op->remove();
      aicModule.insert(aicModule.end(), op);
    }
  }
  for (auto *op : aivPart) {
    if (aicvSharedPart.count(op) != 0) {
      Operation *opCopy = builder.clone(*op);
      opCopy->remove();
      aivModule.insert(aivModule.end(), opCopy);
    } else {
      op->remove();
      aivModule.insert(aivModule.end(), op);
    }
  }
  for (auto *op : aicvSharedPart) {
    op->remove();
  }

  return std::make_pair(aicModule, aivModule);
}

SmallVector<IRFilePair>
saveToFiles(const SmallVector<IRModulePair> &llvmModules,
            const std::string &outputFile, bool setKeepFlag,
            bool needToMangleOutputFilepath) {
  SmallVector<IRFilePair> result;

  needToMangleOutputFilepath =
      needToMangleOutputFilepath || (llvmModules.size() > 1);

  for (auto &[llvmModule, target] : llvmModules) {
    std::string append;
    if (needToMangleOutputFilepath) {
      if (target == SubCoreTarget::AIC) {
        append = "_mix_aic";
      } else if (target == SubCoreTarget::AIV) {
        append = "_mix_aiv";
      } else if (target == SubCoreTarget::HOST) {
        append = "_host";
      } else {
        llvm_unreachable("Not all subcore target handled");
      }
    }

    std::string llOutputFile = outputFile;
    if (outputFile != "-") {
      llOutputFile =
          tryAppendFileName(tryReplaceExtension(outputFile, "ll"), append);
    }

    std::string moduleStr;
    llvm::raw_string_ostream rso(moduleStr);
    llvmModule->print(rso, nullptr);
    rso << "\n";
    std::string newModuleStr = modifyForVersionMismatch(moduleStr);

    // TODO: how to handle stdout for multiple files
    std::string errorMessage;
    std::unique_ptr<llvm::ToolOutputFile> tempLLVMFile =
        openOutputFile(llOutputFile, &errorMessage);

    if (!tempLLVMFile) {
      llvm::errs() << "open file for " << llOutputFile
                   << " failed : " << errorMessage << "\n";
      result.push_back(std::make_pair(nullptr, target));
      break;
    }

    tempLLVMFile->os() << newModuleStr;
    tempLLVMFile->os().flush();

    LLVM_DEBUG(setKeepFlag = true);
    if (setKeepFlag) {
      tempLLVMFile->keep();
    }

    result.push_back(std::make_pair(std::move(tempLLVMFile), target));
  }

  return result;
}

SmallVector<std::string> getLinkArgs() {
#if (!BISHENGIR_PUBLISH)
  SmallVector<std::string> linkArgs;
  StringRef rtLib = getenv("RT_LIB");
  if (rtLib.empty())
    return linkArgs;
  linkArgs.push_back("-L" + rtLib.str());
  linkArgs.push_back("-lruntime");
  linkArgs.push_back("-lascendcl");
  return linkArgs;
#else
  return {};
#endif
}

bool isDebugOrShmemRelated(const ::llvm::StringRef &name) {
  return name.starts_with("_mlir_ciface_print_") ||
         name.starts_with("_mlir_ciface_assert_") ||
         name.starts_with("_mlir_ciface_init_debug") ||
         name.starts_with("_mlir_ciface_finish_debug") ||
         name.starts_with("_mlir_ciface_aclshmem");
}
bool isDebugOrShmemPresent(ModuleOp moduleOp /* don't need & */) {
  bool present = false;
  moduleOp.walk([&](Operation *opInner) {
    if (isa<LLVM::CallOp>(opInner)) {
      LLVM::CallOp callOp = cast<LLVM::CallOp>(opInner);
      ::std::optional<::llvm::StringRef> callee = callOp.getCallee();
      if (callee.has_value() && isDebugOrShmemRelated(callee.value())) {
        present = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return present;
}

void setMetadataForIntrinsicCall(llvm::CallInst *llCall,
                                 llvm::StringRef apiName,
                                 llvm::StringRef stubManglingName) {
  llvm::LLVMContext &ctx = llCall->getContext();
  llvm::MDNode *apiNameNode =
      llvm::MDNode::get(ctx, llvm::MDString::get(ctx, apiName));
  llvm::MDNode *stubManglingNameNode =
      llvm::MDNode::get(ctx, llvm::MDString::get(ctx, stubManglingName));
  llCall->setMetadata("asan.cce.api.name", apiNameNode);
  llCall->setMetadata("asan.stub.mangling.name", stubManglingNameNode);
}

bool getMetadataInfo(llvm::StringRef name,
                     std::pair<llvm::StringRef, llvm::StringRef> &info) {
  if (name == "llvm.hivm.SET.FLAG.IMM") {
    info = {"set_flag", "_Z27__sanitizer_report_set_flagPU3AS1hmmljjj"};
    return true;
  } else if (name == "llvm.hivm.SET.FLAG.REG") {
    info = {"set_flag", "_Z27__sanitizer_report_set_flagPU3AS1hmmljjm"};
    return true;
  } else if (name == "llvm.hivm.WAIT.FLAG.IMM") {
    info = {"wait_flag", "_Z28__sanitizer_report_wait_flagPU3AS1hmmljjj"};
    return true;
  } else if (name == "llvm.hivm.WAIT.FLAG.REG") {
    info = {"wait_flag", "_Z28__sanitizer_report_wait_flagPU3AS1hmmljjm"};
    return true;
  } else if (name == "llvm.hivm.SET.CROSS.CORE") {
    info = {"ffts_cross_core_sync",
            "_Z39__sanitizer_report_ffts_cross_core_syncPU3AS1hmmljm"};
    return true;
  } else if (name == "llvm.hivm.WAIT.FLAG.DEV.REG") {
    info = {"wait_flag_dev", "_Z32__sanitizer_report_wait_flag_devPU3AS1hmmll"};
    return true;
  } else if (name == "llvm.hivm.WAIT.FLAG.DEV.PIPE.REG") {
    info = {"wait_flag_dev", "_Z32__sanitizer_report_wait_flag_devPU3AS1hmmll"};
    return true;
  } else if (name == "llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM") {
    info = {"wait_flag_dev", "_Z32__sanitizer_report_wait_flag_devPU3AS1hmmll"};
    return true;
  }
  return false;
}

void attachMetadataToLLVMIR(llvm::Module &llMod) {
  for (llvm::Function &llFunc : llMod) {
    if (llFunc.isDeclaration()) {
      continue;
    }
    for (llvm::BasicBlock &llBB : llFunc) {
      for (llvm::Instruction &llI : llBB) {
        if (llvm::CallInst *llCall = dyn_cast<llvm::CallInst>(&llI)) {
          llvm::Function *llCallee = llCall->getCalledFunction();
          if (llCallee && llCallee->isIntrinsic()) { // starts with "llvm."
            llvm::StringRef name = llCallee->getName();
            std::pair<llvm::StringRef, llvm::StringRef> info;
            if (getMetadataInfo(name, info)) {
              setMetadataForIntrinsicCall(llCall, info.first, info.second);
            }
          }
        }
      }
    }
  }
}

/// This is a utility function to run a pre-constructed pass pipeline on the
/// input module.
LogicalResult runPipeline(
    ModuleOp mod,
    const std::function<void(mlir::PassManager &, const HIVMCMainConfig &)>
        &buildPipeline,
    const HIVMCMainConfig &config, const std::string &pipelineName) {
  bishengir::BiShengIRPassManager passManager(mod->getContext(),
                                              ModuleOp::getOperationName(),
                                              OpPassManager::Nesting::Implicit);
  buildPipeline(passManager, config);

  // Apply MLIR PassManager command line options.
  // Ignore the result because the invocation point of this function might not
  // necessarily be the command line, so the options might not be loaded.
  (void)mlir::applyPassManagerCLOptions(passManager);
  (void)bishengir::applyPassManagerCLOptions(passManager);

  std::optional<mlir::TimingScope> pipelineTimingScope;
  mlir::TimingScope *timingScope = getCurrentCompileTimingScope();
  if (timingScope) {
    pipelineTimingScope.emplace(timingScope->nest(pipelineName));
    passManager.enableTiming(*pipelineTimingScope);
  }

  if (failed(passManager.run(mod))) {
    return mod->emitError("Failed to run " + pipelineName + " pipeline\n");
  }

  return success();
}

CompileTiming::CompileTiming() {
  mlir::applyDefaultTimingManagerCLOptions(manager);
  if (manager.isEnabled())
    rootScope = std::make_unique<mlir::TimingScope>(manager.getRootScope());
  previousTiming = currentCompileTiming;
  currentCompileTiming = this;
}

CompileTiming::~CompileTiming() { currentCompileTiming = previousTiming; }

int ExternalToolProfiler::run(StringRef name, std::function<int()> fn) {
  std::optional<mlir::TimingScope> toolTimingScope;
  mlir::TimingScope *timingScope = getCurrentCompileTimingScope();
  if (timingScope)
    toolTimingScope.emplace(timingScope->nest(name));
  return fn();
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

  std::string profilerName = llvm::sys::path::filename(binPath).str();
  if (ExternalToolProfiler::run(profilerName, [&]() {
        return llvm::sys::ExecuteAndWait(binPath, arguments);
      }) != 0) {
    llvm::errs() << "[ERROR] Executing: ";
    llvm::interleave(
        arguments, llvm::errs(),
        [](const StringRef &arg) { llvm::errs() << arg; }, " ");
    llvm::errs() << "\n";
    return failure();
  }
  return success();
}

LogicalResult checkOptionValidity(const HIVMCMainConfig &config) {
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

} // namespace bishengir

std::error_code hivmcCanonicalizePath(StringTmpPath &path) {
  if (path == "-") {
    return {};
  }
  std::error_code errorCode = llvm::sys::fs::make_absolute(path);
  if (errorCode)
    return errorCode;
  llvm::sys::path::remove_dots(path, /*removedotdot*/ true);
  return {};
}

//===--------------------------------------------------------------------===//
// getBitcodePathsBySubCoreTarget
//===--------------------------------------------------------------------===//

std::map<SubCoreTarget, std::string>
bishengir::getBitcodePathsBySubCoreTarget(mlir::ModuleOp mod) {
  std::map<SubCoreTarget, std::string> result;
  auto addIfSet = [&](SubCoreTarget target, mlir::Attribute attr) {
    if (!attr)
      return;
    if (auto a = llvm::dyn_cast<mlir::hivm::AIC_BITCODEAttr>(attr))
      result[target] = a.getPath().str();
    else if (auto a = llvm::dyn_cast<mlir::hivm::AIV_BITCODEAttr>(attr))
      result[target] = a.getPath().str();
    else if (auto a = llvm::dyn_cast<mlir::hivm::MIX_AIC_BITCODEAttr>(attr))
      result[target] = a.getPath().str();
    else if (auto a = llvm::dyn_cast<mlir::hivm::MIX_AIV_BITCODEAttr>(attr))
      result[target] = a.getPath().str();
    else if (auto a = llvm::dyn_cast<mlir::hivm::HOST_BITCODEAttr>(attr))
      result[target] = a.getPath().str();
  };
  addIfSet(SubCoreTarget::AIC, mod->getAttr(mlir::hivm::AIC_BITCODEAttr::name));
  addIfSet(SubCoreTarget::AIV, mod->getAttr(mlir::hivm::AIV_BITCODEAttr::name));
  addIfSet(SubCoreTarget::MIX_AIC,
           mod->getAttr(mlir::hivm::MIX_AIC_BITCODEAttr::name));
  addIfSet(SubCoreTarget::MIX_AIV,
           mod->getAttr(mlir::hivm::MIX_AIV_BITCODEAttr::name));
  addIfSet(SubCoreTarget::HOST,
           mod->getAttr(mlir::hivm::HOST_BITCODEAttr::name));
  return result;
}

std::string bishengir::getExecutablePath(const char *argv0, void *mainAddr) {
  std::string path = llvm::sys::fs::getMainExecutable(argv0, mainAddr);
  if (!path.empty()) {
    llvm::SmallString<256> realPath;
    if (!llvm::sys::fs::real_path(path, realPath))
      return std::string(realPath.str());
    return path;
  }
  llvm::SmallString<256> absPath(argv0);
  if (llvm::sys::fs::make_absolute(absPath))
    return "";
  llvm::SmallString<256> realPath;
  if (llvm::sys::fs::real_path(absPath, realPath))
    return std::string(absPath.str());
  return std::string(realPath.str());
}

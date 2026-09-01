//===- BiShengIRHIVMCompile.cpp - BiShengIR HIVM Compile Tool Support C++-*-==//
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

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Tools/hivmc/HIVMCA3.h"
#include "bishengir/Tools/hivmc/PassPipelineA3.h"
#include "bishengir/Tools/hivmc/Utility.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/hivmc/AdapterSanitizer.h"
#include "bishengir/Tools/hivmc/Config.h"

#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <regex>
#include <utility>

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;

namespace {

template <typename KEYT, typename VALUET>
VALUET getOrDefault(const std::map<KEYT, VALUET> &map, KEYT key,
                    VALUET defaultV) {
  if (map.find(key) == map.end()) {
    return defaultV;
  }
  return map.at(key);
}

/// Trim prefix and postfix blank of string
std::string trimString(std::string src) {
  std::regex re("(^\\s*)|(\\s*$)");
  return std::regex_replace(src, re, "");
}

/// Split the string s with the separator string into the result string vectors
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
[[maybe_unused]] std::optional<std::string>
findDirectoryInPath(StringRef targetDir) {
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

  return nullptr;
}

/// Get the path to BiSheng Compiler.
/// 1. First locate the BiSheng Compiler's binary via the `BISHENG_INSTALL_PATH`
///    environment variable.
/// 2. If the compiler path is not present, search bisheng in PATH environment
///    variable
/// 3. If still cant find, connect path from ASCEND_HOME_PATH environment
///    variable
/// this function will return the full patch with `bin` directory
std::string getBiShengCompilerInstallPath() {
  // 1. find bisheng from BISHENG_INSTALL_PATH
  std::string maybeBiShengInstallPath = getBiShengInstallPath();
  if (!maybeBiShengInstallPath.empty())
    return maybeBiShengInstallPath;

  // 2. find bisheng from PATH
  if (llvm::ErrorOr<std::string> P = llvm::sys::findProgramByName("bisheng")) {
    StringRef bishengFullPath = StringRef(P.get());
    if (!bishengFullPath.empty()) {
      return llvm::sys::path::parent_path(bishengFullPath).str();
    }
  }

  // 3. find bisheng from ASCEND_HOME_PATH
  const char *kAscendHomePathEnv = "ASCEND_HOME_PATH";
  const char *AscendHomePath = getenv(kAscendHomePathEnv);
  SmallString<128> bishengInstallPathStr(AscendHomePath);
  sys::path::append(bishengInstallPathStr, "bin");
  SmallString<128> bishengPathStr(bishengInstallPathStr);
  sys::path::append(bishengPathStr, "bisheng");
  if (llvm::sys::fs::exists(bishengPathStr))
    return std::string(bishengInstallPathStr);

  llvm::errs()
      << "[ERROR] Cannot find the binary path of the BiSheng compiler.\n";
  return "";
}

/// Get the BiSheng Compiler binary name.
StringRef getBiShengCompilerName() {
  const char *kBiShengBinaryName = "bisheng";
  return kBiShengBinaryName;
}

///.NEXT: Refactor.
#if (!BISHENGIR_PUBLISH)
StringRef getHostDebugPath() {
  const char *kHostDebugPathEnv = "HOST_DEBUG_PATH";
  return getenv(kHostDebugPathEnv);
}
#endif

///.NEXT: Refactor.
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
  return downgradeStacksaveStr;
}

std::string tryReplaceExtension(StringRef path, StringRef newExtension) {
  if (path == "-") {
    return std::string(path);
  }
  SmallVector<char> result(path.begin(), path.end());
  llvm::sys::path::replace_extension(result, newExtension);
  return std::string(result.data(), result.size());
}

std::string tryModifyFileName(StringRef path, StringRef prepend = "",
                              StringRef append = "") {
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

  // for cloning shared Ops
  OpBuilder builder(mod.getOperation()->getContext());
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

std::string tryAppendFileName(StringRef path, StringRef append) {
  return tryModifyFileName(path, "", append);
}

SmallVector<IRFilePair>
saveToFiles(const SmallVector<IRModulePair> &llvmModules,
            const std::string &outputFile, bool setKeepFlag,
            bool needToMangleOutputFilepath = false) {
  SmallVector<IRFilePair> result;

  needToMangleOutputFilepath =
      needToMangleOutputFilepath || (llvmModules.size() > 1);

  for (auto &[llvmModule, target] : llvmModules) {
    std::string append;
    if (needToMangleOutputFilepath) {
      if (target == SubCoreTarget::AIC) {
        append = ".mix_aic";
      } else if (target == SubCoreTarget::AIV) {
        append = ".mix_aiv";
      } else if (target == SubCoreTarget::HOST) {
        append = ".host";
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
      LDBG("Saved BiShengLIR to " << llOutputFile);
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

std::optional<std::string> compileDeviceKernel(
    StringRef bishengPath, const std::string &llvmirFilepath,
    const std::string &outputBinFilepath, SubCoreTarget arch,
    const HIVMCMainConfig &config,
    bool isMixKernel, bool mixKernelBothDebugOrShmem,
    std::map<SubCoreTarget, std::string> bitcodePaths) {
  if (!bitcodePaths.count(arch))
    return std::nullopt;
  std::string metaOpFilePath = bitcodePaths[arch];

  StringRef archArg = "";
  if (arch == SubCoreTarget::AIC) {
    archArg = "--cce-aicore-arch=dav-c220-cube";
  } else {
    archArg = "--cce-aicore-arch=dav-c220-vec";
  }

  std::string outputBin = tryReplaceExtension(outputBinFilepath, "o");

  SmallVector<StringRef> arguments;
  arguments.push_back(""); // occupied for bin
  arguments.push_back(llvmirFilepath);
  arguments.push_back("-o");
  arguments.push_back(outputBin);
  arguments.push_back(archArg);
  arguments.push_back("--cce-aicore-only");
  if (config.shouldEnableDebugVariables()) {
    arguments.push_back("-O0");
  } else {
    arguments.push_back("-O2");
  }

  arguments.push_back("-cce-bitcode-is-aicore");
  arguments.push_back("-Wno-override-module");
  if (isMixKernel) {
    arguments.push_back("-mllvm");
    arguments.push_back("-enable-mix=true");
  }
  if (config.shouldEnableSanitizer()) {
    arguments.push_back("--cce-enable-sanitizer");
    arguments.push_back("-g");
  } else if (config.shouldEnableDebugInfo() ||
             config.shouldEnableDebugVariables()) {
    arguments.push_back("-g");
  }
  arguments.push_back("-cce-link-aicore-ll-module");
  arguments.push_back(metaOpFilePath);
  std::vector<std::string> inputBitcode = config.getExtraDeviceBCPaths();
  for (const std::string &bc : inputBitcode) {
    arguments.push_back("-cce-link-aicore-ll-module");
    arguments.push_back(bc);
  }
  if (config.shouldInjectBarrierAllSync()) {
    arguments.push_back("--cce-auto-sync=bar-all");
    arguments.push_back("-mllvm");
    arguments.push_back("-cce-remove-auto-sync=true");
    arguments.push_back("-mllvm");
    arguments.push_back("-enable-deps-filter=off");
  }
  arguments.push_back("-mllvm");
  arguments.push_back("-cce-aicore-dcci-insert-for-scalar=false");

  std::string paramSizeArg =
      "--cce-aicore-input-parameter-size=" +
      std::to_string(config.deviceMaxInputParamSizeInBytes());
  arguments.push_back(paramSizeArg);

  StringRef bishengBin = getBiShengCompilerName();
  if ((!isMixKernel) || (!mixKernelBothDebugOrShmem)) {
    // link once
    if (failed(execute(bishengBin, bishengPath, arguments)))
      return std::nullopt;
    return outputBin;
  }
  // TODO: may avoid StringRef since its lifetime is hard to maintain
  // link the llir with meta_op_mix and produce the
  // final .o
  SubCoreTarget mixCoreType =
      arch == SubCoreTarget::AIC ? SubCoreTarget::MIX_AIC
                                            : SubCoreTarget::MIX_AIV;
  if (!bitcodePaths.count(mixCoreType))
    return std::nullopt;
  std::string metaOpMixPartFilePath = bitcodePaths[mixCoreType];
  arguments.push_back("-cce-link-aicore-ll-module");
  arguments.push_back(metaOpMixPartFilePath);
  if (failed(execute(bishengBin, bishengPath, arguments)))
    return std::nullopt;
  return outputBin;
}

std::optional<std::string> linkMixKernel(StringRef bishengPath,
                                         const SmallVector<StringRef> &inputs,
                                         const std::string &output,
                                         const HIVMCMainConfig &config) {
  StringRef exe = "ld.lld";

  SmallVector<StringRef> arguments;
  arguments.push_back(""); // occupied for bin
  arguments.push_back("-m");
  arguments.push_back("aicorelinux");
  arguments.push_back("-Ttext");
  arguments.push_back("0");
  arguments.push_back("-z");
  arguments.push_back("separate-loadable-segments");
  arguments.push_back("-z");
  arguments.push_back("norelro");
  arguments.push_back("-q"); // generate relocations
  arguments.push_back("-r");

  SmallVector<std::string> linkArgs;
  if (config.shouldEnableSanitizer()) {
    linkArgs.push_back("-L" + getAscendPath().str() +
                       "/tools/mssanitizer/lib64");
    linkArgs.push_back("-lsanitizer_stub_dav-c220-vec");
    linkArgs.push_back("-lsanitizer_stub_dav-c220-cube");
    arguments.insert(arguments.end(), linkArgs.begin(), linkArgs.end());
  }

  arguments.push_back("-o");
  arguments.push_back(output);
  for (auto &input : inputs) {
    arguments.push_back(input);
  }

  if (failed(execute(exe, bishengPath, arguments)))
    return std::nullopt;

  return output;
}

LogicalResult relocBinary(const std::string &unrelocFile,
                          const HIVMCMainConfig &config) {
  auto unrelocBin = tryReplaceExtension(unrelocFile, "o");
  SmallVector<StringRef> arguments;
  arguments.push_back(""); // occupied for bin
  arguments.push_back("-m");
  arguments.push_back("aicorelinux");
  arguments.push_back("-Ttext");
  arguments.push_back("0");
  arguments.push_back(unrelocBin);
  arguments.push_back("-q");
  arguments.push_back("-static");

  SmallVector<std::string> linkArgs;
  if (config.shouldEnableSanitizer()) {
    linkArgs.push_back("-L" + getAscendPath().str() +
                       "/tools/mssanitizer/lib64");
    linkArgs.push_back("-lsanitizer_stub_dav-c220-vec");
    linkArgs.push_back("-lsanitizer_stub_dav-c220-cube");
    arguments.insert(arguments.end(), linkArgs.begin(), linkArgs.end());
  }

  arguments.push_back("-o");
  arguments.push_back(unrelocBin);

  StringRef exe = "ld.lld";
  auto bishengPath = getBiShengCompilerInstallPath();
  if (failed(execute(exe, bishengPath, arguments)))
    return failure();

  return success();
}

bool lowerDeviceToBinary(
    const SmallVector<IRFilePair> &outputLLVMIRs, const std::string &outputFile,
    const HIVMCMainConfig &config, bool isMixKernel, bool mixKernelBothDebugOrShmem,
    std::map<SubCoreTarget, std::string> bitcodePaths) {
  LDBG("Lowering device module from BiShengLIR to binary");
  SmallVector<std::optional<std::string>> outputObjects;

  for (auto &pair : outputLLVMIRs) {
    std::optional<std::string> result;
    auto bishengPath = getBiShengCompilerInstallPath();

    // For mix kernels, derive .o path from the .ll file's path (which already has mangle suffix)
    // For non-mix, use outputFile directly
    std::string kernelOutputPath = (outputLLVMIRs.size() > 1)
                                       ? tryReplaceExtension(
                                             pair.first->getFilename().str(), "o")
                                       : outputFile;

    result = compileDeviceKernel(bishengPath, pair.first->getFilename().str(),
                                 kernelOutputPath, pair.second, config, isMixKernel,
                                 mixKernelBothDebugOrShmem, bitcodePaths);
    outputObjects.push_back(result);
  }

  if (std::any_of(
          outputObjects.begin(), outputObjects.end(),
          [](const std::optional<std::string> &o) { return !o.has_value(); })) {
    return false;
  }

  if (outputLLVMIRs.size() > 1) {
    SmallVector<StringRef> linkerInputs(outputObjects.size());
    std::transform(outputObjects.begin(), outputObjects.end(),
                   linkerInputs.begin(),
                   [](const std::optional<std::string> &o) {
                     return StringRef(o.value());
                   });
    auto bishengPath = getBiShengCompilerInstallPath();
    auto finalObjectMaybe =
        linkMixKernel(bishengPath, linkerInputs,
                      tryReplaceExtension(outputFile, "o"), config);
    if (!finalObjectMaybe) {
      return false;
    }
  }

  return true;
}

SmallVector<IRModulePair>
translateDeviceKernelToLLVM(ArrayRef<ModuleOp> modulesToLower,
                            const HIVMCMainConfig &config,
                            llvm::LLVMContext &llvmContext) {
  SmallVector<IRModulePair> result;

  for (ModuleOp m : modulesToLower) {
    SubCoreTarget target;
    if (hivm::isAICModule(m)) {
      target = SubCoreTarget::AIC;
    } else if (hivm::isAIVModule(m)) {
      target = SubCoreTarget::AIV;
    } else {
      llvm_unreachable("Can only lower AIC or AIV module!");
    }
    hivm::removeModuleCoreTypeAttr(m);
    std::unique_ptr<llvm::Module> llvmModule =
        translateModuleToLLVMIR(m, llvmContext);
    if (!llvmModule) {
      m->emitError(
          "Failed to translate module from LLVM Dialect IR to BiShengLIR\n");
    }
    if (config.shouldEnableSanitizer() &&
        failed(setSanitizerAddrArgName(m, llvmModule))) {
      m->emitError("Sanitizer arg rename failed \n");
    }
    result.push_back(std::make_pair(std::move(llvmModule), target));
  }

  return result;
}

#if (!BISHENGIR_PUBLISH)
std::optional<std::string> compileToCPUDynamicLib(
    StringRef hostDebugPath, const std::string &llvmirFilepath,
    SubCoreTarget arch,
    std::map<SubCoreTarget, std::string> bitcodePaths) {
  if (!bitcodePaths.count(arch))
    return std::nullopt;
  std::string metaOpFilePath = bitcodePaths[arch];
  std::string outputBin = tryReplaceExtension(llvmirFilepath, "so");

  SmallVector<StringRef> arguments;
  arguments.push_back(""); // occupied for bin
  arguments.push_back("-shared");
  arguments.push_back(llvmirFilepath);
  arguments.push_back(metaOpFilePath);
  arguments.push_back("-o");
  arguments.push_back(outputBin);

  StringRef clangBin = "clang++";
  if (failed(execute(clangBin, hostDebugPath, arguments)))
    return std::nullopt;

  return outputBin;
}

LogicalResult lowerToCPUDynamicLib(
    const SmallVector<IRFilePair> &outputLLVMIRs,
    std::map<SubCoreTarget, std::string> bitcodePaths) {
  SmallVector<std::optional<std::string>> outputObjects;
  if (outputLLVMIRs.size() > 1) {
    llvm_unreachable("Just support one module to cpu target compilation yet");
  }

  for (auto &pair : outputLLVMIRs) {
    std::optional<std::string> result;

    StringRef hostDebugPath = getHostDebugPath();
    result =
        compileToCPUDynamicLib(hostDebugPath, pair.first->getFilename().str(),
                               pair.second, bitcodePaths);

    outputObjects.push_back(result);
  }

  return success(!std::any_of(
      outputObjects.begin(), outputObjects.end(),
      [](const std::optional<std::string> &o) { return !o.has_value(); }));
}
#endif

LogicalResult runDeviceBiShengLIRCompile(
    ModuleOp mod, const HIVMCMainConfig &config, const std::string &outputFile,
    const std::string &tempFilesPath,
    std::map<SubCoreTarget, std::string> bitcodePaths) {
  LDBG("Lowering device module from BiShengLIR to binary");
  // split the input mix module to 2 separate modules
  SmallVector<ModuleOp, 2> modulesAfterSplit;
  SmallVector<OwningModuleRef> moduleCleanUp;
  LDBG("Split mix device kernel");
  hivm::TModuleCoreTypeAttr coreTypeAttr = hivm::getModuleCoreTypeAttr(mod);
  bool isMixKernel = false;
  bool mixKernelBothDebugOrShmem = false;
  if (coreTypeAttr &&
      coreTypeAttr.getModuleCoreType() == hivm::TModuleCoreType::MIX) {
    std::optional<std::pair<ModuleOp, ModuleOp>> splitMaybe =
        splitMixModule(mod);
    if (!splitMaybe) {
      return failure();
    }
    modulesAfterSplit = {splitMaybe.value().first, splitMaybe.value().second};
    moduleCleanUp.emplace_back(modulesAfterSplit[0]);
    moduleCleanUp.emplace_back(modulesAfterSplit[1]);
    isMixKernel = true;
    // short-circuiting is fine
    mixKernelBothDebugOrShmem = isDebugOrShmemPresent(modulesAfterSplit[0]) &&
                         isDebugOrShmemPresent(modulesAfterSplit[1]);
  } else {
    modulesAfterSplit.push_back(mod);
  }

  llvm::LLVMContext llvmContext;
  LDBG("Translating device from LLVM Dialect IR to BiShengLIR");
  // translate all mlir modules to llvmir modules
  SmallVector<IRModulePair> llvmModules =
      translateDeviceKernelToLLVM(modulesAfterSplit, config, llvmContext);
  if (std::any_of(llvmModules.begin(), llvmModules.end(),
                  [](const IRModulePair &pair) { return !pair.first; })) {
    return mod.emitError("Failed to convert BiShengHIR to BiShengLIR");
  }

  // attach necessary metadata to the generated llvm ir
  if (config.shouldEnableSanitizer()) {
    for (IRModulePair &pair : llvmModules) {
      llvm::Module &llMod = *(pair.first);
      attachMetadataToLLVMIR(llMod);
    }
  }

  // save llvm ir modules temporarily.
  //
  // NOTICE: in future, this saving may not be necessary.
  bool setKeepFlag = !config.shouldCompileLIR();
#if (!BISHENGIR_PUBLISH)
  // For security reasons, we want to honor saving of .ll files with --save-temps
  // option in internal builds only.
  if (!config.shouldSaveTemps().empty())
	  setKeepFlag = true;
#endif
  SmallVector<IRFilePair> llvmTempFiles =
      saveToFiles(llvmModules, tempFilesPath, setKeepFlag);
  if (std::any_of(llvmTempFiles.begin(), llvmTempFiles.end(),
                  [](const IRFilePair &pair) { return !pair.first; })) {
    return mod.emitError("Failed to save BiShengLIR to files");
  }

  if (!config.shouldCompileLIR()) {
    LDBG("No need to compile BiShengLIR, returning early");
    return success();
  }

#if (!BISHENGIR_PUBLISH)
  if (config.enableCPUTraceIntrinsic()) {
    return lowerToCPUDynamicLib(llvmTempFiles, bitcodePaths);
  }
#endif

  // NEXT: remove isMixKernel and mixKernelBothDebugOrShmem after refactoring  to
  // enable bisheng link two bc files compile llvm ir files to the final binary
  if (!lowerDeviceToBinary(llvmTempFiles, outputFile, config, isMixKernel,
                           mixKernelBothDebugOrShmem, bitcodePaths)) {
    return mod.emitError("Failed to compile BiShengLIR to binary");
  }

  // if enable bin relocation, do relocation to verify func is all defined
  // note: the reloc binary will replace outputFile
  if (config.shouldRelocateBinary() &&
      failed(relocBinary(outputFile, config))) {
    return mod.emitError("Failed to relocate binary");
  }

  return success();
}

LogicalResult lowerHostToBinary(
    const std::string &tempLLVMFile, const std::string &outputFileName,
    bool compileWithFatObj,
    const SmallVector<hacc::utils::ExternalFuncInfo> &externalFuncs,
    std::map<SubCoreTarget, std::string> bitcodePaths) {
  LDBG("Lowering host module from BiShengHIR to binary");
  SmallVector<StringRef> arguments;

  arguments.clear();
  arguments.push_back(""); // occupied for bin

  LDBG("Linking External Functions");
  LDBG("Num External Funcs: " << externalFuncs.size());

  for (const auto &extFunc : externalFuncs) {
    if (!llvm::sys::fs::exists(extFunc.srcPath)) {
      llvm::errs() << "External source file not found: " << extFunc.srcPath
                   << "\n";
      return failure();
    }
    LDBG("ExtFunc: " << extFunc.funcName << " FuncSrc: " << extFunc.srcPath);
    arguments.push_back(extFunc.srcPath);
  }
  auto bishengPath = getBiShengCompilerInstallPath();
  if (!bitcodePaths.count(SubCoreTarget::HOST))
    return failure();
  std::string hostTilingFuncFilePath =
      bitcodePaths[SubCoreTarget::HOST];

  arguments.push_back("-x");
  arguments.push_back("ir");
  arguments.push_back(tempLLVMFile);
  arguments.push_back(hostTilingFuncFilePath);
  arguments.push_back("--shared");
  LDBG("Compile with fat obj: " << (compileWithFatObj ? "True" : "False"));
  if (compileWithFatObj)
    arguments.push_back("--cce-fatobj-link");
  auto linkArgs = getLinkArgs();
  arguments.insert(arguments.end(), linkArgs.begin(), linkArgs.end());
  arguments.push_back("-o");
  std::string outputLib = "-";
  if (!outputFileName.empty() && outputFileName != "-") {
    outputLib =
        tryModifyFileName(tryReplaceExtension(outputFileName, ".so"), "lib");
  }
  arguments.push_back(outputLib);

  StringRef bishengBin = getBiShengCompilerName();
  if (failed(execute(bishengBin, bishengPath, arguments)))
    return failure();

  return success();
}

LogicalResult runHostBiShengLIRCompile(
    ModuleOp mod, const HIVMCMainConfig &config, const std::string &outputFile,
    std::map<SubCoreTarget, std::string> bitcodePaths) {
  if (!(config.shouldCompileLIR() && hacc::existHost(mod)))
    return success();

  mod = hacc::filterFuncsInModule(mod,
                                  /*shouldInclude=*/hacc::notExportedAsDag);
  bool compileWithFatObj = hacc::existEntryHost(mod);
  hivm::removeModuleCoreTypeAttr(mod);

  LDBG("Lowering host module from BiShengHIR to LLVM Dialect IR");
  auto buildPipeline = std::bind(buildBiShengHIRHIVMToLLVMPipeline,
                                 std::placeholders::_1, config);
  if (failed(
          runPipeline(mod, buildPipeline, config, "Host BiShengHIR to LLVM"))) {
    return failure();
  }
  // Collect external funcs before module translation to llvm-ir
  SmallVector<hacc::utils::ExternalFuncInfo> externalFuncs =
      hacc::utils::collectExternalFuncs(mod);

  LLVM_DEBUG({
    if (!externalFuncs.empty()) {
      llvm::dbgs() << "Found " << externalFuncs.size()
                   << " external functions:\n";
      for (const auto &func : externalFuncs) {
        llvm::dbgs() << "  " << func.funcName << " in " << func.srcPath << "\n";
      }
    } else {
      llvm::dbgs() << "No external functions found\n";
    }
  });

  llvm::LLVMContext llvmContext;
  LDBG("Translating host module from LLVM Dialect IR to BiShengLIR");
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(mod, llvmContext);
  if (!llvmModule)
    return mod->emitError(
        "Failed to translate host module from LLVM Dialect IR to BiShengLIR\n");
  SmallVector<IRModulePair> llvmHostFile;
  llvmHostFile.emplace_back(std::move(llvmModule),
                            SubCoreTarget::HOST);

  LDBG("Lowering host module from BiShengLIR to binary");
  auto tempLLVMFileName = saveToFiles(llvmHostFile, outputFile, false, true);
  if (failed(lowerHostToBinary(tempLLVMFileName[0].first->getFilename().str(),
                               outputFile, compileWithFatObj, externalFuncs,
                               bitcodePaths)))
    return mod->emitError(
        "Failed to lower host module from BiShengLIR to binary\n");

  return success();
}
} // namespace


FailureOr<PipelineModuleResult> bishengir::runBiShengLIRCompileA3(
    ModuleOp module, HIVMCMainConfig config,
    const std::map<SubCoreTarget, std::string> &bitcodePaths) {

  PipelineModuleResult res;

  auto separatedModule = hacc::separateHostDeviceModule(module);
  auto hostModule = separatedModule.first;
  auto deviceModule = separatedModule.second;
  res.emplace_back(hostModule);
  res.emplace_back(deviceModule);
  // if "-" is used as part of some actual file names, replace it with "_"
  std::string outputFile = config.outputFile();
  if (outputFile == "-") {
    outputFile = "_";
  }

  // Handle --save-temps=<directory> option to store temp files
  // Use a separate path for temp files to avoid affecting the final output path
  std::string tempFilesOutputPath = outputFile;
  if (!config.shouldSaveTemps().empty()) {
    llvm::SmallString<256> saveTempsDir(config.shouldSaveTemps());
    if (llvm::sys::fs::make_absolute(saveTempsDir)) {
      llvm::errs() << "[ERROR] Failed to get absolute path for save-temps.\n";
      return failure();
    }
    if (!llvm::sys::fs::exists(saveTempsDir))
      if (auto ec = llvm::sys::fs::create_directories(saveTempsDir)) {
        llvm::errs() << "[ERROR] Failed to create save-temps directory: " << saveTempsDir << "\n";
        return failure();
      }
    llvm::sys::path::append(saveTempsDir, llvm::sys::path::filename(outputFile));
    tempFilesOutputPath = std::string(saveTempsDir);
  }

  size_t deviceInputParamSize = hacc::countDeviceArgSizeInByte(deviceModule);
  config.updateMaxInputParamsSizeInBytes(deviceInputParamSize);
  if (failed(runDeviceBiShengLIRCompile(deviceModule, config, outputFile,
                                        tempFilesOutputPath, bitcodePaths))) {
    deviceModule->emitError("Failed to compile BiShengLIR for device");
    return failure();
  }
  if (!hacc::existHost(hostModule)) {
    return res;
  }

  // set tmp output file path for host ir
  // TODO: find a better way to create temp files
  config.compileHost(true).setHostOutputFile(
      tryReplaceExtension(outputFile, "o"));
  if (failed(runHostBiShengLIRCompile(hostModule, config, outputFile,
                                      bitcodePaths))) {
    deviceModule->emitError("Failed to compile BiShengLIR for host");
    return failure();
  }
  return res;
}

mlir::LogicalResult bishengir::runHIVMCCompileA3(ModuleOp module, HIVMCMainConfig config) {
  auto buildPipeline = std::bind(buildBiShengHIRHIVMToLLVMPipeline,
                                 std::placeholders::_1, config);
  if (failed(runPipeline(module, buildPipeline, config,
                         "BiShengHIR HIVM To LLVM"))) {
    return failure();
  }
  auto bitcodePaths = getBitcodePathsBySubCoreTarget(module);
  return runBiShengLIRCompileA3(module, config, bitcodePaths);
}

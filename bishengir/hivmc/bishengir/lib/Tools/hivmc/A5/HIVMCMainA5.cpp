
//===-------- HIVMCMainA5.cpp - HIVMC A5 Compile Tool Support C++-*--------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/hivmc/HIVMCA5.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Triton/Pipelines/Passes.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/hivmc/AdapterSanitizer.h"
#include "bishengir/Tools/hivmc/Config.h"
#include "bishengir/Tools/hivmc/PassPipelineA5.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;
using namespace object;

namespace {

static constexpr llvm::StringLiteral kSIMTModule = "hacc.simt_module";
using MixedModules = std::pair<ModuleOp, SmallVector<ModuleOp, 2>>;

MixedModules getMixedModules(ModuleOp topMod) {
  MixedModules res;
  res.first = nullptr;
  for (auto subMod : topMod.getOps<ModuleOp>()) {
    if (subMod->hasAttr(kSIMTModule)) {
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

std::optional<std::string>
compileDeviceKernel(StringRef bishengPath, const std::string &llvmirFilepath,
                    const std::string &outputBinFilepath, SubCoreTarget coreType,
                    const HIVMCMainConfig &config, bool isMixKernel,
                    bool mixKernelHasDebugOrShmem,
                    const std::map<SubCoreTarget, std::string> &bitcodePaths) {
  bool regbaseCompile = hacc::utils::isRegBasedArch(config.getTargetBackend());
  std::string extension = config.shouldSaveLinkedIR() ? ".ll" : ".o";
  std::string outputPath = config.shouldSaveLinkedIR()
                               ? tryAppendFileName(outputBinFilepath, "_linked")
                               : outputBinFilepath;

  std::string outputBin = tryReplaceExtension(outputPath, extension);

  SmallVector<StringRef> arguments;
  arguments.push_back(""); // occupied for bin
  arguments.push_back(llvmirFilepath);
  arguments.push_back("-o");
  arguments.push_back(outputBin);
  std::string archBase = hacc::utils::isAscend950(config.getTargetBackend())
                             ? "dav-c310"
                             : "dav-c220";
  std::string archSuffix =
      (coreType == SubCoreTarget::AIC || coreType == SubCoreTarget::MIX_AIC)
          ? "-cube"
          : "-vec";
  std::string archArg = "--cce-aicore-arch=" + archBase + archSuffix;
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
  // Add extra device BC files if specified
  std::vector<std::string> bitcodePathList = config.getExtraDeviceBCPaths();
  if (!bitcodePaths.count(coreType)) {
    llvm::errs()
        << "[WARNING]cannot find bitcode from hivm attribute for coreType:"
        << static_cast<int>(coreType) << " \n";
  } else {
    std::string metaOpFilePath = bitcodePaths.at(coreType);
    bitcodePathList.push_back(metaOpFilePath);
  }

  if (regbaseCompile) {
    arguments.push_back("-mllvm");
    arguments.push_back("--cce-vf-auto-sync=global");
  }
  if (config.shouldInjectBarrierAllSync()) {
    arguments.push_back("--cce-auto-sync=bar-all");
    arguments.push_back("-mllvm");
    arguments.push_back("-cce-remove-auto-sync=true");
    arguments.push_back("-mllvm");
    arguments.push_back("-enable-deps-filter=off");
    // InjectBarrierAll should also inject barrier-all in the VF
    arguments.push_back("-mllvm");
    arguments.push_back("--cce-vf-mem-bar-all=true");
  }
  arguments.push_back("-mllvm");
  arguments.push_back("-cce-aicore-dcci-insert-for-scalar=false");
  if (hacc::utils::isAscend950(config.getTargetBackend())) {
    arguments.push_back("-mllvm");
    arguments.push_back("-cce-vf-aa-between-iters=true");
    arguments.push_back("-mllvm");
    arguments.push_back("-cce-vf-enable-loop-fusion=false");
  }

  if (config.shouldEnableVFFusion())
    arguments.push_back("--cce-simd-vf-fusion=false");

  // for example, the user can call bishengir-compile with
  // `-append-bisheng-options='--cce-simd-vf-fusion=false'`
  // to disable VF fusion optimization in bisheng
  const std::string appendBishengOptions = config.getAppendBiShengOptions();
  std::vector<std::string> appendBishengOptionsList;
  if (!appendBishengOptions.empty()) {
    splitString(appendBishengOptions, " ", appendBishengOptionsList);
    std::for_each(appendBishengOptionsList.begin(),
                  appendBishengOptionsList.end(),
                  [&](std::string &arg) { arguments.push_back(arg); });
  }

  if (config.shouldSaveLinkedIR()) {
    arguments.push_back("-Xclang");
    arguments.push_back("-disable-llvm-optzns");
    arguments.push_back("-S");
    arguments.push_back("-emit-llvm");
  }

  if (config.shouldDisableFMA()) {
    arguments.push_back("-mllvm");
    arguments.push_back("-cce-simt-fpmath-combine=false");
  }

  // AFAIK, this option will be enabled by default in a future bisheng release
  arguments.push_back("--cce-long-scbz=true");

  std::string paramSizeArg =
      "--cce-aicore-input-parameter-size=" +
      std::to_string(config.deviceMaxInputParamSizeInBytes());
  arguments.push_back(paramSizeArg);

  StringRef bishengBin = getBiShengCompilerName();
  if ((!isMixKernel) || (!mixKernelHasDebugOrShmem)) {
    if (bitcodePathList.empty()) {
      llvm::errs() << "[ERROR]cannot find any bitcode files for coreType:"
                   << static_cast<int>(coreType) << " \n";
      return std::nullopt;
    }
    for (const auto &BCPath : bitcodePathList) {
      arguments.push_back("-cce-link-aicore-ll-module");
      arguments.push_back(BCPath);
    }
    if (failed(execute(bishengBin, bishengPath, arguments))) {
      llvm::errs() << "[ERROR] failed to execute bisheng\n";
      return std::nullopt;
    }
    return outputBin;
  }
  // for device_print, link the llir with meta_op_mix and produce the final .o
  // TODO: may avoid StringRef since its lifetime is hard to maintain
  SubCoreTarget mixTarget = coreType == SubCoreTarget::AIC
                                ? SubCoreTarget::MIX_AIC
                                : SubCoreTarget::MIX_AIV;
  if (!bitcodePaths.count(mixTarget)) {
    llvm::errs()
        << "[WARNING] cannot find bitcode from hivm attribute for mixTarget "
        << static_cast<int>(mixTarget) << " \n";
  } else {
    std::string metaOpMixPartFilePath = bitcodePaths.at(mixTarget);
    bitcodePathList.push_back(metaOpMixPartFilePath);
  }

  if (bitcodePathList.empty()) {
    llvm::errs() << "[ERROR]cannot find any bitcode files\n";
    return std::nullopt;
  }
  for (const auto &BCPath : bitcodePathList) {
    arguments.push_back("-cce-link-aicore-ll-module");
    arguments.push_back(BCPath);
  }
  if (failed(execute(bishengBin, bishengPath, arguments))) {
    llvm::errs() << "[ERROR] failed to execute bisheng\n";
    return std::nullopt;
  }
  if (config.shouldSaveLinkedIR()) {
    llvm::errs()
        << "Compile Command using ccec: \n"
        << llvm::sys::findProgramByName(bishengBin, {bishengPath}).get() << " "
        << outputBin << archBase << archSuffix
        << " -O2 --cce-aicore-only --cce-generic-addrspace=off "
           "-cce-bitcode-is-aicore -Wno-override-module";
    if (regbaseCompile)
      llvm::errs() << " -mllvm --cce-vf-auto-sync=global ";
    llvm::errs() << " -o " << tryReplaceExtension(outputBin, ".o") << "\n";
  }
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
  auto bishengPath = getBiShengInstallPath();
  if (failed(execute(exe, bishengPath, arguments)))
    return failure();

  return success();
}

bool lowerDeviceToBinary(
    const SmallVector<IRFilePair> &outputLLVMIRs, const std::string &outputFile,
    const HIVMCMainConfig &config, bool isMixKernel,
    bool mixKernelHasDebugOrShmem,
    const std::map<SubCoreTarget, std::string> &bitcodePaths) {
  LDBG("Lowering device module from BiShengLIR to binary");
  SmallVector<std::optional<std::string>> outputObjects;

  for (auto &pair : outputLLVMIRs) {
    std::optional<std::string> result;
    auto bishengPath = getBiShengInstallPath();
    SubCoreTarget coreType = pair.second;

    // For mix kernels, derive .o path from the .ll file's path (which already has mangle suffix)
    // For non-mix, use outputFile directly
    std::string kernelOutputPath = (outputLLVMIRs.size() > 1)
                                       ? tryReplaceExtension(
                                             pair.first->getFilename().str(), "o")
                                       : outputFile;

    result = compileDeviceKernel(bishengPath, pair.first->getFilename().str(),
                                 kernelOutputPath, coreType, config, isMixKernel,
                                 mixKernelHasDebugOrShmem, bitcodePaths);
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
    auto bishengPath = getBiShengInstallPath();
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
translateDeviceKernelToLLVM(const SmallVector<ModuleOp, 2> &modulesToLower,
                            const HIVMCMainConfig &config,
                            llvm::LLVMContext &llvmContext) {
  SmallVector<IRModulePair> result;

  for (ModuleOp m : modulesToLower) {
    SubCoreTarget target;
    if (hivm::isAICModule(m)) {
      target = SubCoreTarget::AIC;
    } else if (hivm::isAIVModule(m)) {
      target = SubCoreTarget::AIV;
    } else if (config.shouldEnableSIMTOnly()) {
      target = SubCoreTarget::AIV;
    } else {
      llvm_unreachable("Can only lower AIC or AIV module!");
    }
    hivm::removeModuleCoreTypeAttr(m);
    // We need to remove hacc's attr. Otherwise, translateModuleToLLVMIR calls
    // HACCDialectLLVMIRTranslationInterface::amendOperation. That func requires
    // LLVM::LLVMFuncOp casted from the input param Op. The input Op could be
    // ModuleOp if hacc's attr exists
    // because hacc dialect has the LLVMTranslationDialectInterface.
    m->removeAttr(hacc::TargetAttr::name);
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
    SubCoreTarget coreType,
    const std::map<SubCoreTarget, std::string> &bitcodePaths) {
  if (!bitcodePaths.count(coreType))
    return std::nullopt;
  std::string metaOpFilePath = bitcodePaths.at(coreType);
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

LogicalResult
lowerToCPUDynamicLib(const SmallVector<IRFilePair> &outputLLVMIRs,
                     const HIVMCMainConfig &config,
                     const std::map<SubCoreTarget, std::string> &bitcodePaths) {
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
    const std::map<SubCoreTarget, std::string> &bitcodePaths) {
  LDBG("Lowering device module from BiShengLIR to binary");
  // split the input mix module to 2 separate modules
  SmallVector<ModuleOp, 2> modulesAfterSplit;

  LDBG("Split mix device kernel");
  hivm::TModuleCoreTypeAttr coreTypeAttr = hivm::getModuleCoreTypeAttr(mod);
  bool isMixKernel = false;
  bool mixKernelHasDebugOrShmem = false;
  if (coreTypeAttr &&
      coreTypeAttr.getModuleCoreType() == hivm::TModuleCoreType::MIX) {
    std::optional<std::pair<ModuleOp, ModuleOp>> splitMaybe =
        splitMixModule(mod);
    if (!splitMaybe) {
      return failure();
    }
    modulesAfterSplit = {splitMaybe.value().first, splitMaybe.value().second};
    {
      isMixKernel = true;
      // short-circuiting is fine
      mixKernelHasDebugOrShmem = isDebugOrShmemPresent(modulesAfterSplit[0]) ||
                                 isDebugOrShmemPresent(modulesAfterSplit[1]);
    }
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
    return failure();
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
  // For security reasons, we want to honor saving of .ll files with
  // --save-temps option in internal builds only.
#if (!BISHENGIR_PUBLISH)
  if (!config.shouldSaveTemps().empty())
    setKeepFlag = true;
#endif
  SmallVector<IRFilePair> llvmTempFiles =
      saveToFiles(llvmModules, tempFilesPath, setKeepFlag);
  if (std::any_of(llvmTempFiles.begin(), llvmTempFiles.end(),
                  [](const IRFilePair &pair) { return !pair.first; })) {
    return failure();
  }

  if (!config.shouldCompileLIR()) {
    return success();
  }

#if (!BISHENGIR_PUBLISH)
  if (config.enableCPUTraceIntrinsic()) {
    return lowerToCPUDynamicLib(llvmTempFiles, config, bitcodePaths);
  }
#endif

  // TODO: remove isMixKernel and mixKernelHasDebugOrShmem after refactoring
  //       to enable bisheng link two bc files
  // compile llvm ir files to the final binary
  if (!lowerDeviceToBinary(llvmTempFiles, outputFile, config, isMixKernel,
                           mixKernelHasDebugOrShmem, bitcodePaths)) {
    return failure();
  }

  if (config.shouldSaveLinkedIR()) {
    return success();
  }

  // if enable bin relocation, do relocation to verify func is all defined
  // note: the reloc binary will replace outputFile
  if (config.shouldRelocateBinary()) {
    if (failed(relocBinary(outputFile, config))) {
      return failure();
    }
  }

  return success();
}

// Checks per-thread stack usage and returns failure if it exceeds the
// configured limit. The check is gated on the --simt-stack-limit cmd-arg;
// if absent, the check is skipped entirely. Triton-Ascend now owns the
// policy (env var, default 1152 = CANN's per-thread stack) and always
// forwards a resolved value here.
LogicalResult checkSIMTStackLimit(const HIVMCMainConfig &config,
                                  const std::string &File) {
  auto cmdLineLimit = config.getSimtStackLimit();
  if (!cmdLineLimit)
    return success();
  int32_t stackLimit = *cmdLineLimit;

  LDBG("[DEBUG] per-thread stack limit is set to " << stackLimit << "\n");

  // The check is disabled only by a negative value. 0 is a valid (strict)
  // limit, not a disable sentinel.
  if (stackLimit < 0)
    return success();

  // ---- Read file ----
  auto FileOrErr = MemoryBuffer::getFile(File, /*IsText=*/false,
                                         /*RequiresNullTerminator=*/false);
  if (!FileOrErr) {
    LDBG("[DEBUG] Failed to open file '"
         << File << "': " << FileOrErr.getError().message());
    return success();
  }

  // ---- Parse binary ----
  auto BinaryOrErr = createBinary(FileOrErr.get()->getMemBufferRef());
  if (!BinaryOrErr) {
    LDBG("[DEBUG] Failed to parse object file '"
         << File << "': " << toString(BinaryOrErr.takeError()));
    return success();
  }

  auto *Obj = dyn_cast<ObjectFile>(BinaryOrErr->get());
  if (!Obj) {
    LDBG("[DEBUG] File '" << File << "' is not an object file");
    return success();
  }

  // ---- Iterate symbols ----
  for (SymbolRef Sym : Obj->symbols()) {
    // ---- Symbol name ----
    Expected<StringRef> NameOrErr = Sym.getName();
    if (!NameOrErr) {
      llvm::consumeError(NameOrErr.takeError());
      continue;
    }

    StringRef Name = *NameOrErr;

    if (!Name.starts_with(".AIV_Kernel_Type_"))
      continue;

    llvm::StringRef KernelName = Name;
    KernelName.consume_front(".ascend.meta.");

    // ---- Section ----
    Expected<section_iterator> SecOrErr = Sym.getSection();
    if (!SecOrErr) {
      llvm::consumeError(SecOrErr.takeError());
      continue;
    }
    if (*SecOrErr == Obj->section_end())
      continue;

    llvm::object::SectionRef Sec = **SecOrErr;

    // ---- Section Name ----
    auto SecNameOrErr = Sec.getName();
    if (!SecNameOrErr) {
      llvm::consumeError(SecNameOrErr.takeError());
      continue;
    }

    llvm::StringRef SecName = *SecNameOrErr;

    if (!SecName.starts_with(".ascend.meta."))
      continue;

    LDBG("[DEBUG] found TLV variable " << Name << " in section " << SecName);

    // --- Section base address ---
    uint64_t SecAddr = Sec.getAddress();

    // ---- Symbol address ----
    auto SymAddrOrErr = Sym.getAddress();
    if (!SymAddrOrErr) {
      llvm::consumeError(SymAddrOrErr.takeError());
      continue;
    }

    uint64_t SymAddr = *SymAddrOrErr;

    // ---- Compute offset ----
    uint64_t TLVOffset;
    if (Obj->isRelocatableObject()) {
      // ET_REL: symbol value is section-relative
      TLVOffset = SymAddr;
    } else {
      if (SymAddr < SecAddr)
        return success();
      TLVOffset = SymAddr - SecAddr;
    }

    // ---- Read section contents ----
    Expected<StringRef> ContentsOrErr = Sec.getContents();
    if (!ContentsOrErr) {
      LDBG("[DEBUG] cannot read content of section");
      return success();
    }

    StringRef Contents = *ContentsOrErr;
    // Safe reinterpretation
    ArrayRef<uint8_t> Bytes(static_cast<const uint8_t *>(
                                static_cast<const void *>(Contents.data())),
                            Contents.size());

    // now we start to read the TLV array.
    // it is defined in ccec/llvm/lib/Target/HiIPU/HiIPUMetaSection.cpp
    // HiIPUMetaSection::genAIVKernelTypeAndSuSimtStackSize
    const size_t sizeof_TLV = 2 + 2 + 4;
    const size_t arr_size = 5;
    const size_t sizeof_arr = sizeof_TLV * arr_size;
    if (TLVOffset + sizeof_arr > Bytes.size()) {
      LDBG("[DEBUG] expect content size " << TLVOffset + sizeof_arr << ", is "
                                          << Bytes.size());
      return success();
    }

    const bool IsLittleEndian = Obj->isLittleEndian();
    const uint8_t AddrSize = Obj->getBytesInAddress();
    DataExtractor DE(Bytes, IsLittleEndian, AddrSize);

    // go to third TLV
    uint64_t Off = TLVOffset + sizeof_TLV * 3;

    uint16_t Type = DE.getU16(&Off);
    uint16_t Len = DE.getU16(&Off);

    // expect FuncMetaType::SIMT_WARP_STACK_SIZE
    if (Type != 0x0009 || Len != 0x0004) {
      LDBG("[DEBUG] Unexpected TLV format in " << Name << " (type=" << Type
                                               << ", len=" << Len << ")");
      return success();
    }

    int32_t StackSize = DE.getSigned(&Off, sizeof(int32_t));

    LDBG("[DEBUG] StackSize = " << StackSize);

    // stackLimit and StackSize are both per-thread bytes. StackSize is read
    // from the SIMT_WARP_STACK_SIZE TLV emitted by the compiler.
    if (StackSize > stackLimit) {
      llvm::errs() << llvm::formatv("[ERROR] SIMT per-thread stack size ({0} "
                                    "bytes) of kernel {1} exceeds limit ({2} "
                                    "bytes/thread)\n",
                                    StackSize, KernelName, stackLimit);
      return failure();
    }
  }

  return success();
}

LogicalResult lowerHostToBinary(
    const std::string &tempLLVMFile, std::string outputFileName,
    bool compileWithFatObj,
    const SmallVector<hacc::utils::ExternalFuncInfo> &externalFuncs,
    const std::map<SubCoreTarget, std::string> &bitcodePaths,
    const HIVMCMainConfig &config) {
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
  auto bishengPath = getBiShengInstallPath();
  arguments.push_back("-x");
  arguments.push_back("ir");
  arguments.push_back(tempLLVMFile);
  std::vector<std::string> bitcodePathList = config.getExtraHostBCPaths();
  if (!bitcodePaths.count(SubCoreTarget::HOST)) {
    llvm::errs()
        << "[WARNING]cannot find bitcode from hivm attribute for host!\n";
  } else {
    std::string hostTilingFuncFilePath = bitcodePaths.at(SubCoreTarget::HOST);
    bitcodePathList.push_back(hostTilingFuncFilePath);
  }

  if (bitcodePathList.empty()) {
    llvm::errs() << "[ERROR]cannot find any bitcode files for host!\n";
    return failure();
  }
  for (const auto &BCPath : bitcodePathList) {
    arguments.push_back(BCPath);
  }
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
    const std::map<SubCoreTarget, std::string> &bitcodePaths) {
  if (!(config.shouldCompileLIR() && hacc::existHost(mod)))
    return success();

  mod = hacc::filterFuncsInModule(mod,
                                  /*shouldInclude=*/hacc::notExportedAsDag);
  bool compileWithFatObj = hacc::existEntryHost(mod);
  hivm::removeModuleCoreTypeAttr(mod);
  // We need to remove hacc's attr. Otherwise, translateModuleToLLVMIR calls
  // HACCDialectLLVMIRTranslationInterface::amendOperation. That func requires
  // LLVM::LLVMFuncOp casted from the input param Op. The input Op could be
  // ModuleOp if hacc's attr exists
  // because hacc dialect has the LLVMTranslationDialectInterface.
  mod->removeAttr(hacc::TargetAttr::name);

  // Remove simt_entry attribute to avoid the internal-linkage set of simt_vf decls.
  mod->walk([&](LLVM::LLVMFuncOp funcOp) {
    if (auto cconvAttr = funcOp->getAttr(hivm_regbaseintrins::kDavinciCallingConvAttrName)) {
      if (cconvAttr.isa<hivm_regbaseintrins::SIMT_EntryAttr>()) {
        funcOp->removeAttr(hivm_regbaseintrins::kDavinciCallingConvAttrName);
      }
    }
  });

  LDBG("Lowering host module from BiShengHIR to LLVM Dialect IR");
  if (failed(runPipeline(mod, buildBiShengHIRAVEToLLVMPipeline, config,
                         "Host BiShengHIR to LLVM"))) {
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
  llvmHostFile.emplace_back(std::move(llvmModule), SubCoreTarget::HOST);

  LDBG("Lowering host module from BiShengLIR to binary");
  auto tempLLVMFileName = saveToFiles(llvmHostFile, outputFile, false, true);
  if (failed(lowerHostToBinary(tempLLVMFileName[0].first->getFilename().str(),
                               outputFile, compileWithFatObj, externalFuncs,
                               bitcodePaths, config)))
    return mod->emitError(
        "Failed to lower host module from BiShengLIR to binary\n");

  return success();
}

bool runSIMTToLLVMCompile(ArrayRef<ModuleOp> modules, HIVMCMainConfig config) {
  config.enableSIMTOnly(true);
  bool result = true;
  for (auto module : modules) {
    result &= runPipeline(module, buildSIMTPipeline, config, "BiShengSIMT")
                  .succeeded();
  }
  return result;
}

bool runSIMDToLLVMCompile(ModuleOp module, HIVMCMainConfig &config) {
  return runPipeline(module, buildBiShengHIRAVEToLLVMPipeline, config,
                     "BiShengSIMD")
      .succeeded();
}
} // namespace

LogicalResult bishengir::runBiShengLIRCompileA5(
    ModuleOp module, HIVMCMainConfig config,
    const std::map<SubCoreTarget, std::string> &bitcodePaths) {
  auto separatedModule = hacc::separateHostDeviceModule(module);
  auto &hostModule = separatedModule.first;
  auto &deviceModule = separatedModule.second;

  // if "-" is used as part of some actual file names, replace it with "_"
  std::string outputFile = config.outputFile();
  if (outputFile == "-" && config.shouldCompileLIR()) {
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
        llvm::errs() << "[ERROR] Failed to create save-temps directory: "
                     << saveTempsDir << "\n";
        return failure();
      }
    llvm::sys::path::append(saveTempsDir,
                            llvm::sys::path::filename(outputFile));
    tempFilesOutputPath = std::string(saveTempsDir);
  }

  size_t deviceInputParamSize = hacc::countDeviceArgSizeInByte(deviceModule);
  config.updateMaxInputParamsSizeInBytes(deviceInputParamSize);
  if (failed(runDeviceBiShengLIRCompile(deviceModule, config, outputFile,
                                        tempFilesOutputPath, bitcodePaths))) {
    deviceModule->emitError("Failed to compile BiShengLIR for device");
    return failure();
  }

  if (failed(
          checkSIMTStackLimit(config, tryReplaceExtension(outputFile, "o")))) {
    return failure();
  }

  if (!(config.shouldCompileLIR() && hacc::existHost(hostModule)))
    return success();

  // set tmp output file path for host ir
  // TODO: find a better way to create temp files
  config.compileHost(true).setHostOutputFile(
      tryReplaceExtension(outputFile, "o"));
  if (failed(runHostBiShengLIRCompile(hostModule, config, outputFile,
                                      bitcodePaths))) {
    deviceModule->emitError("Failed to compile BiShengLIR for host");
    return failure();
  }

  return success();
}
LogicalResult bishengir::runHIVMCCompileA5(ModuleOp hirCompileModule,
                                         HIVMCMainConfig config) {


  MLIRContext *ctx = hirCompileModule->getContext();
  mlir::DiagnosticEngine &diagEngine = ctx->getDiagEngine();
  std::vector<Diagnostic> collectedDiagnostics;
  // Collect diagnostics and emit them afterwards because we have tuning
  // mechanism.
  auto handlerID =
      diagEngine.registerHandler([&collectedDiagnostics](Diagnostic &diag) {
        collectedDiagnostics.emplace_back(std::move(diag));
  });

  LogicalResult runPipelineStatus = failure();
  if (config.shouldEnableSimdSimtMixCompile()) {
    auto [mainMod, simtMods] = getMixedModules(hirCompileModule);
    // SIMT modules run triton lowering pipeline
    // Main module runs regular pipeline
    if (runSIMTToLLVMCompile(simtMods, config) &&
        runSIMDToLLVMCompile(mainMod, config)) {
      // Once both are lowered, flatten into single module
      runPipelineStatus =
          runPipeline(hirCompileModule, buildFinalMixVFCompilePipeline, config,
                      "BiShengFinishLLVM");
    }
  } else if (config.shouldEnableSIMTOnly()) {
    runPipelineStatus = success(runSIMTToLLVMCompile(hirCompileModule, config));
  } else {
    runPipelineStatus = success(runSIMDToLLVMCompile(hirCompileModule, config));
  }

  if (config.shouldOnlyRunHIVMPipeline()) {
    config.lowerToLLVMFlag(false);
  }

  if (config.shouldEnableCPURunner()) {
    config.lowerToLLVMFlag(false);
  }

  // Restore to the default handler.
  diagEngine.eraseHandler(handlerID);

  if (failed(runPipelineStatus)) {
    for (auto &diag : llvm::reverse(collectedDiagnostics)) {
      diagEngine.emit(std::move(diag));
    }
    return failure();
  }

  if (config.shouldOnlyRunHIVMPipeline()) {
    auto outputFile = config.outputFile();
    std::string errorMessage;
    auto fileHandle = mlir::openOutputFile(outputFile, &errorMessage);
    if (!fileHandle) {
      llvm::errs() << "[ERROR] Failed to open: " << outputFile
                   << " error message: " << errorMessage << "\n";
      return failure();
    }
    hirCompileModule.print(
        fileHandle->os(),
        mlir::OpPrintingFlags().enableDebugInfo(
            config.shouldEnableSanitizer() || config.shouldEnableDebugInfo()));
    fileHandle->keep();
    return success();
  }

  if (config.shouldEnableCPURunner()) {
    auto outputFile = config.outputFile();
    auto fileHandle = mlir::openOutputFile(outputFile);
    assert(fileHandle != nullptr);
    fileHandle->os() << hirCompileModule << '\n';
    fileHandle->keep();
    return success();
  }

  auto bishengPath = getBiShengInstallPath();
  auto bitcodePaths = getBitcodePathsBySubCoreTarget(hirCompileModule);
  if (failed(runBiShengLIRCompileA5(hirCompileModule, config, bitcodePaths))) {
    return failure();
  }

  return success();
}

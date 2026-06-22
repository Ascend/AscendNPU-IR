//===- BiShengIRCompileConfig.cpp - BiShengIR Compile Config -----*- C++-*-===//
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

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Tools/bishengir-compile/Config.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace bishengir;
using namespace llvm;

namespace {
static cl::OptionCategory featCtrlCategory("BiShengIR Feature Control Options");
static cl::OptionCategory dfxCtrlCategory("BiShengIR DFX Control Options");
static cl::OptionCategory
    generalOptCategory("BiShengIR General Optimization Options");
static cl::OptionCategory
    hfusionOptCategory("BiShengIR HFusion Optimization Options");
static cl::OptionCategory
    hivmOptCategory("BiShengIR HIVM Optimization Options");
static cl::OptionCategory targetCategory("BiShengIR Target Options");
static llvm::cl::OptionCategory
    enableCPURunnerCategory("BiShengIR CPU Runner Options");
static cl::OptionCategory
    sharedWithDownstreamToolchainCategory("Options Shared with HIVMC");

/// This class is intended to manage the handling of command line options for
/// creating bishengir-compile config. This is a singleton.
/// Options that are not exposed to the user should not be added here.
struct BiShengIRCompileMainConfigCLOptions : public BiShengIRCompileMainConfig {
  BiShengIRCompileMainConfigCLOptions() {
    // These options are static but all uses ExternalStorage to initialize the
    // members of the parent class. This is unusual but since this class is a
    // singleton it basically attaches command line option to the singleton
    // members.

#define GEN_OPTION_REGISTRATIONS
#include "bishengir/Tools/bishengir-compile/CompileOptions.cpp.inc"

    // -------------------------------------------------------------------------//
    //                        Input & Output setting options
    // -------------------------------------------------------------------------//

    static cl::opt<std::string, /*ExternalStorage=*/true> inputFilename(
        cl::Positional, cl::desc("<input file>"), cl::location(inputFileFlag),
        cl::init("-"));

    static cl::opt<std::string, /*ExternalStorage=*/true> outputFile(
        "o", cl::desc("Specify output bin name"), cl::location(outputFileFlag),
        cl::init("-"));

    //===--------------------------------------------------------------------===//
    //                          CPU Runner Options
    //===--------------------------------------------------------------------===//

#if MLIR_ENABLE_EXECUTION_ENGINE
    static llvm::cl::opt<CPURunnerMetadata<false>, /*ExternalStorage=*/true,
                         CPURunnerMetadataParser<false>>
        enableCPURunner{
            "enable-cpu-runner",
            llvm::cl::desc(
                "Enable CPU runner lowering pipeline on the final output."),
            llvm::cl::location(enableCPURunnerFlag),
            llvm::cl::cat(enableCPURunnerCategory)};

    static llvm::cl::opt<CPURunnerMetadata<true>, /*ExternalStorage=*/true,
                         CPURunnerMetadataParser<true>>
        enableCPURunnerBefore{
            "enable-cpu-runner-before",
            llvm::cl::desc("Enable BiShengIR CPU runner before "
                           "the specified pass and stop the execution."),
            llvm::cl::location(enableCPURunnerBeforeFlag),
            llvm::cl::cat(enableCPURunnerCategory)};

    static llvm::cl::opt<CPURunnerMetadata<true>, /*ExternalStorage=*/true,
                         CPURunnerMetadataParser<true>>
        enableCPURunnerAfter{
            "enable-cpu-runner-after",
            llvm::cl::desc(
                "Enable BiShengIR CPU runner after the specified pass "
                "and stop the execution."),
            llvm::cl::location(enableCPURunnerAfterFlag),
            llvm::cl::cat(enableCPURunnerCategory)};
#endif // MLIR_ENABLE_EXECUTION_ENGINE

    // when enableSanitizer/enableMemoryDisplay is true, enable
    // printDebugInfoOpt
    auto &opts = cl::getRegisteredOptions();
    if ((enableSanitizer || enableMemoryDisplay || enableDebugInfo) &&
        (opts.count("mlir-print-debuginfo") != 0)) {
      static_cast<cl::opt<bool> *>(opts["mlir-print-debuginfo"])
          ->setValue(true);
    }
  }
};
} // namespace

ManagedStatic<BiShengIRCompileMainConfigCLOptions> clOptionsConfig;

namespace option_handler {
template <typename T, bool ExternalStorage>
std::string handleOpt(const cl::opt<T, ExternalStorage> &opt) {
  llvm::report_fatal_error("not handled");
}

template <bool ExternalStorage>
std::string handleOpt(const cl::opt<bool, ExternalStorage> &opt) {
  return opt.getValue() ? "true" : "false";
}

template <bool ExternalStorage>
std::string handleOpt(const cl::opt<std::string, ExternalStorage> &opt) {
  return opt.getValue();
}

#define HANDLE_OPT_INT_OR_FLOAT(TYPE)                                          \
  template <bool ExternalStorage>                                              \
  std::string handleOpt(const cl::opt<TYPE, ExternalStorage> &opt) {           \
    return std::to_string(opt.getValue());                                     \
  }

HANDLE_OPT_INT_OR_FLOAT(unsigned)

template <bool ExternalStorage>
std::string
handleOpt(const cl::opt<MultiBufferStrategy, ExternalStorage> &opt) {
  const std::map<MultiBufferStrategy, std::string> keyMap = {
      {MultiBufferStrategy::NO_LIMIT, "no-limit"},
      {MultiBufferStrategy::ONLY_CUBE, "only-cube"},
      {MultiBufferStrategy::ONLY_VECTOR, "only-vector"},
      {MultiBufferStrategy::CUBE_NO_L0C, "no-l0c"},
  };
  return keyMap.at(opt.getValue());
}
} // namespace option_handler

void BiShengIRCompileMainConfig::collectHIVMCArgs() {
  std::vector<std::string> collectedArgs;
  auto &opts = cl::getRegisteredOptions();
  // Warning: please do not modify this part unless you know what you're doing.
  for (auto &[optStr, opt] : opts) {
    std::string optValue = "";

#define GEN_OPTION_COLLECTION
#include "bishengir/Tools/bishengir-compile/CompileOptions.cpp.inc"

    if (optValue.empty())
      continue;

    collectedArgs.push_back(optStr.str() + "=" + optValue);
  }

  for (auto &args : clOptionsConfig->getHIVMCArgs()) {
    if (args.empty())
      continue;

    for (auto arg : llvm::split(args, " "))
      collectedArgs.push_back(arg.str());
  }

  // collect all --link-aicore-bitcode args and merge them
  std::vector<std::string> filteredArgs;
  for (const auto &arg : collectedArgs) {
    llvm::StringRef argRef(arg);
    if (argRef.starts_with("--link-aicore-bitcode=") ||
        argRef.starts_with("link-aicore-bitcode="))
      continue;
    filteredArgs.push_back(arg);
  }
  collectedArgs = std::move(filteredArgs);

  // Collect .bc paths from --link-aicore-bitcode only.
  std::vector<std::string> linkPaths;
  for (const auto &path : clOptionsConfig->getLinkAicoreBitcode()) {
    if (!path.empty())
      linkPaths.push_back(path);
  }
  if (!linkPaths.empty()) {
    std::string linkOpt = "link-aicore-bitcode=";
    for (size_t i = 0; i < linkPaths.size(); ++i) {
      if (i > 0)
        linkOpt += ',';
      linkOpt += linkPaths[i];
    }
    collectedArgs.push_back(std::move(linkOpt));
  }

  clOptionsConfig->setHIVMCArgs(collectedArgs);
}

void BiShengIRCompileMainConfig::registerCLOptions() {
  // Make sure that the options struct has been initialized.
  *clOptionsConfig;
}

BiShengIRCompileMainConfig BiShengIRCompileMainConfig::createFromCLOptions() {
  BiShengIRCompileMainConfig::collectHIVMCArgs();
  return *clOptionsConfig;
}

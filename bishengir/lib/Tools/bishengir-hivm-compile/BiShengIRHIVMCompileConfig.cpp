//===- BiShengIRHIVMCompileConfig.cpp - BiShengIR HIVM Compile Config ---- ===//
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
#include "bishengir/Tools/Utils/Utils.h"
#include "bishengir/Tools/bishengir-hivm-compile/Config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"

using namespace bishengir;
using namespace llvm;

namespace {
static cl::OptionCategory
    featCtrlCategory("BiShengIR HIVM Feature Control Options");
static cl::OptionCategory dfxCtrlCategory("BiShengIR HIVM DFX Control Options");
static cl::OptionCategory
    generalOptCategory("BiShengIR HIVM General Optimization Options");
static cl::OptionCategory
    hivmOptCategory("BiShengIR HIVM Optimization Options");
static cl::OptionCategory targetCategory("BiShengIR HIVM Target Options");
static llvm::cl::OptionCategory
    enableCPURunnerCategory("BiShengIR HIVM CPU Runner Options");
static cl::OptionCategory
    sharedWithDownstreamToolchainCategory("Options Shared with hivmc");

/// This class is intended to manage the handling of command line options for
/// creating bishengir-hivm-compile config. This is a singleton.
/// Options that are not exposed to the user should not be added here.
struct BiShengIRHIVMCompileMainConfigCLOptions
    : public BiShengIRHIVMCompileMainConfig {
  BiShengIRHIVMCompileMainConfigCLOptions() {
    // These options are static but all uses ExternalStorage to initialize the
    // members of the parent class. This is unusual but since this class is a
    // singleton it basically attaches command line option to the singleton
    // members.

#define GEN_OPTION_REGISTRATIONS
#include "bishengir/Tools/bishengir-hivm-compile/HIVMCompileOptions.cpp.inc"

    // -------------------------------------------------------------------------//
    //                       Input & Output setting options
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

    // when enableSanitizer is true, enable printDebugInfoOpt
    auto &opts = cl::getRegisteredOptions();
    if ((enableSanitizer || enableDebugInfo) &&
        opts.count("mlir-print-debuginfo")) {
      static_cast<cl::opt<bool> *>(opts["mlir-print-debuginfo"])
          ->setValue(true);
    }
  }
};
} // namespace

ManagedStatic<BiShengIRHIVMCompileMainConfigCLOptions> clOptionsConfig;

void BiShengIRHIVMCompileMainConfig::registerCLOptions() {
  // Make sure that the options struct has been initialized.
  *clOptionsConfig;
}

BiShengIRHIVMCompileMainConfig
BiShengIRHIVMCompileMainConfig::createFromCLOptions() {
  StringTmpPath path(clOptionsConfig->getOutputFile());
  llvm::cantFail(llvm::errorCodeToError(canonicalizePath(path)),
                 "failed to canonicalize output file path.");
  clOptionsConfig->setOutputFile(path.str().str());
  BiShengIRHIVMCompileMainConfig::collectHIVMCArgs();
  return *clOptionsConfig;
}

namespace option_handler {
template <typename T, bool ExternalStorage>
std::string handleOpt(const cl::opt<T, ExternalStorage> &opt) {
  llvm_unreachable("not handled");
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

} // namespace option_handler

void BiShengIRHIVMCompileMainConfig::collectHIVMCArgs() {
  std::vector<std::string> collectedArgs;
  auto &opts = cl::getRegisteredOptions();
  // Warning: please do not modify this part unless you know what you're doing.
  for (auto &[optStr, opt] : opts) {
    std::string optValue = "";

#define GEN_OPTION_COLLECTION
#include "bishengir/Tools/bishengir-hivm-compile/HIVMCompileOptions.cpp.inc"

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

  clOptionsConfig->setHIVMCArgs(collectedArgs);
}
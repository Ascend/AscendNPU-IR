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
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-compile/Utility.h"

#if BISHENGIR_ENABLE_TRITON_COMPILE
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
#endif

#include "llvm/ADT/STLExtras.h" // interleaveComma
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h" // report_fatal_error
#include "llvm/Support/ManagedStatic.h"

using namespace bishengir;
using namespace llvm;
using namespace mlir::triton;

#if BISHENGIR_ENABLE_TRITON_COMPILE
static proton::ConvertProtonToProtonGPUOptions protonGPUCompileConfig;

namespace bishengir {
const proton::ConvertProtonToProtonGPUOptions &getProtonGPUCompileConfig() {
  return protonGPUCompileConfig;
}
} // namespace bishengir
#endif

namespace {
static cl::OptionCategory featCtrlCategory("BiShengIR Feature Control Options");
static cl::OptionCategory dfxCtrlCategory("BiShengIR DFX Control Options");
static cl::OptionCategory
    generalOptCategory("BiShengIR General Optimization Options");
static cl::OptionCategory
    hfusionOptCategory("BiShengIR HFusion Optimization Options");
static cl::OptionCategory
    hivmOptCategory("BiShengIR HIVM Optimization Options");
static cl::OptionCategory protonCategory("BiShengIR Proton Options");
static cl::OptionCategory targetCategory("BiShengIR Target Options");
static cl::OptionCategory
    simtOptCategory("BiShengIR SIMT Optimization Options");
static cl::OptionCategory
    sharedWithDownstreamToolchainCategory("Options Shared with HIVMC");
static cl::OptionCategory enableCPURunnerCategory("BiShengIR CPU Runner Options");

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

    static cl::opt<std::string, /*ExternalStorage=*/true> outputFile(
        "o", cl::desc("Specify output bin name"), cl::location(outputFileFlag),
        cl::init("-"));

#if (defined(MLIR_ENABLE_EXECUTION_ENGINE) && MLIR_ENABLE_EXECUTION_ENGINE) || \
    defined(BISHENGIR_ENABLE_EXECUTION_ENGINE)
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
#endif

    static cl::opt<int32_t, /*ExternalStorage=*/true> simtStackLimitOpt(
        "simt-stack-limit",
        cl::desc("Per-thread stack size limit (bytes) for SIMT kernels. The "
                 "compiler fails compilation if a kernel's per-thread stack "
                 "usage exceeds this limit. If unset, the check is skipped "
                 "— Triton-Ascend owns the policy (env var, default) and "
                 "always forwards a resolved value. Set to a negative value "
                 "to disable the check; 0 is a valid (strict) limit."),
        cl::value_desc("bytes-per-thread"),
        cl::location(simtStackLimitFlag),
        cl::cat(sharedWithDownstreamToolchainCategory));

    // when enableSanitizer is true, enable printDebugInfoOpt
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

template <typename T>
std::string handleValue(const T &value) {
  if constexpr (std::is_same_v<T, bool>) {
    return value ? "true" : "false";
  } else if constexpr (std::is_same_v<T, std::string>) {
    return value;
  } else if constexpr (std::is_integral_v<T>) {
    return std::to_string(value);
  } else {
    llvm_unreachable("not handled");
  }
}

template <typename T, typename ParserT>
std::string handleGenericParserOpt(ParserT &parser, const T &value) {
  const cl::OptionValue<T> optValue(value);
  for (unsigned i = 0, e = parser.getNumOptions(); i != e; ++i)
    if (optValue.compare(parser.getOptionValue(i)))
      return parser.getOption(i).str();
  llvm_unreachable("failed to stringify option value");
}

template <typename T, bool ExternalStorage>
std::string handleOpt(cl::opt<T, ExternalStorage> &opt) {
  if constexpr (std::is_base_of_v<
                           cl::generic_parser_base,
                           std::remove_reference_t<decltype(opt.getParser())>>) {
    return handleGenericParserOpt(opt.getParser(), opt.getValue());
  } else {
    return handleValue(opt.getValue());
  }
}
} // namespace option_handler

void BiShengIRCompileMainConfig::collectHIVMCArgs(
    BiShengIRCompileMainConfig &config) {
  std::vector<std::string> collectedArgs;
  auto &opts = cl::getRegisteredOptions();
  for (auto &[optStr, opt] : opts) {
    if (opt->getNumOccurrences() == 0)
      continue;

    std::string optValue = "";

#define GEN_OPTION_COLLECTION
#include "bishengir/Tools/bishengir-compile/CompileOptions.cpp.inc"

    if (optValue.empty())
      continue;

    collectedArgs.push_back(optStr.str() + "=" + optValue);
  }

  for (auto &args : config.getHIVMCArgs()) {
    if (args.empty())
      continue;

    for (auto arg : llvm::split(args, " "))
      if (!arg.empty())
      collectedArgs.push_back(arg.str());
  }

  std::vector<std::string> filteredArgs;
  for (const auto &arg : collectedArgs) {
    llvm::StringRef argRef(arg);
    if (argRef.starts_with("--link-aicore-bitcode=") ||
        argRef.starts_with("link-aicore-bitcode="))
      continue;
    filteredArgs.push_back(arg);
  }
  collectedArgs = std::move(filteredArgs);

  std::vector<std::string> linkPaths;
  for (const auto &path : config.getLinkAicoreBitcode()) {
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

  config.setHIVMCArgs(collectedArgs);
}

bool BiShengIRCompileMainConfig::isSharedWithDownstreamToolchain(
    llvm::StringRef argName) {
  auto &opts = cl::getRegisteredOptions();
  auto it = opts.find(argName);
  if (it == opts.end())
    return true;

  return llvm::any_of(it->second->Categories, [](const cl::OptionCategory *cat) {
    return cat == &sharedWithDownstreamToolchainCategory;
  });
}

void BiShengIRCompileMainConfig::registerCLOptions() {
  // Make sure that the options struct has been initialized.
  *clOptionsConfig;
}

BiShengIRCompileMainConfig BiShengIRCompileMainConfig::createFromCLOptions() {
  BiShengIRCompileMainConfig::collectHIVMCArgs(*clOptionsConfig);
  // Enforce <= 3 items
  if (clOptionsConfig->getSimtTritonGrid().size() > 3) {
    report_fatal_error(
        "Invalid --simt-triton-grid: at most 3 elements allowed x,y,z.\n");
  }
  StringTmpPath path(clOptionsConfig->getOutputFile());
  llvm::cantFail(llvm::errorCodeToError(canonicalizePath(path)),
                 "failed to canonicalize output file path.");
  clOptionsConfig->setOutputFile(path.str().str());
  return *clOptionsConfig;
}

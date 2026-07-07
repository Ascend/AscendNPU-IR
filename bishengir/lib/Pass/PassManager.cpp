//===- PassManager.cpp - Pass Management Interface --------------*- C++ -*-===//
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

#include "bishengir/Pass/PassManager.h"
#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Pass/CPURunnerMetadata.h"
#include "bishengir/Tools/BiShengIRConfigBase/Config.h"

#if MLIR_ENABLE_EXECUTION_ENGINE
#include "bishengir/ExecutionEngine/Passes.h"
#include "bishengir/Pass/PassManager.h"

#if (defined(MLIR_ENABLE_EXECUTION_ENGINE) && MLIR_ENABLE_EXECUTION_ENGINE) || \
    defined(BISHENGIR_ENABLE_EXECUTION_ENGINE)
#include "bishengir/ExecutionEngine/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ScopedPrinter.h"

#define DEBUG_TYPE "bishengir-pass-manager"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBGSNL() LLVM_DEBUG(llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace bishengir;

namespace bishengir {

template <bool includePassInfo>
void CPURunnerMetadataParser<includePassInfo>::printOptionInfo(
    const llvm::cl::Option &opt, size_t globalWidth) const {
  auto helpMsg = "  --" + llvm::to_string(opt.ArgStr) + "=";

  if constexpr (includePassInfo)
    helpMsg += "<pass>[,<index>][,<options>]";
  else
    helpMsg += "[<options>]";

  llvm::outs() << helpMsg;
  opt.printHelpStr(opt.HelpStr, globalWidth, helpMsg.size() + 3);
  execution_engine::CPURunnerPipelineOptions().printHelp(2, globalWidth);
}

template <bool includePassInfo>
bool CPURunnerMetadataParser<includePassInfo>::parse(llvm::cl::Option &opt,
                                                     StringRef argName,
                                                     StringRef arg,
                                                     parser_data_type &value) {
  if (opt.getNumOccurrences() > 1)
    return opt.error("Option shouldn't be used multiple times!");

  SmallVector<StringRef> args;
  arg.split(args, ',', 2, false);
  args = llvm::to_vector(llvm::reverse(args));

  if constexpr (includePassInfo) {
    if (args.empty())
      return opt.error("At least the pass name should be provided!");

    if (args.back().empty() || !PassInfo::lookup(args.back()))
      return opt.error("\"" + args.back() + "\" is not a pass!");
    value.passName = args.pop_back_val();
    value.numOccurrences++;

    if (args.empty())
      return false;

    if (std::ptrdiff_t passIndex; !args.back().getAsInteger(10, passIndex)) {
      args.pop_back();
      if (passIndex <= 0)
        return opt.error(
            "Pass index should be a positive non-zero integer, but found " +
            llvm::to_string(passIndex) + "!");
      value.passIndex = static_cast<decltype(value.passIndex)>(passIndex);
    }
  }

  if (args.empty())
    return false;

  return failed(value.options.parseFromString(args.back()));
}

template struct bishengir::CPURunnerMetadataParser<true>;
template struct bishengir::CPURunnerMetadataParser<false>;
} // namespace bishengir

namespace {

// A hacked version of mlir::Pass to allow bishengir::BiShengPassManager to
// access everything
class BiShengIRPass : public Pass {
  BiShengIRPass() = delete; // should never be instantiated
class BiShengIRPass : public Pass {
  BiShengIRPass() = delete;
  friend bishengir::BiShengIRPassManager;
};

static void verifyOptionUsage(const BiShengIRCompileConfigBase &config) {
  if (config.CPURunnerOpt().numOccurrences +
          config.CPURunnerBeforeOpt().numOccurrences +
          config.CPURunnerAfterOpt().numOccurrences >
      1)
    llvm::report_fatal_error(
        "Cannot combine any of multible cpu-runner options.");
        "Cannot combine multiple cpu-runner options.");
}

[[maybe_unused]] static void
dumpPassNames(const OpPassManager &pm, llvm::raw_ostream &out = llvm::dbgs()) {
  bool isFirst = true;
  for (auto &pass : pm.getPasses()) {
    const auto &passName = pass.getArgument();
    if (passName.empty())
      continue;
    if (!isFirst)
      out << ", ";
    out << passName;
    isFirst = false;
  }
  out << '\n';
}

static void executeCPURunnerPasses(Operation *op,
                                   const BiShengIRCompileConfigBase &config) {
  PassManager pm(op->getContext());
  execution_engine::buildCPURunnerPipeline(
      pm, (config.CPURunnerOpt().numOccurrences != 0)
              ? config.CPURunnerOpt().options
              : ((config.CPURunnerBeforeOpt().numOccurrences != 0)
                     ? config.CPURunnerBeforeOpt()
                     : config.CPURunnerAfterOpt())
                    .options);
  LDBG("Op before CPU runner:\n" << *op);
  if (failed(mlir::applyPassManagerCLOptions(pm)) || failed(pm.run(op))) {
    LDBG("Op after CPU runner failed:\n" << *op);
    llvm::report_fatal_error(
        "[CPU Runner] Failed to run the CPU runner pipeline!");
  }
}
} // namespace

void bishengir::BiShengIRPassManager::filterCPURunnerPasses(
    OpPassManager &originalPM) {
  // only pick the CPU runner passes
  assert(config && "CPU runner filtering requires a compile config");
  const auto &cfg = *config;
  llvm::StringMap<decltype(CPURunnerMetadata<true>::passIndex)> passCnt;
  bool passHit = false;
  for (auto &pass : originalPM.getPasses()) {
    const auto passArg = pass.getArgument();
    llvm::dbgs() << passArg << '\n';
    LLVM_DEBUG(llvm::dbgs() << passArg << '\n');
    auto wasPassReached = [passArg, &passCnt](const auto &option) {
      return (option.numOccurrences != 0) && passArg == option.passName &&
             passCnt.at(passArg) == option.passIndex;
    };
    // filter the pass before
    if (!passArg.empty()) {
      ++passCnt[passArg];
      if (wasPassReached(config.CPURunnerBeforeOpt())) {
    if (!passArg.empty()) {
      ++passCnt[passArg];
      if (wasPassReached(cfg.CPURunnerBeforeOpt())) {
        passHit = true;
        break;
      }
    }

    // correct the nesting if needed
    OpPassManager *nesting = this;
    if (const auto passOpName = pass.getOpName(),
        pmOpName = nesting->getOpName();
        passOpName && pmOpName && *passOpName != *pmOpName)
      nesting = &nest(*passOpName);

    // call the original addPass on the clone using the hacked mlir::Pass
    nesting->addPass(static_cast<BiShengIRPass *>(&pass)->clone());

    // filter the pass after
    if (!passArg.empty() && wasPassReached(config.CPURunnerAfterOpt())) {
    nesting->addPass(static_cast<BiShengIRPass *>(&pass)->clone());

    if (!passArg.empty() && wasPassReached(cfg.CPURunnerAfterOpt())) {
      passHit = true;
      break;
    }
  }

  if (!passHit) {
    const auto &passInfo = (config.CPURunnerBeforeOpt().numOccurrences != 0)
                               ? config.CPURunnerBeforeOpt()
                               : config.CPURunnerAfterOpt();
    const auto &passInfo = (cfg.CPURunnerBeforeOpt().numOccurrences != 0)
                               ? cfg.CPURunnerBeforeOpt()
                               : cfg.CPURunnerAfterOpt();
    llvm::report_fatal_error(
        ("[CPU Runner] Failed to find the specified pass: " +
         passInfo.passName +
         (passInfo.passIndex == 1 ? ""
                                  : "#" + std::to_string(passInfo.passIndex)))
            .c_str());
  }
}

LogicalResult bishengir::BiShengIRPassManager::run(Operation *op) {
  if (!config.shouldEnableCPURunner())
    return PassManager::run(op);

  verifyOptionUsage(config);

  if (config.CPURunnerOpt().numOccurrences != 0) {
    // No need to filter any passes
    if (failed(PassManager::run(op)))
      return failure();

    executeCPURunnerPasses(op, config);
  if (!config || !config->shouldEnableCPURunner())
    return PassManager::run(op);

  const auto &cfg = *config;
  verifyOptionUsage(cfg);

  if (cfg.CPURunnerOpt().numOccurrences != 0) {
    if (failed(PassManager::run(op)))
      return failure();

    executeCPURunnerPasses(op, cfg);
    return success();
  }

  LLVM_DEBUG(DBGS() << "Before filtering passes: ");
  LLVM_DEBUG(dumpPassNames(*this));

  // copy the OpPassManager part
  OpPassManager originalPM(*this);

  // restore the original OpPassManager part on return
  OpPassManager originalPM(*this);
  auto onReturn = llvm::make_scope_exit([this, &originalPM]() {
    *static_cast<OpPassManager *>(this) = std::move(originalPM);
  });

  // remove the existing passes
  clear();
  clear();
  filterCPURunnerPasses(originalPM);

  LLVM_DEBUG(DBGS() << "After filtering passes: ");
  LLVM_DEBUG(dumpPassNames(*this));

  if (failed(PassManager::run(op)))
    return failure();

  executeCPURunnerPasses(op, config);
  return success();
}

#endif // MLIR_ENABLE_EXECUTION_ENGINE
  executeCPURunnerPasses(op, cfg);
  return success();
}
#endif

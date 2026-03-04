//===- PassManager.cpp - Pass Management Interface --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"

#ifdef BISHENGIR_ENABLE_EXECUTION_ENGINE
#include "bishengir/ExecutionEngine/Passes.h"
#include "bishengir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ScopedPrinter.h"

#define DEBUG_TYPE "bishengir-pass-manager"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBGSNL() LLVM_DEBUG(llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {
template <bool includePassInfo> struct CPURunnerMetadata;

template <> struct CPURunnerMetadata<false> {
  mlir::execution_engine::CPURunnerPipelineOptions options;
};

template <> struct CPURunnerMetadata<true> : public CPURunnerMetadata<false> {
  std::string passName;
  std::size_t passIndex = 1;
};

template <bool includePassInfo>
struct CPURunnerMetadataParser
    : public llvm::cl::parser<CPURunnerMetadata<includePassInfo>> {
  using parser_data_type = CPURunnerMetadata<includePassInfo>;

  explicit CPURunnerMetadataParser(llvm::cl::Option &o)
      : llvm::cl::parser<parser_data_type>(o) {}

  void printOptionInfo(const llvm::cl::Option &opt,
                       size_t globalWidth) const final {
    auto helpMsg = "  --" + llvm::to_string(opt.ArgStr) + "=";

    if constexpr (includePassInfo)
      helpMsg += "<pass>[,<index>][,<options>]";
    else
      helpMsg += "[<options>]";

    llvm::outs() << helpMsg;
    opt.printHelpStr(opt.HelpStr, globalWidth, helpMsg.size() + 3);
    execution_engine::CPURunnerPipelineOptions().printHelp(2, globalWidth);
  }

  // Return true on error.
  static bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
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

      if (args.empty())
        return false;

      if (std::ptrdiff_t passIndex; !args.back().getAsInteger(10, passIndex)) {
        args.pop_back();
        if (value.passIndex <= 0)
          return opt.error(
              "Pass index should be a positive non-zero integer, but found " +
              llvm::to_string(value.passIndex) + "!");
        value.passIndex = static_cast<size_t>(passIndex);
      }
    }

    if (args.empty())
      return false;

    return failed(value.options.parseFromString(args.back()));
  }
};

//===--------------------------------------------------------------------===//
// CPU Runner Options
//===--------------------------------------------------------------------===//
static llvm::cl::OptionCategory enableCPURunnerCategory{
    "BiShengIR Runner Options"};
static llvm::cl::opt<CPURunnerMetadata<false>, false,
                     CPURunnerMetadataParser<false>>
    enableCPURunner{
        "enable-cpu-runner",
        llvm::cl::desc(
            "Enable CPU runner lowering pipeline on the final output."),
        llvm::cl::cat(enableCPURunnerCategory)};
static llvm::cl::opt<CPURunnerMetadata<true>, false,
                     CPURunnerMetadataParser<true>>
    enableCPURunnerBefore{
        "enable-cpu-runner-before",
        llvm::cl::desc("Enable BiShengIR CPU runner before "
                       "the specified pass and stop the execution."),
        llvm::cl::cat(enableCPURunnerCategory)};
static llvm::cl::opt<CPURunnerMetadata<true>, false,
                     CPURunnerMetadataParser<true>>
    enableCPURunnerAfter{
        "enable-cpu-runner-after",
        llvm::cl::desc("Enable BiShengIR CPU runner after the specified pass "
                       "and stop the execution."),
        llvm::cl::cat(enableCPURunnerCategory)};

// A hacked version of mlir::Pass to allow bishengir::BiShengPassManager to
// access everything
class BiShengIRPass : public Pass {
  BiShengIRPass() = delete; // should never be instantiated
  friend bishengir::BiShengIRPassManager;
};

static void verifyOptionUsage() {
  if (enableCPURunner.getNumOccurrences() +
          enableCPURunnerBefore.getNumOccurrences() +
          enableCPURunnerAfter.getNumOccurrences() >
      1)
    llvm::report_fatal_error("Cannot combine any of \"" +
                             enableCPURunner.ArgStr + "\", \"" +
                             enableCPURunnerBefore.ArgStr + "\", or \"" +
                             enableCPURunnerAfter.ArgStr + "\".");

  const auto compileConfig =
      bishengir::BiShengIRCompileMainConfig::createFromCLOptions();
  if (compileConfig.shouldCompileLIR())
    llvm::report_fatal_error(
        "LIR compilation should be disabled for the CPU runner.");

  if (!compileConfig.shouldManageHostResource())
    llvm::report_fatal_error(
        "Managing host resources should be enabled for the CPU runner.");
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

static void executeCPURunnerPasses(Operation *op) {
  PassManager pm(op->getContext());
  execution_engine::buildCPURunnerPipeline(
      pm,
      enableCPURunner.getNumOccurrences()
          ? enableCPURunner.options
          : (enableCPURunnerBefore.getNumOccurrences() ? enableCPURunnerBefore
                                                       : enableCPURunnerAfter)
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
  llvm::StringMap<decltype(CPURunnerMetadata<true>::passIndex)> passCnt;
  bool passHit = false;
  for (auto &pass : originalPM.getPasses()) {
    const auto passArg = pass.getArgument();
    llvm::dbgs() << passArg << '\n';
    auto wasPassReached = [passArg, &passCnt](const auto &option) {
      return option.getNumOccurrences() && passArg == option.passName &&
             passCnt.at(passArg) == option.passIndex;
    };
    // filter the pass before
    if (!passArg.empty()) {
      ++passCnt[passArg];
      if (wasPassReached(enableCPURunnerBefore)) {
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
    if (!passArg.empty() && wasPassReached(enableCPURunnerAfter)) {
      passHit = true;
      break;
    }
  }

  if (!passHit) {
    const auto &passInfo = enableCPURunnerBefore.getNumOccurrences()
                               ? enableCPURunnerBefore
                               : enableCPURunnerAfter;
    llvm::report_fatal_error(
        ("[CPU Runner] Failed to find the specified pass: " +
         passInfo.passName +
         (passInfo.passIndex == 1 ? ""
                                  : "#" + std::to_string(passInfo.passIndex)))
            .c_str());
  }
}

LogicalResult bishengir::BiShengIRPassManager::run(Operation *op) {
  if (!bishengir::BiShengIRCompileMainConfig::shouldEnableCPURunner())
    return PassManager::run(op);

  verifyOptionUsage();

  if (enableCPURunner.getNumOccurrences()) {
    // No need to filter any passes
    if (failed(PassManager::run(op)))
      return failure();

    executeCPURunnerPasses(op);
    return success();
  }

  LLVM_DEBUG(DBGS() << "Before filtering passes: ");
  LLVM_DEBUG(dumpPassNames(*this));

  // copy the OpPassManager part
  OpPassManager originalPM(*this);

  // restore the original OpPassManager part on return
  auto onReturn = llvm::make_scope_exit([this, &originalPM]() {
    *static_cast<OpPassManager *>(this) = std::move(originalPM);
  });

  // remove the existing passes
  clear();

  filterCPURunnerPasses(originalPM);

  LLVM_DEBUG(DBGS() << "After filtering passes: ");
  LLVM_DEBUG(dumpPassNames(*this));

  if (failed(PassManager::run(op)))
    return failure();

  executeCPURunnerPasses(op);
  return success();
}
#endif // BISHENGIR_ENABLE_EXECUTION_ENGINE

bool bishengir::BiShengIRCompileMainConfig::shouldEnableCPURunner() {
#ifdef BISHENGIR_ENABLE_EXECUTION_ENGINE
  return enableCPURunner.getNumOccurrences() ||
         enableCPURunnerBefore.getNumOccurrences() ||
         enableCPURunnerAfter.getNumOccurrences();
#else
  return false;
#endif // BISHENGIR_ENABLE_EXECUTION_ENGINE
}

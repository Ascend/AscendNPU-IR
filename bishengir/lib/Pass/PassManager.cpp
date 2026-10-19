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
#include "bishengir/Pass/PassExecutionPolicy.h"
#include "bishengir/Tools/BiShengIRConfigBase/Config.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bishengir-pass-manager"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBGSNL() LLVM_DEBUG(llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace bishengir;

namespace bishengir {

// Always compiled, independently of MLIR_ENABLE_EXECUTION_ENGINE: the pass
// filtering rides on the MLIR action infrastructure (MLIRIR), not on the
// execution engine, and it used to be registered unconditionally from the
// header constructors.
BiShengIRPassManager::~BiShengIRPassManager() {
  // The context keeps the last registered handler and it may outlive this
  // manager, so deactivate the shared policy state: the stale handler becomes
  // inert for the per-run Mix/Aiv filter (see PolicyState in the header).
  policyState->managerAlive.store(false, std::memory_order_release);
}

void BiShengIRPassManager::initializeActionHandler(MLIRContext *ctx) {
  // The handler is stored in the context and can outlive this manager; capture
  // the shared state, never `this`. The state outlives the manager as well, so
  // a stale handler never dereferences destroyed memory.
  std::shared_ptr<PolicyState> state = policyState;
  ctx->registerActionHandler([state](llvm::function_ref<void()> execute,
                                     const mlir::tracing::Action &action) {
    auto *passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&action);
    if (!passAction) {
      execute();
      return;
    }

    mlir::Operation *op = passAction->getOp();
    const mlir::Pass &pass = passAction->getPass();

    // 1. Aiv-mode global denylist, keyed by pass argument. Mix mode
    // short-circuits inside the policy, so full pipelines pay one
    // predictable branch per pass execution. Only a live manager filters:
    // its mode belongs to the pipeline it is running.
    if (state->managerAlive.load(std::memory_order_acquire) &&
        state->policy.shouldSkipGlobally(pass, op))
      return;

    llvm::StringRef passArg = pass.getArgument();

    // Adaptor passes (empty argument) orchestrate nested pipelines — always
    // let them through so nested ops still get processed.
    if (passArg.empty()) {
      execute();
      return;
    }

    // Helper: returns true if passArg is excluded by a FilterPassesAttr.
    auto isFiltered = [&](mlir::Operation *candidate) -> bool {
      auto attr = candidate->getAttrOfType<mlir::annotation::FilterPassesAttr>(
          mlir::annotation::FilterPassesAttr::name);
      if (!attr)
        return false;
      llvm::SmallVector<llvm::StringRef> allowed;
      attr.getPasses().getValue().split(allowed, ',');
      for (llvm::StringRef entry : allowed)
        if (entry.trim() == passArg)
          return false;
      return true; // attr present but passArg not listed
    };

    // 2. Per-op FilterPassesAttr whitelist — skip entirely.
    if (isFiltered(op)) {
      LLVM_DEBUG(DBGS() << "skip '" << passArg
                        << "' reason=FilterPassesAttr\n");
      return;
    }

    // Op is a module — temporarily remove child ops that are filtered for
    // this pass, execute, then restore them in order.
    if (auto mod = llvm::dyn_cast<mlir::ModuleOp>(op)) {
      mlir::Block *body = mod.getBody();

      // Collect ops to hide.
      llvm::SmallVector<mlir::Operation *> hidden;
      for (mlir::Operation &childOp : llvm::make_early_inc_range(*body)) {
        if (isFiltered(&childOp)) {
          hidden.push_back(&childOp);
          childOp.remove();
        }
      }

      execute();

      // Restore in original order: insert each op after its predecessor.
      for (auto *hiddenOp : llvm::reverse(hidden))
        body->push_front(hiddenOp);
      return;
    }

    execute();
  });
}

} // namespace bishengir

#if MLIR_ENABLE_EXECUTION_ENGINE
#include "bishengir/ExecutionEngine/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ScopedPrinter.h"

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

template struct CPURunnerMetadataParser<true>;
template struct CPURunnerMetadataParser<false>;
} // namespace bishengir

namespace {

// A hacked version of mlir::Pass to allow bishengir::BiShengPassManager to
// access everything
class BiShengIRPass : public Pass {
  BiShengIRPass() = delete; // should never be instantiated
  friend bishengir::BiShengIRPassManager;
};

static void verifyOptionUsage(const BiShengIRCompileConfigBase &config) {
  if (config.CPURunnerOpt().numOccurrences +
          config.CPURunnerBeforeOpt().numOccurrences +
          config.CPURunnerAfterOpt().numOccurrences >
      1)
    llvm::report_fatal_error(
        "Cannot combine any of multible cpu-runner options.");
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
  llvm::StringMap<decltype(CPURunnerMetadata<true>::passIndex)> passCnt;
  bool passHit = false;
  for (auto &pass : originalPM.getPasses()) {
    const auto passArg = pass.getArgument();
    llvm::dbgs() << passArg << '\n';
    auto wasPassReached = [passArg, &passCnt](const auto &option) {
      return (option.numOccurrences != 0) && passArg == option.passName &&
             passCnt.at(passArg) == option.passIndex;
    };
    // filter the pass before
    if (!passArg.empty()) {
      ++passCnt[passArg];
      if (wasPassReached(config.CPURunnerBeforeOpt())) {
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
      passHit = true;
      break;
    }
  }

  if (!passHit) {
    const auto &passInfo = (config.CPURunnerBeforeOpt().numOccurrences != 0)
                               ? config.CPURunnerBeforeOpt()
                               : config.CPURunnerAfterOpt();
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

  executeCPURunnerPasses(op, config);
  return success();
}

#endif // MLIR_ENABLE_EXECUTION_ENGINE

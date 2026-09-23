//===- PassExecutionPolicy.cpp - Pipeline-wide pass filtering ---*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Pass/PassExecutionPolicy.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "bishengir-pass-manager"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace bishengir {

//===----------------------------------------------------------------------===//
// PipelineMode detection
//===----------------------------------------------------------------------===//

/// The `mix_mode` function attribute spelling. It is a plain StringAttr with
/// exactly two recognized values, "mix" (full pipeline) and "aiv" (pure VV
/// pipeline); it is consumed by several existing passes (regbase
/// inferMixedCV, OptimizeScalarTransfers, MarkSyncBlockLockWithSubblock).
static constexpr StringRef kMixModeAttrName = "mix_mode";
static constexpr StringRef kMixModeValueMix = "mix";
static constexpr StringRef kMixModeValueAiv = "aiv";

PipelineMode detectPipelineMode(ModuleOp module) {
  // Collect the mode declared by every function that carries `mix_mode`.
  // Functions without the attribute do not constrain the mode: the module
  // may legitimately contain host helpers or device functions that are not
  // part of the VV pipeline.
  std::optional<PipelineMode> detected;
  module.walk([&](func::FuncOp func) {
    auto modeAttr = func->getAttrOfType<StringAttr>(kMixModeAttrName);
    if (!modeAttr)
      return WalkResult::advance();

    PipelineMode funcMode = PipelineMode::Mix;
    StringRef value = modeAttr.getValue();
    if (value == kMixModeValueAiv) {
      funcMode = PipelineMode::Aiv;
    } else if (value != kMixModeValueMix) {
      // Unknown value: writers of this attribute are external toolchains,
      // so fail safe to the full pipeline instead of erroring out.
      LLVM_DEBUG(DBGS() << "detectPipelineMode: unknown mix_mode='" << value
                        << "' on function '" << func.getSymName()
                        << "', treating as mix\n");
    }

    if (detected && *detected != funcMode) {
      // Disagreeing functions: not a pure VV module, run the full pipeline.
      LLVM_DEBUG(DBGS() << "detectPipelineMode: functions disagree on "
                        << kMixModeAttrName << ", falling back to mix\n");
      detected = PipelineMode::Mix;
      return WalkResult::interrupt();
    }
    detected = funcMode;
    return WalkResult::advance();
  });

  return detected.value_or(PipelineMode::Mix);
}

//===----------------------------------------------------------------------===//
// AivDisabledPasses denylist
//===----------------------------------------------------------------------===//

namespace {
/// The single mutable handle to the denylist. It is a function-local static so
/// that it is constructed on first use, independent of static initialization
/// order across the pipeline libraries that register entries.
llvm::StringSet<> &mutableAivDisabledPasses() {
  static llvm::StringSet<> aivDisabledPasses;
  return aivDisabledPasses;
}
} // namespace

const llvm::StringSet<> &getAivDisabledPasses() {
  // PipelineMode itself is NOT stored here: it is per-run state, because one
  // MLIRContext may compile different modules sequentially.
  return mutableAivDisabledPasses();
}

void registerAivDisabledPasses(std::initializer_list<StringRef> args) {
  // Registration happens during static pipeline registration, before any
  // compile runs on any thread.
  auto &set = mutableAivDisabledPasses();
  for (StringRef arg : args)
    set.insert(arg);
}

//===----------------------------------------------------------------------===//
// PassExecutionPolicy
//===----------------------------------------------------------------------===//

bool PassExecutionPolicy::shouldSkipGlobally(const Pass &pass,
                                             Operation *op) const {
  // Detect the pipeline mode once per run, from the top-level module the
  // pipeline executes on. Every action of a run sees the same module tree, so
  // the first action is representative and the mode is frozen afterwards,
  // keeping the per-action cost a single branch in Mix mode.
  if (!modeDetected.load(std::memory_order_acquire)) {
    // MLIR can dispatch per-operation actions from worker threads, so two
    // threads may detect at the same time. Both read the same module and
    // compute the same mode, so this is a benign race on a plain value store.
    Operation *root = op;
    while (root && !isa<ModuleOp>(root))
      root = root->getParentOp();
    PipelineMode detected = PipelineMode::Mix;
    if (root)
      detected = detectPipelineMode(cast<ModuleOp>(root));
    mode.store(detected, std::memory_order_release);
    modeDetected.store(true, std::memory_order_release);
    LLVM_DEBUG(DBGS() << "PassExecutionPolicy: pipeline mode="
                      << (detected == PipelineMode::Aiv ? "aiv" : "mix")
                      << "\n");
  }

  if (mode.load(std::memory_order_acquire) != PipelineMode::Aiv)
    return false;

  // The denylist is keyed by pass argument, mirroring the FilterPassesAttr
  // pass identity. Adaptor passes (empty argument) only orchestrate nested
  // pipelines and are never skipped.
  StringRef passArg = pass.getArgument();
  if (passArg.empty() || !getAivDisabledPasses().contains(passArg))
    return false;

  // Exemption: the delayed cross-core graph sync solver needs the backup
  // functions produced by the mix-CV flow, so an op that is (or a module whose
  // body contains) such a backup function is never denylist-filtered. This is
  // deliberately conservative — it can only run *more* passes, never drop a
  // step of the backup closed loop.
  auto hasBackup = [](Operation *candidate) {
    return candidate->hasAttr("hivm.backup_function");
  };
  if (isa<ModuleOp>(op)) {
    for (Block &block : op->getRegion(0))
      for (Operation &child : block)
        if (hasBackup(&child))
          return false;
  } else if (hasBackup(op)) {
    return false;
  }

  LLVM_DEBUG(DBGS() << "skip '" << passArg << "' reason=AivDisabledPasses\n");
  return true;
}

} // namespace bishengir

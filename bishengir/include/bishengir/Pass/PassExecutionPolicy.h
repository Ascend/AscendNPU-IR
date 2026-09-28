//===- PassExecutionPolicy.h - Pipeline-wide pass filtering -*- C++ -*-===//
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

#ifndef BISHENGIR_PASS_PASSEXECUTIONPOLICY_H
#define BISHENGIR_PASS_PASSEXECUTIONPOLICY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace bishengir {

/// Pipeline-wide execution mode derived from the module being compiled.
/// Mix is the full/default mode: the complete pipeline runs unchanged. Aiv
/// is the pure vector-kernel mode: passes in the AivDisabledPasses denylist
/// are skipped for the whole pipeline run.
///
/// This is state of a single compilation/run: it lives in the
/// BiShengIRPassManager that runs the pipeline and must never be stored in a
/// process- or context-wide global, because one MLIRContext may run
/// different modules (and therefore different modes) sequentially.
enum class PipelineMode {
  Mix, // full/default pipeline mode
  Aiv, // pure VV (pure-AIV vector) pipeline mode
};

/// Detect the pipeline mode from the module's functions: the mode is Aiv
/// only when at least one function carries a `mix_mode` attribute and every
/// function carrying `mix_mode` declares `"aiv"`. Any other case — no
/// `mix_mode` at all, a non-`"aiv"` value, or disagreeing values — keeps the
/// default full (Mix) mode. The `mix_mode` spelling and its "aiv" value are
/// the ones already used across the codebase (e.g. regbase::inferMixedCV,
/// HFusion OptimizeScalarTransfers, HIVM MarkSyncBlockLockWithSubblock).
PipelineMode detectPipelineMode(mlir::ModuleOp module);

/// The central denylist of pass arguments (see Pass::getArgument()) that the
/// whole pipeline skips when running in Aiv mode. It is a denylist, not an
/// allowlist: passes that are not listed keep executing in Aiv mode.
///
/// The registry is a process-wide set of static pass argument strings, so
/// unlike the per-run PipelineMode it is safe to share. It is read-only
/// through this accessor; the only mutation point is
/// registerAivDisabledPasses(), called while a pipeline is being built, before
/// it runs. HIVM RegBase registers its entries that way; the registry starts
/// empty so builds without that pipeline behave as if Aiv mode disabled
/// nothing.
const llvm::StringSet<> &getAivDisabledPasses();

/// Register extra pass arguments in the Aiv denylist. Idempotent; called
/// before the pipeline that may consult it runs.
void registerAivDisabledPasses(std::initializer_list<llvm::StringRef> args);

/// Policy deciding, for one pipeline run, whether a pass is skipped
/// globally (for every operation it would run on). This is the central
/// Mix/Aiv filter consulted by the BiShengIRPassManager action handler
/// before the per-operation annotation.filter_passes (FilterPassesAttr)
/// filtering; it intentionally mirrors the FilterPassesAttr pass identity
/// (Pass::getArgument()).
///
/// The policy detects the pipeline mode lazily from the module the pass
/// executes on, on the first action of the run, and freezes it for the rest
/// of the run: the mode is a property of the module being compiled, and the
/// passes that would rewrite `mix_mode` are not in the denylist. Detection is
/// guarded by an atomic flag because MLIR may dispatch per-operation pass
/// actions from worker threads; concurrent first detections compute the same
/// value from the same module, so the race is benign by construction.
class PassExecutionPolicy {
public:
  /// Returns true when the pass must be skipped for the whole pipeline run:
  /// Aiv mode and the pass argument is in AivDisabledPasses, and the
  /// operation the pass would run on (or, for module ops, one of the ops
  /// visible in the module body) is not exempt from the Aiv denylist.
  /// Adaptor passes (empty argument) are never skipped — they only
  /// orchestrate nested pipelines.
  bool shouldSkipGlobally(const mlir::Pass &pass, mlir::Operation *op) const;

  /// The lazily-detected mode; PipelineMode::Mix until the first
  /// shouldSkipGlobally call of a run detects otherwise.
  PipelineMode getMode() const { return mode.load(std::memory_order_acquire); }

private:
  mutable std::atomic<PipelineMode> mode{PipelineMode::Mix};
  mutable std::atomic<bool> modeDetected{false};
};

} // namespace bishengir

#endif // BISHENGIR_PASS_PASSEXECUTIONPOLICY_H

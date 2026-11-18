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
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace bishengir {

/// Kernel kind of the module being compiled, taken from
/// `hivm.module_core_type`. Unspecified means the attribute is missing or
/// unrecognized: every pass runs. Mix, Aiv, and Aic match MixKernel,
/// PureAivKernel, and PureAicKernel on HIVMPass.
///
/// This is state of a single compilation/run: it lives in the
/// BiShengIRPassManager that runs the pipeline and must never be stored in a
/// process- or context-wide global, because one MLIRContext may run
/// different modules (and therefore different modes) sequentially.
enum class PipelineMode {
  Unspecified, // no recognized module core type
  Mix,         // hivm.module_core_type = MIX
  Aiv,         // hivm.module_core_type = AIV
  Aic,         // hivm.module_core_type = AIC
};

/// Detect the pipeline mode from the module attribute `hivm.module_core_type`.
/// `AIV`, `AIC`, and `MIX` select that kernel kind. A missing or unrecognized
/// attribute selects Unspecified, so the pipeline does not filter. Function
/// attributes such as `mix_mode` are not consulted: the module core type is
/// the summary already maintained for the device functions in the module.
PipelineMode detectPipelineMode(mlir::ModuleOp module);

/// Record that `args` apply to `kind`. Idempotent. A pass may be registered
/// for more than one kind. Passes that are never registered have no kernel
/// kind list and are not filtered. HIVM RegBase registers the lists generated
/// from HIVMPass `kernelKinds` before the pipeline runs.
void registerKernelKindPasses(PipelineMode kind,
                              std::initializer_list<llvm::StringRef> args);

/// True when `passArg` was registered for `kind`. Unspecified, an empty
/// argument, and a pass with no registration return false.
bool passAppliesToKernelKind(llvm::StringRef passArg, PipelineMode kind);

/// Pass arguments whose HIVMPass `classifyKernel` bit is set. After such
/// a pass runs, BiShengIRPassManager invokes the classify-kernel callback.
/// Idempotent. HIVM registers the passes generated from Passes.td.
void registerClassifyKernelPasses(std::initializer_list<llvm::StringRef> args);

/// True when `passArg` was registered with registerClassifyKernelPasses.
bool passClassifiesKernel(llvm::StringRef passArg);

/// Callback installed by HIVM. It re-infers function core types after a pass
/// with `classifyKernel` set. maybeClassifyKernel only calls this when that
/// bit is set. The Pass library does not know the inference pass. The last
/// registration wins. Called only for passes that actually ran.
using ClassifyKernelCallback = mlir::LogicalResult (*)(mlir::Operation *op);
void registerClassifyKernelCallback(ClassifyKernelCallback callback);
ClassifyKernelCallback getClassifyKernelCallback();

/// Called when an Aiv-denylisted pass is skipped, in place of that pass.
/// Keyed by the skipped pass argument. No entry means the skip does nothing.
/// The last registration for an argument wins. The callback runs on the
/// operation the skipped pass would have run on, and receives that pass so
/// substitutes can honor its options (for example cleanup vs stamp).
using AivSubstituteCallback = mlir::LogicalResult (*)(mlir::Operation *op,
                                                      const mlir::Pass &pass);
void registerAivSubstitute(llvm::StringRef skippedPassArg,
                           AivSubstituteCallback callback);
AivSubstituteCallback getAivSubstitute(llvm::StringRef skippedPassArg);

/// Policy deciding, for one pipeline run, whether a pass is skipped
/// globally (for every operation it would run on). This is the kernel-kind
/// filter consulted by the BiShengIRPassManager action handler before the
/// per-operation annotation.filter_passes (FilterPassesAttr) filtering; it
/// intentionally mirrors the FilterPassesAttr pass identity
/// (Pass::getArgument()).
///
/// The policy reads `hivm.module_core_type` on the module the pass executes
/// on before every decision. A later pass may replace or delete that
/// attribute (InferFuncCoreType, classify-kernel follow-up), and the next
/// decision uses the new value. A missing, deleted, or unrecognized attribute
/// selects Unspecified and runs every pass. A recognized kind skips a pass
/// whose registered kernel-kind list does not include that kind. Concurrent
/// actions read the same attribute; the stored mode is only the last
/// decision and is not a frozen key.
class PassExecutionPolicy {
public:
  /// Returns true when the pass must be skipped for the whole pipeline run:
  /// the module core type is a recognized kernel kind, the pass was
  /// registered with a kernel-kind list, that list does not include the
  /// kind, and the operation the pass would run on (or, for module ops, one
  /// of the ops visible in the module body) is not exempt. Adaptor passes
  /// (empty argument) and passes with no kernel-kind registration are never
  /// skipped.
  bool shouldSkipGlobally(const mlir::Pass &pass, mlir::Operation *op) const;

  /// The mode from the most recent shouldSkipGlobally decision.
  /// PipelineMode::Unspecified until the first call of a run.
  PipelineMode getMode() const { return mode.load(std::memory_order_acquire); }

private:
  mutable std::atomic<PipelineMode> mode{PipelineMode::Unspecified};
};

} // namespace bishengir

#endif // BISHENGIR_PASS_PASSEXECUTIONPOLICY_H

//===- Passes.h - HIVM v2 pipeline entry points -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry points for the HIVM v2 pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_PIPELINES_V2_PASSES_H
#define BISHENGIR_DIALECT_HIVM_PIPELINES_V2_PASSES_H

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hivm {
namespace v2 {

struct V2PipelineOptions : public mlir::PassPipelineOptions<V2PipelineOptions> {
  PassOptions::Option<std::string> target{*this, "target",
                                          llvm::cl::desc("Target device name"),
                                          llvm::cl::init("Ascend950PR_9579")};

  /// Append the device spec from `target` (bishengir-opt only).
  PassOptions::Option<bool> setTargetSpec{
      *this, "set-target-spec",
      llvm::cl::desc("Append the target device spec from --target"),
      llvm::cl::init(true)};

  PassOptions::Option<bool> enablePreload{*this, "enable-preload",
                                          llvm::cl::desc("Run create-preload"),
                                          llvm::cl::init(false)};

  PassOptions::Option<bool> enableMultiBuffer{
      *this, "enable-multi-buffer",
      llvm::cl::desc("Mark/enable multi-buffering on the unsplit MIX"),
      llvm::cl::init(false)};

  // V2 uses graph solvers except for intra-core barrier-all mode.
  PassOptions::Option<bool> disableHIVMAutoInjectSync{
      *this, "disable-hivm-auto-inject-sync",
      llvm::cl::desc("Disable intra-core auto-sync"), llvm::cl::init(false)};

  PassOptions::Option<bool> disableAutoInjectBlockSync{
      *this, "disable-auto-inject-block-sync",
      llvm::cl::desc("Disable cross-core auto-sync"), llvm::cl::init(false)};

  PassOptions::Option<bool> enableHIVMInjectBarrierAllSync{
      *this, "enable-hivm-inject-barrier-all-sync",
      llvm::cl::desc("Use inject-sync in barrier-all mode"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableHIVMInjectBlockAllSync{
      *this, "enable-hivm-inject-block-all-sync",
      llvm::cl::desc("Use cross-core-gss in block-all mode"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableHIVMUnitFlagSync{
      *this, "enable-hivm-unit-flag-sync",
      llvm::cl::desc("Enable unit-flag modes for intra-core synchronization"),
      llvm::cl::init(false)};

  PassOptions::Option<std::string> hivmSyncSolverVersion{
      *this, "hivm-sync-solver-version",
      llvm::cl::desc("Select the auto-sync solver implementation"),
      llvm::cl::init("v2")};
  PassOptions::Option<PlanMemoryUBCopyBankMode> planMemoryUBCopyBankMode{
      *this, "plan-memory-ub-copy-bank-mode",
      llvm::cl::desc("PlanMemory UB-to-UB copy bank conflict avoidance mode"),
      llvm::cl::values(
          clEnumValN(PlanMemoryUBCopyBankMode::OFF, "off", "Disable"),
          clEnumValN(PlanMemoryUBCopyBankMode::PREFER, "prefer", "Prefer"),
          clEnumValN(PlanMemoryUBCopyBankMode::REQUIRE, "require", "Require")),
      llvm::cl::init(PlanMemoryUBCopyBankMode::OFF)};
};

/// Phase builders shared by the opt pipeline and the compile driver.
void buildV2TensorPhase(OpPassManager &pm, const V2PipelineOptions &options);
void buildV2BufferizationPhase(OpPassManager &pm,
                               const V2PipelineOptions &options);
void buildV2PostBufferizationPhase(OpPassManager &pm,
                                   const V2PipelineOptions &options);
void buildV2PerCoreTail(OpPassManager &pm, const V2PipelineOptions &options);

/// Add the "hivm-v2-pipeline" pipeline to `pm`.
void buildV2HIVMPipeline(OpPassManager &pm, const V2PipelineOptions &options);

/// Register the v2 pipeline.
void registerV2HIVMPipelines();

} // namespace v2
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_V2_PASSES_H

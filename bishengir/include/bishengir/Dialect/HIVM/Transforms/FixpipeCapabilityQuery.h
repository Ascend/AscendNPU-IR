//===- FixpipeCapabilityQuery.h - Inline Fixpipe capability helpers ------===//
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
//
// Shared helpers for Fixpipe rewrite consumers that query the production
// execution profile. Missing #hacc.target is not Ascend910B. Only Native is
// treated as an affirmative rewrite permit.
//
// Remaining W5 work that still needs owner decisions (not invented here):
// - W5.b: late address-space / layout / shape / quant_scale presence schema
// - W5.d: FIXPIPE-TD-001 optional dma_mode ("unresolved") semantics
// - Pure address-space / L1-vs-UB insert heuristics in InsertFixpipe that are
//   not execution-schema questions
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_FIXPIPECAPABILITYQUERY_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_FIXPIPECAPABILITYQUERY_H

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Target/NPU/DMA/Fixpipe/FixpipeCapability.h"
#include "mlir/Pass/AnalysisManager.h"

#include <optional>

namespace mlir::hivm {

/// Device and architecture for one module. The architecture is computed once
/// when the analysis is built.
struct CachedModuleTarget {
  hacc::TargetDevice device;
  bishengir::npu::NPUArchitectureKind architecture;
};

/// Resolved `#hacc.target` for every module under one pass anchor. An inner
/// module overrides an outer one. The pass manager caches one instance per
/// anchor operation. It is not a process-wide device.
class ModuleTargetAnalysis {
public:
  explicit ModuleTargetAnalysis(Operation *root);

  /// Nearest enclosing module's resolved target. A module created after this
  /// analysis was built is resolved on demand.
  std::optional<CachedModuleTarget> lookup(Operation *op) const;

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<ModuleTargetAnalysis>();
  }

private:
  llvm::DenseMap<Operation *, std::optional<CachedModuleTarget>> byModule;
};

/// Makes `analysis` the lookup used by `findEnclosingModuleTarget` on this
/// thread until the scope ends. Passes that do not change `#hacc.target`
/// should preserve `ModuleTargetAnalysis`.
class ModuleTargetAnalysisScope {
public:
  explicit ModuleTargetAnalysisScope(const ModuleTargetAnalysis &analysis);
  ~ModuleTargetAnalysisScope();
  ModuleTargetAnalysisScope(const ModuleTargetAnalysisScope &) = delete;
  ModuleTargetAnalysisScope &
  operator=(const ModuleTargetAnalysisScope &) = delete;

  static const ModuleTargetAnalysis *current();

private:
  const ModuleTargetAnalysis *previous = nullptr;
  static thread_local const ModuleTargetAnalysis *active;
};

/// Nearest enclosing module target. Uses `ModuleTargetAnalysisScope` when a
/// pass has installed one. Missing attributes are skipped; there is no
/// Ascend910B default.
std::optional<CachedModuleTarget> findEnclosingModuleTarget(Operation *op);

/// Architecture cached for the module enclosing `op`. This does not call
/// `hacc::getArchitecture` again when a pass analysis is active.
std::optional<bishengir::npu::NPUArchitectureKind>
lookupModuleArchitecture(Operation *op);

/// Query a candidate that does not exist as IR yet. Either address space left
/// unset makes the result Unknown.
bishengir::npu::NativeSupport queryFixpipeCreate(
    Operation *anchor, Type srcElementType, Type dstElementType,
    FixpipeDMAMode dmaMode,
    FixpipePreQuantMode preQuantMode = FixpipePreQuantMode::NO_QUANT,
    FixpipePreReluMode preRelu = FixpipePreReluMode::NO_RELU,
    bool channelSplit = false,
    std::optional<AddressSpace> srcAddressSpace = std::nullopt,
    std::optional<AddressSpace> dstAddressSpace = std::nullopt,
    bool quantScalePresent = false);

/// True when `queryFixpipeCreate` returns Native.
bool isNativeFixpipeCreate(
    Operation *anchor, Type srcElementType, Type dstElementType,
    FixpipeDMAMode dmaMode,
    FixpipePreQuantMode preQuantMode = FixpipePreQuantMode::NO_QUANT,
    FixpipePreReluMode preRelu = FixpipePreReluMode::NO_RELU,
    bool channelSplit = false,
    std::optional<AddressSpace> srcAddressSpace = std::nullopt,
    std::optional<AddressSpace> dstAddressSpace = std::nullopt,
    bool quantScalePresent = false);

/// Attributes that are not part of the Fixpipe execution schema but can
/// interact with a candidate dma_mode change. Non-default values refuse the
/// NZ2DN transpose pilot until those relations are modeled.
bool hasUnmodeledNz2dnConflict(FixpipeOp op);

} // namespace mlir::hivm

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_FIXPIPECAPABILITYQUERY_H

//===- FixpipeCapabilityQuery.cpp - Inline Fixpipe capability helpers ----===//
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

#include "bishengir/Dialect/HIVM/Transforms/FixpipeCapabilityQuery.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"

namespace mlir::hivm {

static std::optional<hacc::TargetDevice>
findEnclosingTargetDeviceUncached(Operation *op) {
  Operation *cursor = op;
  while (cursor) {
    ModuleOp module = dyn_cast<ModuleOp>(cursor);
    if (!module)
      module = cursor->getParentOfType<ModuleOp>();
    if (!module)
      return std::nullopt;
    if (hacc::utils::hasTargetAttr(module))
      return hacc::utils::getTargetDevice(module);
    cursor = module->getParentOp();
  }
  return std::nullopt;
}

static std::optional<CachedModuleTarget> resolveModuleTarget(Operation *op) {
  std::optional<hacc::TargetDevice> device =
      findEnclosingTargetDeviceUncached(op);
  if (!device)
    return std::nullopt;
  return CachedModuleTarget{*device, hacc::getArchitecture(*device)};
}

thread_local const ModuleTargetAnalysis *ModuleTargetAnalysisScope::active =
    nullptr;

ModuleTargetAnalysisScope::ModuleTargetAnalysisScope(
    const ModuleTargetAnalysis &analysis)
    : previous(active) {
  active = &analysis;
}

ModuleTargetAnalysisScope::~ModuleTargetAnalysisScope() { active = previous; }

const ModuleTargetAnalysis *ModuleTargetAnalysisScope::current() {
  return active;
}

ModuleTargetAnalysis::ModuleTargetAnalysis(Operation *root) {
  auto record = [&](ModuleOp module) {
    byModule.try_emplace(module.getOperation(), resolveModuleTarget(module));
  };
  for (Operation *cursor = root; cursor; cursor = cursor->getParentOp()) {
    if (auto module = dyn_cast<ModuleOp>(cursor))
      record(module);
  }
  root->walk([&](ModuleOp module) { record(module); });
}

std::optional<CachedModuleTarget>
ModuleTargetAnalysis::lookup(Operation *op) const {
  ModuleOp module = dyn_cast<ModuleOp>(op);
  if (!module)
    module = op->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;
  auto it = byModule.find(module.getOperation());
  if (it != byModule.end())
    return it->second;
  return resolveModuleTarget(op);
}

std::optional<CachedModuleTarget> findEnclosingModuleTarget(Operation *op) {
  if (const ModuleTargetAnalysis *analysis =
          ModuleTargetAnalysisScope::current())
    return analysis->lookup(op);
  return resolveModuleTarget(op);
}

std::optional<bishengir::npu::NPUArchitectureKind>
lookupModuleArchitecture(Operation *op) {
  if (std::optional<CachedModuleTarget> target = findEnclosingModuleTarget(op))
    return target->architecture;
  return std::nullopt;
}

bool hasUnmodeledNz2dnConflict(FixpipeOp op) {
  if (op.getDualDstMode().getDualDstMode() != FixpipeDualDstMode::NO_DUAL)
    return true;
  if (op.getSubBlockIdx() != FixpipeSubBlock::SUB_BLOCK_0)
    return true;
  // Default c0_pad_en is true. An explicit false is unconfirmed with NZ2DN.
  if (!op.getC0PadEn())
    return true;
  return false;
}

} // namespace mlir::hivm

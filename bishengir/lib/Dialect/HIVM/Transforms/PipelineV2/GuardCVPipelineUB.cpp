//===- GuardCVPipelineUB.cpp - Drop preload when UB estimate overflows ----===//
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

#include "Util.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>

#define DEBUG_TYPE "guard-cv-pipeline-ub"

namespace mlir {
namespace hivm {
#define GEN_PASS_DEF_GUARDCVPIPELINEUB
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace hivm
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct GuardCVPipelineUBPass
    : public mlir::hivm::impl::GuardCVPipelineUBBase<GuardCVPipelineUBPass> {
  using Base::Base;
  void runOnOperation() final;
};

constexpr int64_t kBitsPerByte = 8;
constexpr int64_t kBytesPerKiB = 1024;
// Cube:vector on the current regbase targets. The unsplit estimate is compared
// with this many copies of the per-core UB budget.
constexpr int64_t kCubeVectorSplit = 2;

bool usesSimtUbBudget(func::FuncOp func) {
  if (auto vf = func->getAttrOfType<VFModeAttr>(VFModeAttr::name)) {
    if (vf.getValue() == VFMode::SIMT || vf.getValue() == VFMode::MIX)
      return true;
  }
  if (auto core =
          func->getAttrOfType<TFuncCoreTypeAttr>(TFuncCoreTypeAttr::name))
    if (core.getFuncCoreType() == TFuncCoreType::MIX)
      return true;
  return false;
}

std::optional<int64_t> availableUbBits(func::FuncOp func, int simtKiB) {
  auto module = func->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;
  if (hacc::utils::isRegBasedArch(module) && usesSimtUbBudget(func))
    return static_cast<int64_t>(simtKiB) * kBytesPerKiB * kBitsPerByte;
  auto spec = hacc::utils::getNPUTargetSpec(module);
  if (!spec)
    return std::nullopt;
  auto ub = spec->getSpecForIdentifierEnum(hacc::DeviceSpec::UB_SIZE);
  if (!ub)
    return std::nullopt;
  return cast<IntegerAttr>(ub.getValue()).getInt();
}

std::optional<int64_t> staticBitSize(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape() ||
      !shaped.getElementType().isIntOrFloat())
    return std::nullopt;
  int64_t bits = shaped.getElementType().getIntOrFloatBitWidth();
  for (int64_t dim : shaped.getShape()) {
    if (bits > std::numeric_limits<int64_t>::max() / std::max<int64_t>(dim, 1))
      return std::nullopt;
    bits *= dim;
  }
  return bits;
}

std::optional<int32_t> preloadNum(ScopeOp scope) {
  auto attr = scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
  if (!attr)
    return std::nullopt;
  return static_cast<int32_t>(attr.getInt());
}

ScopeOp parentPreloadScope(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto scope = dyn_cast<ScopeOp>(parent);
    if (scope && scope->hasAttr(PreloadNumAttr::name))
      return scope;
  }
  return nullptr;
}

bool isViewLike(Operation *op) {
  return isa<ViewLikeOpInterface>(op) ||
         isa<bufferization::ToTensorOp, bufferization::ToMemrefOp,
             memref::MemorySpaceCastOp, memref::CastOp>(op);
}

bool effectOnAlloc(Operation *op, Value alloc, bool write) {
  if (isa<ScopeOp, annotation::MarkOp>(op))
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  utils::collectAllEffects(op, effects);
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    if (!value)
      continue;
    bool matches = write ? isa<MemoryEffects::Write>(effect.getEffect())
                         : isa<MemoryEffects::Read>(effect.getEffect());
    if (matches && traceAlloc(value) == traceAlloc(alloc))
      return true;
  }
  return false;
}

void collectSideScopes(Value root, SmallVectorImpl<ScopeOp> &producers,
                       SmallVectorImpl<ScopeOp> &consumers) {
  DenseSet<Value> seen;
  SmallVector<Value> worklist = {root};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (isa<annotation::MarkOp>(user))
        continue;
      ScopeOp scope = parentPreloadScope(user);
      bool wrote = effectOnAlloc(user, root, /*write=*/true);
      bool read = effectOnAlloc(user, root, /*write=*/false);
      if (scope && wrote)
        producers.push_back(scope);
      if (scope && (read || (!wrote && !isViewLike(user))))
        consumers.push_back(scope);
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
}

int32_t multiBufferCount(ArrayRef<int32_t> producers,
                         ArrayRef<int32_t> consumers) {
  if (producers.empty() || consumers.empty())
    return 1;
  int32_t maxDiff = 0;
  for (int32_t producer : producers)
    for (int32_t consumer : consumers)
      maxDiff = std::max(maxDiff, std::abs(producer - consumer));
  return maxDiff + 1;
}

SmallVector<int32_t> numsOf(ArrayRef<ScopeOp> scopes) {
  SmallVector<int32_t> nums;
  for (ScopeOp scope : scopes)
    if (std::optional<int32_t> num = preloadNum(scope))
      nums.push_back(*num);
  return nums;
}

bool isUbTightlyCoupled(memref::AllocOp alloc) {
  if (!utils::getAnnotateOpWithAttr(alloc.getResult(),
                                    HIVMTightlyCoupledBufferAttr::name))
    return false;
  auto space = getOptionalHIVMAddressSpace(alloc.getType());
  return space && *space == AddressSpace::UB;
}

void collectConsumerScopes(Value value, ScopeOp producer,
                           SmallVectorImpl<ScopeOp> &consumers) {
  DenseSet<Operation *> seen;
  SmallVector<Operation *> worklist(value.getUsers().begin(),
                                    value.getUsers().end());
  while (!worklist.empty()) {
    Operation *user = worklist.pop_back_val();
    if (!seen.insert(user).second || isa<annotation::MarkOp>(user))
      continue;
    if (ScopeOp scope = parentPreloadScope(user)) {
      if (scope != producer)
        consumers.push_back(scope);
      continue;
    }
    for (Value result : user->getResults())
      worklist.append(result.getUsers().begin(), result.getUsers().end());
  }
}

int64_t estimateLoopUbBits(scf::ForOp loop) {
  int64_t total = 0;
  DenseSet<Operation *> countedAllocs;
  loop.walk([&](memref::AllocOp alloc) {
    if (!isUbTightlyCoupled(alloc))
      return;
    SmallVector<ScopeOp> producers;
    SmallVector<ScopeOp> consumers;
    collectSideScopes(alloc.getResult(), producers, consumers);
    auto bits = staticBitSize(alloc.getType());
    if (!bits)
      return;
    int32_t copies = multiBufferCount(numsOf(producers), numsOf(consumers));
    total += *bits * copies;
    countedAllocs.insert(alloc);
    LLVM_DEBUG(llvm::dbgs() << "[guard-cv-pipeline-ub] tcb " << alloc.getType()
                            << " x" << copies << "\n");
  });

  loop.walk([&](ScopeOp scope) {
    if (!scope.getRegion().hasOneBlock())
      return;
    auto producerNum = preloadNum(scope);
    if (!producerNum)
      return;
    auto terminator = scope.getRegion().front().getTerminator();
    for (auto [operand, result] :
         llvm::zip(terminator->getOperands(), scope.getResults())) {
      if (memref::AllocOp alloc = traceAlloc(operand))
        if (countedAllocs.contains(alloc))
          continue;
      SmallVector<ScopeOp> consumers;
      collectConsumerScopes(result, scope, consumers);
      if (consumers.empty())
        continue;
      auto bits = staticBitSize(result.getType());
      if (!bits)
        continue;
      int32_t copies = multiBufferCount({*producerNum}, numsOf(consumers));
      total += *bits * copies;
      LLVM_DEBUG(llvm::dbgs() << "[guard-cv-pipeline-ub] tensor "
                              << result.getType() << " x" << copies << "\n");
    }
  });
  return total;
}

void inlineScope(ScopeOp scope) {
  if (!scope.getRegion().hasOneBlock())
    return;
  Block &block = scope.getRegion().front();
  Operation *terminator = block.getTerminator();
  scope.setNoInline(false);
  for (auto [result, operand] :
       llvm::zip_equal(scope.getResults(), terminator->getOperands()))
    result.replaceAllUsesWith(operand);
  scope->getBlock()->getOperations().splice(
      Block::iterator(scope), block.getOperations(), block.begin(),
      std::prev(block.end()));
  scope.erase();
}

void stripPreloadAttrs(Operation *op) {
  op->removeAttr(PreloadNumAttr::name);
  op->removeAttr(MaxPreloadNumAttr::name);
  op->removeAttr(PreloadLocalBufferAttr::name);
  op->removeAttr(PreloadWorkspaceAttr::name);
  op->removeAttr(CVPipelinedLoopAttr::name);
  op->removeAttr(LegacyCVPipelineAttr::name);
}

func::FuncOp findSnapshot(func::FuncOp func) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  if (!module)
    return nullptr;
  for (func::FuncOp candidate : module.getOps<func::FuncOp>()) {
    auto ref = candidate->getAttrOfType<FlatSymbolRefAttr>(
        kCVPipelineSnapshotAttrName);
    if (ref && ref.getValue() == func.getSymName())
      return candidate;
  }
  return nullptr;
}

void restoreFromSnapshot(func::FuncOp func, func::FuncOp snapshot) {
  func.getBody().dropAllReferences();
  func.getBody().getBlocks().clear();
  IRMapping mapping;
  snapshot.getBody().cloneInto(&func.getBody(), mapping);
}

void dropPreload(scf::ForOp loop) {
  SmallVector<ScopeOp> scopes;
  loop.walk([&](ScopeOp scope) { scopes.push_back(scope); });
  for (ScopeOp scope : llvm::reverse(scopes))
    inlineScope(scope);
  loop.walk([&](Operation *op) { stripPreloadAttrs(op); });
  stripPreloadAttrs(loop);
}

void GuardCVPipelineUBPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (isCVPipelineSnapshot(func) || hasLegacyCVPipelinedLoop(func))
    return;
  std::optional<int64_t> available = availableUbBits(func, simtVFDynamicSize);
  if (!available)
    return;
  int64_t budget = *available * kCubeVectorSplit;
  SmallVector<scf::ForOp> loops;
  func.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(CVPipelinedLoopAttr::name))
      loops.push_back(loop);
  });
  bool overflow = false;
  for (scf::ForOp loop : loops) {
    int64_t estimated = estimateLoopUbBits(loop);
    LLVM_DEBUG(llvm::dbgs() << "[guard-cv-pipeline-ub] estimated " << estimated
                            << " budget " << budget << "\n");
    if (estimated > budget)
      overflow = true;
  }
  if (!overflow)
    return;
  if (func::FuncOp snapshot = findSnapshot(func)) {
    restoreFromSnapshot(func, snapshot);
    return;
  }
  for (scf::ForOp loop : loops)
    if (estimateLoopUbBits(loop) > budget)
      dropPreload(loop);
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createGuardCVPipelineUBPass(
    const GuardCVPipelineUBOptions &options) {
  return std::make_unique<GuardCVPipelineUBPass>(options);
}

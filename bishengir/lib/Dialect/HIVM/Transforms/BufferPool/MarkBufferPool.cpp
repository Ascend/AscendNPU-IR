//===- MarkBufferPool.cpp - Mark shared load buffer capacity -----===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/BufferPoolUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/CustomMacroSync.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir {
#define GEN_PASS_DEF_MARKBUFFERPOOL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::buffer_pool;

namespace {
using PipePair = std::pair<PIPE, PIPE>;
using PipeCounts = DenseMap<PipePair, int64_t>;

struct RootResources {
  Value root;
  int64_t slots = 1;
  int64_t l1SlotBytes = 0;
  SmallVector<PipePair> directions;
};

struct Pool {
  SmallVector<LoadInfo *, 4> members;
  int64_t size = 0;
  int64_t id = -1;
};

struct FunctionPlan {
  func::FuncOp function;
  SmallVector<LoadInfo, 4> loads;
  SmallVector<Pool, 4> pools;
};

static int64_t getSlotCount(Value root) {
  auto mark = utils::getAnnotateOpWithAttr(root, MultiBufferAttr::name);
  if (!mark)
    return 1;
  auto count = (*mark)->getAttrOfType<IntegerAttr>(MultiBufferAttr::name);
  return count.getInt();
}

static bool compatible(const LoadInfo &a, const LoadInfo &b) {
  auto aAlloc = a.root.getDefiningOp<memref::AllocOp>();
  auto bAlloc = b.root.getDefiningOp<memref::AllocOp>();
  return a.body == b.body && a.core == b.core &&
         a.root.getType() == b.root.getType() &&
         aAlloc.getAlignmentAttr() == bAlloc.getAlignmentAttr();
}

static SmallVector<PipePair> getDirections(ArrayRef<Access> accesses) {
  llvm::SetVector<PipePair> pairs;
  for (const Access &writer : accesses) {
    if (!writer.write)
      continue;
    for (const Access &other : accesses) {
      if (writer.pipe == other.pipe)
        continue;
      pairs.insert({writer.pipe, other.pipe});
      pairs.insert({other.pipe, writer.pipe});
    }
  }
  return pairs.takeVector();
}

/// Use the same static shape and target alignment as PlanMemory.
static FailureOr<int64_t> getSlotBytes(memref::AllocOp alloc, int64_t align) {
  auto type = alloc.getType();
  auto count = utils::getStaticTotalSize(type.getShape());
  if (!count || !type.getElementType().isIntOrFloat() || align <= 0)
    return failure();
  int64_t bits;
  if (llvm::MulOverflow(*count, int64_t(type.getElementTypeBitWidth()), bits))
    return failure();
  int64_t bytes = llvm::divideCeilSigned(bits, int64_t(8));
  if (auto explicitAlign = alloc.getAlignment())
    align = std::max(align, int64_t(*explicitAlign));
  if (bytes <= 0 || bytes > std::numeric_limits<int64_t>::max() - align)
    return failure();
  return llvm::alignTo(bytes, align);
}

static int64_t requiredSlots(const Pool &pool) {
  DenseMap<Operation *, DenseSet<Value>> inputs;
  for (const LoadInfo *load : pool.members)
    for (Operation *reader : load->readers)
      inputs[reader].insert(load->root);
  int64_t result = 1;
  for (const auto &[reader, roots] : inputs)
    result = std::max(result, int64_t(roots.size()));
  return result;
}

/// Fixed IDs are counted once per directed bank. Macro requirements use the
/// same resolution as GSS; a dynamic pre-existing ID has no safe summary here.
static bool collectInternalReservations(func::FuncOp function,
                                        PipeCounts &reservations) {
  DenseMap<PipePair, DenseSet<int64_t>> fixed;
  bool known = true;
  bool hasL1Padding = false;
  function.walk([&](Operation *op) {
    if (auto padding = dyn_cast<VBrcOp>(op))
      hasL1Padding |= getOptionalHIVMAddressSpace(padding.getDst().getType()) ==
                      AddressSpace::L1;
    if (isa<SetFlagOp, WaitFlagOp>(op)) {
      auto set = op->getAttrOfType<PipeAttr>("set_pipe");
      auto wait = op->getAttrOfType<PipeAttr>("wait_pipe");
      auto event = op->getAttrOfType<EventAttr>("static_event_id");
      if (!set || !wait || !event) {
        known = false;
        return;
      }
      fixed[{set.getPipe(), wait.getPipe()}].insert(int64_t(event.getEvent()));
    }
    if (auto macro = dyn_cast<CustomMacroOp>(op)) {
      for (auto slot : macro.getSyncEventSlots()) {
        auto pair = syncsolver::resolveSlotPipePair(slot, macro);
        if (pair.first == PIPE::PIPE_UNASSIGNED ||
            pair.second == PIPE::PIPE_UNASSIGNED) {
          known = false;
          continue;
        }
        if (auto event = slot.getEvent())
          fixed[pair].insert(int64_t(event.getEvent()));
        else
          ++reservations[pair];
      }
    }
  });
  for (const auto &[pair, ids] : fixed) {
    int64_t available = syncsolver::getHWAvailableEventIdNum(
        syncsolver::SyncMode::INTRA_CORE_SYNC, pair.first, pair.second);
    for (int64_t id : ids)
      if (id >= 0 && id < available)
        ++reservations[pair];
  }
  // L1 padding shares one ordinary GSS event across pool members and outside
  // buffers. Reserve its capacity without choosing the eventual event ID.
  if (hasL1Padding)
    ++reservations[{PIPE::PIPE_MTE1, PIPE::PIPE_MTE2}];
  return known;
}

static LogicalResult analyzeFunction(FunctionPlan &plan) {
  auto function = plan.function;
  bool invalid = false;
  function.walk([&](scope::ScopeOp scope) {
    if (!scope->hasAttr(LoadPreloadAttr::name))
      return;
    // SplitMixKernel temporarily keeps empty scopes in the opposite core.
    Operation *transfer = findLoadTransfer(scope);
    if (!transfer)
      return;
    auto info = analyzeLoad(scope, transfer);
    if (failed(info)) {
      invalid = true;
      return;
    }
    plan.loads.push_back(std::move(*info));
  });
  if (invalid)
    return failure();
  if (plan.loads.empty())
    return success();

  DenseMap<Value, const LoadInfo *> loadsByRoot;
  DenseMap<int64_t, size_t> existing;
  int64_t nextId = 0;
  for (LoadInfo &load : plan.loads) {
    if (!loadsByRoot.try_emplace(load.root, &load).second)
      return load.scope.emitError("multiple independent load scopes write the "
                                  "same root buffer");
    auto scopeId = load.scope->getAttrOfType<IntegerAttr>(MemPoolIdAttr::name);
    auto rootMark =
        utils::getAnnotateOpWithAttr(load.root, MemPoolIdAttr::name);
    if (scopeId) {
      // Existing pools already own their capacity. Read it for resource
      // accounting when this pass also encounters newly separated loads.
      auto size =
          (*rootMark)->getAttrOfType<IntegerAttr>(MemPoolSizeAttr::name);
      auto [it, inserted] =
          existing.try_emplace(scopeId.getInt(), plan.pools.size());
      if (inserted) {
        Pool pool;
        pool.id = scopeId.getInt();
        pool.size = size.getInt();
        plan.pools.push_back(std::move(pool));
      }
      Pool &pool = plan.pools[it->second];
      pool.members.push_back(&load);
      nextId = std::max(nextId, scopeId.getInt() + 1);
    }
  }
  auto spec =
      hacc::utils::getNPUTargetSpec(function->getParentOfType<ModuleOp>());
  if (!spec)
    return success(); // No capacity information: keep independent buffers.
  auto getBytes = [&](hacc::DeviceSpec name) -> int64_t {
    auto entry = spec->getSpecForIdentifierEnum(name);
    if (!entry)
      return 0;
    auto value = dyn_cast<IntegerAttr>(entry.getValue());
    return value ? value.getInt() / 8 : 0;
  };
  int64_t capacity = getBytes(hacc::DeviceSpec::L1_SIZE);
  int64_t l1Align = getBytes(hacc::DeviceSpec::L1_ALIGN_SIZE);
  int64_t l1Used = 0;
  PipeCounts committed;
  bool known = collectInternalReservations(function, committed);
  SmallVector<RootResources, 8> roots;
  DenseMap<Value, size_t> rootIndex;
  function.walk([&](memref::AllocOp alloc) {
    auto space = getOptionalHIVMAddressSpace(alloc.getType());
    if (!space || *space == AddressSpace::GM)
      return;
    RootResources resources;
    resources.root = alloc;
    const LoadInfo *load = loadsByRoot.lookup(alloc);
    resources.slots = load ? load->originalSlots : getSlotCount(alloc);
    SmallVector<Access> accesses;
    if (resources.slots > std::numeric_limits<int32_t>::max() ||
        (!load && failed(collectAccesses(alloc, accesses)))) {
      known = false;
      return;
    }
    resources.directions = getDirections(load ? load->accesses : accesses);
    if (*space == AddressSpace::L1) {
      auto size = getSlotBytes(alloc, l1Align);
      if (failed(size)) {
        known = false;
        return;
      }
      resources.l1SlotBytes = *size;
      int64_t bytes;
      if (llvm::MulOverflow(resources.slots, *size, bytes) ||
          llvm::AddOverflow(l1Used, bytes, l1Used)) {
        known = false;
        return;
      }
    }
    rootIndex[alloc] = roots.size();
    roots.push_back(std::move(resources));
  });
  if (!known || capacity <= 0 || l1Align <= 0)
    return success(); // Independent buffers retain their original capacities.

  // Already committed pools occupy one address table and one group per bank.
  for (const Pool &pool : plan.pools) {
    llvm::SetVector<PipePair> directions;
    for (LoadInfo *load : pool.members) {
      RootResources &root = roots[rootIndex.lookup(load->root)];
      l1Used -= root.l1SlotBytes * root.slots;
      directions.insert(root.directions.begin(), root.directions.end());
      root.slots = 0;
    }
    l1Used +=
        roots[rootIndex[pool.members.front()->root]].l1SlotBytes * pool.size;
    for (PipePair pair : directions)
      committed[pair] += pool.size;
  }
  SmallVector<Pool, 4> candidates;
  for (LoadInfo &load : plan.loads) {
    if (load.scope->hasAttr(MemPoolIdAttr::name))
      continue;
    auto it = llvm::find_if(candidates, [&](const Pool &pool) {
      return compatible(*pool.members.front(), load);
    });
    if (it == candidates.end()) {
      candidates.emplace_back();
      it = std::prev(candidates.end());
    }
    it->members.push_back(&load);
  }
  for (Pool &pool : candidates) {
    if (pool.members.size() < 2)
      continue;
    DenseSet<Value> memberRoots;
    llvm::SetVector<PipePair> directions;
    int64_t originalSlots = 0;
    for (LoadInfo *load : pool.members) {
      memberRoots.insert(load->root);
      originalSlots += load->originalSlots;
      const auto &pairs = roots[rootIndex.lookup(load->root)].directions;
      directions.insert(pairs.begin(), pairs.end());
    }
    if (directions.empty())
      continue;
    PipeCounts used = committed;
    for (const RootResources &root : roots)
      if (!memberRoots.contains(root.root))
        for (PipePair pair : root.directions)
          used[pair] += root.slots;
    int64_t syncLimit = std::numeric_limits<int32_t>::max();
    for (PipePair pair : directions) {
      int64_t available = syncsolver::getHWAvailableEventIdNum(
          syncsolver::SyncMode::INTRA_CORE_SYNC, pair.first, pair.second);
      syncLimit = std::min(syncLimit, available - used[pair]);
    }
    int64_t memoryLimit = originalSlots;
    int64_t slotBytes =
        roots[rootIndex[pool.members.front()->root]].l1SlotBytes;
    if (slotBytes)
      memoryLimit += llvm::divideFloorSigned(capacity - l1Used, slotBytes);
    pool.size = std::min(memoryLimit, syncLimit);
    if (pool.size < requiredSlots(pool))
      continue;
    if (nextId >= std::numeric_limits<int32_t>::max())
      return function.emitError("buffer pool id exceeds the i32 range");
    pool.id = nextId++;
    // Commit budgets only after the complete candidate passes its checks.
    l1Used += (pool.size - originalSlots) * slotBytes;
    for (LoadInfo *load : pool.members)
      roots[rootIndex[load->root]].slots = 0;
    for (PipePair pair : directions)
      committed[pair] += pool.size;
    plan.pools.push_back(std::move(pool));
  }
  return success();
}

static annotation::MarkOp getOrCreateMark(Value root, OpBuilder &builder) {
  if (auto mark = utils::getAnnotateOpWithAttr(root, MultiBufferAttr::name))
    return cast<annotation::MarkOp>(*mark);
  for (Operation *user : root.getUsers())
    if (auto mark = dyn_cast<annotation::MarkOp>(user))
      return mark;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(root.getDefiningOp());
  return builder.create<annotation::MarkOp>(root.getLoc(), root);
}

static void applyPlan(FunctionPlan &plan) {
  OpBuilder builder(plan.function.getContext());
  DenseMap<Value, annotation::MarkOp> marks;
  for (LoadInfo &load : plan.loads) {
    auto mark = getOrCreateMark(load.root, builder);
    mark->setAttr(PreloadLocalBufferAttr::name, builder.getI32IntegerAttr(1));
    marks[load.root] = mark;
  }
  for (Pool &pool : plan.pools) {
    for (LoadInfo *load : pool.members) {
      // A repeated pass reuses committed capacity and preserves numbering.
      if (load->scope->hasAttr(MemPoolIdAttr::name))
        continue;
      auto id = builder.getI32IntegerAttr(pool.id);
      auto size = builder.getI32IntegerAttr(pool.size);
      auto mark = marks[load->root];
      mark->setAttr(MemPoolIdAttr::name, id);
      mark->setAttr(MemPoolSizeAttr::name, size);
      mark->setAttr(MultiBufferAttr::name, size);
      mark->setAttr(HIVMMemoryUniqueAttr::name, builder.getUnitAttr());
      load->scope->setAttr(MemPoolIdAttr::name, id);
    }
  }
}

struct MarkBufferPoolPass : impl::MarkBufferPoolBase<MarkBufferPoolPass> {
  void runOnOperation() override {
    SmallVector<std::unique_ptr<FunctionPlan>, 4> plans;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal() || isVF(function))
        continue;
      auto plan = std::make_unique<FunctionPlan>();
      plan->function = function;
      if (failed(analyzeFunction(*plan)))
        return signalPassFailure();
      plans.push_back(std::move(plan));
    }
    for (auto &plan : plans)
      applyPlan(*plan);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkBufferPoolPass() {
  return std::make_unique<MarkBufferPoolPass>();
}

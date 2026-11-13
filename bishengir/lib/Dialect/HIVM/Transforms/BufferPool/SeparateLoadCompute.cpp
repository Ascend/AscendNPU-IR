//===- SeparateLoadCompute.cpp - Separate GM loads from computation -------===//
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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/BufferPoolUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TightlyCoupledBufferUtils.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir {
#define GEN_PASS_DEF_SEPARATELOADCOMPUTE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::buffer_pool;

namespace {
struct LoadGroup {
  memref::AllocOp alloc;
  Operation *anchor = nullptr;
  Operation *loop = nullptr;
  Block *schedulingBlock = nullptr;
  TCoreTypeAttr core;
  SmallVector<Operation *> transfers;
  SmallVector<Operation *> writes;
  llvm::SetVector<Operation *> inputs;
  std::optional<TaskInfo> task;
  std::optional<int64_t> stage;
};

static bool isTransfer(Operation *op) {
  return isa<hivm::LoadOp, hivm::ND2NZOp>(op);
}

static bool isGMSource(Value source, func::FuncOp function) {
  if (auto space = getOptionalHIVMAddressSpace(source.getType()))
    return *space == AddressSpace::GM;
  Value root = getBufferRoot(source);
  if (auto space = getOptionalHIVMAddressSpace(root.getType()))
    return *space == AddressSpace::GM;
  auto argument = dyn_cast<BlockArgument>(root);
  return argument && argument.getOwner() == &function.getBody().front() &&
         function->hasAttr("hacc.entry");
}

/// Preparations must only initialize this destination. Do not move sync,
/// scalar state reads, or another buffer's initialization.
static bool isInitialization(Operation *op, Value root) {
  bool writes = false;
  bool legal =
      !op->walk([&](Operation *nested) {
           if (nested->hasTrait<OpTrait::IsTerminator>())
             return WalkResult::advance();
           if (auto ifOp = dyn_cast<scf::IfOp>(nested))
             return ifOp.getNumResults() == 0 ? WalkResult::advance()
                                              : WalkResult::interrupt();
           if (isa<hivm::VBrcOp>(nested)) {
             auto dps = cast<DestinationStyleOpInterface>(nested);
             if (dps.getNumDpsInits() != 1 ||
                 getBufferRoot(dps.getDpsInitOperand(0)->get()) != root ||
                 llvm::any_of(dps.getDpsInputs(), [](Value value) {
                   return isa<ShapedType>(value.getType());
                 }))
               return WalkResult::interrupt();
             writes = true;
             return WalkResult::advance();
           }
           return isMemoryEffectFree(nested) && nested->getNumRegions() == 0
                      ? WalkResult::advance()
                      : WalkResult::interrupt();
         }).wasInterrupted();
  return legal && writes;
}

static bool writesRoot(Operation *op, Value root) {
  return op
      ->walk([&](Operation *nested) {
        if (auto dps = dyn_cast<DestinationStyleOpInterface>(nested))
          for (Value destination : dps.getDpsInits())
            if (getBufferRoot(destination) == root)
              return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static bool collectInput(Value value, LoadGroup &group,
                         DominanceInfo &dominance) {
  if (value == group.alloc.getResult())
    return true;
  // Reuse loop-invariant address calculations at their original position.
  if (cast<LoopLikeOpInterface>(group.loop).isDefinedOutsideOfLoop(value))
    return dominance.properlyDominates(value, group.anchor);
  Operation *def = value.getDefiningOp();
  if (def && group.inputs.contains(def))
    return true;
  // Retain loop-argument dependencies even when the address expression
  // originally precedes the compute scope. Enable can then remap its progress.
  bool canClone = def && def->getNumRegions() == 0 && isMemoryEffectFree(def) &&
                  !isa<bufferization::ToTensorOp>(def);
  if (!canClone)
    return (!def || !group.loop->isAncestor(def)) &&
           dominance.properlyDominates(value, group.anchor);
  for (Value operand : def->getOperands())
    if (!collectInput(operand, group, dominance))
      return false;
  group.inputs.insert(def);
  return true;
}

static bool collectInputs(Operation *op, LoadGroup &group,
                          DominanceInfo &dominance) {
  for (Value value : op->getOperands())
    if (!collectInput(value, group, dominance))
      return false;
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(op->getRegions(), captures);
  for (Value value : captures)
    if (!collectInput(value, group, dominance))
      return false;
  return true;
}

/// Check the complete destination access chain before moving any operations.
/// Padding belongs to the load group; all other accesses must be readers in
/// the same scheduling loop. Unknown uses keep the original guarded load.
static bool hasSupportedReaders(const LoadGroup &group,
                                SmallVectorImpl<Operation *> &readers) {
  SmallVector<Access> accesses;
  if (failed(collectAccesses(group.alloc->getResult(0), accesses)))
    return false;
  for (const Access &access : accesses) {
    if (llvm::any_of(group.writes, [&](Operation *write) {
          return write == access.op || write->isAncestor(access.op);
        }))
      continue;
    if (access.write || !group.loop->isAncestor(access.op))
      return false;
    // Intermediate copies and cross-core readers require the TCB path.
    if (isa<hivm::CopyOp, hivm::ND2NZOp>(access.op))
      return false;
    if (hivm::detail::queryCoreTypeHelper(access.op) == TCoreType::VECTOR)
      return false;
    readers.push_back(access.op);
  }
  return !readers.empty();
}

static bool analyzeGroup(LoadGroup &group, DominanceInfo &dominance) {
  Operation *first = group.transfers.front();
  Block *block = first->getBlock();
  if (llvm::any_of(group.transfers,
                   [&](Operation *op) { return op->getBlock() != block; }))
    return false;

  bool hasComputeScope = false;
  for (Operation *parent = first->getParentOp();
       parent != group.schedulingBlock->getParentOp();
       parent = parent->getParentOp()) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(parent)) {
      if (scopeOp->hasAttr(LoadPreloadAttr::name))
        return false;
      hasComputeScope = true;
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      if (!ifOp->hasAttr("ssbuffer.if") ||
          !ifOp.getThenRegion().isAncestor(first->getParentRegion()))
        return false;
      if (!ifOp.getElseRegion().empty() &&
          ifOp.getElseRegion()
              .walk([](Operation *op) {
                return isTransfer(op) ? WalkResult::interrupt()
                                      : WalkResult::advance();
              })
              .wasInterrupted())
        return false;
      hasComputeScope = true;
    } else {
      return false;
    }
    if (!group.core)
      if (auto core = parent->getAttrOfType<TCoreTypeAttr>(
              hivm::kPipelinedLoopCoreTypeAttrName))
        if (core.getTcoretype() == TCoreType::CUBE ||
            core.getTcoretype() == TCoreType::VECTOR)
          group.core = core;
    group.anchor = parent;
  }
  if (!hasComputeScope || !group.anchor)
    return false;
  if (!group.core) {
    auto core = hivm::detail::queryCoreTypeHelper(first);
    if (!core || (*core != TCoreType::CUBE && *core != TCoreType::VECTOR))
      return false;
    group.core = TCoreTypeAttr::get(first->getContext(), *core);
  }

  // Vector consumers may become VF calls after Separate. Keep their loads
  // attached to the original computation until call-based consumers are
  // supported.
  if (group.core.getTcoretype() != TCoreType::CUBE)
    return false;

  // Prefetching must not read data produced by a later task in this loop.
  for (Operation *transfer : group.transfers) {
    Value source = cast<DestinationStyleOpInterface>(transfer)
                       .getDpsInputOperand(0)
                       ->get();
    SmallVector<Access> accesses;
    if (failed(collectAccesses(getBufferRoot(source), accesses)) ||
        llvm::any_of(accesses, [&](const Access &access) {
          return access.write && group.loop->isAncestor(access.op);
        }))
      return false;
  }
  // Lift static allocations to the scheduling block's prefix, never across a
  // loop. An allocation with operands may depend on earlier body operations.
  if (group.loop->isAncestor(group.alloc) && group.alloc->getNumOperands() != 0)
    return false;
  if (!dominance.properlyDominates(group.alloc.getOperation(), group.anchor)) {
    if (!group.anchor->isAncestor(group.alloc) ||
        group.alloc->getParentOfType<LoopLikeOpInterface>().getOperation() !=
            group.loop)
      return false;
  }

  Operation *last = group.transfers.back();
  for (Operation &op : *block) {
    if (llvm::is_contained(group.transfers, &op))
      group.writes.push_back(&op);
    else if (writesRoot(&op, group.alloc.getResult())) {
      if (!isInitialization(&op, group.alloc.getResult()))
        return false;
      group.writes.push_back(&op);
    }
    if (&op == last)
      break;
  }
  SmallVector<Operation *> readers;
  if (!hasSupportedReaders(group, readers))
    return false;
  auto task = analyzeTask(group.schedulingBlock, group.writes, readers);
  if (failed(task))
    return false;
  auto loop = cast<LoopLikeOpInterface>(group.loop);
  if (!loop.isDefinedOutsideOfLoop(task->tripStart) ||
      !loop.isDefinedOutsideOfLoop(task->tripBound))
    return false;
  for (Operation *reader : readers) {
    if (task->guard) {
      if (!task->guard.getThenRegion().isAncestor(reader->getParentRegion()))
        return false;
    } else {
      for (Operation *op = reader; op != group.loop; op = op->getParentOp())
        if (isa<scf::IfOp>(op))
          return false;
    }
  }

  group.task = *task;
  for (Operation *reader : readers)
    for (Operation *op = reader; op != group.loop; op = op->getParentOp())
      if (auto stage = op->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)) {
        group.stage =
            std::max(group.stage.value_or(stage.getInt()), stage.getInt());
        break;
      }
  return true;
}

static bool inferSSBufferStages(MutableArrayRef<LoadGroup> groups) {
  SmallVector<std::optional<TaskInfo>> tasks(groups.size());
  for (auto [i, group] : llvm::enumerate(groups)) {
    if (group.task && !group.stage && group.task->guard &&
        group.task->guard->hasAttr("ssbuffer.if"))
      tasks[i] = group.task;
  }
  struct Constraint {
    size_t before;
    size_t after;
    int64_t distance;
  };
  SmallVector<Constraint> constraints;
  for (auto [i, task] : llvm::enumerate(tasks)) {
    if (!task)
      continue;
    SmallVector<Value> terms;
    collectConjuncts(task->guard.getCondition(), terms);
    for (Value term : terms) {
      auto cmp = term.getDefiningOp<arith::CmpIOp>();
      auto either = term.getDefiningOp<arith::OrIOp>();
      SmallVector<arith::CmpIOp, 2> alternatives;
      if (cmp)
        alternatives.push_back(cmp);
      else if (either) {
        auto a = either.getLhs().getDefiningOp<arith::CmpIOp>();
        auto b = either.getRhs().getDefiningOp<arith::CmpIOp>();
        if (a && b)
          alternatives = {a, b};
      }
      DenseSet<Value> visited;
      llvm::SetVector<Value> conditionCounters;
      collectCounters(term, groups[i].schedulingBlock, visited,
                      conditionCounters);
      for (auto [j, other] : llvm::enumerate(tasks)) {
        if (i == j || !other || task->counter == other->counter ||
            !conditionCounters.contains(other->counter))
          continue;
        std::optional<int64_t> threshold;
        bool valid = !alternatives.empty();
        for (auto comparison : alternatives) {
          if (comparison.getPredicate() != arith::CmpIPredicate::sge ||
              comparison.getLhs() != other->counter) {
            valid = false;
            break;
          }
          if (comparison.getRhs() == other->bound)
            continue; // The producer's completion also admits a short tail.
          auto number = getConstantInt(comparison.getRhs());
          if (!number)
            return false;
          threshold = threshold ? std::min(*threshold, *number) : *number;
        }
        auto start = getConstantInt(other->start);
        // Unrecognized inter-task conditions, completion-only or dynamic
        // startup cannot be encoded by a static stage difference. Leave the
        // original guarded loads intact instead of guessing equal stages.
        if (!valid || !threshold || !start)
          return false;
        int64_t offset;
        if (llvm::SubOverflow(*threshold, *start, offset))
          return false;
        int64_t distance = llvm::divideCeilSigned(offset, other->step);
        if (distance > 0)
          constraints.push_back({j, i, distance});
      }
    }
  }
  SmallVector<int64_t> delays(groups.size(), 0);
  for (size_t iteration = 0; iteration < groups.size(); ++iteration) {
    bool changed = false;
    for (Constraint edge : constraints) {
      int64_t next;
      if (llvm::AddOverflow(delays[edge.before], edge.distance, next) ||
          next >= std::numeric_limits<int32_t>::max())
        return false;
      if (next <= delays[edge.after])
        continue;
      if (iteration + 1 == groups.size())
        return false; // Cyclic startup constraints have no static schedule.
      delays[edge.after] = next;
      changed = true;
    }
    if (!changed)
      break;
  }
  int64_t maxDelay = *llvm::max_element(delays);
  for (auto [i, group] : llvm::enumerate(groups))
    if (tasks[i])
      group.stage = maxDelay - delays[i] + 1;
  return true;
}

/// Preserve the existing compute-stage differences in a positive i32 range.
/// SSBuffer stages are inferred while the original guards are still present.
static bool inferLoadStages(MutableArrayRef<LoadGroup> groups) {
  if (groups.empty())
    return true;
  int64_t minStage = std::numeric_limits<int64_t>::max();
  int64_t maxStage = std::numeric_limits<int64_t>::min();
  for (const LoadGroup &group : groups) {
    if (!group.stage)
      continue;
    minStage = std::min(minStage, *group.stage);
    maxStage = std::max(maxStage, *group.stage);
  }
  if (minStage <= maxStage) {
    int64_t span;
    if (llvm::SubOverflow(maxStage, minStage, span) ||
        span >= std::numeric_limits<int32_t>::max())
      return false;
    for (LoadGroup &group : groups)
      if (group.stage)
        group.stage = *group.stage - minStage + 1;
  }
  if (!inferSSBufferStages(groups))
    return false;
  // A single unguarded stream has no relative scheduling constraint.
  if (groups.size() == 1 && !groups.front().stage && groups.front().task &&
      !groups.front().task->guard)
    groups.front().stage = 1;
  return true;
}

static bool canSeparateGroup(LoadGroup &group, DominanceInfo &dominance) {
  Operation *first = group.transfers.front();

  // Do not cross waits, unknown effects, or earlier computation. Check every
  // enclosing scope/then block, not only the block containing the DMA.
  for (Operation *current = first; current != group.anchor;
       current = current->getParentOp()) {
    for (Operation &preceding : *current->getBlock()) {
      if (&preceding == current)
        break;
      if (llvm::is_contained(group.writes, &preceding) ||
          isa<memref::AllocOp, annotation::MarkOp, bufferization::ToTensorOp>(
              &preceding) ||
          isMemoryEffectFree(&preceding))
        continue;
      return false;
    }
  }
  for (Operation *write : group.writes)
    if (!collectInputs(write, group, dominance))
      return false;
  return true;
}

static void separateGroup(LoadGroup &group, OpBuilder &builder) {
  // Keep all loads, including their scalar address computations, ahead of
  // computation. Advance past definitions already at the insertion point.
  auto moveToPrefix = [&](Operation *op) {
    if (op->getBlock() == builder.getInsertionBlock() &&
        op->getIterator() == builder.getInsertionPoint())
      builder.setInsertionPointAfter(op);
    else
      op->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
  };
  if (group.loop->isAncestor(group.alloc)) {
    moveToPrefix(group.alloc);
    SmallVector<annotation::MarkOp> marks;
    for (Operation *user : group.alloc->getUsers())
      if (auto mark = dyn_cast<annotation::MarkOp>(user))
        if (group.schedulingBlock->findAncestorOpInBlock(*mark))
          marks.push_back(mark);
    for (annotation::MarkOp mark : marks)
      moveToPrefix(mark);
  }

  auto loadScope =
      builder.create<scope::ScopeOp>(group.anchor->getLoc(), TypeRange{});
  loadScope.setNoInline(true);
  loadScope->setAttr(LoadPreloadAttr::name,
                     builder.getI32IntegerAttr(*group.stage));
  loadScope->setAttr(hivm::kPipelinedLoopCoreTypeAttrName, group.core);
  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.createBlock(&loadScope.getRegion());
  IRMapping mapping;
  for (Operation *input : group.inputs)
    builder.clone(*input, mapping);
  for (Operation *write : group.writes)
    builder.clone(*write, mapping);
  builder.create<scope::ReturnOp>(loadScope.getLoc());
  for (Operation *write : llvm::reverse(group.writes))
    write->erase();
}

static void separateFunction(func::FuncOp function) {
  if (function.isExternal() || isVF(function))
    return;
  llvm::SetVector<Block *> blocks;
  function.walk([&](Operation *op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (isSchedulingLoop(op))
        blocks.insert(forOp.getBody());
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (ifOp->hasAttr("ssbuffer.if"))
        blocks.insert(ifOp->getBlock());
    }
  });
  for (Block *block : blocks) {
    auto loop = dyn_cast<LoopLikeOpInterface>(block->getParentOp());
    if (!loop || !isa<scf::ForOp, scf::WhileOp>(loop.getOperation()))
      continue;
    SmallVector<LoadGroup, 4> groups;
    DenseMap<Value, unsigned> indices;
    block->walk([&](Operation *op) {
      // A scheduling block does not include transfers in nested loops.
      if (!isTransfer(op) || op->getNumResults() != 0 ||
          op->getParentOfType<LoopLikeOpInterface>() != loop)
        return;
      auto dps = cast<DestinationStyleOpInterface>(op);
      if (!isGMSource(dps.getDpsInputOperand(0)->get(), function))
        return;
      Value root = getBufferRoot(dps.getDpsInitOperand(0)->get());
      auto alloc = root.getDefiningOp<memref::AllocOp>();
      if (!alloc || !alloc.getType().hasStaticShape() ||
          llvm::is_contained(alloc.getType().getShape(), int64_t(0)) ||
          isTightlyCoupledValue(root))
        return;
      auto space = getOptionalHIVMAddressSpace(alloc.getType());
      if (space && *space != AddressSpace::L1 && *space != AddressSpace::UB)
        return;
      auto [it, inserted] = indices.try_emplace(root, groups.size());
      if (inserted) {
        groups.emplace_back();
        groups.back().alloc = alloc;
        groups.back().loop = loop.getOperation();
        groups.back().schedulingBlock = block;
      }
      groups[it->second].transfers.push_back(op);
    });
    // Recover all stages before moving loads out of their original guards.
    DominanceInfo dominance(function);
    llvm::erase_if(groups, [&](LoadGroup &group) {
      return !analyzeGroup(group, dominance);
    });
    if (!inferLoadStages(groups))
      continue;
    // Keep source order within the load prefix. Stage inference above still
    // sees the original computation guards and counter dependencies.
    OpBuilder builder(block, block->begin());
    for (LoadGroup &group : groups) {
      DominanceInfo dominance(function);
      if (group.stage && canSeparateGroup(group, dominance))
        separateGroup(group, builder);
    }
  }
}

struct SeparateLoadComputePass
    : public impl::SeparateLoadComputeBase<SeparateLoadComputePass> {
  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>())
      separateFunction(function);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createSeparateLoadComputePass() {
  return std::make_unique<SeparateLoadComputePass>();
}

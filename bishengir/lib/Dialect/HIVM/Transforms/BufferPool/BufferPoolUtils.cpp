//===- BufferPoolUtils.cpp - Independent load scheduling utilities --------===//
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

#include "bishengir/Dialect/HIVM/Transforms/BufferPoolUtils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::buffer_pool;

Value hivm::buffer_pool::getBufferRoot(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(def))
      value = view.getViewSource();
    else if (auto cast = dyn_cast<memref::CastOp>(def))
      value = cast.getSource();
    else if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def))
      value = cast.getSource();
    else if (auto tensor = dyn_cast<bufferization::ToTensorOp>(def))
      value = tensor.getMemref();
    else
      break;
  }
  return value;
}

// Follow MultiBufferLoopAdapter's anchor-based region selection. Pool callers
// always have an anchor, so no builder insertion point or default is needed.
Block *hivm::buffer_pool::getLoopBodyBlock(LoopLikeOpInterface loop,
                                           Operation *anchor) {
  if (!loop)
    return nullptr;
  Operation *op = loop.getOperation();
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return forOp.getBody();
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    Region *region = anchor->getParentRegion();
    while (region && region->getParentOp() != whileOp.getOperation())
      region = region->getParentOp()->getParentRegion();
    if (region == &whileOp.getBefore())
      return &whileOp.getBefore().front();
    if (region == &whileOp.getAfter())
      return &whileOp.getAfter().front();
  }
  return nullptr;
}

bool hivm::buffer_pool::isSchedulingLoop(Operation *op) {
  if (!isa<scf::ForOp, scf::WhileOp>(op) ||
      !op->hasAttr(hivm::CVPipelinedLoopAttr::name))
    return false;
  return !op->walk([&](Operation *nested) {
              return nested != op && isa<LoopLikeOpInterface>(nested)
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            }).wasInterrupted();
}

namespace {
LogicalResult collect(Value value, SmallVectorImpl<Access> &accesses,
                      DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return success();
  for (OpOperand &use : value.getUses()) {
    Operation *op = use.getOwner();
    if (isa<annotation::MarkOp, memref::DimOp, memref::DeallocOp>(op))
      continue;
    if (isa<ViewLikeOpInterface, memref::CastOp, memref::MemorySpaceCastOp,
            bufferization::ToTensorOp, arith::SelectOp>(op)) {
      for (Value result : op->getResults())
        if (failed(collect(result, accesses, visited)))
          return failure();
      continue;
    }
    if (isa<scope::ReturnOp>(op) ||
        (isa<scf::YieldOp>(op) && isa<scf::IfOp>(op->getParentOp()))) {
      if (failed(collect(op->getParentOp()->getResult(use.getOperandNumber()),
                         accesses, visited)))
        return failure();
      continue;
    }
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    auto pipeOp = dyn_cast<OpPipeInterface>(op);
    if (!dps || !pipeOp)
      return failure();
    bool write = dps.isDpsInit(&use);
    PIPE pipe = pipeOp.isSinglePipeOp()
                    ? pipeOp.getPipe()
                    : (write ? pipeOp.getOutPipe() : pipeOp.getInPipe());
    // GSS decomposes L1 matrix operands into virtual MTE1 readers.
    if (!write && isa<MmadL1Op, BatchMmadL1Op, MmadMxL1Op>(op) &&
        getOptionalHIVMAddressSpace(value.getType()) == AddressSpace::L1)
      pipe = PIPE::PIPE_MTE1;
    if (pipe == PIPE::PIPE_UNASSIGNED)
      return failure();
    accesses.push_back({op, pipe, write});
  }
  return success();
}
} // namespace

LogicalResult
mlir::hivm::buffer_pool::collectAccesses(Value root,
                                         SmallVectorImpl<Access> &accesses) {
  DenseSet<Value> visited;
  return collect(root, accesses, visited);
}

Operation *mlir::hivm::buffer_pool::findLoadTransfer(scope::ScopeOp scope) {
  Operation *transfer = nullptr;
  scope.walk([&](Operation *op) {
    if (!isa<LoadOp, ND2NZOp>(op))
      return WalkResult::advance();
    transfer = op;
    return WalkResult::interrupt();
  });
  return transfer;
}

FailureOr<LoadInfo> mlir::hivm::buffer_pool::analyzeLoad(scope::ScopeOp scope,
                                                         Operation *transfer,
                                                         bool planned) {
  LoadInfo info;
  info.scope = scope;
  auto fail = [&](StringRef message) -> FailureOr<LoadInfo> {
    scope.emitError(message);
    return failure();
  };
  // Separate identifies scheduling regions. Later stages consume its load
  // markers without requiring the original CV/SSBuffer attributes to survive.
  auto loop = scope->getParentOfType<LoopLikeOpInterface>();
  info.body = getLoopBodyBlock(loop, scope);
  if (!info.body)
    return fail("independent load requires a for or while scheduling loop");
  info.loop = loop.getOperation();
  auto core = scope->getAttrOfType<TCoreTypeAttr>("hivm.loop_core_type");
  info.core = core.getTcoretype();
  // Separate groups transfers by one static allocation. PlanMemory later
  // replaces that allocation with a pointer_cast.
  auto dps = cast<DestinationStyleOpInterface>(transfer);
  info.root = getBufferRoot(dps.getDpsInitOperand(0)->get());
  if (planned) {
    auto pointer = cast<PointerCastOp>(info.root.getDefiningOp());
    info.originalSlots = pointer.getAddrs().size();
  } else {
    auto space = getOptionalHIVMAddressSpace(info.root.getType());
    if (!space || (*space != AddressSpace::L1 && *space != AddressSpace::UB))
      return fail("independent load requires L1 or UB storage");
    if (utils::getAnnotateOpWithAttr(info.root,
                                     HIVMTightlyCoupledBufferAttr::name))
      return fail("independent load cannot use a tightly coupled buffer");
    if (auto mark =
            utils::getAnnotateOpWithAttr(info.root, MultiBufferAttr::name)) {
      auto count = (*mark)->getAttrOfType<IntegerAttr>(MultiBufferAttr::name);
      info.originalSlots = count.getInt();
    }
  }
  if (failed(collectAccesses(info.root, info.accesses)))
    return fail("cannot determine all independent-load buffer accesses");
  for (const Access &access : info.accesses) {
    if (scope->isAncestor(access.op))
      continue;
    if (access.write || !info.body->findAncestorOpInBlock(*access.op))
      return fail("independent-load buffer escapes its scheduling loop or has "
                  "an external writer");
    info.readers.insert(access.op);
  }
  if (info.readers.empty())
    return fail("independent load has no identifiable consumer");
  return info;
}

std::optional<int64_t> mlir::hivm::buffer_pool::getConstantInt(Value value) {
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return std::nullopt;
  return integer.getSExtValue();
}

void mlir::hivm::buffer_pool::collectCounters(
    Value value, Block *body, DenseSet<Value> &visited,
    llvm::SetVector<Value> &counters) {
  if (!visited.insert(value).second)
    return;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() == body && arg.getType().isIntOrIndex())
      counters.insert(arg);
    return;
  }
  Operation *op = value.getDefiningOp();
  if (!op || !isMemoryEffectFree(op) || op->getNumRegions())
    return;
  for (Value operand : op->getOperands())
    collectCounters(operand, body, visited, counters);
}

void mlir::hivm::buffer_pool::collectConjuncts(
    Value value, SmallVectorImpl<Value> &conjuncts) {
  if (auto andOp = value.getDefiningOp<arith::AndIOp>()) {
    collectConjuncts(andOp.getLhs(), conjuncts);
    collectConjuncts(andOp.getRhs(), conjuncts);
    return;
  }
  conjuncts.push_back(value);
}

namespace {
static Value unwrapScopeResult(Value value) {
  while (auto result = dyn_cast<OpResult>(value)) {
    auto scope = dyn_cast<scope::ScopeOp>(result.getOwner());
    if (!scope)
      break;
    value =
        scope.getBody()->getTerminator()->getOperand(result.getResultNumber());
  }
  return value;
}

static std::optional<int64_t> getIncrement(Value value, Value counter) {
  value = unwrapScopeResult(value);
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;
  if (add.getLhs() == counter)
    return getConstantInt(add.getRhs());
  if (add.getRhs() == counter)
    return getConstantInt(add.getLhs());
  return std::nullopt;
}

static Value findBound(Value condition, Value counter) {
  SmallVector<Value> conjuncts;
  collectConjuncts(condition, conjuncts);
  for (Value term : conjuncts) {
    auto cmp = term.getDefiningOp<arith::CmpIOp>();
    if (cmp && cmp.getLhs() == counter &&
        cmp.getPredicate() == arith::CmpIPredicate::slt)
      return cmp.getRhs();
  }
  return {};
}
struct PreloadRange {
  Value counter;
  Value lower;
  Value upper;
  scf::IfOp guard;
};

static std::optional<PreloadRange> getPreloadRange(Operation *reader,
                                                   Operation *loop) {
  for (Operation *op = reader->getParentOp(); op && op != loop;
       op = op->getParentOp()) {
    auto guard = dyn_cast<scf::IfOp>(op);
    if (!guard || !utils::getAnnotateOpWithAttr(guard.getCondition(),
                                                PreloadNumAttr::name))
      continue;
    SmallVector<Value> terms;
    collectConjuncts(guard.getCondition(), terms);
    PreloadRange range;
    range.guard = guard;
    for (Value term : terms) {
      auto cmp = term.getDefiningOp<arith::CmpIOp>();
      if (!cmp)
        continue;
      if (cmp.getPredicate() == arith::CmpIPredicate::slt) {
        range.counter = cmp.getLhs();
        range.upper = cmp.getRhs();
      }
    }
    for (Value term : terms) {
      auto cmp = term.getDefiningOp<arith::CmpIOp>();
      if (cmp && cmp.getPredicate() == arith::CmpIPredicate::sge &&
          cmp.getLhs() == range.counter)
        range.lower = cmp.getRhs();
    }
    if (range.counter && range.lower && range.upper)
      return range;
  }
  return std::nullopt;
}

static std::optional<int64_t> findUpdate(Value next, Value counter,
                                         SmallVectorImpl<scf::IfOp> &guards) {
  next = unwrapScopeResult(next);
  if (auto step = getIncrement(next, counter))
    return step;
  auto result = dyn_cast<OpResult>(next);
  if (!result)
    return std::nullopt;
  auto guard = dyn_cast<scf::IfOp>(result.getOwner());
  if (!guard || guard.getElseRegion().empty())
    return std::nullopt;
  unsigned n = result.getResultNumber();
  if (unwrapScopeResult(guard.elseBlock()->getTerminator()->getOperand(n)) !=
      counter)
    return std::nullopt;
  guards.push_back(guard);
  return findUpdate(guard.thenBlock()->getTerminator()->getOperand(n), counter,
                    guards);
}

static LogicalResult recoverTaskUpdate(Value next, TaskInfo &task,
                                       SmallVectorImpl<scf::IfOp> &guards) {
  auto step = findUpdate(next, task.counter, guards);
  if (!step || *step <= 0)
    return failure();
  task.step = *step;
  for (scf::IfOp guard : llvm::reverse(guards))
    if (Value bound = findBound(guard.getCondition(), task.counter)) {
      task.bound = bound;
      task.guard = guard;
      break;
    }
  return success();
}

// A while must keep running while this task has unconsumed inputs. An OR
// may admit other tasks; an unrelated early-exit condition cannot be ignored.
static bool keepsTaskRunning(Value condition, Value counter, Value bound) {
  if (auto either = condition.getDefiningOp<arith::OrIOp>())
    return keepsTaskRunning(either.getLhs(), counter, bound) ||
           keepsTaskRunning(either.getRhs(), counter, bound);
  auto cmp = condition.getDefiningOp<arith::CmpIOp>();
  return cmp && cmp.getPredicate() == arith::CmpIPredicate::slt &&
         cmp.getLhs() == counter && cmp.getRhs() == bound;
}

static FailureOr<TaskInfo> analyzeWhileTask(scf::WhileOp loop, Block *body,
                                            BlockArgument counter) {
  bool before = body == &loop.getBefore().front();
  BlockArgument beforeCounter = counter;
  if (!before) {
    beforeCounter = dyn_cast<BlockArgument>(
        loop.getConditionOp().getArgs()[counter.getArgNumber()]);
    if (!beforeCounter || beforeCounter.getOwner() != &loop.getBefore().front())
      return failure();
  }
  unsigned index = beforeCounter.getArgNumber();
  Value next = loop.getYieldedValues()[index];
  if (before) {
    auto forwarded = dyn_cast<BlockArgument>(next);
    if (!forwarded || forwarded.getOwner() != &loop.getAfter().front())
      return failure();
    next = loop.getConditionOp().getArgs()[forwarded.getArgNumber()];
  }
  TaskInfo task;
  task.counter = counter;
  task.start = loop.getInits()[index];
  SmallVector<scf::IfOp> guards;
  if (failed(recoverTaskUpdate(next, task, guards)))
    return failure();
  Value condition = loop.getConditionOp().getCondition();
  // Before also executes on the terminating visit, so its task needs a guard.
  if (!task.bound && !before && guards.empty())
    task.bound = findBound(condition, beforeCounter);
  if (!task.bound ||
      (!keepsTaskRunning(condition, beforeCounter, task.bound) &&
       !(before && keepsTaskRunning(condition, next, task.bound))))
    return failure();
  task.tripStart = task.start;
  task.tripBound = task.bound;
  task.tripStep = task.step;
  return task;
}

} // namespace

FailureOr<TaskInfo> mlir::hivm::buffer_pool::analyzeTask(const LoadInfo &load) {
  auto scope = load.scope;
  return analyzeTask(load.body, {scope.getOperation()},
                     load.readers.getArrayRef());
}

FailureOr<TaskInfo>
mlir::hivm::buffer_pool::analyzeTask(Block *body,
                                     ArrayRef<Operation *> loadOperations,
                                     ArrayRef<Operation *> readers) {
  llvm::SetVector<Value> counters;
  DenseSet<Value> visited;
  for (Operation *operation : loadOperations)
    operation->walk([&](Operation *op) {
      for (Value operand : op->getOperands())
        collectCounters(operand, body, visited, counters);
    });
  if (counters.size() > 1)
    return failure();
  if (auto loop = dyn_cast<scf::WhileOp>(body->getParentOp())) {
    if (!counters.empty())
      return analyzeWhileTask(loop, body,
                              cast<BlockArgument>(counters.front()));
    // A constant source still needs an unambiguous original task counter.
    std::optional<TaskInfo> task;
    for (BlockArgument counter : body->getArguments()) {
      if (!counter.getType().isIntOrIndex())
        continue;
      auto candidate = analyzeWhileTask(loop, body, counter);
      if (failed(candidate))
        continue;
      if (task)
        return failure();
      task = *candidate;
    }
    if (!task)
      return failure();
    return *task;
  }
  auto loop = dyn_cast<scf::ForOp>(body->getParentOp());
  if (!loop)
    return failure();
  auto loopStep = getConstantInt(loop.getStep());
  if (!loopStep || *loopStep <= 0)
    return failure();
  TaskInfo task;
  task.counter = cast<BlockArgument>(counters.empty() ? loop.getInductionVar()
                                                      : counters.front());
  task.tripStart = loop.getLowerBound();
  task.tripBound = loop.getUpperBound();
  task.tripStep = *loopStep;
  std::optional<PreloadRange> range;
  for (Operation *reader : readers)
    if (auto candidate = getPreloadRange(reader, loop)) {
      range = *candidate;
      break;
    }
  if (range) {
    task.tripStart = range->lower;
    task.tripBound = range->upper;
  }
  if (task.counter == loop.getInductionVar()) {
    task.start = task.tripStart;
    task.bound = task.tripBound;
    task.step = *loopStep;
    if (range) {
      task.guard = range->guard;
    }
    return task;
  }
  unsigned index = task.counter.getArgNumber() - 1;
  task.start = loop.getInitArgs()[index];
  Value next = loop.getBody()->getTerminator()->getOperand(index);
  SmallVector<scf::IfOp> guards;
  if (failed(recoverTaskUpdate(next, task, guards)))
    return failure();
  if (task.bound) {
    task.tripStart = task.start;
    task.tripBound = task.bound;
    task.tripStep = task.step;
    return task;
  }
  if (!guards.empty() && !range)
    return failure(); // A conditional task needs an identifiable range.
  if (range)
    task.guard = range->guard;
  return task;
}

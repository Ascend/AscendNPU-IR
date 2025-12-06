//===--------- SyncSolverCodeGen.cpp ---- Graph Sync Solver ---------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <deque>
#include <memory>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "hivm-graph-sync-solver-code-gen"

using namespace mlir;
using namespace hivm::syncsolver;

// Choose where to insert generated sync ops (handles function-scope, ghost
// blocks, op-based insertion).
void Solver::setProperInsertionPoint(IRRewriter &rewriter,
                                     OperationBase *opBase,
                                     bool insertAfterOp) {
  if (opBase->parentOp == funcIr.get()) {
    if (insertAfterOp) {
      auto returnOp = utils::getAssumedUniqueReturnOp(func);
      rewriter.setInsertionPoint(returnOp);
    } else {
      if (!resultFuncIrWasGenerated) {
        auto *scopeOp = dyn_cast<Scope>(opBase);
        assert(scopeOp != nullptr && scopeOp->body.front() != nullptr);
        rewriter.setInsertionPoint(scopeOp->body.front()->op);
      } else {
        rewriter.setInsertionPointToStart(
            &dyn_cast<func::FuncOp>(funcIr->op).getBody().front());
      }
    }
  } else if (auto *ghostOp = dyn_cast<Ghost>(opBase)) {
    assert(!insertAfterOp);
    rewriter.setInsertionPointToEnd(ghostOp->block);
  } else {
    assert(opBase->op != nullptr);
    if (insertAfterOp) {
      rewriter.setInsertionPointAfter(opBase->op);
    } else {
      rewriter.setInsertionPoint(opBase->op);
    }
  }
}

// Determine a proper Location for newly generated ops based on opBase context.
Location Solver::getProperLoc(OperationBase *opBase) {
  assert(opBase != nullptr);
  if (auto *ghostOp = dyn_cast<Ghost>(opBase)) {
    assert(ghostOp != nullptr && ghostOp->block != nullptr && ghostOp->block->getParentOp() != nullptr);
    return ghostOp->block->getParentOp()->getLoc();
  }
  if (opBase->op == nullptr && opBase->parentOp != nullptr) {
    return getProperLoc(opBase->parentOp);
  }
  assert(opBase->op != nullptr);
  return opBase->op->getLoc();
}

// Insert a PipeBarrierOp at the resolved insertion point and location.
void Solver::insertBarrierOp(IRRewriter &rewriter, OperationBase *opBase,
                             BarrierOp *barrierOp, bool insertAfterOp) {
  assert(opBase != nullptr && barrierOp != nullptr);
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  Location loc = getProperLoc(opBase);
  auto pipe = PipeAttr::get(func->getContext(), barrierOp->pipe);
  rewriter.create<hivm::PipeBarrierOp>(loc, pipe);
}

// Insert SetFlagOp(s) handling multi-buffer and conditional (first/last iter)
// wrapping.
void Solver::insertSetFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                             SetFlagOp *setFlagOp, bool insertAfterOp) {
  assert(opBase != nullptr && setFlagOp != nullptr);
  if (llvm::succeeded(handleMmadL1SyncOps(rewriter, opBase, setFlagOp))) {
    return;
  }
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  auto *ctx = func->getContext();
  Location loc = getProperLoc(opBase);
  if (setFlagOp->checkLastIter) {
    assert(setFlagOp->op != nullptr);
    auto forOp = setFlagOp->op->getParentOfType<scf::ForOp>();
    assert(forOp != nullptr);
    assert(setFlagOp->op->getParentOp() == forOp);
    Value cond = getIsLastIterationValue(forOp, loc, rewriter);
    auto ifOp = rewriter.create<scf::IfOp>(loc, cond);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }
  auto setPipe = PipeAttr::get(ctx, setFlagOp->pipeSrc);
  auto waitPipe = PipeAttr::get(ctx, setFlagOp->pipeDst);
  if (!setFlagOp->allAtOnce && setFlagOp->eventIds.size() > 1) {
    assert(setFlagOp->eventIds.size() == 2);
    assert(setFlagOp->multibufferLoopPar != nullptr);
    auto selectedBuffer = getMultiBufferSelectOp(rewriter, setFlagOp);
    rewriter.create<hivm::SetFlagOp>(loc, setPipe, waitPipe, EventAttr{},
                                     selectedBuffer);
  } else {
    for (auto eventId : setFlagOp->eventIds) {
      auto eventIdAttr = EventAttr::get(ctx, eventId);
      rewriter.create<hivm::SetFlagOp>(loc, setPipe, waitPipe, eventIdAttr,
                                       Value{});
    }
  }
}

// Insert WaitFlagOp(s) handling multi-buffer and conditional wrapping.
void Solver::insertWaitFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                              WaitFlagOp *waitFlagOp, bool insertAfterOp) {
  assert(opBase != nullptr && waitFlagOp != nullptr);
  if (llvm::succeeded(handleMmadL1SyncOps(rewriter, opBase, waitFlagOp))) {
    return;
  }
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  auto *ctx = func->getContext();
  Location loc = getProperLoc(opBase);
  if (waitFlagOp->checkFirstIter) {
    assert(waitFlagOp->op != nullptr);
    auto forOp = waitFlagOp->op->getParentOfType<scf::ForOp>();
    assert(forOp != nullptr);
    assert(waitFlagOp->op->getParentOp() == forOp);
    Value cond = getIsFirstIterationValue(forOp, loc, rewriter);
    auto ifOp = rewriter.create<scf::IfOp>(loc, cond);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }
  auto setPipe = PipeAttr::get(ctx, waitFlagOp->pipeSrc);
  auto waitPipe = PipeAttr::get(ctx, waitFlagOp->pipeDst);
  if (!waitFlagOp->allAtOnce && waitFlagOp->eventIds.size() > 1) {
    assert(waitFlagOp->eventIds.size() == 2);
    assert(waitFlagOp->multibufferLoopPar != nullptr);
    auto selectedBuffer = getMultiBufferSelectOp(rewriter, waitFlagOp);
    rewriter.create<hivm::WaitFlagOp>(loc, setPipe, waitPipe, EventAttr{},
                                      selectedBuffer);
  } else {
    for (auto eventId : waitFlagOp->eventIds) {
      auto eventIdAttr = EventAttr::get(ctx, eventId);
      rewriter.create<hivm::WaitFlagOp>(loc, setPipe, waitPipe, eventIdAttr,
                                        Value{});
    }
  }
}

// Build/select a runtime i64 value that picks which buffer/event to use for
// multi-buffer sync.
Value Solver::getMultiBufferSelectOp(IRRewriter &rewriter, SetWaitOp *syncOp) {
  assert(syncOp != nullptr);
  assert(syncOp->eventIds.size() == 2);
  assert(syncOp->multibufferLoopPar != nullptr);
  auto forOp = llvm::dyn_cast_if_present<scf::ForOp>(
      syncOp->multibufferLoopPar.getOperation());
  assert(forOp != nullptr);
  auto eventsPair = std::make_pair(syncOp->eventIds[0], syncOp->eventIds[1]);
  if (bufferSelectedMem[syncOp->multibufferLoopPar].contains(eventsPair)) {
    return bufferSelectedMem[syncOp->multibufferLoopPar][eventsPair];
  }
  Value counter;
  PatternRewriter::InsertionGuard guard(rewriter);
  if (nestedIndexModularMem.contains(syncOp->multibufferLoopPar)) {
    counter = nestedIndexModularMem[syncOp->multibufferLoopPar];
  } else {
    counter = createNestedIndexModular(rewriter, syncOp->multibufferLoopPar);
    nestedIndexModularMem[syncOp->multibufferLoopPar] = counter;
  }
  assert(counter.getDefiningOp() != nullptr);
  rewriter.setInsertionPointAfter(counter.getDefiningOp());
  Location loc = counter.getDefiningOp()->getLoc();
  Value firstID = rewriter.create<arith::ConstantIntOp>(
      loc, static_cast<uint64_t>(syncOp->eventIds[0]), rewriter.getI64Type());
  Value secondID = rewriter.create<arith::ConstantIntOp>(
      loc, static_cast<uint64_t>(syncOp->eventIds[1]), rewriter.getI64Type());
  Value bufferSelected = rewriter.create<arith::SelectOp>(
      loc, rewriter.getI64Type(), counter, firstID, secondID);
  bufferSelectedMem[syncOp->multibufferLoopPar][eventsPair] = bufferSelected;
  return bufferSelected;
}

// Helper wrappers for explicit multi-buffer insertion variants.
void Solver::insertMultiBufferSetFlagOp(IRRewriter &rewriter,
                                        OperationBase *opBase,
                                        SetFlagOp *setFlagOp,
                                        bool insertAfterOp) {
  assert(opBase != nullptr && setFlagOp != nullptr);
  assert(opBase->op != nullptr);
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  auto *ctx = func->getContext();
  auto setPipe = PipeAttr::get(ctx, setFlagOp->pipeSrc);
  auto waitPipe = PipeAttr::get(ctx, setFlagOp->pipeDst);
  Location loc = getProperLoc(opBase);
  assert(setFlagOp->eventIds.size() == 2);
  assert(setFlagOp->multibufferLoopPar != nullptr);
  auto selectedBuffer = getMultiBufferSelectOp(rewriter, setFlagOp);
  rewriter.create<hivm::SetFlagOp>(loc, setPipe, waitPipe, EventAttr{},
                                   selectedBuffer);
}

void Solver::insertMultiBufferWaitFlagOp(IRRewriter &rewriter,
                                         OperationBase *opBase,
                                         WaitFlagOp *waitFlagOp,
                                         bool insertAfterOp) {
  assert(opBase != nullptr && waitFlagOp != nullptr);
  assert(opBase->op != nullptr);
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  auto *ctx = func->getContext();
  auto setPipe = PipeAttr::get(ctx, waitFlagOp->pipeSrc);
  auto waitPipe = PipeAttr::get(ctx, waitFlagOp->pipeDst);
  Location loc = getProperLoc(opBase);
  assert(waitFlagOp->eventIds.size() == 2);
  assert(waitFlagOp->multibufferLoopPar != nullptr);
  auto selectedBuffer = getMultiBufferSelectOp(rewriter, waitFlagOp);
  rewriter.create<hivm::WaitFlagOp>(loc, setPipe, waitPipe, EventAttr{},
                                    selectedBuffer);
}

// Get an event id Value for a given SetWaitOp (creates constant or uses
// select).
Value Solver::getEventIdValue(IRRewriter &rewriter, SetWaitOp *setWaitOp,
                              Location loc) {
  assert(setWaitOp != nullptr);
  assert(!setWaitOp->eventIds.empty());
  if (setWaitOp->eventIds.size() > 1) {
    return getMultiBufferSelectOp(rewriter, setWaitOp);
  }
  rewriter.setInsertionPointToStart(&func.getBody().front());
  return rewriter.create<arith::ConstantIntOp>(
      loc, static_cast<uint64_t>(setWaitOp->eventIds[0]),
      rewriter.getI64Type());
}

// Attempt to attach sync args to MmadL1 ops by recognizing special load L0 / L1
// patterns.
llvm::LogicalResult Solver::handleMmadL1SyncOps(IRRewriter &rewriter,
                                                OperationBase *opBase,
                                                SyncOp *syncOp) {
  if (opBase->parentOp == nullptr || opBase->parentOp->parentOp == nullptr) {
    return llvm::failure();
  }
  hivm::MmadL1Op mmadl1Op;
  if (auto *mmadL1Loop = dyn_cast<MmadL1LoopOp>(opBase->parentOp->parentOp)) {
    mmadl1Op = llvm::dyn_cast<hivm::MmadL1Op>(mmadL1Loop->op);
    assert(mmadl1Op != nullptr);
  }
  if (mmadl1Op == nullptr) {
    return llvm::failure();
  }
  assert(isa<LoadL0AOp>(opBase) || isa<LoadL0BOp>(opBase));
  assert(isa<SetFlagOp>(syncOp) || isa<WaitFlagOp>(syncOp));
  if (auto *setFlagOp = dyn_cast<SetFlagOp>(syncOp)) {
    if (isa<LoadL0AOp>(opBase)) {
      mmadl1SyncArgsMap[mmadl1Op].L1AWaitL0Event =
          getEventIdValue(rewriter, setFlagOp, mmadl1Op->getLoc());
    } else if (isa<LoadL0BOp>(opBase)) {
      mmadl1SyncArgsMap[mmadl1Op].L1BWaitL0Event =
          getEventIdValue(rewriter, setFlagOp, mmadl1Op->getLoc());
    }
  } else if (auto *waitFlagOp = dyn_cast<WaitFlagOp>(syncOp)) {
    if (isa<LoadL0AOp>(opBase)) {
      mmadl1SyncArgsMap[mmadl1Op].L0WaitL1AEvent =
          getEventIdValue(rewriter, waitFlagOp, mmadl1Op->getLoc());
    } else if (isa<LoadL0BOp>(opBase)) {
      mmadl1SyncArgsMap[mmadl1Op].L0WaitL1BEvent =
          getEventIdValue(rewriter, waitFlagOp, mmadl1Op->getLoc());
    }
  }
  return llvm::success();
}

Value Solver::getLoopDBCond(IRRewriter &rewriter, Operation *op) {
  auto parentLoop = op->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return nullptr;
  }
  if (loopDBCondMap.contains(parentLoop)) {
    return loopDBCondMap[parentLoop];
  }
  return loopDBCondMap[parentLoop] = createNestedIndexForOp(rewriter, op);
}

// Create and propagate sync args into MmadL1 op arguments.
void Solver::insertMmadL1SyncArgs(IRRewriter &rewriter) {
  for (auto &[mmadL1Op, syncArgs] : mmadl1SyncArgsMap) {
    rewriter.setInsertionPoint(mmadL1Op);
    auto defaultValue = rewriter.create<arith::ConstantIntOp>(
        mmadL1Op->getLoc(), -1, rewriter.getI64Type());
    syncArgs.KLoopDBCond = getLoopDBCond(rewriter, mmadL1Op.getOperation());
    SmallVector<Value> newArgs;
    newArgs.push_back(syncArgs.L0WaitL1AEvent);
    newArgs.push_back(syncArgs.L0WaitL1BEvent);
    newArgs.push_back(syncArgs.L1AWaitL0Event);
    newArgs.push_back(syncArgs.L1BWaitL0Event);
    newArgs.push_back(syncArgs.KLoopDBCond);
    newArgs.push_back(syncArgs.BackPipeMPipeMTE1Event0);
    newArgs.push_back(syncArgs.BackPipeMPipeMTE1Event1);
    for (auto &val : newArgs) {
      if (!val || val == Value{}) {
        val = defaultValue;
      }
    }
    mmadL1Op.getSyncRelatedArgsMutable().assign(newArgs);
  }
}

// Unit-flag helpers: compute final mode and create runtime conditions if
// needed.
hivm::UNIT_FLAG Solver::getUnitFlagMode(RWOperation *rwOp) {
  static DenseMap<std::pair<UNIT_FLAG, UNIT_FLAG>, UNIT_FLAG> possibleStates = {
      {std::make_pair(UNIT_FLAG::DISABLED, UNIT_FLAG::DISABLED),
       UNIT_FLAG::DISABLED},
      {std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE,
                      UNIT_FLAG::ENABLED_WITH_UPDATE),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE, UNIT_FLAG::DISABLED),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::DISABLED, UNIT_FLAG::ENABLED_WITH_UPDATE),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::ENABLED_WITH_UPDATE,
                      UNIT_FLAG::ENABLED_ONLY_FIRST_ITER),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::DISABLED, UNIT_FLAG::ENABLED_ONLY_FIRST_ITER),
       UNIT_FLAG::ENABLED_ONLY_FIRST_ITER},
      {std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER,
                      UNIT_FLAG::ENABLED_WITH_UPDATE),
       UNIT_FLAG::ENABLED_WITH_UPDATE},
      {std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER, UNIT_FLAG::DISABLED),
       UNIT_FLAG::ENABLED_ONLY_LAST_ITER},
      {std::make_pair(UNIT_FLAG::ENABLED_ONLY_LAST_ITER,
                      UNIT_FLAG::ENABLED_ONLY_FIRST_ITER),
       UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS},
  };
  auto it = possibleStates.find(
      std::make_pair(rwOp->unitFlagModeAsSet, rwOp->unitFlagModeAsWait));
  if (it == possibleStates.end()) {
    llvm_unreachable("unit-flag state not handled");
  }
  return it->second;
}

Value Solver::getIsNotDeadLoopValue(scf::ForOp forOp, Location loc,
                                    IRRewriter &rewriter) {
  Value upperBound = forOp.getUpperBound();
  Value lowerBound = forOp.getLowerBound();
  return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                        lowerBound, upperBound);
}

std::optional<mlir::Value> Solver::getUnitFlagCond(IRRewriter &rewriter,
                                                   RWOperation *rwOp) {
  assert(rwOp != nullptr && rwOp->op != nullptr);
  OpBuilder::InsertionGuard guard(rewriter);
  auto loc = rwOp->op->getLoc();
  SmallVector<Value> conditions;
  if (rwOp->linkedUnitFlagOpAsWait != nullptr &&
      (rwOp->linkedUnitFlagOpAsWait->unitFlagModeAsSet ==
           UNIT_FLAG::ENABLED_ONLY_LAST_ITER ||
       rwOp->linkedUnitFlagOpAsWait->unitFlagModeAsSet ==
           UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS)) {
    if (auto forOp = dyn_cast<scf::ForOp>(
            rwOp->linkedUnitFlagOpAsWait->op->getParentOp())) {
      rewriter.setInsertionPoint(forOp);
      Value cond = getIsNotDeadLoopValue(forOp, loc, rewriter);
      conditions.push_back(cond);
    }
  }
  if (rwOp->linkedUnitFlagOpAsSet != nullptr &&
      (rwOp->linkedUnitFlagOpAsSet->unitFlagModeAsWait ==
           UNIT_FLAG::ENABLED_ONLY_FIRST_ITER ||
       rwOp->linkedUnitFlagOpAsSet->unitFlagModeAsWait ==
           UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS)) {
    if (auto forOp = dyn_cast<scf::ForOp>(
            rwOp->linkedUnitFlagOpAsSet->op->getParentOp())) {
      rewriter.setInsertionPoint(rwOp->op);
      Value cond = getIsNotDeadLoopValue(forOp, loc, rewriter);
      conditions.push_back(cond);
    }
  }
  if (conditions.empty()) {
    return nullptr;
  } else if (conditions.size() == 1) {
    return conditions[0];
  } else if (conditions.size() == 2) {
    rewriter.setInsertionPoint(rwOp->op);
    return rewriter.create<arith::OrIOp>(loc, conditions[0], conditions[1]);
  } else {
    llvm_unreachable("unexpected/unhandled number of unit-flag conditions.");
  }
}

void Solver::handleUnitFlagEnabledOps(IRRewriter &rewriter) {
  for (auto *rwOp : unitFlagFeaturedOps) {
    auto unitFlagMode = getUnitFlagMode(rwOp);
    auto unitFlagCond = getUnitFlagCond(rewriter, rwOp);
    if (unitFlagMode == UNIT_FLAG::DISABLED) {
      return;
    }
    if (auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(rwOp->op)) {
      rewriter.setInsertionPoint(fixpipeOp);
      fixpipeOp.setUnitFlagModeAttr(
          UnitFlagAttr::get(rwOp->op->getContext(), unitFlagMode));
      if (unitFlagCond.has_value() && unitFlagCond.value()) {
        fixpipeOp.getUnitFlagCondMutable().assign(unitFlagCond.value());
      }
    } else if (auto mmadl1Op = dyn_cast<hivm::MmadL1Op>(rwOp->op)) {
      rewriter.setInsertionPoint(mmadl1Op);
      mmadl1Op.setUnitFlagModeAttr(
          UnitFlagAttr::get(rwOp->op->getContext(), unitFlagMode));
      if (unitFlagCond.has_value() && unitFlagCond.value()) {
        mmadl1Op.getUnitFlagCondMutable().assign(unitFlagCond.value());
      }
    } else {
      llvm_unreachable("Unsupport op to have unit-flag enabled.");
    }
  }
}

// Insert a PIPE_ALL barrier before function return.
void Solver::insertBarrierAllBeforeReturn(IRRewriter &rewriter) {
  auto returnOp = utils::getAssumedUniqueReturnOp(func);
  assert(returnOp != nullptr);
  rewriter.setInsertionPoint(returnOp);
  Location loc = returnOp->getLoc();
  auto pipe = PipeAttr::get(func->getContext(), hivm::PIPE::PIPE_ALL);
  rewriter.create<hivm::PipeBarrierOp>(loc, pipe);
}

// Collect indices for all Set/Wait ops to facilitate merging decisions.
void Solver::collectSetWaitOpsIndexes(OperationBase *op, SyncMap &syncMapBefore,
                                      SyncMap &syncMapAfter) {
  assert(op != nullptr);
  codeGenInclusiveStartIndex[op] = globalCodeGenIndex++;
  if (syncMapBefore.count(op)) {
    for (auto &syncOp : syncMapBefore[op]) {
      if (auto *setWaitOp = dyn_cast<SetWaitOp>(syncOp.get())) {
        for (auto eventId : setWaitOp->eventIds) {
          setWaitFlagOpsIndex[{setWaitOp->pipeSrc, setWaitOp->pipeDst, eventId}]
              .insert({globalCodeGenIndex++, setWaitOp});
        }
      }
    }
  }
  codeGenStartIndex[op] = globalCodeGenIndex++;
  if (auto *scopeOp = dyn_cast<Scope>(op)) {
    for (auto &childOp : scopeOp->body) {
      collectSetWaitOpsIndexes(childOp.get(), syncMapBefore, syncMapAfter);
    }
  }
  codeGenEndIndex[op] = globalCodeGenIndex++;
  if (syncMapAfter.count(op)) {
    for (auto &syncOp : syncMapAfter[op]) {
      if (auto *setWaitOp = dyn_cast<SetWaitOp>(syncOp.get())) {
        for (auto eventId : setWaitOp->eventIds) {
          setWaitFlagOpsIndex[{setWaitOp->pipeSrc, setWaitOp->pipeDst, eventId}]
              .insert({globalCodeGenIndex++, setWaitOp});
        }
      }
    }
  }
  codeGenInclusiveEndIndex[op] = globalCodeGenIndex++;
}

void Solver::resetAndBuildSetWaitOpIndex(SyncMap &syncMapBefore,
                                         SyncMap &syncMapAfter) {
  globalCodeGenIndex = 0;
  codeGenStartIndex.clear();
  codeGenEndIndex.clear();
  codeGenInclusiveStartIndex.clear();
  codeGenInclusiveEndIndex.clear();
  setWaitFlagOpsIndex.clear();
  collectSetWaitOpsIndexes(funcIr.get(), syncMapBefore, syncMapAfter);
}

// Check whether a backward-sync event id can be merged at scope level.
bool Solver::checkMergeable(Scope *scopeOp, hivm::PIPE pipeSrc,
                            hivm::PIPE pipeDst, hivm::EVENT eventId,
                            bool shouldBeUsedAtleastOnce) {
  auto &index = setWaitFlagOpsIndex[{pipeSrc, pipeDst, eventId}];
  auto it = index.lower_bound({codeGenInclusiveStartIndex[scopeOp], nullptr});
  bool usedAtleastOnce =
      it != index.end() && it->first < codeGenInclusiveEndIndex[scopeOp];
  if (shouldBeUsedAtleastOnce && !usedAtleastOnce) {
    return false;
  }
  if (auto *conditionOp = dyn_cast<Condition>(scopeOp)) {
    return checkMergeable(conditionOp->getTrueScope(), pipeSrc, pipeDst,
                          eventId, true) &&
           checkMergeable(conditionOp->getFalseScope(), pipeSrc, pipeDst,
                          eventId, true);
  }
  if (auto *loopOp = dyn_cast<Loop>(scopeOp)) {
    for (auto &childOp : loopOp->body) {
      if (auto *childScopeOp = dyn_cast<Scope>(childOp.get())) {
        if (!checkMergeable(childScopeOp, pipeSrc, pipeDst, eventId, false)) {
          return false;
        }
      }
    }
    for (auto &childOp : loopOp->body) {
      if (auto *childScopeOp = dyn_cast<Scope>(childOp.get())) {
        if (checkMergeable(childScopeOp, pipeSrc, pipeDst, eventId, true)) {
          return true;
        }
      }
    }
    return false;
  }
  for (auto &childOp : scopeOp->body) {
    auto it1 =
        index.lower_bound({codeGenInclusiveStartIndex[childOp.get()], nullptr});
    auto it2 = index.lower_bound({codeGenEndIndex[childOp.get()], nullptr});
    bool used = it1 != index.end() &&
                it1->first < codeGenInclusiveEndIndex[childOp.get()];
    bool before =
        it1 != index.end() && it1->first < codeGenStartIndex[childOp.get()];
    bool after = it2 != index.end() &&
                 it2->first < codeGenInclusiveEndIndex[childOp.get()];
    if (!used) {
      continue;
    }
    if (before || after) {
      return false;
    }
    if (!backwardSyncEvents[childOp.get()][{pipeSrc, pipeDst}].contains(
            eventId) ||
        backwardSyncEventsAfterMerge[childOp.get()][{pipeSrc, pipeDst}]
            .contains(eventId)) {
      return false;
    }
  }
  return true;
}

// Attempt to merge backward sync events across children and prune duplicates.
void Solver::mergeBackwardSyncEventIds(OperationBase *op) {
  auto *scopeOp = llvm::dyn_cast_if_present<Scope>(op);
  if (scopeOp == nullptr) {
    return;
  }
  for (auto &op : scopeOp->body) {
    mergeBackwardSyncEventIds(op.get());
  }

  if (llvm::isa_and_present<Condition, Loop>(op->parentOp)) {
    return;
  }

  auto *conditionOp = dyn_cast<Condition>(op);
  if (conditionOp != nullptr) {
    if (!conditionOp->hasFalseScope()) {
      return;
    }
  }

  auto &parentBackwardSyncEvents = backwardSyncEvents[scopeOp];
  llvm::DenseSet<std::tuple<hivm::PIPE, hivm::PIPE, hivm::EVENT>> toBeErased;
  for (uint64_t pipeSrcInt = 0;
       pipeSrcInt < static_cast<uint64_t>(hivm::PIPE::PIPE_NUM); pipeSrcInt++) {
    for (uint64_t pipeDstInt = 0;
         pipeDstInt < static_cast<uint64_t>(hivm::PIPE::PIPE_NUM);
         pipeDstInt++) {
      for (uint64_t eventIdInt = 0; eventIdInt < 8; eventIdInt++) {
        hivm::PIPE pipeSrc = static_cast<hivm::PIPE>(pipeSrcInt);
        hivm::PIPE pipeDst = static_cast<hivm::PIPE>(pipeDstInt);
        hivm::EVENT eventId = static_cast<hivm::EVENT>(eventIdInt);
        if (parentBackwardSyncEvents[{pipeSrc, pipeDst}].contains(eventId)) {
          continue;
        }
        if (checkMergeable(scopeOp, pipeSrc, pipeDst, eventId, true)) {
          toBeErased.insert({pipeSrc, pipeDst, eventId});
          parentBackwardSyncEvents[{pipeSrc, pipeDst}].insert(eventId);
        }
      }
    }
  }

  if (isa<Condition, Loop>(scopeOp)) {
    for (auto &op : scopeOp->body) {
      if (auto *block = llvm::dyn_cast<Scope>(op.get())) {
        for (auto &childOp : block->body) {
          if (auto *childScopeOp = dyn_cast<Scope>(childOp.get())) {
            for (auto [pipeSrc, pipeDst, eventId] : toBeErased) {
              backwardSyncEvents[childScopeOp][{pipeSrc, pipeDst}].erase(
                  eventId);
            }
          }
        }
      }
    }
  } else {
    for (auto &childOp : scopeOp->body) {
      if (auto *childScopeOp = dyn_cast<Scope>(childOp.get())) {
        for (auto [pipeSrc, pipeDst, eventId] : toBeErased) {
          backwardSyncEvents[childScopeOp][{pipeSrc, pipeDst}].erase(eventId);
        }
      }
    }
  }
}

SyncBeforeAfterMap Solver::getBeforeAfterSyncMaps() {
  SyncMap syncMapBefore, syncMapAfter;
  std::vector<ConflictPair *> conflictPairs;
  for (auto &conflictPair : chosenConflictedPairs) {
    conflictPairs.push_back(conflictPair.get());
  }
  for (auto &conflictPair : persistentChosenConflictedPairs) {
    conflictPairs.push_back(conflictPair.get());
  }

  for (auto *conflictPair : conflictPairs) {
    if (conflictPair->isUseless) {
      continue;
    }
    if (conflictPair->replacedWithUnitFlag) {
      continue;
    }
    assert(conflictPair->opSet != nullptr && conflictPair->opWait != nullptr);
    if (conflictPair->isBarrier()) {
      auto barrierOp =
          std::make_unique<BarrierOp>(nullptr, nullptr, conflictPair->waitPipe);
      LLVM_DEBUG(barrierOp->debugId = conflictPair->debugId);
      syncMapBefore[conflictPair->opWait].push_back(std::move(barrierOp));
    } else {
      auto setOp = std::make_unique<SetFlagOp>(
          conflictPair->opSet->op, conflictPair->opSet->parentOp,
          conflictPair->eventIds, conflictPair->setPipe,
          conflictPair->waitPipe);
      auto waitOp = std::make_unique<WaitFlagOp>(
          conflictPair->opWait->op, conflictPair->opWait->parentOp,
          conflictPair->eventIds, conflictPair->setPipe,
          conflictPair->waitPipe);
      if (conflictPair->multibufferLoopPar != nullptr) {
        setOp->multibufferLoopPar = conflictPair->multibufferLoopPar;
        waitOp->multibufferLoopPar = conflictPair->multibufferLoopPar;
      }
      if (conflictPair->setOnLastIterOnly) {
        setOp->checkLastIter = true;
      }
      if (conflictPair->waitOnFirstIterOnly) {
        waitOp->checkFirstIter = true;
      }
      LLVM_DEBUG({
        setOp->debugId = conflictPair->debugId;
        waitOp->debugId = conflictPair->debugId;
      });
      syncMapAfter[conflictPair->opSet].push_back(std::move(setOp));
      syncMapBefore[conflictPair->opWait].push_front(std::move(waitOp));
    }
  }

  resetAndBuildSetWaitOpIndex(syncMapBefore, syncMapAfter);
  auto *scopeOp = dyn_cast<Scope>(funcIr.get());
  assert(scopeOp != nullptr && scopeOp->body.front() != nullptr);
  mergeBackwardSyncEventIds(scopeOp->body.front().get());

  for (auto &[op, mp] : backwardSyncEvents) {
    if (mp.empty()) {
      continue;
    }
    auto *scopeOp = dyn_cast<Scope>(op);
    assert(scopeOp != nullptr);
    for (auto [setWaitPipes, eventIdsSet] : mp) {
      if (eventIdsSet.empty()) {
        continue;
      }
      llvm::SmallVector<hivm::EVENT> eventIds(eventIdsSet.begin(),
                                              eventIdsSet.end());
      auto [setPipe, waitPipe] = setWaitPipes;
      auto setOp = std::make_unique<SetFlagOp>(scopeOp->op, scopeOp->parentOp,
                                               eventIds, setPipe, waitPipe);
      auto waitOp = std::make_unique<WaitFlagOp>(scopeOp->op, scopeOp->parentOp,
                                                 eventIds, setPipe, waitPipe);
      setOp->allAtOnce = true;
      waitOp->allAtOnce = true;
      syncMapBefore[scopeOp].push_back(std::move(setOp));
      syncMapAfter[scopeOp].push_front(std::move(waitOp));
    }
  }
  return std::make_pair(std::move(syncMapBefore), std::move(syncMapAfter));
}

// Generate MLIR ops from computed sync maps (inserting via rewriter).
void Solver::generateResultOps() {
  IRRewriter rewriter(func->getContext());
  auto [syncMapBefore, syncMapAfter] = getBeforeAfterSyncMaps();
  for (auto &[op, syncOps] : syncMapBefore) {
    assert(op != nullptr);
    for (auto &syncOp : syncOps) {
      if (auto *barrierOp = dyn_cast<BarrierOp>(syncOp.get())) {
        insertBarrierOp(rewriter, op, barrierOp, false);
      } else if (auto *setFlagOp = dyn_cast<SetFlagOp>(syncOp.get())) {
        insertSetFlagOp(rewriter, op, setFlagOp, false);
      } else if (auto *waitFlagOp = dyn_cast<WaitFlagOp>(syncOp.get())) {
        insertWaitFlagOp(rewriter, op, waitFlagOp, false);
      }
    }
  }
  for (auto &[op, syncOps] : syncMapAfter) {
    assert(op != nullptr);
    for (auto &syncOp : llvm::reverse(syncOps)) {
      if (auto *barrierOp = dyn_cast<BarrierOp>(syncOp.get())) {
        insertBarrierOp(rewriter, op, barrierOp, true);
      } else if (auto *setFlagOp = dyn_cast<SetFlagOp>(syncOp.get())) {
        insertSetFlagOp(rewriter, op, setFlagOp, true);
      } else if (auto *waitFlagOp = dyn_cast<WaitFlagOp>(syncOp.get())) {
        insertWaitFlagOp(rewriter, op, waitFlagOp, true);
      }
    }
  }

  insertMmadL1SyncArgs(rewriter);
  handleUnitFlagEnabledOps(rewriter);
  insertBarrierAllBeforeReturn(rewriter);
}

// Insert generated sync ops into funcIr (in-memory IR) for testing/inspection.
void Solver::generateFuncIrResultOps() {
  auto [syncMapBefore, syncMapAfter] = getBeforeAfterSyncMaps();
  for (auto &e : syncMapBefore) {
    auto *op = e.first;
    assert(op != nullptr);
    auto &syncOps = e.second;
    if (syncOps.empty()) {
      continue;
    }
    auto *parentScopeOp = llvm::dyn_cast_if_present<Scope>(op->parentOp);
    assert(parentScopeOp != nullptr);
    auto it = std::find_if(
        parentScopeOp->body.begin(), parentScopeOp->body.end(),
        [&](const std::unique_ptr<OperationBase> &o) { return o.get() == op; });
    assert(it != parentScopeOp->body.end());
    for (auto &syncOp : syncOps) {
      it = parentScopeOp->body.insert(it, std::move(syncOp));
      ++it;
    }
  }
  for (auto &e : syncMapAfter) {
    auto *op = e.first;
    assert(op != nullptr);
    auto &syncOps = e.second;
    if (syncOps.empty()) {
      continue;
    }
    auto *parentScopeOp = llvm::dyn_cast_if_present<Scope>(op->parentOp);
    assert(parentScopeOp != nullptr);
    auto it = std::find_if(
        parentScopeOp->body.begin(), parentScopeOp->body.end(),
        [&](const std::unique_ptr<OperationBase> &o) { return o.get() == op; });
    assert(it != parentScopeOp->body.end());
    ++it;
    for (auto &syncOp : syncOps) {
      it = parentScopeOp->body.insert(it, std::move(syncOp));
      ++it;
    }
  }
  resultFuncIrWasGenerated = true;
}

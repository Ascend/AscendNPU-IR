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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverCodeGen.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/SCF/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <deque>
#include <memory>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "hivm-gss-code-gen"

using namespace mlir;
using namespace hivm::syncsolver;

// Choose where to insert generated sync ops (handles function-scope,
// placeholder, blocks, op-based insertion).
void CodeGenerator::setProperInsertionPoint(IRRewriter &rewriter,
                                            OperationBase *opBase,
                                            bool insertAfterOp) {
  if (opBase->parentOp == funcIr.get()) {
    if (insertAfterOp) {
      auto returnOp = utils::getAssumedUniqueReturnOp(funcOp);
      rewriter.setInsertionPoint(returnOp);
    } else {
      if (!resultFuncIrWasGenerated) {
        auto *scopeOp = dyn_cast<Scope>(opBase);
        const auto &it = llvm::find_if(
            scopeOp->body, [](const std::unique_ptr<OperationBase> &opBase) {
              return opBase->op != nullptr;
            });
        assert(it != scopeOp->body.end());
        auto *frontOp = (*it)->op;
        assert(frontOp != nullptr);
        rewriter.setInsertionPoint(frontOp);
      } else {
        rewriter.setInsertionPointToStart(
            &dyn_cast<func::FuncOp>(funcIr->op).getBody().front());
      }
    }
  } else if (auto *placeHolderOp = dyn_cast<PlaceHolder>(opBase)) {
    if (placeHolderOp->block != nullptr) {
      if (placeHolderOp->scopeBegin) {
        rewriter.setInsertionPointToStart(placeHolderOp->block);
      } else {
        if (!placeHolderOp->block->empty() &&
            placeHolderOp->block->back()
                .hasTrait<mlir::OpTrait::IsTerminator>()) {
          rewriter.setInsertionPoint(&placeHolderOp->block->back());
        } else {
          rewriter.setInsertionPointToEnd(placeHolderOp->block);
        }
      }
    } else if (placeHolderOp->beforeOp != nullptr ||
               placeHolderOp->afterOp != nullptr) {
      assert(placeHolderOp->beforeOp == nullptr ||
             placeHolderOp->afterOp == nullptr);
      OperationBase *linkedOpBase = placeHolderOp->beforeOp != nullptr
                                        ? placeHolderOp->beforeOp
                                        : placeHolderOp->afterOp;
      assert(linkedOpBase != nullptr);
      assert(linkedOpBase->op != nullptr);
      if (insertAfterOp) {
        rewriter.setInsertionPointAfter(linkedOpBase->op);
      } else {
        rewriter.setInsertionPoint(linkedOpBase->op);
      }
    } else {
      llvm_unreachable(
          "setProperInsertionPoint: unhandled place-holder op case.");
    }
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
Location CodeGenerator::getProperLoc(OperationBase *opBase) {
  assert(opBase != nullptr);
  if (auto *placeHolderOp = dyn_cast<PlaceHolder>(opBase)) {
    if (placeHolderOp->block != nullptr) {
      assert(placeHolderOp->block->getParentOp() != nullptr);
      return placeHolderOp->block->getParentOp()->getLoc();
    } else if (placeHolderOp->beforeOp != nullptr) {
      assert(placeHolderOp->afterOp == nullptr);
      assert(placeHolderOp->beforeOp->op != nullptr);
      return placeHolderOp->beforeOp->op->getLoc();
    } else if (placeHolderOp->afterOp != nullptr) {
      assert(placeHolderOp->beforeOp == nullptr);
      assert(placeHolderOp->afterOp->op != nullptr);
      return placeHolderOp->afterOp->op->getLoc();
    } else {
      llvm_unreachable("getProperLoc: unhandled place-holder op case.");
    }
  }
  if (opBase->op == nullptr && opBase->parentOp != nullptr) {
    return getProperLoc(opBase->parentOp);
  }
  assert(opBase->op != nullptr);
  return opBase->op->getLoc();
}

void CodeGenerator::insertBlockOp(IRRewriter &rewriter, OperationBase *opBase,
                                  BarrierOp *barrierOp, bool insertAfterOp) {
  assert(opBase != nullptr && barrierOp != nullptr);
  if (barrierOp->pipe != PIPE::PIPE_ALL) {
    llvm_unreachable("barriers in cross-core sync are expected to be of type "
                     "barrier-all only.");
  }

  PatternRewriter::InsertionGuard guard(rewriter);
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);

  auto *ctx = funcOp->getContext();
  Location loc = getProperLoc(opBase);
  auto cubeCoreAttr = hivm::TCoreTypeAttr::get(ctx, TCoreType::CUBE);
  auto vectorCoreAttr = hivm::TCoreTypeAttr::get(ctx, TCoreType::VECTOR);

  const auto intraBlockSyncFlagIdAttr1 =
      rewriter.getI64IntegerAttr(blockAllIntraSyncFlagId1);
  const auto intraBlockSyncFlagIdAttr2 =
      rewriter.getI64IntegerAttr(blockAllIntraSyncFlagId2);

  auto pipeSAttr = PipeAttr::get(ctx, PIPE::PIPE_S);
  auto pipeAllAttr = PipeAttr::get(ctx, PIPE::PIPE_ALL);

  rewriter.create<PipeBarrierOp>(loc, pipeAllAttr);
  rewriter.create<hivm::SyncBlockSetOp>(loc, vectorCoreAttr, pipeSAttr,
                                        pipeSAttr, intraBlockSyncFlagIdAttr1);
  rewriter.create<hivm::SyncBlockWaitOp>(loc, cubeCoreAttr, pipeSAttr,
                                         pipeSAttr, intraBlockSyncFlagIdAttr1);
  rewriter.create<hivm::SyncBlockSetOp>(loc, cubeCoreAttr, pipeSAttr, pipeSAttr,
                                        intraBlockSyncFlagIdAttr2);
  rewriter.create<hivm::SyncBlockWaitOp>(loc, vectorCoreAttr, pipeSAttr,
                                         pipeSAttr, intraBlockSyncFlagIdAttr2);
}

// Insert a PipeBarrierOp at the resolved insertion point and location.
void CodeGenerator::insertBarrierOp(IRRewriter &rewriter, OperationBase *opBase,
                                    BarrierOp *barrierOp, bool insertAfterOp) {
  assert(opBase != nullptr && barrierOp != nullptr);
  if (options.isCrossCoreMode()) {
    insertBlockOp(rewriter, opBase, barrierOp, insertAfterOp);
    return;
  }
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  Location loc = getProperLoc(opBase);
  auto pipe = PipeAttr::get(funcOp->getContext(), barrierOp->pipe);
  rewriter.create<PipeBarrierOp>(loc, pipe);
}

// Insert SetFlagOp(s) handling multi-buffer and conditional (first/last iter)
// wrapping.
void CodeGenerator::insertSetFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                                    SetFlagOp *setFlagOp, bool insertAfterOp) {
  assert(opBase != nullptr && setFlagOp != nullptr);
  if (options.isCrossCoreMode()) {
    insertSetBlockFlagOp(rewriter, opBase, setFlagOp, insertAfterOp);
    return;
  }
  if (llvm::succeeded(handleMmadL1SyncOps(rewriter, opBase, setFlagOp))) {
    return;
  }
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  auto *ctx = funcOp->getContext();
  Location loc = getProperLoc(opBase);
  assert(!setFlagOp->checkFirstIter);
  if (setFlagOp->checkLastIter) {
    auto *parentLoop = OperationBase::getParentloop(setFlagOp);
    auto forOp = dyn_cast<scf::ForOp>(parentLoop->op);
    assert(forOp != nullptr);
    Value cond = getIsLastIterationValue(forOp, loc, rewriter);
    auto ifOp = rewriter.create<scf::IfOp>(loc, cond);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }
  auto setPipe = PipeAttr::get(ctx, setFlagOp->pipeSrc);
  auto waitPipe = PipeAttr::get(ctx, setFlagOp->pipeDst);
  if (!setFlagOp->allAtOnce && setFlagOp->eventIds.size() > 1) {
    auto selectedBuffer = getMultiBufferSelectOp(rewriter, setFlagOp);
    rewriter.create<hivm::SetFlagOp>(loc, setPipe, waitPipe, EventAttr{},
                                     selectedBuffer);
  } else {
    for (auto eventId : setFlagOp->eventIds) {
      auto eventIdAttr = EventAttr::get(ctx, static_cast<hivm::EVENT>(eventId));
      rewriter.create<hivm::SetFlagOp>(loc, setPipe, waitPipe, eventIdAttr,
                                       Value{});
    }
  }
}

// Insert WaitFlagOp(s) handling multi-buffer and conditional wrapping.
void CodeGenerator::insertWaitFlagOp(IRRewriter &rewriter,
                                     OperationBase *opBase,
                                     WaitFlagOp *waitFlagOp,
                                     bool insertAfterOp) {
  assert(opBase != nullptr && waitFlagOp != nullptr);
  if (options.isCrossCoreMode()) {
    insertWaitBlockFlagOp(rewriter, opBase, waitFlagOp, insertAfterOp);
    return;
  }
  if (llvm::succeeded(handleMmadL1SyncOps(rewriter, opBase, waitFlagOp))) {
    return;
  }
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  auto *ctx = funcOp->getContext();
  Location loc = getProperLoc(opBase);
  assert(!waitFlagOp->checkLastIter);
  if (waitFlagOp->checkFirstIter) {
    auto *parentLoop = OperationBase::getParentloop(waitFlagOp);
    auto forOp = dyn_cast<scf::ForOp>(parentLoop->op);
    assert(forOp != nullptr);
    Value cond = getIsFirstIterationValue(forOp, loc, rewriter);
    auto ifOp = rewriter.create<scf::IfOp>(loc, cond);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }
  auto setPipe = PipeAttr::get(ctx, waitFlagOp->pipeSrc);
  auto waitPipe = PipeAttr::get(ctx, waitFlagOp->pipeDst);
  if (!waitFlagOp->allAtOnce && waitFlagOp->eventIds.size() > 1) {
    auto selectedBuffer = getMultiBufferSelectOp(rewriter, waitFlagOp);
    rewriter.create<hivm::WaitFlagOp>(loc, setPipe, waitPipe, EventAttr{},
                                      selectedBuffer);
  } else {
    for (auto eventId : waitFlagOp->eventIds) {
      auto eventIdAttr = EventAttr::get(ctx, static_cast<hivm::EVENT>(eventId));
      rewriter.create<hivm::WaitFlagOp>(loc, setPipe, waitPipe, eventIdAttr,
                                        Value{});
    }
  }
}

// Insert SetBlockFlagOp(s) handling multi-buffer and conditional (first/last
// iter) wrapping.
void CodeGenerator::insertSetBlockFlagOp(IRRewriter &rewriter,
                                         OperationBase *opBase,
                                         SetFlagOp *setFlagOp,
                                         bool insertAfterOp) {
  assert(opBase != nullptr && setFlagOp != nullptr);
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  auto *ctx = funcOp->getContext();
  Location loc = getProperLoc(opBase);
  assert(setFlagOp->coreType == hivm::TCoreType::VECTOR ||
         setFlagOp->coreType == hivm::TCoreType::CUBE);
  auto coreTypeAttr = hivm::TCoreTypeAttr::get(ctx, setFlagOp->coreType);
  assert(!setFlagOp->checkLastIter);
  auto setPipe = PipeAttr::get(ctx, setFlagOp->pipeSrc);
  auto waitPipe = PipeAttr::get(ctx, setFlagOp->pipeDst);
  if (!setFlagOp->allAtOnce && setFlagOp->eventIds.size() > 1) {
    auto selectedBuffer = getMultiBufferBlockSelectOp(rewriter, setFlagOp);
    rewriter.create<hivm::SyncBlockSetOp>(loc, coreTypeAttr, setPipe, waitPipe,
                                          selectedBuffer);
  } else {
    for (auto eventId : setFlagOp->eventIds) {
      rewriter.create<hivm::SyncBlockSetOp>(
          loc, coreTypeAttr, setPipe, waitPipe,
          rewriter.getI64IntegerAttr(eventId));
    }
  }
}

// Insert WaitBlockFlagOp(s) handling multi-buffer and conditional wrapping.
void CodeGenerator::insertWaitBlockFlagOp(IRRewriter &rewriter,
                                          OperationBase *opBase,
                                          WaitFlagOp *waitFlagOp,
                                          bool insertAfterOp) {
  assert(opBase != nullptr && waitFlagOp != nullptr);
  setProperInsertionPoint(rewriter, opBase, insertAfterOp);
  auto *ctx = funcOp->getContext();
  Location loc = getProperLoc(opBase);
  assert(waitFlagOp->coreType == hivm::TCoreType::VECTOR ||
         waitFlagOp->coreType == hivm::TCoreType::CUBE);
  auto coreTypeAttr = hivm::TCoreTypeAttr::get(ctx, waitFlagOp->coreType);
  assert(!waitFlagOp->checkFirstIter);
  auto setPipe = PipeAttr::get(ctx, waitFlagOp->pipeSrc);
  auto waitPipe = PipeAttr::get(ctx, waitFlagOp->pipeDst);
  if (!waitFlagOp->allAtOnce && waitFlagOp->eventIds.size() > 1) {
    auto selectedBuffer = getMultiBufferBlockSelectOp(rewriter, waitFlagOp);
    rewriter.create<hivm::SyncBlockWaitOp>(loc, coreTypeAttr, setPipe, waitPipe,
                                           selectedBuffer);
  } else {
    for (auto eventId : waitFlagOp->eventIds) {
      rewriter.create<hivm::SyncBlockWaitOp>(
          loc, coreTypeAttr, setPipe, waitPipe,
          rewriter.getI64IntegerAttr(eventId));
    }
  }
}

// Build/select a runtime i64 value that picks which buffer/event to use for
// multi-buffer sync.
Value CodeGenerator::getNestedIndexModular(IRRewriter &rewriter,
                                           SetWaitOp *syncOp) {
  auto multibufferLoop = syncOp->eventIdInfo.multibufferLoop;
  assert(multibufferLoop != nullptr);

  int64_t eventIdNum = static_cast<int64_t>(syncOp->eventIds.size());
  auto key = std::make_pair(multibufferLoop, eventIdNum);
  auto [it, isInserted] = nestedIndexModularMem.insert({key, Value{}});
  if (!isInserted) {
    return it->second;
  }

  PatternRewriter::InsertionGuard guard(rewriter);
  Value modularIndex =
      createNestedIndexModular(rewriter, multibufferLoop, eventIdNum);
  return it->second = modularIndex;
}

Value CodeGenerator::getMultiBufferSelectOp(IRRewriter &rewriter,
                                            SetWaitOp *syncOp) {
  assert(syncOp != nullptr);
  if (!syncOp->eventIdInfo.multibufferLoop) {
    return nullptr;
  }

  auto multibufferLoop = syncOp->eventIdInfo.multibufferLoop;
  assert(llvm::isa_and_present<scf::ForOp>(multibufferLoop));
  auto [it, isInserted] =
      bufferSelectedMem[multibufferLoop].insert({syncOp->eventIds, Value{}});
  if (!isInserted) {
    return it->second;
  }

  Value counter = getNestedIndexModular(rewriter, syncOp);
  assert(counter.getDefiningOp() != nullptr);

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(counter.getDefiningOp());
  Location loc = counter.getDefiningOp()->getLoc();

  Value bufferSelected;
  if (syncOp->eventIds.size() == 2) {
    counter = rewriter.create<arith::IndexCastOp>(
        counter.getLoc(), rewriter.getI1Type(), counter);
    Value firstID = rewriter.create<arith::ConstantIntOp>(
        loc, syncOp->eventIds[0], rewriter.getI64Type());
    Value secondID = rewriter.create<arith::ConstantIntOp>(
        loc, syncOp->eventIds[1], rewriter.getI64Type());
    bufferSelected = rewriter.create<arith::SelectOp>(
        loc, rewriter.getI64Type(), counter, firstID, secondID);
  } else {
    Value selectedValue{nullptr};
    for (auto [i, eventId] : llvm::enumerate(syncOp->eventIds)) {
      auto eventIdAttr =
          rewriter.getIntegerAttr(rewriter.getI64Type(), syncOp->eventIds[i]);
      Value eventIdValue = rewriter.create<arith::ConstantOp>(loc, eventIdAttr);
      if (!selectedValue) {
        selectedValue = eventIdValue;
        continue;
      }
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value cmpEqOp = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, counter, iVal);
      selectedValue = rewriter.create<arith::SelectOp>(
          loc, rewriter.getI64Type(), cmpEqOp, eventIdValue, selectedValue);
    }
    bufferSelected = selectedValue;
  }

  return it->second = bufferSelected;
}

Value CodeGenerator::getCVMultiBufferSelectOpConsecutive(IRRewriter &rewriter,
                                                         SetWaitOp *syncOp) {
  assert(syncOp != nullptr);
  auto multibufferLoop = isa<SetFlagOp>(syncOp)
                             ? syncOp->eventIdInfo.multibufferUnrollLoop1
                             : syncOp->eventIdInfo.multibufferUnrollLoop2;
  if (!multibufferLoop) {
    return nullptr;
  }
  for (size_t i = 1; i < syncOp->eventIds.size(); i++) {
    if (syncOp->eventIds[i - 1] + 1 != syncOp->eventIds[i]) {
      return nullptr;
    }
  }
  auto forOp = dyn_cast<scf::ForOp>(multibufferLoop.getOperation());
  assert(forOp && forOp->hasAttr(kMultibufferUnrollAttrName));
  assert(scf::utils::isNormalized(forOp));
  auto loc = forOp->getLoc();
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value loopIndVar = forOp.getInductionVar();
  auto eventIdAttr =
      rewriter.getIntegerAttr(rewriter.getIndexType(), syncOp->eventIds[0]);
  auto eventIdValue = rewriter.create<arith::ConstantOp>(loc, eventIdAttr);
  Value eventId = rewriter.create<arith::AddIOp>(loc, rewriter.getIndexType(),
                                                 loopIndVar, eventIdValue);
  return getValueOrCreateCastToI64(rewriter, loc, eventId);
}

Value CodeGenerator::getCVMultiBufferSelectOp(IRRewriter &rewriter,
                                              SetWaitOp *syncOp) {
  assert(syncOp != nullptr);
  auto multibufferLoop = isa<SetFlagOp>(syncOp)
                             ? syncOp->eventIdInfo.multibufferUnrollLoop1
                             : syncOp->eventIdInfo.multibufferUnrollLoop2;
  if (!multibufferLoop) {
    return nullptr;
  }
  auto forOp = dyn_cast<scf::ForOp>(multibufferLoop.getOperation());
  assert(forOp && forOp->hasAttr(kMultibufferUnrollAttrName));
  assert(scf::utils::isNormalized(forOp));
  auto loc = forOp->getLoc();
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value loopIndVar = forOp.getInductionVar();
  Value selectedValue{nullptr};
  for (auto [i, eventId] : llvm::enumerate(syncOp->eventIds)) {
    auto eventIdAttr =
        rewriter.getIntegerAttr(rewriter.getI64Type(), syncOp->eventIds[i]);
    Value eventIdValue = rewriter.create<arith::ConstantOp>(loc, eventIdAttr);
    if (!selectedValue) {
      selectedValue = eventIdValue;
      continue;
    }
    Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value cmpEqOp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, loopIndVar, iVal);
    selectedValue = rewriter.create<arith::SelectOp>(
        loc, rewriter.getI64Type(), cmpEqOp, eventIdValue, selectedValue);
  }
  return selectedValue;
}

Value CodeGenerator::getMultiBufferBlockSelectOp(IRRewriter &rewriter,
                                                 SetWaitOp *syncOp) {
  assert(syncOp != nullptr);
  if (auto selectOp = getMultiBufferSelectOp(rewriter, syncOp)) {
    return selectOp;
  }
  if (auto selectOp = getCVMultiBufferSelectOpConsecutive(rewriter, syncOp)) {
    return selectOp;
  }
  if (auto selectOp = getCVMultiBufferSelectOp(rewriter, syncOp)) {
    return selectOp;
  }
  llvm_unreachable("unhandled cross-core sync case with multibuffer enabled.");
}

// Get an event id Value for a given SetWaitOp (creates constant or uses
// select).
Value CodeGenerator::getEventIdValue(IRRewriter &rewriter, SetWaitOp *setWaitOp,
                                     Location loc) {
  assert(setWaitOp != nullptr);
  assert(!setWaitOp->eventIds.empty());
  if (setWaitOp->eventIds.size() > 1) {
    return getMultiBufferSelectOp(rewriter, setWaitOp);
  }
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  return rewriter.create<arith::ConstantIntOp>(loc, setWaitOp->eventIds[0],
                                               rewriter.getI64Type());
}

// Attempt to attach sync args to MmadL1 ops by recognizing special load L0 / L1
// patterns.
llvm::LogicalResult CodeGenerator::handleMmadL1SyncOps(IRRewriter &rewriter,
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
      mmadl1SyncArgsMap[mmadl1Op].l1AWaitL0Event =
          getEventIdValue(rewriter, setFlagOp, mmadl1Op->getLoc());
    } else if (isa<LoadL0BOp>(opBase)) {
      mmadl1SyncArgsMap[mmadl1Op].l1BWaitL0Event =
          getEventIdValue(rewriter, setFlagOp, mmadl1Op->getLoc());
    }
  } else if (auto *waitFlagOp = dyn_cast<WaitFlagOp>(syncOp)) {
    if (isa<LoadL0AOp>(opBase)) {
      mmadl1SyncArgsMap[mmadl1Op].l0WaitL1AEvent =
          getEventIdValue(rewriter, waitFlagOp, mmadl1Op->getLoc());
    } else if (isa<LoadL0BOp>(opBase)) {
      mmadl1SyncArgsMap[mmadl1Op].l0WaitL1BEvent =
          getEventIdValue(rewriter, waitFlagOp, mmadl1Op->getLoc());
    }
  }
  return llvm::success();
}

Value CodeGenerator::getLoopDBCond(IRRewriter &rewriter, Operation *op) {
  if (!checkAllParentLoopsAreForLoops(op)) {
    return nullptr;
  }
  auto parentLoop = op->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return nullptr;
  }
  if (loopDBCondMap.contains(parentLoop)) {
    return loopDBCondMap[parentLoop];
  }
  Value moduleIndex = createNestedIndexForOp(rewriter, op);
  Value moduleIdx = rewriter.create<arith::IndexCastOp>(
      moduleIndex.getLoc(), rewriter.getI64Type(), moduleIndex);
  return loopDBCondMap[parentLoop] = moduleIdx;
}

// Create and propagate sync args into MmadL1 op arguments.
void CodeGenerator::insertMmadL1SyncArgs(IRRewriter &rewriter) {
  for (auto &[mmadL1Op, syncArgs] : mmadl1SyncArgsMap) {
    rewriter.setInsertionPoint(mmadL1Op);
    auto defaultValue = rewriter.create<arith::ConstantIntOp>(
        mmadL1Op->getLoc(), -1, rewriter.getI64Type());
    if (options.isMemBasedArch) {
      syncArgs.kLoopDBCond = getLoopDBCond(rewriter, mmadL1Op.getOperation());
    }
    SmallVector<Value> newArgs;
    newArgs.push_back(syncArgs.l0WaitL1AEvent);
    newArgs.push_back(syncArgs.l0WaitL1BEvent);
    newArgs.push_back(syncArgs.l1AWaitL0Event);
    newArgs.push_back(syncArgs.l1BWaitL0Event);
    newArgs.push_back(syncArgs.kLoopDBCond);
    newArgs.push_back(syncArgs.bwdPipeMPipeMTE1Event0);
    newArgs.push_back(syncArgs.bwdPipeMPipeMTE1Event1);
    for (auto &val : newArgs) {
      if (!val || val == Value{}) {
        val = defaultValue;
      }
    }
    mmadL1Op.getSyncRelatedArgsMutable().assign(newArgs);
  }
}

void CodeGenerator::handleUnitFlagEnabledOps(IRRewriter &rewriter) {
  for (auto *rwOp : unitFlagFeaturedOps) {
    auto unitFlagInfo = rwOp->mergedUnitFlagInfo;
    auto unitFlagEnabledOp =
        llvm::dyn_cast_if_present<UnitFlagEnabledInterface>(rwOp->op);
    assert(unitFlagEnabledOp != nullptr);
    if (auto unitFlagArgsOpt =
            unitFlagInfo.getUnitFlagArgs(unitFlagEnabledOp, rewriter)) {
      auto [unitFlagModes, unitFlagConds] = unitFlagArgsOpt.value();
      assert(unitFlagModes.size() <= 1 ||
             unitFlagModes.size() == unitFlagConds.size());
      if (!unitFlagModes.empty()) {
        unitFlagEnabledOp.setUnitFlagModes(unitFlagModes);
        unitFlagEnabledOp.setUnitFlagConditions(unitFlagConds);
      }
    }
  }
}

// Insert a PIPE_ALL barrier before function return.
void CodeGenerator::insertBarrierAllBeforeReturn(IRRewriter &rewriter) {
  if (!hacc::utils::isDeviceEntry(funcOp)) {
    return;
  }
  auto returnOp = utils::getAssumedUniqueReturnOp(funcOp);
  assert(returnOp != nullptr);
  rewriter.setInsertionPoint(returnOp);
  Location loc = returnOp->getLoc();
  auto pipe = PipeAttr::get(funcOp->getContext(), hivm::PIPE::PIPE_ALL);
  rewriter.create<hivm::PipeBarrierOp>(loc, pipe);
}

// Generate MLIR ops from computed sync maps (inserting via rewriter).
void CodeGenerator::generateResultOps() {
  IRRewriter rewriter(funcOp->getContext());
  for (auto &[op, syncOps] : syncMapAfter) {
    if (auto *placeHolder = dyn_cast<PlaceHolder>(op)) {
      if (placeHolder->scopeBegin != nullptr ||
          placeHolder->scopeEnd != nullptr) {
        auto &syncMapBeforeRef = syncMapBefore[op];
        for (auto &syncOp : syncOps) {
          syncMapBeforeRef.emplace_back(std::move(syncOp));
        }
        syncOps.clear();
      }
    }
  }
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

  if (options.isIntraCoreMode()) {
    insertMmadL1SyncArgs(rewriter);
    handleUnitFlagEnabledOps(rewriter);
    insertBarrierAllBeforeReturn(rewriter);
  }
}

// Insert generated sync ops into funcIr (in-memory IR) for testing/inspection.
void CodeGenerator::generateFuncIrResultOps() {
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

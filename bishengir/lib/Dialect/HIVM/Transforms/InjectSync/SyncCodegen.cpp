//===------------- SyncCodegen.h ----Sync information collection ---------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCodegen.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCodegen.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "bishengir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

void SyncCodegen::Build() {
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);
  UpdateOpInsertSync(rewriter);

  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op2InsertSync.count(op)) {
      for (auto &syncBefore : op2InsertSync[op].pipeBefore) {
        SyncInsert(rewriter, op, syncBefore, true);
      }
      for (auto &syncAfter : llvm::reverse(op2InsertSync[op].pipeAfter)) {
        SyncInsert(rewriter, op, syncAfter, false);
      }
    }
  });

  if (syncAnalysisMode == SyncAnalysisMode::NORMALSYNC) {
    UpdateMmadL1SyncTemplateInter();
  }
}

void SyncCodegen::UpdateMmadL1SyncTemplateInter() {
  func_->walk<WalkOrder::PreOrder>([&](hivm::MmadL1Op mmadL1Op) {
    auto iter = mmadL12SyncTemplateInter.find(mmadL1Op);
    checkCondition(iter != mmadL12SyncTemplateInter.end(),
                   "mmadL1 must has SyncTemplateInter");
    SmallVector<Value> newArgs;
    newArgs.push_back(iter->second.MmadL1WaitL1AEvent);
    newArgs.push_back(iter->second.MmadL1WaitL1BEvent);
    newArgs.push_back(iter->second.L1AWaitMmadL1Event);
    newArgs.push_back(iter->second.L1B2WaitMmadL1Event);
    newArgs.push_back(iter->second.KLoopDBCond);
    newArgs.push_back(iter->second.BackPipeMPipeMTE1DBEvent0);
    newArgs.push_back(iter->second.BackPipeMPipeMTE1DBEvent1);
    auto syncArgs = mmadL1Op.getSyncRelatedArgsMutable();
    syncArgs.assign(newArgs);
  });
}

void SyncCodegen::UpdateOpInsertSync(IRRewriter &rewriter) {
  for (auto &nowElement : syncIR) {
    if (auto *compoundElement =
            dyn_cast<CompoundInstanceElement>(nowElement.get())) {
      handleEnableUnitFlag(rewriter, compoundElement);
      UpdateCompoundOpInsertSync(compoundElement);
      UpdateSyncTemplateInterForBackPipeMPipeMTE1DB(compoundElement);
    } else if (auto *placeHolder =
                   dyn_cast<PlaceHolderInstanceElement>(nowElement.get())) {
      updatePlaceHolderOpInsertSync(placeHolder);
    } else if (auto *loopElement =
                   dyn_cast<LoopInstanceElement>(nowElement.get())) {
      UpdateLoopOpInsertSync(loopElement);
    } else if (auto *branchElement =
                   dyn_cast<BranchInstanceElement>(nowElement.get())) {
      UpdateBranchOpInsertSync(branchElement);
    }
  }
}

void SyncCodegen::handleEnableUnitFlag(
    IRRewriter &rewriter, CompoundInstanceElement *nowCompound) const {
  auto unitFlagMode = nowCompound->getUnitFlagMode();
  auto unitFlagCond =
      nowCompound->getUnitFlagCond(nowCompound->elementOp->getLoc(), rewriter);
  if (unitFlagMode == UNIT_FLAG::DISABLED) {
    return;
  }
  if (auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(nowCompound->elementOp)) {
    rewriter.setInsertionPoint(fixpipeOp);
    fixpipeOp.setUnitFlagModeAttr(
        UnitFlagAttr::get(nowCompound->elementOp->getContext(), unitFlagMode));
    if (unitFlagCond.has_value() && unitFlagCond.value()) {
      fixpipeOp.getUnitFlagCondMutable().assign(unitFlagCond.value());
    }
  } else if (auto mmadl1Op = dyn_cast<hivm::MmadL1Op>(nowCompound->elementOp)) {
    rewriter.setInsertionPoint(mmadl1Op);
    mmadl1Op.setUnitFlagModeAttr(
        UnitFlagAttr::get(nowCompound->elementOp->getContext(), unitFlagMode));
    if (unitFlagCond.has_value() && unitFlagCond.value()) {
      mmadl1Op.getUnitFlagCondMutable().assign(unitFlagCond.value());
    }
  } else {
    llvm_unreachable("Unsupport op to have unit-flag enabled.");
  }
}

void SyncCodegen::updatePlaceHolderOpInsertSync(
    PlaceHolderInstanceElement *placeHolder) {
  Operation *terminatorOp = nullptr;
  auto *parentScope = syncIR[placeHolder->parentScopeId].get();
  if (auto *branchOp = dyn_cast<BranchInstanceElement>(parentScope)) {
    if (auto ifOp = dyn_cast<scf::IfOp>(branchOp->elementOp)) {
      if (branchOp->getBranchKind() == KindOfBranch::IF_BEGIN) {
        terminatorOp = ifOp.getThenRegion().front().getTerminator();
      } else {
        terminatorOp = ifOp.getElseRegion().front().getTerminator();
      }
    }
  } else if (auto *loopOp = dyn_cast<LoopInstanceElement>(parentScope)) {
    if (auto forOp = dyn_cast<scf::ForOp>(loopOp->elementOp)) {
      terminatorOp = forOp.getRegion().front().getTerminator();
    }
  }
  assert(terminatorOp != nullptr);
  assert(placeHolder->pipeAfter.empty());
  auto iter = op2InsertSync.find(terminatorOp);
  if (iter != op2InsertSync.end()) {
    // There are two MacroOp elements insert sync.
    iter->second.pipeBefore.insert(iter->second.pipeBefore.end(),
                                   placeHolder->pipeBefore.begin(),
                                   placeHolder->pipeBefore.end());
  } else {
    SyncPipeBuild pipeBuild;
    pipeBuild.pipeBefore = placeHolder->pipeBefore;
    op2InsertSync[terminatorOp] = pipeBuild;
  }
}

void SyncCodegen::UpdateCompoundOpInsertSync(
    CompoundInstanceElement *nowCompound) {
  auto iter = op2InsertSync.find(nowCompound->elementOp);
  if (iter != op2InsertSync.end()) {
    // There are two MacroOp elements insert sync.
    iter->second.pipeAfter.insert(iter->second.pipeAfter.end(),
                                  nowCompound->pipeAfter.begin(),
                                  nowCompound->pipeAfter.end());
    iter->second.pipeBefore.insert(iter->second.pipeBefore.end(),
                                   nowCompound->pipeBefore.begin(),
                                   nowCompound->pipeBefore.end());
  } else {
    SyncPipeBuild pipeBuild;
    pipeBuild.pipeBefore = nowCompound->pipeBefore;
    pipeBuild.pipeAfter = nowCompound->pipeAfter;
    op2InsertSync[nowCompound->elementOp] = pipeBuild;
  }
}

void SyncCodegen::UpdateSyncTemplateInterForBackPipeMPipeMTE1DB(
    CompoundInstanceElement *nowCompound) {
  auto mmadL1Op =
      llvm::dyn_cast_if_present<hivm::MmadL1Op>(nowCompound->elementOp);
  if (!mmadL1Op) {
    return;
  }
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPointToStart(&func_.getBody().front());
  auto iter = mmadL12SyncTemplateInter.find(mmadL1Op);
  if (iter == mmadL12SyncTemplateInter.end()) {
    InitDefaultSyncTemplateInterForMmadL1Op(rewriter, mmadL1Op);
  }
  if (!nowCompound->PipeMTE1ToPipeMSync ||
      nowCompound->PipeMTE1ToPipeMSync->uselessSync) {
    // There is no reverse or Eventid conflict, and the library itself
    // needs to be inserted for synchronization.
    return;
  }
  checkCondition(nowCompound->PipeMTE1ToPipeMSync->eventIds.size() == 1,
                 "expected PipeMTE1ToPipeMSync eventIds to be of size 1");
  auto backPipeMPipeMTE1DBEvent = rewriter.create<arith::ConstantIntOp>(
      nowCompound->elementOp->getLoc(),
      nowCompound->PipeMTE1ToPipeMSync->eventIds[0], rewriter.getI64Type());
  // mmadL1 Sync IR two updates, namely BackPipeMPipeMTE1DBEvent0 and
  // BackPipeMPipeMTE1DBEvent1.
  if (nowCompound->defVec.empty()) {
    mmadL12SyncTemplateInter[mmadL1Op].BackPipeMPipeMTE1DBEvent1 =
        backPipeMPipeMTE1DBEvent;
  } else {
    mmadL12SyncTemplateInter[mmadL1Op].BackPipeMPipeMTE1DBEvent0 =
        backPipeMPipeMTE1DBEvent;
  }
}

void SyncCodegen::InitDefaultSyncTemplateInterForMmadL1Op(
    IRRewriter &rewriter, hivm::MmadL1Op mmadL1Op) {
  auto defaultValue = rewriter.create<arith::ConstantIntOp>(
      mmadL1Op.getOperation()->getLoc(), -1, rewriter.getI64Type());
  SyncTemplateInter syncTemplateInter(defaultValue, defaultValue, defaultValue,
                                      defaultValue, defaultValue, defaultValue,
                                      defaultValue);
  Value KLoopDBCond = createNestedIndexForOp(rewriter, mmadL1Op.getOperation());
  if (KLoopDBCond) {
    syncTemplateInter.KLoopDBCond = KLoopDBCond;
  }
  mmadL12SyncTemplateInter[mmadL1Op] = syncTemplateInter;
}

void SyncCodegen::UpdateLoopOpInsertSync(LoopInstanceElement *nowElement) {
  SyncPipeBuild pipeBuild;
  if (nowElement->getLoopKind() == KindOfLoop::LOOP_END) {
    auto *loopBegin =
        dyn_cast<LoopInstanceElement>(syncIR[nowElement->beginId].get());
    checkCondition(loopBegin != nullptr,
                   "dyn_cast failed for LoopInstanceElement");
    checkCondition(loopBegin->pipeAfter.empty(),
                   "The node does not exist in synchronization!");
    checkCondition(nowElement->pipeBefore.empty(),
                   "The node does not exist in synchronization!");
    pipeBuild.pipeBefore = loopBegin->pipeBefore;
    pipeBuild.pipeAfter = nowElement->pipeAfter;
    op2InsertSync[nowElement->elementOp] = pipeBuild;
  }
}

void SyncCodegen::UpdateBranchOpInsertSync(BranchInstanceElement *nowElement) {
  SyncPipeBuild pipeBuild;
  if (nowElement->getBranchKind() == KindOfBranch::IF_END) {
    auto *branchBeginPtr =
        dyn_cast<BranchInstanceElement>(syncIR[nowElement->beginId].get());
    checkCondition(branchBeginPtr != nullptr,
                   "dyn_cast failed for BranchInstanceElement");
    checkCondition(branchBeginPtr->pipeAfter.empty(),
                   "The node does not exist in synchronization!");
    checkCondition(nowElement->pipeBefore.empty(),
                   "The node does not exist in synchronization!");
    pipeBuild.pipeBefore = branchBeginPtr->pipeBefore;
    pipeBuild.pipeAfter = nowElement->pipeAfter;
    op2InsertSync[nowElement->elementOp] = pipeBuild;
  }
}

void SyncCodegen::SyncInsert(IRRewriter &rewriter, Operation *op,
                             SyncOperation *sync, bool beforeInsert) {
  if (sync->uselessSync) {
    // Useless Sync does not require actual generation.
    return;
  }
  if (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER) {
    CreateBarrierOp(rewriter, op, sync, beforeInsert);
  } else if (sync->GetType() == SyncOperation::TYPE::SET_EVENT ||
             sync->GetType() == SyncOperation::TYPE::WAIT_EVENT) {
    checkCondition(!sync->eventIds.empty(),
                   "eventIds expected to not be empty");
    if (sync->eventIds.size() == 1) {
      CreateSetWaitOpForSingleBuffer(rewriter, op, sync, beforeInsert);
    } else {
      CreateSetWaitOpForMultiBuffer(rewriter, op, sync, beforeInsert);
    }
  } else if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET ||
             sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    checkCondition(!sync->eventIds.empty(),
                   "eventIds expected to not be empty");
    if (sync->eventIds.size() == 1) {
      CreateSetWaitBlockOpForSingleBuffer(rewriter, op, sync, beforeInsert);
    } else {
      CreateSetWaitBlockOpForMultiBuffer(rewriter, op, sync, beforeInsert);
    }
  } else if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_ALL) {
    CreateBlockSyncAllOp(rewriter, op, sync, beforeInsert);
  } else if (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER_CUBE ||
             sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER_VECTOR) {
    CreateBlockSyncBarrierOp(rewriter, op, sync, beforeInsert);
  } else {
    llvm_unreachable("Sync type not supported! ");
  }
}

void SyncCodegen::CreateBarrierOp(IRRewriter &rewriter, Operation *op,
                                  SyncOperation *sync, bool beforeInsert) {
  // Set sync insertion position.
  if (beforeInsert) {
    rewriter.setInsertionPoint(op);
  } else {
    rewriter.setInsertionPointAfter(op);
  }
  auto setPipe = PipeAttr::get(func_->getContext(), sync->GetActualSrcPipe());
  Location loc = op->getLoc();
  rewriter.create<hivm::PipeBarrierOp>(loc, setPipe);
}

void SyncCodegen::CreateSetWaitBlockOpForSingleBuffer(IRRewriter &rewriter,
                                                      Operation *op,
                                                      SyncOperation *sync,
                                                      bool beforeInsert) {
  // Set block sync insertion position.
  if (beforeInsert) {
    rewriter.setInsertionPoint(op);
  } else {
    rewriter.setInsertionPointAfter(op);
  }
  auto setPipe = PipeAttr::get(func_->getContext(), sync->GetActualSrcPipe());
  auto waitPipe = PipeAttr::get(func_->getContext(), sync->GetActualDstPipe());
  auto coreTypeAttr =
      hivm::TCoreTypeAttr::get(func_->getContext(), sync->syncCoreType);
  Location loc = op->getLoc();
  if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    rewriter.create<SyncBlockWaitOp>(
        loc, coreTypeAttr, setPipe, waitPipe,
        rewriter.getI64IntegerAttr(sync->eventIds[0]));
  } else if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET) {
    rewriter.create<SyncBlockSetOp>(
        loc, coreTypeAttr, setPipe, waitPipe,
        rewriter.getI64IntegerAttr(sync->eventIds[0]));
  }
}

void SyncCodegen::CreateSetWaitBlockOpForMultiBuffer(IRRewriter &rewriter,
                                                     Operation *op,
                                                     SyncOperation *sync,
                                                     bool beforeInsert) {
  if (sync->eventIds.size() > MAX_MULTI_BUFFER_NUM) {
    llvm_unreachable("Sync supports up to 16 buffers! ");
  }

  Location loc = op->getLoc();
  LoopLikeOpInterface forOp = op->getParentOfType<LoopLikeOpInterface>();
  if (!scf::utils::isNormalized(forOp)) {
    // TODO: call normalize loop pass before plan memory, currently CVPipeling
    // ensure the loop is normalized
    op->emitOpError("parent loop is not normalized");
    return;
  }
  rewriter.setInsertionPoint(op);
  auto mayLoopIndVars = forOp.getLoopInductionVars();
  checkCondition(mayLoopIndVars.has_value() && !mayLoopIndVars.value().empty(),
                 "forOp expected to have at least 1 induction variable");
  Value loopIndVar = mayLoopIndVars.value()[0];
  auto eventIdAttr =
      rewriter.getIntegerAttr(loopIndVar.getType(), sync->eventIds[0]);
  auto eventIdValue =
      rewriter.create<arith::ConstantOp>(op->getLoc(), eventIdAttr);
  Value id = rewriter.create<arith::AddIOp>(op->getLoc(), loopIndVar.getType(),
                                            loopIndVar, eventIdValue);
  if (!id.getType().isInteger(64)) {
    if (id.getType().isIndex()) {
      id = rewriter.create<arith::IndexCastOp>(op->getLoc(),
                                               rewriter.getIntegerType(64), id);
    } else if (id.getType().isInteger()) {
      id = rewriter.create<arith::ExtSIOp>(op->getLoc(),
                                           rewriter.getIntegerType(64), id);
    } else {
      llvm_unreachable("unhandled casting type");
    }
  }

  // Set sync insertion position.
  if (beforeInsert) {
    rewriter.setInsertionPoint(op);
  } else {
    rewriter.setInsertionPointAfter(op);
  }

  auto setPipe = PipeAttr::get(func_->getContext(), sync->GetSrcPipe());
  auto waitPipe = PipeAttr::get(func_->getContext(), sync->GetDstPipe());
  auto coreTypeAttr =
      hivm::TCoreTypeAttr::get(func_->getContext(), sync->syncCoreType);

  if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    rewriter.create<SyncBlockWaitOp>(loc, coreTypeAttr, setPipe, waitPipe, id);
  } else if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET) {
    rewriter.create<SyncBlockSetOp>(loc, coreTypeAttr, setPipe, waitPipe, id);
  }
}

void SyncCodegen::CreateBlockSyncBarrierOp(IRRewriter &rewriter, Operation *op,
                                           const SyncOperation *sync,
                                           bool beforeInsert) {
  if (beforeInsert) {
    rewriter.setInsertionPoint(op);
  } else {
    rewriter.setInsertionPointAfter(op);
  }
  hivm::SyncBlockMode syncBlockMode =
      sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER_CUBE
          ? hivm::SyncBlockMode::BARRIER_CUBE
          : hivm::SyncBlockMode::BARRIER_VECTOR;
  auto syncMode =
      hivm::SyncBlockModeAttr::get(func_.getContext(), syncBlockMode);
  rewriter.create<SyncBlockOp>(op->getLoc(), syncMode, IntegerAttr{}, Value{},
                               hivm::PipeAttr{}, hivm::PipeAttr{});
}

void SyncCodegen::CreateBlockSyncAllOp(IRRewriter &rewriter, Operation *op,
                                       SyncOperation *sync, bool beforeInsert) {
  // Set block sync insertion position.
  if (beforeInsert) {
    rewriter.setInsertionPoint(op);
  } else {
    rewriter.setInsertionPointAfter(op);
  }
  hivm::SyncBlockMode syncBlockMode = sync->syncCoreType == TCoreType::CUBE
                                          ? hivm::SyncBlockMode::ALL_CUBE
                                          : hivm::SyncBlockMode::ALL_VECTOR;
  Location loc = op->getLoc();
  auto syncMode =
      hivm::SyncBlockModeAttr::get(func_.getContext(), syncBlockMode);
  auto pipeAttr = PipeAttr::get(func_->getContext(), sync->GetActualSrcPipe());
  if (sync->syncCoreType == TCoreType::CUBE) {
    rewriter.create<SyncBlockOp>(loc, syncMode,
                                 rewriter.getI64IntegerAttr(sync->eventIds[0]),
                                 Value{}, pipeAttr, hivm::PipeAttr{});
  } else {
    rewriter.create<SyncBlockOp>(loc, syncMode,
                                 rewriter.getI64IntegerAttr(sync->eventIds[0]),
                                 Value{}, hivm::PipeAttr{}, pipeAttr);
  }
}

bool SyncCodegen::IsNeedLowerSyncToTemplate(Operation *op,
                                            const SyncOperation *sync) const {
  bool isVirtualMTE2 =
      sync->GetSrcPipe() == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A ||
      sync->GetSrcPipe() == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B ||
      sync->GetDstPipe() == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A ||
      sync->GetDstPipe() == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B;
  if (!isVirtualMTE2) {
    return false;
  }
  if (!isa<hivm::MmadL1Op>(op)) {
    return false;
  }
  return true;
}

bool SyncCodegen::NeedLowerSyncToTemplate(IRRewriter &rewriter, Operation *op,
                                          SyncOperation *sync, Value eventId) {
  if (!IsNeedLowerSyncToTemplate(op, sync)) {
    return false;
  }
  if (!eventId) {
    Location loc = op->getLoc();
    checkCondition(sync->eventIds.size() == 1,
                   "sync operation expected to have exactly 1 eventId");
    rewriter.setInsertionPointToStart(&func_.getBody().front());
    eventId = rewriter.create<arith::ConstantIntOp>(loc, sync->eventIds[0],
                                                    rewriter.getI64Type());
  }
  auto mmadL1Op = dyn_cast<hivm::MmadL1Op>(op);
  auto iter = mmadL12SyncTemplateInter.find(mmadL1Op);
  checkCondition(iter != mmadL12SyncTemplateInter.end(),
                 "mmadL1Op expected to be found in mmadL12SyncTemplateInter");
  if (sync->GetType() == SyncOperation::TYPE::WAIT_EVENT) {
    if (sync->GetSrcPipe() == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A) {
      sync->uselessSync = true;
      iter->second.MmadL1WaitL1AEvent = eventId;
      return true;
    }
    if (sync->GetSrcPipe() == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B) {
      iter->second.MmadL1WaitL1BEvent = eventId;
      sync->uselessSync = true;
      return true;
    }
  } else if (sync->GetType() == SyncOperation::TYPE::SET_EVENT) {
    if (sync->GetDstPipe() == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A) {
      iter->second.L1AWaitMmadL1Event = eventId;
      sync->uselessSync = true;
      return true;
    }
    if (sync->GetDstPipe() == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B) {
      iter->second.L1B2WaitMmadL1Event = eventId;
      sync->uselessSync = true;
      return true;
    }
  }
  return false;
}

void SyncCodegen::CreateSetWaitOpForSingleBuffer(IRRewriter &rewriter,
                                                 Operation *op,
                                                 SyncOperation *sync,
                                                 bool beforeInsert) {
  if (NeedLowerSyncToTemplate(rewriter, op, sync)) {
    return;
  }

  // Set sync insertion position.
  if (beforeInsert) {
    rewriter.setInsertionPoint(op);
  } else {
    rewriter.setInsertionPointAfter(op);
  }
  auto setPipe = PipeAttr::get(func_->getContext(), sync->GetActualSrcPipe());
  auto waitPipe = PipeAttr::get(func_->getContext(), sync->GetActualDstPipe());
  Location loc = op->getLoc();
  checkCondition(sync->eventIds.size() == 1,
                 "sync operation expected to have exactly 1 eventId");
  auto iterId = eventIdMap.find(sync->eventIds[0]);
  checkCondition(iterId != eventIdMap.end(),
                 "iterId expected to be found in eventIdMap");
  auto eventIdAttr = EventAttr::get(func_->getContext(), iterId->second);
  if (sync->GetType() == SyncOperation::TYPE::WAIT_EVENT) {
    rewriter.create<hivm::WaitFlagOp>(loc, setPipe, waitPipe, eventIdAttr,
                                      Value{});
  } else if (sync->GetType() == SyncOperation::TYPE::SET_EVENT) {
    rewriter.create<hivm::SetFlagOp>(loc, setPipe, waitPipe, eventIdAttr,
                                     Value{});
  }
}

void SyncCodegen::CreateSetWaitOpForMultiBuffer(IRRewriter &rewriter,
                                                Operation *op,
                                                SyncOperation *sync,
                                                bool beforeInsert) {
  if (sync->eventIds.size() > 2) {
    llvm_unreachable("Sync supports up to 2 buffers! ");
  }
  Value bufferSelected = GetBufferSelected(rewriter, op, sync);
  if (NeedLowerSyncToTemplate(rewriter, op, sync, bufferSelected)) {
    return;
  }
  // Set sync insertion position.
  if (beforeInsert) {
    rewriter.setInsertionPoint(op);
  } else {
    rewriter.setInsertionPointAfter(op);
  }
  auto setPipe = PipeAttr::get(func_->getContext(), sync->GetActualSrcPipe());
  auto waitPipe = PipeAttr::get(func_->getContext(), sync->GetActualDstPipe());
  Location loc = op->getLoc();
  if (sync->GetType() == SyncOperation::TYPE::WAIT_EVENT) {
    rewriter.create<hivm::WaitFlagOp>(loc, setPipe, waitPipe, EventAttr{},
                                      bufferSelected);
  } else if (sync->GetType() == SyncOperation::TYPE::SET_EVENT) {
    rewriter.create<hivm::SetFlagOp>(loc, setPipe, waitPipe, EventAttr{},
                                     bufferSelected);
  }
}

Value SyncCodegen::GetBufferSelected(IRRewriter &rewriter, Operation *op,
                                     SyncOperation *sync) {
  Value bufferSelected;
  auto it = SyncIndex2SelectBuffer.find(sync->GetSyncIndex());
  if (it != SyncIndex2SelectBuffer.end()) {
    bufferSelected = it->second;
  } else {
    checkCondition(sync->lowestCommonAncestorBuffer != nullptr,
                   "sync operation expected to have lowestCommonAncestorBuffer "
                   "initialized");
    auto *defineOp = sync->lowestCommonAncestorBuffer.getDefiningOp();
    if (!defineOp) {
      llvm_unreachable("defineOp is not defined");
      return nullptr;
    }
    LoopLikeOpInterface parentLoop =
        defineOp->getParentOfType<LoopLikeOpInterface>();
    Value counter;
    auto iter = loop2BufferCounter.find(parentLoop);
    if (iter != loop2BufferCounter.end()) {
      counter = iter->second;
    } else {
      // Construct a ternary expression for select.
      counter = createNestedIndexModular(rewriter, defineOp);
      loop2BufferCounter[parentLoop] = counter;
    }
    // Insert selector after the defined value.
    rewriter.setInsertionPointAfter(counter.getDefiningOp());
    Location locDefineOp = counter.getDefiningOp()->getLoc();
    Value firstID = rewriter.create<arith::ConstantIntOp>(
        locDefineOp, sync->eventIds[0], rewriter.getI64Type());
    Value secondID = rewriter.create<arith::ConstantIntOp>(
        locDefineOp, sync->eventIds[1], rewriter.getI64Type());
    bufferSelected = rewriter.create<arith::SelectOp>(
        locDefineOp, rewriter.getI64Type(), counter, firstID, secondID);
    SyncIndex2SelectBuffer[sync->GetSyncIndex()] = bufferSelected;
  }
  return bufferSelected;
}

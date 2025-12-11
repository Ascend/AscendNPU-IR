//=========- SyncCommon.cpp --sync correlation common code --------------=====//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>
#include <utility>

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

bool SyncOperation::operator==(const SyncOperation &other) const {
  if (this->GetDstPipe() == other.GetDstPipe()) {
    if (this->GetSrcPipe() == other.GetSrcPipe()) {
      if (this->GetType() == other.GetType()) {
        return this->GetEventIDs().empty() && other.GetEventIDs().empty();
      }
    }
  }
  return false;
}

std::string SyncOperation::TypeName(SyncOperation::TYPE t) {
  static std::map<TYPE, std::string> typeNameMap = {
      {TYPE::SET_EVENT, "set_flag"},
      {TYPE::WAIT_EVENT, "wait_flag"},
      {TYPE::PIPE_BARRIER, "pipe_barrier"},
      {TYPE::PIPE_BARRIER_CUBE, "pipe_barrier_cube"},
      {TYPE::PIPE_BARRIER_VECTOR, "pipe_barrier_vector"},
      {TYPE::SYNC_BLOCK_SET, "sync_block_set"},
      {TYPE::SYNC_BLOCK_WAIT, "sync_block_wait"},
      {TYPE::SYNC_BLOCK_ALL, "sync_block_all"}};
  auto typeIt = typeNameMap.find(t);
  if (typeIt != typeNameMap.cend()) {
    return typeIt->second;
  }
  llvm_unreachable("Not supported sync type");
  return "";
}

std::string SyncOperation::GetCoreTypeName(TCoreType t) const {
  static std::map<TCoreType, std::string> coreTypeNameMap = {
      {TCoreType::CUBE, "CUBE"},
      {TCoreType::VECTOR, "VECTOR"},
      {TCoreType::CUBE_OR_VECTOR, "CUBE_OR_VECTOR"},
      {TCoreType::CUBE_AND_VECTOR, "CUBE_AND_VECTOR"}};
  auto typeIt = coreTypeNameMap.find(t);
  if (typeIt != coreTypeNameMap.cend()) {
    return typeIt->second;
  }
  llvm_unreachable("Not supported sync type");
  return "";
}

std::unique_ptr<SyncOperation>
SyncOperation::GetMatchSync(unsigned index) const {
  TYPE newType{TYPE::PIPE_BARRIER};
  static std::map<TYPE, TYPE> syncPair = {
      {TYPE::SET_EVENT, TYPE::WAIT_EVENT},
      {TYPE::WAIT_EVENT, TYPE::SET_EVENT},
      {TYPE::PIPE_BARRIER, TYPE::PIPE_BARRIER},
      {TYPE::SYNC_BLOCK_SET, TYPE::SYNC_BLOCK_WAIT},
      {TYPE::SYNC_BLOCK_WAIT, TYPE::SYNC_BLOCK_SET},
      {TYPE::SYNC_BLOCK_ALL, TYPE::SYNC_BLOCK_ALL},
  };
  auto syncIt = syncPair.find(this->type_);
  if (syncIt != syncPair.cend()) {
    newType = syncIt->second;
  }

  auto res =
      std::make_unique<SyncOperation>(newType, this->srcPipe_, this->dstPipe_,
                                      kSyncIndex_, index, this->forEndIndex_);
  res->eventIds = this->eventIds;
  return res;
}

void SyncOperation::SetPipeAll() {
  // set current sync to pipe_all
  this->type_ = TYPE::PIPE_BARRIER;
  this->srcPipe_ = hivm::PIPE::PIPE_ALL;
  this->dstPipe_ = hivm::PIPE::PIPE_ALL;
}

bool SyncOperation::isSyncSetType() const {
  auto type = this->GetType();
  return type == TYPE::SET_EVENT || type == TYPE::SYNC_BLOCK_SET;
}

bool SyncOperation::isSyncWaitType() const {
  auto type = this->GetType();
  return type == TYPE::WAIT_EVENT || type == TYPE::SYNC_BLOCK_WAIT;
}

bool SyncOperation::isBarrierType() const {
  auto type = this->GetType();
  return type == TYPE::PIPE_BARRIER || type == TYPE::PIPE_BARRIER_CUBE ||
         type == TYPE::PIPE_BARRIER_VECTOR;
}

bool InstanceElement::RemoveSync(SyncOps &syncVector,
                                 const SyncOperation *sync) {
  auto it = std::find(syncVector.begin(), syncVector.end(), sync);
  if (it == syncVector.end()) {
    return false;
  }
  syncVector.erase(it);
  return true;
}

std::unique_ptr<InstanceElement>
LoopInstanceElement::CloneFor(KindOfLoop loopKind) const {
  unsigned index =
      loopKind == KindOfLoop::LOOP_BEGIN ? this->beginId : this->endId;
  checkCondition(this->beginId != this->endId,
                 "LoopInstanceElement clone failed.");
  auto res =
      std::make_unique<LoopInstanceElement>(index, beginId, endId, loopKind);
  res->elementOp = elementOp;
  return res;
}

std::unique_ptr<BranchInstanceElement>
BranchInstanceElement::CloneBranch(KindOfBranch branchKind) const {
  if (branchKind == KindOfBranch::ELSE_BEGIN) {
    auto res = std::make_unique<BranchInstanceElement>(
        branchId, beginId, branchId, endId, KindOfBranch::ELSE_BEGIN);
    res->elementOp = elementOp;
    return res;
  }
  if (branchKind == KindOfBranch::IF_END) {
    auto res = std::make_unique<BranchInstanceElement>(
        endId, beginId, branchId, endId, KindOfBranch::IF_END);
    res->elementOp = elementOp;
    return res;
  }
  checkCondition(branchKind == KindOfBranch::IF_BEGIN,
                 "element expected to be of kind IF_BEGIN");
  auto res = std::make_unique<BranchInstanceElement>(
      beginId, beginId, branchId, endId, KindOfBranch::IF_BEGIN);
  res->elementOp = elementOp;
  return res;
}

std::unique_ptr<PlaceHolderInstanceElement>
PlaceHolderInstanceElement::Clone() const {
  return std::make_unique<PlaceHolderInstanceElement>(this->kIndex,
                                                      this->parentScopeId);
}

bool LoopInstanceElement::classof(const InstanceElement *e) {
  checkCondition(e != nullptr,
                 "give a nullptr for LoopInstanceElement'sconst classof");
  return e->GetKind() == KindTy::LOOP;
}

bool CompoundInstanceElement::classof(const InstanceElement *e) {
  checkCondition(e != nullptr,
                 "give a nullptr for CompoundInstanceElement's classof");
  return e->GetKind() == KindTy::COMPOUND;
}

bool BranchInstanceElement::classof(const InstanceElement *e) {
  checkCondition(e != nullptr,
                 "give a nullptr for BranchInstanceElement's classof");
  return e->GetKind() == KindTy::BRANCH;
}

bool PlaceHolderInstanceElement::classof(const InstanceElement *e) {
  checkCondition(e != nullptr,
                 "give a nullptr for PlaceHolderInstanceElement's classof");
  return e->GetKind() == KindTy::PLACE_HOLDER;
}

UNIT_FLAG CompoundInstanceElement::getUnitFlagMode() const {
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
      std::make_pair(unitFlagModeAsSet, unitFlagModeAsWait));
  if (it == possibleStates.end()) {
    llvm_unreachable("unit-flag state not handled");
  }
  return it->second;
}

Value getIsNotDeadLoopValue(scf::ForOp forOp, Location loc,
                            IRRewriter &rewriter) {
  Value upperBound = forOp.getUpperBound();
  Value lowerBound = forOp.getLowerBound();
  return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                        lowerBound, upperBound);
}

std::optional<mlir::Value>
CompoundInstanceElement::getUnitFlagCond(Location loc, IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<Value> conditions;
  if (linkedUnitFlagCompAsWait &&
      (linkedUnitFlagCompAsWait->unitFlagModeAsSet ==
           UNIT_FLAG::ENABLED_ONLY_LAST_ITER ||
       linkedUnitFlagCompAsWait->unitFlagModeAsSet ==
           UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS)) {
    if (auto forOp = dyn_cast<scf::ForOp>(
            linkedUnitFlagCompAsWait->elementOp->getParentOp())) {
      rewriter.setInsertionPoint(forOp);
      Value cond = getIsNotDeadLoopValue(forOp, loc, rewriter);
      conditions.push_back(cond);
    }
  }
  if (linkedUnitFlagCompAsSet &&
      (linkedUnitFlagCompAsSet->unitFlagModeAsWait ==
           UNIT_FLAG::ENABLED_ONLY_FIRST_ITER ||
       linkedUnitFlagCompAsSet->unitFlagModeAsWait ==
           UNIT_FLAG::ENABLED_ONLY_FIRST_AND_LAST_ITERS)) {
    if (auto forOp = dyn_cast<scf::ForOp>(
            linkedUnitFlagCompAsSet->elementOp->getParentOp())) {
      rewriter.setInsertionPoint(elementOp);
      Value cond = getIsNotDeadLoopValue(forOp, loc, rewriter);
      conditions.push_back(cond);
    }
  }

  if (conditions.empty()) {
    return nullptr;
  } else if (conditions.size() == 1) {
    return conditions[0];
  } else if (conditions.size() == 2) {
    rewriter.setInsertionPoint(elementOp);
    return rewriter.create<arith::OrIOp>(loc, conditions[0], conditions[1]);
  } else {
    llvm_unreachable("unexpected/unhandled number of unit-flag conditions.");
  }
}

namespace mlir::hivm {

bool checkAllParentLoopsAreForLoops(Operation *op) {
  while ((op = op->getParentOfType<LoopLikeOpInterface>())) {
    if (!isa<scf::ForOp>(op)) {
      return false;
    }
  }
  return true;
}

void checkSyncIRIndex(const SyncIRs &syncIR, int index) {
  if (index < 0 || index >= static_cast<int>(syncIR.size())) {
    llvm_unreachable("index out of bounds when accessing syncIR");
  }
}

void checkCondition(bool condition, const std::string &message) {
  if (!condition) {
    llvm_unreachable(message.c_str());
  }
}

} // namespace mlir::hivm
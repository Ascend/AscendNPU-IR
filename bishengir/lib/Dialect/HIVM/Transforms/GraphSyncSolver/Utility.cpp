//===------------- Utility.cpp ---- Graph Sync Solver ---------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace hivm::syncsolver;

int ConflictPair::globalIdCounter = 0;
int EventIdNode::globalIdCounter = 0;

bool Occurrence::sameScope(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1->parentOcc != nullptr);
  assert(occ2->parentOcc != nullptr);
  return occ1->parentOcc == occ2->parentOcc;
}

int Occurrence::getDepth(Occurrence *occ) {
  int ret = 0;
  while (occ != nullptr) {
    occ = occ->parentOcc;
    ret++;
  }
  return ret;
}

Occurrence *Occurrence::getParentWithOp(OperationBase *op, bool assertExists) {
  assert(op != nullptr);
  Occurrence *occ = this;
  while (occ != nullptr) {
    if (occ->op == op) {
      return occ;
    }
    occ = occ->parentOcc;
  }
  assert(!assertExists);
  return nullptr;
}

Occurrence *Occurrence::getParentWithOp(Operation *op, bool assertExists) {
  assert(op != nullptr);
  Occurrence *occ = this;
  while (occ != nullptr) {
    if (occ->op != nullptr && occ->op->op == op) {
      return occ;
    }
    occ = occ->parentOcc;
  }
  assert(!assertExists);
  return nullptr;
}

Occurrence *Occurrence::getNthParent(int dist) {
  Occurrence *occ = this;
  while (dist--) {
    assert(occ != nullptr);
    occ = occ->parentOcc;
  }
  assert(occ != nullptr);
  return occ;
}

std::pair<Occurrence *, Occurrence *> Occurrence::getLCAPair(Occurrence *occ1,
                                                             Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  int depth1 = getDepth(occ1);
  int depth2 = getDepth(occ2);
  if (depth1 < depth2) {
    occ2 = occ2->getNthParent(depth2 - depth1);
  } else if (depth1 > depth2) {
    occ1 = occ1->getNthParent(depth1 - depth2);
  }
  while (occ1->parentOcc != occ2->parentOcc) {
    occ1 = occ1->parentOcc;
    occ2 = occ2->parentOcc;
  }
  assert(occ1 != occ2);
  return std::make_pair(occ1, occ2);
}

Occurrence *Occurrence::getParentloop(Occurrence *occ) {
  assert(occ != nullptr);
  Occurrence *cur = occ->parentOcc;
  while (cur != nullptr && !isa<Loop>(cur->op)) {
    cur = cur->parentOcc;
  }
  return cur;
}

Occurrence *Occurrence::getParentCondition(Occurrence *occ) {
  assert(occ != nullptr);
  Occurrence *cur = occ->parentOcc;
  while (cur != nullptr && !isa<Condition>(cur->op)) {
    cur = cur->parentOcc;
  }
  return cur;
}

Occurrence *Occurrence::getUnlikelyParentCondition(Occurrence *occ) {
  assert(occ != nullptr && occ->op != nullptr);
  if (auto *parentConditionOp =
          OperationBase::getUnlikelyParentCondition(occ->op)) {
    return occ->getParentWithOp(parentConditionOp, /*assertExists=*/true);
  }
  return nullptr;
}

bool Occurrence::isProperAncestor(Occurrence *occ) {
  int depth1 = getDepth(this);
  int depth2 = getDepth(occ);
  if (depth1 >= depth2) {
    return false;
  }
  return occ->getNthParent(depth2 - depth1) == this;
}

llvm::SmallVector<Occurrence *> Occurrence::getAllParents() {
  llvm::SmallVector<Occurrence *> collectedParents;
  Occurrence *occ = this->parentOcc;
  while (occ != nullptr) {
    collectedParents.push_back(occ);
    occ = occ->parentOcc;
  }
  return collectedParents;
}

llvm::SmallVector<OperationBase *> OperationBase::getAllParents() {
  llvm::SmallVector<OperationBase *> collectedParents;
  OperationBase *op = this->parentOp;
  while (op != nullptr) {
    collectedParents.push_back(op);
    op = op->parentOp;
  }
  return collectedParents;
}

bool OperationBase::sameScope(OperationBase *op1, OperationBase *op2) {
  assert(op1->parentOp != nullptr);
  assert(op2->parentOp != nullptr);
  return op1->parentOp == op2->parentOp;
}

int OperationBase::getDepth(OperationBase *op) {
  int ret = 0;
  while (op != nullptr) {
    op = op->parentOp;
    ret++;
  }
  return ret;
}

OperationBase *OperationBase::getNthParent(int dist) {
  OperationBase *op = this;
  while (dist--) {
    assert(op != nullptr);
    op = op->parentOp;
  }
  return op;
}

std::pair<OperationBase *, OperationBase *>
OperationBase::getLCAPair(OperationBase *op1, OperationBase *op2) {
  assert(op1 != nullptr && op2 != nullptr);
  int depth1 = getDepth(op1);
  int depth2 = getDepth(op2);
  if (depth1 < depth2) {
    op2 = op2->getNthParent(depth2 - depth1);
  } else if (depth1 > depth2) {
    op1 = op1->getNthParent(depth1 - depth2);
  }
  while (op1->parentOp != op2->parentOp) {
    op1 = op1->parentOp;
    op2 = op2->parentOp;
  }
  return std::make_pair(op1, op2);
}

OperationBase *OperationBase::getParentloop(OperationBase *op) {
  assert(op != nullptr);
  OperationBase *cur = op->parentOp;
  while (cur != nullptr && !isa<Loop>(cur)) {
    cur = cur->parentOp;
  }
  return cur;
}

OperationBase *OperationBase::getParentCondition(OperationBase *op) {
  assert(op != nullptr);
  OperationBase *cur = op->parentOp;
  while (cur != nullptr && !isa<Condition>(cur)) {
    cur = cur->parentOp;
  }
  return cur;
}

bool OperationBase::isProperAncestor(OperationBase *op) {
  int depth1 = getDepth(this);
  int depth2 = getDepth(op);
  if (depth1 >= depth2) {
    return false;
  }
  return op->getNthParent(depth2 - depth1) == this;
}

OperationBase *OperationBase::getUnlikelyParentCondition(OperationBase *op) {
  auto *cur = OperationBase::getParentCondition(op);
  while (cur != nullptr) {
    auto *conditionOp = dyn_cast<Condition>(cur);
    assert(conditionOp != nullptr);
    if (conditionOp->isUnlikely &&
        conditionOp->getTrueScope()->isProperAncestor(op)) {
      return cur;
    }
    cur = OperationBase::getParentCondition(cur);
  }
  return nullptr;
}

namespace mlir::hivm::syncsolver {

// Check if two integer ranges intersect (half-open semantics: [l, r) )
bool checkRangesIntersect(int l1, int r1, int l2, int r2) {
  // return !(r1 <= l2 || r2 <= l1);
  return r1 > l2 && r2 > l1;
}

// Return explicit integer ranges covered by a conflict pair (barrier -> empty).
std::vector<std::pair<int, int>> getRanges(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  if (conflictPair->isBarrier()) {
    return {};
  }
  std::vector<std::pair<int, int>> ret;
  ret.emplace_back(conflictPair->startIndex, conflictPair->endIndex);
  return ret;
}

// Return the hardware-available EVENT ids for a given (setPipe, waitPipe) pair.
// Respects reserved ids for special pipe pairs and returns a vector of usable
// ids.
int64_t getHWAvailableEventIdNum(SyncMode syncMode, hivm::PIPE setPipe,
                                 hivm::PIPE waitPipe) {
  if (syncMode == SyncMode::INTRA_CORE_SYNC) {
    const llvm::DenseMap<std::tuple<PIPE, PIPE>, int64_t> reservedEventIdNum = {
        {{hivm::PIPE::PIPE_V, hivm::PIPE::PIPE_S}, 1},
        {{hivm::PIPE::PIPE_S, hivm::PIPE::PIPE_V}, 1},
        {{hivm::PIPE::PIPE_MTE2, hivm::PIPE::PIPE_V}, 1},
        {{hivm::PIPE::PIPE_M, hivm::PIPE::PIPE_FIX}, 1},
        {{hivm::PIPE::PIPE_FIX, hivm::PIPE::PIPE_M}, 1},
    };
    int64_t eventIdNum = INTRA_CORE_EVENT_ID_NUM;
    auto it = reservedEventIdNum.find({setPipe, waitPipe});
    if (it != reservedEventIdNum.end()) {
      eventIdNum -= it->second;
    }
    return eventIdNum;
  } else if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    int64_t eventIdNum = CROSS_CORE_EVENT_ID_NUM;
    return eventIdNum;
  } else if (syncMode == SyncMode::TEST_INTRA_CORE_MODE) {
    int64_t eventIdNum = TEST_INTRA_CORE_EVENT_ID_NUM;
    return eventIdNum;
  } else if (syncMode == SyncMode::TEST_CROSS_CORE_MODE) {
    int64_t eventIdNum = TEST_CROSS_CORE_EVENT_ID_NUM;
    return eventIdNum;
  }
  llvm_unreachable("getHWAvailableEventIdNum: unhandled SyncMode");
}

SmallVector<int64_t> getHWAvailableEventIds(SyncMode syncMode,
                                            hivm::PIPE setPipe,
                                            hivm::PIPE waitPipe) {
  if (syncMode == SyncMode::INTRA_CORE_SYNC) {
    const llvm::DenseMap<std::tuple<PIPE, PIPE>, int64_t> reservedEventIdNum = {
        {{hivm::PIPE::PIPE_V, hivm::PIPE::PIPE_S}, 1},
        {{hivm::PIPE::PIPE_S, hivm::PIPE::PIPE_V}, 1},
        {{hivm::PIPE::PIPE_MTE2, hivm::PIPE::PIPE_V}, 1},
        {{hivm::PIPE::PIPE_M, hivm::PIPE::PIPE_FIX}, 1},
        {{hivm::PIPE::PIPE_FIX, hivm::PIPE::PIPE_M}, 1},
    };
    int64_t eventIdNum = INTRA_CORE_EVENT_ID_NUM;
    auto it = reservedEventIdNum.find({setPipe, waitPipe});
    if (it != reservedEventIdNum.end()) {
      eventIdNum -= it->second;
    }
    SmallVector<int64_t> hwAvailableEventIds(eventIdNum);
    std::iota(hwAvailableEventIds.begin(), hwAvailableEventIds.end(),
              static_cast<int64_t>(0));
    return hwAvailableEventIds;
  } else if (syncMode == SyncMode::CROSS_CORE_SYNC) {
    int64_t eventIdNum = CROSS_CORE_EVENT_ID_NUM;
    SmallVector<int64_t> hwAvailableEventIds(eventIdNum);
    std::iota(hwAvailableEventIds.begin(), hwAvailableEventIds.end(),
              static_cast<int64_t>(0));
    return hwAvailableEventIds;
  } else if (syncMode == SyncMode::TEST_INTRA_CORE_MODE) {
    int64_t eventIdNum = TEST_INTRA_CORE_EVENT_ID_NUM;
    SmallVector<int64_t> availableEventIds(eventIdNum);
    std::iota(availableEventIds.begin(), availableEventIds.end(),
              static_cast<int64_t>(0));
    return availableEventIds;
  } else if (syncMode == SyncMode::TEST_CROSS_CORE_MODE) {
    int64_t eventIdNum = TEST_CROSS_CORE_EVENT_ID_NUM;
    SmallVector<int64_t> availableEventIds(eventIdNum);
    std::iota(availableEventIds.begin(), availableEventIds.end(),
              static_cast<int64_t>(0));
    return availableEventIds;
  }
  llvm_unreachable("getHWAvailableEventIds: unhandled SyncMode");
}

// Build a Value that is true for the first iteration of the given scf::ForOp.
// Inserted at the start of the loop body and compares induction var with lower.
Value getIsFirstIterationValue(scf::ForOp forOp, Location loc,
                               IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value lowerBound = forOp.getLowerBound();
  Value currentInd = forOp.getInductionVar();
  Value isFirstIter = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lowerBound, currentInd);
  return isFirstIter;
}

// Build a Value that is true for the last iteration of the given scf::ForOp.
// Compares next induction value with the upper bound.
Value getIsLastIterationValue(scf::ForOp forOp, Location loc,
                              IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value currentInd = forOp.getInductionVar();
  Value nextInd = rewriter.create<arith::AddIOp>(loc, currentInd, step);
  Value isLastIter = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, nextInd, upperBound);
  return isLastIter;
}

// Convert a Value to its string representation for debugging/logging.
std::string op2str(Value val) {
  std::string printBuffer;
  llvm::raw_string_ostream os(printBuffer);
  val.print(os);
  return os.str();
}

// Convert an Operation pointer to its string representation.
std::string op2str(Operation *op) {
  std::string printBuffer;
  llvm::raw_string_ostream os(printBuffer);
  op->print(os);
  return os.str();
}

// Verify that all loop-like parents of `op` are SCF ForOps. Used to ensure
// certain multi-buffer/loop transformations are safe to apply.
bool checkAllParentLoopsAreForLoops(Operation *op) {
  while (op != nullptr) {
    auto parLoop = op->getParentOfType<LoopLikeOpInterface>();
    if (parLoop != nullptr && !isa<scf::ForOp>(parLoop)) {
      return false;
    }
    op = parLoop;
  }
  return true;
}

Value getValueOrCreateCastToI64(IRRewriter &rewriter, Location loc, Value val) {
  assert(isa<OpResult>(val));
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(val);
  if (!val.getType().isInteger(64)) {
    if (val.getType().isIndex()) {
      val = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIntegerType(64), val);
    } else if (val.getType().isInteger()) {
      val = rewriter.create<arith::ExtSIOp>(loc, rewriter.getIntegerType(64),
                                            val);
    } else {
      llvm_unreachable("unhandled casting type");
    }
  }
  return val;
}

hivm::TCoreType getOppositeCoreType(hivm::TCoreType coreType) {
  switch (coreType) {
  case hivm::TCoreType::CUBE:
    return hivm::TCoreType::VECTOR;
  case hivm::TCoreType::VECTOR:
    return hivm::TCoreType::CUBE;
  case hivm::TCoreType::CUBE_OR_VECTOR:
    return hivm::TCoreType::CUBE_OR_VECTOR;
  case hivm::TCoreType::CUBE_AND_VECTOR:
    return hivm::TCoreType::CUBE_AND_VECTOR;
  }
}

std::pair<Occurrence *, Occurrence *> getLCAPairOcc(Occurrence *occ1,
                                                    Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  assert(parOp1 != nullptr && parOp2 != nullptr);
  assert(parOp1->parentOp != nullptr && parOp2->parentOp != nullptr);
  assert(parOp1->parentOp == parOp2->parentOp);
  auto *parOcc1 = occ1->getParentWithOp(parOp1->parentOp);
  auto *parOcc2 = occ2->getParentWithOp(parOp2->parentOp);
  assert(parOcc1 != nullptr && parOcc2 != nullptr);
  assert(parOcc1 != occ1 && parOcc2 != occ2);
  auto *setOcc = occ1->getNthParent(occ1->depth - parOcc1->depth - 1);
  auto *waitOcc = occ2->getNthParent(occ2->depth - parOcc2->depth - 1);
  assert(setOcc != nullptr && waitOcc != nullptr &&
         setOcc->parentOcc != nullptr && waitOcc->parentOcc != nullptr);
  return std::make_pair(setOcc, waitOcc);
}

bool isEmptyScope(Scope *scope) {
  for (auto &childOp : scope->body) {
    if (isa<RWOperation>(childOp.get())) {
      return false;
    }
    if (auto *childScope = dyn_cast<Scope>(childOp.get())) {
      if (!isEmptyScope(childScope)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace mlir::hivm::syncsolver

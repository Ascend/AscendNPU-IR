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
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include <utility>
#include <vector>

using namespace mlir;
using namespace hivm::syncsolver;

int ConflictPair::globalDebugIdCounter = 0;

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

Occurrence *Occurrence::getParentWithOp(OperationBase *op) {
  assert(op != nullptr);
  Occurrence *occ = this;
  while (occ->op != nullptr && occ->op != op && occ->parentOcc != nullptr) {
    occ = occ->parentOcc;
  }
  assert(occ->op == op);
  return occ;
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

llvm::DenseMap<std::pair<Occurrence *, Occurrence *>,
               std::pair<Occurrence *, Occurrence *>>
    Occurrence::getLCAOccMem;
std::pair<Occurrence *, Occurrence *> Occurrence::getLCAPair(Occurrence *occ1,
                                                             Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto [it, inserted] = getLCAOccMem.insert({{occ1, occ2}, {nullptr, nullptr}});
  if (!inserted) {
    return it->second;
  }
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
  return it->second = std::make_pair(occ1, occ2);
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

bool Occurrence::isProperAncestor(Occurrence *occ) {
  int depth1 = getDepth(this);
  int depth2 = getDepth(occ);
  if (depth1 >= depth2) {
    return false;
  }
  return occ->getNthParent(depth2 - depth1) == this;
}

std::vector<Occurrence *> Occurrence::getAllParents() {
  std::vector<Occurrence *> collectedParents;
  Occurrence *occ = this->parentOcc;
  while (occ != nullptr) {
    collectedParents.push_back(occ);
    occ = occ->parentOcc;
  }
  return collectedParents;
}

std::vector<OperationBase *> OperationBase::getAllParents() {
  std::vector<OperationBase *> collectedParents;
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

llvm::DenseMap<std::pair<OperationBase *, OperationBase *>,
               std::pair<OperationBase *, OperationBase *>>
    OperationBase::getLCAOpMem;
std::pair<OperationBase *, OperationBase *>
OperationBase::getLCAPair(OperationBase *op1, OperationBase *op2) {
  assert(op1 != nullptr && op2 != nullptr);
  auto [it, inserted] = getLCAOpMem.insert({{op1, op2}, {nullptr, nullptr}});
  if (!inserted) {
    return it->second;
  }
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
  // assert(op1 != op2);
  return it->second = std::make_pair(op1, op2);
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

namespace mlir::hivm::syncsolver {

// Check if two integer ranges intersect (half-open semantics: [l, r) )
bool checkIntersect(int l1, int r1, int l2, int r2) {
  // return !(r1 <= l2 || r2 <= l1);
  return r1 > l2 && r2 > l1;
}

// Check whether two ConflictPair entries conflict in pipe and time ranges.
bool checkIntersect(ConflictPair *conflictPair1, ConflictPair *conflictPair2) {
  assert(conflictPair1 != nullptr && conflictPair2 != nullptr);
  if (conflictPair1->setPipe != conflictPair2->setPipe ||
      conflictPair1->waitPipe != conflictPair2->waitPipe) {
    return false;
  }
  for (auto [l1, r1] : getRanges(conflictPair1)) {
    for (auto [l2, r2] : getRanges(conflictPair2)) {
      if (checkIntersect(l1, r1 + 1, l2, r2 + 1)) {
        return true;
      }
    }
  }
  return false;
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
SmallVector<hivm::EVENT> getHWAvailableEventIds(hivm::PIPE setPipe,
                                                hivm::PIPE waitPipe) {
  const llvm::DenseMap<std::pair<hivm::PIPE, hivm::PIPE>, uint64_t>
      reservedEventIdNum = {
          {{hivm::PIPE::PIPE_V, hivm::PIPE::PIPE_S}, 1},
          {{hivm::PIPE::PIPE_S, hivm::PIPE::PIPE_V}, 1},
          {{hivm::PIPE::PIPE_MTE2, hivm::PIPE::PIPE_V}, 1},
      };
  uint64_t eventIdNum = NORM_EVENT_ID_NUM;
  auto it = reservedEventIdNum.find({setPipe, waitPipe});
  if (it != reservedEventIdNum.end()) {
    eventIdNum -= it->second;
  }
  SmallVector<hivm::EVENT> hwAvailableEventIds;
  for (uint64_t i = 0; i < eventIdNum; i++) {
    hivm::EVENT eventId = static_cast<hivm::EVENT>(i);
    hwAvailableEventIds.push_back(eventId);
  }
  return hwAvailableEventIds;
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
bool checkAllLoopParentsAreForLoops(Operation *op) {
  while (op != nullptr) {
    auto parLoop = op->getParentOfType<LoopLikeOpInterface>();
    if (parLoop != nullptr && !isa<scf::ForOp>(parLoop)) {
      return false;
    }
    op = parLoop;
  }
  return true;
}

} // namespace mlir::hivm::syncsolver

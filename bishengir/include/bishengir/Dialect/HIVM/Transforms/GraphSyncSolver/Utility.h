//===------------- Utility.h ---- Graph Sync Solver -----------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_UTILITY_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_UTILITY_H

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <memory>

// #define NORM_EVENT_ID_NUM (uint8_t)4
#define NORM_EVENT_ID_NUM (uint8_t)8

namespace mlir::hivm::syncsolver {

struct Occurrence {
  OperationBase *op{nullptr};
  Occurrence *parentOcc{nullptr};
  int depth{-1};
  int startIndex{-1};
  int endIndex{-1};
  int syncIrIndex{-1};
  int syncIrEndIndex{-1};
  int loopSplitIndex{-1};

  Occurrence(OperationBase *op, Occurrence *parentOcc, int depth,
             int startIndex, int endIdx)
      : op(op), parentOcc(parentOcc), depth(depth), startIndex(startIndex),
        endIndex(endIdx) {}

  // Return true if occ1 and occ2 have the same immediate parent occurrence.
  static bool sameScope(Occurrence *occ1, Occurrence *occ2);

  // Return depth (number of ancestors + 1) for the given occurrence.
  static int getDepth(Occurrence *occ);

  // Walk up parents to find the first ancestor occurrence associated with 'op'.
  Occurrence *getParentWithOp(OperationBase *op);

  // Return the ancestor that is `dist` levels above this occurrence.
  Occurrence *getNthParent(int dist);

  static llvm::DenseMap<std::pair<Occurrence *, Occurrence *>,
                        std::pair<Occurrence *, Occurrence *>>
      getLCAOccMem;

  static void resetLCAMem() { getLCAOccMem.clear(); }

  // Compute/return the pair of sibling occurrences just below their LCA.
  static std::pair<Occurrence *, Occurrence *> getLCAPair(Occurrence *occ1,
                                                          Occurrence *occ2);

  // Find and return the nearest parent occurrence that is a loop.
  static Occurrence *getParentloop(Occurrence *occ);

  // Find and return the nearest parent occurrence that is a condition.
  static Occurrence *getParentCondition(Occurrence *occ);

  // Return true if this occurrence is a strict ancestor of `occ`.
  bool isProperAncestor(Occurrence *occ);

  // Collect and return all occurrence parents (in upward order).
  std::vector<Occurrence *> getAllParents();
};

struct ConflictPair {

  static int globalDebugIdCounter;

  int debugId{-1};
  OperationBase *op1{nullptr};
  OperationBase *op2{nullptr};
  OperationBase *opSet{nullptr};
  OperationBase *opWait{nullptr};
  hivm::PIPE setPipe{hivm::PIPE::PIPE_UNASSIGNED};
  hivm::PIPE waitPipe{hivm::PIPE::PIPE_UNASSIGNED};
  int startIndex{-1};
  int endIndex{-1};
  llvm::SmallVector<hivm::EVENT> eventIds;
  bool isInnerBackward{false};
  bool isUseless{false};
  bool couldNotRun{false};
  LoopLikeOpInterface multibufferLoopPar{nullptr};
  bool setOnLastIterOnly{false};
  bool waitOnFirstIterOnly{false};
  bool replacedWithUnitFlag{false};

  ConflictPair(OperationBase *op1, OperationBase *op2, OperationBase *opSet,
               OperationBase *opWait, hivm::PIPE setPipe, hivm::PIPE waitPipe,
               int startIndex, int endIndex)
      : op1(op1), op2(op2), opSet(opSet), opWait(opWait), setPipe(setPipe),
        waitPipe(waitPipe), startIndex(startIndex), endIndex(endIndex) {
    debugId = globalDebugIdCounter++;
    if (setPipe == waitPipe) {
      this->opSet = opWait;
      this->startIndex = endIndex;
    }
  };

  bool isBarrier() const { return setPipe == waitPipe; }

  // Human-readable description of the conflict pair for debug printing.
  std::string str() const;

  // Update the stored set/wait operation pointers and their indices from
  // occurrences.
  void updateSetWaitOps(Occurrence *setOcc, Occurrence *waitOcc) {
    if (setOcc != nullptr) {
      opSet = setOcc->op;
      startIndex = setOcc->endIndex;
    }
    if (waitOcc != nullptr) {
      opWait = waitOcc->op;
      endIndex = waitOcc->startIndex;
    }
  }

  std::unique_ptr<ConflictPair> clone() {
    auto clonedConflictPair = std::make_unique<ConflictPair>(
        op1, op2, opSet, opWait, setPipe, waitPipe, startIndex, endIndex);
    clonedConflictPair->eventIds = eventIds;
    clonedConflictPair->isInnerBackward = isInnerBackward;
    clonedConflictPair->isUseless = isUseless;
    clonedConflictPair->multibufferLoopPar = multibufferLoopPar;
    clonedConflictPair->setOnLastIterOnly = setOnLastIterOnly;
    clonedConflictPair->waitOnFirstIterOnly = waitOnFirstIterOnly;
    return clonedConflictPair;
  }

  std::unique_ptr<ConflictPair> clone(uint32_t startIndex, uint32_t endIndex) {
    auto clonedConflictPair = this->clone();
    clonedConflictPair->startIndex = static_cast<int>(startIndex);
    clonedConflictPair->endIndex = static_cast<int>(endIndex);
    return clonedConflictPair;
  }
};

struct MmadL1SyncArgs {
  MmadL1SyncArgs() = default;
  MmadL1SyncArgs(Value L0WaitL1AEvent, Value L0WaitL1BEvent,
                 Value L1AWaitL0Event, Value L1BWaitL0Event, Value KLoopDBCond,
                 Value BackPipeMPipeMTE1Event0, Value BackPipeMPipeMTE1Event1)
      : L0WaitL1AEvent(L0WaitL1AEvent), L0WaitL1BEvent(L0WaitL1BEvent),
        L1AWaitL0Event(L1AWaitL0Event), L1BWaitL0Event(L1BWaitL0Event),
        KLoopDBCond(KLoopDBCond),
        BackPipeMPipeMTE1Event0(BackPipeMPipeMTE1Event0),
        BackPipeMPipeMTE1Event1(BackPipeMPipeMTE1Event1) {}

  Value L0WaitL1AEvent;
  Value L0WaitL1BEvent;
  Value L1AWaitL0Event;
  Value L1BWaitL0Event;
  Value KLoopDBCond;
  Value BackPipeMPipeMTE1Event0;
  Value BackPipeMPipeMTE1Event1;
};

// Check if two integer ranges intersect.
bool checkIntersect(int l1, int r1, int l2, int r2);

// Check whether two ConflictPair ranges/event mapping intersect (same
// pipes/events).
bool checkIntersect(ConflictPair *conflictPair1, ConflictPair *conflictPair2);

// Return explicit integer ranges covered by a conflict pair (empty for
// barrier).
std::vector<std::pair<int, int>> getRanges(ConflictPair *conflictPair);

// Return hardware-available EVENT ids for a given (setPipe, waitPipe) pair.
llvm::SmallVector<hivm::EVENT> getHWAvailableEventIds(hivm::PIPE setPipe,
                                                      hivm::PIPE waitPipe);

// Create a boolean Value that is true for the first iteration of `forOp`.
Value getIsFirstIterationValue(scf::ForOp forOp, Location loc,
                               IRRewriter &rewriter);

// Create a boolean Value that is true for the last iteration of `forOp`.
Value getIsLastIterationValue(scf::ForOp forOp, Location loc,
                              IRRewriter &rewriter);

// Helper to stringify a Value to std::string for logging.
std::string op2str(Value val);

// Helper to stringify an Operation pointer to std::string for logging.
std::string op2str(Operation *op);

// Verify that all loop-like parents of `op` are SCF ForOps (returns true if
// so).
bool checkAllLoopParentsAreForLoops(Operation *op);

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_UTILITY_H

//===- FusibleBlock.cpp - Fusible block contains fusible ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlock.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include <algorithm>
#include <queue>

#define DEBUG_TYPE "hfusion-fuse"

namespace mlir {
namespace hfusion {
namespace opfusion {

void FusibleBlock::dump() {
  llvm::dbgs() << "FusibleBlock {\n";
  llvm::dbgs() << "ins:\n";
  for (Value val : getInputs()) {
    val.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "outs:\n";
  for (Value val : getOutputs()) {
    val.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "Ops:\n";
  for (Operation *op : getOps()) {
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "OpWithAuxs:\n";
  for (Operation *op : getOpWithAuxs()) {
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "}\n";
}

void FusibleBlock::visitOutValues() {
  if (opWithAuxs_.empty())
    visitAuxiliaryOps();
  SetVector<Operation *> observedOuts =
      outsModification_.empty() ? ops_ : outsModification_;
  std::queue<Operation *> operationQueue;
  DenseSet<Operation *> visited;
  for (Operation *op : observedOuts) {
    visited.insert(op);
    operationQueue.push(op);
  }

  while (!operationQueue.empty()) {
    Operation *curOp = operationQueue.front();
    operationQueue.pop();
    for (const Value &res : curOp->getResults()) {
      if (llvm::any_of(res.getUsers(), [&](Operation *user) {
            return !opWithAuxs_.contains(user);
          }))
        outs_.insert(res);
    }
    for (Operation *user : curOp->getUsers()) {
      if (visited.contains(user))
        continue;
      if (opWithAuxs_.contains(user)) {
        visited.insert(user);
        operationQueue.push(user);
      }
    }
  }
  fillNonEdgeOps();
  assert(!outs_.empty());
}

void FusibleBlock::fillNonEdgeOps() {
  for (Operation *op : ops_) {
    for (const Value &res : op->getResults()) {
      if (!outs_.count(res)) {
        nonEdgeOps_.insert(res.getDefiningOp());
      }
    }
  }
}

void FusibleBlock::processOperandForBFS(const Value &operand,
                                        Operation *pivotOp,
                                        DenseSet<Operation *> &visited,
                                        std::queue<Operation *> &workQueue,
                                        const InclusionCheck &shouldInclude,
                                        const DenseSet<Operation *> &blocker) {
  if (Operation *defOp = operand.getDefiningOp()) {
    LLVM_DEBUG(llvm::dbgs() << "Checking defOp: " << *defOp << "\n";);
    if (!visited.contains(defOp) && shouldInclude(defOp, pivotOp) &&
        !blocker.contains(defOp)) {
      visited.insert(defOp);
      workQueue.push(defOp);
    }
  }
}

void FusibleBlock::auxBFS(const SetVector<Operation *> &initialOps,
                          DenseSet<Operation *> &visited,
                          const InclusionCheck &shouldInclude,
                          const DenseSet<Operation *> &blocker) {
  std::queue<Operation *> workQueue;
  for (Operation *op : initialOps) {
    workQueue.push(op);
    visited.insert(op);
  }

  while (!workQueue.empty()) {
    Operation *op = workQueue.front();
    workQueue.pop();
    LLVM_DEBUG(llvm::dbgs() << "Relaxing " << *op << "\n";);

    // Process direct operands
    for (const Value &operand : op->getOperands()) {
      processOperandForBFS(operand, op, visited, workQueue, shouldInclude,
                           blocker);
    }

    // Process region operands
    for (Region &region : op->getRegions()) {
      region.walk([&](Operation *regionOp) {
        for (const Value &operand : regionOp->getOperands()) {
          processOperandForBFS(operand, op, visited, workQueue, shouldInclude,
                               blocker);
        }
      });
    }
  }
}

void FusibleBlock::auxBFSDown(const SetVector<Operation *> &initialOps,
                              DenseSet<Operation *> &visited,
                              const InclusionCheck &shouldInclude,
                              const DenseSet<Operation *> &blocker) {
  std::queue<Operation *> workQueue;
  for (Operation *op : initialOps) {
    workQueue.push(op);
    visited.insert(op);
  }

  while (!workQueue.empty()) {
    Operation *op = workQueue.front();
    workQueue.pop();
    LLVM_DEBUG(llvm::dbgs() << "Relaxing " << *op << "\n";);

    // Process direct operands
    for (Operation *nextOp : op->getUsers()) {
      if (visited.contains(nextOp) || blocker.contains(nextOp) ||
          !shouldInclude(nextOp, op))
        continue;

      // don't fuse if there is an operand dependency with the infusible
      // operations. example,
      // A --> B
      // |     |
      // V     |
      // C <- /
      if (llvm::any_of(nextOp->getOperands(), [&](Value opr) {
            Operation *defOp = opr.getDefiningOp();
            if (defOp == nullptr)
              return false;
            if (fusibleHelper_->isBuffer(defOp))
              return false;
            return !visited.contains(defOp) || blocker.contains(defOp);
          }))
        continue;

      visited.insert(nextOp);
      workQueue.push(nextOp);
    }
  }
}

void FusibleBlock::visitAuxiliaryOps() {
  assert(opWithAuxs_.empty());

  // First BFS: Process edge operations
  SetVector<Operation *> edgeOps;
  for (Operation *op : ops_) {
    for (Operation *user : op->getUsers()) {
      if (!ops_.contains(user)) {
        edgeOps.insert(op);
      }
    }
  }

  // Define inclusion check function for buffer
  auto auxBufferCheck = [this](Operation *defOp, Operation *pivotOp) {
    return isValidAuxOrBuffer(defOp, pivotOp);
  };

  auto blacklistedAux = [](Operation *defOp, Operation *pivotOp) {
    return defOp->getParentOp() == pivotOp->getParentOp() &&
           !FusibleHelper::isImportantPattern(defOp) &&
           !reshape_utils::isUnsupportedOp(defOp) && !isa<func::CallOp>(defOp);
  };

  auto nextAuxCheck = [](Operation *nextOp, Operation *curOp) {
    return FusibleHelper::isZeroRankElemwise(nextOp);
  };

  // Add auxiliary ops based on moveOutToParam
  if (fusibleHelper_->moveOutToParam()) {
    // Only add operations that weren't visited in edge BFS
    // This is to prioritize the out's "buffer" shouldnt be fused
    // This needs to be adjusted if the priority of buffer of non edge must be
    // fused

    DenseSet<Operation *> visited;
    DenseSet<Operation *> emptyBlocker;
    auxBFS(edgeOps, visited, auxBufferCheck, emptyBlocker);

    DenseSet<Operation *> edgeBufferBlocker;
    for (Operation *op : visited) {
      LLVM_DEBUG(llvm::dbgs() << "Ok visited " << *op << "\n";);
      if (!ops_.contains(op) && fusibleHelper_->isBuffer(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Ok inserting to blocker\n";);
        edgeBufferBlocker.insert(op);
      }
    }

    DenseSet<Operation *> auxVis;
    auxBFS(ops_, auxVis, auxBufferCheck, edgeBufferBlocker);
    opWithAuxs_.insert(auxVis.begin(), auxVis.end());
  } else {
    // Add all visited operations
    DenseSet<Operation *> visited;
    DenseSet<Operation *> emptyBlocker;
    auxBFS(ops_, visited, auxBufferCheck, emptyBlocker);
    opWithAuxs_.insert(visited.begin(), visited.end());
  }

  SetVector<Operation *> bufferOnly;
  for (auto &bufferLikeOps : opWithAuxs_) {
    if (isValidBuffer(bufferLikeOps, ops_.front())) {
      bufferOnly.insert(bufferLikeOps);
    }
  }

  DenseSet<Operation *> visitedExtra;
  DenseSet<Operation *> emptyBlocker;
  auxBFS(bufferOnly, visitedExtra, blacklistedAux, emptyBlocker);

  opWithAuxs_.insert(visitedExtra.begin(), visitedExtra.end());

  DenseSet<Operation *> visitedEnd;
  auxBFSDown(ops_, visitedEnd, nextAuxCheck, emptyBlocker);
  opWithAuxs_.insert(visitedEnd.begin(), visitedEnd.end());

  SmallVector<Operation *> remainingOps(opWithAuxs_.begin(), opWithAuxs_.end());
  sort(remainingOps.begin(), remainingOps.end(),
       [](Operation *a, Operation *b) {
         Block *blockA = a->getBlock();
         Block *blockB = b->getBlock();

         if (blockA == blockB)
           return a->isBeforeInBlock(b);

         // Iterate through parent's blocks to determine order
         for (Block &block : *blockA->getParent()) {
           if (&block == blockA)
             return true;
           if (&block == blockB)
             return false;
         }
         return false;
       });
  opWithAuxs_ =
      SetVector<Operation *>(remainingOps.begin(), remainingOps.end());
}

bool FusibleBlock::isValidAuxOrBuffer(Operation *defOp, Operation *pivotOp) {
  // Check if defOp has the same parent and meets inclusion criteria
  LLVM_DEBUG(llvm::dbgs() << "\nChecking criteria\n";);
  LLVM_DEBUG(llvm::dbgs() << *defOp << " " << *pivotOp << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "same parent: "
                          << (defOp->getParentOp() == pivotOp->getParentOp())
                          << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "is aux: " << fusibleHelper_->isAuxiliary(defOp)
                          << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "is buffer: " << fusibleHelper_->isBuffer(defOp)
                          << "\n";);
  return defOp->getParentOp() == pivotOp->getParentOp() &&
         !ops_.contains(defOp) &&
         (FusibleHelper::isAuxiliary(defOp) || FusibleHelper::isBuffer(defOp) ||
          FusibleHelper::isZeroRankElemwise(defOp) ||
          FusibleHelper::isPossibleCountingAux(defOp));
}

bool FusibleBlock::isValidBuffer(Operation *defOp, Operation *pivotOp) {
  // Check if defOp has the same parent and meets inclusion criteria
  return defOp->getParentOp() == pivotOp->getParentOp() &&
         !ops_.contains(defOp) && fusibleHelper_->isBuffer(defOp);
}

void FusibleBlock::visitInValues() {
  for (Operation *op : getOpWithAuxs()) {
    for (const Value &operand : op->getOperands()) {
      LLVM_DEBUG(llvm::dbgs()
                     << "Checking current op with auxs " << *op << "\n";);
      if (Operation *defOp = operand.getDefiningOp()) {
        if (!opWithAuxs_.contains(defOp)) {
          LLVM_DEBUG(llvm::dbgs() << "Putting " << *defOp << "\n";);
          ins_.insert(operand);
        }
      } else {
        ins_.insert(operand);
      }
    }
  }
}
} // namespace opfusion
} // namespace hfusion
} // namespace mlir
//===--------- PlanContext.cpp - Plan data structures for AV2
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hfusion-auto-vectorize-v2"

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/PlanContext.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir {
namespace hfusion {

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

void PlanContext::registerAndAnalyzeOp(Operation *op,
                                       const std::string &label) {
  FusableOpInfo &opInfo = opInfoMap[op];
  opInfo.label = label;

  // Compute numLoops, shape, maxElemBitWidth.
  SmallVector<Type> allTypes;
  if (isa<linalg::TransposeOp>(op) &&
      !analysis::isVsstbPatternTransposeOp(op)) {
    allTypes.append(op->getResultTypes().begin(), op->getResultTypes().end());
    allTypes.append(op->getOperandTypes().begin(), op->getOperandTypes().end());
  } else {
    allTypes.append(op->getOperandTypes().begin(), op->getOperandTypes().end());
    allTypes.append(op->getResultTypes().begin(), op->getResultTypes().end());
  }

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    opInfo.numReductionLoops = linalgOp.getNumReductionLoops();
    opInfo.numLoops = linalgOp.getNumLoops();
    // For LinalgOp, its shape corresponds to the shape of operand/result type
    // whose rank is equal to its numLoops.
    for (Type ty : allTypes) {
      if (auto shapedType = dyn_cast<ShapedType>(ty)) {
        if (shapedType.getRank() == opInfo.numLoops) {
          opInfo.shape.append(shapedType.getShape().begin(),
                              shapedType.getShape().end());
          break;
        }
      }
    }
    Block *body = &linalgOp->getRegion(0).front();
    body->walk([&](Operation *op) {
      allTypes.append(op->getOperandTypes().begin(),
                      op->getOperandTypes().end());
      allTypes.append(op->getResultTypes().begin(), op->getResultTypes().end());
    });
  } else {
    // For non-LinalgOp, its numLoops corresponds to the largest rank of
    // operand/result type, its shape corresponds to the shape of operand/result
    // type with largest rank.
    ShapedType typeWithLargestRank;
    for (Type ty : allTypes) {
      if (auto shapedType = dyn_cast<ShapedType>(ty)) {
        if (shapedType.getRank() > opInfo.numLoops) {
          opInfo.numLoops = shapedType.getRank();
          typeWithLargestRank = shapedType;
        }
      }
    }
    for (auto i : typeWithLargestRank.getShape())
      opInfo.shape.push_back(i);
  }

  unsigned maxElemBitWidth = 1;
  for (Type ty : allTypes) {
    Type elemType = getElementTypeOrSelf(ty);
    if (elemType.isIndex())
      continue;
    unsigned currElemBitWidth = elemType.getIntOrFloatBitWidth();
    currElemBitWidth = (currElemBitWidth == 64) ? 32 : currElemBitWidth;
    maxElemBitWidth = std::max(maxElemBitWidth, currElemBitWidth);
  }
  opInfo.maxElemBitWidth = maxElemBitWidth;
}

void PlanContext::initFusableOpInfoFrom(func::FuncOp func) {
  unsigned fusableOpCount = 1;
  MLIRContext *context = func.getContext();
  func.walk([&](Operation *op) {
    if (scope::utils::isInCubeScope(op))
      return;

    if (!isFusableOp(op))
      return;
    // Name the fusable op uniquely so that it can be matched later
    std::string label =
        "hfusion-auto-vectorize-target-" + std::to_string(fusableOpCount++);
    op->setAttr(label, UnitAttr::get(context));
    registerAndAnalyzeOp(op, label);
  });
  LLVM_DEBUG(llvm::dbgs() << "========Dumping func with label begin========\n");
  LLVM_DEBUG(llvm::dbgs() << *func << "\n");

  // Find conflict fusable ops for every fusable op.
  computeConflictLists(func);
#ifndef NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "========Dumping conflict lists begin========\n");
  for (auto info : opInfos()) {
    LLVM_DEBUG(llvm::dbgs() << "========Dumping op========\n");
    LLVM_DEBUG(llvm::dbgs() << *info.first << "\n");
    LLVM_DEBUG(llvm::dbgs() << "========Dumping conflict list========\n");
    for (auto op : info.second.conflictList)
      LLVM_DEBUG(llvm::dbgs() << *op << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
#endif
}

void PlanContext::computeTileSize() {
  for (auto fusedNode : nodes()) {
    for (Operation *fusedOp : fusedNode->ops()) {
      FusableOpInfo &opInfo = getInfo(fusedOp);
      SmallVector<int64_t> tileSize;
      SmallVector<int64_t> tileInterchange;
      fusedNode->estimateTileSizeForOp(fusedOp, tileSize, tileInterchange);
      opInfo.tileSize = tileSize;
      opInfo.tileInterchange = tileInterchange;
    }
  }
}

//===----------------------------------------------------------------------===//
// Conflict analysis (internal helpers)
//===----------------------------------------------------------------------===//

namespace {

void findPreviousAndFollowingFusableOpOf(Operation *barrierOp, Block *block,
                                         DenseSet<Operation *> &previousOps,
                                         DenseSet<Operation *> &followingOps) {
  block->walk([&](Operation *op) {
    if (isOpInBlock(op, block) && isFusableOp(op)) {
      if (op->isBeforeInBlock(barrierOp)) {
        previousOps.insert(op);
      } else {
        followingOps.insert(op);
      }
    }
  });
}

bool hasMemRefInOperands(Operation *op, Value memRef) {
  for (auto operand : op->getOperands()) {
    auto optMemRef =
        mlir::utils::tracebackMemRefToAllocOrBlockArgument(operand);
    if (optMemRef.has_value() && (memRef == optMemRef.value())) {
      return true;
    }
  }
  return false;
}

void computeConflictListsForCopyOpOperand(PlanContext &ctx,
                                          DenseSet<Operation *> &previousOps,
                                          DenseSet<Operation *> &followingOps,
                                          Value operand) {

  auto optMemRef = mlir::utils::tracebackMemRefToAllocOrBlockArgument(operand);
  if (!optMemRef.has_value()) {
    return;
  }
  auto memRef = optMemRef.value();

  for (auto previousOp : previousOps) {
    if (hasMemRefInOperands(previousOp, memRef)) {
      for (auto followingOp : followingOps) {
        ctx.getInfo(previousOp).conflictList.insert(followingOp);
        ctx.getInfo(followingOp).conflictList.insert(previousOp);
      }
    }
  }

  for (auto followingOp : followingOps) {
    if (hasMemRefInOperands(followingOp, memRef)) {
      for (auto previousOp : previousOps) {
        ctx.getInfo(previousOp).conflictList.insert(followingOp);
        ctx.getInfo(followingOp).conflictList.insert(previousOp);
      }
    }
  }
}

} // namespace

void PlanContext::computeConflictLists(func::FuncOp func) {
  func.walk([&](Block *block) {
    if (isa<func::FuncOp, scf::ForOp, scf::IfOp, scf::WhileOp, scope::ScopeOp>(
            block->getParentOp())) {
      block->walk([&](Operation *op) {
        if (isOpInBlock(op, block) && isNonVectorizableOp(op)) {
          DenseSet<Operation *> upstreamOps;
          DenseSet<Operation *> visitedUpstreamOps;
          findUpstreamFusableOpOf(op, block, upstreamOps, visitedUpstreamOps);
          DenseSet<Operation *> downstreamOps;
          DenseSet<Operation *> visitedDownstreamOps;
          findDownstreamFusableOpOf(op, block, downstreamOps,
                                    visitedDownstreamOps);
          for (auto upstreamOp : upstreamOps) {
            for (auto downstreamOp : downstreamOps) {
              getInfo(upstreamOp).conflictList.insert(downstreamOp);
              getInfo(downstreamOp).conflictList.insert(upstreamOp);
            }
          }

          if (enableCrossIfFusion) {
            // Only region-bearing ops or explicit sync/copy ops need the
            // previous/following scan and the body walk below.
            if (op->getNumRegions() == 0 &&
                !isa<hivm::SyncBlockOp, hivm::SyncBlockSetOp,
                     hivm::SyncBlockWaitOp, hivm::CreateSyncBlockLockOp,
                     hivm::SyncBlockLockOp, hivm::SyncBlockUnlockOp>(op) &&
                !isa<hivm::CopyOp, memref::CopyOp>(op))
              return;

            DenseSet<Operation *> previousOps;
            DenseSet<Operation *> followingOps;
            findPreviousAndFollowingFusableOpOf(op, block, previousOps,
                                                followingOps);

            // Walk the op body: CopyOps create operand-specific conflicts;
            // sync ops trigger a full barrier between all previous and
            // following fusable ops.
            auto walker = [this, &previousOps, &followingOps](Operation *op) {
              if (isa<hivm::CopyOp, memref::CopyOp>(op)) {
                auto copyOp = cast<CopyOpInterface>(op);
                computeConflictListsForCopyOpOperand(
                    *this, previousOps, followingOps, copyOp.getTarget());
                computeConflictListsForCopyOpOperand(
                    *this, previousOps, followingOps, copyOp.getSource());
                return WalkResult::advance();
              }
              if (isa<hivm::AnchorOp>(op))
                return WalkResult::interrupt();
              return isa<hivm::SyncBlockOp, hivm::SyncBlockSetOp,
                         hivm::SyncBlockWaitOp, hivm::CreateSyncBlockLockOp,
                         hivm::SyncBlockLockOp, hivm::SyncBlockUnlockOp>(op)
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            };
            if (op->walk(walker).wasInterrupted()) {
              for (auto previousOp : previousOps) {
                for (auto followingOp : followingOps) {
                  getInfo(previousOp).conflictList.insert(followingOp);
                  getInfo(followingOp).conflictList.insert(previousOp);
                }
              }
            }
            return;
          }

          if (isa<hivm::CopyOp, memref::CopyOp>(op)) {
            auto copyOp = cast<CopyOpInterface>(op);
            DenseSet<Operation *> previousOps;
            DenseSet<Operation *> followingOps;
            findPreviousAndFollowingFusableOpOf(op, block, previousOps,
                                                followingOps);
            computeConflictListsForCopyOpOperand(
                *this, previousOps, followingOps, copyOp.getTarget());
            computeConflictListsForCopyOpOperand(
                *this, previousOps, followingOps, copyOp.getSource());
          }

          if (isa<hivm::SyncBlockOp, hivm::SyncBlockSetOp,
                  hivm::SyncBlockWaitOp, hivm::CreateSyncBlockLockOp,
                  hivm::SyncBlockLockOp, hivm::SyncBlockUnlockOp, scf::ForOp,
                  scf::WhileOp, scf::IfOp, scope::ScopeOp>(op)) {
            DenseSet<Operation *> previousOps;
            DenseSet<Operation *> followingOps;
            findPreviousAndFollowingFusableOpOf(op, block, previousOps,
                                                followingOps);
            for (auto previousOp : previousOps) {
              for (auto followingOp : followingOps) {
                getInfo(previousOp).conflictList.insert(followingOp);
                getInfo(followingOp).conflictList.insert(previousOp);
              }
            }
          }
        }
      });
    }
  });
}

bool PlanContext::hasConflict(const FusableOpInfo &a, Operation *b) const {
  return a.conflictList.contains(b);
}
bool PlanContext::hasConflict(Operation *a, Operation *b) const {
  return hasConflict(getInfo(a), b);
}

} // namespace hfusion
} // namespace mlir

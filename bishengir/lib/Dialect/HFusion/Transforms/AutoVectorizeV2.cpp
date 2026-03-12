//===--------- AutoVectorizeV2.cpp - Auto vectorization pass
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <memory>

#define DEBUG_TYPE "hfusion-auto-vectorize-v2"

namespace mlir {
#define GEN_PASS_DEF_AUTOVECTORIZEV2
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct FusedNode {
  std::string loopLabel;
  DenseSet<Operation *> fusedOps;
  DenseSet<Operation *> fusedLeafNodes;
};

// Every fusable op(including LinalgOp and interleave/deinterleave) corresponds
// to a FusableOpInfo.
struct FusableOpInfo {
  std::string label;
  int64_t numLoops = 0;
  unsigned numReductionLoops = 0;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> tileSize;
  SmallVector<int64_t> tileInterchange;
  unsigned maxElemBitWidth = 1;
  DenseSet<Operation *> conflictList;
  std::shared_ptr<FusedNode> fusedNode = nullptr;
};

static inline bool isOpInBlock(Operation *op, Block *block) {
  return op && op->getParentOp() == block->getParentOp();
}

static bool isCubeScopeOp(Operation *op) {
  auto scopeOp = dyn_cast<scope::ScopeOp>(op);
  if (!scopeOp)
    return false;

  auto attr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!attr)
    return false;

  return attr.getTcoretype() == mlir::hivm::TCoreType::CUBE;
}

static bool isVsstbPatternTransposeOp(Operation *op) {
  auto transpose = dyn_cast<linalg::TransposeOp>(op);
  if (!transpose) {
    return false;
  }

  auto inputType = dyn_cast<ShapedType>(transpose.getInput().getType());
  if (!inputType || !inputType.hasStaticShape()) {
    return false;
  }

  auto elemType = inputType.getElementType();
  if (!(elemType.isBF16() || elemType.isF16() || elemType.isF32() ||
        elemType.isFloat8E4M3FN() || elemType.isFloat8E5M2())){
    return false;
  }

  auto perm = transpose.getPermutation();
  int64_t rank = perm.size();
  // Rule 0: Should be 3-dim transpose
  if (rank != 3)
    return false;

  // Rule 1: Should not be inner axis transpose
  if (perm[rank - 1] != rank - 1)
    return false;

  // Rule 2: Last axis should fit in exactly 32 bytes
  ArrayRef<int64_t> shape = inputType.getShape();
  // Calculate element width in bytes
  uint64_t elemByteWidth = llvm::divideCeil(
      inputType.getElementType().getIntOrFloatBitWidth(), utils::INTR_BITS_PER_BYTE);
  int64_t lastDim = shape[rank - 1];
  return lastDim * elemByteWidth == 32;
}

static bool userCanFuseIntoVsstbPatternTransposeOp(Operation *op) {
  if (llvm::any_of(op->getUsers(), [](Operation *user) {
        return isVsstbPatternTransposeOp(user);
      })) {
    return true;
  }
  for (Operation *user : op->getUsers()) {
    if (!isa<linalg::GenericOp>(user))
      continue;
    if (userCanFuseIntoVsstbPatternTransposeOp(user))
      return true;
  }
  return false;
}

static bool isExpandShapeOpCanFuseIntoVsstbPatternTranspose(Operation *op) {
  auto expandShape = dyn_cast<tensor::ExpandShapeOp>(op);
  if (!expandShape) {
    return false;
  }
  // FIXME: expand_shape with one-dim src will cause error when tile after fusing
  // into vsstb pattern transpose, see issue:
  // https://codehub-y.huawei.com/CompilerKernel/BiShengCompiler/AscendNPU-IR/issues/1100
  auto srcType = dyn_cast<TensorType>(expandShape.getSrc().getType());
  auto resType = dyn_cast<TensorType>(expandShape.getResult().getType());
  if (srcType.getShape().size() != 2 || resType.getShape().size() != 3) {
    return false;
  }

  return userCanFuseIntoVsstbPatternTransposeOp(op);
}

static bool isNonVectorizableOp(Operation *op) {
  if (hivm::util::isSIMTVF(op))
    return true;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    if (hfusion::isSingleElementLinalgOp(linalgOp) &&
        !isa<linalg::GenericOp>(linalgOp))
      return true;
  if (isExpandShapeOpCanFuseIntoVsstbPatternTranspose(op))
    return false;
  return isa<hfusion::LoadOp, hfusion::StoreOp, hfusion::ReduceWithIndexOp,
             hfusion::GatherOp, hfusion::MulExtOp, hfusion::CumsumOp,
             hfusion::CumprodOp, hfusion::PrintOp, hfusion::SortOp,
             hfusion::CastOp, hfusion::CompareOp, tensor::ExtractOp,
             tensor::DimOp, tensor::ReshapeOp, tensor::InsertSliceOp,
             tensor::ExtractSliceOp, tensor::CastOp, tensor::CollapseShapeOp,
             tensor::ExpandShapeOp, tensor::ConcatOp, hivm::CopyOp,
             hivm::CustomOp, hivm::CustomMacroOp, hivm::DebugOp, hivm::StoreOp,
             hivm::BitcastOp, hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp, scf::WhileOp,
             scf::ForOp, scf::IfOp, func::CallOp>(op) ||
         hfusion::isSimtOps(op) || hfusion::isMatmulOps(op) ||
         hfusion::opCanFuseIntoMatmul(op);
}

static bool isMemrefLinalgOp(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (llvm::any_of(linalgOp.getDpsInits(), [](Value dst) {
          return isa<MemRefType>(dst.getType());
        })) {
      return true;
    }
  }
  return false;
}

static bool isFusableOp(Operation *op) {
  if (!op)
    return false;
  if (isExpandShapeOpCanFuseIntoVsstbPatternTranspose(op))
    return true;
  return !isNonVectorizableOp(op) &&
         isa<linalg::LinalgOp, hfusion::InterleaveOp, hfusion::DeinterleaveOp>(
             op);
}

static void computeNumLoopsAndShapeAndMaxElemBitWidth(Operation *op,
                                                      FusableOpInfo &opInfo) {
  SmallVector<Type> allTypes;
  if (isa<linalg::TransposeOp>(op) && !isVsstbPatternTransposeOp(op)) {
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

static void
computeTileSize(llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
                SmallVector<std::shared_ptr<FusedNode>> &fusedNodes,
                int64_t vectorLength) {
  for (auto fusedNode : fusedNodes) {
    unsigned maxElemBitWidthInFusedNode = 1;
    for (Operation *fusedOp : fusedNode->fusedOps) {
      maxElemBitWidthInFusedNode =
          std::max(maxElemBitWidthInFusedNode,
                   fusableOpInfoMap[fusedOp].maxElemBitWidth);
    }
    bool shouldMultiAxisVectorize = false;
    for (Operation *fusedLeafNode : fusedNode->fusedLeafNodes) {
      if (isVsstbPatternTransposeOp(fusedLeafNode)) {
        shouldMultiAxisVectorize = true;
        break;
      }
    }
    for (Operation *fusedOp : fusedNode->fusedOps) {
      FusableOpInfo &opInfo = fusableOpInfoMap[fusedOp];
      SmallVector<int64_t> tileSize(opInfo.numLoops, 1);
      if (shouldMultiAxisVectorize && opInfo.numLoops > 2) {
        int64_t maxElemByteWidthInFusedNode =
            maxElemBitWidthInFusedNode / utils::INTR_BITS_PER_BYTE;
        int64_t remainBytes = vectorLength;
        int64_t allocAxisNum = 0;
        for (int64_t i = opInfo.numLoops - 1; i >= 0; --i) {
          allocAxisNum++;
          // max vectorize axis is 2
          if (allocAxisNum == 2) {
            tileSize[i] = remainBytes / maxElemByteWidthInFusedNode;
            break;
          } else {
            tileSize[i] = std::min(opInfo.shape[i],
                                   remainBytes / maxElemByteWidthInFusedNode);
            remainBytes /= tileSize[i];
          }
        }
        if (isVsstbPatternTransposeOp(fusedOp)) {
          auto transpose = dyn_cast<linalg::TransposeOp>(fusedOp);
          SmallVector<int64_t> transposeTileSize;
          SmallVector<int64_t> transposeTileInterchange;
          for (int64_t idx : transpose.getPermutation()) {
            transposeTileSize.push_back(tileSize[idx]);
            transposeTileInterchange.push_back(idx);
          }
          opInfo.tileSize = transposeTileSize;
          opInfo.tileInterchange = transposeTileInterchange;
          continue;
        }
      } else {
        tileSize[opInfo.numLoops - 1] =
            maxElemBitWidthInFusedNode == 1
                ? vectorLength
                : vectorLength /
                      (maxElemBitWidthInFusedNode / utils::INTR_BITS_PER_BYTE);
      }
      opInfo.tileSize = tileSize;
    }
  }
}

static void
findDownstreamFusableOpOf(Operation *op, Block *block,
                          DenseSet<Operation *> &downstreamFusableOps,
                          DenseSet<Operation *> &visitedOps) {
  if (visitedOps.contains(op))
    return;
  visitedOps.insert(op);
  for (Operation *user : op->getUsers()) {
    if (isOpInBlock(user, block) && isFusableOp(user)) {
      downstreamFusableOps.insert(user);
    }
    if (!isOpInBlock(user, block) &&
        isa<scf::ForOp, scf::IfOp>(user->getParentOp())) {
      user = user->getParentOp();
    }
    findDownstreamFusableOpOf(user, block, downstreamFusableOps, visitedOps);
  }
}

static void findUpstreamFusableOpOf(Operation *op, Block *block,
                                    DenseSet<Operation *> &upstreamFusableOps,
                                    DenseSet<Operation *> &visitedOps) {
  if (visitedOps.contains(op))
    return;
  visitedOps.insert(op);
  for (Value operand : op->getOperands()) {
    Operation *operandOp = operand.getDefiningOp();
    if (operandOp) {
      if (isOpInBlock(operandOp, block) && isFusableOp(operandOp)) {
        upstreamFusableOps.insert(operandOp);
      }
      findUpstreamFusableOpOf(operandOp, block, upstreamFusableOps, visitedOps);
    }
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    for (auto &childOp : forOp.getBody()->getOperations()) {
      findUpstreamFusableOpOf(&childOp, block, upstreamFusableOps, visitedOps);
    }
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    // Traverse the then block.
    for (auto &childOp : ifOp.getBody()->getOperations()) {
      findUpstreamFusableOpOf(&childOp, block, upstreamFusableOps, visitedOps);
    }
    // Also traverse the else block so that fusable ops referenced there
    // are correctly identified as upstream of this scf.if.
    if (Block *elseBlock = ifOp.elseBlock()) {
      for (auto &childOp : elseBlock->getOperations()) {
        findUpstreamFusableOpOf(&childOp, block, upstreamFusableOps, visitedOps);
      }
    }
  }
}

static void
findPreviousAndFollowingFusableOpOf(Operation *barrierOp, Block *block,
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

/// If two fusable ops are conflict with each other, they cannot be fused into
/// the same VF:
/// 1. The producer(upstream op) and consumer(downstream op) of
///    NonVectorizableOp are confilict with each other. For example:
///       A(FusableOp)
///       B(NonVectorizableOp, use A)
///       C(FusableOp, use B)
///    Then A and C are confilict with each other
/// 2. The previous and following of scf.for/hivm.hir.sync_block_wait/
///    hivm.hir.sync_block_set are confilict with each other. For example:
///       A(FusableOp)
///       hivm.hir.sync_block_wait
///       B(FusableOp)
///    Then A and B are confilict with each other
static void computeConflictLists(
    func::FuncOp func,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap) {
  func.walk([&](Block *block) {
    if (isa<func::FuncOp, scf::ForOp, scf::IfOp, scf::WhileOp>(block->getParentOp())) {
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
              fusableOpInfoMap[upstreamOp].conflictList.insert(downstreamOp);
              fusableOpInfoMap[downstreamOp].conflictList.insert(upstreamOp);
            }
          }
          if (isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp, scf::ForOp, scf::WhileOp>(
                  op)) {
            DenseSet<Operation *> previousOps;
            DenseSet<Operation *> followingOps;
            findPreviousAndFollowingFusableOpOf(op, block, previousOps,
                                                followingOps);
            for (auto previousOp : previousOps) {
              for (auto followingOp : followingOps) {
                fusableOpInfoMap[previousOp].conflictList.insert(followingOp);
                fusableOpInfoMap[followingOp].conflictList.insert(previousOp);
              }
            }
          }
        }
      });
    }
  });
}

// A fusable op is output node when its all users is NonVectorizableOp or
// terminator op.
static bool isFusableOutputNode(Operation *op, Block *block) {
  if (!isFusableOp(op))
    return false;
  if (op->getUsers().empty())
    return true;
  for (Operation *user : op->getUsers()) {
    if (isa<annotation::MarkOp>(user))
      continue;
    if (!(isNonVectorizableOp(user) || !isOpInBlock(user, block) ||
          isa<scf::YieldOp, func::ReturnOp>(user) ||
          isa<bufferization::BufferizationDialect>(user->getDialect())))
      return false;
  }
  return true;
}

static void interchangeForLeafNodes(
    SmallVector<int64_t> commonAxis, DenseSet<Operation *> fusedLeafNodes,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap) {
  for (Operation *leafNode : fusedLeafNodes) {
    FusableOpInfo &leafNodeInfo = fusableOpInfoMap[leafNode];
    leafNodeInfo.tileInterchange = commonAxis;
    for (unsigned i = 0; i < leafNodeInfo.numLoops; i++) {
      if (!llvm::is_contained(commonAxis, i)) {
        leafNodeInfo.tileInterchange.push_back(i);
      }
    }
  }
}

static bool hasCommonAxis(
    Operation *op1, Operation *op2,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap) {
  assert(fusableOpInfoMap.contains(op1) && fusableOpInfoMap.contains(op2));
  auto shape1 = fusableOpInfoMap[op1].shape;
  auto shape2 = fusableOpInfoMap[op2].shape;
  if (shape1 == shape2)
    return true;
  if (shape1.size() > 1 && shape2.size() > 1 && shape1[0] == shape2[0])
    return true;
  return false;
}

static bool isProducerConsumedImpl(Operation *target, Operation *source,
                                   DenseSet<Value> &visited) {
  if (!target || !source) {
    return false;
  }
  for (Value targetResult : target->getResults()) {
    if (visited.contains(targetResult)) {
      continue;
    }
    visited.insert(targetResult);
    // dfs check if any of the target result users is consumed by source op
    bool consumed =
        llvm::any_of(targetResult.getUsers(), [&](Operation *resultUser) {
          return source->isAncestor(resultUser) ||
                 isProducerConsumedImpl(resultUser, source, visited);
        });
    if (consumed) {
      return true;
    }
  }
  return false;
}

static bool isProducerConsumed(Operation *target, Operation *source) {
  DenseSet<Value> visited;
  if (target->isBeforeInBlock(source)) {
    return isProducerConsumedImpl(target, source, visited);
  } else {
    return isProducerConsumedImpl(source, target, visited);
  }
}

static void visitUsersOfLeafNodeRecursively(
    Operation *op, Operation *lastLeafNode, Block *block,
    DenseSet<Operation *> &usersToBeMovedSet) {
  DominanceInfo domInfo;
  for (Operation *user : op->getUsers()) {
    if (!user || user->hasTrait<OpTrait::IsTerminator>() ||
        domInfo.properlyDominates(lastLeafNode, user)) {
      continue;
    }
    while (!isOpInBlock(user, block)) {
      user = user->getParentOp();
    }
    if (usersToBeMovedSet.contains(user)) {
      continue;
    }
    usersToBeMovedSet.insert(user);
    visitUsersOfLeafNodeRecursively(user, lastLeafNode, block, usersToBeMovedSet);
  }
}

// fuse sibling will clone all users of those front siblings behind fused loop which will
// cause existing handle lost or IR order changed. So we move those front siblings and their
// users to their new positions after fuse.
// For example: we have 4 nodes in order A B C(use A) D, and A and D should fuse sibling,
// then we move A before D and move C hehind D, after moving the order will be B A D C(use A).
static void moveLeafNodesAndTheirUsers(SmallVector<Operation *> &leafNodeGroup,
                                       Block *block) {
  if (leafNodeGroup.size() == 1)
    return;

  llvm::sort(leafNodeGroup, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  Operation *lastLeafNode = leafNodeGroup.back();
  DenseSet<Operation *> usersToBeMovedSet;
  for (Operation *leafNode : llvm::drop_end(leafNodeGroup)) {
    visitUsersOfLeafNodeRecursively(leafNode, lastLeafNode, block, usersToBeMovedSet);
  }
  SmallVector<Operation *> usersToBeMoved(usersToBeMovedSet.begin(),
                                          usersToBeMovedSet.end());
  llvm::sort(usersToBeMoved, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });

  Operation *prevMovedLeafNode = lastLeafNode;
  for (Operation *leafNodeToBeMoved : llvm::drop_end(leafNodeGroup)) {
    leafNodeToBeMoved->moveBefore(prevMovedLeafNode);
    prevMovedLeafNode = leafNodeToBeMoved;
  }
  Operation *prevMovedUser = lastLeafNode;
  for (Operation *userToBeMoved : usersToBeMoved) {
    userToBeMoved->moveAfter(prevMovedUser);
    prevMovedUser = userToBeMoved;
  }
}

// Normally, we should fuse the producer into the closest fusedNode which
// contains its consumers. But in some context, we should give up fusing and
// return nullptr:
// 1. the producer is fill op or vsstb pattern transpose op
// 2. there is fusedOps within the closest fusedNode conflict with the producer
// 3. there is normal(not vsstb pattern) transpose user within the closest
//    fusedNode
static std::shared_ptr<FusedNode> findBestFusedNodeForProducer(
    Block *block, Operation *producer,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap) {
  // here we do not fuse FillOp and put FillOp into a single VF, see issue:
  // https://codehub-y.huawei.com/CompilerKernel/BiShengKernel/BiSheng/issues/3687
  if (mlir::hfusion::isFillOp(producer))
    return nullptr;

  // Since the producers and consumers of a transpose op have opposite axes, we
  // cannot fuse them into the same fusedNode. For vsstb pattern transpose, we
  // fuse the producers into this transpose op; for other pattern transpose, we
  // fuse this transpose op into its consumer ops. For example: if we have user
  // chain: op1 -> vsstb transpose -> op2, we will fuse op1 into transpose op;
  // if we have user chain: op1 -> normal transpose -> op2, we will fuse
  // transpose op into op2
  if (isVsstbPatternTransposeOp(producer))
    return nullptr;

  Operation *closestUser = nullptr;
  for (auto user : producer->getUsers()) {
    if (isOpInBlock(user, block)) {
      if (!closestUser || user->isBeforeInBlock(closestUser))
        closestUser = user;
    }
  }
  if (!closestUser || !isFusableOp(closestUser))
    return nullptr;

  auto bestFusedNode = fusableOpInfoMap[closestUser].fusedNode;
  assert(bestFusedNode);
  if (bestFusedNode->fusedOps.size() > 15)
    return nullptr;
  FusableOpInfo &producerInfo = fusableOpInfoMap[producer];
  // If the closest fuseNode is conflict with the producer, give up fusing.
  if (llvm::any_of(bestFusedNode->fusedOps, [&](Operation *fusedOp) {
        return producerInfo.conflictList.contains(fusedOp);
      }))
    return nullptr;

  int numUsersInBestFusedNode = 0;
  for (auto user : DenseSet<Operation *>(producer->getUsers().begin(),
                                         producer->getUsers().end())) {
    if (bestFusedNode->fusedOps.contains(user)) {
      numUsersInBestFusedNode++;
      if (isa<linalg::TransposeOp>(user) && !isVsstbPatternTransposeOp(user))
        return nullptr;
      if (!hasCommonAxis(producer, user, fusableOpInfoMap))
        return nullptr;
    }
  }

  if (numUsersInBestFusedNode > 1) {
    AffineMap map;
    for (OpOperand &use : producer->getUses()) {
      Operation *user = use.getOwner();
      if (bestFusedNode->fusedOps.contains(user)) {
        if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(user)) {
          if (!map) {
            map = consumerLinalgOp.getMatchingIndexingMap(&use);
          } else if (map != consumerLinalgOp.getMatchingIndexingMap(&use)) {
            return nullptr;
          }
        }
      }
    }
  }
  // If producer is reduction and user is broadcast, only fuse last-axis
  // reduction producer into last-axis broadcast op. Other fusion context will
  // generate memref.alloc inside VF and cannot be outlined, so give up fusing,
  // see issue:
  // https://codehub-y.huawei.com/CompilerKernel/BiShengCompiler/AscendNPU-IR/issues/638
  if (producerInfo.numReductionLoops > 0) {
    bool producerIsNonLastAxisReduce = false;
    auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producer);
    SmallVector<unsigned> reductionDims;
    producerLinalgOp.getReductionDims(reductionDims);
    if (reductionDims[0] != producerInfo.numLoops - 1)
      producerIsNonLastAxisReduce = true;

    for (OpOperand &use : producer->getUses()) {
      Operation *user = use.getOwner();
      if (bestFusedNode->fusedOps.contains(user)) {
        if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(user)) {
          AffineMap map = consumerLinalgOp.getMatchingIndexingMap(&use);
          if (map.getNumResults() < map.getNumDims()) {
            // last-axis broadcast, the affine map will be
            // `affine_map<(d0,d1)->(d0)>` non-last-axis broadcast, the affine
            // map will be `affine_map<(d0,d1)->(d1)>`
            for (auto indexAndResult : llvm::enumerate(map.getResults()))
              if (auto d = dyn_cast<AffineDimExpr>(indexAndResult.value()))
                if (d.getPosition() != indexAndResult.index())
                  return nullptr;
            if (producerIsNonLastAxisReduce)
              return nullptr;
          }
        }
      }
    }
  }

  return bestFusedNode;
}

static void updateConflictLists(
    SmallVector<Operation *> &leafNodeGroup, Block *block,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap) {
  DenseSet<Operation *> upstreamOps;
  DenseSet<Operation *> visitedUpstreamOps;
  DenseSet<Operation *> downstreamOps;
  DenseSet<Operation *> visitedDownstreamOps;
  for (Operation *leafNode : leafNodeGroup) {
    findUpstreamFusableOpOf(leafNode, block, upstreamOps, visitedUpstreamOps);
    findDownstreamFusableOpOf(leafNode, block, downstreamOps,
                              visitedDownstreamOps);
  }
  for (auto upstreamOp : upstreamOps) {
    for (auto downstreamOp : downstreamOps) {
      fusableOpInfoMap[upstreamOp].conflictList.insert(downstreamOp);
      fusableOpInfoMap[downstreamOp].conflictList.insert(upstreamOp);
    }
  }
}

/// For reduction op, tile_reduction_using_for has better performance than
/// tile_using_for. Firstly we should tile parallel axis by tile_using_for,
/// then tile reduction axis by tile_reduction_using_for.
static Value
tileReductionOp(OpBuilder &builder, transform::SequenceOp seqOp, Operation *op,
                Value &linalgOpHandle, SmallVector<int64_t> tileSize,
                std::string label,
                SmallVector<std::pair<std::string, SmallVector<int64_t>>>
                    &otherVectorizableOps) {
  assert(isa<linalg::LinalgOp>(op));
  auto reductionOp = cast<linalg::LinalgOp>(op);
  assert(reductionOp.getNumParallelLoops() > 0);
  assert(reductionOp.getNumReductionLoops() == 1);
  auto loc = seqOp->getLoc();
  // get parallel axis and tile.
  SmallVector<unsigned> parallelDims;
  reductionOp.getParallelDims(parallelDims);
  SmallVector<int64_t> parallelAxisTileSize(reductionOp.getNumLoops(), 0);
  for (auto i : parallelDims)
    parallelAxisTileSize[i] = tileSize[i];
  transform::TileUsingForOp parallelAxisTilingResult =
      builder.create<transform::TileUsingForOp>(loc, linalgOpHandle,
                                                parallelAxisTileSize);
  // get reduction axis and tile.
  SmallVector<unsigned> reductionDims;
  reductionOp.getReductionDims(reductionDims);
  SmallVector<int64_t> reductionAxisTileSize(reductionOp.getNumLoops(), 0);
  for (auto i : reductionDims)
    reductionAxisTileSize[i] = tileSize[i];
  transform::TileReductionUsingForOp reductionAxisTilingResult =
      builder.create<transform::TileReductionUsingForOp>(
          loc, parallelAxisTilingResult.getTiledLinalgOp(),
          reductionAxisTileSize);
  builder.create<transform::AnnotateOp>(
      loc, reductionAxisTilingResult.getForOp(), "reductionLoop", nullptr);
  // fillOp and combiningLinalgOp should also be vectorized, here we add them
  // into otherVectorizableOps which will be vectorized after tiling and
  // fusing.
  for (auto it : llvm::enumerate(reductionAxisTilingResult.getFillOp())) {
    Value fillOp = it.value();
    std::string fillOpLabel = label + "_fill_" + std::to_string(it.index());
    builder.create<transform::AnnotateOp>(loc, fillOp, fillOpLabel, nullptr);
    otherVectorizableOps.push_back(std::make_pair(fillOpLabel, tileSize));
  }
  std::string splitLinalgOpLabel = label + "_split";
  builder.create<transform::AnnotateOp>(
      loc, reductionAxisTilingResult.getSplitLinalgOp(), splitLinalgOpLabel,
      nullptr);
  otherVectorizableOps.push_back(std::make_pair(splitLinalgOpLabel, tileSize));
  // If combiningLinalgOp(is a linalg.reduce op) has dyncamic shape, it
  // cannot be vectorized, so we convert it to a linalg.generic op.
  Value generalizedCombiningLinalgOp = builder.create<transform::GeneralizeOp>(
      loc, builder.getType<transform::AnyOpType>(),
      reductionAxisTilingResult.getCombiningLinalgOp());
  std::string combiningLinalgOpLabel = label + "_combining";
  builder.create<transform::AnnotateOp>(loc, generalizedCombiningLinalgOp,
                                        combiningLinalgOpLabel, nullptr);
  otherVectorizableOps.push_back(
      std::make_pair(combiningLinalgOpLabel, tileSize));
  return parallelAxisTilingResult.getLoops().front();
}

static void
collectFusableFuncInModule(Operation *moduleOp,
                           SmallVector<func::FuncOp> &fusableFuncList) {
  moduleOp->walk([&](func::FuncOp func) {
    auto fusionKind = mlir::hfusion::tryGetFusionKind(func);
    if (hacc::utils::isDevice(func)) {
      if (fusionKind.has_value() &&
          (fusionKind.value() == mlir::hfusion::FusionKind::ShallowCV ||
           fusionKind.value() == mlir::hfusion::FusionKind::SingleCube)) {
        // Skip this for now
      } else
        fusableFuncList.push_back(func);
    }
  });
}

static transform::SequenceOp buildTransformSequenceOp(OpBuilder &builder,
                                                      func::FuncOp func) {
  builder.setInsertionPointAfter(func);
  transform::SequenceOp seqOp = builder.create<transform::SequenceOp>(
      func.getLoc(), TypeRange(), transform::FailurePropagationMode::Propagate,
      builder.getType<transform::AnyOpType>(),
      [](OpBuilder &b, Location nested, Value rootH) {
        b.create<transform::YieldOp>(nested, ValueRange());
      });
  builder.setInsertionPointToStart(seqOp.getBodyBlock());
  return seqOp;
}

class AutoVectorizeV2 : public impl::AutoVectorizeV2Base<AutoVectorizeV2> {
public:
  explicit AutoVectorizeV2(const AutoVectorizeV2Options &options)
      : AutoVectorizeV2Base(options) {}
  void runOnOperation() override;

private:
  unsigned loopCount = 0;
  void initFusableOpInfo(
      func::FuncOp func, int64_t vectorLength,
      llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap);
  void buildTileAndFuseTransformSequenceForBlock(
      OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
      llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps);
  void buildVectorizeTransformSequence(
      OpBuilder &builder, transform::SequenceOp seqOp,
      llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps);
  Value getOpTransformHandle(std::string label, OpBuilder &builder,
                             transform::SequenceOp seqOp);
  void planFuseSiblingForLeafNodes(
      Block *block, SmallVector<SmallVector<Operation *>> &leafNodeGroups,
      llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
      SmallVector<std::shared_ptr<FusedNode>> &fusedNodes);
  void planFuseProducersIntoConsumers(
      Block *block, SmallVector<SmallVector<Operation *>> &leafNodeGroups,
      SmallVector<Operation *> &producersToBeFusedInto,
      llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
      SmallVector<std::shared_ptr<FusedNode>> &fusedNodes);
  void planFuseProducerIntoFusedNode(
      Block *block, Operation *producer,
      SmallVector<SmallVector<Operation *>> &leafNodeGroups,
      SmallVector<Operation *> &producersToBeFusedInto,
      llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
      SmallVector<std::shared_ptr<FusedNode>> &fusedNodes);
  void tileAndFuseSiblingForLeafNodes(
      OpBuilder &builder, transform::SequenceOp seqOp,
      SmallVector<SmallVector<Operation *>> &leafNodeGroups,
      llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps);
  void fuseProducersIntoConsumers(
      OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
      SmallVector<Operation *> &producersToBeFusedInto,
      llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps);
  void applyCleanUp(OpBuilder &builder, transform::SequenceOp seqOp);
};

void AutoVectorizeV2::initFusableOpInfo(
    func::FuncOp func, int64_t vectorLength,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap) {
  unsigned fusableOpCount = 1;
  func.walk([&](Operation *op) {
    if (mlir::hfusion::isInCubeScope(op))
      return;

    if (!isFusableOp(op))
      return;
    // Name the fusable op uniquely so that it can be matched later
    std::string label =
        "hfusion-auto-vectorize-target-" + std::to_string(fusableOpCount++);
    op->setAttr(label, UnitAttr::get(&getContext()));

    FusableOpInfo opInfo;
    opInfo.label = label;
    computeNumLoopsAndShapeAndMaxElemBitWidth(op, opInfo);
    fusableOpInfoMap[op] = opInfo;
  });
  LLVM_DEBUG(llvm::dbgs() << "========Dumping func with label begin========\n");
  LLVM_DEBUG(llvm::dbgs() << *func << "\n");

  // Find confict fusable ops for every fusable op.
  computeConflictLists(func, fusableOpInfoMap);
#ifndef NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "========Dumping conflict lists begin========\n");
  for (auto info : fusableOpInfoMap) {
    LLVM_DEBUG(llvm::dbgs() << "========Dumping op========\n");
    LLVM_DEBUG(llvm::dbgs() << *info.first << "\n");
    LLVM_DEBUG(llvm::dbgs() << "========Dumping conflict list========\n");
    for (auto op : info.second.conflictList)
      LLVM_DEBUG(llvm::dbgs() << *op << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
#endif
}

void AutoVectorizeV2::planFuseSiblingForLeafNodes(
    Block *block, SmallVector<SmallVector<Operation *>> &leafNodeGroups,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
    SmallVector<std::shared_ptr<FusedNode>> &fusedNodes) {
  // Collect leafNodes in the block
  SmallVector<Operation *> leafNodes;
  block->walk([&](Operation *op) {
    if (isOpInBlock(op, block) && isFusableOutputNode(op, block)) {
      leafNodes.push_back(op);
    }
  });
  if (leafNodes.empty())
    return;
  // Group leafNodes, all leafNodes in the same group will be fused siblings
  for (auto leafNode : leafNodes) {
    if (leafNodeGroups.size() == 0) {
      leafNodeGroups.push_back(SmallVector<Operation *>{leafNode});
      continue;
    }
    if (isMemrefLinalgOp(leafNode)) {
      leafNodeGroups.push_back(SmallVector<Operation *>{leafNode});
      continue;
    }

    bool isInserted = false;
    for (SmallVector<Operation *> &leafNodeGroup : leafNodeGroups) {
      if (leafNodeGroup.size() > 15 || isMemrefLinalgOp(leafNodeGroup[0]))
        continue;
      // All leafNodes within a group have the same shape and do not conflict
      // with each other.
      auto leafNodeInfo = fusableOpInfoMap[leafNode];
      if (hasCommonAxis(leafNode, leafNodeGroup[0], fusableOpInfoMap) &&
          llvm::all_of(leafNodeGroup, [&](Operation *otherLeafNode) {
            return !leafNodeInfo.conflictList.contains(otherLeafNode);
          })) {
        leafNodeGroup.push_back(leafNode);
        isInserted = true;
        break;
      };
    }

    if (!isInserted)
      leafNodeGroups.push_back(SmallVector<Operation *>{leafNode});
  }
  // Every leafNodeGroup will form a fusedNode.
  for (SmallVector<Operation *> &leafNodeGroup : leafNodeGroups) {
    std::shared_ptr<FusedNode> fusedNode = std::make_shared<FusedNode>();
    fusedNodes.push_back(fusedNode);
    fusedNode->loopLabel = "outlined-loop-target-" + std::to_string(++loopCount);
    for (Operation *leafNode : leafNodeGroup) {
      fusedNode->fusedOps.insert(leafNode);
      fusedNode->fusedLeafNodes.insert(leafNode);
      fusableOpInfoMap[leafNode].fusedNode = fusedNode;
    }
    moveLeafNodesAndTheirUsers(leafNodeGroup, block);
  }
}

// Determine how to fuse producers layer by layer.
void AutoVectorizeV2::planFuseProducersIntoConsumers(
    Block *block, SmallVector<SmallVector<Operation *>> &leafNodeGroups,
    SmallVector<Operation *> &producersToBeFusedInto,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
    SmallVector<std::shared_ptr<FusedNode>> &fusedNodes) {
  std::queue<Operation *> queue;
  for (SmallVector<Operation *> leafNodeGroup : leafNodeGroups) {
    for (Operation *leafNode : leafNodeGroup) {
      queue.push(leafNode);
    }
  }
  while (!queue.empty()) {
    auto consumer = queue.front();
    queue.pop();
    for (Value operand : consumer->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (isFusableOp(producer)) {
        // It means that current producer has been handled if the value of
        // fusableOpInfoMap[producer].fusedNode is not nullptr.
        if (!isOpInBlock(producer, block) ||
            fusableOpInfoMap[producer].fusedNode)
          continue;

        // If all users of the producer has been handled, we can fuse this
        // producer into proper fused node.
        if (llvm::all_of(producer->getUsers(), [&](Operation *user) {
              if (isFusableOp(user))
                return fusableOpInfoMap[user].fusedNode != nullptr;
              return true;
            })) {
          planFuseProducerIntoFusedNode(block, producer, leafNodeGroups,
                                        producersToBeFusedInto,
                                        fusableOpInfoMap, fusedNodes);
          queue.push(producer);
        }
      }
    }
  }
}

void AutoVectorizeV2::planFuseProducerIntoFusedNode(
    Block *block, Operation *producer,
    SmallVector<SmallVector<Operation *>> &leafNodeGroups,
    SmallVector<Operation *> &producersToBeFusedInto,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
    SmallVector<std::shared_ptr<FusedNode>> &fusedNodes) {
  FusableOpInfo &producerInfo = fusableOpInfoMap[producer];
  std::shared_ptr<FusedNode> bestFusedNode =
      findBestFusedNodeForProducer(block, producer, fusableOpInfoMap);
  if (bestFusedNode) {
    producersToBeFusedInto.push_back(producer);
    bestFusedNode->fusedOps.insert(producer);
    producerInfo.fusedNode = bestFusedNode;

    // consumer leafNodes should interchange when tiling because of reduction
    // producer.
    if (producerInfo.numReductionLoops) {
      for (OpOperand &use : producer->getUses()) {
        Operation *user = use.getOwner();
        if (bestFusedNode->fusedLeafNodes.contains(user)) {
          // FIXME: here only find common axis for LinalgOp, also should find
          // for non LinalgOp(interleave, deinterleave...)
          if (auto consumer = dyn_cast<linalg::LinalgOp>(user)) {
            SmallVector<int64_t> commonAxis;
            AffineMap indexingMap = consumer.getMatchingIndexingMap(&use);
            for (AffineExpr e : indexingMap.getResults()) {
              if (auto d = dyn_cast<AffineDimExpr>(e)) {
                commonAxis.push_back(d.getPosition());
              }
            }
            interchangeForLeafNodes(commonAxis, bestFusedNode->fusedLeafNodes,
                                    fusableOpInfoMap);
            break;
          }
        }
      }
    }
  } else {
    bool isInserted = false;
    for (SmallVector<Operation *> &leafNodeGroup : leafNodeGroups) {
      if (leafNodeGroup.size() > 15)
        continue;
      // All leafNodes within a group have the common axis and do not conflict
      // with each other.
      if (hasCommonAxis(producer, leafNodeGroup[0], fusableOpInfoMap) &&
          llvm::all_of(leafNodeGroup, [&](Operation *otherLeafNode) {
            return !producerInfo.conflictList.contains(otherLeafNode) &&
                   !isProducerConsumed(producer, otherLeafNode);
          })) {
        std::shared_ptr<FusedNode> fusedNode = fusableOpInfoMap[leafNodeGroup[0]].fusedNode;
        fusedNode->fusedOps.insert(producer);
        fusedNode->fusedLeafNodes.insert(producer);
        producerInfo.fusedNode = fusedNode;
        leafNodeGroup.push_back(producer);
        isInserted = true;
        moveLeafNodesAndTheirUsers(leafNodeGroup, block);
        updateConflictLists(leafNodeGroup, block, fusableOpInfoMap);
        break;
      };
    }
    if (!isInserted) {
      leafNodeGroups.push_back(SmallVector<Operation *>{producer});
      std::shared_ptr<FusedNode> fusedNode = std::make_shared<FusedNode>();
      fusedNodes.push_back(fusedNode);
      fusedNode->loopLabel = "outlined-loop-target-" + std::to_string(++loopCount);
      fusedNode->fusedOps.insert(producer);
      fusedNode->fusedLeafNodes.insert(producer);
      producerInfo.fusedNode = fusedNode;
    }
  }
}

Value AutoVectorizeV2::getOpTransformHandle(std::string label,
                                            OpBuilder &builder,
                                            transform::SequenceOp seqOp) {
  DictionaryAttr opAttr = builder.getDictionaryAttr(
      builder.getNamedAttr(label, builder.getUnitAttr()));
  Value linalgOpHandle =
      builder
          .create<transform::MatchOp>(
              seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
              seqOp.getBodyBlock()->getArguments().front(), ArrayAttr(),
              transform::MatchInterfaceEnumAttr{}, opAttr, DictionaryAttr{},
              TypeAttr{}, ArrayAttr{})
          .getResults();
  return linalgOpHandle;
}

void AutoVectorizeV2::tileAndFuseSiblingForLeafNodes(
    OpBuilder &builder, transform::SequenceOp seqOp,
    SmallVector<SmallVector<Operation *>> &leafNodeGroups,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps) {
  auto loc = seqOp->getLoc();
  for (SmallVector<Operation *> &leafNodeGroup : leafNodeGroups) {
    SmallVector<Value> tiledLoopHandles;
    bool hasFillOp = false;
    for (Operation *leafNode : leafNodeGroup) {
      if (mlir::hfusion::isFillOp(leafNode))
        hasFillOp = true;
      FusableOpInfo &leafNodeInfo = fusableOpInfoMap[leafNode];
      Value leafNodeHandle =
          getOpTransformHandle(leafNodeInfo.label, builder, seqOp);
      if (hfusion::shouldUseTileReductionUsingForV2(leafNode)) {
        tiledLoopHandles.push_back(tileReductionOp(
            builder, seqOp, leafNode, leafNodeHandle, leafNodeInfo.tileSize,
            leafNodeInfo.label, otherVectorizableOps));
      } else {
        transform::TileUsingForOp tilingResult =
            builder.create<transform::TileUsingForOp>(
                loc, leafNodeHandle, leafNodeInfo.tileSize,
                leafNodeInfo.tileInterchange);
        tiledLoopHandles.push_back(tilingResult.getLoops().front());
        if (isVsstbPatternTransposeOp(leafNode)) {
          assert(tilingResult.getLoops().size() >= 2);
          builder.create<transform::AnnotateOp>(
              loc, tilingResult.getLoops()[1],
              builder.getStringAttr("unroll_for_vsstb"), nullptr);
        }
      }
    }
    assert(!std::empty(tiledLoopHandles) && "Should fuse more than one loops");
    applyCleanUp(builder, seqOp);
    Value fusedLoopHandle = tiledLoopHandles.front();
    for (Value nextLoopHandle : llvm::drop_begin(tiledLoopHandles)) {
      fusedLoopHandle =
          builder
              .create<transform::LoopFuseSiblingOp>(
                  loc, builder.getType<transform::AnyOpType>(),
                  /*target=*/fusedLoopHandle, /*source=*/nextLoopHandle, true)
              .getFusedLoop();
    }
    if (hasFillOp)
      builder.create<transform::AnnotateOp>(loc, fusedLoopHandle,
                                            "outlinedLoopWithFill", nullptr);
    builder.create<transform::AnnotateOp>(
        loc, fusedLoopHandle, fusableOpInfoMap[leafNodeGroup[0]].fusedNode->loopLabel,
        nullptr);
    if (tiledLoopHandles.size() > 1)
      applyCleanUp(builder, seqOp);
  }
}

void AutoVectorizeV2::fuseProducersIntoConsumers(
    OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
    SmallVector<Operation *> &producersToBeFusedInto,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps) {
  auto loc = seqOp.getLoc();
  for (Operation *producer : producersToBeFusedInto) {
    FusableOpInfo &producerInfo = fusableOpInfoMap[producer];
    Value producerHandle =
        getOpTransformHandle(producerInfo.label, builder, seqOp);
    std::shared_ptr<FusedNode> fusedNode = producerInfo.fusedNode;
    Value containingLoopHandle =
        getOpTransformHandle(fusedNode->loopLabel, builder, seqOp);
    builder.create<transform::ApplyPatternsOp>(
        loc, containingLoopHandle, [](OpBuilder &innerBuilder, Location loc) {
          innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(loc);
        });
    transform::FuseIntoContainingOp fuseIntoOp =
        builder.create<transform::FuseIntoContainingOp>(
            loc, builder.getType<transform::AnyOpType>(),
            builder.getType<transform::AnyOpType>(), producerHandle,
            containingLoopHandle, /*merge_multiple_extract_uses*/true);
    Value fusedOp = fuseIntoOp.getFusedOp();
    Value newContainingLoopHandle = fuseIntoOp.getNewContainingOp();
    builder.create<transform::ApplyPatternsOp>(
        loc, newContainingLoopHandle, [](OpBuilder &innerBuilder, Location loc) {
          innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(loc);
        });
    if (hfusion::shouldUseTileReductionUsingForV2(producer)) {
      tileReductionOp(builder, seqOp, producer, fusedOp, producerInfo.tileSize,
                      producerInfo.label, otherVectorizableOps);
    } else if (!isa<tensor::ExpandShapeOp>(producer)) {
      builder.create<transform::TileUsingForOp>(loc, fusedOp,
                                                producerInfo.tileSize);
    }
    builder.create<transform::ApplyPatternsOp>(
        loc, newContainingLoopHandle, [](OpBuilder &innerBuilder, Location loc) {
          innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(loc);
        });
    applyCleanUp(builder, seqOp);
  }
}

void AutoVectorizeV2::buildTileAndFuseTransformSequenceForBlock(
    OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps) {
  SmallVector<std::shared_ptr<FusedNode>> fusedNodes;
  SmallVector<SmallVector<Operation *>> leafNodeGroups;
  planFuseSiblingForLeafNodes(block, leafNodeGroups, fusableOpInfoMap,
                              fusedNodes);
  SmallVector<Operation *> producersToBeFusedInto;
  planFuseProducersIntoConsumers(block, leafNodeGroups, producersToBeFusedInto,
                                 fusableOpInfoMap, fusedNodes);
  computeTileSize(fusableOpInfoMap, fusedNodes, vectorLength);
#ifndef NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "========Dumping LeafNodeGroups begin========\n");
  for (auto leafNodeGroup : leafNodeGroups) {
    LLVM_DEBUG(llvm::dbgs() << "========Dumping group========\n");
    for (auto op : leafNodeGroup) {
      LLVM_DEBUG(llvm::dbgs() << *op);
      LLVM_DEBUG(llvm::dbgs() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "----shape:[");
      for (auto i : fusableOpInfoMap[op].shape)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
      LLVM_DEBUG(llvm::dbgs() << "----tilesize:[");
      for (auto i : fusableOpInfoMap[op].tileSize)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "====Dumping ProducersToBeFusedInto begin====\n");
  for (auto op : producersToBeFusedInto) {
    LLVM_DEBUG(llvm::dbgs() << *op);
    LLVM_DEBUG(llvm::dbgs() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "----shape:[");
    for (auto i : fusableOpInfoMap[op].shape)
      LLVM_DEBUG(llvm::dbgs() << i << ",");
    LLVM_DEBUG(llvm::dbgs() << "]----\n");
    LLVM_DEBUG(llvm::dbgs() << "----tilesize:[");
    for (auto i : fusableOpInfoMap[op].tileSize)
      LLVM_DEBUG(llvm::dbgs() << i << ",");
    LLVM_DEBUG(llvm::dbgs() << "]----\n");
  }
#endif
  tileAndFuseSiblingForLeafNodes(builder, seqOp, leafNodeGroups,
                                 fusableOpInfoMap, otherVectorizableOps);
  fuseProducersIntoConsumers(builder, seqOp, block, producersToBeFusedInto,
                             fusableOpInfoMap, otherVectorizableOps);
}

void AutoVectorizeV2::buildVectorizeTransformSequence(
    OpBuilder &builder, transform::SequenceOp seqOp,
    llvm::MapVector<Operation *, FusableOpInfo> &fusableOpInfoMap,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps) {
  auto loc = seqOp.getLoc();
  for (auto info : fusableOpInfoMap) {
    // Here only vectorize LinalgOp, interleave/deinterleave will be vectorized
    // in convert-hfusion-to-hivmave pass.
    if (isa<linalg::LinalgOp>(info.first)) {
      FusableOpInfo &opInfo = info.second;
      builder.create<transform::VectorizeOp>(
          loc, getOpTransformHandle(opInfo.label, builder, seqOp),
          SmallVector<Value>(), opInfo.tileSize, nullptr,
          SmallVector<bool>(opInfo.tileSize.size(), false));
    }
  }
  for (auto vectorizableOp : otherVectorizableOps) {
    builder.create<transform::VectorizeOp>(
        loc, getOpTransformHandle(vectorizableOp.first, builder, seqOp),
        SmallVector<Value>(), vectorizableOp.second, nullptr,
        SmallVector<bool>(vectorizableOp.second.size(), false));
  }
}

void AutoVectorizeV2::applyCleanUp(OpBuilder &builder,
                                   transform::SequenceOp seqOp) {
  auto loopLikeAttr = transform::MatchInterfaceEnumAttr::get(
      builder.getContext(), transform::MatchInterfaceEnum::LoopLikeInterface);
  Value loopLikeHandle = builder
                             .create<transform::MatchOp>(
                                 builder.getInsertionPoint()->getLoc(),
                                 builder.getType<transform::AnyOpType>(),
                                 seqOp.getBodyBlock()->getArguments().front(),
                                 ArrayAttr(), loopLikeAttr, DictionaryAttr(),
                                 DictionaryAttr{}, TypeAttr{}, ArrayAttr{})
                             .getResults();
  builder.create<transform::ApplyLoopInvariantCodeMotionOp>(
      loopLikeHandle.getLoc(), loopLikeHandle);

  Value funcHandle = builder.create<transform::MatchOp>(
      builder.getInsertionPoint()->getLoc(),
      seqOp.getBodyBlock()->getArguments().front(),
      ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
  auto bodyBuilder = [](OpBuilder &innerBuilder, Location loc) {
    innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(loc);
    innerBuilder
        .create<transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp>(
            loc);
  };
  transform::ApplyPatternsOp applyPatternsOp =
      builder.create<transform::ApplyPatternsOp>(funcHandle.getLoc(),
                                                 /*target=*/funcHandle,
                                                 /*bodyBuilder=*/bodyBuilder);
  applyPatternsOp.setApplyCse(true);
  applyPatternsOp.setDisablePatternsAttr(builder.getArrayAttr(
      SmallVector<Attribute>{builder.getStringAttr("SimplifyTrivialLoops")}));
}

void AutoVectorizeV2::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  IRRewriter rewriter(context);
  OpBuilder builder(context);

  SmallVector<func::FuncOp> fusableFuncList;
  collectFusableFuncInModule(op, fusableFuncList);

  for (func::FuncOp func : fusableFuncList) {
    // Clone original func before apply transform seqence
    builder.setInsertionPointAfter(func);
    func::FuncOp clonedFunc = dyn_cast<func::FuncOp>(builder.clone(*func));
    auto funcName = clonedFunc.getSymName().str();
    SymbolTable::setSymbolName(clonedFunc,
                               StringAttr::get(context, "cloned_" + funcName));

    llvm::MapVector<Operation *, FusableOpInfo> fusableOpInfoMap;
    initFusableOpInfo(func, vectorLength, fusableOpInfoMap);
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        otherVectorizableOps;
    transform::SequenceOp seqOp = buildTransformSequenceOp(builder, func);
    func.walk([&](Block *block) {
      if (mlir::hfusion::isInCubeScope(block->getParentOp()) ||
          isCubeScopeOp(block->getParentOp()))
        return;

      if (isa<func::FuncOp, scf::ForOp, scf::IfOp, scf::WhileOp,
              scope::ScopeOp>(block->getParentOp()))
        buildTileAndFuseTransformSequenceForBlock(
            builder, seqOp, block, fusableOpInfoMap, otherVectorizableOps);
    });
    buildVectorizeTransformSequence(builder, seqOp, fusableOpInfoMap,
                                    otherVectorizableOps);
    // Apply transform
    transform::TransformOptions options;
    options.enableExpensiveChecks(false);
    LogicalResult result = transform::applyTransformNamedSequence(
        func, seqOp, func->getParentOfType<ModuleOp>(), options);
    seqOp->erase();
    // If any error occurs during applying, use cloned func to run
    // auto-vectorize pass.
    if (failed(result)) {
      LLVM_DEBUG(llvm::dbgs() << "========Failed========\n");
      rewriter.eraseOp(func);
      SymbolTable::setSymbolName(clonedFunc,
                                 StringAttr::get(context, funcName));

      PassManager pm(op->getContext());
      pm.addPass(mlir::hfusion::createHFusionAutoVectorizePass());
      std::ignore = pm.run(op);
      return;
    } else {
      rewriter.eraseOp(clonedFunc);
      func.walk([&](Block *block) {
        if (isa<func::FuncOp, scf::ForOp, scf::IfOp>(block->getParentOp()))
          sortTopologically(block);
      });
    }
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createHFusionAutoVectorizeV2Pass(
    const AutoVectorizeV2Options &options) {
  return std::make_unique<AutoVectorizeV2>(options);
}

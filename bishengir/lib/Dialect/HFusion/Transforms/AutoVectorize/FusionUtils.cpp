//===--------- FusionUtils.cpp - Op classification helpers for AV2
//----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/FusionUtils.h"
#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace mlir {
namespace hfusion {

transform::SequenceOp buildTransformSequenceOp(OpBuilder &builder,
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

//===----------------------------------------------------------------------===//
// Op classification helpers
//===----------------------------------------------------------------------===//

bool isOpInBlock(Operation *op, Block *block) {
  return op && op->getParentOp() == block->getParentOp();
}

bool isMemrefLinalgOp(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (llvm::any_of(linalgOp.getDpsInits(), [](Value dst) {
          return isa<MemRefType>(dst.getType());
        })) {
      return true;
    }
  }
  return false;
}

bool hasRankReducingIndexingMap(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  auto numLoops = linalgOp.getNumLoops();
  if (numLoops <= 2)
    return false;
  auto check = [numLoops](const auto &map) {
    return map.getNumResults() != numLoops;
  };
  auto indexing = linalgOp.getIndexingMapsArray();
  auto outsCnt = linalgOp.getNumDpsInits();
  return llvm::any_of(llvm::drop_end(indexing, outsCnt), check);
}

namespace {
bool isScalarPredSelectOp(Operation *op) {
  auto selectOp = dyn_cast<arith::SelectOp>(op);
  if (!selectOp)
    return false;
  auto condType = selectOp.getCondition().getType();
  return condType.isInteger(1) || condType.isIndex();
}
} // namespace

bool isNonVectorizableOp(Operation *op) {
  if (hivm::util::isSIMTVF(op) || hfusion::isSimtOps(op)) {
    return true;
  }

  if (isScalarPredSelectOp(op)) {
    return true;
  }

  if (hfusion::isMatmulOps(op) || hfusion::opCanFuseIntoMatmul(op)) {
    return true;
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (linalgOp && hfusion::isSingleElementLinalgOp(linalgOp) &&
      !isa<linalg::GenericOp>(linalgOp) && !isa<linalg::MapOp>(linalgOp)) {
    return true;
  }

  if (analysis::isExpandShapeOpCanFuseIntoVsstbPatternTranspose(op)) {
    return false;
  }

  return isa<hfusion::LoadOp, hfusion::StoreOp, hfusion::ReduceWithIndexOp,
             hfusion::GatherOp, hfusion::MulExtOp, hfusion::CumsumOp,
             hfusion::CumprodOp, hfusion::CummaxOp, hfusion::CumminOp,
             hfusion::PrintOp, hfusion::SortOp, hfusion::CastOp,
             hfusion::CompareOp, tensor::ExtractOp, tensor::DimOp,
             tensor::ReshapeOp, tensor::InsertSliceOp, tensor::ExtractSliceOp,
             tensor::CastOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             tensor::ConcatOp, hivm::CopyOp, hivm::CustomOp,
             hivm::CustomMacroOp, hivm::DebugOp, hivm::StoreOp, hivm::BitcastOp,
             hivm::VGatherOp, hivm::SyncBlockOp, hivm::VSortOp,
             hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp,
             hivm::CreateSyncBlockLockOp, hivm::SyncBlockLockOp,
             hivm::SyncBlockUnlockOp, scf::WhileOp, scf::ForOp, scf::IfOp,
             scope::ScopeOp, func::CallOp, memref::CopyOp,
             bufferization::MaterializeInDestinationOp>(op);
}

bool isFusableOp(Operation *op) {
  if (!op)
    return false;
  if (analysis::isExpandShapeOpCanFuseIntoVsstbPatternTranspose(op))
    return true;
  return !isNonVectorizableOp(op) &&
         isa<linalg::LinalgOp, hfusion::InterleaveOp, hfusion::DeinterleaveOp>(
             op);
}

//===----------------------------------------------------------------------===//
// Graph traversal helpers
//===----------------------------------------------------------------------===//

void findDownstreamFusableOpOf(Operation *op, Block *block,
                               DenseSet<Operation *> &downstreamFusableOps,
                               DenseSet<Operation *> &visitedOps) {
  if (visitedOps.contains(op))
    return;
  visitedOps.insert(op);
  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    Value target = copyOp.getTarget();
    for (Operation *targetUser : target.getUsers()) {
      if (isOpInBlock(targetUser, block) && isFusableOp(targetUser))
        downstreamFusableOps.insert(targetUser);
      findDownstreamFusableOpOf(targetUser, block, downstreamFusableOps,
                                visitedOps);
    }
  }
  if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(op)) {
    Value tensor = toTensorOp.getResult();
    for (Operation *tensorUser : tensor.getUsers()) {
      if (isOpInBlock(tensorUser, block) && isFusableOp(tensorUser))
        downstreamFusableOps.insert(tensorUser);
      findDownstreamFusableOpOf(tensorUser, block, downstreamFusableOps,
                                visitedOps);
    }
  }
  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    Value memref = storeOp.getMemref();
    for (Operation *memUser : memref.getUsers()) {
      if (isOpInBlock(memUser, block) && isFusableOp(memUser))
        downstreamFusableOps.insert(memUser);
      findDownstreamFusableOpOf(memUser, block, downstreamFusableOps,
                                visitedOps);
    }
  }
  for (Operation *user : op->getUsers()) {
    if (isOpInBlock(user, block) && isFusableOp(user)) {
      downstreamFusableOps.insert(user);
    }
    if (!isOpInBlock(user, block) &&
        isa<scf::ForOp, scf::IfOp, scf::WhileOp, scope::ScopeOp>(
            user->getParentOp())) {
      user = user->getParentOp();
    }
    findDownstreamFusableOpOf(user, block, downstreamFusableOps, visitedOps);
  }
}

void findUpstreamFusableOpOf(Operation *op, Block *block,
                             DenseSet<Operation *> &upstreamFusableOps,
                             DenseSet<Operation *> &visitedOps) {
  if (visitedOps.contains(op))
    return;
  visitedOps.insert(op);
  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    Value source = copyOp.getSource();
    if (Operation *sourceOp = source.getDefiningOp()) {
      if (isOpInBlock(sourceOp, block) && isFusableOp(sourceOp))
        upstreamFusableOps.insert(sourceOp);
      findUpstreamFusableOpOf(sourceOp, block, upstreamFusableOps, visitedOps);
    }
    for (Operation *sourceUser : source.getUsers()) {
      if (sourceUser != op && isOpInBlock(sourceUser, block))
        findUpstreamFusableOpOf(sourceUser, block, upstreamFusableOps,
                                visitedOps);
    }
  }
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
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    for (auto &childOp : whileOp.getBeforeBody()->getOperations()) {
      findUpstreamFusableOpOf(&childOp, block, upstreamFusableOps, visitedOps);
    }
    for (auto &childOp : whileOp.getAfterBody()->getOperations()) {
      findUpstreamFusableOpOf(&childOp, block, upstreamFusableOps, visitedOps);
    }
  } else if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
    for (auto &childOp : scopeOp.getBody()->getOperations()) {
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
        findUpstreamFusableOpOf(&childOp, block, upstreamFusableOps,
                                visitedOps);
      }
    }
  }
}

} // namespace hfusion
} // namespace mlir

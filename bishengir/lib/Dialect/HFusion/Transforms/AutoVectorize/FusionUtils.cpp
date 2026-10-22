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
  return op && block && op->getBlock() == block;
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
             hfusion::GatherOp, hfusion::MulExtOp, hfusion::MulExtUiOp,
             hfusion::CumsumOp, hfusion::CumprodOp, hfusion::CummaxOp,
             hfusion::CumminOp, hfusion::PrintOp, hfusion::SortOp,
             hfusion::CastOp, hfusion::CompareOp, tensor::ExtractOp,
             tensor::DimOp, tensor::ReshapeOp, tensor::InsertSliceOp,
             tensor::ExtractSliceOp, tensor::CastOp, tensor::CollapseShapeOp,
             tensor::ExpandShapeOp, tensor::ConcatOp, hivm::CopyOp,
             hivm::CustomOp, hivm::CustomMacroOp, hivm::DebugOp, hivm::StoreOp,
             hivm::BitcastOp, hivm::VGatherOp, hivm::SyncBlockOp, hivm::VSortOp,
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

} // namespace hfusion
} // namespace mlir

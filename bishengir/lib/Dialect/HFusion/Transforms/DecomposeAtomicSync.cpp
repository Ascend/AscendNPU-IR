//===- DecomposeAtomicSync.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HFusion/Transforms/DecomposeAtomicSync.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::hfusion;

namespace {

static constexpr llvm::StringLiteral kAlreadySync = "already_sync";

static Value getSnapshotBuffer(Operation *snapshotOp) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(snapshotOp))
    return copyOp.getTarget();
  if (auto loadOp = dyn_cast<hivm::LoadOp>(snapshotOp))
    return loadOp.getDst();
  return nullptr;
}

static bool snapshotHasReturnedValueUsers(Operation *snapshotOp) {
  Value buffer = getSnapshotBuffer(snapshotOp);
  if (!buffer)
    return false;
  return llvm::range_size(buffer.getUsers()) > 1;
}

static Operation *findReturnedValueSnapshotOp(Operation *atomicOp,
                                              Value gmTarget) {
  Operation *op = atomicOp->getPrevNode();
  while (op) {
    if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      if (copyOp.getSource() == gmTarget)
        return copyOp;
    }
    if (auto loadOp = dyn_cast<hivm::LoadOp>(op)) {
      if (loadOp.getSrc() == gmTarget)
        return loadOp;
    }
    op = op->getPrevNode();
  }
  return nullptr;
}

static LogicalResult addSyncForReturnedValue(Operation *atomicOp,
                                             Operation *syncStartOp,
                                             PatternRewriter &rewriter) {
  if (atomicOp->hasAttr(kAlreadySync))
    return failure();

  Location loc = atomicOp->getLoc();

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(syncStartOp);
  auto lockVar = hivm::createSyncBlockLockVar(rewriter, loc);
  rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar.getResult());
  rewriter.setInsertionPointAfter(atomicOp);
  rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar.getResult());
  atomicOp->setAttr(kAlreadySync, UnitAttr::get(atomicOp->getContext()));
  return success();
}

struct HFusionAtomicStoreReturnedValueSyncPattern
    : public OpRewritePattern<hfusion::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::StoreOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module || !hacc::utils::isRegBasedArch(module))
      return failure();

    AtomicKind kind = op.getAtomicKind();
    if (kind != AtomicKind::ADD && kind != AtomicKind::MAX &&
        kind != AtomicKind::MIN)
      return failure();

    if (op.getDpsInits().empty())
      return failure();
    Value gmDst = op.getDpsInits().front();
    Operation *snapshotOp = findReturnedValueSnapshotOp(op, gmDst);
    if (!snapshotOp || !snapshotHasReturnedValueUsers(snapshotOp))
      return failure();

    return addSyncForReturnedValue(op, snapshotOp, rewriter);
  }
};

struct HFusionAtomicCasReturnedValueSyncPattern
    : public OpRewritePattern<hfusion::AtomicCasOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::AtomicCasOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module || !hacc::utils::isRegBasedArch(module))
      return failure();

    Operation *syncStartOp = op;
    if (op->getNumResults() == 0) {
      Value gmDst = op.getDst();
      Operation *snapshotOp = findReturnedValueSnapshotOp(op, gmDst);
      if (!snapshotOp || !snapshotHasReturnedValueUsers(snapshotOp))
        return failure();
      syncStartOp = snapshotOp;
    }

    return addSyncForReturnedValue(op, syncStartOp, rewriter);
  }
};

struct LinalgAtomicCasReturnedValueSyncPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module || !hacc::utils::isRegBasedArch(module))
      return failure();

    auto atomicAttr = op->getAttrOfType<StringAttr>("GenericAtomicRMW");
    if (!atomicAttr || atomicAttr.getValue() != "cas")
      return failure();

    if (op.getNumDpsInputs() < 1)
      return failure();
    Value gmDst = op.getDpsInputs()[0];
    Operation *snapshotOp = findReturnedValueSnapshotOp(op, gmDst);
    if (!snapshotOp || !snapshotHasReturnedValueUsers(snapshotOp))
      return failure();

    return addSyncForReturnedValue(op, snapshotOp, rewriter);
  }
};

} // namespace

void mlir::hfusion::populateHFusionDecomposeAtomicSyncPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<HFusionAtomicStoreReturnedValueSyncPattern,
               HFusionAtomicCasReturnedValueSyncPattern,
               LinalgAtomicCasReturnedValueSyncPattern>(ctx);
}

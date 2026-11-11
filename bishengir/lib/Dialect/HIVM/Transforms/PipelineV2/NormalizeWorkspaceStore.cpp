//===- NormalizeWorkspaceStore.cpp - Normalize workspace stores -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Util.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "normalize-workspace-store"

namespace mlir {
namespace hivm {

using bishengir::memref_ext::AllocWorkspaceOp;

#define GEN_PASS_DEF_NORMALIZEWORKSPACESTORE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

namespace {

static bool isInsideLegacyPipelinedLoop(Operation *op) {
  for (Operation *curr = op->getParentOp(); curr != nullptr;
       curr = curr->getParentOp()) {
    if (curr->hasAttr(LegacyCVPipelineAttr::name))
      return true;
  }
  return false;
}

struct NormalizeWorkspaceStorePass
    : public impl::NormalizeWorkspaceStoreBase<NormalizeWorkspaceStorePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<StoreOp> candidates;
    funcOp.walk([&](StoreOp storeOp) {
      if (isInsideLegacyPipelinedLoop(storeOp))
        return;
      if (storeOp.getNumResults() != 1)
        return;
      auto toTensorOp =
          storeOp.getDst().getDefiningOp<bufferization::ToTensorOp>();
      if (!toTensorOp || !toTensorOp.getResult().hasOneUse())
        return;
      Value rootVal = traceValueDef(toTensorOp.getOperand());
      if (!rootVal ||
          !isa_and_nonnull<AllocWorkspaceOp>(rootVal.getDefiningOp()))
        return;
      candidates.push_back(storeOp);
    });

    for (StoreOp storeOp : candidates) {
      auto toTensorOp =
          storeOp.getDst().getDefiningOp<bufferization::ToTensorOp>();
      if (!toTensorOp || !toTensorOp.getResult().hasOneUse())
        continue;

      Value memrefVal = toTensorOp.getOperand();
      Value resultTensor = storeOp.getResultTensor();

      OpBuilder builder(storeOp);
      auto newStore = builder.create<StoreOp>(storeOp.getLoc(), TypeRange{},
                                              storeOp.getSrc(), memrefVal);
      newStore->setAttrs(storeOp->getAttrs());

      auto newToTensor = builder.create<bufferization::ToTensorOp>(
          toTensorOp.getLoc(), toTensorOp.getType(), memrefVal,
          toTensorOp.getRestrict(), toTensorOp.getWritable());

      for (Operation *user :
           llvm::make_early_inc_range(resultTensor.getUsers())) {
        auto markOp = dyn_cast<annotation::MarkOp>(user);
        if (!markOp)
          continue;

        if (auto coreAttr =
                markOp->getAttrOfType<TCoreTypeAttr>(TCoreTypeAttr::name)) {
          if (!newStore->hasAttr(TCoreTypeAttr::name))
            newStore->setAttr(TCoreTypeAttr::name, coreAttr);
          markOp->removeAttr(TCoreTypeAttr::name);
        }
        if (auto markerAttr = markOp->getAttrOfType<TCoreTypeMarkerAttr>(
                TCoreTypeMarkerAttr::name)) {
          if (!newStore->hasAttr(TCoreTypeAttr::name)) {
            newStore->setAttr(TCoreTypeAttr::name,
                              TCoreTypeAttr::get(markOp.getContext(),
                                                 markerAttr.getTcoretype()));
          }
          markOp->removeAttr(TCoreTypeMarkerAttr::name);
        }
        if (markOp.isAttrEmpty())
          markOp.erase();
      }

      resultTensor.replaceAllUsesWith(newToTensor.getResult());
      storeOp.erase();
      toTensorOp.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createNormalizeWorkspaceStorePass() {
  return std::make_unique<NormalizeWorkspaceStorePass>();
}

} // namespace hivm
} // namespace mlir

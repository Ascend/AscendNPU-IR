//===- DuplicateUnitMaskBroadcast.cpp ----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass duplicates ave.hir.broadcast ops whose mask operand has type
// vector<1xi1> and whose result has multiple users, so that each broadcast
// result value has at most one user.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_DUPLICATEUNITMASKBROADCAST
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivmave;

#define DEBUG_TYPE "duplicate-unit-mask-broadcast"

namespace {

struct DuplicateUnitMaskBroadcastPass
    : public impl::DuplicateUnitMaskBroadcastBase<
          DuplicateUnitMaskBroadcastPass> {
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Collect all candidate broadcast ops first to avoid iterator invalidation.
    SmallVector<hivmave::VFBroadcastScalarMaskOp> candidates;
    funcOp.walk([&](hivmave::VFBroadcastScalarMaskOp bcastOp) {
      // Only process broadcasts whose mask is vector<1xi1>.
      auto maskType = dyn_cast<VectorType>(bcastOp.getMask().getType());
      if (!maskType || maskType.getNumElements() != 1 ||
          !maskType.getElementType().isInteger(1))
        return;

      // Skip if the result has zero or one user.
      if (bcastOp.getResult().hasOneUse() || bcastOp.getResult().use_empty())
        return;

      candidates.push_back(bcastOp);
    });

    if (candidates.empty())
      return;

    OpBuilder builder(&getContext());

    for (auto bcastOp : candidates) {
      // Re-collect users since earlier iterations may have modified uses.
      SmallVector<Operation *> users(bcastOp.getResult().getUsers());
      if (users.size() <= 1)
        continue;

      // Keep the first user using the original op; duplicate for the rest.
      auto maskDefOp = bcastOp.getMask().getDefiningOp();
      for (unsigned i = 1, e = users.size(); i < e; ++i) {
        builder.setInsertionPoint(users[i]);

        // Also duplicate the mask-defining op so each broadcast copy gets
        // its own independent mask.
        Value newMask = bcastOp.getMask();
        if (maskDefOp) {
          auto *clonedMaskOp = builder.clone(*maskDefOp);
          newMask = clonedMaskOp->getResult(0);
        }

        auto newBcast = builder.create<hivmave::VFBroadcastScalarMaskOp>(
            bcastOp.getLoc(), bcastOp.getResult().getType(), bcastOp.getSrc(),
            newMask);
        users[i]->replaceUsesOfWith(bcastOp.getResult(), newBcast.getResult());
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[" << DEBUG_TYPE << "] Duplicated broadcast: "
                     << bcastOp << " for " << (users.size() - 1)
                     << " additional user(s)\n";
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivmave::createDuplicateUnitMaskBroadcastPass() {
  return std::make_unique<DuplicateUnitMaskBroadcastPass>();
}

//===---------------- PeelLoopsContainingTranspose.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "peel-loops-containing-transpose"

namespace mlir {
#define GEN_PASS_DEF_PEELLOOPSCONTAININGTRANSPOSE
#include "bishengir/Dialect/Vector/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;

namespace {

struct PeelLoopsContainingTransposePass
    : public impl::PeelLoopsContainingTransposeBase<
        PeelLoopsContainingTransposePass> {
  using PeelLoopsContainingTransposeBase<PeelLoopsContainingTransposePass>::
      PeelLoopsContainingTransposeBase;
  void runOnOperation() override;
};

} // namespace

/// In subsequent NormalizeVector pass will convert transpose vector.transfer_read into
/// vector.gather, this conversion requires static memref type. This pass peel the parent
/// scf.ForOp of vector.transfer_read to guarantee static memref types.
void PeelLoopsContainingTransposePass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  funcOp.walk([&](vector::TransferReadOp readOp) {
    auto permMap = readOp.getPermutationMap();
    // we don't touch the ones with (1) identity, (2) broadcast (constant)
    if (!permMap.isPermutation() || permMap.isIdentity() || permMap.isConstant())
      return;

    auto memrefType = dyn_cast<MemRefType>(readOp.getSource().getType());
    if (!(memrefType && memrefType.hasRank()))
      return;
    auto destType = dyn_cast<VectorType>(readOp.getResult().getType());
    if (!(destType && destType.hasStaticShape()))
      return;

    scf::ForOp partialIteration;
    scf::ForOp target = readOp->getParentOfType<scf::ForOp>();
    if (target) {
      std::ignore =
          scf::peelForLoopAndSimplifyBounds(rewriter, target, partialIteration);
    }
  });
}

std::unique_ptr<Pass> vector::createPeelLoopsContainingTransposePass() {
  return std::make_unique<PeelLoopsContainingTransposePass>();
}
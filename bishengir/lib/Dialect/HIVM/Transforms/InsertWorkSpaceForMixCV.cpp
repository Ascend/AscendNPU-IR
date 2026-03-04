//===-------------------- InsertWorkSpaceForMixCV.cpp----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts workspace for mix cv function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTWORKSPACEFORMIXCV
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "insert-workspace"

namespace {
struct InsertWorkSpaceForMixCVPass
    : public impl::InsertWorkSpaceForMixCVBase<InsertWorkSpaceForMixCVPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

// Match CC/CV/VC/VV junction and replace emptyOp with workspace
// CC: mmadL1 -> fixpipe -> load -> mmadL1
// CV: mmadL1 -> fixpipe -> load -> vector
// VC: vector -> store -> load -> mmadL1
// VV: vector -> store -> load -> vector
struct InsertWorkSpace : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto src = loadOp.getSrc();
    auto maybeStoreDefOp = traceDefOp<hivm::StoreOp>(src);
    auto maybeSrcDefiningOp = maybeStoreDefOp.has_value()
                                  ? maybeStoreDefOp
                                  : traceDefOp<hivm::FixpipeOp>(src);
    if (!maybeSrcDefiningOp.has_value()) {
      return failure();
    }

    auto srcDefiningOp = maybeSrcDefiningOp.value();
    auto gmStoreOp = cast<DestinationStyleOpInterface>(srcDefiningOp);
    auto emptyDefOp = traceDefOp<tensor::EmptyOp>(gmStoreOp.getDpsInits()[0]);
    if (!emptyDefOp.has_value()) {
      return failure();
    }

    auto emptyOp = emptyDefOp.value();
    auto dstType = cast<ShapedType>(emptyOp->getResultTypes()[0]);
    rewriter.setInsertionPoint(emptyOp);
    auto dstTensor =
        getLocalWorkSpaceTensor(rewriter, emptyOp->getLoc(), dstType.getShape(),
                                getElementTypeOrSelf(dstType));
    rewriter.replaceAllUsesWith(emptyOp->getResult(0), dstTensor);

    return success();
  }
};

void InsertWorkSpaceForMixCVPattern(RewritePatternSet &patterns) {
  patterns.add<InsertWorkSpace>(patterns.getContext());
}

void InsertWorkSpaceForMixCVPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto context = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(context);
  InsertWorkSpaceForMixCVPattern(patterns);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createInsertWorkSpaceForMixCVPass() {
  return std::make_unique<InsertWorkSpaceForMixCVPass>();
}
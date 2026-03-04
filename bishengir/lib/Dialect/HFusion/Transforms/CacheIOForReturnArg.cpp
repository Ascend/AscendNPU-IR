//===---------------------- CacheIOForReturnArg.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cache-io"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_CACHEIOFORRETURNARG
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

OpOperand *traceReshapeAndRewriteInverse(PatternRewriter &rewriter,
                                         OpOperand *outValue) {
  DenseSet<Operation *> createdOperations;
  while (true) {
    bool insideImportantRegion =
        llvm::any_of(outValue->get().getUsers(), [](Operation *user) {
          return opfusion::FusibleHelper::isImportantPattern(user);
        });
    // Time to stop if
    if (insideImportantRegion) {
      break;
    }
    bool reshapePropagated = false;
    // %arg
    // %expanded = expand_shape %arg
    // return %arg
    //
    // |
    // V
    //
    // %arg
    // %expanded = expand_shape %arg
    // %inverse_expand = collapse_shape %expanded
    // return %inverse_expand

    auto outUsers = outValue->get().getUsers();
    for (Operation *user : outUsers) {
      if (createdOperations.contains(user))
        continue;
      Operation *inverseReshape = nullptr;
      rewriter.setInsertionPoint(outValue->getOwner());
      if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(user)) {
        // Expand shape here
        inverseReshape =
            tensor::reshape_utils::createExpandInverse(rewriter, expandOp);
      }
      if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(user)) {
        inverseReshape =
            tensor::reshape_utils::createCollapseInverse(rewriter, collapseOp);
      }
      if (!inverseReshape)
        continue;
      createdOperations.insert(inverseReshape);
      // Set the opOperands with the result of expand
      Value inverseRes = inverseReshape->getResult(0);
      rewriter.modifyOpInPlace(outValue->getOwner(),
                               [&]() { outValue->set(inverseRes); });
      // Old out value, replace the value with the inverse

      // %arg
      // %expanded = expand_shape %arg
      // %inverse_expand = collapse_shape %expanded
      // return [Replace this from %arg to %inverse_expand]

      for (auto &inverseOpr : inverseReshape->getOpOperands()) {
        // %arg
        // [user == %expanded] = expand_shape %arg
        // %inverse_expand = collapse_shape [outValue := %expanded]
        // return %inverse_expand
        // Get the source OpOperand
        outValue = &inverseOpr;
        LDBG("new outValue is " << *outValue->getOwner() << "\n");
      }
      reshapePropagated = true;
      break;
    }
    LDBG(*outValue->getOwner()->getParentOp());
    if (!reshapePropagated)
      break;
  }
  return outValue;
}

struct CacheIOForReturnArgPattern : public OpRewritePattern<func::ReturnOp> {
public:
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const final {
    if (llvm::all_of(returnOp->getOperands(), [](Value operand) {
          return isa<BlockArgument>(operand);
        })) {
      return rewriter.notifyMatchFailure(
          returnOp, "no need to cache io when return values are all arguments");
    }

    SmallVector<OpOperand *> returnArgs;
    for (OpOperand &returnValue : returnOp->getOpOperands()) {
      if (!isa<BlockArgument>(returnValue.get())) {
        continue;
      }
      returnArgs.push_back(&returnValue);
    }

    if (returnArgs.empty()) {
      return rewriter.notifyMatchFailure(returnOp,
                                         "no arg is returned directly");
    }

    for (OpOperand *returnArg : returnArgs) {
      auto returnIdx = returnArg->getOperandNumber();
      BlockArgument ba = cast<BlockArgument>(returnArg->get());

      // Create reshapes following the inverse order of other use chain from
      // returned arg.
      // e.g.
      //   trace 1: %arg0 -> return %arg0
      //   trace 2: %arg0 -> %collapse_arg0 -> elemwise %collapse_arg0 -> return
      // traceReshapeAndRewriteInverse will modify `trace 1` into:
      //   trace new: %arg0 -> collapse %arg0 -> expand %collapse_arg0 -> return
      //   where the `expand %collapse_arg0` is the inverse of `collapse %arg0`
      // and traceReshapeAndRewriteInverse will return OpOperande:
      //   expand %collapse_arg0
      // based on this, we can do cache io:
      //   trace cache io: %arg0 -> %collapse_arg0 -> %load -> %store ->
      //                   expand store -> return
      OpOperand *tracedInsideRegion =
          traceReshapeAndRewriteInverse(rewriter, returnArg);
      SmallPtrSet<Operation *, 4> newOps;
      Operation *outOp = tracedInsideRegion->getOwner();
      Location loc = returnOp->getLoc();
      Value inValue = tracedInsideRegion->get();

      // create load based on the src of inversed operand
      rewriter.setInsertionPointAfterValue(inValue);
      auto emptyForLoad = utils::createEmptyOp(rewriter, loc, inValue);
      auto loadOp = rewriter.create<hfusion::LoadOp>(loc, ValueRange{inValue},
                                                     ValueRange{emptyForLoad});
      Value loadResult = loadOp->getResult(0);

      // create store from load result
      auto emptyForStore = utils::createEmptyOp(rewriter, loc, inValue);
      auto storeOp = rewriter.create<hfusion::StoreOp>(
          loc, ValueRange{loadResult}, ValueRange{emptyForStore});
      // mark return number
      auto resultOprNumAttr = rewriter.getI64IntegerAttr(returnIdx);
      storeOp->setAttr(hfusion::ReturnOperandNumAttr::name, resultOprNumAttr);
      Value storeResult = storeOp->getResult(0);

      // update inversed operand to store result
      rewriter.modifyOpInPlace(outOp,
                               [&]() { tracedInsideRegion->set(storeResult); });
      // mark cached io attr
      func::FuncOp func = returnOp->getParentOfType<func::FuncOp>();
      MLIRContext *ctx = func->getContext();
      func.setArgAttr(ba.getArgNumber(), hacc::CachedIOAttr::name,
                      UnitAttr::get(ctx));
      func.setResultAttr(returnIdx, hacc::CachedIOAttr::name,
                         UnitAttr::get(ctx));
    }
    return success();
  }
};

struct CacheIOForReturnArgPass
    : public impl::CacheIOForReturnArgBase<CacheIOForReturnArgPass> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<CacheIOForReturnArgPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createCacheIOForReturnArg() {
  return std::make_unique<CacheIOForReturnArgPass>();
}

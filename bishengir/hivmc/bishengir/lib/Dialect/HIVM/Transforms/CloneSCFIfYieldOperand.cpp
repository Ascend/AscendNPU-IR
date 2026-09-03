//===- HIVMCloneSCFIfYieldOperand.cpp - Clone SCF If Yield Operand Pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_CLONESCFIFYIELDOPERAND
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-clone-scf-if-yield-operand"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
bool isYieldValueStaticShapeTensor(Value yieldValue) {
  auto tensorType = dyn_cast<TensorType>(yieldValue.getType());
  if (!tensorType) {
    LDBG("yield value: " << yieldValue << " is not a tensor");
    return false;
  }
  if (ShapedType::isDynamicShape(tensorType.getShape())) {
    LDBG("yield value: " << yieldValue << " has dynamic dims");
    return false;
  }
  return true;
}

void cloneYieldValue(PatternRewriter &rewriter, scf::YieldOp yieldOp, int idx) {
  auto yieldValue = yieldOp->getOperand(idx);
  rewriter.setInsertionPoint(yieldOp);
  Value dstValue =
      utils::createEmptyOp(rewriter, yieldOp->getLoc(), yieldValue);
  auto copyOp =
      rewriter.create<hivm::CopyOp>(yieldOp->getLoc(), yieldValue.getType(),
                                    /*src*/ yieldValue, /*dst*/ dstValue);
  rewriter.modifyOpInPlace(yieldOp, [&]() {
    yieldOp.getResultsMutable()[idx].assign(copyOp.getResult(0));
  });
}

/// This pass clones scf.if yield operand if yield operands are same or it is
/// used after this scf.if, avoid inplace by PlanMemory.
struct CloneSCFIfYieldOperandPass
    : public impl::CloneSCFIfYieldOperandBase<CloneSCFIfYieldOperandPass> {
  void runOnOperation() override;
};

class CloneSameYieldOperandsPattern : public OpRewritePattern<scf::YieldOp> {
  using OpRewritePattern<scf::YieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_nonnull<scf::IfOp>(yieldOp->getParentOp())) {
      return failure();
    }
    llvm::MapVector<Value, SmallVector<size_t>> yieldValueToIndices;
    size_t yieldSize = yieldOp->getOperands().size();
    for (size_t i = 0; i < yieldSize; i++) {
      auto yieldValue = yieldOp->getOperand(i);
      if (isYieldValueStaticShapeTensor(yieldValue)) {
        yieldValueToIndices[yieldValue].push_back(i);
      }
    }
    bool modified = false;
    for (auto &pair : yieldValueToIndices) {
      // Check whether same yield values exist
      if (pair.second.size() <= 1) {
        continue;
      }
      // Replace the same yield values with the new ones
      for (size_t i = 1; i < pair.second.size(); i++) {
        // Add copy of the same yield values before scf.if yield.
        // Copy times is one less than the number of same yield operands.
        cloneYieldValue(rewriter, yieldOp, pair.second[i]);
        modified = true;
      }
    }
    return success(modified);
  }
};

class CloneYieldOperandUseAfterSCFIfPattern
    : public OpRewritePattern<scf::YieldOp> {
  using OpRewritePattern<scf::YieldOp>::OpRewritePattern;

  // Example:
  //
  // 1 %a = tensor.empty()
  // 2 %res = if () {
  // 3   %b = vadd(%a, %cst_1)
  // 4   yield %b
  // 5 } else {
  // 6   yield %a
  // 7 } (after ifOp, we can't read %a, because %a will alias with %b and %a
  //      maybe be modified in line 3)
  // 8 use %a

  // if %a have user(line 8) after ifOp, we need to copy %b before line 4 and
  // change the yield value. so that %a will alias with %b' and we will not
  // modify init value of %a.

  // Example after clone yield value:
  //
  // 1 %a = tensor.empty()
  // 2 %res = if () {
  // 3   %b = vadd(%a, %cst_1)
  // 4   %b' = copy ins(%b) outs(%b')
  // 5   yield %b'
  // 6 } else {
  // 7   yield %a
  // 8 }
  // 9 use %a

  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_nonnull<scf::IfOp>(yieldOp->getParentOp())) {
      return failure();
    }
    auto yieldSize = yieldOp->getOperands().size();
    bool modified = false;
    for (size_t i = 0; i < yieldSize; i++) {
      auto yieldValue = yieldOp->getOperand(i);
      if (!isYieldValueStaticShapeTensor(yieldValue)) {
        continue;
      }
      auto *defOp = yieldValue.getDefiningOp();
      // defOp is null means yieldvalue is block argument, this case will be
      // handle in CloneYieldOperandAliasByForOperand.
      if (!defOp) {
        continue;
      }
      // For the case like that the yieldValue is defined in an upper-level
      // block, and it will be used after ifOp(yieldOp->getParentOp()), ifOp
      // will properlyDominates user.
      // If user is in ifOp (we don't care user is in thenblock or elseblock),
      // ifOp will also properlyDominates user. But if yieldValue will not be
      // used after ifOp, even though the init yieldvalue will be dirty, but we
      // don't read the it. So we don't need copy it, too.
      DominanceInfo domInfo;
      bool needCopyYieldValue = llvm::any_of(
          yieldValue.getUsers(), [&domInfo, &yieldOp](Operation *user) {
            // user is not in if op and if Op properly dominates user.
            return user->getParentOp() != yieldOp->getParentOp() &&
                   domInfo.properlyDominates(yieldOp->getParentOp(), user);
          });
      if (needCopyYieldValue && !isa<hivm::CopyOp>(defOp)) {
        cloneYieldValue(rewriter, yieldOp, i);
        modified = true;
      }
    }
    return success(modified);
  }
};

class CloneYieldOperandAliasByForOperand : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  // Example:
  // 1  %a = tensor.empty()
  // 2  scf.for i iter_arg(%arg0 = %a)
  // 3    %res = if () {
  // 4      yield %arg0
  // 5    } else {
  // 6      write %c (may read %arg0, so we need clone yield value)
  // 7      yield %c
  // 8    }
  // 9    yield %res
  // 10 }

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_nonnull<scf::ForOp>(ifOp->getParentOp())) {
      return failure();
    }
    auto &blockYieldOp = ifOp->getBlock()->back();
    auto forOp = cast<scf::ForOp>(ifOp->getParentOp());
    // map %res to 0(index)
    llvm::DenseMap<Value, size_t> blockYieldValue2Index;
    for (auto [idx, blockYieldValue] :
         llvm::enumerate(blockYieldOp.getOperands())) {
      blockYieldValue2Index[blockYieldValue] = idx;
    }
    bool modified = false;
    auto ifResultSize = ifOp.getResults().size();

    for (size_t i = 0; i < ifResultSize; i++) {
      auto resValue = ifOp->getResult(i);
      // map contain %res map[%res] will get an iter_arg idx
      if (!blockYieldValue2Index.contains(resValue)) {
        continue;
      }
      // get iter_arg
      auto iterArgValue =
          forOp.getRegionIterArg(blockYieldValue2Index[resValue]);
      modified |= checkAndClone(rewriter, ifOp.thenYield(), i, iterArgValue);
      modified |= checkAndClone(rewriter, ifOp.elseYield(), i, iterArgValue);
    }
    return success(modified);
  }

  bool checkAndClone(PatternRewriter &rewriter, scf::YieldOp yieldOp,
                     size_t idx, Value iterArgValue) const {
    auto yieldValue = yieldOp->getOperand(idx);
    if (isYieldValueStaticShapeTensor(yieldValue) &&
        iterArgValue != yieldValue) {
      // if yieldValue is not iter_arg
      auto *defOp = yieldValue.getDefiningOp();
      if (!defOp || !isa<hivm::CopyOp>(defOp)) {
        cloneYieldValue(rewriter, yieldOp, idx);
        return true;
      }
    }
    return false;
  }
};

void CloneSCFIfYieldOperandPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<CloneYieldOperandUseAfterSCFIfPattern,
               CloneSameYieldOperandsPattern,
               CloneYieldOperandAliasByForOperand>(ctx);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createCloneSCFIfYieldOperandPass() {
  return std::make_unique<CloneSCFIfYieldOperandPass>();
}

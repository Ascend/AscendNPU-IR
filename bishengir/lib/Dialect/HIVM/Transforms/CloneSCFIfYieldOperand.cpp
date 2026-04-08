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
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

bool checkUseAfterOverWrite(Operation *WriteDefOp, Value usedYieldValue,
                            scf::YieldOp usedYieldOp) {
  auto *moduleBlock = utils::getTopLevelModuleOp(WriteDefOp).getBody();
  // trace all blocks that contains ifOp until moduleBlock.
  DenseMap<Block *, Operation *> block2Op;
  for (auto *op = WriteDefOp; op != nullptr && op->getBlock() != moduleBlock;
       op = op->getParentOp()) {
    block2Op.try_emplace(op->getBlock(), op);
  }
  return llvm::any_of(
      usedYieldValue.getUsers(),
      [block2Op, moduleBlock, usedYieldOp](Operation *user) {
        if (user == usedYieldOp) {
          return false;
        }
        // trace all blocks that contains user until funcBlock.
        for (auto *op = user; op != nullptr; op = op->getParentOp()) {
          auto it = block2Op.find(op->getBlock());
          if (op->getBlock() != moduleBlock && it != block2Op.end()) {
            auto *currentOp = it->second;
            return currentOp->isBeforeInBlock(op);
          }
        }
        return false;
      });
}

// After writeYieldValue defined, there should not be other Op use
// usedYieldValue, otherwise need clone.
bool checkNeedClone(Value writeYieldValue, Value usedYieldValue,
                    scf::YieldOp usedYieldOp) {
  if (!isYieldValueStaticShapeTensor(writeYieldValue)) {
    return false;
  }
  auto *WriteDefOp = writeYieldValue.getDefiningOp();
  if (!WriteDefOp) {
    // It means that WriteDefOp is block argument, we will not write it until
    // yield value.
    return false;
  } else if (isa<bufferization::ToTensorOp>(WriteDefOp)) {
    // Whether to clone buffer need to be further discussed
    return false;
  } else if (isa<scf::IfOp>(WriteDefOp)) {
    auto idx = cast<OpResult>(writeYieldValue).getResultNumber();
    auto overWriteIfOp = cast<scf::IfOp>(WriteDefOp);
    return checkNeedClone(overWriteIfOp.thenYield().getOperand(idx),
                          usedYieldValue, usedYieldOp) ||
           checkNeedClone(overWriteIfOp.elseYield().getOperand(idx),
                          usedYieldValue, usedYieldOp);
  } else if (isa<scope::ScopeOp>(WriteDefOp)) {
    auto idx = cast<OpResult>(writeYieldValue).getResultNumber();
    auto writeScopeOp = cast<scope::ScopeOp>(WriteDefOp);
    return checkNeedClone(writeScopeOp.getBody()->back().getOperand(idx),
                          usedYieldValue, usedYieldOp);
  } else if (isa<scf::ForOp>(WriteDefOp)) {
    auto idx = cast<OpResult>(writeYieldValue).getResultNumber();
    auto writeForOp = cast<scf::ForOp>(WriteDefOp);
    return checkNeedClone(writeForOp.getInitArgs()[idx], usedYieldValue,
                          usedYieldOp);
  } else if (isa<tensor::ExtractSliceOp>(WriteDefOp)) {
    auto writeExtractSliceOp = cast<tensor::ExtractSliceOp>(WriteDefOp);
    return checkNeedClone(writeExtractSliceOp.getSource(), usedYieldValue,
                          usedYieldOp);
  } else if (isa<tensor::ExpandShapeOp>(WriteDefOp)) {
    auto writeExpandShapeOp = cast<tensor::ExpandShapeOp>(WriteDefOp);
    return checkNeedClone(writeExpandShapeOp.getSrc(), usedYieldValue,
                          usedYieldOp);
  }
  return checkUseAfterOverWrite(WriteDefOp, usedYieldValue, usedYieldOp);
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
  LDBG("clone yield value: " << yieldValue);
}

/// This pass clones scf.if yield operand if yield operands are same or it is
/// used after this scf.if, avoid inplace by PlanMemory.
struct CloneSCFIfYieldOperandPass
    : public impl::CloneSCFIfYieldOperandBase<CloneSCFIfYieldOperandPass> {
  void runOnOperation() override;
};

class CloneSameYieldOperandsPattern : public OpRewritePattern<scf::YieldOp> {
public:
  explicit CloneSameYieldOperandsPattern(MLIRContext *context,
                                         PatternBenefit benefit = 1)
      : OpRewritePattern<scf::YieldOp>(context, benefit) {}
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

class CloneSCFIfYieldOperandUseAfterWritePattern
    : public OpRewritePattern<scf::IfOp> {
public:
  explicit CloneSCFIfYieldOperandUseAfterWritePattern(
      MLIRContext *context, PatternBenefit benefit = 100)
      : OpRewritePattern<scf::IfOp>(context, benefit) {}
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

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
  // 4   yield %b
  // 5 } else {
  // 6   %a' = copy ins(%a) outs(%a')
  // 7   yield %a'
  // 8 }
  // 9 use %a

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp.getResults().empty()) {
      return failure();
    }
    auto modified1 = copyYieldOperandUseAfterSCFIf(rewriter, ifOp.thenYield(),
                                                   ifOp.elseYield());
    auto modified2 = copyYieldOperandUseAfterSCFIf(rewriter, ifOp.elseYield(),
                                                   ifOp.thenYield());
    return success(modified1 || modified2);
  }

  bool checkDefOutOfFor(Value usedYieldValue, scf::YieldOp currBrYieldOp) const {
    auto *usedDefOp = usedYieldValue.getDefiningOp();
    if (!usedDefOp) {
      return false;
    }
    auto parentForOp = currBrYieldOp->getParentOfType<scf::ForOp>();
    DominanceInfo domInfo;
    return parentForOp && domInfo.properlyDominates(usedDefOp, parentForOp, false);
  }
  bool copyYieldOperandUseAfterSCFIf(PatternRewriter &rewriter,
                                     scf::YieldOp writeYieldOp,
                                     scf::YieldOp currBrYieldOp) const {
    auto yieldSize = writeYieldOp->getOperands().size();
    bool modified = false;
    for (size_t i = 0; i < yieldSize; i++) {
      auto writeYieldValue = writeYieldOp->getOperand(i);
      auto currBrYieldValue = currBrYieldOp->getOperand(i);
      if (!isYieldValueStaticShapeTensor(currBrYieldValue)) {
        continue;
      }
      // 1  %a = tensor.empty()
      // 2  %res = if () {
      // 3    write %c
      // 4    use %a
      // 5    yield %c
      // 6  } else {
      // 7    write %c
      // 8    use %a
      // 9    yield %c
      // 10 }
      // 11 %res1 = if () {
      // 12   yield %res
      // 13 } else {
      // 14   yield %a
      // 15 }

      // If writeYieldValue is result of IfOp, and write %c will modify %res.
      // so that checkNeedClone will find op that write %res recursively. And
      // after this Op, there should not be other op use %a
      if (checkDefOutOfFor(currBrYieldValue, currBrYieldOp) ||
          checkNeedClone(writeYieldValue, currBrYieldValue, currBrYieldOp)) {
        cloneYieldValue(rewriter, currBrYieldOp, i);
        modified = true;
      }
    }
    return modified;
  }
};

class CloneSCFForYieldOperandUseAfterWritePattern
    : public OpRewritePattern<scf::ForOp> {
public:
  explicit CloneSCFForYieldOperandUseAfterWritePattern(
      MLIRContext *context, PatternBenefit benefit = 100)
      : OpRewritePattern<scf::ForOp>(context, benefit) {}
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  // Example:
  // 1  %a = tensor.empty()
  // 2  scf.for i iter_arg(%arg0 = %a) {
  // 3    write %res
  // 4    use %arg0
  // 5    copy %res to %res'(need clone here)
  // 6    yield %res'
  // 7  }

  // After writing %res, we can't use %arg0 anymore, otherwise we need to clone
  // %res. In fact, similar work also occurs in NormalizeIterUseAfterYieldInit
  // pattern of planmemory, but it seems that this pattern don't work properly
  // in some scenarios. When I try to clone yield operand for SCFIf, I find my
  // solution may also work for cloning operand for SCFFor. So I add this
  // pattern and will integrate this pattern and NormalizeIterUseAfterYieldInit
  // pattern of planmemory in furture.

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto forResultSize = forOp->getResults().size();
    if (forResultSize == 0) {
      return failure();
    }
    auto forYieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->back());
    bool modified = false;
    for (size_t i = 0; i < forResultSize; i++) {
      auto writeYieldValue = forYieldOp.getOperand(i);
      auto iterArg = forOp.getRegionIterArgs()[i];
      if (!isYieldValueStaticShapeTensor(iterArg)) {
        continue;
      }
      if (checkNeedClone(writeYieldValue, iterArg, forYieldOp)) {
        cloneYieldValue(rewriter, forYieldOp, i);
        modified = true;
      }
    }
    return success(modified);
  }
};

// Example:
// 1  %a = tensor.empty()
// 2  scf.for i iter_arg(%arg0 = %a)
// 3    %res = if () {
// 4      copy %arg0 to %arg0'
// 5      yield %arg0'
// 6    } else {
// 7      write %c
// 8      use %arg0(is dirty)
// 9      yield %c
// 10   }
// 11   copy %res to %res'(need clone here)
// 12   yield %res'
// 13}

// CloneSCFIfYieldOperandUseAfterWritePattern will clone %arg0, and
// CloneSCFForYieldOperandUseAfterWritePattern will clone %res.

void CloneSCFIfYieldOperandPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<CloneSCFIfYieldOperandUseAfterWritePattern,
               CloneSCFForYieldOperandUseAfterWritePattern,
               CloneSameYieldOperandsPattern>(ctx);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createCloneSCFIfYieldOperandPass() {
  return std::make_unique<CloneSCFIfYieldOperandPass>();
}

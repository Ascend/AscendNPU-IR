//===- DecomposeMulti.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include <numeric>
namespace mlir {
#define GEN_PASS_DEF_DECOMPOSEMULTI
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir
#define DEBUG_TYPE "hfusion-decompose-multi"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::utils::debugger;

namespace {

using OutlineIndices = std::set<int64_t>;

template <class LinalgType>
struct DecomposeMultiPattern : public OpRewritePattern<LinalgType> {
  using OpRewritePattern<LinalgType>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgType linalgOp,
                                PatternRewriter &rewriter) const override {
    assert(isa<linalg::LinalgOp>(linalgOp.getOperation()) &&
           "LinalgType must be derived from linalg::LinalgOp");
    unsigned inputNum = static_cast<unsigned>(linalgOp.getNumDpsInputs());
    unsigned outputNum = static_cast<unsigned>(linalgOp.getNumDpsInits());
    if (linalgOp->getNumResults() == 1)
      return failure();
    assert(outputNum == linalgOp->getNumResults());
    LLVM_DEBUG(llvm::dbgs() << "OK checking " << linalgOp << "\n";);
    // Assert that the number of block arguments is twice the number of inputs
    if (linalgOp->getRegions().size() != 1)
      return failure();
    if (!linalgOp->getRegions().front().hasOneBlock())
      return failure();
    Block &block = linalgOp->getRegions().front().getBlocks().front();
    // Create a bitvector to mark which inputs/outputs are extractable
    llvm::SmallBitVector extractable(linalgOp.getNumDpsInits(), true);
    SmallVector<SmallVector<int64_t>> extractIndexing(
        linalgOp.getNumDpsInits());
    // Check if each result is extractable
    for (unsigned i = 0; i < outputNum; ++i) {
      Value result = block.getTerminator()->getOperand(i);
      auto *definer = result.getDefiningOp();
      if (!definer) {
        LLVM_DEBUG(llvm::dbgs() << "defining op not found\n";);
        extractable[i] = false;
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "Checking definer " << *definer << " for " << i
                              << "\n";);
      if (!definer->hasOneUse()) {
        LLVM_DEBUG(llvm::dbgs() << " definer has multiple use\n";);
        extractable[i] = false;
        continue;
      }
      // Check if the user operation only uses block arguments i and i +
      // inputNum
      for (Value operand : definer->getOperands()) {
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          unsigned argIndex = blockArg.getArgNumber();
          if (!blockArg.hasOneUse()) {
            extractable[i] = false;
            LLVM_DEBUG(llvm::dbgs() << blockArg << " has multiple use\n";);
            break;
          }
          extractIndexing[i].push_back(argIndex);
        } else {
          LLVM_DEBUG(llvm::dbgs() << operand << " operand not a block arg";);
          extractable[i] = false;
          break;
        }
      }
    }

    // If nothing is extractable, return failure
    if (extractable.none())
      return failure();

    SmallVector<int64_t> remainingMex(linalgOp->getNumOperands());
    std::iota(remainingMex.begin(), remainingMex.end(), 0);
    OutlineIndices remainingIndices;
    remainingIndices.insert(remainingMex.begin(), remainingMex.end());

    // Create new linalg operations for extractable inputs/outputs
    for (unsigned i = 0; i < extractable.size(); ++i) {
      if (extractable[i]) {
        LLVM_DEBUG(llvm::dbgs() << "Extracting " << i << "\n";);
        OutlineIndices tmpSet;
        tmpSet.insert(i + inputNum);
        remainingIndices.erase(i + inputNum);
        for (auto inputIndex : extractIndexing[i]) {
          tmpSet.insert(inputIndex);
          remainingIndices.erase(inputIndex);
          LLVM_DEBUG(llvm::dbgs() << "got inputIndex " << inputIndex << "\n";);
          assert(inputIndex < inputNum || inputIndex == i + inputNum);
        }
        outline(rewriter, linalgOp, tmpSet);
      }
    }

    // If not all inputs/outputs were extracted, create a new op with the
    // remaining ones
    if (!remainingIndices.empty())
      outline(rewriter, linalgOp, remainingIndices);
    return success();
  }

  // Helper function to outline a new operation
  void outline(PatternRewriter &rewriter, linalg::LinalgOp linalgOp,
               const OutlineIndices &indices) const {
    LLVM_DEBUG(llvm::dbgs() << to_string(indices) << "\n";);
    LLVM_DEBUG(llvm::dbgs()
                   << linalgOp << " processing this linalg operation\n";);
    auto inputNum = linalgOp.getNumDpsInputs();
    SmallVector<Value> newOperands;
    SmallVector<Type> newResultTypes;

    LLVM_DEBUG(llvm::dbgs() << "\n --Extracting operands and result types\n";);
    for (auto i : indices) {
      if (i < inputNum) {
        newOperands.push_back(linalgOp.getDpsInputOperand(i)->get());
      } else {
        newOperands.push_back(linalgOp.getDpsInitOperand(i - inputNum)->get());
        newResultTypes.push_back(linalgOp->getResult(i - inputNum).getType());
      }
      LLVM_DEBUG(llvm::dbgs() << "Processing index " << i << ": "
                              << newOperands.back() << "\n\n";);
    }
    LLVM_DEBUG(llvm::dbgs() << "\n -- ok start cloning -- \n";);
#ifndef NDEBUG
    for (auto op : newOperands) {
      LLVM_DEBUG(llvm::dbgs() << "-- op: " << op << "\n";);
    }
#endif
    auto *clonedOp =
        clone(rewriter, linalgOp.getOperation(), newResultTypes, newOperands);
    auto newSingleOp = cast<linalg::LinalgOp>(clonedOp);

    auto &newBlock = newSingleOp->getRegions().front().getBlocks().front();

    // Modify the yield operation
    Operation *yieldOp = newBlock.getTerminator();
    SmallVector<Value> newYieldOperands;
    LLVM_DEBUG(llvm::dbgs() << to_string(indices) << "\n";);
    for (auto i : indices) {
      if (i >= inputNum) {
        LLVM_DEBUG(llvm::dbgs()
                       << "Ok getting operand " << i - inputNum << "\n";);
        newYieldOperands.push_back(yieldOp->getOperand(i - inputNum));
      }
    }
    yieldOp->setOperands(newYieldOperands);
    LLVM_DEBUG(llvm::dbgs() << "Creating new " << *newSingleOp << "\n";);
    // Remove unused arguments
    auto &ops = newBlock.getOperations();

    for (auto it = ops.rbegin(); it != ops.rend();) {
      Operation *op = &*it;
      ++it; // Advance the iterator before potentially erasing

      LLVM_DEBUG(llvm::dbgs() << *op << "\n";);
      if (isOpTriviallyDead(op)) {
        LLVM_DEBUG(llvm::dbgs() << "dead and erasing\n";);
        op->erase();
      }
    }
    for (int i = static_cast<int>(newBlock.getNumArguments()) - 1; i >= 0;
         --i) {
      if (!indices.count(i)) {
        newBlock.eraseArgument(i);
      }
    }
    // Replace the old results with the new ones
    int newResultIndex = 0;
    for (auto i : indices) {
      if (i >= inputNum) {
        linalgOp->getResult(i - inputNum)
            .replaceAllUsesWith(newSingleOp->getResult(newResultIndex++));
      }
    }
    if (isa<linalg::GenericOp>(newSingleOp)) {
      auto indexingMaps = newSingleOp.getIndexingMapsArray();
      SmallVector<AffineMap> newIndexingMaps;
      newIndexingMaps.reserve(indices.size());
      int32_t inputCount = 0;
      int32_t outputCount = 0;
      for (auto i : indices) {
        newIndexingMaps.push_back(indexingMaps[i]);
        if (i >= inputNum)
          outputCount++;
        else
          inputCount++;
      }
      newSingleOp->setAttr("indexing_maps",
                           rewriter.getAffineMapArrayAttr(newIndexingMaps));
      newSingleOp->setAttr(
          "operandSegmentSizes",
          rewriter.getDenseI32ArrayAttr({inputCount, outputCount}));
    }
    LLVM_DEBUG(llvm::dbgs() << "The new single op " << *newSingleOp << "\n";);
  }
};

struct DecomposeMultiPass
    : public impl::DecomposeMultiBase<DecomposeMultiPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<DecomposeMultiPattern<linalg::ReduceOp>>(context);
    patterns.add<DecomposeMultiPattern<linalg::GenericOp>>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createDecomposeMulti() {
  return std::make_unique<DecomposeMultiPass>();
}

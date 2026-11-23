//===----------------------- AnalyzeArithVectorMask.cpp--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

#define DEBUG_TYPE "arith-vector-mask-analyze"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_ANALYZEARITHVECTORMASK
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::math;

namespace {
struct ArithVectorMaskAnalysisPass
    : public impl::AnalyzeArithVectorMaskBase<ArithVectorMaskAnalysisPass> {
  using AnalyzeArithVectorMaskBase<
      ArithVectorMaskAnalysisPass>::AnalyzeArithVectorMaskBase;

public:
  void runOnOperation() override;
};
} // namespace

// Mark the value could reach to which mask op
void markReachableInfo(Value val, Operation *maskOp, int idx,
                       IRRewriter &rewriter) {
  if (idx < 0 || !val)
    return;
  auto valDefOp = val.getDefiningOp();
  annotation::MarkOp mark;
  if (utils::getAnnotateOpWithAttr(val, utils::reachedMaskOpsIdx)) {
    // If the value has annotation, append the reachable mask op index
    mark = dyn_cast<annotation::MarkOp>(
        utils::getAnnotateOpWithAttr(val, utils::reachedMaskOpsIdx).value());
    auto markedIdxArray =
        mark->template getAttrOfType<ArrayAttr>(utils::reachedMaskOpsIdx)
            .getValue();
    SmallVector<int32_t> idxs = {idx};
    for (auto &attr : markedIdxArray) {
      int id = dyn_cast<IntegerAttr>(attr).getInt();
      if (id != idx)
        idxs.push_back(id);
    }
    mark->setAttr(utils::reachedMaskOpsIdx, rewriter.getI32ArrayAttr(idxs));
  } else {
    // If the value has no annotation, create new annotation to record the
    // reachable mask op index
    rewriter.setInsertionPointAfter(valDefOp);
    mark = rewriter.create<annotation::MarkOp>(valDefOp->getLoc(), val);
    mark->setAttr(utils::reachedMaskOpsIdx, rewriter.getI32ArrayAttr({idx}));
  }
}

// Analyze whether the value could reach the mask op with specified index
void analyzeUseAndMark(Value val, Operation *maskOp, int maskOpIdx,
                       IRRewriter &rewriter,
                       llvm::SmallPtrSet<Value, 16> &visited) {
  // Skip already-visited values.
  if (!visited.insert(val).second)
    return;
  auto valDefOp = val.getDefiningOp();
  // If val is not defined by op(from args), or is from
  // CreateMaskOp, ConstantMaskOp, ..., then skip the analysis
  if (!valDefOp || isa<vector::CreateMaskOp>(valDefOp) ||
      isa<vector::ConstantMaskOp>(valDefOp) ||
      isa<arith::ConstantOp>(valDefOp) || isa<vector::LoadOp>(valDefOp) ||
      isa<memref::SubViewOp>(valDefOp) || isa<vector::ReductionOp>(valDefOp)) {
    return;
  }
  if (isa<vector::TransferReadOp>(valDefOp)) {
    auto transferReadOp = dyn_cast<vector::TransferReadOp>(valDefOp);
    // In transferReadOp, mask is optional
    if (!transferReadOp.getMask())
      return;
  }
  markReachableInfo(val, maskOp, maskOpIdx, rewriter);
  // Get val definingOp's operand, analyze and mark recursively
  for (unsigned i = 0; i < valDefOp->getNumOperands(); i++) {
    Value defOpOperand = valDefOp->getOperand(i);
    analyzeUseAndMark(defOpOperand, maskOp, maskOpIdx, rewriter, visited);
  }
}

void ArithVectorMaskAnalysisPass::runOnOperation() {
  func::FuncOp Func = getOperation();
  IRRewriter rewriter(Func.getContext());
  DominanceInfo domInfo;

  // Assign an index to each mask value in one walk. Hoisting and CSE in the
  // pipeline unify equivalent constant masks before this analysis.
  DenseMap<Operation *, int> opToIdx;
  SmallVector<Value> maskValues;
  Func.walk([&](Operation *op) {
    if (!isa<vector::ConstantMaskOp, vector::CreateMaskOp>(op))
      return;
    opToIdx.try_emplace(op, maskValues.size());
    maskValues.push_back(op->getResult(0));
  });

  Func.walk<WalkOrder::PostOrder>([&](Operation *op) {
    // Regard the mask of MaskedStoreOp and TransferWriteOp as the anchor
    Value outVec;
    Value maskOp;
    if (auto maskedStoreOp = dyn_cast<vector::MaskedStoreOp>(*op)) {
      outVec = maskedStoreOp.getValueToStore();
      maskOp = maskedStoreOp.getMask();
    } else if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(*op)) {
      outVec = transferWriteOp.getVector();
      // In transferWriteOp, mask is optional, so here mask may be null
      maskOp = transferWriteOp.getMask();
    }
    if (maskOp) {
      int maskOpIdx = -1;
      annotation::MarkOp maskOpMark;
      if (utils::getAnnotateOpWithAttr(maskOp, utils::maskOpIdx)) {
        maskOpMark = dyn_cast<annotation::MarkOp>(
            utils::getAnnotateOpWithAttr(maskOp, utils::maskOpIdx).value());
        maskOpIdx =
            maskOpMark->template getAttrOfType<IntegerAttr>(utils::maskOpIdx)
                .getValue()
                .getZExtValue();
      } else {
        // If the value has no annotation, create new annotation to record the
        // reachable mask op index
        auto *maskDef = maskOp.getDefiningOp();
        if (maskDef == nullptr) {
          return; // block-arg mask: nothing to annotate/reuse → skip anchor
        }
        auto opIt = opToIdx.find(maskDef);
        maskOpIdx = (opIt != opToIdx.end()) ? opIt->second : -1;
        rewriter.setInsertionPointAfter(maskDef);
        maskOpMark =
            rewriter.create<annotation::MarkOp>(maskOp.getLoc(), maskOp);
        maskOpMark->setAttr(utils::maskOpIdx,
                            rewriter.getI32IntegerAttr(maskOpIdx));
      }
      if (maskOpIdx != -1) {
        llvm::SmallPtrSet<Value, 16> visited;
        analyzeUseAndMark(outVec, maskOp.getDefiningOp(), maskOpIdx, rewriter,
                          visited);
      }
    }
  });

  // If a op has more than one different reached masks, we cannot use one of
  // them but should create a new mask with PgePattern::ALL, so here we erase
  // the reachedMaskOpsIdx. After this stage, all utils::reachedMaskOpsIdx
  // markOp has only one element.
  SmallVector<annotation::MarkOp> reachedMaskMarks;
  Func.walk([&](annotation::MarkOp markOp) {
    if (markOp.isAnnotatedByStaticAttr(utils::reachedMaskOpsIdx))
      reachedMaskMarks.push_back(markOp);
  });

  // Process marks on a use-def chain from use to def. This relies on:
  // 1. Marks being inserted immediately after the marked op.
  // 2. SSA dependencies not crossing blocks under MLIR region barrier ops.
  for (annotation::MarkOp markOp : llvm::reverse(reachedMaskMarks)) {
    if (markOp.isAnnotatedByStaticAttr(utils::reachedMaskOpsIdx)) {
      auto markedIdxArray =
          markOp->template getAttrOfType<ArrayAttr>(utils::reachedMaskOpsIdx)
              .getValue();
      if (markedIdxArray.size() == 1) {
        int reachedMaskIdx = dyn_cast<IntegerAttr>(markedIdxArray[0]).getInt();
        if (reachedMaskIdx < 0 ||
            static_cast<size_t>(reachedMaskIdx) >= maskValues.size()) {
          rewriter.eraseOp(markOp);
          continue;
        }
        // If the annotated value is also used outside the masked data flow,
        // do not mark it. Marking it would cause the LLVM lowering to
        // materialize it under the restricted predicate, and then silently
        // reuse that restricted-predicate result in the unmasked context,
        // leaving the upper lanes of the vector invalid.
        //
        // Note: unlike an earlier revision, a mark is NOT erased when the
        // annotated value properly dominates the mask. An SSA value feeding
        // ops in two sibling blocks always properly dominates those blocks'
        // masks, so that rule would erase the very annotation that makes
        // cross-block mask reuse possible. Keeping the mark is safe:
        // findReuseableMask re-verifies dominance (or same-block clone)
        // before reusing, and an unhonorable mark degrades to a fresh mask.
        Value annotatedVal = markOp.getSrc();
        // Check whether a user still belongs to the same masked data flow as
        // the annotated value.
        auto isUserInSameMaskedFlow = [reachedMaskIdx](Operation *user) {
          // Users with results must stay in the reached-mask data flow.
          if (user->getNumResults() != 0) {
            return static_cast<bool>(utils::getAnnotateOpWithAttr(
                user->getResult(0), utils::reachedMaskOpsIdx));
          }

          // Store-like users have no result, so check whether their mask
          // matches the mask that reached the annotated value.
          Value userMask;
          if (auto maskedStoreOp = dyn_cast<vector::MaskedStoreOp>(user)) {
            userMask = maskedStoreOp.getMask();
          } else if (auto transferWriteOp =
                         dyn_cast<vector::TransferWriteOp>(user)) {
            userMask = transferWriteOp.getMask();
          }
          if (!userMask)
            return false;

          auto userMaskMark =
              utils::getAnnotateOpWithAttr(userMask, utils::maskOpIdx);
          if (!userMaskMark)
            return false;

          auto mark = dyn_cast<annotation::MarkOp>(userMaskMark.value());
          if (!mark)
            return false;
          auto maskIdxAttr =
              mark->template getAttrOfType<IntegerAttr>(utils::maskOpIdx);
          if (!maskIdxAttr)
            return false;

          return static_cast<int>(maskIdxAttr.getInt()) == reachedMaskIdx;
        };
        bool hasUnmaskedUser = false;
        for (Operation *user : annotatedVal.getUsers()) {
          // Skip annotation mark ops themselves.
          if (isa<annotation::MarkOp>(user))
            continue;
          // Any user outside the same masked flow makes this a mixed use.
          if (!isUserInSameMaskedFlow(user)) {
            hasUnmaskedUser = true;
            break;
          }
        }
        if (hasUnmaskedUser) {
          rewriter.eraseOp(markOp);
          continue;
        }
        // Keep: resolve the accumulated array to a single index.
        markOp->setAttr(utils::reachedMaskOpsIdx, markedIdxArray[0]);
        // Record the direct mask reference consumed by findReuseableMask.
        // A same-block mask may follow the annotated value when this analysis
        // runs without hoisting. Move only the mark to keep its operand in
        // scope; lowering still checks dominance before reusing the mask.
        Value mask = maskValues[reachedMaskIdx];
        Operation *maskDef = mask.getDefiningOp();
        if (!domInfo.dominates(maskDef, markOp.getOperation())) {
          if (maskDef->getBlock() != markOp->getBlock())
            continue;
          rewriter.moveOpAfter(markOp.getOperation(), maskDef);
        }
        rewriter.modifyOpInPlace(markOp, [&]() {
          markOp->setAttr(markOp.getKeysAttrName(),
                          rewriter.getStrArrayAttr(utils::reachedMaskValue));
          markOp.getValuesMutable().assign(mask);
        });
      } else {
        rewriter.eraseOp(markOp);
      }
    }
  }
}

std::unique_ptr<Pass> hivm::createArithVectorMaskAnalysisPass() {
  return std::make_unique<ArithVectorMaskAnalysisPass>();
}

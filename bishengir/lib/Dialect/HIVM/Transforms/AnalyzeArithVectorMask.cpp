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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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

// Get all mask ops, including vector.create_mask, vector.constant_mask
static SmallVector<Operation *> getAllMaskOps(func::FuncOp funcOp) {
  SmallVector<Operation *> retVec;
  funcOp->walk([&](Operation *op) {
    if (isa<vector::CreateMaskOp>(op))
      retVec.push_back(op);
    if (isa<vector::ConstantMaskOp>(op))
      retVec.push_back(op);
  });
  return retVec;
}

int getIdxInMaskOps(Value mask) {
  int idx = -1;
  Operation *op = mask.getDefiningOp();
  assert(op!=nullptr);
  func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return idx;
  SmallVector<Operation *> maskOps = getAllMaskOps(funcOp);
  auto it = std::find(maskOps.begin(), maskOps.end(), op);
  if (it != maskOps.end()) {
    idx = it - maskOps.begin();
  }
  return idx;
}

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
                       IRRewriter &rewriter) {
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
    analyzeUseAndMark(defOpOperand, maskOp, maskOpIdx, rewriter);
  }
}

void ArithVectorMaskAnalysisPass::runOnOperation() {
  func::FuncOp Func = getOperation();
  IRRewriter rewriter(Func.getContext());
  std::map<int, annotation::MarkOp> maskOpMarkIdx;
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
        maskOpMarkIdx[maskOpIdx] = maskOpMark;
      } else {
        // If the value has no annotation, create new annotation to record the
        // reachable mask op index
        maskOpIdx = getIdxInMaskOps(maskOp);
        rewriter.setInsertionPointAfter(maskOp.getDefiningOp());
        maskOpMark =
            rewriter.create<annotation::MarkOp>(maskOp.getLoc(), maskOp);
        maskOpMark->setAttr(utils::maskOpIdx,
                            rewriter.getI32IntegerAttr(maskOpIdx));
        maskOpMarkIdx[maskOpIdx] = maskOpMark;
      }
      if (maskOpIdx != -1)
        analyzeUseAndMark(outVec, maskOp.getDefiningOp(), maskOpIdx, rewriter);
    }
  });

  // If a op has more than one different reached masks, we cannot use one of
  // them but should create a new mask with PgePattern::ALL, so here we erase
  // the reachedMaskOpsIdx. After this stage, all utils::reachedMaskOpsIdx
  // markOp has only one element.
  Func.walk([&](annotation::MarkOp markOp) {
    if (markOp.isAnnotatedByStaticAttr(utils::reachedMaskOpsIdx)) {
      auto markedIdxArray =
          markOp->template getAttrOfType<ArrayAttr>(utils::reachedMaskOpsIdx)
              .getValue();
      if (markedIdxArray.size() == 1) {
        auto maskOpMark = maskOpMarkIdx.find(
            dyn_cast<IntegerAttr>(markedIdxArray[0]).getInt());
        DominanceInfo domInfo;
        if (domInfo.properlyDominates(
                markOp.getOperation(), // reached_mask_ops_idx
                maskOpMark->second)) { // mask_op_idx
          rewriter.eraseOp(markOp);
        } else {
          // If the annotated value is also used outside the masked data flow,
          // do not mark it. Marking it would cause the LLVM lowering to
          // materialize it under the restricted predicate, and then silently
          // reuse that restricted-predicate result in the unmasked context,
          // leaving the upper lanes of the vector invalid.
          Value annotatedVal = markOp.getSrc();
          bool hasUnmaskedUser = false;
          for (Operation *user : annotatedVal.getUsers()) {
            // Skip annotation mark ops themselves.
            if (isa<annotation::MarkOp>(user))
              continue;
            // Skip ops with no results (e.g., the transfer_write anchor).
            if (user->getNumResults() == 0)
              continue;
            // If this user's result has no reached_mask_ops_idx annotation it
            // lies outside the masked chain — the value has mixed uses.
            if (!utils::getAnnotateOpWithAttr(user->getResult(0),
                                              utils::reachedMaskOpsIdx)) {
              hasUnmaskedUser = true;
              break;
            }
          }
          if (hasUnmaskedUser)
            rewriter.eraseOp(markOp);
          else
            markOp->setAttr(utils::reachedMaskOpsIdx, markedIdxArray[0]);
        }

      } else {
        rewriter.eraseOp(markOp);
      }
    }
  });
}

std::unique_ptr<Pass> hivm::createArithVectorMaskAnalysisPass() {
  return std::make_unique<ArithVectorMaskAnalysisPass>();
}
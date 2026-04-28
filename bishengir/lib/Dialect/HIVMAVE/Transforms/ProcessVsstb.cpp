//===------------- ProcessVsstb.cpp - process vsstb op --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ave-process-vsstb"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_PROCESSVSSTB
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivmave;

// From one op find source back to the load op, each intermediate usage point
// cannot be in multiple branches.
static void findLoadOp(Operation *op, SmallVector<Operation *> &opList,
                       SmallVector<VFLoadOp> &loadList) {
  if (!op)
    return;
  if (VFLoadOp loadOp = dyn_cast<VFLoadOp>(op)) {
    if (loadOp.getPattern() == hivmave::LoadDist::NORM && loadOp->hasOneUse()) {
      loadList.push_back(loadOp);
    }
  } else {
    if (!isa<hivmave::AVEElementwiseOp>(op)) {
      return;
    }
    opList.push_back(op);
    for (auto oprand : op->getOperands()) {
      Operation *srcOp = oprand.getDefiningOp();
      // If exist non-elementwise op, must keep origin vector sequence,
      // can not merge load with dintlv.
      bool keepSequence = false;
      for (auto u : oprand.getUsers()) {
        if (!isa<hivmave::AVEElementwiseOp>(u)) {
          keepSequence = true;
        }
      }
      if (!keepSequence)
        findLoadOp(srcOp, opList, loadList);
    }
  }
}

// Check the two ub address of load ops can be combined with DINTLV_B32.
static bool checkTwoLoadCanCombine(SmallVector<Operation *> &opList1,
                                   SmallVector<VFLoadOp> &loadList1,
                                   SmallVector<Operation *> &opList2,
                                   SmallVector<VFLoadOp> &loadList2,
                                   VFLoadOp &op1, VFLoadOp &op2) {
  // the operations from laod to truncf must equal
  if (opList1.size() != opList2.size())
    return false;
  for (size_t i = 0; i < opList1.size(); i++) {
    if (opList1[i]->getName() != opList2[i]->getName())
      return false;
  }
  // only merge one loadop
  if (loadList1.size() != 1)
    return false;
  op1 = loadList1[0];
  op2 = loadList2[0];
  if (!op1 || !op2)
    return false;
  if (op1.getIndices() != op2.getIndices())
    return false;
  if (op1.getVectorType() != op2.getVectorType())
    return false;
  if (op1.getVectorType().getElementTypeBitWidth() != 32)
    return false;
  // todo: check the two ub address of load ops are continue.
  return true;
}

// find vsstb dst value upon function parameter
static Value findVsstbDst(Value memrefVal) {
  Operation *defOp = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
    defOp = blockArg.getOwner()->getParentOp();
  } else {
    defOp = memrefVal.getDefiningOp();
  }
  if (dyn_cast<func::FuncOp>(defOp)) {
    // Check whether the memref object is a function parameter.
    return memrefVal;
  } else if (auto subViewOp = dyn_cast<memref::SubViewOp>(defOp)) {
    return findVsstbDst(subViewOp.getSource());
  } else if (auto viewOp = dyn_cast<memref::ViewOp>(defOp)) {
    return findVsstbDst(viewOp.getSource());
  } else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(defOp)) {
    return findVsstbDst(collapseOp.getSrc());
  }
  return nullptr;
}

// Check the two vsstb ops save to the same ub area.
static bool
checkTwoVsstbCanCombine(SmallVector<VFStoreWithStrideOp, 2> &oldStoreOps) {
  if (oldStoreOps.size() == 2) {
    Value vsstbDst1 = findVsstbDst(oldStoreOps[0].getBase());
    Value vsstbDst2 = findVsstbDst(oldStoreOps[1].getBase());
    return vsstbDst1 == vsstbDst2;
  }
  return false;
}

static Value getNewMask(Value oldMask, VectorType newMaskType, Location loc,
                        PatternRewriter &rewriter, bool needDouble = false) {
  Operation *defOp = oldMask.getDefiningOp();
  while (defOp) {
    if (auto plt = dyn_cast<VFPltOp>(defOp)) {
      Value trueShape = plt.getTrueShape();
      if (needDouble) {
        Value trueShapeI = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI32Type(), trueShape);
        Value trueShapeDouble =
            rewriter.create<arith::AddIOp>(loc, trueShapeI, trueShapeI);
        trueShape = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), trueShapeDouble);
      }
      Value newMask =
          rewriter
              .create<hivmave::VFPltOp>(loc, newMaskType,
                                        rewriter.getIndexType(), trueShape)
              .getResults()[0];
      return newMask;
    } else if (auto pge = dyn_cast<VFPgeOp>(defOp)) {
      uint32_t trueShape = getNumfromPgePattern(pge);
      if (needDouble)
        trueShape += trueShape;
      PgePatternAttr pgePatternAttr =
          hivmave::getPgePatternAttr(rewriter, trueShape,
                                     newMaskType.getNumElements())
              .value();
      Value newMask =
          rewriter.create<hivmave::VFPgeOp>(loc, newMaskType, pgePatternAttr);
      return newMask;
    } else if (auto cast = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
      defOp = cast->getOperand(0).getDefiningOp();
    } else if (auto extract = dyn_cast<LLVM::ExtractValueOp>(defOp)) {
      defOp = extract->getOperand(0).getDefiningOp();
    } else {
      llvm::errs() << "not process yet " << *defOp << "\n";
      break;
    }
  };
  return rewriter.create<hivmave::VFPgeOp>(loc, newMaskType, PgePattern::ALL);
}

struct Unroll64F32ForLoopPattern : public OpRewritePattern<scf::ForOp> {
  Unroll64F32ForLoopPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context, /*benefit=*/10) {}

  static VectorType createFullB16VectorType(MLIRContext *ctx, Type elemType) {
    int64_t vecSizeB16 = util::VL / 2;
    if (elemType.isF16()) {
      return VectorType::get({vecSizeB16}, Float16Type::get(ctx));
    } else if (elemType.isBF16()) {
      return VectorType::get({vecSizeB16}, BFloat16Type::get(ctx));
    }
    return nullptr;
  }

  static void getStoreWithStrideAfterTruncOps(
      SmallVector<VFTruncFOp, 2> &vtruncfOps,
      SmallVector<VFStoreWithStrideOp, 2> &oldStoreOps, Block *forBody) {
    vtruncfOps.clear();
    oldStoreOps.clear();
    for (Operation &op : forBody->getOperations()) {
      if (auto storeOp = dyn_cast<VFStoreWithStrideOp>(&op)) {
        Operation *srcOp = storeOp->getOperands().back().getDefiningOp();
        if (isa<VFTruncFOp>(srcOp)) {
          vtruncfOps.push_back(cast<VFTruncFOp>(srcOp));
          oldStoreOps.push_back(storeOp);
        }
      }
    }
  }

  LogicalResult rewriteTruncAndStore(
      PatternRewriter &rewriter, SmallVector<VFTruncFOp, 2> &vtruncfOps,
      SmallVector<VFStoreWithStrideOp, 2> &oldStoreOps, bool useDIntlv) const {
    Location loc = vtruncfOps[0].getLoc();
    VectorType vecType = cast<VectorType>(vtruncfOps[0].getType());
    Type elemType = vecType.getElementType();
    VectorType newVecType =
        createFullB16VectorType(rewriter.getContext(), elemType);
    if (!newVecType)
      return failure();
    int64_t vecSizeB16 = util::VL / 2;
    VectorType newMaskType =
        VectorType::get({vecSizeB16}, rewriter.getI1Type());
    if (oldStoreOps.size() == 1 || !checkTwoVsstbCanCombine(oldStoreOps)) {
      for (size_t i = 0; i < oldStoreOps.size(); i++) {
        // If for body has only one vsstb.
        // The src vector must dintlv with 0 vector before vsstb.
        rewriter.setInsertionPointAfter(vtruncfOps[i]);
        auto deInterleaveOp = rewriter.create<VFDeInterleaveOp>(
            loc, vecType, vecType, vtruncfOps[i].getRes(),
            vtruncfOps[i].getRes());
        Value res1 = deInterleaveOp.getRes1();
        // create new StoreWithStrideOp with new mask and vector
        rewriter.setInsertionPointAfter(oldStoreOps[i]);
        Value newMask =
            getNewMask(oldStoreOps[i].getMask(), newMaskType, loc, rewriter);
        rewriter.create<VFStoreWithStrideOp>(
            loc, oldStoreOps[i].getBase(), oldStoreOps[i].getIndices(),
            oldStoreOps[i].getStride(), newMask, res1);
        // remove old store
        rewriter.eraseOp(oldStoreOps[i]);
      }
      return success();
    } else if (oldStoreOps.size() == 2) {
      // If for body has two vsstb.
      // If the src two load ops can be conbined with DINTLV_B32,
      // then change truncOp with even/odd mode and use OrOp to combine the two
      // vector before vsstb,
      // or use DintlvOp to combine the two vector before vsstb.
      bool useDIntlv = true;
      SmallVector<VFLoadOp> loadList1, loadList2;
      SmallVector<Operation *> opList1, opList2;
      VFLoadOp oldLoad1, oldLoad2;
      findLoadOp(vtruncfOps[0], opList1, loadList1);
      findLoadOp(vtruncfOps[1], opList2, loadList2);
      // If two load-op merged, the two trunc-op will change by even/odd mod.
      // To avoid affecting other users, this code requires that trunc-op has
      // only one user.
      if (checkTwoLoadCanCombine(opList1, loadList1, opList2, loadList2,
                                 oldLoad1, oldLoad2) &&
          vtruncfOps[0]->hasOneUse() && vtruncfOps[1]->hasOneUse()) {
        // change two norm load to one intlv load
        useDIntlv = false;
        rewriter.setInsertionPointAfter(oldLoad1);
        hivmave::VFLoadOp newLoadOp = rewriter.create<hivmave::VFLoadOp>(
            loc, oldLoad1.getVectorType(), oldLoad1.getVectorType(),
            hivmave::LoadDist::DINTLV_B32, oldLoad1.getBase(),
            oldLoad1.getIndices());
        rewriter.replaceAllOpUsesWith(oldLoad1, newLoadOp.getResult(0));
        rewriter.replaceAllOpUsesWith(oldLoad2, newLoadOp.getResult(1));
      }

      rewriter.setInsertionPointAfter(vtruncfOps[1]);
      Value newMask = getNewMask(oldStoreOps[0].getMask(), newMaskType, loc,
                                 rewriter, true);
      // change TruncF to output with 128*f16 and set with even/odd part
      Value storeVal;
      if (useDIntlv) {
        auto deInterleaveOp = rewriter.create<VFDeInterleaveOp>(
            loc, newVecType, newVecType, vtruncfOps[0].getRes(),
            vtruncfOps[1].getRes());
        storeVal = deInterleaveOp.getRes1();
        rewriter.setInsertionPointAfter(deInterleaveOp);
      } else {
        VFTruncFOp newTruncOp1 = rewriter.create<VFTruncFOp>(
            loc, newVecType, vtruncfOps[0].getSrc(), vtruncfOps[0].getMask(),
            vtruncfOps[0].getRnd(), vtruncfOps[0].getSat(),
            VCVT_PartType::PART_EVEN);
        VFTruncFOp newTruncOp2 = rewriter.create<VFTruncFOp>(
            loc, newVecType, vtruncfOps[1].getSrc(), vtruncfOps[1].getMask(),
            vtruncfOps[1].getRnd(), vtruncfOps[1].getSat(),
            VCVT_PartType::PART_ODD);
        auto orOp =
            rewriter.create<VFOrOp>(loc, newVecType, newTruncOp1.getRes(),
                                    newTruncOp2.getRes(), newMask, nullptr);
        storeVal = orOp.getResult();
        rewriter.setInsertionPointAfter(orOp);
      }
      // use one 128*f16 vsstb instead of two 64*f16 vsstb
      Value base = oldStoreOps[0].getBase();
      ValueRange indices = oldStoreOps[0].getIndices();
      Value stride = oldStoreOps[0].getStride();
      rewriter.create<VFStoreWithStrideOp>(loc, base, indices, stride, newMask,
                                           storeVal);
      // remove old store
      rewriter.eraseOp(oldStoreOps[0]);
      rewriter.eraseOp(oldStoreOps[1]);
      return success();
    }
    return failure();
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<VFTruncFOp, 2> vtruncfOps;
    SmallVector<VFStoreWithStrideOp, 2> oldStoreOps;
    getStoreWithStrideAfterTruncOps(vtruncfOps, oldStoreOps, forOp.getBody());
    if (oldStoreOps.size() == 1 && forOp->getAttr("unroll_for_vsstb")) {
      forOp->removeAttr("unroll_for_vsstb");
      auto outerFor = forOp->getParentOfType<scf::ForOp>();
      // Unrolll
      auto unrollResult =
          loopUnrollByFactor(forOp, /*unrollFactor=*/2, /*annotateFn=*/nullptr);
      if (failed(unrollResult)) {
        return failure();
      }
      Block *unrolledBody = nullptr;
      if (!unrollResult->epilogueLoopOp.has_value() &&
          !unrollResult->mainLoopOp.has_value()) {
        unrolledBody = outerFor.getBody();
      } else if (unrollResult->mainLoopOp.has_value()) {
        unrolledBody = unrollResult->mainLoopOp->getBody();
      } else if (unrollResult->epilogueLoopOp.has_value()) {
        unrolledBody = unrollResult->epilogueLoopOp->getBody();
      }
      if (!unrolledBody) {
        return failure();
      }
      getStoreWithStrideAfterTruncOps(vtruncfOps, oldStoreOps, unrolledBody);
    } else if (forOp->getAttr("unroll_for_vsstb")) {
      forOp->removeAttr("unroll_for_vsstb");
      return failure();
    } else if (oldStoreOps.size() != 1 && oldStoreOps.size() != 2) {
      return failure();
    }

    return rewriteTruncAndStore(rewriter, vtruncfOps, oldStoreOps, true);
  }
};

// Change mask with wide bitwise to narrow bitwise adapt to datatype
// If the store vector type wide is B32, use vector<64*i1> as mask type.
// If the store vector type wide B16, use vector<128*i1> as mask type.
// If the store vector type wide B8, use vector<256*i1> as mask type.
// change from
// %4 = ave.hir.pge <ALL> : vector<64xi1>
// ave.hir.store_with_stride %subview_5[%c0, %c0], %c1040, %4, %res_4
// to
// %4 = ave.hir.pge <VL64> : vector<128xi1>
// ave.hir.store_with_stride %subview_5[%c0, %c0], %c1040, %4, %res_4
struct NarrowVsstbMaskPattern
    : public OpRewritePattern<hivmave::VFStoreWithStrideOp> {
  NarrowVsstbMaskPattern(MLIRContext *context)
      : OpRewritePattern<hivmave::VFStoreWithStrideOp>(context,
                                                       /*benefit=*/10) {}

  LogicalResult matchAndRewrite(hivmave::VFStoreWithStrideOp storeOp,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(storeOp);
    Location loc = storeOp.getLoc();
    auto dElemType = storeOp.getVectorType().getElementType();
    auto vectorRegLength = util::VL_BITS / dElemType.getIntOrFloatBitWidth();
    auto oldMask = storeOp.getMask();
    auto oldMaskLength = cast<VectorType>(oldMask.getType()).getNumElements();
    if (oldMaskLength == vectorRegLength)
      return failure();
    VectorType newMaskType =
        VectorType::get({vectorRegLength}, rewriter.getI1Type());
    Value newMask = getNewMask(oldMask, newMaskType, loc, rewriter);
    Value base = storeOp.getBase();
    ValueRange indices = storeOp.getIndices();
    Value stride = storeOp.getStride();
    Value storeVal = storeOp.getVal();
    rewriter.create<VFStoreWithStrideOp>(loc, base, indices, stride, newMask,
                                         storeVal);
    rewriter.eraseOp(storeOp);
    return success();
  }
};

namespace {
struct processVsstbPass : public impl::ProcessVsstbBase<processVsstbPass> {
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<Unroll64F32ForLoopPattern>(context);
    patterns.add<NarrowVsstbMaskPattern>(context);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> hivmave::createProcessVsstbPass() {
  return std::make_unique<processVsstbPass>();
}

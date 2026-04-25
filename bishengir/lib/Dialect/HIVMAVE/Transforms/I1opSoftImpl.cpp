//===------------- I1opSoftImpl.cpp - soft impl i1 op ---------------------===//
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
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "ave-i1op-soft-impl"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_I1OPSOFTIMPL
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivmave;

std::pair<Value, Value> getBaseMemrefAndOffset(PatternRewriter &rewriter,
                                               Location &loc, Value srcMem,
                                               Value currOffset) {
  // TODO: Supports offset calculation in complex scenarios. #ISSUE#84
  Operation *defOp = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(srcMem)) {
    defOp = blockArg.getOwner()->getParentOp();
  } else {
    defOp = srcMem.getDefiningOp();
  }
  if (dyn_cast<func::FuncOp>(defOp)) {
    // Check whether the memref object is a function parameter.
    // The memref object address in the parameters is aligned.
    return {srcMem, currOffset};
  } else if (auto subViewOp = dyn_cast<memref::SubViewOp>(defOp)) {
    auto srcValue = subViewOp.getSource();
    auto offsetOperands = subViewOp.getMixedOffsets();

    if (srcValue.getType().getRank() != 1)
      llvm::report_fatal_error(
          "process i1 load error: can not process vector with multi dim");

    Value newOffset;
    if (auto v = dyn_cast<Value>(offsetOperands[0])) {
      if (auto blockArg = dyn_cast<BlockArgument>(v)) {
        if (scf::ForOp forOp =
                dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
          Value upperBound = forOp.getUpperBound();
          std::optional<int64_t> constVal = getConstantIntValue(upperBound);
          if (constVal.has_value() && constVal > util::VL / 2)
            llvm::report_fatal_error(
                "The offset of the vector is larger than the length of preg");
        }
      }
      newOffset = rewriter.create<arith::AddIOp>(loc, v, currOffset);
    } else if (Attribute attr = dyn_cast<Attribute>(offsetOperands[0])) {
      int64_t offset = dyn_cast<IntegerAttr>(attr).getSInt();
      if (offset > util::VL / 2)
        llvm::report_fatal_error(
            "The offset of the vector is larger than the length of preg");
      Value cstVal = rewriter.create<arith::ConstantIndexOp>(loc, offset);
      newOffset = rewriter.create<arith::AddIOp>(loc, cstVal, currOffset);
    }

    return getBaseMemrefAndOffset(rewriter, loc, srcValue, newOffset);
  }
  return {srcMem, currOffset};
}

// process load + brc i1
struct loadBroadcastPattern : public OpRewritePattern<hivmave::VFLoadOp> {
  loadBroadcastPattern(MLIRContext *context)
      : OpRewritePattern<hivmave::VFLoadOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(mlir::hivmave::VFLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    VectorType orgVectorTy = loadOp.getVectorType();
    int64_t vecSize = orgVectorTy.getNumElements();
    Type vecElemTy = orgVectorTy.getElementType();
    MemRefType memRefTy = loadOp.getMemRefType();
    if (!vecElemTy.isInteger(1) || !memRefTy.hasStaticShape() ||
        memRefTy.getNumElements() != 1 || memRefTy.getRank() != 1)
      return failure();
    LDBG("Process operation : " << loadOp);

    Location loc = loadOp.getLoc();
    rewriter.setInsertionPointAfter(loadOp);
    Value constZeroI8 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(rewriter.getI8Type()));
    Value constZeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value constOne = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getOneAttr(rewriter.getI8Type()));
    VectorType i8VecTy = VectorType::get({util::VL}, rewriter.getI8Type());
    VectorType i8MaskTy = VectorType::get({util::VL}, rewriter.getI1Type());
    // find base address of i1 vector and load from base address
    Value srcMemref = loadOp.getBase();
    Value srcOffset = loadOp.getIndices()[0];
    auto [baseMemref, baseOffset] =
        getBaseMemrefAndOffset(rewriter, loc, srcMemref, srcOffset);
    VFLoadOp newLoad = rewriter.create<hivmave::VFLoadOp>(
        loc, orgVectorTy, baseMemref, constZeroIndex);
    Value newLoadVal = newLoad.getResult(0);
    if (vecSize != util::VL)
      newLoadVal =
          rewriter.create<UnrealizedConversionCastOp>(loc, i8MaskTy, newLoadVal)
              .getResult(0);
    // convert i1 to i8
    VFBroadcastScalarOp brcZero = rewriter.create<hivmave::VFBroadcastScalarOp>(
        loc, i8VecTy, constZeroI8);
    VFBroadcastScalarOp brcOne =
        rewriter.create<hivmave::VFBroadcastScalarOp>(loc, i8VecTy, constOne);
    VFSelectOp selI8 = rewriter.create<hivmave::VFSelectOp>(
        loc, i8VecTy, newLoadVal, brcOne, brcZero);
    // use reduce-xori to move the active element to the lowest position
    Value allI8Mask = rewriter.create<hivmave::VFPgeOp>(
        loc, i8MaskTy,
        PgePatternAttr::get(rewriter.getContext(), PgePattern::ALL));
    auto indexOne = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), constOne);
    auto postOffset = rewriter.create<arith::AddIOp>(loc, baseOffset, indexOne);
    VFPltOp p1 = rewriter.create<hivmave::VFPltOp>(
        loc, i8MaskTy, rewriter.getIndexType(), postOffset);
    VFPltOp p2 = rewriter.create<hivmave::VFPltOp>(
        loc, i8MaskTy, rewriter.getIndexType(), baseOffset);
    PregXorOp pXor = rewriter.create<hivmave::PregXorOp>(
        loc, i8MaskTy, MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8),
        p1.getResults()[0], p2.getResults()[0], allI8Mask);
    ReductionOp shiftI8 = rewriter.create<hivmave::ReductionOp>(
        loc, i8VecTy, hivmave::CombiningKind::XORI, selI8, pXor);
    // brc I8
    // TODO: I1 is broadcast to all positions. Bit width optimization need be
    // considered. #ISSUE#84
    VFBroadcastVectorOp brcI8 = rewriter.create<hivmave::VFBroadcastVectorOp>(
        loc, i8VecTy, shiftI8, allI8Mask, rewriter.getBoolAttr(true));
    // get preg by cmp not-equal with zero vector
    VFBroadcastScalarOp brcZeroI8 =
        rewriter.create<hivmave::VFBroadcastScalarOp>(loc, i8VecTy,
                                                      constZeroI8);
    Value newPreg = rewriter.create<hivmave::VFCmpOp>(
        loc, i8MaskTy, hivmave::CmpType::NE, brcI8, brcZeroI8, allI8Mask);

    if (vecSize != util::VL)
      newPreg =
          rewriter.create<UnrealizedConversionCastOp>(loc, orgVectorTy, newPreg)
              .getResult(0);
    // replace old load
    rewriter.replaceAllUsesWith(loadOp->getResults()[0], newPreg);
    rewriter.eraseOp(loadOp);
    return success();
  }
};

namespace {
struct i1opSoftImplPass : public impl::I1opSoftImplBase<i1opSoftImplPass> {
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<loadBroadcastPattern>(context);
    mlir::GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> hivmave::createI1opSoftImplPass() {
  return std::make_unique<i1opSoftImplPass>();
}

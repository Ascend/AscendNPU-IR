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

static constexpr llvm::StringLiteral i1ProcessedAttr = "1xi1 processed";

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
      newOffset = rewriter.create<arith::AddIOp>(loc, v, currOffset);
    } else if (Attribute attr = dyn_cast<Attribute>(offsetOperands[0])) {
      int64_t offset = dyn_cast<IntegerAttr>(attr).getSInt();
      Value cstVal = rewriter.create<arith::ConstantIndexOp>(loc, offset);
      newOffset = rewriter.create<arith::AddIOp>(loc, cstVal, currOffset);
    }

    return getBaseMemrefAndOffset(rewriter, loc, srcValue, newOffset);
  }
  return {srcMem, currOffset};
}

// Decompose a memory offset into a VL-aligned base and an intra-VL offset.
//
// Given `currOffset`, computes:
//   base = floor(currOffset / VL) * VL   // VL-aligned base address
//   offsetInVL = currOffset - base           // remainder within [0, VL)
//
// This is used when loading i1 vectors: the hardware vector load must start
// at a VL-aligned boundary, and the actual data position within the loaded
// vector is tracked separately via `offsetInVL`.  For example, with VL=256
// and offset=300:
//   base = 256, offsetInVL = 44
//
// Returns {base, offsetInVL} both cast back to index type.
std::pair<Value, Value> getBaseAndOffetInVL(PatternRewriter &rewriter,
                                            Location &loc, Value currOffset) {
  Value constVL = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(util::VL));
  Value constByteSize = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(util::BITS_PER_BYTE));
  Value i32Offset = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI32Type(), currOffset);
  Value numVL = rewriter.create<arith::DivSIOp>(loc, i32Offset, constVL);
  Value newBase = rewriter.create<arith::MulIOp>(loc, numVL, constVL);
  Value newOffset = rewriter.create<arith::SubIOp>(loc, i32Offset, newBase);
  // Load indice will be convert to llvm.gep.
  // The offset of the gep instruction is measured in bytes.
  Value newBaseInByte =
      rewriter.create<arith::DivSIOp>(loc, newBase, constByteSize);
  Value indexBase = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), newBaseInByte);
  Value indexOffset = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), newOffset);
  return {indexBase, indexOffset};
}

/// Convert an i1 vector to a predicate register via i8 expansion:
///   i1 mask → select(i8 0/1) → broadcast → cmp(NE, 0) → constrain(B8)
/// If `offsetInVL` has value, inserts a PregXor + Reduction(XORI) before
/// the broadcast to shift the active element to the lowest position
/// (used for unaligned loads).
static Value convertI1ToPreg(VectorType orgVectorTy, Value i1Val,
                             Value offsetInVL, PatternRewriter &rewriter,
                             Location loc) {
  int64_t vecSize = orgVectorTy.getNumElements();
  VectorType i8VecTy = VectorType::get({util::VL}, rewriter.getI8Type());
  VectorType i8MaskTy = VectorType::get({util::VL}, rewriter.getI1Type());
  if (vecSize != util::VL)
    i1Val = rewriter.create<UnrealizedConversionCastOp>(loc, i8MaskTy, i1Val)
                .getResult(0);
  Value allI8Mask = rewriter.create<hivmave::VFPgeOp>(
      loc, i8MaskTy,
      PgePatternAttr::get(rewriter.getContext(), PgePattern::ALL));
  Value constZeroI8 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(rewriter.getI8Type()));
  Value constOneI8 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getOneAttr(rewriter.getI8Type()));

  // i1 → i8: select 1 for true lanes, 0 for false lanes.
  VFBroadcastScalarOp brcZero =
      rewriter.create<hivmave::VFBroadcastScalarOp>(loc, i8VecTy, constZeroI8);
  VFBroadcastScalarOp brcOne =
      rewriter.create<hivmave::VFBroadcastScalarOp>(loc, i8VecTy, constOneI8);
  Value selI8 = rewriter.create<hivmave::VFSelectOp>(loc, i8VecTy, i1Val,
                                                     brcOne, brcZero);

  // For unaligned loads, shift the active element to position 0 via
  // reduce-xori so that the broadcast propagates the correct value.
  Value shiftResult = selI8;
  if (offsetInVL) {
    Value indexOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    auto postOffsetInVL =
        rewriter.create<arith::AddIOp>(loc, offsetInVL, indexOne);
    VFPltOp p1 = rewriter.create<hivmave::VFPltOp>(
        loc, i8MaskTy, rewriter.getIndexType(), postOffsetInVL);
    VFPltOp p2 = rewriter.create<hivmave::VFPltOp>(
        loc, i8MaskTy, rewriter.getIndexType(), offsetInVL);
    PregXorOp pXor = rewriter.create<hivmave::PregXorOp>(
        loc, i8MaskTy, MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8),
        p1.getResults()[0], p2.getResults()[0], allI8Mask);
    shiftResult = rewriter.create<hivmave::ReductionOp>(
        loc, i8VecTy, hivmave::CombiningKind::XORI, selI8, pXor);
  }

  // Broadcast the (possibly shifted) i8 value, then compare NE with zero
  // to produce the final predicate register.
  VFBroadcastVectorOp brcI8 = rewriter.create<hivmave::VFBroadcastVectorOp>(
      loc, i8VecTy, shiftResult, allI8Mask, rewriter.getBoolAttr(true));
  Value newPreg = rewriter.create<hivmave::VFCmpOp>(
      loc, i8MaskTy, hivmave::CmpType::NE, brcI8, brcZero, allI8Mask);

  // Constrain the layout to B8 for VectorLayout analysis.
  newPreg = hivmave::constrainVectorLayout(newPreg, hivmave::VecMemType::B8,
                                           rewriter);
  if (vecSize != util::VL)
    newPreg =
        rewriter.create<UnrealizedConversionCastOp>(loc, orgVectorTy, newPreg)
            .getResult(0);
  return newPreg;
}

// process load + brc i1
struct loadBroadcastPattern : public OpRewritePattern<hivmave::VFLoadOp> {
  loadBroadcastPattern(MLIRContext *context)
      : OpRewritePattern<hivmave::VFLoadOp>(context, /*benefit=*/10) {}

  void rewriteLoadI1(mlir::hivmave::VFLoadOp loadOp,
                     PatternRewriter &rewriter) const {
    Location loc = loadOp.getLoc();
    rewriter.setInsertionPointAfter(loadOp);
    SmallVector<Operation *> oldUsers(loadOp.getRes().getUsers());
    Value newPreg = convertI1ToPreg(loadOp.getVectorType(), loadOp.getRes(),
                                    Value(), rewriter, loc);
    for (Operation *user : oldUsers)
      user->replaceUsesOfWith(loadOp.getRes(), newPreg);
    loadOp->setAttr(i1ProcessedAttr, rewriter.getUnitAttr());
  }

  void rewriteLoadI1Unaligned(mlir::hivmave::VFLoadOp loadOp,
                              PatternRewriter &rewriter) const {
    Location loc = loadOp.getLoc();
    rewriter.setInsertionPointAfter(loadOp);
    // Find base memref and accumulated offset through the SubViewOp chain.
    Value srcMemref = loadOp.getBase();
    Value srcOffset = loadOp.getIndices()[0];
    auto [baseMemref, baseOffset] =
        getBaseMemrefAndOffset(rewriter, loc, srcMemref, srcOffset);
    // Decompose offset into a VL-aligned base and an intra-VL offset.
    auto [baseIndices, offsetInVL] =
        getBaseAndOffetInVL(rewriter, loc, baseOffset);
    // Create a new aligned load from the computed base address.
    VectorType orgVectorTy = loadOp.getVectorType();
    VFLoadOp newLoad = rewriter.create<hivmave::VFLoadOp>(
        loc, orgVectorTy, baseMemref, baseIndices);
    // Convert i1 → preg, with PregXor+Reduction shift for unalignment.
    Value newPreg = convertI1ToPreg(orgVectorTy, newLoad.getResult(0),
                                    offsetInVL, rewriter, loc);
    rewriter.replaceAllUsesWith(loadOp.getResult(0), newPreg);
    rewriter.eraseOp(loadOp);
    newLoad->setAttr(i1ProcessedAttr, rewriter.getUnitAttr());
  }

  LogicalResult matchAndRewrite(mlir::hivmave::VFLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    VectorType orgVectorTy = loadOp.getVectorType();
    Type vecElemTy = orgVectorTy.getElementType();
    MemRefType memRefTy = loadOp.getMemRefType();
    if (loadOp->hasAttr(i1ProcessedAttr))
      return failure();
    if (!vecElemTy.isInteger(1) || !memRefTy.hasStaticShape() ||
        memRefTy.getNumElements() != 1 || memRefTy.getRank() != 1)
      return failure();
    LDBG("Process operation : " << loadOp);

    if (!loadOp->hasAttr(UnalignedAttr::name)) {
      rewriteLoadI1(loadOp, rewriter);
    } else {
      rewriteLoadI1Unaligned(loadOp, rewriter);
    }
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

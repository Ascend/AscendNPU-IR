//===- ComplexReductionIntermediateLowering.cpp - Lower Vector ops to fit HIVM
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hivm-complex-reduction-intermediate-lowering"

namespace mlir {
#define GEN_PASS_DEF_HIVMCOMPLEXREDUCTIONINTERMEDIATELOWERING
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::hivm;
using namespace mlir::hivmave;
namespace {

static Value createConstantBroadcastOp(VectorType vecTy, Type elemType,
                                       DenseElementsAttr denseAttr,
                                       Location loc,
                                       PatternRewriter &rewriter) {
  Value scalarValue;
  if (llvm::isa<FloatType>(elemType)) {
    auto initVal = denseAttr.getSplatValue<APFloat>();
    scalarValue = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, initVal));
  }
  if (llvm::isa<IntegerType>(elemType)) {
    auto initVal = denseAttr.getSplatValue<APInt>();
    if (elemType.isInteger(1)) {
      bool allTrue = denseAttr.getValues<bool>()[0];
      scalarValue =
          mlir::utils::createPRegFromConstantOp(vecTy, allTrue, rewriter);
    } else
      scalarValue = rewriter.create<arith::ConstantOp>(
          loc, elemType, rewriter.getIntegerAttr(elemType, initVal));
  }
  Operation *brcOp = nullptr;
  // Check if the tile element type is FP8 (f8e4m3fn or f8e5m2)
  if ((elemType.isFloat8E4M3FN() || elemType.isFloat8E5M2()) &&
      vecTy.getRank() <= 1) {
    if (!vecTy || vecTy.getShape().size() != 1) {
      auto zeroAttr = rewriter.getZeroAttr(elemType);
      return rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(vecTy, zeroAttr));
    }
    auto vecSize = vecTy.getShape();
    auto i8VecTy = VectorType::get({vecSize}, rewriter.getI8Type());
    // Check if the constant attribute is zero (splat value)
    if (denseAttr.getSplatValue<APFloat>().isZero()) {
      // Keep the same process logic for scalar value as i8 did above
      scalarValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerType(8),
          rewriter.getIntegerAttr(rewriter.getIntegerType(8), 0));
    }
    // Broadcast the scalar value to i8 vector type and cast back
    brcOp = mlir::utils::getBroadcastOp(scalarValue, i8VecTy, rewriter, loc);
    return rewriter.create<vector::BitCastOp>(loc, TypeRange{vecTy},
                                              brcOp->getResult(0));
  }
  if (vecTy.getRank() > 1) {
    auto trimmedType = mlir::hivm::util::trimNonScalableUnitDims(vecTy);
    trimmedType = VectorType::get({trimmedType.getNumElements()},
                                  trimmedType.getElementType());
    brcOp =
        mlir::utils::getBroadcastOp(scalarValue, trimmedType, rewriter, loc);
    if (vecTy != trimmedType) {
      auto ucc = rewriter.create<UnrealizedConversionCastOp>(
          loc, vecTy, brcOp->getResult(0));
      return ucc.getResult(0);
    } else
      return brcOp->getResult(0);
  } else {
    brcOp = mlir::utils::getBroadcastOp(scalarValue, vecTy, rewriter, loc);
    return brcOp->getResult(0);
  }
}

/// Lower xori reduction op to ave.hir.vintlv and ave.hir.vxor.
/// Interleave reduction vector with constant 0 vector and xor the result two
/// vectors. Repeat above operation until the vector size is 1.
/// for example:
/// clang-format off
/// before:
/// %6 = ave.hir.reduction <xori>, %4, %5 : vector<64xi32>, vector<64xi1>
///   -> vector<64xi32>
/// after:
/// %cst = arith.constant dense<0> : vector<64xi32>
/// %6 = ave.hir.vor %4, %cst, %5 : vector<64xi32>, vector<64xi1> // size 64
/// %7 = ave.hir.pge <ALL> : vector<64xi1>
/// %res1, %res2 = ave.hir.vintlv %6, %cst : vector<64xi32> // size 32
/// %8 = ave.hir.vxor %res1, %res2, %7 : vector<64xi32>, vector<64xi1>
/// %res1_2, %res2_3 = ave.hir.vintlv %8, %cst : vector<64xi32> // size 16
/// %9 = ave.hir.vxor %res1_2, %res2_3, %7 : vector<64xi32>, vector<64xi1>
/// %res1_4, %res2_5 = ave.hir.vintlv %9, %cst : vector<64xi32> // size 8
/// %10 = ave.hir.vxor %res1_4, %res2_5, %7 : vector<64xi32>, vector<64xi1>
/// %res1_6, %res2_7 = ave.hir.vintlv %10, %cst : vector<64xi32> // size 4
/// %11 = ave.hir.vxor %res1_6, %res2_7, %7 : vector<64xi32>, vector<64xi1>
/// %res1_8, %res2_9 = ave.hir.vintlv %11, %cst : vector<64xi32> // size 2
/// %12 = ave.hir.vxor %res1_8, %res2_9, %7 : vector<64xi32>, vector<64xi1>
/// %res1_10, %res2_11 = ave.hir.vintlv %12, %cst : vector<64xi32> // size 1
/// %13 = ave.hir.vxor %res1_10, %res2_11, %7 : vector<64xi32>, vector<64xi1>
/// clang-format on
struct LowerAveXorReduction : public OpRewritePattern<hivmave::ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hivmave::ReductionOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getKind() != hivmave::CombiningKind::XORI)
      return failure();
    Location loc = op.getLoc();
    Value inputVec = op.getVector();
    Value mask = op.getMask();
    VectorType vecTy = mlir::cast<VectorType>(inputVec.getType());
    auto elemType = vecTy.getElementType();
    auto zeroAttr = rewriter.getZeroAttr(elemType);
    auto denseAttr = DenseElementsAttr::get(vecTy, zeroAttr);
    Value vecZero;
    if (!elemType || !denseAttr || !denseAttr.isSplat() ||
        !(mlir::utils::isValidHIVMTileVectorType(vecTy)) ||
        mlir::utils::isValidTwoDimVectorType(vecTy)) {
      vecZero = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(vecTy, zeroAttr));
    } else {
      vecZero =
          createConstantBroadcastOp(vecTy, elemType, denseAttr, loc, rewriter);
    }
    // zero out the non active elements
    Value normalizeInput = rewriter.create<hivmave::VFOrOp>(
        loc, vecTy, inputVec, vecZero, mask, nullptr);

    Value reduced = normalizeInput;

    unsigned curSize = static_cast<unsigned>(
        mlir::hivm_regbaseintrins::getVectorSizeByElementType(elemType));
    if (curSize == 0) {
      return failure();
    }
    Value newMask = hivmave::createMaskByPGE(vecTy, rewriter, loc);
    // interleave returns two vectors (interleaved pairs), xor them,
    // and keep the xor result for the next interleave
    while (curSize > 1) {
      unsigned halfSize = curSize / 2;
      auto interOp = rewriter.create<hivmave::VFInterleaveOp>(
          loc, vecTy, vecTy, reduced, vecZero);
      Value inter0 = interOp.getResult(0);
      Value inter1 = interOp.getResult(1);

      Value xorRes = rewriter.create<hivmave::VFXorOp>(
          loc, vecTy, inter0, inter1, newMask, nullptr);

      reduced = xorRes;
      curSize = halfSize;
    }
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

struct ComplexReductionIntermediateLoweringPass
    : public impl::HIVMComplexReductionIntermediateLoweringBase<
          ComplexReductionIntermediateLoweringPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void ComplexReductionIntermediateLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.add<LowerAveXorReduction>(ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass>
mlir::hivmave::createComplexReductionIntermediateLoweringPass() {
  return std::make_unique<ComplexReductionIntermediateLoweringPass>();
}

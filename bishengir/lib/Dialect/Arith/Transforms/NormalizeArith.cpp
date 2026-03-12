//===----------------------- NormalizeArith.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Arith/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZEARITH
#include "bishengir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace {

struct NormalizeArithPass
    : public impl::NormalizeArithBase<NormalizeArithPass> {
  using NormalizeArithBase<NormalizeArithPass>::NormalizeArithBase;

public:
  void runOnOperation() override;
};

// arith.uitofp : vector<1x1x128xi1> to vector<1x1x128xf16>

FloatAttr createF16AttrWithBF16ValueBits(MLIRContext *ctx) {
  // BF16 encoding of 1.0 is 0x3F80
  uint16_t bf16Bits = 0x3F80;
  // Construct a 16-bit APInt from the BF16 bits
  llvm::APInt bitPattern(16, bf16Bits);
  // Use f16 semantics to interpret the BF16 bit pattern
  llvm::APFloat fakeF16Value(llvm::APFloat::IEEEhalf(), bitPattern);

  // Create a FloatAttr with f16 type but BF16 bit pattern
  return FloatAttr::get(FloatType::getF16(ctx), fakeF16Value);
}

static std::optional<DenseElementsAttr>
getDenseAttr(VectorType dstVecTy, double splatValue, PatternRewriter &rewriter,
             bool isBitCastedToBF16 = false) {
  Attribute scalarAttr;
  Type elemType = dstVecTy.getElementType();
  if (elemType.isF16()) {
    if (isBitCastedToBF16)
      scalarAttr = createF16AttrWithBF16ValueBits(rewriter.getContext());
    else
      scalarAttr = rewriter.getF16FloatAttr(splatValue);
  } else if (elemType.isF32()) {
    scalarAttr = rewriter.getF32FloatAttr(splatValue);
  } else if (elemType.isInteger()) {
    int64_t intVal = static_cast<int64_t>(splatValue);
    scalarAttr = rewriter.getIntegerAttr(elemType, intVal);
  } else {
    llvm::errs() << "Unsupported element type for splat constant\n";
    return std::nullopt;
  }
  return DenseElementsAttr::get(dstVecTy, scalarAttr);
}

static Operation *getArithSelectOnZerosAndOnes(Value condition,
                                               VectorType dstVecTy,
                                               Location loc,
                                               PatternRewriter &rewriter,
                                               bool isBitCastedToBF16 = false) {

  auto denseZeros = getDenseAttr(dstVecTy, 0.0, rewriter);
  auto denseOnes = getDenseAttr(dstVecTy, 1.0, rewriter, isBitCastedToBF16);
  if (!denseZeros.has_value() || !denseOnes.has_value())
    return nullptr;
  auto zerosVal =
      rewriter.create<arith::ConstantOp>(loc, dstVecTy, *denseZeros);
  auto onesVal = rewriter.create<arith::ConstantOp>(loc, dstVecTy, *denseOnes);
  auto selectOp = rewriter.create<arith::SelectOp>(loc, dstVecTy, condition,
                                                   onesVal, zerosVal);
  return selectOp;
}



template <typename BinaryOp>
struct NormalizeArithMathScalarBinary : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    if (lhs.getType().isIntOrFloat() && rhs.getType().isIntOrFloat()) {
      Type elementTypeLhs = lhs.getType();
      int64_t vecSizeLhs =
        mlir::utils::getVectorSizeByElementType(elementTypeLhs);
      VectorType brcVecLhsTy = VectorType::get({vecSizeLhs}, lhs.getType());
      Type elementTypeRhs = rhs.getType();
      int64_t vecSizeRhs =
        mlir::utils::getVectorSizeByElementType(elementTypeRhs);
      VectorType brcVecRhsTy = VectorType::get({vecSizeRhs}, rhs.getType());

      Value lhsBrc = rewriter.create<vector::BroadcastOp>(loc, brcVecLhsTy, lhs);
      Value rhsBrc = rewriter.create<vector::BroadcastOp>(loc, brcVecRhsTy, rhs);
      auto brcBinary = rewriter.create<BinaryOp>(loc, lhsBrc, rhsBrc);
      auto extractOp = rewriter.create<vector::ExtractOp>(loc, brcBinary.getResult(), 0);
      rewriter.replaceOp(op, extractOp);
      return success();
    }
    return failure();
  }
};

template <typename UnaryOp>
struct NormalizeArithMathScalarUnary : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto hs = op.getOperand();    
    if (hs.getType().isIntOrFloat()) {
      Type elementTypehs = hs.getType();
      int64_t vecSizehs =
        mlir::utils::getVectorSizeByElementType(elementTypehs);
      VectorType brcVecTy = VectorType::get({vecSizehs}, hs.getType());
      Value hsBrc = rewriter.create<vector::BroadcastOp>(loc, brcVecTy, hs);

      auto brcUnary = rewriter.create<UnaryOp>(loc, hsBrc);
      auto extractOp = rewriter.create<vector::ExtractOp>(loc, brcUnary.getResult(), 0);
      rewriter.replaceOp(op, extractOp);
      return success();
    }
    return failure();
    
  }
};

template <typename CastOp>
struct ConvertCastBoolToOtherTypeBySelect : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {

    if (!isa<CastOpInterface>(op.getOperation()))
      return failure();
    auto src = op.getIn();
    VectorType srcVecTy = dyn_cast<VectorType>(src.getType());
    auto dst = op.getOut();
    VectorType dstVecTy = dyn_cast<VectorType>(dst.getType());
    // only normalize when input is a vector of i1
    if (!srcVecTy || !dstVecTy ||
        srcVecTy.getElementType().getIntOrFloatBitWidth() != 1)
      return failure();
    Location loc = op.getLoc();
    // v300 does not support vsel on bf16
    // FIXME: If src is from the result of compare op, v300 does not support
    // vcmp on bf16, the bf16 operands of compare op will be cast to f32 in
    // LegalizeBF16Pass, so after lowering to HIVMIntrinOp, vcmp of f32 will
    // get 64xi1 and cannot be used to instruct the selection of f16 or bf16,
    // so here we select f32 and cast f32 to bf16. But if src is not from the
    // result of compare op, select f32 may cause wrong result.
    if (dstVecTy.getElementType().isBF16()) {
      // using truncf to cast bf16 from f32
      VectorType f32VecTy =
          VectorType::get(dstVecTy.getShape(), rewriter.getF32Type());
      Operation *f32Select =
          getArithSelectOnZerosAndOnes(src, f32VecTy, loc, rewriter, true);
      auto castToBf16Op = rewriter.create<arith::TruncFOp>(
          loc, dstVecTy, cast<arith::SelectOp>(f32Select).getResult());
      rewriter.replaceOp(op, castToBf16Op);
      return success();
    }
    // When condition shape is [1, 64] and dstVecTy dtype is i8,
    // need to transfer select to: select + cast
    auto srcNumel = utils::getStaticTotalSize(srcVecTy.getShape());
    if (srcNumel == 64 && dstVecTy.getElementType().isInteger(8)) {
      VectorType I32VecTy =
          VectorType::get(dstVecTy.getShape(), rewriter.getI32Type());
      Operation *I32Select =
          getArithSelectOnZerosAndOnes(src, I32VecTy, loc, rewriter, true);
      auto castToI8Op = rewriter.create<arith::TruncIOp>(
          loc, dstVecTy, cast<arith::SelectOp>(I32Select).getResult());
      rewriter.replaceOp(op, castToI8Op);
      return success();
    }

    auto selectOp = getArithSelectOnZerosAndOnes(src, dstVecTy, loc, rewriter);
    rewriter.replaceOp(op, selectOp);
    return success();
  }
};

// This pattern is driven by the following test cases:
//
// 1) vector<i32> -> vector<index> index_cast feeding ONLY vector.gather
//    should be eliminated:
//
//    %idx_i = arith.index_cast %idx : vector<64xi32> to vector<64xindex>
//    %v = vector.gather %tbl[%c0] [%idx_i] ...
//
//    After rewrite, vector.gather directly consumes vector<64xi32> indices
//    and the index_cast must disappear.
template <typename CastOp>
struct ElimVectorIndexCastPattern : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  static bool isVecIntToVecIndex(Type srcTy, Type dstTy) {
    auto srcV = dyn_cast<VectorType>(srcTy);
    auto dstV = dyn_cast<VectorType>(dstTy);
    if (!srcV || !dstV)
      return false;
    if (srcV.getShape() != dstV.getShape())
      return false;
    auto srcElem = dyn_cast<IntegerType>(srcV.getElementType());
    if (!srcElem)
      return false;
    if (!isa<IndexType>(dstV.getElementType()))
      return false;
    return true;
  }

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getIn();
    Value dst = op.getOut();
    if (!isVecIntToVecIndex(src.getType(), dst.getType()))
      return failure();

    const bool allUsersAreGather =
        llvm::all_of(op->getUsers(), [](const auto &user) {
          return isa<vector::GatherOp>(user);
        });

    if (!allUsersAreGather)
      return failure();

    rewriter.replaceOp(op, src);
    return success();
  }
};

} // namespace

void NormalizeArithPass::runOnOperation() {
  auto funcOp = getOperation();
  if (!hivm::isVF(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertCastBoolToOtherTypeBySelect<arith::ExtSIOp>,
               ConvertCastBoolToOtherTypeBySelect<arith::ExtUIOp>,
               ConvertCastBoolToOtherTypeBySelect<arith::SIToFPOp>,
               ConvertCastBoolToOtherTypeBySelect<arith::UIToFPOp>,
               ElimVectorIndexCastPattern<arith::IndexCastOp>,
               ElimVectorIndexCastPattern<arith::IndexCastUIOp>,
               NormalizeArithMathScalarBinary<arith::AddFOp>,
               NormalizeArithMathScalarBinary<arith::DivFOp>,
               NormalizeArithMathScalarUnary<math::SqrtOp>>(&getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> arith::createNormalizeArithPass() {
  return std::make_unique<NormalizeArithPass>();
}

//===--------- LegalizeScalar.cpp - Legalize Scalar Op Pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hfusion-legalize-scalar"

namespace mlir {
#define GEN_PASS_DEF_LEGALIZESCALARPASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

template <typename OpType>
struct LegalizeScalarArithOps : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type elemType = op.getResult().getType();

    if(op->template getParentOfType<linalg::LinalgOp>()) {
      return failure();
    }

    if constexpr (std::is_same_v<arith::AddFOp, OpType>) {
      if (!elemType.isBF16()) {
        return failure();
      }
    } else if constexpr (std::is_same_v<arith::SubFOp, OpType>) {
      if (!elemType.isBF16()) {
        return failure();
      }
    } else if constexpr (std::is_same_v<arith::MulFOp, OpType>) {
      if (!elemType.isBF16()) {
        return failure();
      }
    } else if constexpr (std::is_same_v<arith::DivFOp, OpType>) {
      if (!elemType.isBF16()) {
        return failure();
      }
    } else if constexpr (std::is_same_v<arith::RemFOp, OpType>) {
      if (!(elemType.isF16() || elemType.isBF16() || elemType.isF32())) {
        return failure();
      }
    }

    auto tensorTy = RankedTensorType::get({1}, elemType);
    Value lhsTensor =
        rewriter.create<tensor::FromElementsOp>(loc, tensorTy, lhs);
    Value rhsTensor =
        rewriter.create<tensor::FromElementsOp>(loc, tensorTy, rhs);
 
    Value tensorOp = rewriter.create<OpType>(loc, lhsTensor, rhsTensor);
    auto extractIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {extractIndex};
    auto extractOp = rewriter.create<tensor::ExtractOp>(
        loc, tensorOp, indices);
    
    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

class LegalizeSelectLikeUiToFp : public OpRewritePattern<arith::UIToFPOp> {
  static Value createFilledTensor(PatternRewriter &rewriter, Location loc, RankedTensorType type, double value) {
    auto elementType = cast<FloatType>(type.getElementType());
    
    Value scalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elementType, value));
    assert(type.getNumDynamicDims() == 0);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, type.getShape(), elementType);
    return rewriter.create<linalg::FillOp>(loc, scalar, empty).getResult(0);
  }

public:
  using OpRewritePattern<arith::UIToFPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::UIToFPOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getIn();
    Value output = op.getOut();
    if (!input.getType() || !output.getType()) {
      return failure();
    }

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!inputType || !outputType) {
      return failure();
    }
    if (inputType.getRank() != outputType.getRank()) {
      return failure();
    }
    auto inputRank = inputType.getRank();
    
    if (!inputType.getElementType().isInteger(1)) {
      return failure();
    }
    auto outputElementType = dyn_cast<FloatType>(outputType.getElementType());
    if (!outputElementType) {
      return failure();
    }

    for (int i = 0; i < inputRank; i++) {
      auto inDim = inputType.getDimSize(i);
      auto outDim = outputType.getDimSize(i);
      if (ShapedType::isDynamic(inDim) || ShapedType::isDynamic(outDim)) {
        return failure();
      }
      if (inDim != outDim) {
        return failure();
      }
    }
    auto zeros = createFilledTensor(rewriter, loc, outputType, 0);
    auto ones = createFilledTensor(rewriter, loc, outputType, 1);
    auto outEmpty = utils::createEmptyOp(rewriter, loc, output);
    auto select = rewriter.create<hfusion::SelectOp>(loc, TypeRange{outputType}, ValueRange{input, ones, zeros}, ValueRange{outEmpty});

    rewriter.replaceOp(op, select);
    return success();
  }
};

template <typename OpType>
struct LegalizeScalarMathOps : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    Type elemType = op.getResult().getType();

    if(op->template getParentOfType<linalg::LinalgOp>()) {
      return failure();
    }

    if constexpr (std::is_same_v<math::SqrtOp, OpType>) {
      if (!(elemType.isF16() || elemType.isBF16())) {
        return failure();
      }
    }

    auto tensorTy = RankedTensorType::get({1}, elemType);
    Value inputTensor =
        rewriter.create<tensor::FromElementsOp>(loc, tensorTy, input);
 
    Value tensorOp = rewriter.create<OpType>(loc, inputTensor);
    auto extractIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {extractIndex};
    auto extractOp = rewriter.create<tensor::ExtractOp>(
        loc, tensorOp, indices);

    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

template <typename OpType>
struct LegalizeScalarCastOps : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    Type inType = input.getType();
    Type outType = op.getResult().getType();

    if(op->template getParentOfType<linalg::LinalgOp>()) {
      return failure();
    }

    if constexpr (std::is_same_v<arith::SIToFPOp, OpType>) {
      const bool isI8ToBF16 = inType.isInteger(8) && outType.isBF16();
      const bool isI32ToBF16 = inType.isInteger(32) && outType.isBF16();
      if (!( isI8ToBF16 || isI32ToBF16)) {
        return failure();
      }
    } else if constexpr (std::is_same_v<arith::FPToSIOp, OpType>) {
      const bool isBF16ToI8 = inType.isBF16() && outType.isInteger(8);
      const bool isBF16ToI16 = inType.isBF16() && outType.isInteger(16);
      const bool isBF16ToI32 = inType.isBF16() && outType.isInteger(32);
      if (!(isBF16ToI8 || isBF16ToI16 || isBF16ToI32)) {
        return failure();
      }
    } else if constexpr (std::is_same_v<arith::UIToFPOp, OpType>) {
      const bool isU8ToBF16 = inType.isInteger(8) && outType.isBF16();
      const bool isU32ToBF16 = inType.isInteger(32) && outType.isBF16();
      if (!(isU8ToBF16 || isU32ToBF16)) {
        return failure();
      }
    } else if constexpr (std::is_same_v<arith::FPToUIOp, OpType>) {
      const bool isBF16ToU8 = inType.isBF16() && outType.isInteger(8) ;
      const bool isBF16ToU16 = inType.isBF16() && outType.isInteger(16);
      if (!(isBF16ToU8 || isBF16ToU16)) {
        return failure();
      }
    }

    auto inTensorTy = RankedTensorType::get({1}, inType);
    auto outTensorTy = RankedTensorType::get({1}, outType);
    auto i32TensorTy = RankedTensorType::get({1}, rewriter.getI32Type());
    Value inTensor = rewriter.create<tensor::FromElementsOp>(
        loc, inTensorTy, input);
 
    Value tensorOp;
    if constexpr (std::is_same_v<arith::FPToSIOp, OpType>) {
      const bool isBF16ToI8 = inType.isBF16() && outType.isInteger(8);
      const bool isBF16ToI16 = inType.isBF16() && outType.isInteger(16);
      if (isBF16ToI8 || isBF16ToI16) {
        Value i32TensorOp = rewriter.create<arith::FPToSIOp>(
            loc, i32TensorTy, inTensor);
        tensorOp = rewriter.create<arith::TruncIOp>(
            loc, outTensorTy, i32TensorOp);
      }
    } else if constexpr (std::is_same_v<arith::FPToUIOp, OpType>) {
      const bool isBF16ToU8 = inType.isBF16() && outType.isInteger(8);
      const bool isBF16ToU16 = inType.isBF16() && outType.isInteger(16);
      if (isBF16ToU8 || isBF16ToU16) {
        Value i32TensorOp = rewriter.create<arith::FPToSIOp>(
            loc, i32TensorTy, inTensor);
        tensorOp = rewriter.create<arith::TruncIOp>(
            loc, outTensorTy, i32TensorOp);
      }
    } else {
      tensorOp = rewriter.create<OpType>(loc, outTensorTy, inTensor);
    }
    auto extractIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {extractIndex};
    auto extractOp = rewriter.create<tensor::ExtractOp>(
        loc, tensorOp, indices);

    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

/// For narrow unsigned integer to bf16 conversions the hardware does not
/// natively support the direct cast. Extend the operand to i32 first, then
/// promote i32 to bf16 via tensor<1> (same strategy as
/// LegalizeScalarCastOps<UIToFPOp>).
///
/// Before:
///   %res = arith.uitofp %a : i16 to bf16
/// After:
///   %ext = arith.extui %a : i16 to i32
///   %ta = tensor.from_elements %ext : tensor<1xi32>
///   %tr = arith.uitofp %ta : tensor<1xi32> to tensor<1xbf16>
///   %res = tensor.extract %tr[%c0] : tensor<1xbf16>
struct LegalizeNarrowUIToBF16 : public OpRewritePattern<arith::UIToFPOp> {
  using OpRewritePattern<arith::UIToFPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::UIToFPOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType();
    if (!resultType.isBF16()) {
      return failure();
    }

    auto operandType = op.getOperand().getType();
    auto intOperandType = dyn_cast<IntegerType>(operandType);
    if (!intOperandType || intOperandType.getWidth() >= 32) {
      return failure();
    }

    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();

    Value extI32 =
        rewriter.create<arith::ExtUIOp>(loc, i32Type, op.getOperand());

    auto inTensorTy = RankedTensorType::get({1}, i32Type);
    auto outTensorTy = RankedTensorType::get({1}, resultType);
    Value inTensor =
        rewriter.create<tensor::FromElementsOp>(loc, inTensorTy, extI32);
    Value tensorOp =
        rewriter.create<arith::UIToFPOp>(loc, outTensorTy, inTensor);
    auto extractIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {extractIndex};
    auto extractOp =
        rewriter.create<tensor::ExtractOp>(loc, tensorOp, indices);

    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

/// For narrow signed integer to bf16 conversions the hardware does not
/// natively support the direct cast. Extend the operand to i32 first, then
/// promote i32 to bf16 via tensor<1> (same strategy as
/// LegalizeScalarCastOps<SIToFPOp>).
///
/// Before:
///   %res = arith.sitofp %a : i16 to bf16
/// After:
///   %ext = arith.extsi %a : i16 to i32
///   %ta = tensor.from_elements %ext : tensor<1xi32>
///   %tr = arith.sitofp %ta : tensor<1xi32> to tensor<1xbf16>
///   %res = tensor.extract %tr[%c0] : tensor<1xbf16>
struct LegalizeNarrowSIToBF16 : public OpRewritePattern<arith::SIToFPOp> {
  using OpRewritePattern<arith::SIToFPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SIToFPOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType();
    if (!resultType.isBF16()) {
      return failure();
    }

    auto operandType = op.getOperand().getType();
    auto intOperandType = dyn_cast<IntegerType>(operandType);
    if (!intOperandType || intOperandType.getWidth() >= 32) {
      return failure();
    }

    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();

    Value extI32 =
        rewriter.create<arith::ExtSIOp>(loc, i32Type, op.getOperand());

    auto inTensorTy = RankedTensorType::get({1}, i32Type);
    auto outTensorTy = RankedTensorType::get({1}, resultType);
    Value inTensor =
        rewriter.create<tensor::FromElementsOp>(loc, inTensorTy, extI32);
    Value tensorOp =
        rewriter.create<arith::SIToFPOp>(loc, outTensorTy, inTensor);
    auto extractIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {extractIndex};
    auto extractOp =
        rewriter.create<tensor::ExtractOp>(loc, tensorOp, indices);

    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

/// For i64 to bf16 sitofp conversions the hardware does not natively support
/// the direct cast. Convert i64 to f32 first, then truncate f32 to bf16.
///
/// Before:
///   %res = arith.sitofp %a : i64 to bf16
/// After:
///   %f32 = arith.sitofp %a : i64 to f32
///   %res = arith.truncf %f32 : f32 to bf16
struct LegalizeWideSIToNarrowFP : public OpRewritePattern<arith::SIToFPOp> {
  using OpRewritePattern<arith::SIToFPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SIToFPOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType();
    if (!resultType.isBF16()) {
      return failure();
    }

    auto operandType = op.getOperand().getType();
    auto intOperandType = dyn_cast<IntegerType>(operandType);
    if (!intOperandType || intOperandType.getWidth() <= 32) {
      return failure();
    }

    auto loc = op.getLoc();
    auto f32Type = rewriter.getF32Type();

    Value f32Val =
        rewriter.create<arith::SIToFPOp>(loc, f32Type, op.getOperand());
    Value bf16Val =
        rewriter.create<arith::TruncFOp>(loc, resultType, f32Val);

    rewriter.replaceOp(op, bf16Val);
    return success();
  }
};

/// For u64 to f16/bf16 uitofp conversions the hardware does not natively
/// support the direct cast. Convert u64 to f32 first, then truncate f32 to
/// the target type.
///
/// Before:
///   %res = arith.uitofp %a : u64 to bf16
/// After:
///   %f32 = arith.uitofp %a : u64 to f32
///   %res = arith.truncf %f32 : f32 to bf16
///
/// Before:
///   %res = arith.uitofp %a : u64 to f16
/// After:
///   %f32 = arith.uitofp %a : u64 to f32
///   %res = arith.truncf %f32 : f32 to f16
struct LegalizeWideUIToNarrowFP : public OpRewritePattern<arith::UIToFPOp> {
  using OpRewritePattern<arith::UIToFPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::UIToFPOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType();
    if (!isa<Float16Type, BFloat16Type>(resultType)) {
      return failure();
    }

    auto operandType = op.getOperand().getType();
    auto intOperandType = dyn_cast<IntegerType>(operandType);
    if (!intOperandType || intOperandType.getWidth() <= 32) {
      return failure();
    }

    auto loc = op.getLoc();
    auto f32Type = rewriter.getF32Type();

    Value f32Val =
        rewriter.create<arith::UIToFPOp>(loc, f32Type, op.getOperand());
    Value truncVal =
        rewriter.create<arith::TruncFOp>(loc, resultType, f32Val);

    rewriter.replaceOp(op, truncVal);
    return success();
  }
};

void populateLegalizeScalarConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LegalizeScalarArithOps<arith::AddFOp>>(patterns.getContext());
  patterns.add<LegalizeScalarArithOps<arith::SubFOp>>(patterns.getContext());
  patterns.add<LegalizeScalarArithOps<arith::MulFOp>>(patterns.getContext());
  patterns.add<LegalizeScalarArithOps<arith::DivFOp>>(patterns.getContext());
  patterns.add<LegalizeScalarArithOps<arith::RemFOp>>(patterns.getContext());
  patterns.add<LegalizeScalarMathOps<math::SqrtOp>>(patterns.getContext());
  patterns.add<LegalizeScalarCastOps<arith::SIToFPOp>>(patterns.getContext());
  patterns.add<LegalizeScalarCastOps<arith::FPToSIOp>>(patterns.getContext());
  patterns.add<LegalizeScalarCastOps<arith::UIToFPOp>>(patterns.getContext());
  patterns.add<LegalizeScalarCastOps<arith::FPToUIOp>>(patterns.getContext());
  patterns.add<LegalizeSelectLikeUiToFp>(patterns.getContext());
  patterns.add<LegalizeNarrowUIToBF16>(patterns.getContext());
  patterns.add<LegalizeNarrowSIToBF16>(patterns.getContext());
  patterns.add<LegalizeWideSIToNarrowFP>(patterns.getContext());
  patterns.add<LegalizeWideUIToNarrowFP>(patterns.getContext());
}
 
namespace {
struct LegalizeScalarPass
    : public impl::LegalizeScalarPassBase<LegalizeScalarPass> {
  void runOnOperation() override;
};
} // namespace
 
void LegalizeScalarPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());
  populateLegalizeScalarConversionPatterns(patterns);
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hfusion::createLegalizeScalarPass() {
  return std::make_unique<LegalizeScalarPass>();
}

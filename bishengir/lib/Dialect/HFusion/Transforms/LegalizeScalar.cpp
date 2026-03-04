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

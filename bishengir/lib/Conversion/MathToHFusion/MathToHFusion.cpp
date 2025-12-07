//===- MathToHFusion.cpp - conversion from Math to HFusion dialect --------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/MathToHFusion/MathToHFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

static bool operateOnTensors(Operation *op) {
  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

template <typename UnaryOp, linalg::UnaryFn linalgFn>
struct ElementwiseOpToLinalgUnary : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    Value inner = op.getOperand();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto unaryAttr = rewriter.getAttr<linalg::UnaryFnAttr>(linalgFn);
    auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
    rewriter.replaceOpWithNewOp<linalg::ElemwiseUnaryOp>(
        op, ValueRange{inner}, ValueRange{dsts}, ArrayRef{fnAttr});
    return success();
  }
};

template <typename BinaryOp, linalg::BinaryFn linalgFn>
struct ElementwiseOpToLinalgBinary : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto binaryAttr = rewriter.getAttr<linalg::BinaryFnAttr>(linalgFn);
    auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
    rewriter.replaceOpWithNewOp<linalg::ElemwiseBinaryOp>(
        op, ValueRange{lhs, rhs}, ValueRange{dsts}, ArrayRef{fnAttr});
    return success();
  }
};

template <typename UnaryOp, hfusion::UnaryFn hfusionFn>
struct ElementwiseOpToHFusionUnary : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    Value inner = op.getOperand();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto unaryAttr = rewriter.getAttr<hfusion::UnaryFnAttr>(hfusionFn);
    auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
    rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
        op, ValueRange{inner}, ValueRange{dsts}, ArrayRef{fnAttr});
    return success();
  }
};

template <typename BinaryOp, hfusion::BinaryFn hfusionFn>
struct ElementwiseOpToHFusionBinary : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto binaryAttr = rewriter.getAttr<hfusion::BinaryFnAttr>(hfusionFn);
    auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
    rewriter.replaceOpWithNewOp<hfusion::ElemwiseBinaryOp>(
        op, ValueRange{lhs, rhs}, ValueRange{dsts}, ArrayRef{fnAttr});
    return success();
  }
};

struct MathFmaToComposeBinaryOp : public OpRewritePattern<math::FmaOp> {
  using OpRewritePattern<math::FmaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op)) {
      return failure();
    }
    auto input0 = op.getA();
    auto input1 = op.getB();
    auto input2 = op.getC();
    auto emptyOutsOp =
        mlir::tensor::createTensorEmptyOp(rewriter, op->getLoc(), input0);
    auto *mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::mul,
            ValueRange{input0, input1}, ValueRange(emptyOutsOp));
    auto *addOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::add,
            ValueRange{mulOp->getResult(0), input2}, ValueRange(emptyOutsOp));
    rewriter.replaceOp(op, addOp);
    return success();
  }
};

/// @brief
// Two kinds of conversions are applied:
// 1. math ops to linalg unary/binary ops
// 2. math ops to hfusion unary/binary ops
void mlir::hfusion::populateMathToHFusionConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      ElementwiseOpToLinalgUnary<math::ExpOp, linalg::UnaryFn::exp>,
      ElementwiseOpToLinalgUnary<math::LogOp, linalg::UnaryFn::log>,
      ElementwiseOpToLinalgUnary<math::AbsFOp, linalg::UnaryFn::abs>,
      ElementwiseOpToLinalgUnary<math::CeilOp, linalg::UnaryFn::ceil>,
      ElementwiseOpToLinalgUnary<math::FloorOp, linalg::UnaryFn::floor>,
      ElementwiseOpToHFusionBinary<mathExt::LdexpOp, hfusion::BinaryFn::ldexp>,
      ElementwiseOpToHFusionBinary<math::PowFOp, hfusion::BinaryFn::powf>,
      ElementwiseOpToHFusionUnary<math::SqrtOp, hfusion::UnaryFn::sqrt>,
      ElementwiseOpToHFusionUnary<math::RsqrtOp, hfusion::UnaryFn::rsqrt>,
      ElementwiseOpToHFusionUnary<math::TanhOp, hfusion::UnaryFn::tanh>,
      ElementwiseOpToHFusionUnary<math::AtanOp, hfusion::UnaryFn::atan>,
      ElementwiseOpToHFusionUnary<math::TanOp, hfusion::UnaryFn::tan>,
      ElementwiseOpToHFusionUnary<math::SinOp, hfusion::UnaryFn::sin>,
      ElementwiseOpToHFusionUnary<math::CosOp, hfusion::UnaryFn::cos>,
      ElementwiseOpToHFusionUnary<math::AbsIOp, hfusion::UnaryFn::absi>,
      ElementwiseOpToHFusionUnary<math::ErfOp, hfusion::UnaryFn::erf>,
      ElementwiseOpToHFusionUnary<math::Log2Op, hfusion::UnaryFn::log2>,
      ElementwiseOpToHFusionUnary<math::Log10Op, hfusion::UnaryFn::log10>,
      ElementwiseOpToHFusionUnary<math::Log1pOp, hfusion::UnaryFn::log1p>,
      ElementwiseOpToHFusionUnary<math::Exp2Op, hfusion::UnaryFn::exp2>,
      ElementwiseOpToHFusionUnary<math::ExpM1Op, hfusion::UnaryFn::expm1>,
      ElementwiseOpToHFusionUnary<mathExt::IlogbOp, hfusion::UnaryFn::ilogb>,
      MathFmaToComposeBinaryOp>(patterns.getContext());
}

namespace {
struct MathToHFusionConversionPass
    : public impl::ConvertMathToHFusionBase<MathToHFusionConversionPass> {
  void runOnOperation() override;
};
} // namespace

void MathToHFusionConversionPass::runOnOperation() {
  auto *module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                         hfusion::HFusionDialect>();
  // also add dialects that maybe created by hfusion dialect ops
  target.addLegalDialect<arith::ArithDialect>();
  // math dialect ops are allowed if they don't operate on tensors
  target.addDynamicallyLegalDialect<math::MathDialect>(
      [](Operation *op) { return !operateOnTensors(op); });

  RewritePatternSet patterns(&getContext());
  populateMathToHFusionConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createMathToHFusionConversionPass() {
  return std::make_unique<MathToHFusionConversionPass>();
}

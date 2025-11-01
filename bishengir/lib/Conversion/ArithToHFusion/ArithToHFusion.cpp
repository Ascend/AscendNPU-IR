//===- ArithToHFusion.cpp - conversion from Arith to HFusion dialect ------===//
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

#include "bishengir/Conversion/ArithToHFusion/ArithToHFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

static bool operateOnTensors(Operation *op) {
  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

template <typename UnaryOp, linalg::UnaryFn linalgFn>
struct ElementwiseOpToLinalgUnary : OpRewritePattern<UnaryOp> {
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
struct ElementwiseOpToLinalgBinary : OpRewritePattern<BinaryOp> {
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
struct ElementwiseOpToHFusionUnary : OpRewritePattern<UnaryOp> {
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
struct ElementwiseOpToHFusionBinary : OpRewritePattern<BinaryOp> {
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

template <typename BinaryOp>
struct MulIExtendedOpLowering : OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto resultType = op.getLow().getType();

    auto mulExtOp = rewriter.create<hfusion::MulExtOp>(op->getLoc(), resultType,
                                                       resultType, lhs, rhs);

    rewriter.replaceOp(op, mulExtOp);
    return success();
  }
};

namespace {
struct BitcastOpToHFusionBitcastOp : OpRewritePattern<arith::BitcastOp> {
  using OpRewritePattern<arith::BitcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::BitcastOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    Value input = op.getOperand();
    Value output = op.getResult();
    ShapedType inputType = dyn_cast_if_present<ShapedType>(input.getType());
    Type outputElemType = getElementTypeOrSelf(output.getType());

    auto bitcastOutputOp = mlir::tensor::createTensorEmptyOpWithTargetElemType(
        rewriter, op->getLoc(), input, outputElemType);
    auto bitcastOp = rewriter.create<hfusion::BitcastOp>(
        op->getLoc(), TypeRange{inputType.clone(outputElemType)},
        ValueRange{input}, ValueRange{bitcastOutputOp});

    rewriter.replaceOp(op, bitcastOp);
    return success();
  }
};
} // namespace

inline bool isOverFlowMode(Type inType, Type outType) {
  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);
  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  const bool isF16ToI8 = inType.isF16() && outType.isInteger(8);
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);
  const bool isI32ToI16 = inType.isInteger(32) && outType.isInteger(16);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  return (isI16ToI8 || isI32ToI16 || isI32ToI8 || isF32ToI16 || isF32ToI8 ||
          isF16ToI8);
}

template <typename CastOp>
struct ElementwiseOpToHFusionCast : OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  static hfusion::RoundMode selectRoundMode(CastOp op) {
    auto inType = getElementTypeOrSelf(op.getIn().getType());
    auto outType = getElementTypeOrSelf(op.getOut().getType());
    if (isa<arith::TruncFOp>(op)) {
      if (inType.isF32() && outType.isF16())
        return hfusion::RoundMode::RINT;
      if (inType.isF32() && outType.isBF16())
        return hfusion::RoundMode::RINT;
      if (inType.isF32() && outType.isF32())
        return hfusion::RoundMode::RINT;
      llvm_unreachable("unsupported datatype for arith::TruncFOp to hfusion");
    } else if (isa<arith::ExtFOp>(op)) {
      if (inType.isF16() && outType.isF32())
        return hfusion::RoundMode::RINT;
      if (inType.isBF16() && outType.isF32())
        return hfusion::RoundMode::RINT;
      llvm_unreachable("unsupported datatype for arith::ExtFOp to hfusion");
    } else if (isa<arith::TruncIOp>(op)) {
      if (isOverFlowMode(inType, outType)) {
        return hfusion::RoundMode::TRUNCWITHOVERFLOW;
      }
      return hfusion::RoundMode::RINT;
    } else if (isa<arith::ExtSIOp>(op) || isa<arith::ExtUIOp>(op)) {
      return hfusion::RoundMode::RINT;
    } else if (isa<arith::FPToSIOp>(op) || isa<arith::SIToFPOp>(op) ||
               isa<arith::FPToUIOp>(op) || isa<arith::UIToFPOp>(op)) {
      if (isOverFlowMode(inType, outType)) {
        return hfusion::RoundMode::TRUNCWITHOVERFLOW;
      }
      return hfusion::RoundMode::TRUNC;
    }
    llvm_unreachable("unsupported arith op to hfusion");
  }

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    Value src = op.getOperand();
    hfusion::RoundMode rounding = selectRoundMode(op);
    auto roundingAttr = rewriter.getAttr<hfusion::RoundModeAttr>(rounding);
    auto modeAttr = rewriter.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(),
                                          roundingAttr);
    rewriter.replaceOpWithNewOp<hfusion::CastOp>(
        op, ValueRange{src}, ValueRange{dsts}, ArrayRef{modeAttr});
    return success();
  }
};

template <typename CompareOp>
struct ElementwiseOpToHFusionCompare : OpRewritePattern<CompareOp> {
  using OpRewritePattern<CompareOp>::OpRewritePattern;

  static hfusion::CompareFn selectPredicate(arith::CmpFOp op) {
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      return CompareFn::veq;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      return CompareFn::vne;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      return CompareFn::vle;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      return CompareFn::vlt;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      return CompareFn::vge;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      return CompareFn::vgt;
    default:
      llvm_unreachable("unsupported arith cmp predicate to hfusion");
    }
  }

  static hfusion::CompareFn selectPredicate(arith::CmpIOp op) {
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return CompareFn::veq;
    case arith::CmpIPredicate::ne:
      return CompareFn::vne;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return CompareFn::vlt;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return CompareFn::vgt;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return CompareFn::vle;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return CompareFn::vge;
    default:
      llvm_unreachable("unsupported arith cmp predicate to hfusion");
    }
  }

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    hfusion::CompareFn predicate = selectPredicate(op);
    auto predicateAttr = rewriter.getAttr<hfusion::CompareFnAttr>(predicate);
    auto modeAttr = rewriter.getNamedAttr(hfusion::CompareFnAttr::getMnemonic(),
                                          predicateAttr);
    rewriter.replaceOpWithNewOp<hfusion::CompareOp>(
        op, ValueRange{lhs, rhs}, ValueRange{dsts}, ArrayRef{modeAttr});
    return success();
  }
};

template <typename SelectOp>
struct ElementwiseOpToHFusionSelect : OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    Value condition = op.getCondition();
    Value trueValue = op.getTrueValue();
    Value falseValue = op.getFalseValue();

    rewriter.replaceOpWithNewOp<hfusion::SelectOp>(
        op, ValueRange{condition, trueValue, falseValue}, ValueRange{dsts});
    return success();
  }
};

struct ConstantSplatOpToLinalgFillPattern
    : OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();

    auto shapedType = dyn_cast<ShapedType>(op.getType());
    if (!shapedType)
      return failure();

    auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(op.getValue());
    if (!denseAttr && !denseAttr.isSplat())
      return failure();

    auto elemType = denseAttr.getElementType();
    if (!elemType.isIntOrIndexOrFloat()) {
      return failure();
    }

    auto emptyOp = mlir::tensor::createTensorEmptyOp(rewriter, op->getLoc(),
                                                     op.getResult());

    TypedAttr typedAttr =
        elemType.isIntOrIndex()
            ? (TypedAttr)*denseAttr.getValues<IntegerAttr>().begin()
            : (TypedAttr)*denseAttr.getValues<FloatAttr>().begin();
    auto inputConstantOp =
        rewriter.create<arith::ConstantOp>(op->getLoc(), elemType, typedAttr);

    auto fillOp = rewriter.create<linalg::FillOp>(
        op->getLoc(), ValueRange(inputConstantOp), ValueRange(emptyOp));
    rewriter.replaceOp(op, fillOp);
    return success();
  }
};

void mlir::hfusion::populateArithToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      ElementwiseOpToLinalgBinary<arith::AddFOp, linalg::BinaryFn::add>,
      ElementwiseOpToLinalgBinary<arith::AddIOp, linalg::BinaryFn::add>,
      ElementwiseOpToLinalgBinary<arith::SubFOp, linalg::BinaryFn::sub>,
      ElementwiseOpToLinalgBinary<arith::SubIOp, linalg::BinaryFn::sub>,
      ElementwiseOpToLinalgBinary<arith::MulFOp, linalg::BinaryFn::mul>,
      ElementwiseOpToLinalgBinary<arith::MulIOp, linalg::BinaryFn::mul>,
      ElementwiseOpToLinalgBinary<arith::DivFOp, linalg::BinaryFn::div>,
      ElementwiseOpToLinalgBinary<arith::DivSIOp, linalg::BinaryFn::div>,
      ElementwiseOpToLinalgBinary<arith::DivUIOp,
                                  linalg::BinaryFn::div_unsigned>,
      ElementwiseOpToLinalgBinary<arith::MaxSIOp, linalg::BinaryFn::max_signed>,
      ElementwiseOpToLinalgBinary<arith::MaxUIOp,
                                  linalg::BinaryFn::max_unsigned>,
      ElementwiseOpToLinalgBinary<arith::MinSIOp, linalg::BinaryFn::min_signed>,
      ElementwiseOpToLinalgBinary<arith::MinUIOp,
                                  linalg::BinaryFn::min_unsigned>,
      ElementwiseOpToLinalgUnary<arith::NegFOp, linalg::UnaryFn::negf>,
      ConstantSplatOpToLinalgFillPattern>(patterns.getContext());
}

void mlir::hfusion::populateArithToHFusionConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      ElementwiseOpToHFusionBinary<arith::AndIOp, hfusion::BinaryFn::vand>,
      ElementwiseOpToHFusionBinary<arith::OrIOp, hfusion::BinaryFn::vor>,
      ElementwiseOpToHFusionBinary<arith::XOrIOp, hfusion::BinaryFn::vxor>,
      ElementwiseOpToHFusionBinary<arith::MinNumFOp, hfusion::BinaryFn::minf>,
      ElementwiseOpToHFusionBinary<arith::MinimumFOp, hfusion::BinaryFn::minf>,
      ElementwiseOpToHFusionBinary<arith::MaxNumFOp, hfusion::BinaryFn::maxf>,
      ElementwiseOpToHFusionBinary<arith::MaximumFOp, hfusion::BinaryFn::maxf>,
      ElementwiseOpToHFusionBinary<arith::RemSIOp, hfusion::BinaryFn::mod>,
      ElementwiseOpToHFusionBinary<arith::ShLIOp, hfusion::BinaryFn::shli>,
      ElementwiseOpToHFusionBinary<arith::ShRSIOp, hfusion::BinaryFn::shrsi>,
      ElementwiseOpToHFusionBinary<arith::ShRUIOp, hfusion::BinaryFn::shrui>,
      ElementwiseOpToHFusionBinary<arith::FloorDivSIOp,
                                   hfusion::BinaryFn::floordivsi>,
      ElementwiseOpToHFusionBinary<arith::CeilDivSIOp,
                                   hfusion::BinaryFn::ceildivsi>,
      ElementwiseOpToHFusionBinary<arith::CeilDivUIOp,
                                   hfusion::BinaryFn::ceildivui>,
      ElementwiseOpToHFusionCast<arith::TruncFOp>,
      ElementwiseOpToHFusionCast<arith::ExtFOp>,
      ElementwiseOpToHFusionCast<arith::FPToSIOp>,
      ElementwiseOpToHFusionCast<arith::FPToUIOp>,
      ElementwiseOpToHFusionCast<arith::SIToFPOp>,
      ElementwiseOpToHFusionCast<arith::UIToFPOp>,
      ElementwiseOpToHFusionCast<arith::ExtSIOp>,
      ElementwiseOpToHFusionCast<arith::ExtUIOp>,
      ElementwiseOpToHFusionCast<arith::TruncIOp>,
      ElementwiseOpToHFusionCast<arith::TruncFOp>,
      ElementwiseOpToHFusionCompare<arith::CmpFOp>,
      ElementwiseOpToHFusionCompare<arith::CmpIOp>,
      ElementwiseOpToHFusionSelect<arith::SelectOp>,
      MulIExtendedOpLowering<arith::MulSIExtendedOp>,
      MulIExtendedOpLowering<arith::MulUIExtendedOp>,
      BitcastOpToHFusionBitcastOp>(patterns.getContext());
}

namespace {
struct ArithToHFusionConversionPass
    : public impl::ConvertArithToHFusionBase<ArithToHFusionConversionPass> {
  void runOnOperation() override;
};
} // namespace

void ArithToHFusionConversionPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                         hfusion::HFusionDialect>();
  // Elementwise arith Ops should be converted.
  target.addDynamicallyLegalDialect<arith::ArithDialect>([](Operation *op) {
    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      auto denseAttr =
          dyn_cast<DenseIntOrFPElementsAttr>(constantOp.getValue());
      if (denseAttr && denseAttr.isSplat())
        return false;
      return true;
    }
    return !operateOnTensors(op);
  });

  RewritePatternSet patterns(&getContext());
  populateArithToLinalgConversionPatterns(patterns);
  populateArithToHFusionConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createArithToHFusionConversionPass() {
  return std::make_unique<ArithToHFusionConversionPass>();
}

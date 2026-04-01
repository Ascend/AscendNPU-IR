//===- HFusionToHIVM.cpp - HFusion to HIVM dialect conversion -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert HFusion dialect to HIVM dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVM.h"
#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVMPass.h"
#include "bishengir/Conversion/HFusionToHIVM/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"
#include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <type_traits>

namespace mlir {
#define GEN_PASS_DEF_CONVERTHFUSIONTOHIVM
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-to-hivm-converter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::utils::debugger;
using namespace mlir;
using namespace mlir::hivm;

namespace {

//===----------------------------------------------------------------------===//
// HFusionToHIVMElemwiseOp
//===----------------------------------------------------------------------===//

class ElemwiseOpConvertor {
public:
  ElemwiseOpConvertor(OpBuilder b, Operation *op) : b(b), op(op) {
    assert(hfusion::reshape_utils::isMarkedAsElementwiseOp(op) &&
           "ElemwiseOpConvertor only converts elemwise op");
  }

  template <typename opType>
  Operation *create() {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    Location loc = dpsOp->getLoc();
    auto resultTypes = dpsOp->getResultTypes();
    auto inputs = dpsOp.getDpsInputs();
    auto inits = dpsOp.getDpsInits();

    bool isSigned = true;
    if (isa<linalg::ElemwiseBinaryOp>(op)) {
      linalg::BinaryFn kind = cast<linalg::ElemwiseBinaryOp>(op).getFun();
      if (kind == linalg::BinaryFn::div_unsigned) {
        isSigned = false;
      }
    }
    opType hivmOp;

    if constexpr (std::is_base_of_v<
                      mlir::hivm::detail::ExtraBufferOpInterfaceTrait<opType>,
                      opType>) {
      // don't need temp buffer, but need to pass extra operand to create op
      if constexpr (std::is_same_v<opType, VDivOp>) {
        hivmOp = b.create<opType>(loc, resultTypes, inputs, inits,
                                  /*temp_buffer=*/Value(), isSigned);
      } else {
        hivmOp = b.create<opType>(loc, resultTypes, inputs, inits,
                                  /*temp_buffer=*/Value());
      }
    } else {
      if constexpr (std::is_same_v<opType, VDivOp>) {
        hivmOp = b.create<opType>(loc, resultTypes, inputs, inits, isSigned);
      } else {
        hivmOp = b.create<opType>(loc, resultTypes, inputs, inits);
      }
    }

    return hivmOp;
  }

  OpBuilder &getBuilder() { return b; }
  Operation *getOp() { return op; }

  ~ElemwiseOpConvertor() {}

private:
  OpBuilder b;
  Operation *op;
};

hivm::CompareMode mapCompareModeHFusionToHiVM(hfusion::CompareFn hsCmpMode) {
  switch (hsCmpMode) {
  case hfusion::CompareFn::veq:
    return hivm::CompareMode::EQ;
  case hfusion::CompareFn::vne:
    return hivm::CompareMode::NE;
  case hfusion::CompareFn::vle:
    return hivm::CompareMode::LE;
  case hfusion::CompareFn::vlt:
    return hivm::CompareMode::LT;
  case hfusion::CompareFn::vge:
    return hivm::CompareMode::GE;
  case hfusion::CompareFn::vgt:
    return hivm::CompareMode::GT;
  case hfusion::CompareFn::vule:
    return hivm::CompareMode::LE;
  case hfusion::CompareFn::vuge:
    return hivm::CompareMode::GE;
  case hfusion::CompareFn::vugt:
    return hivm::CompareMode::GT;
  case hfusion::CompareFn::vult:
    return hivm::CompareMode::LT;
  }
}

template <>
Operation *ElemwiseOpConvertor::create<hivm::VCmpOp>() {
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  hfusion::CompareFn hsCmpMode = cast<hfusion::CompareOp>(op).getCompareFn();
  hivm::CompareMode hvCmpMode = mapCompareModeHFusionToHiVM(hsCmpMode);

  return b.create<hivm::VCmpOp>(dpsOp->getLoc(), dpsOp->getResultTypes(),
                                dpsOp.getDpsInputs(), dpsOp.getDpsInits(),
                                hvCmpMode);
}

static hivm::RoundMode mapRoundModeHFusionToHiVM(hfusion::RoundMode hsRndMode) {
  switch (hsRndMode) {
  case (hfusion::RoundMode::RINT):
    return hivm::RoundMode::RINT;
  case (hfusion::RoundMode::ROUND):
    return hivm::RoundMode::ROUND;
  case (hfusion::RoundMode::CEIL):
    return hivm::RoundMode::CEIL;
  case (hfusion::RoundMode::FLOOR):
    return hivm::RoundMode::FLOOR;
  case (hfusion::RoundMode::TRUNC):
    return hivm::RoundMode::TRUNC;
  case (hfusion::RoundMode::ODD):
    return hivm::RoundMode::ODD;
  case (hfusion::RoundMode::TRUNCWITHOVERFLOW):
    return hivm::RoundMode::TRUNCWITHOVERFLOW;
  }
}
hivm::TypeFn mapCastHFusionToHiVM(hfusion::TypeFn casting) {
  switch (casting) {
  case hfusion::TypeFn::cast_signed:
    return hivm::TypeFn::cast_signed;
  case hfusion::TypeFn::cast_unsigned:
    return hivm::TypeFn::cast_unsigned;
  case hfusion::TypeFn::bitcast:
    return hivm::TypeFn::bitcast;
  }
  llvm_unreachable("unsupported hfusion::TypeFn");
}

template <>
Operation *ElemwiseOpConvertor::create<hivm::VCastOp>() {
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  auto castOp = cast<hfusion::CastOp>(op);
  hivm::RoundMode roundMode = mapRoundModeHFusionToHiVM(castOp.getRoundMode());
  hivm::TypeFn casting = mapCastHFusionToHiVM(castOp.getCast());

  return b.create<hivm::VCastOp>(dpsOp->getLoc(), dpsOp->getResultTypes(),
                                 dpsOp.getDpsInputs(), dpsOp.getDpsInits(),
                                 roundMode, casting);
}

Operation *convertNegfToMulOp(ElemwiseOpConvertor &b) {
  auto elemwiseOp = cast<linalg::ElemwiseUnaryOp>(b.getOp());
  auto input = elemwiseOp.getDpsInputs()[0];
  auto elementType = getElementTypeOrSelf(input);
  
  OpBuilder &builder = b.getBuilder();
  Value negOne = builder.create<arith::ConstantOp>(
      elemwiseOp->getLoc(), elementType, builder.getFloatAttr(elementType, -1.0));
  
  auto operation = cast<DestinationStyleOpInterface>(b.getOp());
  Location location = operation->getLoc();
  auto resultTypes = operation->getResultTypes();
  auto inits = operation.getDpsInits();
  
  return builder.create<hivm::VMulOp>(location, resultTypes, ValueRange{input, negOne}, inits);
}

Operation *convertUnaryLinalgOp(ElemwiseOpConvertor &b, linalg::UnaryFn kind) {
  switch (kind) {
  case linalg::UnaryFn::exp:
    return b.create<hivm::VExpOp>();
  case linalg::UnaryFn::log:
    return b.create<hivm::VLnOp>();
  case linalg::UnaryFn::abs:
    return b.create<hivm::VAbsOp>();
  case linalg::UnaryFn::negf:
    return convertNegfToMulOp(b);
  default:
    llvm_unreachable("unsupported linalg unary operation kind");
  }
}

Operation *convertBinaryLinalgOp(ElemwiseOpConvertor &b,
                                 linalg::BinaryFn kind) {
  switch (kind) {
  case linalg::BinaryFn::add:
    return b.create<hivm::VAddOp>();
  case linalg::BinaryFn::mul:
    return b.create<hivm::VMulOp>();
  case linalg::BinaryFn::sub:
    return b.create<hivm::VSubOp>();
  case linalg::BinaryFn::div:
    return b.create<hivm::VDivOp>();
  case linalg::BinaryFn::div_unsigned:
    return b.create<hivm::VDivOp>();
  case linalg::BinaryFn::max_signed:
    return b.create<hivm::VMaxOp>();
  case linalg::BinaryFn::min_signed:
    return b.create<hivm::VMinOp>();
  case linalg::BinaryFn::max_unsigned:
    return b.create<hivm::VMaxOp>();
  case linalg::BinaryFn::min_unsigned:
    return b.create<hivm::VMinOp>();
  default:
    llvm_unreachable("unsupported linalg binary operation kind");
  }
}

Operation *convertUnaryHFusionOp(ElemwiseOpConvertor &b,
                                 hfusion::UnaryFn kind) {
  switch (kind) {
  case hfusion::UnaryFn::relu:
    return b.create<hivm::VReluOp>();
  case hfusion::UnaryFn::sqrt:
    return b.create<hivm::VSqrtOp>();
  case hfusion::UnaryFn::rsqrt:
    return b.create<hivm::VRsqrtOp>();
  case hfusion::UnaryFn::rec:
    return b.create<hivm::VRecOp>();
  case hfusion::UnaryFn::vnot:
    return b.create<hivm::VNotOp>();
  case hfusion::UnaryFn::tanh:
    return b.create<hivm::VTanhOp>();
  case hfusion::UnaryFn::sin:
    return b.create<hivm::VSinOp>();
  case hfusion::UnaryFn::cos:
    return b.create<hivm::VCosOp>();
  case hfusion::UnaryFn::absi:
    return b.create<hivm::VAbsOp>();
  case hfusion::UnaryFn::erf:
    return b.create<hivm::VErfOp>();
  default:
    llvm_unreachable("unsupported hfusion unary operation kind");
  }
}

Operation *convertBinaryHFusionOp(ElemwiseOpConvertor &b,
                                  hfusion::BinaryFn kind) {
  switch (kind) {
  case hfusion::BinaryFn::vor:
    return b.create<hivm::VOrOp>();
  case hfusion::BinaryFn::vand:
    return b.create<hivm::VAndOp>();
  case hfusion::BinaryFn::minf:
    return b.create<hivm::VMinOp>();
  case hfusion::BinaryFn::maxf:
    return b.create<hivm::VMaxOp>();
  case hfusion::BinaryFn::powi:
    return b.create<hivm::VPowOp>();
  case hfusion::BinaryFn::shli:
    return b.create<hivm::VShLOp>();
  case hfusion::BinaryFn::shrsi:
  case hfusion::BinaryFn::shrui:
    return b.create<hivm::VShROp>();
  // This is here just so it pass tests. This operation never gets gneerated
  case hfusion::BinaryFn::divfhp:
    return b.create<hivm::VDivOp>();
  case hfusion::BinaryFn::modui:
    return b.create<hivm::VModUIOp>();
  case hfusion::BinaryFn::mod:
    return b.create<hivm::VModOp>();
  case hfusion::BinaryFn::vxor:
    return b.create<hivm::VXorOp>();
  default:
    llvm_unreachable("unsupported hfusion binary operation kind");
  }
}

Operation *convertCastHFusionOp(ElemwiseOpConvertor &b) {
  return b.create<hivm::VCastOp>();
}

Operation *convertCompareHFusionOp(ElemwiseOpConvertor &b) {
  return b.create<hivm::VCmpOp>();
}

Operation *convertTernaryHFusionOp(ElemwiseOpConvertor &b,
                                   hfusion::TernaryFn kind) {
  return b.create<hivm::VSelOp>();
}

Value brcOperand(OpBuilder &b, Location loc, Value scalarVal,
                 Value brcInitVal) {
  Type brcInitType = brcInitVal.getType();
  const bool isTensorType = isa<TensorType>(brcInitType);

  auto resultTypeRange = isTensorType ? TypeRange(brcInitVal) : TypeRange();
  auto vbrcOp =
      b.create<hivm::VBrcOp>(loc, resultTypeRange, scalarVal, brcInitVal,
                             b.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
  Value newVal = isTensorType ? vbrcOp.getResult()[0] : brcInitVal;

  return newVal;
}

bool isScalarOperand(Value val) { return val.getType().isIntOrFloat(); }

void getInvalidScalarOperands(HIVMStructuredOp *hivmOp,
                              SmallVector<size_t> &scalarOperands) {
  Operation *op = hivmOp->getOperation();
  // TODO: remove the vsub and vdiv special process and support scalar operands
  // for hivm ops
  if (auto *vsubOp = dyn_cast<hivm::VSubOp>(hivmOp)) {
    Type scalarSrc0Type = vsubOp->getSrc()[0].getType();
    Type scalarSrc1Type = vsubOp->getSrc()[0].getType();
    if (scalarSrc0Type.isIntOrFloat() && scalarSrc1Type.isIntOrFloat()) {
      scalarOperands.push_back(0);
      return;
    }
  }
  if (auto *vdivOp = dyn_cast<hivm::VDivOp>(hivmOp)) {
    Type scalarSrc0Type = vdivOp->getSrc()[0].getType();
    Type scalarSrc1Type = vdivOp->getSrc()[0].getType();
    if (scalarSrc0Type.isIntOrFloat() && scalarSrc1Type.isIntOrFloat()) {
      scalarOperands.push_back(0);
      return;
    }
  }
  for (size_t idx = 0; idx < op->getNumOperands() - 1; ++idx) {
    auto oprd = op->getOperand(idx);
    if (isScalarOperand(oprd) && hivmOp->isVectorOnlyOperand(idx)) {
      scalarOperands.push_back(idx);
    }
  }
}

void convertInvalidScalarOperandByBrc(
    Operation *op, SmallVector<size_t> &invalidscalarOperands) {
  OpBuilder b(op);
  Value dstVal = op->getOperand(op->getNumOperands() - 1);
  Type dstType = dstVal.getType();
  auto dstShapedType = mlir::dyn_cast<ShapedType>(dstType);
  for (size_t invalidIdx : invalidscalarOperands) {
    auto operand = op->getOperand(invalidIdx);
    Type srcType = operand.getType();
    auto dstElementType = dstShapedType.getElementType();
    Value brcResult;
    // Distinguishing Scalars and Vectors
    if (auto srcShapedType = mlir::dyn_cast<ShapedType>(srcType)) {
      auto srcElementType = srcShapedType.getElementType();
      srcType = srcElementType;
    }
    if (dstElementType == srcType) {
      brcResult = utils::createEmptyOp(b, op->getLoc(), dstVal);
    } else {
      brcResult = utils::createEmptyOpWithTargetElemType(
        b, op->getLoc(), dstVal, srcType
      );
    }
    Value newOperand = brcOperand(b, op->getLoc(), operand, brcResult);
    op->setOperand(invalidIdx, newOperand);
  }
}

LogicalResult tryConvertInvalidScalarOperandByCommutative(
    HIVMStructuredOp *hivmOp, SmallVector<size_t> &invalidscalarOperands) {
  Operation *op = hivmOp->getOperation();
  if (!op->hasTrait<OpTrait::CommutativeOpTrait>()) {
    return failure();
  }
  // swap input operands if allowed by commutative law
  for (int64_t idx = 0; idx < hivmOp->getNumDpsInputs(); ++idx) {
    if (invalidscalarOperands.empty()) {
      return success();
    }
    Value operand = hivmOp->getDpsInputOperand(idx)->get();
    // cases where swapping operands is not possible
    // 1. current operand is scalar -- should not swap with another scalar
    // 2. current operand is vector, but only vector is allowed at current idx
    if (isScalarOperand(operand) || hivmOp->isVectorOnlyOperand(idx)) {
      continue;
    }

    auto invalidOperandIt = invalidscalarOperands.back();
    invalidscalarOperands.pop_back();

    Value scalarOperands = op->getOperand(invalidOperandIt);
    // swap operand and invalidOperandIt
    op->setOperand(invalidOperandIt, operand);
    op->setOperand(idx, scalarOperands);
  }

  return invalidscalarOperands.empty() ? success() : failure();
}

void convertInvalidScalarOperands(Operation *op) {
  assert(isa<HIVMStructuredOp>(op));
  auto hivmOp = cast<HIVMStructuredOp>(op);
  SmallVector<size_t> scalarOperands;
  getInvalidScalarOperands(&hivmOp, scalarOperands);

  if (scalarOperands.empty()) {
    return;
  }
  if (succeeded(tryConvertInvalidScalarOperandByCommutative(&hivmOp,
                                                            scalarOperands))) {
    return;
  }
  convertInvalidScalarOperandByBrc(op, scalarOperands);
}

LogicalResult elementwiseMatchAndRewriteHelper(Operation *op,
                                               PatternRewriter &rewriter) {
  OpBuilder b(op);
  ElemwiseOpConvertor builder(b, op);
  Operation *hivmOp = nullptr;

  if (isa<linalg::ElemwiseUnaryOp>(op)) {
    linalg::UnaryFn kind = cast<linalg::ElemwiseUnaryOp>(op).getFun();
    hivmOp = convertUnaryLinalgOp(builder, kind);
  } else if (isa<linalg::ElemwiseBinaryOp>(op)) {
    linalg::BinaryFn kind = cast<linalg::ElemwiseBinaryOp>(op).getFun();
    hivmOp = convertBinaryLinalgOp(builder, kind);
  } else if (isa<hfusion::ElemwiseUnaryOp>(op)) {
    hfusion::UnaryFn kind = cast<hfusion::ElemwiseUnaryOp>(op).getFun();
    hivmOp = convertUnaryHFusionOp(builder, kind);
  } else if (isa<hfusion::ElemwiseBinaryOp>(op)) {
    hfusion::BinaryFn kind = cast<hfusion::ElemwiseBinaryOp>(op).getFun();
    hivmOp = convertBinaryHFusionOp(builder, kind);
  } else if (isa<hfusion::CastOp>(op)) {
    hivmOp = convertCastHFusionOp(builder);
  } else if (isa<hfusion::CompareOp>(op)) {
    hivmOp = convertCompareHFusionOp(builder);
  } else if (isa<hfusion::SelectOp>(op)) {
    hfusion::TernaryFn kind = hfusion::TernaryFn::select;
    hivmOp = convertTernaryHFusionOp(builder, kind);
  } else {
    llvm_unreachable("undhandled conversion");
  }
  convertInvalidScalarOperands(hivmOp);
  rewriter.replaceOp(op, hivmOp->getResults());
  return success();
}

template <typename SrcOp>
class HFusionElemwiseOpConverter : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const final {
    return elementwiseMatchAndRewriteHelper(op, rewriter);
  }
};

class ExtractScalarForBinaryShiftOp
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  explicit ExtractScalarForBinaryShiftOp(MLIRContext *context,
                                         PatternBenefit benefit = 100)
      : OpRewritePattern<hfusion::ElemwiseBinaryOp>(context, benefit) {}

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp binOp,
                                PatternRewriter &rewriter) const final {
    hfusion::BinaryFn kind = binOp.getFun();
    DenseSet<hfusion::BinaryFn> supported = {hfusion::BinaryFn::shli,
                                             hfusion::BinaryFn::shrsi,
                                             hfusion::BinaryFn::shrui};
    if (!supported.contains(kind)) {
      return failure();
    }
    Value rhs = binOp.getInputs()[1];
    if (rhs.getType().isIntOrIndexOrFloat()) {
      return failure();
    }
    if (!utils::isScalarLike(rhs)) {
      return failure();
    }
    std::optional<Value> scalarMaybe =
        utils::extractScalarValue(rewriter, binOp->getLoc(), rhs);
    if (!scalarMaybe.has_value()) {
      return failure();
    }
    Value scalar = scalarMaybe.value();
    auto *rhsOperand = binOp.getDpsInputOperand(1);
    rewriter.modifyOpInPlace(binOp, [&]() { rhsOperand->assign(scalar); });
    return success();
  }
};

struct LinalgFillOpToHIVMBrcOp : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() && !op.hasPureTensorSemantics()) {
      return op.emitOpError(
          "linalg::FillOp should have pure buffer or tensor Semantics!");
    }
    auto inputs = op.getInputs();
    assert(inputs.size() == 1);
    auto inits = op.getDpsInits();
    assert(inits.size() == 1);
    auto resultTypeRange =
        op.hasPureBufferSemantics() ? TypeRange() : TypeRange(op->getResults());
    auto hivmVBrcOp = rewriter.create<hivm::VBrcOp>(
        op.getLoc(), resultTypeRange, inputs[0], inits[0],
        rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
    rewriter.replaceOp(op, hivmVBrcOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LinalgBrcOpToHIVMBrcOp
//===----------------------------------------------------------------------===//
/// Convert linalg.broadcast to hivm.hir.vbrc, expand_shape is required.
/// e.g.
///   linalg.broadcast
///     ins(%input:memref<8x32xf32>)
///     outs(%init:memref<8x16x32xf32>)
///     dimensions = [1]
/// converts to
///   %tmp = memref.expand_shape %input [[0, 1], [2]]
///          : memref<8x32xf32> into memref<8x1x32xf32>
///   hivm.hir.vbrc
///     ins(%tmp:memref<8x1x32xf32>)
///     outs(%init:memref<8x16x32xf32>)
///     broadcast_dims = [1]
/// note that input's rank of linalg.broadcast is always less than init's rank,
/// while src'rank of hivm.hir.vbrc is the same as dst's rank.
struct LinalgBrcOpToHIVMBrcOp : public OpRewritePattern<linalg::BroadcastOp> {
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() && !op.hasPureTensorSemantics()) {
      return op.emitOpError(
          "linalg::BroadcastOp should have pure buffer or tensor Semantics!");
    }

    Value expandShapeOp = hfusion_conversion_utils::createExpandShapeOp(
        op, rewriter, op.getInput(), op.getInit().getType());
    auto resultTypeRange =
        op.hasPureBufferSemantics() ? TypeRange() : TypeRange(op.getResult());
    auto brcDimsAttr = op.getDimensionsAttr();
    auto hivmVBrcOp = rewriter.create<hivm::VBrcOp>(
        op.getLoc(), resultTypeRange, expandShapeOp, op.getInit(), brcDimsAttr);

    rewriter.replaceOp(op, hivmVBrcOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LinalgToHIVMCopyOp
//===----------------------------------------------------------------------===//

struct LinalgToHIVMCopyOp : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp op,
                                PatternRewriter &rewriter) const override {
    // convert linalg::CopyOp to hivm::CopyOp
    auto src = cast<::mlir::Value>(*op.getInputs().begin());
    auto dst = cast<::mlir::Value>(*op.getOutputs().begin());
    auto res = op.getResultTensors();
    rewriter.replaceOpWithNewOp<hivm::CopyOp>(op, TypeRange(res), src, dst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LinalgToHIVMTransposeOp
//===----------------------------------------------------------------------===//

struct LinalgToHIVMTransposeOp : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() && !op.hasPureTensorSemantics()) {
      return op.emitOpError(
          "linalg::TansposeOp should have pure buffer or tensor Semantics!");
    }
    auto resultTypeRange =
        op.hasPureBufferSemantics() ? TypeRange() : TypeRange(op.getResult());
    rewriter.replaceOpWithNewOp<hivm::VTransposeOp>(op, resultTypeRange,
                                                    op.getInput(), op.getInit(),
                                                    op.getPermutationAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMArangeOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMArangeOp : public OpRewritePattern<hfusion::ArangeOp> {
  using OpRewritePattern<hfusion::ArangeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ArangeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() && !op.hasPureTensorSemantics()) {
      return op.emitOpError(
          "hfusion::ArangeOp should have pure buffer or tensor Semantics!");
    }
    auto resultTypeRange = op.hasPureBufferSemantics()
                               ? TypeRange()
                               : TypeRange(op->getResultTypes());
    rewriter.replaceOpWithNewOp<hivm::VArangeOp>(
        op, resultTypeRange, op.getInit(), op.getOffset(), op.getStrides());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMGatherOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMGatherOp : public OpRewritePattern<hfusion::GatherOp> {
  using OpRewritePattern<hfusion::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::GatherOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics() && !op.hasPureTensorSemantics()) {
      return op.emitOpError(
          "hfusion::GatherOp should have pure buffer or tensor Semantics!");
    }

    const auto rank = op.getSrc().getType().getRank();
    if (rank - 1 != static_cast<int64_t>(op.getAxis())) {
      return op.emitOpError(
          "can only lower hfusion.gather to hivm gather if axis is last dim");
    }

    auto resultTypeRange = op.hasPureBufferSemantics()
                               ? TypeRange()
                               : TypeRange(op->getResultTypes());
    rewriter.replaceOpWithNewOp<hivm::VGatherOp>(
        op, resultTypeRange, op.getSrc(), op.getIndex(), op.getInit());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionLoadOpToHIVMLoadOp
//===----------------------------------------------------------------------===//

struct HFusionLoadOpToHIVMLoadOp : public OpRewritePattern<hfusion::LoadOp> {
  using OpRewritePattern<hfusion::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::LoadOp op,
                                PatternRewriter &rewriter) const override {
    // convert hfusion::LoadOp to hivm::LoadOp
    auto src = cast<::mlir::Value>(*op.getInputs().begin());
    auto dst = cast<::mlir::Value>(*op.getOutputs().begin());
    auto res = op.getResultTensors();
    rewriter.replaceOpWithNewOp<hivm::LoadOp>(op, TypeRange(res), src, dst);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionStoreOpToHIVMStoreOp
//===----------------------------------------------------------------------===//

hivm::AtomicKind mapAtomicKindHFusionToHiVM(hfusion::AtomicKind hsAtKind) {
  hivm::AtomicKind hvAtKind;
  switch (hsAtKind) {
  case (hfusion::AtomicKind::NONE):
    hvAtKind = hivm::AtomicKind::NONE;
    break;
  case (hfusion::AtomicKind::ADD):
    hvAtKind = hivm::AtomicKind::ADD;
    break;
  case (hfusion::AtomicKind::MAX):
  case (hfusion::AtomicKind::UMAX): // In A5, it should never reach here
    hvAtKind = hivm::AtomicKind::MAX;
    break;
  case (hfusion::AtomicKind::MIN):
  case (hfusion::AtomicKind::UMIN): // In A5, it should never reach here
    hvAtKind = hivm::AtomicKind::MIN;
    break;
  case (hfusion::AtomicKind::AND):
    hvAtKind = hivm::AtomicKind::AND;
    break;
  case (hfusion::AtomicKind::OR):
    hvAtKind = hivm::AtomicKind::OR;
    break;
  case (hfusion::AtomicKind::XOR):
    hvAtKind = hivm::AtomicKind::XOR;
    break;
  case (hfusion::AtomicKind::CAS):
    hvAtKind = hivm::AtomicKind::CAS;
    break;
  case (hfusion::AtomicKind::XCHG):
    hvAtKind = hivm::AtomicKind::XCHG;
    break;
  }
  return hvAtKind;
}

struct HFusionStoreOpToHIVMStoreOp : public OpRewritePattern<hfusion::StoreOp> {
  using OpRewritePattern<hfusion::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::StoreOp op,
                                PatternRewriter &rewriter) const override {
    // convert hfusion::StoreOp to hivm::StoreOp
    auto src = cast<::mlir::Value>(*op.getInputs().begin());
    auto dst = cast<::mlir::Value>(*op.getOutputs().begin());
    auto res = op.getResultTensors();

    auto newStoreOp =
        rewriter.create<hivm::StoreOp>(op.getLoc(), TypeRange(res), src, dst);

    // Add atomic attr to hivm.store
    // hfusion.store has default atomic attr
    // then hivm.store should has one too.
    auto hsAtomicKind = op.getAtomicKind();
    hivm::AtomicKind hvAtomicKind = mapAtomicKindHFusionToHiVM(hsAtomicKind);
    newStoreOp.setAtomicKind(hvAtomicKind);

    rewriter.replaceOp(op, newStoreOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMBitcastOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMBitcastOp : public OpRewritePattern<hfusion::BitcastOp> {
  using OpRewritePattern<hfusion::BitcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::BitcastOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getInputs().front();
    Type dstType = op.getOutputs().front().getType();

    if (!op.hasPureTensorSemantics()) {
      return op->emitOpError(
          "hfusion.bitcast must be in Pure Tensor Semantics\n");
    }

    // TODO:  This check should be moved to verifier,
    // and/or change the design of hfusion.bitcast to non-destination style.
    if (!mlir::hfusion::reshape_utils::isContainerAllocator(
            op.getOutputs().front().getDefiningOp())) {
      LDBG("precision loss warning: hfusion.bitcast outs() must be a container "
           "allocator (empty tensor)");
    }

    hivm::BitcastOp hivmOp =
        rewriter.create<hivm::BitcastOp>(op->getLoc(), dstType, src);

    rewriter.replaceOp(op, hivmOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionPrintOpToHIVMDebugOp
//===----------------------------------------------------------------------===//

struct HFusionPrintOpToHIVMDebugOp : public OpRewritePattern<hfusion::PrintOp> {
  using OpRewritePattern<hfusion::PrintOp>::OpRewritePattern;

  static constexpr llvm::StringLiteral HIVMDebugTypePrint = "print";

  LogicalResult matchAndRewrite(hfusion::PrintOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    Value opArg = op.getArg();

    (void)(rewriter.replaceOpWithNewOp<hivm::DebugOp>(
        op, HIVMDebugTypePrint, op.getPrefix(), op.getHex(), opArg,
        hivm::TCoreTypeAttr::get(op->getContext(),
                                 hivm::TCoreType::CUBE_OR_VECTOR)));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionAssertOpToHIVMDebugOp
//===----------------------------------------------------------------------===//

struct HFusionAssertOpToHIVMDebugOp
    : public OpRewritePattern<hfusion::AssertOp> {
  using OpRewritePattern<hfusion::AssertOp>::OpRewritePattern;

  static constexpr llvm::StringLiteral HIVMDebugTypeAssert = "assert";

  LogicalResult matchAndRewrite(hfusion::AssertOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    (void)(rewriter.replaceOpWithNewOp<hivm::DebugOp>(
        op, HIVMDebugTypeAssert, op.getMsg(), false /* hex */, op.getCond(),
        hivm::TCoreTypeAttr::get(op->getContext(),
                                 hivm::TCoreType::CUBE_OR_VECTOR)));

    return success();
  }
};

struct HFusionToHIVMBarrierOp : public OpRewritePattern<hfusion::BarrierOp> {
  using OpRewritePattern<hfusion::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::BarrierOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto pipeAll = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
    rewriter.replaceOpWithNewOp<hivm::PipeBarrierOp>(op, pipeAll);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMMulExtOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMMulExtOp : public OpRewritePattern<hfusion::MulExtOp> {
  using OpRewritePattern<hfusion::MulExtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::MulExtOp op,
                                PatternRewriter &rewriter) const override {
    // convert hfusion::MulExtOp to hivm::VMulExtOp
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto hivmMulExtOp = rewriter.create<hivm::VMulExtOp>(
        op->getLoc(), op->getResultTypes(), ValueRange({lhs, rhs}),
        ValueRange{dsts});
    convertInvalidScalarOperands(hivmMulExtOp);
    rewriter.replaceOp(op, hivmMulExtOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HfusionToHIVMInterleaveOp
//===----------------------------------------------------------------------===//
struct HfusionToHIVMInterleaveOp
    : public OpRewritePattern<hfusion::InterleaveOp> {
  using OpRewritePattern<hfusion::InterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::InterleaveOp op,
                                PatternRewriter &rewriter) const override {
    // convert hfusion::InterleaveOp to hivm::VInterleaveOp
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto hivmInterleaveOp = rewriter.create<hivm::VInterleaveOp>(
        op->getLoc(), op->getResultTypes(), ValueRange(op.getInput()), dsts[0],
        hfusion::InterleaveOp::getInterLeaveChannelNums());
    rewriter.replaceOp(op, hivmInterleaveOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMDeinterleaveOp
//===----------------------------------------------------------------------===//
class HFusionToHIVMDeinterleaveOp
    : public OpRewritePattern<hfusion::DeinterleaveOp> {
  using OpRewritePattern<hfusion::DeinterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::DeinterleaveOp op,
                                PatternRewriter &rewriter) const override {
    // convert hfusion::DeinterleaveOp to hivm::VDeinterleaveOp
    Value input = op.getInput();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    hivm::DeinterleaveMode hivmDeinterleaveMode =
        hivm::symbolizeDeinterleaveMode(op.getDeInterLeaveChannelIdx()).value();

    // TODO: hfusion::DeinterleaveOp support channel num other than 2
    auto hivmDeinterleaveOp = rewriter.create<hivm::VDeinterleaveOp>(
        op->getLoc(), op->getResultTypes(), input, ValueRange{dsts},
        hfusion::DeinterleaveOp::getDeInterLeaveChannelNum(),
        hivmDeinterleaveMode);
    rewriter.replaceOp(op, hivmDeinterleaveOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMFlipOp
//===----------------------------------------------------------------------===//
struct HFusionToHIVMFlipOp : public OpRewritePattern<hfusion::FlipOp> {
  using OpRewritePattern<hfusion::FlipOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::FlipOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto hivmFlipOp = rewriter.create<hivm::VFlipOp>(
        op->getLoc(), op->getResultTypes(), op.getInput(), dsts[0],
        op.getFlipAxis());
    rewriter.replaceOp(op, hivmFlipOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMCumOp
//===----------------------------------------------------------------------===//
template <typename HFUSIONOP, typename HIVMOP>
struct HFusionToHIVMCumOp : public OpRewritePattern<HFUSIONOP> {
  using OpRewritePattern<HFUSIONOP>::OpRewritePattern;

  LogicalResult matchAndRewrite(HFUSIONOP op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    rewriter.replaceOpWithNewOp<HIVMOP>(op, op->getResultTypes(), op.getInput(),
                                        dsts[0], op.getCumDims(),
                                        op.getReverse());
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// HFusionToHIVM AtomicCasOp and AtomicXchgOp
// ===----------------------------------------------------------------------===//
template <typename HFUSIONOP, typename HIVMOP>
struct HFusionToHIVMAtomicOp : public OpRewritePattern<HFUSIONOP> {
  using OpRewritePattern<HFUSIONOP>::OpRewritePattern;

  LogicalResult matchAndRewrite(HFUSIONOP op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto hivmAtomicOp = rewriter.create<HIVMOP>(
        op->getLoc(), op->getResultTypes(), op.getInput(), op.getDst());
    rewriter.replaceOp(op, hivmAtomicOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMSortOp
//===----------------------------------------------------------------------===//
struct HFusionToHIVMSortOp : public OpRewritePattern<hfusion::SortOp> {
  using OpRewritePattern<hfusion::SortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::SortOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    rewriter.replaceOpWithNewOp<hivm::VSortOp>(
        op, op->getResultTypes(), op.getSrc(), ValueRange{dsts},
        op.getDescending(), op.getSortAxis());
    return success();
  }
};

struct HFusionAttrsLowering : public OpRewritePattern<annotation::MarkOp> {
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(annotation::MarkOp op,
                                PatternRewriter &rewriter) const override {
    auto hfusionMultiBufferAttr = hfusion::MultiBufferAttr::name;
    auto hfusionStrideAlignDimsAttr = hfusion::StrideAlignDimsAttr::name;
    auto hfusionStrideAlignValueInByteAttr =
        hfusion::StrideAlignValueInByteAttr::name;

    bool attrMatchStatus = false;

    for (auto iter : op->getAttrDictionary()) {
      if (iter.getName() == hfusionMultiBufferAttr) {
        auto attrVal = op->getAttr(hfusionMultiBufferAttr);
        op->removeAttr(hfusionMultiBufferAttr);
        auto hivmMultiBufferAttr = hivm::MultiBufferAttr::name;
        op->setAttr(hivmMultiBufferAttr, attrVal);
        attrMatchStatus = true;
      } else if (iter.getName() == hfusionStrideAlignDimsAttr) {
        auto attrVal = op->getAttr(hfusionStrideAlignDimsAttr);
        op->removeAttr(hfusionStrideAlignDimsAttr);
        auto hivmStrideAlignDimsAttr = hivm::StrideAlignDimsAttr::name;
        op->setAttr(hivmStrideAlignDimsAttr, attrVal);
        attrMatchStatus = true;
      } else if (iter.getName() == hfusionStrideAlignValueInByteAttr) {
        auto attrVal = op->getAttr(hfusionStrideAlignValueInByteAttr);
        op->removeAttr(hfusionStrideAlignValueInByteAttr);
        auto hivmStrideAlignValueInByteAttr =
            hivm::StrideAlignValueInByteAttr::name;
        op->setAttr(hivmStrideAlignValueInByteAttr, attrVal);
        attrMatchStatus = true;
      }
    }
    if (!attrMatchStatus) {
      return failure();
    }
    return success();
  }
};

struct HFusionBindSubBlockAttrLowing : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!forOp->hasAttrOfType<UnitAttr>(hfusion::BindSubBlockAttr::name))
      return failure();
    rewriter.modifyOpInPlace(
        forOp, [&]() { forOp->removeAttr(hfusion::BindSubBlockAttr::name); });
    setSubBlockMapping(rewriter, forOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMEmbeddingGatherOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMEmbeddingGatherOp
    : public OpRewritePattern<hfusion::EmbeddingGatherOp> {
  using OpRewritePattern<hfusion::EmbeddingGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::EmbeddingGatherOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto idx = op.getIndex();
    auto dst = op.getDst();
    auto bound = op.getBound();
    auto offsets = op.getOffsets();
    auto numels = op.getNumels();
    auto ret = op.getResult();
    auto retTy = ret.getType();

    auto gatherOp = rewriter.create<hivm::EmbeddingGatherOp>(
        loc, retTy, src, idx, dst, bound, offsets, numels);

    gatherOp->setAttr(VFModeAttr::name,
                      VFModeAttr::get(op->getContext(), VFMode::SIMT));

    rewriter.replaceOp(op, gatherOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMGatherLoadOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMGatherLoadOp
    : public OpRewritePattern<hfusion::GatherLoadOp> {
  using OpRewritePattern<hfusion::GatherLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::GatherLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<hivm::GatherLoadOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands());

    if (auto boundary = op.getBoundaryCheckAttr()) {
      newOp.setBoundaryCheck(boundary);
    }
    if (auto padding = op.getPaddingAttr()) {
      auto attrVal = (hivm::PaddingOption)padding.getValue();
      newOp.setPaddingAttr(
          hivm::PaddingOptionAttr::get(op->getContext(), attrVal));
    }
    if (auto evict = op.getEvictAttr()) {
      auto attrVal = (hivm::EvictionPolicy)evict.getValue();
      newOp.setEvictAttr(
          hivm::EvictionPolicyAttr::get(op->getContext(), attrVal));
    }
    if (auto cache = op.getCacheAttr()) {
      auto attrVal = (hivm::CacheModifier)cache.getValue();
      newOp.setCacheAttr(
          hivm::CacheModifierAttr::get(op->getContext(), attrVal));
    }
    if (auto isVolatile = op.getIsVolatileAttr()) {
      newOp.setIsVolatileAttr(isVolatile);
    }

    rewriter.replaceOp(op, newOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMScatterStoreOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMScatterStoreOp
    : public OpRewritePattern<hfusion::ScatterStoreOp> {
  using OpRewritePattern<hfusion::ScatterStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ScatterStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<hivm::ScatterStoreOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands());

    if (auto boundaryCheck = op.getBoundaryCheckAttr()) {
      newOp.setBoundaryCheck(boundaryCheck);
    }
    if (auto evict = op.getEvictAttr()) {
      auto attrVal = (hivm::EvictionPolicy)evict.getValue();
      newOp.setEvictAttr(
          hivm::EvictionPolicyAttr::get(op->getContext(), attrVal));
    }
    if (auto cache = op.getCacheAttr()) {
      auto attrVal = (hivm::CacheModifier)cache.getValue();
      newOp.setCacheAttr(
          hivm::CacheModifierAttr::get(op->getContext(), attrVal));
    }

    rewriter.replaceOp(op, newOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMIndirectLoadOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMIndirectLoadOp
    : public OpRewritePattern<hfusion::IndirectLoadOp> {
  using OpRewritePattern<hfusion::IndirectLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::IndirectLoadOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ret = op.getResult();
    auto retTy = ret.getType();
    auto src = op.getSrc();
    auto offset = op.getOffsets();
    auto dst = op.getDst();
    auto mask = op.getMask();
    auto other = op.getOther();

    auto indirectLoadOp = rewriter.create<hivm::IndirectLoadOp>(
        loc, retTy, src, offset, dst, mask, other);

    indirectLoadOp->setAttr(VFModeAttr::name,
                            VFModeAttr::get(op->getContext(), VFMode::SIMT));

    rewriter.replaceOp(op, indirectLoadOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMIndirectStoreOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMIndirectStoreOp
    : public OpRewritePattern<hfusion::IndirectStoreOp> {
  using OpRewritePattern<hfusion::IndirectStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::IndirectStoreOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto offset = op.getOffsets();
    auto dst = op.getDst();
    auto mask = op.getMask();

    auto indirectStoreOp =
        rewriter.create<hivm::IndirectStoreOp>(loc, dst, offset, src, mask);

    indirectStoreOp->setAttr(VFModeAttr::name,
                             VFModeAttr::get(op->getContext(), VFMode::SIMT));

    rewriter.replaceOp(op, indirectStoreOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMGatherTOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMGatherTOp : public OpRewritePattern<hfusion::GatherTOp> {
  using OpRewritePattern<hfusion::GatherTOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::GatherTOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ret = op.getResult();
    auto retTy = ret.getType();
    auto src = op.getSrc();
    auto index = op.getIndex();
    auto dst = op.getDst();
    auto bound = op.getBound();
    auto dim = op.getDim();
    auto src_stride = op.getSrcStride();
    auto index_shape = op.getIndexShape();
    auto offsets = op.getOffsets();

    auto gatherTOp =
        rewriter.create<hivm::GatherTOp>(loc, retTy, src, index, dst, bound,
                                         dim, src_stride, index_shape, offsets);

    gatherTOp->setAttr(VFModeAttr::name,
                       VFModeAttr::get(op->getContext(), VFMode::SIMT));

    rewriter.replaceOp(op, gatherTOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMIndexPutOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMIndexPutOp : public OpRewritePattern<hfusion::IndexPutOp> {
  using OpRewritePattern<hfusion::IndexPutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::IndexPutOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto dst = op.getDst();
    auto index = op.getIndex();
    auto value = op.getValue();
    auto scatter_dim = op.getScatterDim();
    auto bound = op.getBound();
    auto end_offset = op.getEndOffset();
    auto start_offset = op.getStartOffset();
    auto dst_stride = op.getDstStride();

    auto indexPutOp = rewriter.create<hivm::IndexPutOp>(
        loc, dst, index, value, scatter_dim, bound, end_offset, start_offset,
        dst_stride);

    indexPutOp->setAttr(VFModeAttr::name,
                        VFModeAttr::get(op->getContext(), VFMode::SIMT));

    rewriter.replaceOp(op, indexPutOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// HFusionToHIVMScatterTOp
//===----------------------------------------------------------------------===//

struct HFusionToHIVMScatterTOp : public OpRewritePattern<hfusion::ScatterTOp> {
  using OpRewritePattern<hfusion::ScatterTOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ScatterTOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto dst = op.getDst();
    auto value = op.getValue();
    auto index_tile = op.getIndexTile();
    auto index_boundary = op.getIndexBoundary();
    auto dim = op.getDim();
    auto dst_stride = op.getDstStride();
    auto index_shape = op.getIndexShape();
    auto offsets = op.getOffsets();

    auto scatterTOp = rewriter.create<hivm::ScatterTOp>(
        loc, dst, value, index_tile, index_boundary, dim, dst_stride,
        index_shape, offsets);

    scatterTOp->setAttr(VFModeAttr::name,
                        VFModeAttr::get(op->getContext(), VFMode::SIMT));

    rewriter.replaceOp(op, scatterTOp);

    return success();
  }
};

void populateHIVMOpRewritingRule(RewritePatternSet &patterns) {
  patterns.add<HFusionAttrsLowering, HFusionBindSubBlockAttrLowing>(
      patterns.getContext());
}

void populateLowerHFusionToHIVMPattern(RewritePatternSet &patterns) {
  // clang-format off
  (void)patterns.add<
    ExtractScalarForBinaryShiftOp,
    HFusionElemwiseOpConverter<linalg::ElemwiseBinaryOp>,
    HFusionElemwiseOpConverter<linalg::ElemwiseUnaryOp>,
    HFusionElemwiseOpConverter<hfusion::ElemwiseUnaryOp>,
    HFusionElemwiseOpConverter<hfusion::ElemwiseBinaryOp>,
    HFusionElemwiseOpConverter<hfusion::CompareOp>,
    HFusionElemwiseOpConverter<hfusion::SelectOp>,
    HFusionElemwiseOpConverter<hfusion::CastOp>,
    HFusionToHIVMBitcastOp,
    LinalgBrcOpToHIVMBrcOp,
    LinalgFillOpToHIVMBrcOp,
    LinalgToHIVMCopyOp,
    HFusionLoadOpToHIVMLoadOp,
    HFusionStoreOpToHIVMStoreOp,
    LinalgToHIVMTransposeOp,
    HFusionToHIVMArangeOp,
    HFusionToHIVMGatherOp,
    HFusionToHIVMEmbeddingGatherOp,
    HFusionToHIVMGatherLoadOp,
    HFusionToHIVMScatterStoreOp,
    HFusionToHIVMIndirectLoadOp,
    HFusionToHIVMIndirectStoreOp,
    HFusionToHIVMGatherTOp,
    HFusionToHIVMIndexPutOp,
    HFusionToHIVMScatterTOp,
    HFusionPrintOpToHIVMDebugOp,
    HFusionAssertOpToHIVMDebugOp,
    HFusionToHIVMBarrierOp,
    HFusionToHIVMMulExtOp,
    HfusionToHIVMInterleaveOp,
    HFusionToHIVMDeinterleaveOp,
    HFusionToHIVMFlipOp,
    HFusionToHIVMSortOp,
    HFusionToHIVMCumOp<hfusion::CumsumOp, hivm::VCumsumOp>,
    HFusionToHIVMCumOp<hfusion::CumprodOp, hivm::VCumprodOp>,
    HFusionToHIVMAtomicOp<hfusion::AtomicCasOp, hivm::AtomicCasOp>,
    HFusionToHIVMAtomicOp<hfusion::AtomicXchgOp, hivm::AtomicXchgOp>
  >(patterns.getContext());
  // clang-format on
}

struct ConvertHFusionToHIVMPass
    : public impl::ConvertHFusionToHIVMBase<ConvertHFusionToHIVMPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    ConvertHFusionToHIVMOptions options = {this->mmMapMode};

    target.addLegalDialect<hivm::HIVMDialect, memref::MemRefDialect,
                           bufferization::BufferizationDialect,
                           tensor::TensorDialect, arith::ArithDialect,
                           affine::AffineDialect, scf::SCFDialect,
                           func::FuncDialect>();
    target.addIllegalDialect<linalg::LinalgDialect, hfusion::HFusionDialect>();

    populateLowerHFusionToHIVMPattern(patterns);
    populateReductionPatternsAndLegality(patterns, target);
    populateMatmulPatternsAndLegality(patterns, target, options);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }

    Operation *moduleOp = getOperation();
    auto *ctx = &getContext();
    moduleOp->walk([&](func::FuncOp funcOp) {
      if (hacc::utils::isHost(funcOp))
        // avoid convert host op to hivm op
        return;

      // rewrite op within cur funcOp
      RewritePatternSet hivmOpPatterns(ctx);
      populateHIVMOpRewritingRule(hivmOpPatterns);
      (void)applyPatternsGreedily(funcOp, std::move(hivmOpPatterns));
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createHFusionToHIVMConversionPass() {
  return std::make_unique<ConvertHFusionToHIVMPass>();
}

std::unique_ptr<Pass> mlir::createHFusionToHIVMConversionPass(
    const ConvertHFusionToHIVMOptions &option) {
  return std::make_unique<ConvertHFusionToHIVMPass>(option);
}
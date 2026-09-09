//===- VectorizeElementwise.cpp - Vectorize HIVM elementwise ops ----------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMVectorize.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::hivm {
namespace {

enum class BitwiseKind { And, Or, Xor };

Value createSplatVector(OpBuilder &builder, Location loc, VectorType vectorType,
                        Attribute value) {
  return builder.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(vectorType, value));
}

Value createVectorBitwiseOp(OpBuilder &builder, Location loc, BitwiseKind kind,
                            Value lhs, Value rhs) {
  auto applyInt = [&](Value lhsInt, Value rhsInt) -> Value {
    switch (kind) {
    case BitwiseKind::And:
      return builder.create<arith::AndIOp>(loc, lhsInt, rhsInt);
    case BitwiseKind::Or:
      return builder.create<arith::OrIOp>(loc, lhsInt, rhsInt);
    case BitwiseKind::Xor:
      return builder.create<arith::XOrIOp>(loc, lhsInt, rhsInt);
    }
    llvm_unreachable("unhandled bitwise kind");
  };

  Type elemType = getElementTypeOrSelf(lhs.getType());
  if (isa<IntegerType>(elemType))
    return applyInt(lhs, rhs);

  auto floatType = cast<FloatType>(elemType);

  auto srcVec = cast<VectorType>(lhs.getType());
  auto intType = IntegerType::get(builder.getContext(), floatType.getWidth());
  auto intVec = VectorType::get(srcVec.getShape(), intType);
  Value lhsInt = builder.create<arith::BitcastOp>(loc, intVec, lhs);
  Value rhsInt = builder.create<arith::BitcastOp>(loc, intVec, rhs);
  Value resultInt = applyInt(lhsInt, rhsInt);
  return builder.create<arith::BitcastOp>(loc, srcVec, resultInt);
}

Value padZero(OpBuilder &builder, Location loc, Type elemType) {
  return builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(elemType));
}

template <arith::AtomicRMWKind FloatKind, arith::AtomicRMWKind IntKind>
Value arithIdentityPad(OpBuilder &builder, Location loc, Type elemType) {
  return arith::getIdentityValue(isa<FloatType>(elemType) ? FloatKind : IntKind,
                                 elemType, builder, loc);
}

bool isSupportedVectorCast(Type srcType, Type dstType, TypeFn casting) {
  unsigned srcWidth = srcType.getIntOrFloatBitWidth();
  unsigned dstWidth = dstType.getIntOrFloatBitWidth();

  if (casting == TypeFn::bitcast)
    return srcWidth == dstWidth;

  if (!isa<FloatType>(srcType) || !isa<FloatType>(dstType))
    return true;

  // extf/truncf cannot represent a conversion between distinct float types
  // of the same width, while bitcast has different semantics.
  return srcType == dstType || srcWidth != dstWidth;
}

Value createCastVectorOp(RewriterBase &rewriter, Location loc, Value src,
                         Type dstElemType, TypeFn casting) {
  auto srcVecType = cast<VectorType>(src.getType());
  Type srcElemType = srcVecType.getElementType();
  Type dstVecType = VectorType::get(srcVecType.getShape(), dstElemType);
  if (!isSupportedVectorCast(srcElemType, dstElemType, casting)) {
    llvm::report_fatal_error("unsupported vector cast");
  }

  if (casting == TypeFn::bitcast)
    return rewriter.create<arith::BitcastOp>(loc, dstVecType, src);

  bool srcIsFloat = isa<FloatType>(srcElemType);
  bool dstIsFloat = isa<FloatType>(dstElemType);
  bool isSigned = casting == TypeFn::cast_signed;
  unsigned srcWidth = srcElemType.getIntOrFloatBitWidth();
  unsigned dstWidth = dstElemType.getIntOrFloatBitWidth();

  if (srcIsFloat && dstIsFloat) {
    if (srcWidth > dstWidth)
      return rewriter.create<arith::TruncFOp>(loc, dstVecType, src);
    if (srcWidth < dstWidth)
      return rewriter.create<arith::ExtFOp>(loc, dstVecType, src);
    return rewriter.create<math::RoundOp>(loc, dstVecType, src);
  }

  if (srcIsFloat) {
    if (isSigned)
      return rewriter.create<arith::FPToSIOp>(loc, dstVecType, src);
    return rewriter.create<arith::FPToUIOp>(loc, dstVecType, src);
  }

  if (dstIsFloat) {
    if (isSigned)
      return rewriter.create<arith::SIToFPOp>(loc, dstVecType, src);
    return rewriter.create<arith::UIToFPOp>(loc, dstVecType, src);
  }

  if (srcWidth > dstWidth)
    return rewriter.create<arith::TruncIOp>(loc, dstVecType, src);
  if (srcWidth < dstWidth) {
    if (isSigned)
      return rewriter.create<arith::ExtSIOp>(loc, dstVecType, src);
    return rewriter.create<arith::ExtUIOp>(loc, dstVecType, src);
  }
  return src;
}

bool isScalarOrRankedLike(Value value, ArrayRef<int64_t> vectorSizes) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  return !shapedType ||
         shapedType.getRank() == static_cast<int64_t>(vectorSizes.size());
}

bool hasArithCompatibleElementType(Value value) {
  Type elemType = getElementTypeOrSelf(value.getType());
  if (isa<FloatType>(elemType))
    return true;
  auto intType = dyn_cast<IntegerType>(elemType);
  return intType && intType.isSignless();
}

// Elementwise vectorization requires integer operands to have signless types.
bool hasArithCompatibleElementTypes(HIVMStructuredOp op) {
  return llvm::all_of(op.getDpsInputs(), hasArithCompatibleElementType) &&
         llvm::all_of(op.getDpsInits(), hasArithCompatibleElementType);
}

bool hasFloatInputElementType(Operation *op) {
  auto hivmOp = dyn_cast<HIVMStructuredOp>(op);
  if (!hivmOp || hivmOp.getDpsInputs().empty())
    return false;
  return isa<FloatType>(
      getElementTypeOrSelf(hivmOp.getDpsInputs().front().getType()));
}

LogicalResult checkElementwisePreconditions(Operation *op,
                                            ArrayRef<int64_t> vectorSizes) {
  if (failed(checkVectorizePreconditions(op, vectorSizes)))
    return failure();

  auto hivmOp = cast<HIVMStructuredOp>(op);
  // Explicit vtranspose has its own lowering.
  // Currently, no use case for OTF transpose in HIVM Op.
  if (!hivmOp.getPermutationArray().empty())
    return failure();

  if (!llvm::all_of(hivmOp.getDpsInputs(), [&](Value input) {
        return isScalarOrRankedLike(input, vectorSizes);
      }))
    return failure();

  if (!hasArithCompatibleElementTypes(hivmOp))
    return failure();

  return success();
}

arith::CmpFPredicate getCmpFPredicate(CompareMode mode) {
  switch (mode) {
  case CompareMode::EQ:
    return arith::CmpFPredicate::OEQ;
  case CompareMode::NE:
    return arith::CmpFPredicate::UNE;
  case CompareMode::LT:
    return arith::CmpFPredicate::OLT;
  case CompareMode::GT:
    return arith::CmpFPredicate::OGT;
  case CompareMode::LE:
    return arith::CmpFPredicate::OLE;
  case CompareMode::GE:
    return arith::CmpFPredicate::OGE;
  }
  llvm_unreachable("unhandled compare mode");
}

arith::CmpIPredicate getCmpIPredicate(CompareMode mode, bool isSigned) {
  switch (mode) {
  case CompareMode::EQ:
    return arith::CmpIPredicate::eq;
  case CompareMode::NE:
    return arith::CmpIPredicate::ne;
  case CompareMode::LT:
    return isSigned ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult;
  case CompareMode::GT:
    return isSigned ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt;
  case CompareMode::LE:
    return isSigned ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule;
  case CompareMode::GE:
    return isSigned ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge;
  }
  llvm_unreachable("unhandled compare mode");
}

Value toI1Condition(RewriterBase &rewriter, Location loc, Value cond) {
  auto condVecType = cast<VectorType>(cond.getType());
  if (condVecType.getElementType().isInteger(1))
    return cond;
  Value zero = createSplatVector(
      rewriter, loc, condVecType,
      rewriter.getIntegerAttr(condVecType.getElementType(), 0));
  return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, cond,
                                        zero);
}

LogicalResult vectorizeElementwiseOp(
    Operation *op, RewriterBase &rewriter, ArrayRef<int64_t> vectorSizes,
    function_ref<Value(OpBuilder &, Location, Type)> makePad,
    function_ref<Value(RewriterBase &, ValueRange)> compute) {
  if (failed(checkElementwisePreconditions(op, vectorSizes)))
    return failure();

  auto hivmOp = cast<HIVMStructuredOp>(op);
  Value output = hivmOp.getDpsInitOperand(0)->get();
  Location loc = hivmOp.getLoc();
  SmallVector<Value> inputs = hivmOp.getDpsInputs();

  Value mask = createShapeMask(rewriter, loc, output, vectorSizes);

  SmallVector<Value> vectorOperands;
  for (Value input : inputs) {
    Value padding =
        makePad(rewriter, loc, getElementTypeOrSelf(input.getType()));
    vectorOperands.push_back(
        readOperand(rewriter, loc, input, vectorSizes, padding, mask));
  }

  Value resultVector = compute(rewriter, vectorOperands);
  auto writeOp =
      createMaskedTransferWrite(rewriter, loc, resultVector, output, mask);

  if (hivmOp->getNumResults() > 0) {
    rewriter.replaceOp(op, writeOp.getResult());
  } else {
    rewriter.eraseOp(op);
  }

  return success();
}

template <typename OpF, typename OpI, arith::AtomicRMWKind NeutralF,
          arith::AtomicRMWKind NeutralI>
LogicalResult vectorizeArithBinary(Operation *op, RewriterBase &rewriter,
                                   ArrayRef<int64_t> vectorSizes) {
  auto makePad = [](OpBuilder &builder, Location loc, Type elemType) -> Value {
    return arithIdentityPad<NeutralF, NeutralI>(builder, loc, elemType);
  };
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    if (isa<FloatType>(getElementTypeOrSelf(vecOps[0].getType())))
      return r.create<OpF>(op->getLoc(), vecOps[0], vecOps[1]);
    return r.create<OpI>(op->getLoc(), vecOps[0], vecOps[1]);
  };
  return vectorizeElementwiseOp(op, rewriter, vectorSizes, makePad, compute);
}

template <typename OpF, typename OpSI, typename OpUI,
          arith::AtomicRMWKind NeutralF, arith::AtomicRMWKind NeutralSI,
          arith::AtomicRMWKind NeutralUI>
LogicalResult vectorizeArithBinary(Operation *op, RewriterBase &rewriter,
                                   ArrayRef<int64_t> vectorSizes,
                                   bool isSigned) {
  auto makePad = [&](OpBuilder &builder, Location loc, Type elemType) -> Value {
    arith::AtomicRMWKind kind = NeutralF;
    if (isa<IntegerType>(elemType))
      kind = isSigned ? NeutralSI : NeutralUI;
    return arith::getIdentityValue(kind, elemType, builder, loc);
  };
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    if (isa<FloatType>(getElementTypeOrSelf(vecOps[0].getType())))
      return r.create<OpF>(op->getLoc(), vecOps[0], vecOps[1]);
    if (isSigned)
      return r.create<OpSI>(op->getLoc(), vecOps[0], vecOps[1]);
    return r.create<OpUI>(op->getLoc(), vecOps[0], vecOps[1]);
  };
  return vectorizeElementwiseOp(op, rewriter, vectorSizes, makePad, compute);
}

LogicalResult vectorizeBitwise(Operation *op, RewriterBase &rewriter,
                               ArrayRef<int64_t> vectorSizes,
                               BitwiseKind kind) {
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    return createVectorBitwiseOp(r, op->getLoc(), kind, vecOps[0], vecOps[1]);
  };
  return vectorizeElementwiseOp(op, rewriter, vectorSizes, padZero, compute);
}

template <typename UnaryOpF, typename UnaryOpI>
LogicalResult vectorizeUnary(Operation *op, RewriterBase &rewriter,
                             ArrayRef<int64_t> vectorSizes) {
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    Type elemType = getElementTypeOrSelf(vecOps[0].getType());
    if (isa<FloatType>(elemType))
      return r.create<UnaryOpF>(op->getLoc(), vecOps[0]);
    return r.create<UnaryOpI>(op->getLoc(), vecOps[0]);
  };
  return vectorizeElementwiseOp(op, rewriter, vectorSizes, padZero, compute);
}

template <typename UnaryOpF>
LogicalResult vectorizeFloatUnary(Operation *op, RewriterBase &rewriter,
                                  ArrayRef<int64_t> vectorSizes) {
  if (!hasFloatInputElementType(op))
    return failure();
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    return r.create<UnaryOpF>(op->getLoc(), vecOps[0]);
  };
  return vectorizeElementwiseOp(op, rewriter, vectorSizes, padZero, compute);
}

} // namespace

//===----------------------------------------------------------------------===//
// Per-op vectorize()
//===----------------------------------------------------------------------===//

LogicalResult VAddOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeArithBinary<arith::AddFOp, arith::AddIOp,
                              arith::AtomicRMWKind::addf,
                              arith::AtomicRMWKind::addi>(*this, rewriter,
                                                          vectorSizes);
}

LogicalResult VSubOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeArithBinary<arith::SubFOp, arith::SubIOp,
                              arith::AtomicRMWKind::addf,
                              arith::AtomicRMWKind::addi>(*this, rewriter,
                                                          vectorSizes);
}

LogicalResult VMulOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeArithBinary<arith::MulFOp, arith::MulIOp,
                              arith::AtomicRMWKind::mulf,
                              arith::AtomicRMWKind::muli>(*this, rewriter,
                                                          vectorSizes);
}

LogicalResult VDivOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  bool isSigned = getIsSigned();
  bool isHighPrecision = getIsHP();
  auto makePad = [](OpBuilder &builder, Location loc, Type elemType) -> Value {
    return arithIdentityPad<arith::AtomicRMWKind::mulf,
                            arith::AtomicRMWKind::muli>(builder, loc, elemType);
  };
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    if (isa<FloatType>(getElementTypeOrSelf(vecOps[0].getType()))) {
      if (isHighPrecision)
        return r.create<mathExt::DivFHPOp>(getLoc(), vecOps[0].getType(),
                                           vecOps[0], vecOps[1]);
      return r.create<arith::DivFOp>(getLoc(), vecOps[0], vecOps[1]);
    }
    if (isSigned)
      return r.create<arith::DivSIOp>(getLoc(), vecOps[0], vecOps[1]);
    return r.create<arith::DivUIOp>(getLoc(), vecOps[0], vecOps[1]);
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, makePad, compute);
}

LogicalResult VMaxOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeArithBinary<arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp,
                              arith::AtomicRMWKind::maximumf,
                              arith::AtomicRMWKind::maxs,
                              arith::AtomicRMWKind::maxu>(
      *this, rewriter, vectorSizes, getIsSigned());
}

LogicalResult VMinOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeArithBinary<arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp,
                              arith::AtomicRMWKind::minimumf,
                              arith::AtomicRMWKind::mins,
                              arith::AtomicRMWKind::minu>(
      *this, rewriter, vectorSizes, getIsSigned());
}

LogicalResult VAndOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeBitwise(*this, rewriter, vectorSizes, BitwiseKind::And);
}

LogicalResult VOrOp::vectorize(RewriterBase &rewriter,
                               ArrayRef<int64_t> vectorSizes) {
  return vectorizeBitwise(*this, rewriter, vectorSizes, BitwiseKind::Or);
}

LogicalResult VXorOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeBitwise(*this, rewriter, vectorSizes, BitwiseKind::Xor);
}

LogicalResult VAbsOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeUnary<math::AbsFOp, math::AbsIOp>(*this, rewriter,
                                                    vectorSizes);
}

LogicalResult VExpOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeFloatUnary<math::ExpOp>(*this, rewriter, vectorSizes);
}

LogicalResult VLnOp::vectorize(RewriterBase &rewriter,
                               ArrayRef<int64_t> vectorSizes) {
  return vectorizeFloatUnary<math::LogOp>(*this, rewriter, vectorSizes);
}

LogicalResult VSqrtOp::vectorize(RewriterBase &rewriter,
                                 ArrayRef<int64_t> vectorSizes) {
  return vectorizeFloatUnary<math::SqrtOp>(*this, rewriter, vectorSizes);
}

LogicalResult VRsqrtOp::vectorize(RewriterBase &rewriter,
                                  ArrayRef<int64_t> vectorSizes) {
  return vectorizeFloatUnary<math::RsqrtOp>(*this, rewriter, vectorSizes);
}

LogicalResult VTanhOp::vectorize(RewriterBase &rewriter,
                                 ArrayRef<int64_t> vectorSizes) {
  return vectorizeFloatUnary<math::TanhOp>(*this, rewriter, vectorSizes);
}

LogicalResult VSinOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeFloatUnary<math::SinOp>(*this, rewriter, vectorSizes);
}

LogicalResult VCosOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeFloatUnary<math::CosOp>(*this, rewriter, vectorSizes);
}

LogicalResult VErfOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  return vectorizeFloatUnary<math::ErfOp>(*this, rewriter, vectorSizes);
}

LogicalResult VCastOp::vectorize(RewriterBase &rewriter,
                                 ArrayRef<int64_t> vectorSizes) {
  Type srcElemType = getElementTypeOrSelf(getDpsInputs().front().getType());
  Type dstElemType =
      getElementTypeOrSelf(getDpsInitOperand(0)->get().getType());
  if (!isSupportedVectorCast(srcElemType, dstElemType, getCast()))
    return failure();

  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    return createCastVectorOp(r, getLoc(), vecOps[0], dstElemType, getCast());
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, padZero, compute);
}

LogicalResult VCmpOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  Type dstElemType =
      getElementTypeOrSelf(getDpsInitOperand(0)->get().getType());
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    auto srcVecType = cast<VectorType>(vecOps[0].getType());
    Type srcElemType = srcVecType.getElementType();
    auto boolVecType = VectorType::get(srcVecType.getShape(), r.getI1Type());

    Value cmp;
    if (isa<FloatType>(srcElemType)) {
      cmp = r.create<arith::CmpFOp>(getLoc(), boolVecType,
                                    getCmpFPredicate(getCompareMode()),
                                    vecOps[0], vecOps[1]);
    } else {
      bool isSigned = srcElemType.getIntOrFloatBitWidth() > 1 && getIsSigned();
      cmp = r.create<arith::CmpIOp>(
          getLoc(), boolVecType, getCmpIPredicate(getCompareMode(), isSigned),
          vecOps[0], vecOps[1]);
    }

    if (dstElemType.isInteger(8))
      return r.create<arith::ExtUIOp>(
          getLoc(), VectorType::get(srcVecType.getShape(), dstElemType), cmp);
    return cmp;
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, padZero, compute);
}

LogicalResult VSelOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    Value condition = toI1Condition(r, getLoc(), vecOps[0]);
    return r.create<arith::SelectOp>(getLoc(), condition, vecOps[1], vecOps[2]);
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, padZero, compute);
}

LogicalResult VShLOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    return r.create<arith::ShLIOp>(getLoc(), vecOps[0], vecOps[1]);
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, padZero, compute);
}

LogicalResult VShROp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  bool isSigned = getIsSigned();
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    if (isSigned)
      return r.create<arith::ShRSIOp>(getLoc(), vecOps[0], vecOps[1]);
    return r.create<arith::ShRUIOp>(getLoc(), vecOps[0], vecOps[1]);
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, padZero, compute);
}

LogicalResult VPowOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    Type elemType = getElementTypeOrSelf(vecOps[0].getType());
    if (isa<FloatType>(elemType))
      return r.create<math::PowFOp>(getLoc(), vecOps[0], vecOps[1]);
    return r.create<math::IPowIOp>(getLoc(), vecOps[0], vecOps[1]);
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, padZero, compute);
}

LogicalResult VRecOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  if (!hasFloatInputElementType(*this))
    return failure();

  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    auto vectorType = cast<VectorType>(vecOps[0].getType());
    auto floatType = cast<FloatType>(vectorType.getElementType());
    Value one = createSplatVector(r, getLoc(), vectorType,
                                  r.getFloatAttr(floatType, 1.0));
    return r.create<arith::DivFOp>(getLoc(), one, vecOps[0]);
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, padZero, compute);
}

LogicalResult VReluOp::vectorize(RewriterBase &rewriter,
                                 ArrayRef<int64_t> vectorSizes) {
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    auto vectorType = cast<VectorType>(vecOps[0].getType());
    Type elemType = vectorType.getElementType();
    Value zero =
        createSplatVector(r, getLoc(), vectorType, r.getZeroAttr(elemType));
    if (isa<FloatType>(elemType))
      return r.create<arith::MaximumFOp>(getLoc(), vecOps[0], zero);
    return r.create<arith::MaxSIOp>(getLoc(), vecOps[0], zero);
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, padZero, compute);
}

LogicalResult VNotOp::vectorize(RewriterBase &rewriter,
                                ArrayRef<int64_t> vectorSizes) {
  auto compute = [&](RewriterBase &r, ValueRange vecOps) -> Value {
    auto srcVec = cast<VectorType>(vecOps[0].getType());
    Type elemType = srcVec.getElementType();
    if (auto intType = dyn_cast<IntegerType>(elemType)) {
      Value ones =
          createSplatVector(r, getLoc(), srcVec, r.getIntegerAttr(intType, -1));
      return r.create<arith::XOrIOp>(getLoc(), vecOps[0], ones);
    }
    auto floatType = cast<FloatType>(elemType);
    auto intType = IntegerType::get(r.getContext(), floatType.getWidth());
    auto intVec = VectorType::get(srcVec.getShape(), intType);
    Value srcInt = r.create<arith::BitcastOp>(getLoc(), intVec, vecOps[0]);
    Value ones =
        createSplatVector(r, getLoc(), intVec, r.getIntegerAttr(intType, -1));
    Value resultInt = r.create<arith::XOrIOp>(getLoc(), srcInt, ones);
    return r.create<arith::BitcastOp>(getLoc(), srcVec, resultInt);
  };
  return vectorizeElementwiseOp(*this, rewriter, vectorSizes, padZero, compute);
}

} // namespace mlir::hivm

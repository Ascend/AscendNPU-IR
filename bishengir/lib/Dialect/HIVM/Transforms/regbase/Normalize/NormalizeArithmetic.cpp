//===-------- NormalizeArithmeticTraits.cpp --------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeArithmeticTemplate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"

namespace mlir {
/// Normalizes `rsqrt(x)` to `rec(sqrt(x))`
struct HIVMNormalizeRSqrtTraits
    : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeRSqrt(hivm::VRsqrtOp op) {
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() && op.getTranspose().empty();
  }
};

/// Normalizes `mul(rec_like(x), y)` to `div(y, x)`
/// (1/b) * a -> a/b
/// a * (1/b) -> a/b
struct HIVMNormalizeMulRecTraits : public hivm::NormalizeTraitsBase {
  using RecOpType = hivm::VRecOp;
  using DivOpType = hivm::VDivOp;

  static bool shouldNormalizeMulRec(hivm::VMulOp op) {
    return op.hasPureTensorSemantics();
  }
};

/// Normalizes `div(1, x)` to `rec(x)`.
struct HIVMNormalizeDivVSToRecTraits
    : public hivm::NormalizeTraitsBase {
public:
  static bool shouldNormalizeDiv(hivm::VDivOp op) {
    return op.hasPureTensorSemantics();
  }
};

struct HIVMNormalizeVPowiToPowfTraits : public hivm::NormalizeTraitsBase {
  static bool shouldNormalizeVPowi(hivm::VPowOp op) {
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
           op.getTranspose().empty();
  }
};

struct HIVMNormalizeMulExtTraits : public hivm::NormalizeTraitsBase {
  static bool shouldNormalizeMulExt(hivm::VMulExtOp op) {
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
           op.getTranspose().empty();
  }

  static Value getMulExtLhs(hivm::VMulExtOp op) { return op.getDpsInputs()[0]; }

  static Value getMulExtRhs(hivm::VMulExtOp op) { return op.getDpsInputs()[1]; }

  static Type getMulExtExtendedType(PatternRewriter &rewriter, Type inputType) {
    return inputType.isInteger(32) ? rewriter.getI64Type() : Type();
  }
};

/// normalize vsub(s, v) to vadd(vmuls(v, -1), s).
/// eg.
///   %res = vsub %brc_s, %x : f32, tensor<16xf32>
///   where %brc_s = vbrc %s : f32
///   is normalized to
///   %mul = vmul %x, -1 : tensor<16xf32>, f32
///   %res = vadd %mul, %s : tensor<16xf32>, f32
struct HIVMNormalizeSubVSToVMulAndVAddTraits
    : public hivm::NormalizeTraitsBase {
  // HIVM final add must keep the vector operand in the vector-only slot.
  static constexpr bool kPlaceScalarFirstInFinalAdd = false;

  static hivm::VBrcOp getScalarBroadcast(hivm::VSubOp op) {
    return op.getSrc()[0].template getDefiningOp<hivm::VBrcOp>();
  }

  static bool shouldNormalizeSubScalarVector(hivm::VSubOp op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty())
      return false;

    auto brcOp = getScalarBroadcast(op);
    if (!brcOp || !brcOp.hasPureTensorSemantics() ||
        !brcOp.getBroadcastDims().empty())
      return false;

    if (!brcOp.getSrc().getType().isIntOrFloat() ||
        !isa<ShapedType>(op.getSrc()[1].getType()))
      return false;
    return true;
  }

  static Value getSubScalar(hivm::VSubOp op) {
    return getScalarBroadcast(op).getSrc();
  }

  static Value getSubVector(hivm::VSubOp op) { return op.getSrc()[1]; }
};

/// Normalize divsi and divui for membase:
/// supports i8/i16/i32/i64 type
/// c = a / b
/// is normalized to
/// fa = castTo<f32>(a)
/// fb = castTo<f32>(b)
/// fc = fa / fb
/// c = castTo<integer>(fc, mode = TRUNC)
struct HIVMNormalizeDivSIandDivUIOpTraits : public hivm::NormalizeTraitsBase {
  static bool shouldNormalizeDivSIandDivUIOp(hivm::VDivOp op) {
    if (!op.hasPureTensorSemantics() || !op.getBroadcast().empty() ||
        !op.getTranspose().empty())
      return false;

    auto resultType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
    if (!resultType)
      return false;

    Type elemType = resultType.getElementType();
    if (!elemType.isInteger())
      return false;

    if (hivm::archisMembased)
      return true;

    // Regbase only supports the i8 widening path here.
    return hivm::archIsRegbased && elemType.isInteger(8);
  }

  /// Materialize a tensor splat of i8 minimum in i16 compute type. The signed
  /// i8 regbase path uses this to repair the -128 / -1 overflow case after the
  /// i16 div has produced +128.
  static Value createI8MinTensor(PatternRewriter &rewriter, Location loc,
                                 Value likeTensor) {
    Value i8MinScalar = utils::createConstantOp<int>(rewriter, loc,
                                                     rewriter.getI16Type(),
                                                     -128);
    Value i8MinInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, likeTensor, rewriter.getI16Type());
    return rewriter
        .create<hivm::VBrcOp>(loc, TypeRange{i8MinInit.getType()}, i8MinScalar,
                              i8MinInit)
        .getResult()[0];
  }


  static Value createAscend950SignedI8ResultCast(PatternRewriter &rewriter,
                                                 Value src) {
    Value f16Value = createCastOp(rewriter, src.getLoc(), src,
                                  rewriter.getF16Type(), CastRoundKind::Trunc);
    return createCastOp(rewriter, f16Value.getLoc(), f16Value,
                        rewriter.getI8Type(), CastRoundKind::Trunc);
  }

  /// Cast the membase f32 div result back to the original integer element type.
  static Value createMembaseResultCast(PatternRewriter &rewriter,
                                       Value divResult, Type srcElemType) {
    if (srcElemType.isInteger(8) || srcElemType.isInteger(16)) {
      Value i32Value = createCastOp(rewriter, divResult.getLoc(), divResult,
                                    rewriter.getI32Type(),
                                    CastRoundKind::Trunc);
      return createCastOp(rewriter, i32Value.getLoc(), i32Value, srcElemType,
                          CastRoundKind::TruncWithOverflow);
    }
    return createCastOp(rewriter, divResult.getLoc(), divResult, srcElemType,
                        CastRoundKind::Trunc);
  }

  static Value createMembaseComputeValue(PatternRewriter &rewriter, Value src) {
    if (isa<ShapedType>(src.getType()))
      return createCastOp(rewriter, src.getLoc(), src, rewriter.getF32Type(),
                          CastRoundKind::Default);
    return mlir::convertScalarToDtype(rewriter, src.getLoc(), src,
                                      rewriter.getF32Type(),
                                      /*isUnsignedCast=*/false);
  }

  static LogicalResult rewriteDivSIandDivUIOp(hivm::VDivOp op,
                                              PatternRewriter &rewriter) {
    if (!hivm::archisMembased)
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto resultType = cast<ShapedType>(op.getDpsInits()[0].getType());
    Type srcElemType = resultType.getElementType();
    auto inputs = op.getDpsInputs();
    Value lhs = createMembaseComputeValue(rewriter, inputs[0]);
    Value rhs = createMembaseComputeValue(rewriter, inputs[1]);
    Value divInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, op.getDpsInits()[0], rewriter.getF32Type());
    Value divResult =
        rewriter
            .create<hivm::VDivOp>(op.getLoc(), TypeRange{divInit.getType()},
                                  ValueRange{lhs, rhs}, ValueRange{divInit},
                                  Value(), /*isSigned=*/true)
            .getResult()[0];
    rewriter.replaceOp(
        op, createMembaseResultCast(rewriter, divResult, srcElemType));
    return success();
  }

  static LogicalResult rewriteI8IntegerDivThroughI16(hivm::VDivOp op,
                                                     PatternRewriter &rewriter) {
    auto resultType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
    if (!resultType || !resultType.getElementType().isInteger(8))
      return failure();

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto inputs = op.getDpsInputs();

    CastSignKind signKind =
        op.getIsSigned() ? CastSignKind::Signed : CastSignKind::Unsigned;

    Value lhsI16 = createCastOp(rewriter, inputs[0].getLoc(), inputs[0],
                                rewriter.getI16Type(), CastRoundKind::RInt,
                                Value(), signKind);
    Value rhsI16 = createCastOp(rewriter, inputs[1].getLoc(), inputs[1],
                                rewriter.getI16Type(), CastRoundKind::RInt,
                                Value(), signKind);

    Value divInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, op.getDpsInits()[0], rewriter.getI16Type());
    Value divI16 =
        rewriter
            .create<hivm::VDivOp>(loc, TypeRange{divInit.getType()},
                                  ValueRange{lhsI16, rhsI16},
                                  ValueRange{divInit}, Value(),
                                  op.getIsSigned())
            .getResult()[0];

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp || !hacc::utils::isAscend950(moduleOp) || !op.getIsSigned()) {
      rewriter.replaceOp(op, createCastOp(rewriter, divI16.getLoc(), divI16,
                                          rewriter.getI8Type(),
                                          CastRoundKind::Trunc));
      return success();
    }

    Value i8ExceedScalar = utils::createConstantOp<int>(
        rewriter, loc, rewriter.getI16Type(), 128);
    Value exceedMask = createCmpOp(rewriter, loc, divI16, i8ExceedScalar,
                                   CompareKind::EQ);
    Value i8MinTensor = createI8MinTensor(rewriter, loc, divI16);
    Value finalI16 =
        rewriter
            .create<hivm::VSelOp>(loc, TypeRange{divI16.getType()},
                                  ValueRange{exceedMask, i8MinTensor, divI16},
                                  ValueRange{divI16}, Value(),
                                  ArrayRef<int64_t>{}, ArrayRef<int64_t>{})
            .getResult()[0];
    rewriter.replaceOp(op,
                       createAscend950SignedI8ResultCast(rewriter, finalI16));
    return success();
  }
};

using NormalizeRSqrtOp =
    NormalizeRSqrtOpTemplate<hivm::VRsqrtOp, HIVMNormalizeRSqrtTraits>;
using NormalizeMulRecOp =
    NormalizeMulRecOpTemplate<hivm::VMulOp, HIVMNormalizeMulRecTraits>;
using NormalizeDivVSToRec =
    NormalizeDivVSToRecTemplate<hivm::VDivOp, HIVMNormalizeDivVSToRecTraits>;
using NormalizeVPowiToPowf =
    NormalizeVPowiToPowfTemplate<hivm::VPowOp, HIVMNormalizeVPowiToPowfTraits>;
using NormalizeMulExtOp =
    NormalizeMulExtOpTemplate<hivm::VMulExtOp, HIVMNormalizeMulExtTraits>;
using NormalizeSubVSToVMulAndVAdd =
    NormalizeSubVSToVMulAndVAddTemplate<hivm::VSubOp,
                                        HIVMNormalizeSubVSToVMulAndVAddTraits>;
using NormalizeDivSIandDivUIOp =
    NormalizeDivSIandDivUIOpTemplate<hivm::VDivOp,
                                     HIVMNormalizeDivSIandDivUIOpTraits>;

} // namespace mlir

void mlir::hivm::populateNormalizeMulRecPatterns(
    RewritePatternSet &patterns) {
  patterns.add<NormalizeMulRecOp>(patterns.getContext());
}

void mlir::hivm::populateNormalizeArithmeticPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeDivVSToRec>(ctx);
  patterns.add<NormalizeVPowiToPowf>(ctx);
  patterns.add<NormalizeSubVSToVMulAndVAdd>(ctx);
  patterns.add<NormalizeRSqrtOp>(ctx);
}

void mlir::hivm::populateNormalizePreFinalArithmeticPatterns(
    RewritePatternSet &patterns) {
  patterns.add<NormalizeMulExtOp>(patterns.getContext());
}

void mlir::hivm::populateNormalizeFinalArithmeticPatterns(
    RewritePatternSet &patterns) {
  patterns.add<NormalizeDivSIandDivUIOp>(patterns.getContext());
}

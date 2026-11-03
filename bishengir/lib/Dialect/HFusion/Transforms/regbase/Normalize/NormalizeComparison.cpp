//===- NormalizeComparison.cpp ----------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeUtils.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeComparisonTemplate.h"

namespace mlir::hfusion {

/// normalize the specific cmp pattern to cast op
/// eg.
///  scalar = const 0
///  src0 = fill(scalar, dst) -> i8
///  y = hfusion.cmpi x, src0 {vne} ->  i1
/// is normalized to
///  y = hfusion.cast x -> i1
struct HFusionCmpToCastTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeCmpToCast(CompareOp op) {
    return op.hasPureTensorSemantics() && op.getCompareFn() == CompareFn::vne;
  }

  /// Returns true only for the original HFusion zero-tensor producer shape
  /// accepted by this pattern: `linalg.fill(0) -> tensor<...>`.
  static bool isZeroTensorProducer(Value value) {
    auto fillOp = value.getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return false;
    auto cstOp =
        fillOp.getInputs()[0].getDefiningOp<arith::ConstantIntOp>();
    return cstOp && cstOp.value() == 0;
  }

  static bool isSupportedCastToBool(Type) { return true; }
};

using NormalizeCmpToCastOpRegBase =
    mlir::NormalizeCmpToCastOpTemplate<CompareOp, HFusionCmpToCastTraits>;

struct HFusionCmpVneTraits : public NormalizeTraitsBase {
  static bool shouldNormalize(CompareOp op) {
    return op.hasPureTensorSemantics() && op.getCompareFn() == CompareFn::vne;
  }
};

using NormalizeCmpVneOp = mlir::NormalizeCmpVneOpTemplate<CompareOp, HFusionCmpVneTraits>;

struct HFusionCmpTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeCmp(CompareOp op) {
    // hfusion::CompareOp is used in cast op when casting to bool(int1).
    // Overflowmode annotation mark is useless in this case,
    // and would cause redundant vf in PreVectorizationFusion Pass.
    // So here we pair and delete overflow mark on CompareOp.
    return op.hasPureTensorSemantics();
  }
};

using NormalizeCmpOp = mlir::NormalizeCmpOpTemplate<CompareOp, HFusionCmpTraits>;

struct HFusionIsInfNanNormalizeTraitsBase : public NormalizeTraitsBase {
  static bool isSupportedElementType(Type elemType) {
    return elemType.isF16() || elemType.isBF16() || elemType.isF32();
  }

  // HFusion isinf/isnan are not DestinationStyleOpInterface ops today.
  // They expose one input / one result through getInput()/getOutput(), so the
  // shared template cannot use the same DPS accessor shape as HIVM here.
  template <typename OpTy>
  static Value getInput(OpTy op) {
    return op.getInput();
  }

  template <typename OpTy>
  static Value getOutput(OpTy op) {
    return op.getOutput();
  }

  /// Converts the intermediate integer mask in `{0, 1}` to the final bool
  /// result tensor by reusing the op's destination and issuing an explicit
  /// `hfusion.cast(... -> i1)`.
  static Value lowerIntMaskToBoolResult(PatternRewriter &rewriter, Location loc,
                                        Value input, Value output) {
    return NormalizeTraitsBase::createCastOp(
        rewriter, loc, input, getElementTypeOrSelf(output.getType()),
        CastRoundKind::RInt, output);
  }
};

/// Normalizes `hfusion.is_inf` to existing HFusion primitive ops:
///   bitcast -> vand(sign_mask) -> add(-inf_bits) -> bitcast -> abs
///   -> bitcast -> min_signed(1) -> mul(-1) -> add(1) -> cast(i1)
struct HFusionIsInfTraits : public HFusionIsInfNanNormalizeTraitsBase {
  static bool shouldNormalize(hfusion::IsInfOp) { return true; }
};

/// Normalizes `hfusion.is_nan` to existing HFusion primitive ops:
///   bitcast -> vand(sign_mask) -> add(-inf_bits) -> min_signed(1)
///   -> max_signed(0) -> cast(i1)
struct HFusionIsNanTraits : public HFusionIsInfNanNormalizeTraitsBase {
  static bool shouldNormalize(hfusion::IsNanOp) { return true; }
};

using NormalizeIsInfOpRegBase =
    mlir::NormalizeIsInfOpTemplate<hfusion::IsInfOp, HFusionIsInfTraits>;
using NormalizeIsNanOpRegBase =
    mlir::NormalizeIsNanOpTemplate<hfusion::IsNanOp, HFusionIsNanTraits>;

/// normalize i8/i32 CompareOp
///   i8 -> f16
///   i32 -> i64 (except vne and veq)
/// e.g.
///   hfusion.compare ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
/// is normalized to
///   %cast1 = hfusion.cast %src1 : tensor<6x6xi32> to tensor<6x6xi64>
///   %cast2 = hfusion.cast %src2 : tensor<6x6xi32> to tensor<6x6xi64>
///   hfusion.compare ins(%cast1, %cast2 : tensor<6x6xi64>, tensor<6x6xi64>)
struct HFusionI8I32CmpTraits : public NormalizeTraitsBase {
  static bool shouldNormalize(CompareOp op) {
    return op.hasPureTensorSemantics();
  }

  static bool isEqOrNeCompare(CompareOp op) {
    return op.getCompareFn() == CompareFn::veq ||
           op.getCompareFn() == CompareFn::vne;
  }
};

using NormalizeI8I32CmpOpRegBase =
    mlir::NormalizeI8I32CmpOpTemplate<CompareOp, HFusionI8I32CmpTraits>;

/// Normalizes `vxor(x, y)` to:
///   tmp_or = vor(x, y)
///   tmp_and = vand(x, y)
///   tmp_not = vnot(tmp_and)
///   result = vand(tmp_not, tmp_or)
struct HFusionXorTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeXor(hfusion::ElemwiseBinaryOp op) {
    return op.hasPureTensorSemantics() &&
           op.getFun() == hfusion::BinaryFn::vxor;
  }
};

using NormalizeXorOpRegBase =
    mlir::NormalizeXorOpTemplate<hfusion::ElemwiseBinaryOp, HFusionXorTraits>;

/// normalize shift i8 as bellow
/// eg.
///   %res = shift %src : i8
/// is normalized to
///   %tmp0 = cast %src i8 to i16
///   %tmp1 = shift %tmp0 : i16
///   %res = cast %tmp1 i16 to i8
struct HFusionShiftI8ToI16Traits : public NormalizeTraitsBase {
  static bool shouldNormalizeShift(hfusion::ElemwiseBinaryOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    auto fun = op.getFun();
    return fun == hfusion::BinaryFn::shli || fun == hfusion::BinaryFn::shrsi ||
           fun == hfusion::BinaryFn::shrui;
  }

  static CastSignKind getShiftCastSignKind(hfusion::ElemwiseBinaryOp op) {
    auto fun = op.getFun();
    return (fun == hfusion::BinaryFn::shrui || fun == hfusion::BinaryFn::shli)
               ? CastSignKind::Unsigned
               : CastSignKind::Signed;
  }
};

using NormalizeShiftI8ToI16RegBase =
    mlir::NormalizeShiftI8ToI16OpTemplate<hfusion::ElemwiseBinaryOp,
                                          HFusionShiftI8ToI16Traits>;

void populateNormalizeI8I32CmpPatterns(RewritePatternSet &patterns) {
  if (!archIsRegbased)
    patterns.add<NormalizeI8I32CmpOpRegBase>(patterns.getContext());
}

void populateNormalizeCmpToCastPatterns(RewritePatternSet &patterns) {
  if (!archIsRegbased)
    patterns.add<NormalizeCmpToCastOpRegBase>(patterns.getContext());
}

void populateNormalizeComparisonCleanupPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeCmpOp>(ctx);
  patterns.add<NormalizeIsInfOpRegBase>(ctx);
  patterns.add<NormalizeIsNanOpRegBase>(ctx);
}

void populateNormalizeBitwiseComparisonPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (!archIsRegbased) {
    patterns.add<NormalizeXorOpRegBase>(ctx);
    patterns.add<NormalizeShiftI8ToI16RegBase>(ctx);
  }
}

void populateNormalizeShiftI8ToI16(RewritePatternSet &patterns) {
  if (archIsRegbased)
    patterns.add<NormalizeShiftI8ToI16RegBase>(patterns.getContext());
}

void populateNormalizeCmpVnePatterns(RewritePatternSet &patterns) {
  if (!archIsRegbased)
    patterns.add<NormalizeCmpVneOp>(patterns.getContext());
}
} // namespace mlir::hfusion

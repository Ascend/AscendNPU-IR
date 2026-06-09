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

#include "bishengir/Transforms/Normalize/NormalizeComparisonTemplate.h"

#include "mlir/IR/PatternMatch.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"

namespace mlir::hivm {

struct HIVMComparisonNormalizeTraitsBase : public NormalizeTraitsBase {
  template <typename OpTy>
  static bool hasPureTensorNoTransformAttrs(OpTy op) {
    return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
           op.getTranspose().empty();
  }
};

/// Normalizes `hivm.hir.vcmp(..., compare_mode = ne)` to:
///   tmp = vcmp(lhs, rhs, eq)
///   result = vnot(tmp)
struct HIVMCmpVneTraits : public HIVMComparisonNormalizeTraitsBase {
  static bool shouldNormalize(VCmpOp op) {
    if (!hasPureTensorNoTransformAttrs(op)) {
      op->emitWarning("NormalizeCmpVneOp: skipping op with non-empty broadcast/transpose attributes");
      return false;
    }
    return op.getCompareMode() == CompareMode::NE;
  }
};

using NormalizeCmpVneOp = mlir::NormalizeCmpVneOpTemplate<VCmpOp, HIVMCmpVneTraits>;

/// Normalizes `hivm.hir.vcmp` on `i8` / `i32` inputs to a wider compare type:
///   i8  -> f16
///   i32 -> i64, except `eq` / `ne`
struct HIVMI8I32CmpTraits : public HIVMComparisonNormalizeTraitsBase {
  static bool shouldNormalize(VCmpOp op) {
    return op.getIsSigned() && hasPureTensorNoTransformAttrs(op);
  }

  static bool isEqOrNeCompare(VCmpOp op) {
    return op.getCompareMode() == CompareMode::EQ ||
           op.getCompareMode() == CompareMode::NE;
  }
};

using NormalizeI8I32CmpOp =
    mlir::NormalizeI8I32CmpOpTemplate<VCmpOp, HIVMI8I32CmpTraits>;

/// Normalizes `hivm.hir.vxor(x, y)` to:
///   tmp_or = vor(x, y)
///   tmp_and = vand(x, y)
///   tmp_not = vnot(tmp_and)
///   result = vand(tmp_not, tmp_or)
struct HIVMXorTraits : public HIVMComparisonNormalizeTraitsBase {
  static bool shouldNormalizeXor(VXorOp op) {
    return hasPureTensorNoTransformAttrs(op);
  }
};

using NormalizeXorOp = mlir::NormalizeXorOpTemplate<VXorOp, HIVMXorTraits>;

/// Normalizes `hivm.hir.vshr` on `tensor<...xi8>` to:
///   vcast(i8 -> i16) -> vshr(i16) -> vcast(i16 -> i8)
struct HIVMShiftRightI8ToI16Traits
    : public HIVMComparisonNormalizeTraitsBase {
  static bool shouldNormalizeShift(VShROp op) {
    return hasPureTensorNoTransformAttrs(op);
  }

  static CastSignKind getShiftCastSignKind(VShROp) {
    return CastSignKind::Signed;
  }
};

using NormalizeShiftRightI8ToI16 =
    mlir::NormalizeShiftI8ToI16OpTemplate<VShROp,
                                          HIVMShiftRightI8ToI16Traits>;

struct HIVMIsInfNanNormalizeTraitsBase : public HIVMComparisonNormalizeTraitsBase {
  /// HIVM `visinf`/`visnan` normalize rewrites through `vabs`, and `vabs`
  /// only supports F16/F32 here. BF16 therefore cannot be normalized by this
  /// pattern and stays outside the supported op contract.
  static bool isSupportedElementType(Type elemType) {
    return elemType.isF16() || elemType.isF32();
  }

  template <typename OpTy>
  static Value getInput(OpTy op) {
    return op.getDpsInputs()[0];
  }

  template <typename OpTy>
  static Value getOutput(OpTy op) {
    return op.getDpsInits()[0];
  }

  /// Converts the intermediate integer mask in `{0, 1}` to the final bool
  /// output tensor.
  static Value lowerIntMaskToBoolResult(PatternRewriter &rewriter, Location loc,
                                        Value input, Value output) {
    Type intElemType = getElementTypeOrSelf(input.getType());
    Type elemType = intElemType.getIntOrFloatBitWidth() == 32
                        ? static_cast<Type>(rewriter.getF32Type())
                        : static_cast<Type>(rewriter.getF16Type());
    Value castValue = NormalizeTraitsBase::createCastOp(
        rewriter, loc, input, elemType, CastRoundKind::RInt);
    Value zeroValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, 0.0));
    Value cmpValue = NormalizeTraitsBase::createCmpOp(rewriter, loc, castValue,
                                                      zeroValue, CompareKind::EQ);
    return NormalizeTraitsBase::createUnaryOp(rewriter, loc, cmpValue, output,
                                              UnaryKind::Not);
  }

  template <typename OpTy>
  static bool shouldNormalize(OpTy op) {
    return hasPureTensorNoTransformAttrs(op);
  }
};

struct HIVMIsInfTraits : public HIVMIsInfNanNormalizeTraitsBase {
  static bool shouldNormalize(VIsInfOp op) {
    return HIVMIsInfNanNormalizeTraitsBase::shouldNormalize(op);
  }
};

struct HIVMIsNanTraits : public HIVMIsInfNanNormalizeTraitsBase {
  static bool shouldNormalize(VIsNanOp op) {
    return HIVMIsInfNanNormalizeTraitsBase::shouldNormalize(op);
  }
};

/// Normalizes `hivm.hir.visinf` to existing HIVM primitive ops:
///   bitcast -> vand(sign_mask) -> vadd(-inf_bits) -> bitcast -> vabs
///   -> bitcast -> vmin(1) -> vmul(-1) -> vadd(1) -> vcast -> vcmp(eq, 0)
///   -> vnot
using NormalizeIsInfOp = mlir::NormalizeIsInfOpTemplate<VIsInfOp, HIVMIsInfTraits>;

/// Normalizes `hivm.hir.visnan` to existing HIVM primitive ops:
///   bitcast -> vand(sign_mask) -> vadd(-inf_bits) -> vmin(1) -> vmax(0)
///   -> vcast -> vcmp(eq, 0) -> vnot
using NormalizeIsNanOp = mlir::NormalizeIsNanOpTemplate<VIsNanOp, HIVMIsNanTraits>;

void populateNormalizeI8I32CmpPatterns(RewritePatternSet &patterns) {
  if (!NormalizeTraitsBase::archIsRegbased())
    patterns.add<NormalizeI8I32CmpOp>(patterns.getContext());
}

void populateNormalizeComparisonCleanupPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeIsInfOp>(ctx);
  patterns.add<NormalizeIsNanOp>(ctx);
}

void populateNormalizeBitwiseComparisonPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (!NormalizeTraitsBase::archIsRegbased()) {
    patterns.add<NormalizeXorOp>(ctx);
    patterns.add<NormalizeShiftRightI8ToI16>(ctx);
  }
}

void populateNormalizeShiftI8ToI16(RewritePatternSet &patterns) {
  if (NormalizeTraitsBase::archIsRegbased())
    patterns.add<NormalizeShiftRightI8ToI16>(patterns.getContext());
}

void populateNormalizeCmpVnePatterns(RewritePatternSet &patterns) {
  if (!NormalizeTraitsBase::archIsRegbased())
    patterns.add<NormalizeCmpVneOp>(patterns.getContext());
}

} // namespace mlir::hivm

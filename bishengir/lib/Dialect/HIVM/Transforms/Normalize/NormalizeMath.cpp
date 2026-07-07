//===- NormalizeMath.cpp ----------------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/NormalizeMathTemplate.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::hivm {
namespace {
template <typename OpTy>
bool shouldNormalizeNonBroadcastOp(OpTy op) {
  return op.hasPureTensorSemantics() && op.getBroadcast().empty() &&
         op.getTranspose().empty();
}

/// Normalizes `hivm.hir.vexp2` to `vexp(ln(2) * x)`.
struct HIVMNormalizeExp2Traits : public NormalizeTraitsBase {
  static bool shouldNormalizeExp2(VExp2Op op) {
    return shouldNormalizeNonBroadcastOp(op);
  }
};

/// Normalizes `hivm.hir.verf` with the shared erf polynomial template.
struct HIVMNormalizeErfTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeErf(VErfOp op) {
    return shouldNormalizeNonBroadcastOp(op);
  }
};

/// normalize vlogb(x) to vln(x) / vln(b) when log base b is not e
/// eg.
/// y = hivm.hir.vlog2 x
///  is normalized to
///  y = hivm.hir.vln x / hivm.hir.vln 2
template <typename OpType, int Base>
struct HIVMNormalizeLogLikeTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeLogLike(OpType op) {
    return shouldNormalizeNonBroadcastOp(op);
  }

  static float getLogBase(OpType) { return static_cast<float>(Base); }

  static Value castBackLogLikeF16Result(PatternRewriter &rewriter, Location loc,
                                        Value result, Value dst) {
    return createCastOp(rewriter, loc, result,
                        getElementTypeOrSelf(dst.getType()),
                        CastRoundKind::Round);
  }
};

/// normalize vlog1p(x) to vln(x + 1)
/// eg.
/// y = hivm.hir.vlog1p x
///  is normalized to
///  y = hivm.hir.vln (x + 1)
struct HIVMNormalizeLog1pTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeLog1p(hivm::VLog1pOp op) {
    return shouldNormalizeNonBroadcastOp(op);
  }
};

using NormalizeExp2Op =
    mlir::NormalizeExp2OpTemplate<VExp2Op, HIVMNormalizeExp2Traits>;
using NormalizeErfOp =
    mlir::NormalizeErfOpTemplate<VErfOp, HIVMNormalizeErfTraits>;

/// normalize vexpm1(x) to vexp(x) - 1
/// eg.
/// y = hivm.hir.vexpm1 x
///  is normalized to
///  y = hivm.hir.vexp(x) - 1
struct HIVMNormalizeExpM1Traits : public NormalizeTraitsBase {
  static bool shouldNormalizeExpM1(hivm::VExpM1Op op) {
    return shouldNormalizeNonBroadcastOp(op);
  }
};

using NormalizeVExpM1Op =
    mlir::NormalizeExpM1OpTemplate<hivm::VExpM1Op,
                                   HIVMNormalizeExpM1Traits>;

/// normalize vilogb(x), which is exponent of frexp(x), to floor(log2(abs(x)))
struct HIVMNormalizeIlogbTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeIlogb(hivm::VIlogbOp op) {
    return shouldNormalizeNonBroadcastOp(op);
  }

  static Value createIlogbResult(PatternRewriter &rewriter, Location loc,
                                 Value log2) {
    return createCastOp(rewriter, loc, log2,
                        getElementTypeOrSelf(log2.getType()),
                        CastRoundKind::Floor);
  }
};

/// normalize vldexp(x, y) to vmul(x, y)
/// eg.
///  y = hivm.hir.vldexp(x, y)
///   is normalized to
///   y = hivm.hir.vmul(x, y)
struct HIVMNormalizeLdexpTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeLdexp(hivm::VLdexpOp op) {
    return shouldNormalizeNonBroadcastOp(op);
  }
};

/// normalize vpow(x, y) to constant-exponent shortcuts or exp + log with
/// boundary-case selects
/// eg.
///  y = hivm.hir.vpow(x, y)
///   is normalized to (constant exponent)
///   y = sqrt(x)  (when y = 0.5), or
///   y = x * x    (when y = 2)
///  or (dynamic exponent)
///   y = select(boundary cases, boundary values,
///              exp(exponent * ln(x)))
struct HIVMNormalizePowfTraits : public NormalizeTraitsBase {
  static bool shouldNormalizePowf(hivm::VPowOp op) {
    return shouldNormalizeNonBroadcastOp(op);
  }

  /// Recovers and returns a scalar `arith::ConstantOp` for the exponent from
  /// the wrapper forms that are common on the HIVM path.
  /// Example:
  ///   1. `hivm.hir.vbrc(0.5)` feeding `vpow`
  ///   2. a shaped `arith.constant dense<0.5>`
  static arith::ConstantOp getPowfExponentScalarConstant(
      Value exponent, PatternRewriter &rewriter) {
    if (auto brcOp = exponent.getDefiningOp<hivm::VBrcOp>()) {
      return dyn_cast_or_null<arith::ConstantOp>(brcOp.getSrc().getDefiningOp());
    }

    if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(
            exponent.getDefiningOp())) {
      if (!isa<ShapedType>(constOp.getType()))
        return constOp;
      std::optional<Value> scalar =
          utils::extractScalarValue(rewriter, exponent.getLoc(), exponent);
      if (scalar.has_value())
        return dyn_cast_or_null<arith::ConstantOp>(
            scalar->getDefiningOp());
    }

    return nullptr;
  }

  /// Computes the parity coefficient `(-2 * mod) + 1`.
  /// On Ascend950 with f32, emits `math.fma` for the fused mul+add.
  static Value computeParityCoefficient(PatternRewriter &rewriter,
                                        Location loc, Value mod,
                                        Value negativeTwo,
                                        Value positiveOne) {
    Type elemType = getElementTypeOrSelf(mod.getType());
    if (archisAscend950 && elemType.isF32()) {
      // math.fma requires all operands to have the same type, so broadcast the
      // scalar constants to match the tensor shape of `mod`.
      auto likeMod = utils::createEmptyOp(rewriter, loc, mod);
      Value negTwoTensor =
          createFillOp(rewriter, loc, negativeTwo, likeMod);
      Value posOneTensor =
          createFillOp(rewriter, loc, positiveOne, likeMod);
      return rewriter.create<math::FmaOp>(loc, mod, negTwoTensor, posOneTensor);
    }
    Value mul = createBinaryOp(rewriter, loc, mod, negativeTwo,
                               utils::createEmptyOp(rewriter, loc, mod),
                               BinaryKind::Mul);
    return createBinaryOp(rewriter, loc, mul, positiveOne,
                          utils::createEmptyOp(rewriter, loc, mod),
                          BinaryKind::Add);
  }
};

using NormalizeVIlogbOp =
    mlir::NormalizeIlogbOpTemplate<hivm::VIlogbOp,
                                   HIVMNormalizeIlogbTraits>;

using NormalizeVLog2Op =
    mlir::NormalizeLogLikeOpTemplate<
        hivm::VLog2Op, HIVMNormalizeLogLikeTraits<hivm::VLog2Op, 2>>;
using NormalizeVLog10Op =
    mlir::NormalizeLogLikeOpTemplate<
        hivm::VLog10Op, HIVMNormalizeLogLikeTraits<hivm::VLog10Op, 10>>;
using NormalizeVLog1pOp =
    mlir::NormalizeLog1pOpTemplate<hivm::VLog1pOp, HIVMNormalizeLog1pTraits>;
using NormalizeVLdexpOp =
    mlir::NormalizeLdexpOpTemplate<hivm::VLdexpOp, HIVMNormalizeLdexpTraits>;
using NormalizeVPowfOp =
    mlir::NormalizePowfOpTemplate<hivm::VPowOp, HIVMNormalizePowfTraits>;

template <typename OpTy, CastSignKind SignKind, BinaryKind ModKind,
          bool SupportsFloat>
struct HIVMModTraits : public NormalizeTraitsBase {
  static bool isSupportedType(Type elemType) {
    if (elemType.isInteger(1))
      return true;
    if (elemType.isInteger(8))
      return true;
    if (elemType.isInteger())
      return false;
    if constexpr (!SupportsFloat)
      return false;
    return elemType.isF16() || elemType.isF32();
  }

  static bool shouldNormalize(OpTy op) {
    return shouldNormalizeNonBroadcastOp(op);
  }

  static CastSignKind getCastSignKind(OpTy) { return SignKind; }

  static BinaryKind getModKind(OpTy) { return ModKind; }

  static Value createDivOpForMod(PatternRewriter &rewriter, Location loc,
                                 Value x, Value y, Type elemType) {
    auto divDst = utils::createEmptyOp(rewriter, loc, x);
    auto divOp = rewriter
                     .create<hivm::VDivOp>(loc, mlir::TypeRange{divDst.getType()},
                                           mlir::ValueRange{x, y},
                                           mlir::ValueRange{divDst})
                     .getResults()[0];
    auto roundModeAttr =
        rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::TRUNC);
    Value truncDst =
        utils::createEmptyOpWithTargetElemType(rewriter, loc, divOp, elemType);
    return rewriter
        .create<hivm::VCastOp>(loc, TypeRange(truncDst.getType()), divOp,
                               truncDst, roundModeAttr,
                               rewriter.getAttr<hivm::TypeFnAttr>(
                                   hivm::TypeFn::cast_signed))
        .getResults()[0];
  }
};

using NormalizeVModOp =
    mlir::NormalizeModOpTemplate<VModOp,
                                 HIVMModTraits<VModOp, CastSignKind::Signed,
                                               BinaryKind::Mod,
                                               /*SupportsFloat=*/true>>;
using NormalizeVModUIOp =
    mlir::NormalizeModOpTemplate<VModUIOp,
                                 HIVMModTraits<VModUIOp,
                                               CastSignKind::Unsigned,
                                               BinaryKind::ModUnsigned,
                                               /*SupportsFloat=*/false>>;

struct HIVMNormalizeCeilandFloorTraits : public NormalizeTraitsBase {
  static bool shouldNormalizeCeilandFloor(VCastOp op) {
    if (!op.hasPureTensorSemantics())
      return false;
    auto roundMode = op.getRoundMode();
    if (roundMode != hivm::RoundMode::CEIL &&
        roundMode != hivm::RoundMode::FLOOR)
      return false;
    Type inType = getElementTypeOrSelf(op.getSingleSrc().getType());
    Type outType = getElementTypeOrSelf(op.getSingleDst().getType());
    if (!((inType.isF16() || inType.isBF16()) && inType == outType))
      return false;
    if (archisAscend950)
      return false;
    return true;
  }

  static CastRoundKind getCeilOrFloorRoundKind(VCastOp op) {
    return op.getRoundMode() == hivm::RoundMode::CEIL ? CastRoundKind::Ceil
                                                       : CastRoundKind::Floor;
  }

  static Value getCeilandFloorInput(VCastOp op) {
    return op.getSingleSrc();
  }

  static Value getCeilandFloorOutput(VCastOp op) {
    return op.getSingleDst();
  }
};

using NormalizeVCeilandFloorOp =
    mlir::NormalizeCeilandFloorOpTemplate<VCastOp,
                                          HIVMNormalizeCeilandFloorTraits>;

} // namespace

void populateNormalizePrimaryMathPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeVLog2Op, NormalizeVLog10Op, NormalizeVLog1pOp,
               NormalizeExp2Op, NormalizeVExpM1Op,
               NormalizeErfOp>(patterns.getContext());
}

void populateNormalizeLateMathPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeVIlogbOp, NormalizeVLdexpOp,
               NormalizeVPowfOp,
               NormalizeVCeilandFloorOp>(patterns.getContext());
}

void populateNormalizeModPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeVModOp, NormalizeVModUIOp>(patterns.getContext());
}

} // namespace mlir::hivm

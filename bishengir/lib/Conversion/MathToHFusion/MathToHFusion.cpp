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

#include <cstdlib>
#include <string>

#include "RegBase/SIMTTrigCommon.h"
#include "bishengir/Conversion/MathToHFusion/MathToHFusion.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "bishengir/Transforms/regbase/Normalize/Utils/TrigTemplateHelpers.h"

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
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

// A/B measurement switch: when BISHENGIR_SIN_NORMALIZE is set, f32 sin
// keeps the legacy HFusion Normalize lowering instead of the template-library
// intrinsic (sin_fp32_simt_<rank>d).
static bool useNormalizeSin() {
  return std::getenv("BISHENGIR_SIN_NORMALIZE") != nullptr;
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

/// Highest memref rank that has a template-library SIMT entry point
/// (`_mlir_ciface_sin_fp32_simt_<rank>d` in RegBase/Vector/Sin.cpp). Keep in
/// sync with SIN_SIMT_MAX_RANK in Template/lib/Template/include/RegBase/
/// SIMTTrigCommon.h.
static constexpr int64_t kMaxTemplateLibTrigRank = SIN_SIMT_MAX_RANK;

/// True if `value` (an f32 tensor) is only a widening view/cast of an f16
/// tensor, with no arithmetic in between that could leave the f16 range. For
/// such values |x| <= 65504 < SIN_COS_THRESHOLD, so the Payne-Hanek branch and
/// its 2/pi table are unreachable.
static bool isF16Derived(Value value) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (Operation *def = value.getDefiningOp()) {
    if (!visited.insert(def).second)
      return false;
    if (auto cast = dyn_cast<hfusion::CastOp>(def)) {
      Value src = cast->getOperand(0);
      if (getElementTypeOrSelf(src.getType()).isF16())
        return true;
      value = src;
      continue;
    }
    if (auto ext = dyn_cast<arith::ExtFOp>(def)) {
      Value src = ext.getIn();
      if (getElementTypeOrSelf(src.getType()).isF16())
        return true;
      value = src;
      continue;
    }
    if (auto generic = dyn_cast<linalg::GenericOp>(def)) {
      bool fromF16 = false;
      generic->getRegion(0).walk([&](arith::ExtFOp ext) {
        if (getElementTypeOrSelf(ext.getIn().getType()).isF16())
          fromF16 = true;
      });
      if (fromF16)
        return true;
      if (generic.getNumDpsInputs() == 1) {
        value = generic.getDpsInputOperand(0)->get();
        continue;
      }
      return false;
    }
    if (isa<tensor::CastOp, tensor::ExtractSliceOp, tensor::CollapseShapeOp,
            tensor::ExpandShapeOp>(def)) {
      if (def->getNumOperands() == 0)
        return false;
      value = def->getOperand(0);
      continue;
    }
    return false;
  }
  return false;
}

/// True if `value` is (through views) narrowed to f16 by a cast.
static bool isNarrowedToF16(Value value) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (true) {
    if (!value.hasOneUse())
      return false;
    Operation *user = *value.user_begin();
    if (!visited.insert(user).second)
      return false;
    if (auto cast = dyn_cast<hfusion::CastOp>(user))
      return getElementTypeOrSelf(cast->getResult(0).getType()).isF16();
    if (auto trunc = dyn_cast<arith::TruncFOp>(user))
      return getElementTypeOrSelf(trunc.getOut().getType()).isF16();
    if (isa<tensor::ExtractSliceOp, tensor::CastOp>(user)) {
      value = user->getResult(0);
      continue;
    }
    return false;
  }
}

/// True if the f32 sin is bracketed by f16 casts: the input is a widening of
/// f16 data and the result is narrowed back to f16. Then the low-precision
/// Normalize recipe (pi reduction + Taylor degree 9) is enough, and the
/// Payne-Hanek table can be skipped entirely.
static bool isF16Bracketed(Value input, Operation *sinOp) {
  if (!isF16Derived(input))
    return false;
  if (sinOp->getNumResults() == 0)
    return false;
  return isNarrowedToF16(sinOp->getResult(0));
}

/// Lower f32 `math.sin` on regbase to the template-library implementation via
/// a `hivm.custom` call whose symbol resolves to
/// `_mlir_ciface_sin_fp32_simt_<rank>d` (RegBase/Vector/Sin.cpp). This keeps
/// sin away from the HFusion Normalize expansion and lets the template-lib
/// SIMT algorithm be used.
///
/// Everything the template library cannot handle - f16/bf16 inputs and ranks
/// above kMaxTemplateLibTrigRank - falls back to the HFusion Normalize
/// expansion here, so this pattern is also the fallback for math.sin.
template <typename MathOp>
struct MathTrigToTemplateLib : public OpRewritePattern<MathOp> {
  using OpRewritePattern<MathOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    Value inner = op.getOperand();
    auto inType = cast<RankedTensorType>(inner.getType());

    if (inType.getElementType().isF32() &&
        inType.getRank() <= kMaxTemplateLibTrigRank) {
      // f16 in and f16 out (both through casts): use the low-precision
      // Normalize recipe inline (pi reduction + Taylor degree 9). The f16
      // accuracy budget makes it more than enough, and it avoids the
      // Payne-Hanek table entirely.
      if (std::is_same_v<MathOp, math::SinOp> && isF16Bracketed(inner, op)) {
        Location loc = op.getLoc();
        Value empty = utils::createEmptyOp(rewriter, loc, inner);
        Value roundedPiMultiple =
            buildRoundedPiMultiple<hfusion::NormalizeTraitsBase>(
                rewriter, loc, inner, rewriter.getF32Type());
        Value rangeReducedInput =
            buildRangeReducedTrigInput<hfusion::NormalizeTraitsBase>(
                rewriter, loc, inner, roundedPiMultiple, kPiApproximations,
                0.0);
        Value sinApproximation =
            buildTaylorApproximation<hfusion::NormalizeTraitsBase>(
                rewriter, loc, rangeReducedInput,
                getTaylorSeriesCoefficients(mlir::TaylerMode::SIN, 5));
        Value sinSign = buildSinParitySign<hfusion::NormalizeTraitsBase>(
            rewriter, loc, roundedPiMultiple);
        Value result = hfusion::NormalizeTraitsBase::createBinaryOp(
            rewriter, loc, sinApproximation, sinSign, empty, BinaryKind::Mul);
        rewriter.replaceOp(op, result);
        return success();
      }

      SmallVector<Value> dsts;
      if (failed(
              tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
        return failure();

      // Stage the same 2/pi table the HFusion Normalize path uses
      // (memref.global -> GM->UB copy -> tensor) and pass it to the SIMT
      // callee. The global name is shared so both paths reuse a single `@tbl`
      // (the lookup index is `expo + 8` in both).
      ModuleOp module = op->template getParentOfType<ModuleOp>();
      if (!module)
        return failure();
      FailureOr<Value> tableOr =
          emitTrigTableFromUI32ArrayAndLoadAsTensorI32<320>(
              rewriter, op, module, op.getLoc(), "tbl", kPayneHanekTable);
      if (failed(tableOr))
        return failure();
      Value tableArg = *tableOr;

      // The library call is emitted with the operand's own memref type, so the
      // entry point has to be selected by rank: the memref descriptor layout
      // (and with it the way the element count is derived) differs per rank.
      std::string base = std::is_same_v<MathOp, math::SinOp> ? "sin_fp32_simt_"
                                                             : "cos_fp32_simt_";
      std::string symbol = base + std::to_string(inType.getRank()) + "d";
      auto customOp = rewriter.create<hivm::CustomOp>(
          op.getLoc(), symbol, ValueRange{inner, tableArg}, ValueRange{dsts});
      customOp.setCoreType(hivm::TCoreType::VECTOR);
      // The callee is a `__simt_vf__` kernel launched from the AIV context:
      // mark it SIMT so the SIMT-VF passes (memory semantics, UB accounting,
      // legalization) and InferVFMode see the real execution model.
      customOp.setVFMode(hivm::VFMode::SIMT);
      customOp.setPipe(hivm::PIPE::PIPE_V);
      customOp.setInlineMode(hivm::InlineMode::AlwaysInline);
      customOp->setAttr("symbol", rewriter.getStringAttr(symbol));
      customOp->setAttr("bitcode", rewriter.getStringAttr(""));
      customOp->setAttr("arg_attrs", rewriter.getArrayAttr({}));
      rewriter.replaceOp(op, customOp.getResults());
      return success();
    }

    // f16/bf16 and ranks above the template-library limit fall back to the
    // HFusion Normalize expansion. For f16 this is the low-precision path
    // (cast to f32, pi reduction, Taylor degree 9), which is enough for the
    // f16 accuracy budget and avoids the Payne-Hanek table entirely.
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto unaryAttr = rewriter.getAttr<hfusion::UnaryFnAttr>(
        std::is_same_v<MathOp, math::SinOp> ? hfusion::UnaryFn::sin
                                            : hfusion::UnaryFn::cos);
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

static bool isSupportedFmaType(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  return elementType.isF16() || elementType.isBF16() || elementType.isF32();
}

struct MathFmaLowering : public OpRewritePattern<math::FmaOp> {
  MathFmaLowering(MLIRContext *context, bool enableFma)
      : OpRewritePattern<math::FmaOp>(context), enableFma(enableFma) {}

  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();

    if (enableFma && isSupportedFmaType(op.getType())) {
      SmallVector<Value> dsts;
      if (failed(
              tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
        return failure();

      auto *fmaOp =
          hfusion::createTernaryOp<hfusion::ElemwiseTernaryOp,
                                   hfusion::TernaryFn, hfusion::TernaryFnAttr>(
              rewriter, op.getLoc(), hfusion::TernaryFn::fma,
              ValueRange{op.getA(), op.getB(), op.getC()}, dsts);
      rewriter.replaceOp(op, fmaOp);
      return success();
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

private:
  bool enableFma;
};

/// @brief
// Two kinds of conversions are applied:
// 1. math ops to linalg unary/binary ops
// 2. math ops to hfusion unary/binary ops
//
// SUFFIX SPLIT: 7 HFusion patterns (Atan2/Atanh/Asin/Acos/Sinh/Acosh/Lgamma)
// are membase-only — forced by regbase's HFusionEnums.td lacking those enum
// cases (regbase UnaryFn stops at ilogb=17; BinaryFn slot 18 is divfhp, not
// master's atan2). _membase keeps all 24 HFusion patterns; _regbase drops
// the 7. Dispatcher chooses via hacc::utils::isRegBasedArch(ModuleOp).

static void
populateMathToHFusionConversionPatterns_membase(RewritePatternSet &patterns) {
  patterns.add<
      ElementwiseOpToLinalgUnary<math::ExpOp, linalg::UnaryFn::exp>,
      ElementwiseOpToLinalgUnary<math::LogOp, linalg::UnaryFn::log>,
      ElementwiseOpToLinalgUnary<math::AbsFOp, linalg::UnaryFn::abs>,
      ElementwiseOpToLinalgUnary<math::CeilOp, linalg::UnaryFn::ceil>,
      ElementwiseOpToLinalgUnary<math::FloorOp, linalg::UnaryFn::floor>,
      ElementwiseOpToHFusionBinary<mathExt::LdexpOp, hfusion::BinaryFn::ldexp>,
      ElementwiseOpToHFusionBinary<math::PowFOp, hfusion::BinaryFn::powf>,
      ElementwiseOpToHFusionBinary<math::Atan2Op, hfusion::BinaryFn::atan2>,
      ElementwiseOpToHFusionBinary<math::CopySignOp,
                                   hfusion::BinaryFn::copysign>,
      ElementwiseOpToHFusionUnary<math::SqrtOp, hfusion::UnaryFn::sqrt>,
      ElementwiseOpToHFusionUnary<math::RsqrtOp, hfusion::UnaryFn::rsqrt>,
      ElementwiseOpToHFusionUnary<math::TanhOp, hfusion::UnaryFn::tanh>,
      ElementwiseOpToHFusionUnary<math::AtanOp, hfusion::UnaryFn::atan>,
      ElementwiseOpToHFusionUnary<math::AtanhOp, hfusion::UnaryFn::atanh>,
      ElementwiseOpToHFusionUnary<math::TanOp, hfusion::UnaryFn::tan>,
      ElementwiseOpToHFusionUnary<math::SinOp, hfusion::UnaryFn::sin>,
      ElementwiseOpToHFusionUnary<math::AsinOp, hfusion::UnaryFn::asin>,
      ElementwiseOpToHFusionUnary<math::CosOp, hfusion::UnaryFn::cos>,
      ElementwiseOpToHFusionUnary<math::AcosOp, hfusion::UnaryFn::acos>,
      ElementwiseOpToHFusionUnary<math::SinhOp, hfusion::UnaryFn::sinh>,
      ElementwiseOpToHFusionUnary<math::CoshOp, hfusion::UnaryFn::cosh>,
      ElementwiseOpToHFusionUnary<math::RoundEvenOp,
                                  hfusion::UnaryFn::nearbyint>,
      ElementwiseOpToHFusionUnary<math::AcoshOp, hfusion::UnaryFn::acosh>,
      ElementwiseOpToHFusionUnary<math::AsinhOp, hfusion::UnaryFn::asinh>,
      ElementwiseOpToHFusionUnary<math::AbsIOp, hfusion::UnaryFn::absi>,
      ElementwiseOpToHFusionUnary<math::ErfOp, hfusion::UnaryFn::erf>,
      ElementwiseOpToHFusionUnary<math::Log2Op, hfusion::UnaryFn::log2>,
      ElementwiseOpToHFusionUnary<math::Log10Op, hfusion::UnaryFn::log10>,
      ElementwiseOpToHFusionUnary<math::Log1pOp, hfusion::UnaryFn::log1p>,
      ElementwiseOpToHFusionUnary<math::Exp2Op, hfusion::UnaryFn::exp2>,
      ElementwiseOpToHFusionUnary<math::ExpM1Op, hfusion::UnaryFn::expm1>,
      ElementwiseOpToHFusionUnary<mathExt::IlogbOp, hfusion::UnaryFn::ilogb>,
      ElementwiseOpToHFusionUnary<mathExt::LgammaOp, hfusion::UnaryFn::lgamma>>(
      patterns.getContext());
  patterns.add<MathFmaLowering>(patterns.getContext(),
                                /*enableFma=*/false);
}

static void
populateMathToHFusionConversionPatterns_regbase(RewritePatternSet &patterns,
                                                bool enableFma) {
  // regbase subset: drops Atan2/Atanh/Asin/Acos/Sinh/Acosh/Lgamma
  // (regbase HFusionEnums.td lacks those enum cases).
  patterns.add<
      ElementwiseOpToLinalgUnary<math::ExpOp, linalg::UnaryFn::exp>,
      ElementwiseOpToLinalgUnary<math::LogOp, linalg::UnaryFn::log>,
      ElementwiseOpToLinalgUnary<math::AbsFOp, linalg::UnaryFn::abs>,
      ElementwiseOpToLinalgUnary<math::CeilOp, linalg::UnaryFn::ceil>,
      ElementwiseOpToLinalgUnary<math::FloorOp, linalg::UnaryFn::floor>,
      ElementwiseOpToHFusionBinary<mathExt::LdexpOp, hfusion::BinaryFn::ldexp>,
      ElementwiseOpToHFusionBinary<math::PowFOp, hfusion::BinaryFn::powf>,
      ElementwiseOpToHFusionBinary<math::CopySignOp,
                                   hfusion::BinaryFn::copysign>,
      ElementwiseOpToHFusionUnary<math::SqrtOp, hfusion::UnaryFn::sqrt>,
      ElementwiseOpToHFusionUnary<math::RsqrtOp, hfusion::UnaryFn::rsqrt>,
      ElementwiseOpToHFusionUnary<math::TanhOp, hfusion::UnaryFn::tanh>,
      ElementwiseOpToHFusionUnary<math::AtanOp, hfusion::UnaryFn::atan>,
      ElementwiseOpToHFusionUnary<math::TanOp, hfusion::UnaryFn::tan>,
      // cos still uses the HFusion Normalize expansion for now (no cos wrapper
      // in the template library yet).
      ElementwiseOpToHFusionUnary<math::CosOp, hfusion::UnaryFn::cos>,
      ElementwiseOpToHFusionUnary<math::CoshOp, hfusion::UnaryFn::cosh>,
      ElementwiseOpToHFusionUnary<math::RoundEvenOp,
                                  hfusion::UnaryFn::nearbyint>,
      ElementwiseOpToHFusionUnary<math::AbsIOp, hfusion::UnaryFn::absi>,
      ElementwiseOpToHFusionUnary<math::ErfOp, hfusion::UnaryFn::erf>,
      ElementwiseOpToHFusionUnary<math::Log2Op, hfusion::UnaryFn::log2>,
      ElementwiseOpToHFusionUnary<math::Log10Op, hfusion::UnaryFn::log10>,
      ElementwiseOpToHFusionUnary<math::Log1pOp, hfusion::UnaryFn::log1p>,
      ElementwiseOpToHFusionUnary<math::Exp2Op, hfusion::UnaryFn::exp2>,
      ElementwiseOpToHFusionUnary<math::ExpM1Op, hfusion::UnaryFn::expm1>,
      ElementwiseOpToHFusionUnary<mathExt::IlogbOp, hfusion::UnaryFn::ilogb>>(
      patterns.getContext());
  if (useNormalizeSin()) {
    // A/B measurement switch: legacy HFusion Normalize lowering.
    patterns
        .add<ElementwiseOpToHFusionUnary<math::SinOp, hfusion::UnaryFn::sin>>(
            patterns.getContext());
  } else {
    // f32 sin with rank <= kMaxTemplateLibTrigRank goes to the template-lib
    // SIMT kernel; everything else falls back to Normalize inside the pattern.
    patterns.add<MathTrigToTemplateLib<math::SinOp>>(patterns.getContext());
  }
  patterns.add<MathFmaLowering>(patterns.getContext(), enableFma);
}

void mlir::hfusion::populateMathToHFusionConversionPatterns(
    RewritePatternSet &patterns, ModuleOp moduleOp) {
  if (moduleOp && hacc::utils::isRegBasedArch(moduleOp))
    populateMathToHFusionConversionPatterns_regbase(patterns,
                                                    /*enableFma=*/false);
  else
    populateMathToHFusionConversionPatterns_membase(patterns);
}

namespace {
struct MathToHFusionConversionPass
    : public impl::ConvertMathToHFusionBase<MathToHFusionConversionPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void MathToHFusionConversionPass::runOnOperation() {
  auto *module = getOperation();
  auto moduleOp = dyn_cast<ModuleOp>(module);

  bool isMemBased = moduleOp && hacc::utils::isMemBasedArch(moduleOp);
  bool isRegBased = moduleOp && hacc::utils::isRegBasedArch(moduleOp);
  // Unknown architectures are neither mem- nor reg-based; leave their scalar
  // math unchanged, matching the previous behavior.
  if (isMemBased || isRegBased) {
    RewritePatternSet scalarPatterns(&getContext());
    if (isMemBased)
      populateScalarMathPromotionPatternsMemBased(scalarPatterns);
    else
      populateScalarMathPromotionPatterns(scalarPatterns);
    if (failed(applyPatternsGreedily(module, std::move(scalarPatterns)))) {
      signalPassFailure();
      return;
    }
  }

  ConversionTarget target(getContext());
  target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                         hfusion::HFusionDialect, hivm::HIVMDialect,
                         memref::MemRefDialect,
                         bufferization::BufferizationDialect>();
  // also add dialects that maybe created by hfusion dialect ops
  target.addLegalDialect<arith::ArithDialect>();
  // math dialect ops are allowed if they don't operate on tensors
  target.addDynamicallyLegalDialect<math::MathDialect>(
      [](Operation *op) { return !operateOnTensors(op); });

  RewritePatternSet patterns(&getContext());
  bool shouldEnableFma =
      moduleOp && hacc::utils::isAscend950(moduleOp) && enableFma;
  if (moduleOp && hacc::utils::isRegBasedArch(moduleOp))
    populateMathToHFusionConversionPatterns_regbase(patterns, shouldEnableFma);
  else
    populateMathToHFusionConversionPatterns_membase(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createMathToHFusionConversionPass() {
  return std::make_unique<MathToHFusionConversionPass>();
}

std::unique_ptr<Pass> mlir::createMathToHFusionConversionPass(
    const ConvertMathToHFusionOptions &options) {
  return std::make_unique<MathToHFusionConversionPass>(options);
}

//===- AscendDPMathOpsLowering.h --===//
//===- Convert Ascend DPX Math Ops to HIVMRegbaseIntrins dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------------------------===//

#include "bishengir/Conversion/AscendDPXToHIVMRegbaseIntrins/AscendDPXMathOpsLowering.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace ascend_dpx;

namespace {

template <typename... T>
struct UnaryDirect {};

template <typename DPX_OP, typename LOWER_OP>
struct UnaryDirect<DPX_OP, LOWER_OP>
    : public mlir::ConvertOpToLLVMPattern<DPX_OP> {
  explicit UnaryDirect(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<DPX_OP>(converter) {}

  LogicalResult
  matchAndRewrite(DPX_OP dpx_op, typename DPX_OP::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = dpx_op.getLoc();
    Value arg = dpx_op.getIns();
    Type res = dpx_op.getResult().getType();
    rewriter.replaceOp(dpx_op, rewriter.create<LOWER_OP>(loc, res, arg));
    return success();
  }
};

template <typename DPX_OP, typename LOWER_F16, typename LOWER_F32,
          typename LOWER_BF16>
struct UnaryDirect<DPX_OP, LOWER_F16, LOWER_F32, LOWER_BF16>
    : public mlir::ConvertOpToLLVMPattern<DPX_OP> {
  explicit UnaryDirect(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<DPX_OP>(converter) {}

  LogicalResult
  matchAndRewrite(DPX_OP dpx_op, typename DPX_OP::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = dpx_op.getLoc();
    Value arg = dpx_op.getIns();
    Type res = dpx_op.getResult().getType();
    if (res.isF16())
      rewriter.replaceOp(dpx_op, rewriter.create<LOWER_F16>(loc, res, arg));
    else if (res.isF32())
      rewriter.replaceOp(dpx_op, rewriter.create<LOWER_F32>(loc, res, arg));
    else if (res.isBF16())
      rewriter.replaceOp(dpx_op, rewriter.create<LOWER_BF16>(loc, res, arg));
    else
      return rewriter.notifyMatchFailure(
          dpx_op, "operand type must be one of f16, f32, f64");
    return success();
  }
};
template <typename... T>
struct BinaryDirect {};

template <typename DPX_OP, typename LOWER_OP>
struct BinaryDirect<DPX_OP, LOWER_OP>
    : public mlir::ConvertOpToLLVMPattern<DPX_OP> {
  explicit BinaryDirect(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<DPX_OP>(converter) {}

  LogicalResult
  matchAndRewrite(DPX_OP dpx_op, typename DPX_OP::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = dpx_op.getLoc();
    Value A = dpx_op.getIn1();
    Value B = dpx_op.getIn2();
    Type res = dpx_op.getResult().getType();
    rewriter.replaceOp(dpx_op, rewriter.create<LOWER_OP>(loc, res, A, B));
    return success();
  }
};

template <typename DPX_OP, typename LOWER_INT_OP, typename LOWER_F_OP>
struct BinaryDirect<DPX_OP, LOWER_INT_OP, LOWER_F_OP>
    : public mlir::ConvertOpToLLVMPattern<DPX_OP> {
  explicit BinaryDirect(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<DPX_OP>(converter) {}

  LogicalResult
  matchAndRewrite(DPX_OP dpx_op, typename DPX_OP::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value A = dpx_op.getIn1();
    Value B = dpx_op.getIn2();
    Type res = dpx_op.getResult().getType();
    if (res.isInteger())
      rewriter.replaceOpWithNewOp<LOWER_INT_OP>(dpx_op, res, A, B);
    else
      rewriter.replaceOpWithNewOp<LOWER_F_OP>(dpx_op, res, A, B);
    return success();
  }
};

// copied over from LLVMTritonRemap
LLVM::CallOp emitOrCreateLibCall(OpBuilder &builder, Location loc,
                                 llvm::StringRef funcName, ValueRange operands,
                                 Type returnType) {
  ModuleOp module;
  if (auto *defOp = operands[0].getDefiningOp()) {
    module = defOp->getParentOfType<mlir::ModuleOp>();
  } else {
    module = operands[0].getParentRegion()->getParentOfType<mlir::ModuleOp>();
  }
  LLVM::LLVMFuncOp func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);

  if (!func) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    SmallVector<Type> types;
    for (auto type : operands.getTypes()) {
      types.push_back(type);
    }
    auto funcType =
        LLVM::LLVMFunctionType::get(returnType, ArrayRef<Type>(types), false);

    func = builder.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);

    auto haccAlwaysInlineAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
        hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE);
    func->setAttr(haccAlwaysInlineAttr, builder.getUnitAttr());
    auto llvmEmitCAttr = LLVM::LLVMDialect::getEmitCWrapperAttrName();
    func->setAttr(llvmEmitCAttr, builder.getUnitAttr());
    func->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                  builder.getStringAttr("private"));
    auto ctx = builder.getContext();
    func->setAttr(mlir::hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
    func->setAttr(hivm_regbaseintrins::kDavinciCallingConvAttrName,
                  hivm_regbaseintrins::SIMT_CallableAttr::get(ctx));
  }

  auto calleeAttr = mlir::SymbolRefAttr::get(builder.getContext(), funcName);
  return builder.create<LLVM::CallOp>(loc, returnType, calleeAttr, operands);
}

// declared in empty namespace so not a bunch of global variables
template <typename opName>
constexpr const char *libFuncName = "";
template <typename opName>
constexpr const char *libFuncNameF16 = nullptr;
template <typename opName>
constexpr const char *libFuncNameF32 = nullptr;
template <typename opName>
constexpr const char *libFuncNameBF16 = nullptr;
template <typename opName>
constexpr const char *libFuncNameI8 = nullptr;
template <typename opName>
constexpr const char *libFuncNameI16 = nullptr;
template <typename opName>
constexpr const char *libFuncNameI32 = nullptr;
template <typename opName>
constexpr const char *libFuncNameI64 = nullptr;
// ideally we somehow move these strings into the template parameters
// of LibCallLowering, but C++ template strings are finnicky
#define TCCC                                                                   \
  template <>                                                                  \
  constexpr const char // try to tidy up big hunk of boilerplate with a macro
// Note: The following lines are sorted alphabetically
// Additional note: make sure your formatter doesn't break lines mid expression
// here, since it will make it much harder to sort
TCCC *libFuncName<AbsOp> = "_mlir_ciface_simt_abs_float";
TCCC *libFuncNameI32<AbsOp> = "_mlir_ciface_simt_abs_int32_t";
TCCC *libFuncName<AcosOp> = "_mlir_ciface_simt_acos_float";
TCCC *libFuncName<AcoshOp> = "_mlir_ciface_simt_acosh_float";
TCCC *libFuncName<AsinOp> = "_mlir_ciface_simt_asin_float";
TCCC *libFuncName<AsinhOp> = "_mlir_ciface_simt_asinh_float";
TCCC *libFuncName<Atan2Op> = "_mlir_ciface_simt_atan2_float";
TCCC *libFuncName<AtanOp> = "_mlir_ciface_simt_atan_float";
TCCC *libFuncNameF16<AtanOp> = "_mlir_ciface_simt_atan_half";
TCCC *libFuncName<AtanhOp> = "_mlir_ciface_simt_atanh_float";
TCCC *libFuncNameI32<BrevOp> = "_mlir_ciface_simt_brev_int32_t";
TCCC *libFuncName<CbrtOp> = "_mlir_ciface_simt_cbrt_float";
TCCC *libFuncName<CosOp> = "_mlir_ciface_simt_cos_float";
TCCC *libFuncNameF16<CosOp> = "_mlir_ciface_simt_cos_half";
TCCC *libFuncNameBF16<CosOp> = "_mlir_ciface_simt_cos_bfloat16_t";
TCCC *libFuncName<CoshOp> = "_mlir_ciface_simt_cosh_float";
TCCC *libFuncName<CospiOp> = "_mlir_ciface_simt_cospi_float";
TCCC *libFuncName<CylBesselI0Op> = "_mlir_ciface_simt_cyl_bessel_i0_float";
TCCC *libFuncName<CylBesselI1Op> = "_mlir_ciface_simt_cyl_bessel_i1_float";
TCCC *libFuncName<ErfOp> = "_mlir_ciface_simt_erf_float";
TCCC *libFuncNameF16<ErfOp> = "_mlir_ciface_simt_erf_half";
TCCC *libFuncNameBF16<ErfOp> = "_mlir_ciface_simt_erf_bfloat16_t";
TCCC *libFuncName<ErfcOp> = "_mlir_ciface_simt_erfc_float";
TCCC *libFuncName<ErfcinvOp> = "_mlir_ciface_simt_erfcinv_float";
TCCC *libFuncName<ErfcxOp> = "_mlir_ciface_simt_erfcx_float";
TCCC *libFuncName<ErfinvOp> = "_mlir_ciface_simt_erfinv_float";
TCCC *libFuncName<Exp10Op> = "_mlir_ciface_simt_exp10_float";
TCCC *libFuncName<Expm1Op> = "_mlir_ciface_simt_expm1_float";
TCCC *libFuncName<FastCosfOp> = "_mlir_ciface_simt_fast_cos_float";
TCCC *libFuncName<FastDividefOp> = "_mlir_ciface_simt_fast_divide_float";
TCCC *libFuncName<FastExp10fOp> = "_mlir_ciface_simt_fast_exp10_float";
TCCC *libFuncName<FastExpfOp> = "_mlir_ciface_simt_fast_exp_float";
TCCC *libFuncName<FastLog10fOp> = "_mlir_ciface_simt_fast_log10_float";
TCCC *libFuncName<FastLog2fOp> = "_mlir_ciface_simt_fast_log2_float";
TCCC *libFuncName<FastLogfOp> = "_mlir_ciface_simt_fast_log_float";
TCCC *libFuncName<FastPowfOp> = "_mlir_ciface_simt_fast_pow_float";
TCCC *libFuncName<FastSinfOp> = "_mlir_ciface_simt_fast_sin_float";
TCCC *libFuncName<FastTanfOp> = "_mlir_ciface_simt_fast_tan_float";
TCCC *libFuncName<FastTanhfOp> = "_mlir_ciface_simt_fast_tanh_float";
TCCC *libFuncName<Float2HalfRnOp> = "_mlir_ciface_simt_float2half_rn_float";
TCCC *libFuncName<FdimOp> = "_mlir_ciface_simt_fdim_float";
TCCC *libFuncName<FinitefOp> = "_mlir_ciface_simt_finite_float";
TCCC *libFuncName<Float2IntRnOp> = "_mlir_ciface_simt_float2int_rn_float";
TCCC *libFuncName<Float2IntRzOp> = "_mlir_ciface_simt_float2int_rz_float";
TCCC *libFuncName<Float2IntRdOp> = "_mlir_ciface_simt_float2int_rd_float";
TCCC *libFuncName<Float2IntRuOp> = "_mlir_ciface_simt_float2int_ru_float";
TCCC *libFuncName<Float2LlRnOp> = "_mlir_ciface_simt_float2ll_rn_float";
TCCC *libFuncName<Float2LlRzOp> = "_mlir_ciface_simt_float2ll_rz_float";
TCCC *libFuncName<Float2LlRdOp> = "_mlir_ciface_simt_float2ll_rd_float";
TCCC *libFuncName<Float2LlRuOp> = "_mlir_ciface_simt_float2ll_ru_float";
TCCC *libFuncName<Float2UintRnOp> = "_mlir_ciface_simt_float2uint_rn_float";
TCCC *libFuncName<Float2UintRzOp> = "_mlir_ciface_simt_float2uint_rz_float";
TCCC *libFuncName<Float2UintRdOp> = "_mlir_ciface_simt_float2uint_rd_float";
TCCC *libFuncName<Float2UintRuOp> = "_mlir_ciface_simt_float2uint_ru_float";
TCCC *libFuncName<Float2UllRnOp> = "_mlir_ciface_simt_float2ull_rn_float";
TCCC *libFuncName<Float2UllRzOp> = "_mlir_ciface_simt_float2ull_rz_float";
TCCC *libFuncName<Float2UllRdOp> = "_mlir_ciface_simt_float2ull_rd_float";
TCCC *libFuncName<Float2UllRuOp> = "_mlir_ciface_simt_float2ull_ru_float";
TCCC *libFuncName<FloatAsIntOp> = "_mlir_ciface_simt_float_as_int_float";
TCCC *libFuncName<FloatAsUintOp> = "_mlir_ciface_simt_float_as_uint_float";
TCCC *libFuncName<FmodOp> = "_mlir_ciface_simt_fmod_float";
TCCC *libFuncName<Half2FloatOp> = "_mlir_ciface_simt_half2float_half";
TCCC *libFuncName<GammaOp> = "_mlir_ciface_simt_gamma_float";
TCCC *libFuncNameI32<HaddOp> = "_mlir_ciface_simt_hadd_int32_t";
TCCC *libFuncName<HypotOp> = "_mlir_ciface_simt_hypot_float";
TCCC *libFuncName<ILogbOp> = "_mlir_ciface_simt_ilogb_float";
TCCC *libFuncNameF16<ILogbOp> = "_mlir_ciface_simt_ilogb_half";
TCCC *libFuncName<Int2FloatRnOp> = "_mlir_ciface_simt_int2float_rn_float";
TCCC *libFuncName<Int2FloatRzOp> = "_mlir_ciface_simt_int2float_rz_float";
TCCC *libFuncName<Int2FloatRdOp> = "_mlir_ciface_simt_int2float_rd_float";
TCCC *libFuncName<Int2FloatRuOp> = "_mlir_ciface_simt_int2float_ru_float";
TCCC *libFuncName<IntAsFloatOp> = "_mlir_ciface_simt_int_as_float_float";
TCCC *libFuncName<IsFiniteOp> = "_mlir_ciface_simt_isfinite_float";
TCCC *libFuncName<IsInfOp> = "_mlir_ciface_simt_isinf_float";
TCCC *libFuncName<IsNanOp> = "_mlir_ciface_simt_isnan_float";
TCCC *libFuncName<J0Op> = "_mlir_ciface_simt_j0_float";
TCCC *libFuncName<J1Op> = "_mlir_ciface_simt_j1_float";
TCCC *libFuncName<JnOp> = "_mlir_ciface_simt_jn_float";
TCCC *libFuncName<LdExpOp> = "_mlir_ciface_simt_ldexp_float";
TCCC *libFuncNameF16<LdExpOp> = "_mlir_ciface_simt_ldexp_half";
TCCC *libFuncName<LgammaOp> = "_mlir_ciface_simt_lgamma_float";
TCCC *libFuncName<Ll2FloatRnOp> = "_mlir_ciface_simt_ll2float_rn_float";
TCCC *libFuncName<Ll2FloatRzOp> = "_mlir_ciface_simt_ll2float_rz_float";
TCCC *libFuncName<Ll2FloatRdOp> = "_mlir_ciface_simt_ll2float_rd_float";
TCCC *libFuncName<Ll2FloatRuOp> = "_mlir_ciface_simt_ll2float_ru_float";
TCCC *libFuncName<LlrintOp> = "_mlir_ciface_simt_llrint_float";
TCCC *libFuncName<LlroundOp> = "_mlir_ciface_simt_llround_float";
TCCC *libFuncName<Log10Op> = "_mlir_ciface_simt_log10_float";
TCCC *libFuncName<Log1pOp> = "_mlir_ciface_simt_log1p_float";
TCCC *libFuncNameF16<Log1pOp> = "_mlir_ciface_simt_log1p_half";
TCCC *libFuncName<Log2Op> = "_mlir_ciface_simt_log2_float";
TCCC *libFuncName<MaxOp> = "_mlir_ciface_simt_fmax_float";
TCCC *libFuncNameI32<MaxOp> = "_mlir_ciface_simt_max_int32_t";
TCCC *libFuncName<MinOp> = "_mlir_ciface_simt_fmin_float";
TCCC *libFuncNameI32<MinOp> = "_mlir_ciface_simt_min_int32_t";
TCCC *libFuncName<LogbOp> = "_mlir_ciface_simt_logb_float";
TCCC *libFuncNameI32<Mul24Op> = "_mlir_ciface_simt_mul24_int32_t";
TCCC *libFuncNameI32<MulhiOp> = "_mlir_ciface_simt_mulhi_int32_t";
TCCC *libFuncName<MulRdOp> = "_mlir_ciface_simt_mul_rd_float";
TCCC *libFuncName<MulRnOp> = "_mlir_ciface_simt_mul_rn_float";
TCCC *libFuncName<MulRuOp> = "_mlir_ciface_simt_mul_ru_float";
TCCC *libFuncName<MulRzOp> = "_mlir_ciface_simt_mul_rz_float";
TCCC *libFuncName<NearbyintOp> = "_mlir_ciface_simt_nearbyint_float";
TCCC *libFuncName<NanfOp> = "_mlir_ciface_simt_nan_float";
TCCC *libFuncName<NextafterOp> = "_mlir_ciface_simt_nextafter_float";
TCCC *libFuncName<Norm3dOp> = "_mlir_ciface_simt_norm3d_float";
TCCC *libFuncName<Norm4dOp> = "_mlir_ciface_simt_norm4d_float";
TCCC *libFuncName<NormcdfOp> = "_mlir_ciface_simt_normcdf_float";
TCCC *libFuncName<NormcdfinvOp> = "_mlir_ciface_simt_normcdfinv_float";
TCCC *libFuncName<PowOp> = "_mlir_ciface_simt_pow_float";
TCCC *libFuncNameF16<PowOp> = "_mlir_ciface_simt_pow_half";
TCCC *libFuncNameBF16<PowOp> = "_mlir_ciface_simt_pow_bfloat16_t";
TCCC *libFuncNameI8<PowOp> = "_mlir_ciface_simt_pow_int8_t";
TCCC *libFuncNameI16<PowOp> = "_mlir_ciface_simt_pow_int16_t";
TCCC *libFuncNameI32<PowOp> = "_mlir_ciface_simt_pow_int32_t";
TCCC *libFuncNameI64<PowOp> = "_mlir_ciface_simt_pow_int64_t";
TCCC *libFuncNameI32<PopcOp> = "_mlir_ciface_simt_popc_int32_t";
TCCC *libFuncName<RcbrtOp> = "_mlir_ciface_simt_rcbrt_float";
TCCC *libFuncName<RcpRdOp> = "_mlir_ciface_simt_rcp_rd_float";
TCCC *libFuncName<RcpRnOp> = "_mlir_ciface_simt_rcp_rn_float";
TCCC *libFuncName<RcpRuOp> = "_mlir_ciface_simt_rcp_ru_float";
TCCC *libFuncName<RcpRzOp> = "_mlir_ciface_simt_rcp_rz_float";
TCCC *libFuncName<RecipOp> = "_mlir_ciface_simt_recip_float";
TCCC *libFuncNameF16<RecipOp> = "_mlir_ciface_simt_recip_half";
TCCC *libFuncName<ReluOp> = "_mlir_ciface_simt_relu_float";
TCCC *libFuncNameF16<ReluOp> = "_mlir_ciface_simt_relu_half";
TCCC *libFuncName<RemainderOp> = "_mlir_ciface_simt_remainder_float";
TCCC *libFuncName<RhypotOp> = "_mlir_ciface_simt_rhypot_float";
TCCC *libFuncName<Rnorm3dOp> = "_mlir_ciface_simt_rnorm3d_float";
TCCC *libFuncName<Rnorm4dOp> = "_mlir_ciface_simt_rnorm4d_float";
TCCC *libFuncName<RoundOp> = "_mlir_ciface_simt_round_float";
TCCC *libFuncName<RSqrtOp> = "_mlir_ciface_simt_rsqrt_float";
TCCC *libFuncNameF16<RSqrtOp> = "_mlir_ciface_simt_rsqrt_half";
TCCC *libFuncNameBF16<RSqrtOp> = "_mlir_ciface_simt_rsqrt_bfloat16_t";
TCCC *libFuncName<RsqrtRdOp> = "_mlir_ciface_simt_rsqrt_rd_float";
TCCC *libFuncName<RsqrtRnOp> = "_mlir_ciface_simt_rsqrt_rn_float";
TCCC *libFuncName<RsqrtRuOp> = "_mlir_ciface_simt_rsqrt_ru_float";
TCCC *libFuncName<RsqrtRzOp> = "_mlir_ciface_simt_rsqrt_rz_float";
TCCC *libFuncName<SadOp> = "_mlir_ciface_simt_sad_int32_t";
TCCC *libFuncName<ScalbnOp> = "_mlir_ciface_simt_scalbn_float";
TCCC *libFuncName<SaturatefOp> = "_mlir_ciface_simt_saturate_float";
TCCC *libFuncName<SignbitOp> = "_mlir_ciface_simt_signbit_float";
TCCC *libFuncName<SinOp> = "_mlir_ciface_simt_sin_float";
TCCC *libFuncNameF16<SinOp> = "_mlir_ciface_simt_sin_half";
TCCC *libFuncNameBF16<SinOp> = "_mlir_ciface_simt_sin_bfloat16_t";
TCCC *libFuncName<SinhOp> = "_mlir_ciface_simt_sinh_float";
TCCC *libFuncName<SinpiOp> = "_mlir_ciface_simt_sinpi_float";
TCCC *libFuncName<SqrtRdOp> = "_mlir_ciface_simt_sqrt_rd_float";
TCCC *libFuncName<SqrtRnOp> = "_mlir_ciface_simt_sqrt_rn_float";
TCCC *libFuncName<SqrtRuOp> = "_mlir_ciface_simt_sqrt_ru_float";
TCCC *libFuncName<SqrtRzOp> = "_mlir_ciface_simt_sqrt_rz_float";
TCCC *libFuncName<SubRdOp> = "_mlir_ciface_simt_sub_rd_float";
TCCC *libFuncName<SubRnOp> = "_mlir_ciface_simt_sub_rn_float";
TCCC *libFuncName<SubRuOp> = "_mlir_ciface_simt_sub_ru_float";
TCCC *libFuncName<SubRzOp> = "_mlir_ciface_simt_sub_rz_float";
TCCC *libFuncName<TanOp> = "_mlir_ciface_simt_tan_float";
TCCC *libFuncNameF16<TanOp> = "_mlir_ciface_simt_tan_half";
TCCC *libFuncName<TanhOp> = "_mlir_ciface_simt_tanh_float";
TCCC *libFuncNameF16<TanhOp> = "_mlir_ciface_simt_tanh_half";
TCCC *libFuncNameBF16<TanhOp> = "_mlir_ciface_simt_tanh_bfloat16_t";
TCCC *libFuncName<TgammaOp> = "_mlir_ciface_simt_tgamma_float";
TCCC *libFuncName<TruncOp> = "_mlir_ciface_simt_trunc_float";
TCCC *libFuncName<Uint2FloatRnOp> = "_mlir_ciface_simt_uint2float_rn_float";
TCCC *libFuncName<Uint2FloatRzOp> = "_mlir_ciface_simt_uint2float_rz_float";
TCCC *libFuncName<Uint2FloatRdOp> = "_mlir_ciface_simt_uint2float_rd_float";
TCCC *libFuncName<Uint2FloatRuOp> = "_mlir_ciface_simt_uint2float_ru_float";
TCCC *libFuncName<UintAsFloatOp> = "_mlir_ciface_simt_uint_as_float_float";
TCCC *libFuncName<Ull2FloatRnOp> = "_mlir_ciface_simt_ull2float_rn_float";
TCCC *libFuncName<Ull2FloatRzOp> = "_mlir_ciface_simt_ull2float_rz_float";
TCCC *libFuncName<Ull2FloatRdOp> = "_mlir_ciface_simt_ull2float_rd_float";
TCCC *libFuncName<Ull2FloatRuOp> = "_mlir_ciface_simt_ull2float_ru_float";
TCCC *libFuncName<UmulhiOp> = "_mlir_ciface_simt_umulhi_uint32_t";
TCCC *libFuncName<Y0Op> = "_mlir_ciface_simt_y0_float";
TCCC *libFuncName<Y1Op> = "_mlir_ciface_simt_y1_float";
TCCC *libFuncName<YnOp> = "_mlir_ciface_simt_yn_float";
TCCC *libFuncName<ReciprocalOp> = "_mlir_ciface_simt_reciprocal_float";

TCCC *libFuncNameI32<BytePermOp> = "_mlir_ciface_simt_byte_perm_int32_t";
TCCC *libFuncName<FmaRnOp> = "_mlir_ciface_simt_fma_rn_float";
TCCC *libFuncName<FmaRzOp> = "_mlir_ciface_simt_fma_rz_float";
TCCC *libFuncName<FmaRdOp> = "_mlir_ciface_simt_fma_rd_float";
TCCC *libFuncName<FmaRuOp> = "_mlir_ciface_simt_fma_ru_float";
TCCC *libFuncName<FmaOp> = "_mlir_ciface_simt_fma_float";

TCCC *libFuncNameI32<FfsOp> = "_mlir_ciface_simt_ffs_int32_t";
TCCC *libFuncNameI32<ClzOp> = "_mlir_ciface_simt_clz_int32_t";
TCCC *libFuncName<CopysignOp> = "_mlir_ciface_simt_copysign_float";
TCCC *libFuncName<DivRnOp> = "_mlir_ciface_simt_div_rn_float";
TCCC *libFuncName<DivRzOp> = "_mlir_ciface_simt_div_rz_float";
TCCC *libFuncName<DivRdOp> = "_mlir_ciface_simt_div_rd_float";
TCCC *libFuncName<DivRuOp> = "_mlir_ciface_simt_div_ru_float";
TCCC *libFuncName<AddRnOp> = "_mlir_ciface_simt_add_rn_float";
TCCC *libFuncName<AddRzOp> = "_mlir_ciface_simt_add_rz_float";
TCCC *libFuncName<AddRdOp> = "_mlir_ciface_simt_add_rd_float";
TCCC *libFuncName<AddRuOp> = "_mlir_ciface_simt_add_ru_float";
TCCC *libFuncNameI32<RhaddOp> = "_mlir_ciface_simt_rhadd_int32_t";
#undef TCCC

template <typename DPX_OP>
struct LibCallLowering : public mlir::ConvertOpToLLVMPattern<DPX_OP> {
  explicit LibCallLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<DPX_OP>(converter) {}
  LogicalResult
  matchAndRewrite(DPX_OP op, typename DPX_OP::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type resultType = op.getResult().getType();
    std::string funcName;
    if (libFuncNameF16<DPX_OP> && resultType.isF16())
      funcName = libFuncNameF16<DPX_OP>;
    else if (libFuncNameF32<DPX_OP> && resultType.isF32())
      funcName = libFuncNameF32<DPX_OP>;
    else if (libFuncNameBF16<DPX_OP> && resultType.isBF16())
      funcName = libFuncNameBF16<DPX_OP>;
    else if (libFuncNameI8<DPX_OP> && resultType.isInteger(8))
      funcName = libFuncNameI8<DPX_OP>;
    else if (libFuncNameI16<DPX_OP> && resultType.isInteger(16))
      funcName = libFuncNameI16<DPX_OP>;
    else if (libFuncNameI32<DPX_OP> && resultType.isInteger(32))
      funcName = libFuncNameI32<DPX_OP>;
    else if (libFuncNameI64<DPX_OP> && resultType.isInteger(64))
      funcName = libFuncNameI64<DPX_OP>;
    else
      funcName = libFuncName<DPX_OP>;
    rewriter.replaceOp(op,
                       emitOrCreateLibCall(rewriter, loc, funcName,
                                           adaptor.getOperands(), resultType));
    return success();
  }
};

struct Exp2OpLowering : public mlir::ConvertOpToLLVMPattern<Exp2Op> {
  explicit Exp2OpLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<Exp2Op>(converter) {}

  LogicalResult
  matchAndRewrite(Exp2Op exp2, Exp2Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = exp2.getLoc();
    Value arg = exp2.getIns();
    Type resultType = exp2.getResult().getType();
    Value ln2 = rewriter.create<LLVM::ConstantOp>(
        loc, resultType, rewriter.getFloatAttr(resultType, std::log(2.0)));
    Value normalized = rewriter.create<LLVM::FMulOp>(loc, resultType, arg, ln2);
    rewriter.replaceOp(
        exp2, rewriter.create<LLVM::ExpOp>(loc, resultType, normalized));
    return success();
  }
};

} // namespace

namespace mlir {

void addAscendDPXMathOpsLoweringPatterns(RewritePatternSet &patterns,
                                         LLVMTypeConverter &converter) {
  patterns.add<
      // clang-format off
BinaryDirect<DivOp, LLVM::SDivOp, LLVM::FDivOp>,
BinaryDirect<UDivOp, LLVM::UDivOp>,
Exp2OpLowering,
LibCallLowering<UmulhiOp>,
LibCallLowering<AbsOp>,
LibCallLowering<AcosOp>,
LibCallLowering<AcoshOp>,
LibCallLowering<AsinOp>,
LibCallLowering<AsinhOp>,
LibCallLowering<Atan2Op>,
LibCallLowering<AtanOp>,
LibCallLowering<AtanhOp>,
LibCallLowering<BrevOp>,
LibCallLowering<CbrtOp>,
LibCallLowering<CoshOp>,
LibCallLowering<CosOp>,
LibCallLowering<CospiOp>,
LibCallLowering<CylBesselI0Op>,
LibCallLowering<CylBesselI1Op>,
LibCallLowering<ErfOp>,
LibCallLowering<ErfcOp>,
LibCallLowering<ErfcinvOp>,
LibCallLowering<ErfcxOp>,
LibCallLowering<ErfinvOp>,
LibCallLowering<Exp10Op>,
LibCallLowering<Expm1Op>,
LibCallLowering<FastCosfOp>,
LibCallLowering<FastDividefOp>,
LibCallLowering<Float2HalfRnOp>,
LibCallLowering<FastExp10fOp>,
LibCallLowering<FastExpfOp>,
LibCallLowering<FastLog10fOp>,
LibCallLowering<FastLog2fOp>,
LibCallLowering<FastLogfOp>,
LibCallLowering<FastPowfOp>,
LibCallLowering<FastSinfOp>,
LibCallLowering<FastTanfOp>,
LibCallLowering<FastTanhfOp>,
LibCallLowering<FdimOp>,
LibCallLowering<FinitefOp>,
LibCallLowering<Float2IntRdOp>,
LibCallLowering<Float2IntRnOp>,
LibCallLowering<Float2IntRuOp>,
LibCallLowering<Float2IntRzOp>,
LibCallLowering<Float2LlRdOp>,
LibCallLowering<Float2LlRnOp>,
LibCallLowering<Float2LlRuOp>,
LibCallLowering<Float2LlRzOp>,
LibCallLowering<Float2UintRdOp>,
LibCallLowering<Float2UintRnOp>,
LibCallLowering<Float2UintRuOp>,
LibCallLowering<Float2UintRzOp>,
LibCallLowering<Half2FloatOp>,
LibCallLowering<Float2UllRdOp>,
LibCallLowering<Float2UllRnOp>,
LibCallLowering<Float2UllRuOp>,
LibCallLowering<Float2UllRzOp>,
LibCallLowering<FloatAsIntOp>,
LibCallLowering<FloatAsUintOp>,
LibCallLowering<FmodOp>,
LibCallLowering<GammaOp>,
LibCallLowering<HaddOp>,
LibCallLowering<HypotOp>,
LibCallLowering<ILogbOp>,
LibCallLowering<Int2FloatRdOp>,
LibCallLowering<Int2FloatRnOp>,
LibCallLowering<Int2FloatRuOp>,
LibCallLowering<Int2FloatRzOp>,
LibCallLowering<IntAsFloatOp>,
LibCallLowering<IsFiniteOp>,
LibCallLowering<IsInfOp>,
LibCallLowering<IsNanOp>,
LibCallLowering<J0Op>,
LibCallLowering<J1Op>,
LibCallLowering<JnOp>,
LibCallLowering<LdExpOp>,
LibCallLowering<LgammaOp>,
LibCallLowering<Ll2FloatRdOp>,
LibCallLowering<Ll2FloatRnOp>,
LibCallLowering<MaxOp>,
LibCallLowering<MinOp>,
LibCallLowering<Ll2FloatRuOp>,
LibCallLowering<Ll2FloatRzOp>,
LibCallLowering<LlrintOp>,
LibCallLowering<LlroundOp>,
LibCallLowering<Log10Op>,
LibCallLowering<Log1pOp>,
LibCallLowering<Log2Op>,
LibCallLowering<LogbOp>,
LibCallLowering<NanfOp>,
LibCallLowering<Mul24Op>,
LibCallLowering<MulhiOp>,
LibCallLowering<MulRdOp>,
LibCallLowering<MulRnOp>,
LibCallLowering<MulRuOp>,
LibCallLowering<MulRzOp>,
LibCallLowering<NearbyintOp>,
LibCallLowering<NextafterOp>,
LibCallLowering<Norm3dOp>,
LibCallLowering<Norm4dOp>,
LibCallLowering<NormcdfOp>,
LibCallLowering<NormcdfinvOp>,
LibCallLowering<PowOp>,
LibCallLowering<PopcOp>,
LibCallLowering<RcbrtOp>,
LibCallLowering<RcpRdOp>,
LibCallLowering<RcpRnOp>,
LibCallLowering<RcpRuOp>,
LibCallLowering<RcpRzOp>,
LibCallLowering<RecipOp>,
LibCallLowering<ReluOp>,
LibCallLowering<RemainderOp>,
LibCallLowering<RhypotOp>,
LibCallLowering<Rnorm3dOp>,
LibCallLowering<Rnorm4dOp>,
LibCallLowering<RoundOp>,
LibCallLowering<RSqrtOp>,
LibCallLowering<RsqrtRdOp>,
LibCallLowering<RsqrtRnOp>,
LibCallLowering<RsqrtRuOp>,
LibCallLowering<RsqrtRzOp>,
LibCallLowering<SadOp>,
LibCallLowering<ScalbnOp>,
LibCallLowering<SaturatefOp>,
LibCallLowering<SignbitOp>,
LibCallLowering<SinhOp>,
LibCallLowering<SinOp>,
LibCallLowering<SinpiOp>,
LibCallLowering<SqrtRdOp>,
LibCallLowering<SqrtRnOp>,
LibCallLowering<SqrtRuOp>,
LibCallLowering<SqrtRzOp>,
LibCallLowering<SubRdOp>,
LibCallLowering<SubRnOp>,
LibCallLowering<SubRuOp>,
LibCallLowering<SubRzOp>,
LibCallLowering<TanOp>,
LibCallLowering<TanhOp>,
LibCallLowering<TgammaOp>,
LibCallLowering<TruncOp>,
LibCallLowering<Uint2FloatRdOp>,
LibCallLowering<Uint2FloatRnOp>,
LibCallLowering<Uint2FloatRuOp>,
LibCallLowering<Uint2FloatRzOp>,
LibCallLowering<UintAsFloatOp>,
LibCallLowering<Ull2FloatRdOp>,
LibCallLowering<Ull2FloatRnOp>,
LibCallLowering<Ull2FloatRuOp>,
LibCallLowering<Ull2FloatRzOp>,
LibCallLowering<Y0Op>,
LibCallLowering<Y1Op>,
LibCallLowering<YnOp>,
LibCallLowering<ReciprocalOp>,
LibCallLowering<BytePermOp>,
LibCallLowering<FfsOp>,
LibCallLowering<ClzOp>,
LibCallLowering<CopysignOp>,
LibCallLowering<DivRdOp>,
LibCallLowering<DivRnOp>,
LibCallLowering<DivRzOp>,
LibCallLowering<DivRuOp>,
LibCallLowering<AddRdOp>,
LibCallLowering<AddRnOp>,
LibCallLowering<AddRzOp>,
LibCallLowering<AddRuOp>,
LibCallLowering<RhaddOp>,
LibCallLowering<FmaRdOp>,
LibCallLowering<FmaRnOp>,
LibCallLowering<FmaOp>,
LibCallLowering<FmaRuOp>,
LibCallLowering<FmaRzOp>,
UnaryDirect<CeilOp, hivm_regbaseintrins::CeilOpF16, hivm_regbaseintrins::CeilOpF32, hivm_regbaseintrins::CeilOpBF16>,
UnaryDirect<ExpOp, LLVM::ExpOp>,
UnaryDirect<FloorOp, hivm_regbaseintrins::FloorOpF16, hivm_regbaseintrins::FloorOpF32, hivm_regbaseintrins::FloorOpBF16>,
UnaryDirect<LogOp, LLVM::LogOp, LLVM::LogOp, LLVM::LogOp>,
UnaryDirect<RintOp, hivm_regbaseintrins::RintOpF16, hivm_regbaseintrins::RintOpF32, hivm_regbaseintrins::RintOpBF16>,
UnaryDirect<SqrtOp, LLVM::SqrtOp, LLVM::SqrtOp, LLVM::SqrtOp>
      // clang-format on
      >(converter);
}

} // namespace mlir

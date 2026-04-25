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
TCCC *libFuncName<AtanOp> = "_mlir_ciface_simt_atan_float";
TCCC *libFuncName<CosOp> = "_mlir_ciface_simt_cos_float";
TCCC *libFuncName<ErfOp> = "_mlir_ciface_simt_erf_float";
TCCC *libFuncName<ILogbOp> = "_mlir_ciface_simt_ilogb_float";
TCCC *libFuncName<IsFiniteOp> = "_mlir_ciface_simt_isfinite_float";
TCCC *libFuncName<IsInfOp> = "_mlir_ciface_simt_isinf_float";
TCCC *libFuncName<IsNanOp> = "_mlir_ciface_simt_isnan_float";
TCCC *libFuncName<LdExpOp> = "_mlir_ciface_simt_ldexp_float";
TCCC *libFuncName<Log1pOp> = "_mlir_ciface_simt_log1p_float";
TCCC *libFuncName<Log2Op> = "_mlir_ciface_simt_log2_float";
TCCC *libFuncName<PowOp> = "_mlir_ciface_simt_pow_float";
TCCC *libFuncName<RecipOp> = "_mlir_ciface_simt_recip_float";
TCCC *libFuncName<ReluOp> = "_mlir_ciface_simt_relu_float";
TCCC *libFuncName<RoundOp> = "_mlir_ciface_simt_round_float";
TCCC *libFuncName<RSqrtOp> = "_mlir_ciface_simt_rsqrt_float";
TCCC *libFuncName<SinOp> = "_mlir_ciface_simt_sin_float";
TCCC *libFuncName<TanhOp> = "_mlir_ciface_simt_tanh_float";
TCCC *libFuncName<TanOp> = "_mlir_ciface_simt_tan_float";
TCCC *libFuncNameBF16<CosOp> = "_mlir_ciface_simt_cos_bfloat16_t";
TCCC *libFuncNameBF16<ErfOp> = "_mlir_ciface_simt_erf_bfloat16_t";
TCCC *libFuncNameBF16<PowOp> = "_mlir_ciface_simt_pow_bfloat16_t";
TCCC *libFuncNameBF16<RSqrtOp> = "_mlir_ciface_simt_rsqrt_bfloat16_t";
TCCC *libFuncNameBF16<SinOp> = "_mlir_ciface_simt_sin_bfloat16_t";
TCCC *libFuncNameBF16<TanhOp> = "_mlir_ciface_simt_tanh_bfloat16_t";
TCCC *libFuncNameF16<AtanOp> = "_mlir_ciface_simt_atan_half";
TCCC *libFuncNameF16<CosOp> = "_mlir_ciface_simt_cos_half";
TCCC *libFuncNameF16<ErfOp> = "_mlir_ciface_simt_erf_half";
TCCC *libFuncNameF16<ILogbOp> = "_mlir_ciface_simt_ilogb_half";
TCCC *libFuncNameF16<LdExpOp> = "_mlir_ciface_simt_ldexp_half";
TCCC *libFuncNameF16<Log1pOp> = "_mlir_ciface_simt_log1p_half";
TCCC *libFuncNameF16<PowOp> = "_mlir_ciface_simt_pow_half";
TCCC *libFuncNameF16<RecipOp> = "_mlir_ciface_simt_recip_half";
TCCC *libFuncNameF16<ReluOp> = "_mlir_ciface_simt_relu_half";
TCCC *libFuncNameF16<RSqrtOp> = "_mlir_ciface_simt_rsqrt_half";
TCCC *libFuncNameF16<SinOp> = "_mlir_ciface_simt_sin_half";
TCCC *libFuncNameF16<TanhOp> = "_mlir_ciface_simt_tanh_half";
TCCC *libFuncNameF16<TanOp> = "_mlir_ciface_simt_tan_half";
TCCC *libFuncName<UmulhiOp> = "_mlir_ciface_simt_umulhi_uint32_t";
// NOTE: pow uniquely supports a lot of integer widths. No other library
// function A) supports integer parameter and returns, and B) supports
// this many bit widths
TCCC *libFuncNameI8<PowOp> = "_mlir_ciface_simt_pow_int8_t";
TCCC *libFuncNameI16<PowOp> = "_mlir_ciface_simt_pow_int16_t";
TCCC *libFuncNameI32<PowOp> = "_mlir_ciface_simt_pow_int32_t";
TCCC *libFuncNameI64<PowOp> = "_mlir_ciface_simt_pow_int64_t";
TCCC *libFuncName<FloatAsIntOp> = "_mlir_ciface_simt_float_as_int_float";
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
// TODO: there is intrinics for BinaryDirect<UmulhiOp, hivm_regbaseintrins::UmulhiOp>, 
// but it does not work fails at CCEC for now use the template function 
LibCallLowering<UmulhiOp>,
LibCallLowering<AtanOp>,
LibCallLowering<CosOp>,
LibCallLowering<ErfOp>,
LibCallLowering<ILogbOp>,
LibCallLowering<IsFiniteOp>,
LibCallLowering<IsInfOp>,
LibCallLowering<IsNanOp>,
LibCallLowering<LdExpOp>,
LibCallLowering<Log1pOp>,
LibCallLowering<Log2Op>,
LibCallLowering<PowOp>,
LibCallLowering<RecipOp>,
LibCallLowering<ReluOp>,
LibCallLowering<RoundOp>,
LibCallLowering<RSqrtOp>,
LibCallLowering<SinOp>,
LibCallLowering<TanhOp>,
LibCallLowering<TanOp>,
LibCallLowering<FloatAsIntOp>,
UnaryDirect<CeilOp, hivm_regbaseintrins::CeilOpF16, hivm_regbaseintrins::CeilOpF32, hivm_regbaseintrins::CeilOpBF16>,
UnaryDirect<ExpOp, LLVM::ExpOp>,
UnaryDirect<FloorOp, hivm_regbaseintrins::FloorOpF16, hivm_regbaseintrins::FloorOpF32, hivm_regbaseintrins::FloorOpBF16>,
// TODO: get hivm_regbaseintrins::LogOpF32 to work, right now if we mapp f32 to hivm_regbaseintrins::LogOpF32 it gives ccec error (need to add simt_callable attr)
UnaryDirect<LogOp, LLVM::LogOp, LLVM::LogOp, LLVM::LogOp>,
UnaryDirect<RintOp, hivm_regbaseintrins::RintOpF16, hivm_regbaseintrins::RintOpF32, hivm_regbaseintrins::RintOpBF16>,
// TODO: get hivm_regbaseintrins::SqrtOpF32 to work, right now if we mapp f32 to hivm_regbaseintrins::SqrtOpF32 it gives ccec error
UnaryDirect<SqrtOp, LLVM::SqrtOp, LLVM::SqrtOp, LLVM::SqrtOp>
      // clang-format on
      >(converter);
}

} // namespace mlir

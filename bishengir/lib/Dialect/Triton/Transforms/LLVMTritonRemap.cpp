//===- LLVMTritonRemap.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Regex.h"
#include <cmath>
#include <regex>
#include <string>

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_TRITONREMAP
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace triton
} // namespace bishengir

using namespace mlir;
using namespace hivm;

#define DEBUG_TYPE "triton-remap"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

struct PTXAtomicInfo {
  StringRef kind;
  StringRef memorySpace;
  StringRef type;
};

struct AscendAtomicInfo {
  StringRef kind;
  StringRef memorySpace;
  StringRef type;
};

struct AsmInfo {
  std::string predicate;
  std::string op;
  std::string space;
  int vec = 1;
  int bits = 0;
  std::string regs;
  std::string mem;
  int totalBits = 0;
};

struct FuncInfo {
  LLVM::LLVMFunctionType funcType;
  StringAttr funcName;
  ArrayAttr funcArgAttr;
};

const DenseMap<StringRef,
               std::function<Value(OpBuilder &, Location, OperandRange)>>
    intrinsicReplacementMap = {
        // intrinsic name
        {"llvm.nvvm.div.full",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           Type resTy = operands[0].getType();
           return builder.create<LLVM::FDivOp>(loc, resTy, operands[0],
                                               operands[1]);
         }},
        {"llvm.nvvm.ex2.approx.f",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           Type ty = operands[0].getType();
           Value ln2 = builder.create<LLVM::ConstantOp>(
               loc, ty, builder.getFloatAttr(ty, std::log(2.0)));
           Value normalized =
               builder.create<LLVM::FMulOp>(loc, ty, operands[0], ln2);
           return builder.create<LLVM::ExpOp>(loc, ty, normalized);
         }},

        // inline asm name
        {"div.full.f32",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           Type resTy = operands[0].getType();
           return builder.create<LLVM::FDivOp>(loc, resTy, operands[0],
                                               operands[1]);
         }},
        {"ex2.approx.f32",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           Type ty = operands[0].getType();
           Value ln2 = builder.create<LLVM::ConstantOp>(
               loc, ty, builder.getFloatAttr(ty, std::log(2.0)));
           Value normalized =
               builder.create<LLVM::FMulOp>(loc, ty, operands[0], ln2);
           return builder.create<LLVM::ExpOp>(loc, ty, normalized);
         }}};

Value emitOrCreateLibCall(OpBuilder &builder, Location loc,
                          llvm::StringRef funcName, OperandRange operands,
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
  auto call =
      builder.create<LLVM::CallOp>(loc, returnType, calleeAttr, operands);
  return call.getResult();
}

const DenseMap<StringRef,
               std::function<Value(OpBuilder &, Location, OperandRange)>>
    llvmCallReplacementMap = {

        {"__nv_floorf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return builder.create<hivm_regbaseintrins::FloorOpF32>(
               loc, operands[0].getType(), operands[0]);
         }},
        {"__nv_sqrtf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return builder.create<LLVM::SqrtOp>(loc, operands[0].getType(),
                                               operands[0]);
         }},
        {"__nv_logf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return builder.create<LLVM::LogOp>(loc, operands[0].getType(),
                                              operands[0]);
         }},
         {"__nv_log2f",
           [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
             return emitOrCreateLibCall(builder, loc,
                                        "_mlir_ciface_simt_log2_float", operands,
                                        operands[0].getType());
         }},
        {"__nv_ceilf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return builder.create<hivm_regbaseintrins::CeilOpF32>(
               loc, operands[0].getType(), operands[0]);
         }},
        {"__hmf_rint",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           auto ctx = builder.getContext();
           if (operands[0].getType() == Float16Type::get(ctx)) {
             return builder.create<hivm_regbaseintrins::RintOpF16>(
                 loc, operands[0].getType(), operands[0]);
           } else if (operands[0].getType() == FloatType::getF32(ctx)) {
             return builder.create<hivm_regbaseintrins::RintOpF32>(
                 loc, operands[0].getType(), operands[0]);
           } else if (operands[0].getType() == BFloat16Type::get(ctx)) {
             return builder.create<hivm_regbaseintrins::RintOpBF16>(
                 loc, operands[0].getType(), operands[0]);
           }
           llvm_unreachable("unsupported type for __hmf_rint");
         }},
        {"__nv_rintf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return builder.create<hivm_regbaseintrins::RintOpF32>(
               loc, operands[0].getType(), operands[0]);
         }},
        {"__nv_umulhi",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_umulhi_uint32_t",
                                      operands, operands[0].getType());
         }},
        {"__nv_cosf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_cos_float", operands,
                                      operands[0].getType());
         }},
        {"__nv_expf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return builder.create<LLVM::ExpOp>(loc, operands[0].getType(),
                                              operands[0]);
         }},
        {"__nv_exp2f",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           Type ty = operands[0].getType();
           Value ln2 = builder.create<LLVM::ConstantOp>(
               loc, ty, builder.getFloatAttr(ty, std::log(2.0)));
           Value normalized =
               builder.create<LLVM::FMulOp>(loc, ty, operands[0], ln2);
           return builder.create<LLVM::ExpOp>(loc, ty, normalized);
         }},
        {"__nv_erff",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_erf_float", operands,
                                      operands[0].getType());
         }},
        {"__nv_rsqrtf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_rsqrt_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_isnan",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_isnan_float", operands,
                                      builder.getI1Type());
         }},
        {"__nv_isnanf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_isnan_float", operands,
                                      builder.getI1Type());
         }},
        {"__hmf_isinf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_isinf_float", operands,
                                      builder.getI1Type());
         }},
        {"__nv_isinff",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_isinf_float", operands,
                                      builder.getI1Type());
         }},
        {"__nv_isfinited",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_isfinite_float",
                                      operands, builder.getI1Type());
         }},
        {"__hmf_tanhf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_tanh_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_tanhDh",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_tanh_half", operands,
                                      operands[0].getType());
         }},
        {"__nv_sinf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_sin_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_powf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_pow_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_powDh",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_pow_half", operands,
                                      operands[0].getType());
         }},
        {"__hmf_powDb",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_pow_bfloat16_t",
                                      operands,
                                      operands[0].getType());
         }},
        {"__hmf_powi",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_pow_int32_t", operands,
                                      operands[0].getType());
         }},
        {"__hmf_tanf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_tan_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_tanDh",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_tan_half", operands,
                                      operands[0].getType());
         }},
        {"__hmf_powi",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_pow_int32_t", operands,
                                      operands[0].getType());
         }},
        {"__nv_fdiv_rn",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
          Type type = operands[0].getType();

          if (type.isF32()) {
              return emitOrCreateLibCall(builder, loc,
                                        "_mlir_ciface_simt_divrn_float", operands,
                                        operands[0].getType());
          }
          return builder.create<LLVM::FDivOp>(loc, type, operands[0], operands[1]);
         }},
        {"__hmf_ldexpf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_ldexp_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_ldexpDh",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_ldexp_half", operands,
                                      operands[0].getType());
         }},
        {"__hmf_log1pf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_log1p_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_log1pDh",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_log1p_half", operands,
                                      operands[0].getType());
         }},
        {"__hmf_recipf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_recip_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_recipDh",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_recip_half", operands,
                                      operands[0].getType());
         }},
        {"__hmf_atanf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_atan_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_atanDh",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_atan_half", operands,
                                      operands[0].getType());
         }},
        {"__hmf_ilogbf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_ilogb_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_ilogbDh",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_ilogb_half", operands,
                                      operands[0].getType());
         }},
        {"__hmf_reluf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_relu_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_reluDh",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_relu_half", operands,
                                      operands[0].getType());
         }},
        {"__hmf_roundf",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_round_float", operands,
                                      operands[0].getType());
         }},
        {"__hmf_float_as_int_fp32",
         [](OpBuilder &builder, Location loc, OperandRange operands) -> Value {
           return emitOrCreateLibCall(builder, loc,
                                      "_mlir_ciface_simt_float_as_int_float", operands,
                                      operands[0].getType());
         }},
};

static Type parseTypeString(OpBuilder &b, StringRef str) {
  if (str.contains("u8") || str.contains("b8") || str.contains("s8"))
    return b.getI8Type();
  if (str.contains("u16") || str.contains("b16") || str.contains("s16"))
    return b.getI16Type();
  if (str.contains("u32") || str.contains("b32") || str.contains("s32"))
    return b.getI32Type();
  if (str.contains("u64") || str.contains("b64") || str.contains("s64"))
    return b.getI64Type();
  if (str.contains("fp16"))
    return b.getF16Type();
  if (str.contains("fp32"))
    return b.getF32Type();
  if (str.contains("fp64"))
    return b.getF64Type();
  if (str.contains("f16x2"))
    return VectorType::get({2}, b.getF16Type());
  if (str.contains("bf16x2"))
    return VectorType::get({2}, b.getBF16Type());
  if (str.contains("bf16"))
    return b.getBF16Type();

  llvm_unreachable("Unhandled type in inline asm.");
}

static Value createAndInitializeVector(OpBuilder &b, Location loc, Type ty,
                                       SmallVector<Value> &init) {
  Value vector = b.create<LLVM::UndefOp>(loc, ty);
  for (size_t i = 0; i < init.size(); i++) {
    Value idx =
        b.create<LLVM::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr(i));
    vector = b.create<LLVM::InsertElementOp>(loc, ty, vector, init[i], idx);
  }

  return vector;
}

namespace {

struct PackRmFromShflSyncPattern : public OpRewritePattern<NVVM::ShflOp> {
  using OpRewritePattern<NVVM::ShflOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NVVM::ShflOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value offset = op.getOffset();
    Value threadMask = op.getThreadMask();
    Value maskAndClamp = op.getMaskAndClamp();
    Value src = op.getVal();
    Type i32Ty = rewriter.getI32Type();

    Operation *argThreadMask = threadMask.getDefiningOp();

    if (argThreadMask) {
      if (auto llvmConst = dyn_cast<LLVM::ConstantOp>(argThreadMask)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(llvmConst.getValue())) {
          APInt val = intAttr.getValue();
          if (val.isZero()) {
            // threadMask is constant 0 means a potential predication
            op.emitWarning(
                "SHFL op with predication is currently not supported yet.");
          }
        }
      }
    }

    // Constants
    auto c31 = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty,
        rewriter.getI32IntegerAttr(31)); // 5-bit mask
    auto c8191 = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty,
        rewriter.getI32IntegerAttr(8191)); // 13-bit mask (0x1FFF)
    auto c8 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                rewriter.getI32IntegerAttr(8));

    // Rm[4:0] = offset & 0x1F
    auto offMasked = rewriter.create<LLVM::AndOp>(loc, offset, c31);

    // mask_and_clamp[12:0] -> shift into [20:8]
    auto mcMasked = rewriter.create<LLVM::AndOp>(loc, maskAndClamp, c8191);
    auto mcShifted = rewriter.create<LLVM::ShlOp>(loc, mcMasked, c8);

    // pack into i32
    auto ctrl = rewriter.create<LLVM::OrOp>(loc, offMasked, mcShifted);

    auto kind = op.getKind();
    Value newRes;
    if (src.getType().isInteger(32)) {
      switch (kind) {
      case NVVM::ShflKind::bfly:
        newRes = rewriter.create<hivm_regbaseintrins::ShflBflyOpI32>(
            loc, src.getType(), src, ctrl);
        break;
      case NVVM::ShflKind::up:
        newRes = rewriter.create<hivm_regbaseintrins::ShflUpOpI32>(
            loc, src.getType(), src, ctrl);
        break;
      case NVVM::ShflKind::down:
        newRes = rewriter.create<hivm_regbaseintrins::ShflDownOpI32>(
            loc, src.getType(), src, ctrl);
        break;
      case NVVM::ShflKind::idx:
        newRes = rewriter.create<hivm_regbaseintrins::ShflIdxOpI32>(
            loc, src.getType(), src, ctrl);
        break;
      }
    }

    rewriter.replaceOp(op, newRes);
    return success();
  }
};

void replacePtxToIntrins(LLVM::LLVMFuncOp funcOp) {
  funcOp.walk([&](NVVM::ThreadIdXOp tidOp) {
    OpBuilder builder(tidOp);
    Value newVal = builder.create<hivm_regbaseintrins::ThreadIdXOp>(
        tidOp.getLoc(), builder.getI32Type());
    tidOp.getResult().replaceAllUsesWith(newVal);
    tidOp.erase();
  });

  funcOp.walk([&](NVVM::SyncWarpOp syncWarpOp) {
    OpBuilder builder(syncWarpOp);
    builder.create<hivm_regbaseintrins::SyncThreadsOp>(syncWarpOp->getLoc());
    syncWarpOp.erase();
  });

  funcOp.walk([&](NVVM::Barrier0Op barOp) {
    OpBuilder builder(barOp);
    builder.create<hivm_regbaseintrins::SyncThreadsOp>(barOp->getLoc());
    barOp.erase();
  });
}

void changeAddrSpace(LLVM::LLVMFuncOp funcOp) {
  funcOp.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(result.getType())) {
        if (ptrType.getAddressSpace() == 3) {
          Type newType = LLVM::LLVMPointerType::get(funcOp->getContext(), 6);
          result.setType(newType);
        }
      }
    }

    for (Value operand : op->getOperands()) {
      if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(operand.getType())) {
        if (ptrType.getAddressSpace() == 3) {
          Type newType = LLVM::LLVMPointerType::get(funcOp->getContext(), 6);
          operand.setType(newType);
        }
      }
    }
  });
}

void convertGridIntrinsicsToArgs(LLVM::LLVMFuncOp funcOp,
                                 MLIRContext *context) {
  Block *entryBlock = &funcOp.getBody().front();
  OpBuilder builder(context);
  auto loc = funcOp.getLoc();

  auto oldType = funcOp.getFunctionType();
  SmallVector<Type, 4> argTypes(oldType.getParams().begin(),
                                oldType.getParams().end());
  // TODO: the nctaid.x/y/z might be duplicated if the grid dims are passed in
  // as kernel parameters. Do we need to optimize this?

  // append six i64s (nctaid.x/y/z then ctaid.x/y/z)
  // always, even if present
  argTypes.append(6, builder.getI64Type());
  argTypes.push_back(LLVM::LLVMPointerType::get(context, 6));

  auto newFuncType =
      LLVM::LLVMFunctionType::get(oldType.getReturnType(), argTypes, false);
  funcOp.setType(newFuncType);

  // add arguments in the same order we appended them:
  // nctaid.x, nctaid.y, nctaid.z, ctaid.x, ctaid.y, ctaid.z, ub(ptr<6>)
  Value nctaidX = entryBlock->addArgument(builder.getI64Type(), loc);
  Value nctaidY = entryBlock->addArgument(builder.getI64Type(), loc);
  Value nctaidZ = entryBlock->addArgument(builder.getI64Type(), loc);
  Value ctaidX = entryBlock->addArgument(builder.getI64Type(), loc);
  Value ctaidY = entryBlock->addArgument(builder.getI64Type(), loc);
  Value ctaidZ = entryBlock->addArgument(builder.getI64Type(), loc);
  Value ub =
      entryBlock->addArgument(LLVM::LLVMPointerType::get(context, 6), loc);

  DenseMap<StringRef, Value> intrinsicToArg = {
      // nctaid.* -> appended nctaid args
      {"llvm.nvvm.read.ptx.sreg.nctaid.x", nctaidX},
      {"llvm.nvvm.read.ptx.sreg.nctaid.y", nctaidY},
      {"llvm.nvvm.read.ptx.sreg.nctaid.z", nctaidZ},
      // ctaid.* -> appended ctaid args
      {"llvm.nvvm.read.ptx.sreg.ctaid.x", ctaidX},
      {"llvm.nvvm.read.ptx.sreg.ctaid.y", ctaidY},
      {"llvm.nvvm.read.ptx.sreg.ctaid.z", ctaidZ},
  };

  DenseMap<StringRef, Value> nvvmOpNameToArg = {
      // nctaid
      {"nvvm.read.ptx.sreg.nctaid.x", nctaidX},
      {"nvvm.read.ptx.sreg.nctaid.y", nctaidY},
      {"nvvm.read.ptx.sreg.nctaid.z", nctaidZ},
      // ctaid
      {"nvvm.read.ptx.sreg.ctaid.x", ctaidX},
      {"nvvm.read.ptx.sreg.ctaid.y", ctaidY},
      {"nvvm.read.ptx.sreg.ctaid.z", ctaidZ},
  };

  // handle llvm.call_intrinsic form
  funcOp.walk([&](LLVM::CallIntrinsicOp callOp) {
    StringRef name = callOp.getIntrin();
    auto it = intrinsicToArg.find(name);
    if (it != intrinsicToArg.end()) {
      OpBuilder builder(callOp);
      Value arg = it->second;
      Value newVal = builder.create<LLVM::TruncOp>(callOp.getLoc(),
                                                   builder.getI32Type(), arg);
      callOp.getResult(0).replaceAllUsesWith(newVal);
      callOp.erase();
    }
  });

  // handle the direct nvvm.* op
  funcOp.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    auto it = nvvmOpNameToArg.find(name);
    if (it != nvvmOpNameToArg.end()) {
      OpBuilder builder(op);
      Value arg = it->second;
      Value newVal = builder.create<LLVM::TruncOp>(op->getLoc(),
                                                   builder.getI32Type(), arg);
      if (!op->getResults().empty()) {
        op->getResult(0).replaceAllUsesWith(newVal);
      }
      op->erase();
    }
  });

  // handle llvm.inline_asm form
  funcOp.walk([&](LLVM::InlineAsmOp asmOp) {
    StringRef asmStr = asmOp.getAsmString();

    Value chosenArg;
    // ctaid.* tokens
    if (asmStr.contains("ctaid.x") || asmStr.contains("%ctaid.x")) {
      chosenArg = ctaidX;
    } else if (asmStr.contains("ctaid.y") || asmStr.contains("%ctaid.y")) {
      chosenArg = ctaidY;
    } else if (asmStr.contains("ctaid.z") || asmStr.contains("%ctaid.z")) {
      chosenArg = ctaidZ;
    }
    // nctaid.* tokens (if inline asm used the nctaid names)
    else if (asmStr.contains("nctaid.x") || asmStr.contains("%nctaid.x")) {
      chosenArg = nctaidX;
    } else if (asmStr.contains("nctaid.y") || asmStr.contains("%nctaid.y")) {
      chosenArg = nctaidY;
    } else if (asmStr.contains("nctaid.z") || asmStr.contains("%nctaid.z")) {
      chosenArg = nctaidZ;
    }

    if (!chosenArg)
      return;

    OpBuilder builder(asmOp);
    Value newVal = builder.create<LLVM::TruncOp>(
        asmOp.getLoc(), builder.getI32Type(), chosenArg);
    if (!asmOp.getResults().empty()) {
      asmOp.getResult(0).replaceAllUsesWith(newVal);
    }
    asmOp.erase();
  });

  funcOp.walk([&](LLVM::AddressOfOp addrOp) {
    if (addrOp.getGlobalName() == "global_smem") {
      addrOp.getResult().replaceAllUsesWith(ub);
    } else {
      for (auto &use :
           llvm::make_early_inc_range(addrOp.getResult().getUses())) {
        use.getOwner()->erase();
      }
    }
    addrOp.erase();
  });

  changeAddrSpace(funcOp);
}

void replaceIntrinsicsWithOps(
    LLVM::LLVMFuncOp funcOp,
    DenseMap<StringRef,
             std::function<Value(OpBuilder &, Location, OperandRange)>>
        intrinsicReplacements) {
  funcOp.walk([&](LLVM::CallIntrinsicOp callOp) {
    StringRef name = callOp.getIntrin();

    auto it = intrinsicReplacements.find(name);
    if (it == intrinsicReplacements.end()) {
      llvm::dbgs() << "cannot find replacement for " << name << "\n";
      return;
    }

    auto &builderFn = it->second;
    OpBuilder builder(callOp);
    Location loc = callOp.getLoc();

    Value newVal = builderFn(builder, loc, callOp.getOperands());

    if (!newVal) {
      LDBG("replacement function returned null for intrinsic: " << name);
      return;
    }

    callOp.getResult(0).replaceAllUsesWith(newVal);

    callOp.erase();
  });

  funcOp.walk([&](LLVM::InlineAsmOp asmOp) {
    StringRef asmStr = asmOp.getAsmString();

    if (asmStr.contains("ld.") || asmStr.contains("st.") ||
        asmStr.contains("atom.") || asmStr.contains("mov.u32 $0, 0x0")) {
      return;
    }

    // normalize inline asm string -> take first token (e.g. "div.full.f32")
    StringRef name = asmStr;
    size_t pos = asmStr.find_first_of(" \t$");
    if (pos != StringRef::npos)
      name = asmStr.substr(0, pos);
    if (!name.empty() && name.front() == ';')
      name = name.drop_front();
    while (!name.empty() && name.back() == ';')
      name = name.drop_back();

    auto it = intrinsicReplacements.find(name);
    if (it == intrinsicReplacements.end()) {
      it = intrinsicReplacements.find(asmStr);
      if (it == intrinsicReplacements.end()) {
        LDBG("No replacement found for normalized intrinsic: '"
             << name << "' (orig: '" << asmStr << "')");
        return;
      }
    }

    auto &builderFn = it->second;
    OpBuilder builder(asmOp);
    Location loc = asmOp.getLoc();

    Value newVal = builderFn(builder, loc, asmOp.getOperands());

    if (!newVal) {
      LDBG("replacement function returned null for inline asm: " << name);
      return;
    }

    if (!asmOp.getResults().empty()) {
      asmOp.getResult(0).replaceAllUsesWith(newVal);
    }
    asmOp.erase();
  });
}

void replaceLLVMCallWithOps(
    LLVM::LLVMFuncOp funcOp,
    DenseMap<StringRef,
             std::function<Value(OpBuilder &, Location, OperandRange)>>
        intrinsicReplacements) {
  funcOp.walk([&](LLVM::CallOp callOp) {
    auto attr = callOp.getCalleeAttr();
    if (!attr) {
      return;
    }

    auto name = attr.getRootReference().getValue();

    auto it = intrinsicReplacements.find(name);
    if (it == intrinsicReplacements.end()) {
      LDBG("cannot find replacement for " << name);
      return;
    }

    auto &builderFn = it->second;
    OpBuilder builder(callOp);
    Location loc = callOp.getLoc();

    Value newVal = builderFn(builder, loc, callOp.getOperands());

    callOp.getResult().replaceAllUsesWith(newVal);
    callOp.erase();
  });
}

template <typename MapT, typename KeyT>
auto lookupOrError(const MapT &map, const KeyT &key) ->
    typename MapT::mapped_type {
  auto it = map.find(key);
  if (it == map.end()) {
    std::string keyStr;
    llvm::raw_string_ostream os(keyStr);
    os << key;
    os.flush();
    llvm::errs() << "No mapping for '" << keyStr << "'\n";
    llvm::report_fatal_error("Unknown map key");
  }
  return it->second;
}

void constructAtomicAsmString(StringRef asmStr, size_t atomPos,
                              llvm::SmallString<64> &newAsmStr,
                              llvm::SmallString<16> &newConstraint) {
  llvm::StringMap<llvm::StringRef> kindMap = {
      {"add", "ADD"},   {"max", "MAX"}, {"min", "MIN"},
      {"exch", "EXCH"}, {"cas", "CAS"},
  };

  llvm::StringMap<llvm::StringRef> spaceMap = {
      {"global", "G"},
      {"shared", "S"},
  };

  llvm::StringMap<llvm::StringRef> typeMap = {
      {"b32", "u32"}, {"u32", "u32"}, {"s32", "s32"}, {"b64", "u64"},
      {"u64", "u64"}, {"f32", "f32"}, {"f16", "f16"},
  };

  llvm::StringRef l2CacheHint = "NMFV"; // default value
  StringRef opcode =
      asmStr.drop_front(atomPos); // "atom.global.gpu.acq_rel.add.u32"
  opcode = opcode.take_while([](char c) { return !isspace(c); });
  // parts = {"atom", "global", "gpu", "acq_rel", "add", "u32"}
  SmallVector<StringRef, 8> parts;
  opcode.split(parts, '.');
  assert(parts.size() > 5);
  PTXAtomicInfo ptxAtomInfo = {parts[4], parts[1], parts[5]};

  AscendAtomicInfo ascendAtomInfo;
  ascendAtomInfo.kind = lookupOrError(kindMap, ptxAtomInfo.kind);
  ascendAtomInfo.memorySpace = lookupOrError(spaceMap, ptxAtomInfo.memorySpace);
  ascendAtomInfo.type = lookupOrError(typeMap, ptxAtomInfo.type);

  llvm::raw_svector_ostream os(newAsmStr);
  os << "MOVI $0, #0 wait:0b0000000 stall:10;\n\t"
     << "ATOM." << ascendAtomInfo.memorySpace << "." << ascendAtomInfo.kind
     << "." << ascendAtomInfo.type << "." << l2CacheHint << " $0,[$1],$2";

  if (ascendAtomInfo.kind.equals_insensitive("CAS")) {
    os << ",$3 ws:0 rs:7 wait:0b0000000 stall:2 @$4";
    newConstraint = "=&R,L,R,R,B";
  } else {
    os << " ws:0 rs:7 wait:0b0000000 stall:2 @$3";
    newConstraint = "=&R,L,R,B";
  }
}

void handleMov(StringRef inst, llvm::SmallString<64> &newAsmStr) {
  auto [opcode, rest] = inst.split(' ');
  SmallVector<llvm::StringRef, 4> operands;
  rest.split(operands, ',', -1, false);

  if (operands.size() != 2) {
    LDBG("Invalid operand count: " << inst);
    // llvm::report_fatal_error("Cannot remap mov instruction");
  }

  StringRef dst = operands[0].trim();
  StringRef src = operands[1].trim();
  llvm::raw_svector_ostream os(newAsmStr);

  if (!src.starts_with("$")) {
    int64_t immVal = 0;

    bool failed = src.getAsInteger(0, immVal);

    if (failed) {
      LDBG("Failed to parse immediate: " << src);
      // llvm::report_fatal_error("Cannot remap mov instruction");
    }

    os << "MOVI " << dst << ", "
       << "#" << immVal << " wait:0b0000000 stall:10";
  } else {
    os << "MOV " << dst << ", " << src << " wait:0b0000000 stall:10";
  }
}

static Value castValueToType(OpBuilder &builder, Location loc, Value val,
                             Type targetType) {
  OpBuilder::InsertionGuard g(builder);
  if (!val)
    return nullptr;
  Type srcType = val.getType();
  if (srcType == targetType)
    return val;
  // int
  if (auto srcInt = dyn_cast<IntegerType>(srcType)) {
    if (auto dstInt = dyn_cast<IntegerType>(targetType)) {
      unsigned s = srcInt.getWidth();
      unsigned d = dstInt.getWidth();
      if (s == d)
        return builder.create<LLVM::BitcastOp>(loc, targetType, val);
      if (s > d)
        return builder.create<LLVM::TruncOp>(loc, targetType, val);
      return builder.create<LLVM::ZExtOp>(loc, targetType, val);
    }
  }
  // vector types: cast elementwise
  if (auto srcVec = dyn_cast<VectorType>(srcType)) {
    if (auto dstVec = dyn_cast<VectorType>(targetType)) {
      int ne = srcVec.getNumElements();
      Value out = builder.create<LLVM::UndefOp>(loc, targetType);
      for (int i = 0; i < ne; ++i) {
        Value idx = builder.create<LLVM::ConstantOp>(
            loc, builder.getI32Type(), builder.getI32IntegerAttr(i));
        Value elt = builder.create<LLVM::ExtractElementOp>(
            loc, srcVec.getElementType(), val, idx);
        Value castedElt =
            castValueToType(builder, loc, elt, dstVec.getElementType());
        out = builder.create<LLVM::InsertElementOp>(loc, targetType, out,
                                                    castedElt, idx);
      }
      return out;
    }
  }
  return builder.create<LLVM::BitcastOp>(loc, targetType, val);
}

void handleMov(OpBuilder &builder, Location loc,
               llvm::StringMap<Value> &variableMap, StringRef instruction) {
  llvm::Regex movMatcher(
      R"(mov\.([^[:space:]]+)[[:space:]]+(\$[0-9]+),[[:space:]]*(\$[0-9]+|0x[0-9a-fA-F]+)[[:space:]]*.*)");
  SmallVector<StringRef, 4> matches;

  if (!movMatcher.match(instruction, &matches)) {
    LDBG("Warning: MOV instrtion failed to match: " << instruction);
    return;
  }
  // matches[0] is the entire instruction
  // matches[2] is the destination, matches[3] is the source
  Value toBeMoved;
  if (matches[3].contains("$")) {
    // the source is a register
    toBeMoved = variableMap[matches[3]];
  } else {
    // moving a integer literal
    Value destVal = variableMap[matches[2]];
    Type literalType =
        destVal ? destVal.getType() : parseTypeString(builder, matches[1]);
    unsigned long long parsedVal = std::stoull(matches[3].str(), nullptr, 16);

    auto intAttr = builder.getIntegerAttr(
        literalType,
        APInt(cast<IntegerType>(literalType).getWidth(), parsedVal, true));
    toBeMoved = builder.create<LLVM::ConstantOp>(loc, literalType, intAttr);
  }

  // Check if the move is predicated
  llvm::Regex maskMatcher(R"(@!?(\$[0-9]+)[[:space:]]+.*)");
  SmallVector<StringRef, 4> masked;
  if (maskMatcher.match(instruction, &masked)) {
    // Generate an if stmt
    scf::IfOp ifOp = builder.create<scf::IfOp>(loc, toBeMoved.getType(),
                                               variableMap[masked[1]], true);

    if (masked[0].contains("!")) {
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    } else
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    builder.create<scf::YieldOp>(loc, toBeMoved);

    if (masked[0].contains("!")) {
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    } else
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

    // yield the old value of the dest register — but CAST it to the moved
    // value's type if its type differs so both branches produce the same result
    // type.
    Value oldVal = variableMap[matches[2]];
    if (oldVal && oldVal.getType() != toBeMoved.getType()) {
      oldVal = castValueToType(builder, loc, oldVal, toBeMoved.getType());
    }
    builder.create<scf::YieldOp>(loc, oldVal);

    builder.setInsertionPointAfter(ifOp);
    variableMap[matches[2]] = ifOp->getResult(0);
  } else
    // Just assign the result to the dest register
    variableMap[matches[2]] = toBeMoved;
}

std::optional<StringRef> handleCreatePolicy(StringRef instruction) {
  static const llvm::Regex matcher(
      // createpolicy .fractional .L2::evict_last.b64          $0, 1.0
      R"(createpolicy\.fractional\.L2::([^.]+)\.b64[[:space:]]+.*)");

  SmallVector<StringRef, 2> matches;
  if (!matcher.match(instruction, &matches)) {
    LDBG("Warning: createpolicy instruction failed to match: " << instruction);
    return {};
  }

  // matches[0] is the entire instruction
  // matches[1] is the policy
  return matches[1];
}

void handleLoad(OpBuilder &builder, Location loc, StringRef instruction,
                LLVM::InlineAsmOp asmOp, llvm::StringMap<Value> &variableMap,
                llvm::DenseMap<mlir::Value, StringRef> &cachePolicy) {
  llvm::StringMap<int32_t> cachePolicyMap = {
      // David does not have evict normal,
      // so map it to -1 and use llvm::loadOp
      {"evict_normal", -1},
      {"evict_first", 0},
      {"evict_last", 1},
  };

  static const llvm::Regex ldMatcher(
      // --- ld op ---
      R"(ld\.)"
      R"(([^[:space:]]+))" // Group 1: load config
      R"([[:space:]]+)"

      // --- Destination registers ---
      R"(\{?)"
      R"(([[:space:]0-9\$,]+))" // Group 2: load dest registers
      R"(\}?)"
      R"(,[[:space:]]*)"

      // --- Source address [a] ---
      R"(\[[[:space:]]*)"
      R"((\$[0-9]+))"       // Group 3: load source address
      R"([[:space:]]*\+.*)" // TODO: handle non zero load offset
      R"(\])"

      // --- Optional cache-policy ---
      R"((,[[:space:]]*(\$[0-9]+))?)" // Group 5: cache policy register
                                      // (optional)
  );
  SmallVector<StringRef, 6> matches;

  if (!ldMatcher.match(instruction, &matches)) {
    LDBG("Warning: LD instrtion failed to match: " << instruction);
    return;
  }

  // matches[0] is the entire instruction
  // matches[1] is the load config
  // matches[2] is the load dest registers
  // matches[3] is the load source address
  // matches[4] is the surrounding group of matches[5], do not use
  // matches[5] is the cache policy register (optional)
  StringRef config = matches[1];
  SmallVector<StringRef> destRegisters;
  matches[2].split(destRegisters, ",", -1, false);
  Value addr = variableMap[matches[3]];
  StringRef cachePolicyReg = matches[5];

  int32_t cache_policy;
  if (cachePolicyReg.empty()) {
    cache_policy = -1;
  } else {
    StringRef cache_policy_str = cachePolicy[variableMap[cachePolicyReg]];
    auto it = cachePolicyMap.find(cache_policy_str);
    if (it == cachePolicyMap.end()) {
      LDBG("No mapping for '" << cache_policy_str << "'\n");
      return;
    }
    cache_policy = it->second;
  }

  if (cache_policy != -1 && !config.contains("global")) {
    LDBG("Warning: Inline ASM ld not handled: " << instruction);
    return;
  }

  // Result types
  Type elementType = parseTypeString(builder, config);
  Type loadResultType = elementType;

  // the type (e.g., b32) is after the last '.',
  int bits;
  config.substr(config.rfind(".") + 2).getAsInteger(10, bits);

  int64_t vec = 1;
  if (config.contains(".v2")) {
    vec = 2;
    loadResultType = VectorType::get({2}, elementType);
  } else if (config.contains(".v4")) {
    vec = 4;
    loadResultType = VectorType::get({4}, elementType);
  } else if (config.contains(".v8")) {
    vec = 8;
    loadResultType = VectorType::get({8}, elementType);
  }

  auto buildLoad = [&]() -> Value {
    if (cache_policy == -1) {
      return builder.create<LLVM::LoadOp>(loc, loadResultType, addr);
    } else {
      int64_t totalBits = vec * bits;

      llvm::SmallString<64> intrStr;
      llvm::raw_svector_ostream os(intrStr);
      os << "llvm.hivm.ldg.cache.s" << totalBits;

      Value l2cache = builder.create<LLVM::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(cache_policy));
      SmallVector<Value> args = {addr, l2cache};

      Value loadResult = builder
                             .create<LLVM::CallIntrinsicOp>(
                                 loc, builder.getIntegerType(totalBits),
                                 builder.getStringAttr(os.str()), args)
                             .getResult(0);

      loadResult =
          builder.create<LLVM::BitcastOp>(loc, loadResultType, loadResult)
              .getResult();

      return loadResult;
    }
  };

  // Check if the move is predicated
  llvm::Regex maskMatcher(R"(@!?(\$[0-9]+)[[:space:]]+.*)");
  SmallVector<StringRef, 4> masked;
  Value yieldedValue;
  if (maskMatcher.match(instruction, &masked)) {
    // Generate an if stmt; ensure the scf.if result type is exactly the
    // loadResultType so i8 loads remain i8.
    scf::IfOp ifOp = builder.create<scf::IfOp>(loc, loadResultType,
                                               variableMap[masked[1]], true);

    if (masked[0].contains("!")) {
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    } else
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    Value loadResult = buildLoad();
    builder.create<scf::YieldOp>(loc, loadResult);

    if (masked[0].contains("!")) {
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    } else
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

    // yield the old value of the dest registers — CAST old elements to the
    // load elementType when necessary so both branches match (prevents i16
    // else-branch when then-branch is i8).
    Value oldValue;
    if (VectorType vecRes = dyn_cast<VectorType>(loadResultType)) {
      SmallVector<Value> init;
      for (int i = 0; i < vecRes.getShape()[0]; i++) {
        Value oldElement = variableMap[destRegisters[i].trim()];
        if (!oldElement)
          oldElement = builder.create<LLVM::ConstantOp>(
              loc, elementType, builder.getIntegerAttr(elementType, 0));
        else if (oldElement.getType() != elementType)
          oldElement = castValueToType(builder, loc, oldElement, elementType);
        init.push_back(oldElement);
      }
      oldValue = createAndInitializeVector(builder, loc, loadResultType, init);
    } else {
      Value oldElement = variableMap[destRegisters[0].trim()];
      if (!oldElement) {
        oldElement = builder.create<LLVM::ConstantOp>(
            loc, elementType, builder.getIntegerAttr(elementType, 0));
      } else if (oldElement.getType() != elementType) {
        oldElement = castValueToType(builder, loc, oldElement, elementType);
      }
      oldValue = oldElement;
    }
    builder.create<scf::YieldOp>(loc, oldValue);

    builder.setInsertionPointAfter(ifOp);

    yieldedValue = ifOp.getResult(0);
  } else
    yieldedValue = buildLoad();

  // Assign the result to each register
  if (destRegisters.size() == 1) {
    variableMap[destRegisters[0].trim()] = yieldedValue;
    return;
  }
  for (size_t i = 0; i < destRegisters.size(); i++) {
    Value idx = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(),
                                                 builder.getI32IntegerAttr(i));
    variableMap[destRegisters[i].trim()] =
        builder.create<LLVM::ExtractElementOp>(loc, elementType, yieldedValue,
                                               idx);
  }
}

void handleStore(OpBuilder &builder, Location loc, StringRef instruction,
                 llvm::StringMap<Value> &variableMap) {
  // TODO: handle non zero load offset
  llvm::Regex stMatcher(
      R"(st\.([^[:space:]]+)[[:space:]]+\[[[:space:]]*(\$[0-9]+)[[:space:]]*\+.*\],[[:space:]]*\{?([[:space:]0-9\$,]+)\}?.*)");
  SmallVector<StringRef, 4> matches;

  if (!stMatcher.match(instruction, &matches)) {
    LDBG("Warning: ST instrtion failed to match: " << instruction);
    return;
  }
  // matches[0] is the entire instruction
  // matches[1] is the store config
  // matches[2] is the store address
  // matches[3] is the store source registers
  SmallVector<StringRef> srcRegisters;
  // split the inline asm with the new line separator
  matches[3].split(srcRegisters, ",", -1, false);

  // Result types
  Type elementType = parseTypeString(builder, matches[1]);
  Type storeSourceType = elementType;

  if (matches[1].contains(".v2")) {
    storeSourceType = VectorType::get({2}, elementType);
  } else if (matches[1].contains(".v4")) {
    storeSourceType = VectorType::get({4}, elementType);
  }

  // The value to be stored
  Value toBeStored;
  if (VectorType vecRes = dyn_cast<VectorType>(storeSourceType)) {
    SmallVector<Value> init;
    for (int i = 0; i < vecRes.getShape()[0]; i++) {
      Value srcVal = variableMap[srcRegisters[i].trim()];
      if (!srcVal) {
        srcVal = builder.create<LLVM::ConstantOp>(
            loc, elementType, builder.getIntegerAttr(elementType, 0));
      } else if (srcVal.getType() != elementType) {
        srcVal = castValueToType(builder, loc, srcVal, elementType);
      }
      init.push_back(srcVal);
    }
    toBeStored = createAndInitializeVector(builder, loc, storeSourceType, init);
  } else {
    Value srcVal = variableMap[srcRegisters[0].trim()];
    if (!srcVal) {
      srcVal = builder.create<LLVM::ConstantOp>(
          loc, elementType, builder.getIntegerAttr(elementType, 0));
    } else if (srcVal.getType() != elementType) {
      srcVal = castValueToType(builder, loc, srcVal, elementType);
    }
    toBeStored = srcVal;
  }

  // Check if the move is predicated
  llvm::Regex maskMatcher(R"(@!?(\$[0-9]+)[[:space:]]+.*)");
  SmallVector<StringRef, 4> masked;
  if (maskMatcher.match(instruction, &masked)) {
    // Generate an if stmt
    scf::IfOp ifOp =
        builder.create<scf::IfOp>(loc, variableMap[masked[1]], true);

    if (masked[0].contains("!")) {
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    } else
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // store the values, as a vector if it's vectorized
    builder.create<LLVM::StoreOp>(loc, toBeStored, variableMap[matches[2]]);

    if (masked[0].contains("!")) {
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    } else
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

    builder.setInsertionPointAfter(ifOp);
  } else {
    builder.create<LLVM::StoreOp>(loc, toBeStored, variableMap[matches[2]]);
  }
}

Value castToRawBits(Value val, OpBuilder &builder) {
  Type ty = val.getType();
  Location loc = val.getLoc();

  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    // Already correct width, signless is already the MLIR integer standard
    if (intTy.isSignless())
      return val;

    // Create signless replacement
    auto newTy = IntegerType::get(builder.getContext(), intTy.getWidth());
    return builder.create<arith::BitcastOp>(loc, newTy, val);
  }

  if (auto floatTy = dyn_cast<FloatType>(ty)) {
    // Convert float to a signless integer type with same width
    auto newTy = IntegerType::get(builder.getContext(), floatTy.getWidth());
    return builder.create<arith::BitcastOp>(loc, newTy, val);
  }

  llvm_unreachable("Unsupported type in castToRawBits");
}

Value castToElementType(OpBuilder &b, Location loc, Value val,
                        Type elementType) {
  Type srcTy = val.getType();

  // Already correct
  if (srcTy == elementType)
    return val;

  unsigned srcW = srcTy.getIntOrFloatBitWidth();

  if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(elementType)) {
    int64_t length = vecTy.getDimSize(0);
    assert(length == 2 && "Only support <2x*16> type");
 
    auto elemTy = vecTy.getElementType();
    unsigned dstW = elemTy.getIntOrFloatBitWidth() * 2;
 
    assert(srcW == dstW && "Atomic operand cast must preserve bit width");

    return b.create<LLVM::BitcastOp>(loc, vecTy, val);
  } else {
    unsigned dstW = elementType.getIntOrFloatBitWidth();
    // Width must match for reinterpretation
    assert(srcW == dstW && "Atomic operand cast must preserve bit width");
 
    // Only legal conversions: reinterpretation
    return b.create<arith::BitcastOp>(loc, elementType, val);
  }
}

void handleAtom(OpBuilder &builder, Location loc, StringRef instruction,
                llvm::StringMap<Value> &variableMap) {

  llvm::StringMap<llvm::StringRef> kindMap = {
      {"add", "ADD"}, {"max", "MAX"}, {"min", "MIN"},   {"and", "AND"},
      {"or", "OR"},   {"xor", "XOR"}, {"exch", "EXCH"}, {"cas", "CAS"},
  };

  llvm::StringMap<llvm::StringRef> spaceMap = {
      {"global", "G"},
      {"shared", "S"},
  };

  llvm::StringMap<llvm::StringRef> typeMap = {
      {"b16", "u16"}, {"b32", "u32"}, {"u32", "u32"},  {"s32", "s32"},
      {"b64", "u64"}, {"u64", "u64"}, {"f16", "fp16"}, {"f32", "fp32"},
      {"f16x2", "f16x2"}, {"bf16x2","bf16x2"},
  };

  // clang-format off
  // atom.<space>.<scope>.<semantics>.<operation>(.<ftz|noftz>)?.<type> dst, [address + offset], src(, src2)?
  // - <space>:   global, shared, etc.
  // - <scope>:   cta, cluster, gpu, sys (ordering visibility domain)
  // - <semantics>: relaxed, acquire, release, acq_rel (memory ordering)
  // - <operation>: add, cas, exch, min, max, and, or, xor, etc.
  // - <ftz|noftz>: (optional) floating-point flush-to-zero modifier (e.g., for f16 atomics)
  // - <type>:     u32, b32, f16, f32, etc.
  //
  // Examples:
  //   atom.global.gpu.acq_rel.add.u32 %r0, [%rd1], %r2
  //   atom.global.gpu.acq_rel.cas.b32 %r0, [%rd1], %r2, %r3
  //   atom.global.gpu.acq_rel.add.noftz.f16 %h0, [%rd1], %h2
  // clang-format on
  llvm::Regex atomMatcher(
      R"(atom\.([^[:space:]]+)[[:space:]]+(\$[0-9]+)[[:space:]]*,[[:space:]]*\[[[:space:]]*(\$[0-9]+)([[:space:]]*\+[^]]+)?[[:space:]]*\][[:space:]]*,[[:space:]]*(\$[0-9]+)(,[[:space:]]*(\$[0-9]+))?.*)");

  SmallVector<StringRef, 8> matches;
  if (!atomMatcher.match(instruction, &matches)) {
    LDBG("ATOM instruction failed to match: " << instruction);
    llvm_unreachable("Unexpected ATOM instruction");
  }

  // matches:
  // [1] = full op config (e.g., "global.gpu.acq_rel.add.u32")
  // [2] = destination register ($0)
  // [3] = address register ($1)
  // [4] = "+ 0";  // TODO support non-zero values
  // [5] = first source operand ($2)
  // [6] = ", $3"
  // [7] = optional second source operand ($3)
  StringRef atomConfig = matches[1];
  StringRef dstReg = matches[2].trim();
  StringRef addrReg = matches[3].trim();
  StringRef srcReg1 = matches[5].trim();
  StringRef srcReg2 = matches.size() > 7 ? matches[7].trim() : "";

  // Split the config: space, scope, semantics, operation, (ftz/noftz)?, type
  SmallVector<StringRef, 8> parts;
  atomConfig.split(parts, '.');
  assert(parts.size() >= 5);
  StringRef space = parts[0];
  StringRef operation = parts[3];
  // ftz/noftz is not supported, pray for the best
  StringRef type = (parts.size() == 6) ? parts[5] : parts[4];
  PTXAtomicInfo ptxAtomInfo = {operation, space, type};

  AscendAtomicInfo ascendAtomInfo;
  ascendAtomInfo.kind = lookupOrError(kindMap, ptxAtomInfo.kind);
  ascendAtomInfo.memorySpace = lookupOrError(spaceMap, ptxAtomInfo.memorySpace);
  ascendAtomInfo.type = lookupOrError(typeMap, ptxAtomInfo.type);

  if (ascendAtomInfo.kind == "CAS" && ascendAtomInfo.type.ends_with("16")) {
    LDBG("A5 does not support 16 bit CAS atomic" << instruction);
    llvm_unreachable("A5 does not support 16 bit CAS atomic");
  }

  llvm::SmallString<64> IntrinsicName("");
  llvm::raw_svector_ostream os(IntrinsicName);
  os << "llvm.hivm.atom." << ascendAtomInfo.kind << "."
     << ascendAtomInfo.memorySpace << "." << ascendAtomInfo.type;

  LDBG("IntrinsicName " << IntrinsicName);

  Type elementType = parseTypeString(builder, ascendAtomInfo.type);
  if (!elementType) {
    LDBG("Unknown atomic type: " << ascendAtomInfo.type);
    llvm_unreachable("Unexpected atomic type encountered");
  }

  llvm::Regex maskMatcher(R"(@!?(\$[0-9]+)[[:space:]]+.*)");
  SmallVector<StringRef, 4> masked;
  bool isPredicated = maskMatcher.match(instruction, &masked);
  bool isNegated = isPredicated && masked[0].contains("!");

  auto createAtomic = [&](OpBuilder &b) -> Value {
    Value ptr = variableMap[addrReg];
    Value val1 = variableMap[srcReg1];
    Value val2 = variableMap[srcReg2];
    if (!ptr || !val1 || (!srcReg2.empty() && !val2)) {
      LDBG("Missing pointer or value for atomic: "
           << addrReg << " / " << srcReg1 << " / " << srcReg2);
      llvm_unreachable("Missing pointer or value for atomic");
    }

    Value l2cache = builder.create<LLVM::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(0));

    // EXCH/CAS accept only u32/u64 (raw bits)
    if (ascendAtomInfo.kind == "EXCH") {
      val1 = castToRawBits(val1, b);
    } else if (ascendAtomInfo.kind == "CAS") {
      val1 = castToRawBits(val1, b);
      val2 = castToRawBits(val2, b);
    }

    SmallVector<Value> args = {ptr, val1};
    if (!srcReg2.empty())
      args.push_back(val2);
    args.push_back(l2cache);

    auto call = builder.create<LLVM::CallIntrinsicOp>(
        loc, TypeRange{elementType}, IntrinsicName, args);

    return call.getResult(0);
  };

  Value yieldedValue;
  if (isPredicated) {
    scf::IfOp ifOp = builder.create<scf::IfOp>(loc, elementType,
                                               variableMap[masked[1]], true);

    if (isNegated)
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    else
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    Value result = createAtomic(builder);
    builder.create<scf::YieldOp>(loc, result);

    if (isNegated)
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    else
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

    // Yield previous value if predication fails
    Value oldValue = variableMap[dstReg];
    if (!oldValue)
      llvm_unreachable("Cannot find previous value");
    oldValue = castToElementType(builder, loc, oldValue, elementType);
    builder.create<scf::YieldOp>(loc, oldValue);

    builder.setInsertionPointAfter(ifOp);
    yieldedValue = ifOp->getResult(0);
  } else {
    yieldedValue = createAtomic(builder);
  }

  // Assign result (atomics return the *old* value)
  variableMap[dstReg] = yieldedValue;
}

Value translateInlineASM(OpBuilder &builder, Location loc,
                         LLVM::InlineAsmOp asmOp,
                         llvm::DenseMap<mlir::Value, StringRef> &cachePolicy) {
  StringRef name = asmOp.getAsmString();
  SmallVector<StringRef> instructions;
  // split the inline asm with the new line separator
  name.split(instructions, ";\0A\09", -1, false);

  StringRef constraint = asmOp.getConstraints();
  SmallVector<StringRef> constraintVariables;
  // split the inline asm constraint
  constraint.split(constraintVariables, ",", -1, false);

  // Initialize all variables in the inline asm's constraints
  llvm::StringMap<Value> variableMap;
  SmallVector<std::string> resultRegisters;
  int operandIndex = 0;
  for (size_t i = 0; i < constraintVariables.size(); i++) {
    if (constraintVariables[i].contains("=")) {
      resultRegisters.push_back("$" + std::to_string(i));
      variableMap[StringRef("$" + std::to_string(i))] = nullptr;
    } else {
      variableMap[StringRef("$" + std::to_string(i))] =
          asmOp->getOperand(operandIndex);
      operandIndex++;
    }
  }

  for (StringRef instruction : instructions) {
    if (instruction.contains("mov."))
      handleMov(builder, loc, variableMap, instruction);
    else if (instruction.contains("createpolicy.")) {
      // this has to be before .contains("st.")
      // because of createpolicy.fractional.L2::evict_last.b64
      if (auto policy = handleCreatePolicy(instruction))
        cachePolicy[asmOp.getResult(0)] = *policy;
      return asmOp.getResult(0);
    } else if (instruction.contains("ld."))
      handleLoad(builder, loc, instruction, asmOp, variableMap, cachePolicy);
    else if (instruction.contains("st."))
      handleStore(builder, loc, instruction, variableMap);
    else if (instruction.contains("atom."))
      handleAtom(builder, loc, instruction, variableMap);
    else {
      LDBG("Warning: Inline ASM not handled: " << instruction);
      return asmOp.getResult(0);
    }
  }

  Value finalValue = nullptr;
  if (resultRegisters.size() > 1) {
    Value structVal =
        builder.create<LLVM::UndefOp>(loc, asmOp->getResultTypes()[0]);

    for (unsigned i = 0; i < resultRegisters.size(); ++i) {
      SmallVector<int64_t, 1> indices = {static_cast<int64_t>(i)};
      structVal = builder.create<LLVM::InsertValueOp>(
          loc, asmOp->getResultTypes()[0], structVal,
          variableMap[resultRegisters[i]], indices);
    }

    finalValue = structVal;
  } else if (resultRegisters.size() == 1)
    finalValue = variableMap[resultRegisters[0]];

  return finalValue;
}

void handleLdSt(StringRef inst, llvm::SmallString<64> &newAsmStr) {
  // TODO: use llvm regex and not convert to std string
  const std::regex pattern(
      R"((^\s*(?:(@\$?\d+)\s+)?ld\.(global|shared)(?:\.v(\d+))?\.b(\d+)\s*(?:(?:\s*\{\s*([^\}]+)\s*\}\s*,\s*\[\s*([^\]]+)\s*\])|(?:\s*(\S+)\s*,\s*\[\s*([^\]]+)\s*\])))|(^\s*(?:(@\$?\d+)\s+)?st\.(global|shared)(?:\.v(\d+))?\.b(\d+)\s*(?:(?:\s*\[\s*([^\]]+)\s*\]\s*,\s*\{\s*([^\}]+)\s*\})|(?:\s*\[\s*([^\]]+)\s*\]\s*,\s*(\S+)))))");
  std::string instStr = inst.str();
  std::smatch match;
  AsmInfo info;
  if (!std::regex_search(instStr, match, pattern)) {
    LDBG("Problem mapping inline asm instruction: " << inst);
    // llvm::report_fatal_error("Cannot match the regex");
  }

  if (match[1].matched) { // ld matched
    info.predicate = match[2];
    info.op = "ld";
    info.space = match[3];
    info.vec = match[4].matched ? std::stoi(match[4]) : 1;
    info.bits = std::stoi(match[5]);
    if (match[6].matched && match[7].matched) { // ld vector
      info.regs = match[6];
      info.mem = match[7];
    } else if (match[8].matched && match[9].matched) { // ld scalar
      info.regs = match[8];
      info.mem = match[9];
    } else {
      llvm::report_fatal_error("Cannot determine load/store operands");
    }
  } else if (match[10].matched) { // st matched
    info.predicate = match[11];
    info.op = "st";
    info.space = match[12];
    info.vec = match[13].matched ? std::stoi(match[13]) : 1;
    info.bits = std::stoi(match[14]);
    if (match[15].matched && match[16].matched) { // st vector
      info.mem = match[15];
      info.regs = match[16];
    } else if (match[17].matched && match[18].matched) { // st scalar
      info.mem = match[17];
      info.regs = match[18];
    } else {
      llvm::report_fatal_error("Cannot determine load/store operands");
    }
  } else {
    LDBG("Problem mapping inline asm instruction: " << inst);
    // llvm::report_fatal_error("Cannot determine load/store operands");
  }
  info.totalBits = info.vec * info.bits;

  const std::map<std::pair<std::string, std::string>, std::string> mapping = {
      {{"ld", "global"}, "LDG"},
      {{"st", "global"}, "STG"},
      {{"ld", "shared"}, "LDS"},
      {{"st", "shared"}, "STS"}};

  auto it = mapping.find({info.op, info.space});
  if (it == mapping.end()) {
    LDBG("Problem mapping inline asm instruction: " << inst);
    // llvm::report_fatal_error("Cannot remap the operation");
  }

  std::string firstReg = info.regs.substr(0, info.regs.find(','));
  std::string type = (info.op == "ld" || info.totalBits >= 32) ? "b" : "u";
  std::string addC =
      (info.space == "global" && info.totalBits >= 32) ? "C.NMFV." : "";

  llvm::raw_svector_ostream os(newAsmStr);
  os << it->second << "." << addC << type << info.totalBits << "\t";
  os << firstReg << ", [" << info.mem << "] "
     << " ws:7 rs:7 wait:0b0000000 stall:15 " << info.predicate;
}

void processAsmStr(StringRef oldAsm, llvm::SmallString<64> &newAsmStr) {
  SmallVector<llvm::StringRef, 8> instructions;
  llvm::raw_svector_ostream os(newAsmStr);
  oldAsm.split(instructions, ';', -1, false);

  for (size_t i = 0; i < instructions.size(); ++i) {
    StringRef inst = instructions[i];
    StringRef trimmed = inst.trim();

    bool isLast = (i == instructions.size() - 1);

    if (trimmed.starts_with("mov.")) {
      handleMov(trimmed, newAsmStr);
    } else {
      handleLdSt(trimmed, newAsmStr);
    }

    if (!isLast) {
      os << ";\n\t";
    }
  }
}

void processAsmConstraint(StringRef oldAsmConstraint,
                          llvm::SmallString<16> &newConstraint) {
  bool ampAdded = false;

  for (char c : oldAsmConstraint) {
    char upper = std::toupper(c);

    if (!ampAdded && upper == '=') {
      newConstraint.push_back('=');
      newConstraint.push_back('&');
      ampAdded = true;
    } else {
      newConstraint.push_back(upper);
    }
  }
}

void replaceInlineAsm(LLVM::LLVMFuncOp funcOp, bool ifElse) {
  llvm::DenseMap<mlir::Value, StringRef> cachePolicy;
  funcOp.walk([&](LLVM::InlineAsmOp asmOp) {
    OpBuilder builder(asmOp);
    StringRef name = asmOp.getAsmString();
    StringRef constraint = asmOp.getConstraints();
    size_t opPos = name.find("atom");
    llvm::SmallString<64> newAsmStr("");
    llvm::SmallString<16> newConstraint("");
    auto operands = asmOp->getOperands();
    OpResult oldRes = asmOp->getResult(0);
    Location loc = asmOp->getLoc();

    if (ifElse) {
      Value converted =
          translateInlineASM(builder, asmOp->getLoc(), asmOp, cachePolicy);

      if (cachePolicy.count(converted))
        return;

      Value finalValue = converted;
      if (!converted) {
        oldRes.dropAllUses();
        asmOp.erase();
        return;
      }

      oldRes.replaceAllUsesWith(finalValue);
      asmOp.erase();
      return;
    }

    if (opPos != StringRef::npos) {
      constructAtomicAsmString(name, opPos, newAsmStr, newConstraint);
    } else {
      processAsmStr(name, newAsmStr);
      processAsmConstraint(constraint, newConstraint);
    }

    builder.create<hivm_regbaseintrins::SyncThreadsOp>(asmOp->getLoc());
    auto newVal = builder.create<LLVM::InlineAsmOp>(
        loc, oldRes.getType(), operands, newAsmStr, newConstraint,
        asmOp.getHasSideEffects(), asmOp.getIsAlignStack(),
        asmOp.getAsmDialectAttr(), asmOp.getOperandAttrsAttr());
    builder.create<hivm_regbaseintrins::SyncThreadsOp>(asmOp->getLoc());

    oldRes.replaceAllUsesWith(newVal.getResult(0));
    asmOp.erase();
  });

  for (auto &[value, _] : cachePolicy) {
    if (value.use_empty())
      value.getDefiningOp()->erase();
  }
}

class AtomicRMWToIntrinsicPattern : public OpRewritePattern<LLVM::AtomicRMWOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op.getContext();

    // Build intrinsic name from type + kind + syncscope
    static const llvm::DenseMap<LLVM::AtomicBinOp, llvm::StringRef> kindMap = {
        {LLVM::AtomicBinOp::xchg, "EXCH"}, {LLVM::AtomicBinOp::add, "ADD"},
        {LLVM::AtomicBinOp::_and, "AND"},  {LLVM::AtomicBinOp::_or, "OR"},
        {LLVM::AtomicBinOp::_xor, "XOR"},  {LLVM::AtomicBinOp::max, "MAX"},
        {LLVM::AtomicBinOp::min, "MIN"},   {LLVM::AtomicBinOp::umax, "MAX"},
        {LLVM::AtomicBinOp::umin, "MIN"},  {LLVM::AtomicBinOp::fadd, "ADD"},
        {LLVM::AtomicBinOp::fsub, "SUB"},  {LLVM::AtomicBinOp::fmax, "MAX"},
        {LLVM::AtomicBinOp::fmin, "MIN"},
    };

    static const llvm::StringMap<llvm::StringRef> syncscopeToSpace = {
        {"device", "G"},
        {"cta", "S"},
    };

    mlir::Type valType = op.getVal().getType();
    llvm::DenseMap<Type, llvm::StringRef> typeMap = {
        {IntegerType::get(ctx, 32, IntegerType::Unsigned), "u32"},
        {IntegerType::get(ctx, 32, IntegerType::Signed), "s32"},
        {Float16Type::get(ctx), "f16"},
        {BFloat16Type::get(ctx), "bf16"},
        {FloatType::getF32(ctx), "f32"},
        {FloatType::getF64(ctx), "f64"},
    };

    llvm::StringRef syncscopeStr = "";
    if (auto ssAttr = op.getSyncscope())
      syncscopeStr = *ssAttr;

    llvm::SmallString<64> intrinsicName;
    llvm::raw_svector_ostream os(intrinsicName);
    os << "llvm.hivm.atom." << kindMap.lookup(op.getBinOp()) << "."
       << syncscopeToSpace.lookup(syncscopeStr) << "."
       << typeMap.lookup(valType);

    Value l2cache = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    SmallVector<Value> args = {op.getOperand(0), op.getOperand(1), l2cache};

    // Create call intrinsic
    rewriter.setInsertionPoint(op);
    Value call = rewriter
                     .create<LLVM::CallIntrinsicOp>(
                         op.getLoc(), TypeRange{valType},
                         rewriter.getStringAttr(os.str()), args)
                     .getResult(0);

    // Optionally ensure it is not DCE-dropped
    assert(call.getDefiningOp() != nullptr);
    call.getDefiningOp()->setAttr("llvm.sideeffect", rewriter.getUnitAttr());

    // Replace or drop dead result
    if (!op.getResult().use_empty())
      rewriter.replaceOp(op, call);
    else
      rewriter.eraseOp(op);

    return success();
  }
};

} // namespace

class TritonRemapPass
    : public bishengir::triton::impl::TritonRemapBase<TritonRemapPass> {
public:
  explicit TritonRemapPass(const bishengir::TritonRemapOptions &options)
      : TritonRemapBase(options) {}

  void runOnOperation() override {
    Operation *op = getOperation();
    auto *context = &getContext();
    SymbolTable symbolTable(op);
    removeAssertFailOps(op);

    op->walk([&](Operation *nestedOp) {
      if (auto funcOp = llvm::dyn_cast<LLVM::LLVMFuncOp>(nestedOp)) {
        if (funcOp->hasAttr("nvvm.kernel")) {
          FuncInfo originalFuncInfo{funcOp.getFunctionType(),
                                    funcOp.getSymNameAttr(),
                                    funcOp.getArgAttrsAttr()};
          std::string newName = originalFuncInfo.funcName.str() + "_vf_simt";
          StringAttr newNameAttr =
              StringAttr::get(funcOp.getContext(), newName);
          if (llvm::failed(symbolTable.rename(funcOp, newNameAttr))) {
            funcOp->emitError()
                << "Failed to rename function from "
                << originalFuncInfo.funcName.str() << " to " << newName;
          }

          rewriteSIMTKernelFunction(funcOp, context);
          createLaunchFunction(funcOp, context, originalFuncInfo);
          pruneUnusedSIMTArgs(funcOp);
        }
      }
      if (auto globalOp = llvm::dyn_cast<LLVM::GlobalOp>(nestedOp)) {
        globalOp->erase();
      }
    });
  }
  void populateRewritePatterns(RewritePatternSet &patterns) {
    patterns.add<AtomicRMWToIntrinsicPattern>(patterns.getContext());
    patterns.add<PackRmFromShflSyncPattern>(patterns.getContext());
  }

private:
  void removeAssertFailOps(Operation *op) {
    op->walk([&](LLVM::CallOp callOp) {
      if (auto callee = callOp.getCallee()) {
        if (*callee == "__assertfail") {
          callOp.erase();
        }
      }
    });
  }

  void pruneUnusedSIMTArgs(LLVM::LLVMFuncOp funcOp) {
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return;

    Block &entryBlock = funcOp.getBody().front();
    unsigned originalVFArgCount = entryBlock.getNumArguments();

    // find unused VF arguments
    SmallVector<int> unusedArgumentInd;
    for (BlockArgument bArg : entryBlock.getArguments()) {
      if (bArg.use_empty()) {
        unusedArgumentInd.push_back(bArg.getArgNumber());
      }
    }
    if (unusedArgumentInd.empty()) {
      return;
    }

    // update call sites to remove operands corresponding to unused args
    auto callSites = funcOp.getSymbolUses(module);
    if (callSites.has_value()) {
      for (auto &symUse : callSites.value()) {
        Operation *userOp = symUse.getUser();
        if (!userOp)
          continue;

        SmallVector<Value> operands(userOp->operand_begin(),
                                    userOp->operand_end());
        SmallVector<Value> newOperands;

        if (auto launch = dyn_cast<hivm_regbaseintrins::LaunchFuncOp>(userOp)) {
          const unsigned launchHead = 3;
          for (unsigned i = 0;
               i < std::min<unsigned>(launchHead, operands.size()); ++i)
            newOperands.push_back(operands[i]);

          for (unsigned opIdx = launchHead; opIdx < operands.size(); ++opIdx) {
            unsigned argIndex = opIdx - launchHead;
            if (argIndex < originalVFArgCount &&
                llvm::is_contained(unusedArgumentInd, (int)argIndex)) {
              continue;
            }
            newOperands.push_back(operands[opIdx]);
          }
          userOp->setOperands(newOperands);
        } else {
          for (unsigned i = 0; i < operands.size(); ++i) {
            if (i < originalVFArgCount &&
                llvm::is_contained(unusedArgumentInd, (int)i)) {
              continue;
            }
            newOperands.push_back(operands[i]);
          }
          userOp->setOperands(newOperands);
        }
      }
    }

    // erase unused VF argument
    entryBlock.eraseArguments([&](BlockArgument bArg) {
      return llvm::is_contained(unusedArgumentInd, (int)bArg.getArgNumber());
    });

    // reebuild VF function type
    auto oldFuncType = funcOp.getFunctionType();
    SmallVector<Type> keptParamTypes;
    auto oldParamTypes = oldFuncType.getParams();
    for (unsigned i = 0; i < oldParamTypes.size(); ++i) {
      if (!llvm::is_contained(unusedArgumentInd, (int)i))
        keptParamTypes.push_back(oldParamTypes[i]);
    }
    auto newFuncType = LLVM::LLVMFunctionType::get(oldFuncType.getReturnType(),
                                                   keptParamTypes, false);
    funcOp.setType(newFuncType);
  }

  void rewriteSIMTKernelFunction(LLVM::LLVMFuncOp funcOp,
                                 MLIRContext *context) {
    convertGridIntrinsicsToArgs(funcOp, context);
    replaceIntrinsicsWithOps(funcOp, intrinsicReplacementMap);
    replaceLLVMCallWithOps(funcOp, llvmCallReplacementMap);
    replacePtxToIntrins(funcOp);
    replaceInlineAsm(funcOp, useIfElse);
    RewritePatternSet patterns(context);
    populateRewritePatterns(patterns);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));

    funcOp.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace() == "nvvm") {
        LDBG("cannot find replacement for " << op->getName().getStringRef());
      }
    });
  }

  using GridDims = std::tuple<Value, Value, Value>;
  GridDims getGridDims(LLVM::LLVMFuncOp funcOp, OpBuilder builder,
                       Location loc) {
    OpBuilder::InsertionGuard g(builder);
    Value gridX, gridY, gridZ;
    Type int64Ty = builder.getI64Type();
    for (auto [idx, value] : llvm::enumerate(funcOp.getArguments())) {
      auto dictAttr = funcOp.getArgAttrDict(idx);
      if (!dictAttr)
        continue;

      Attribute attr = dictAttr.get(gpu::GPUBlockMappingAttr::name);
      auto blockMappingAttr = dyn_cast_or_null<gpu::GPUBlockMappingAttr>(attr);
      if (!blockMappingAttr)
        continue;

      gpu::MappingId mapping = blockMappingAttr.getBlock();
      Value gridDim = value;
      // Target type is i64, cast if needed
      if (value.getType() != int64Ty) {
        gridDim = castValueToType(builder, loc, value, int64Ty);
      }
      switch (mapping) {
      case gpu::MappingId::DimX:
        gridX = gridDim;
        break;
      case gpu::MappingId::DimY:
        gridY = gridDim;
        break;
      case gpu::MappingId::DimZ:
        gridZ = gridDim;
        break;
      default:
        llvm_unreachable("Unknown mapping");
      }
    }
    // If the grid dims in the function is invalid, or the user specified that
    // we should use the flag.
    if ((!gridX || !gridY || !gridZ) || useGridFlag) {
      LDBG("Falling back to use compile options in remapper: gridDimX="
           << gridDimX << ", gridDimY=" << gridDimY
           << ", gridDimZ=" << gridDimZ);
      gridX = builder.create<LLVM::ConstantOp>(
          loc, int64Ty, builder.getI64IntegerAttr(gridDimX));
      gridY = builder.create<LLVM::ConstantOp>(
          loc, int64Ty, builder.getI64IntegerAttr(gridDimY));
      gridZ = builder.create<LLVM::ConstantOp>(
          loc, int64Ty, builder.getI64IntegerAttr(gridDimZ));
    }
    return {gridX, gridY, gridZ};
  }

  void createLaunchFunction(LLVM::LLVMFuncOp vf, MLIRContext *context,
                            const FuncInfo &originalFuncInfo) {
    OpBuilder builder(context);
    builder.setInsertionPoint(vf);
    auto loc = vf.getLoc();
    auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, originalFuncInfo.funcName, originalFuncInfo.funcType);
    // Forward argument attributes
    wrapperFunc.setArgAttrsAttr(originalFuncInfo.funcArgAttr);
    auto kernelAttr =
        StringAttr::get(context, hivm_regbaseintrins::kDavinciKernelAttrName);
    wrapperFunc->setAttr(kernelAttr, builder.getUnitAttr());
    auto targetAttr =
        StringAttr::get(context, hivm_regbaseintrins::kDavinciTargetAttrName);
    wrapperFunc->setAttr(targetAttr, hivm_regbaseintrins::SIMT_TargetAttr::get(
                                         context, "dav-c310"));
    auto entryAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
        hacc::HACCToLLVMIRTranslateAttr::ENTRY);
    wrapperFunc->setAttr(entryAttr, builder.getUnitAttr());
    wrapperFunc->setAttr(
        hacc::HACCFuncTypeAttr::name,
        hacc::HACCFuncTypeAttr::get(context, hacc::HACCFuncType::DEVICE));
    Block *entryBlock = wrapperFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);
    populateEntry(vf, context, builder, loc, wrapperFunc);
  }

  void populateEntry(LLVM::LLVMFuncOp funcOp, MLIRContext *context,
                     OpBuilder builder, Location loc,
                     LLVM::LLVMFuncOp wrapperFuncOp) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return;
    }

    int64_t numWarp = -1;
    if (auto numWarpAttr = moduleOp->getAttr("ttg.num-warps")) {
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(numWarpAttr)) {
        numWarp = intAttr.getInt();
      }
    }

    int64_t numThreadPerWarp = -1;
    if (auto numThreadAttr = moduleOp->getAttr("ttg.threads-per-warp")) {
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(numThreadAttr)) {
        numThreadPerWarp = intAttr.getInt();
      }
    }

    if (numWarp <= 0 || numThreadPerWarp <= 0) {
      llvm::report_fatal_error(
          "Cannot determine num-warps or threads-per-warp! num-warps: " +
          Twine(numWarp) + ", threads-per-warp: " + Twine(numThreadPerWarp));
    }

    int64_t vfLaunchBound = numWarp * numThreadPerWarp;
    constexpr int64_t VF_LAUNCH_BOUND_THRESHOLD = 2048;
    if (vfLaunchBound > VF_LAUNCH_BOUND_THRESHOLD) {
      llvm::report_fatal_error(
          "vf LAUNCH_BOUND exceeds maximum number: " + Twine(vfLaunchBound) +
          " > " + Twine(VF_LAUNCH_BOUND_THRESHOLD));
    } else if (vfLaunchBound == VF_LAUNCH_BOUND_THRESHOLD) {
      LDBG("vfLaunchBound of " << VF_LAUNCH_BOUND_THRESHOLD
                               << " may trigger too much register spilling!");
    }

    funcOp->setAttr(hivm_regbaseintrins::kDavinciCallingConvAttrName,
                    hivm_regbaseintrins::SIMT_EntryAttr::get(
                        funcOp->getContext(), vfLaunchBound));

    SmallVector<Value> args = {wrapperFuncOp.getArguments().begin(),
                               wrapperFuncOp.getArguments().end()};
    Type int64Ty = IntegerType::get(context, 64);
    // launch block sizes (threads per block)
    Value blockX = builder.create<LLVM::ConstantOp>(
        loc, int64Ty, builder.getI64IntegerAttr(vfLaunchBound));
    Value blockY = builder.create<LLVM::ConstantOp>(
        loc, int64Ty, builder.getI64IntegerAttr(1));
    Value blockZ = builder.create<LLVM::ConstantOp>(
        loc, int64Ty, builder.getI64IntegerAttr(1));

    Value newIdx = builder.create<GetBlockIdxInstrOp>(loc, int64Ty);
    auto [gridX, gridY, gridZ] = getGridDims(wrapperFuncOp, builder, loc);
    // get grid x,y,z from linear grid id using x
    // px = pid % Gx
    // tmp = pid / Gx
    // py = tmp % Gy
    // pz = tmp / Gy
    Value tmp0 = newIdx;
    Value px = builder.create<LLVM::URemOp>(loc, int64Ty, tmp0, gridX);
    Value tmp1 = builder.create<LLVM::UDivOp>(loc, int64Ty, tmp0, gridX);
    Value py = builder.create<LLVM::URemOp>(loc, int64Ty, tmp1, gridY);
    Value tmp2 = builder.create<LLVM::UDivOp>(loc, int64Ty, tmp1, gridY);
    Value pz = tmp2;

    auto ub = triton::util::allocateSharedMemory(wrapperFuncOp, builder, loc);
    // always append six i64 grid-related args (nctaid.x,y,z then
    // ctaid.x,y,z) so it can easily support both flag and arg way of giving
    // the grid
    args.push_back(gridX);
    args.push_back(gridY);
    args.push_back(gridZ);
    args.push_back(px);
    args.push_back(py);
    args.push_back(pz);
    args.push_back(ub);
    builder.create<hivm_regbaseintrins::LaunchFuncOp>(
        funcOp->getLoc(), SymbolRefAttr::get(funcOp), blockX, blockY, blockZ,
        args);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }
};

std::unique_ptr<mlir::Pass> bishengir::triton::createTritonRemapPass(
    const bishengir::TritonRemapOptions &options) {
  return std::make_unique<TritonRemapPass>(options);
}

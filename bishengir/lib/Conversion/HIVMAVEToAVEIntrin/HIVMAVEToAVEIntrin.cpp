//===- HIVMAVEToAVEIntrin.cpp - Conversion from HIVMAVE to AVEIntrins ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace mlir {
#define GEN_PASS_DEF_CONVERTHIVMAVETOAVEINTRIN
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivmave;
using namespace mlir::hivm_regbaseintrins;

template <typename TargetTy, typename... Args>
inline Value createTargetTyIfValid(OpBuilder &builder, const Location &loc,
                                   bool condition, Args &&...args) {
  if constexpr (!std::is_same_v<TargetTy, VunsupportedBinaryInstrOp> &&
                !std::is_same_v<TargetTy, VunsupportedUnaryInstrOp>) {
    if (condition)
      return builder.create<TargetTy>(loc, std::forward<Args>(args)...);
  }
  return nullptr;
}

template <typename IntrV128F16OpTy, typename IntrV128S16OpTy,
          typename IntrV128U16OpTy, typename IntrV256S8OpTy,
          typename IntrV256U8OpTy, typename IntrV64F32OpTy,
          typename IntrV64S32OpTy, typename IntrV64U32OpTy,
          typename IntrV128BF16OpTy, typename... Args>
inline Value createCorrespondingIntr(OpBuilder &builder, const Location &loc,
                                     Type elementType, Args &&...args) {
  auto canBeSigned = [&elementType](unsigned int width) {
    return elementType.isSignedInteger(width) ||
           elementType.isSignlessInteger(width);
  };
  auto canBeUnsigned = [&elementType](unsigned int width) {
    return elementType.isUnsignedInteger(width) ||
           elementType.isSignlessInteger(width);
  };
  if (Value v = createTargetTyIfValid<IntrV128F16OpTy>(
          builder, loc, elementType.isF16(), std::forward<Args>(args)...))
    return v;
  if (Value v = createTargetTyIfValid<IntrV128S16OpTy>(
          builder, loc, canBeSigned(16), std::forward<Args>(args)...))
    return v;
  if (Value v = createTargetTyIfValid<IntrV128U16OpTy>(
          builder, loc, canBeUnsigned(16), std::forward<Args>(args)...))
    return v;
  if (Value v = createTargetTyIfValid<IntrV256S8OpTy>(
          builder, loc, canBeSigned(8), std::forward<Args>(args)...))
    return v;
  if (Value v = createTargetTyIfValid<IntrV256U8OpTy>(
          builder, loc, canBeUnsigned(8), std::forward<Args>(args)...))
    return v;
  if (Value v = createTargetTyIfValid<IntrV64F32OpTy>(
          builder, loc, elementType.isF32(), std::forward<Args>(args)...))
    return v;
  if (Value v = createTargetTyIfValid<IntrV64S32OpTy>(
          builder, loc, canBeSigned(32), std::forward<Args>(args)...))
    return v;
  if (Value v = createTargetTyIfValid<IntrV64U32OpTy>(
          builder, loc, canBeUnsigned(32), std::forward<Args>(args)...))
    return v;
  if (Value v = createTargetTyIfValid<IntrV128BF16OpTy>(
          builder, loc, elementType.isBF16(), std::forward<Args>(args)...))
    return v;
  return nullptr;
}

/// Base conversion for Binary ops that can be lowered to one of the intrinsics
/// based on the vector length and vector element type.
template <typename OpTy, typename IntrV128F16OpTy, typename IntrV128S16OpTy,
          typename IntrV128U16OpTy, typename IntrV256S8OpTy,
          typename IntrV256U8OpTy, typename IntrV64F32OpTy,
          typename IntrV64S32OpTy, typename IntrV64U32OpTy,
          typename IntrV128BF16OpTy>
struct BinaryLowerToIntrinsic : public OpConversionPattern<OpTy> {
  explicit BinaryLowerToIntrinsic(LLVMTypeConverter &converter)
      : OpConversionPattern<OpTy>(converter, &converter.getContext()) {}

  const LLVMTypeConverter &getTypeConverter() const {
    return *static_cast<const LLVMTypeConverter *>(
        OpConversionPattern<OpTy>::getTypeConverter());
  }
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto vecType = cast<VectorType>(lhs.getType());
    uint64_t totalSize = static_cast<uint64_t>(vecType.getNumElements());
    Type elementType = vecType.getElementType();
    auto dataWidth = elementType.getIntOrFloatBitWidth();
    Value mask =
        getVLRegValueOrSelf(op.getMask(), rewriter);
    auto vlLength = util::VL_BITS / dataWidth;
    VectorType oriVecType = vecType;
    if (totalSize != vlLength) {
      vecType = VectorType::get(SmallVector<int64_t>{vlLength}, elementType);
      lhs = rewriter.create<UnrealizedConversionCastOp>(loc, vecType, lhs)
                .getResult(0);
      rhs = rewriter.create<UnrealizedConversionCastOp>(loc, vecType, rhs)
                .getResult(0);
    }
    Value res =
        createCorrespondingIntr<IntrV128F16OpTy, IntrV128S16OpTy,
                                IntrV128U16OpTy, IntrV256S8OpTy, IntrV256U8OpTy,
                                IntrV64F32OpTy, IntrV64S32OpTy, IntrV64U32OpTy,
                                IntrV128BF16OpTy>(rewriter, loc, elementType,
                                                  vecType, lhs, rhs, mask);
    if (!res) {
      return rewriter.notifyMatchFailure(op, "cannot legalize op");
    }
    if (oriVecType != vecType) {
      Operation *ucc =
          rewriter.create<UnrealizedConversionCastOp>(loc, oriVecType, res);
      rewriter.replaceOp(op, ucc);
    } else {
      rewriter.replaceOp(op, res);
    }
    return success();
  }
};

/// Before conversion:
/// ```mlir
//    %c0_i32_0 = arith.constant 0 : i32
///   ave.hir.membar %c0_i32_0
/// ```
/// After conversion:
/// ```mlir
///   "hivm_regbaseintrins.intr.hivm.mem.bar.vv.all"() : () -> ()
/// ```
struct HIVMMemBarOpLowering
    : public ConvertOpToLLVMPattern<hivmave::VFMemBarOp> {
  explicit HIVMMemBarOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<hivmave::VFMemBarOp>(converter) {}
  LogicalResult
  matchAndRewrite(hivmave::VFMemBarOp op, hivmave::VFMemBarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    int32_t membar_type_value = 0;
    if (auto constOp = src.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue())) {
        membar_type_value = intAttr.getInt();
      }
    }
    Operation *membar;
    switch (membar_type_value) {
    case 0:
      membar = rewriter.create<MemBarVVALLInstrOp>(loc);
      break;
    case 1:
      membar = rewriter.create<MemBarVSTVLDInstrOp>(loc);
      break;
    case 2:
      membar = rewriter.create<MemBarVLDVSTInstrOp>(loc);
      break;
    case 3:
      membar = rewriter.create<MemBarVSTVSTInstrOp>(loc);
      break;
    case 4:
      membar = rewriter.create<MemBarVSALLInstrOp>(loc);
      break;
    case 5:
      membar = rewriter.create<MemBarVSTLDInstrOp>(loc);
      break;
    case 6:
      membar = rewriter.create<MemBarVLDSTInstrOp>(loc);
      break;
    case 7:
      membar = rewriter.create<MemBarVSTSTInstrOp>(loc);
      break;
    case 8:
      membar = rewriter.create<MemBarSVALLInstrOp>(loc);
      break;
    case 9:
      membar = rewriter.create<MemBarSTVLDInstrOp>(loc);
      break;
    case 10:
      membar = rewriter.create<MemBarLDVSTInstrOp>(loc);
      break;
    case 11:
      membar = rewriter.create<MemBarSTVSTInstrOp>(loc);
      break;
    case 12:
      membar = rewriter.create<MemBarSSALLTInstrOp>(loc);
      break;
    case 13:
      membar = rewriter.create<MemBarSTLDInstrOp>(loc);
      break;
    case 14:
      membar = rewriter.create<MemBarLDSTInstrOp>(loc);
      break;
    case 15:
      membar = rewriter.create<MemBarSTSTInstrOp>(loc);
      break;
    default:
      llvm::report_fatal_error("Invalid membar_type_value!");
    }
    rewriter.replaceOp(op, membar);
    return success();
  }
};

struct HIVMVpackOpLowering : public ConvertOpToLLVMPattern<hivmave::VFVpackOp> {
  explicit HIVMVpackOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<hivmave::VFVpackOp>(converter) {}

  LogicalResult
  matchAndRewrite(hivmave::VFVpackOp op, hivmave::VFVpackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value part = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(adaptor.getPart()));
    VectorType vecType = cast<VectorType>(op.getRes().getType());
    Type elemType = vecType.getElementType();
    Type llvmVecType = getTypeConverter()->convertType(vecType);
    if (!llvmVecType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert vector type to LLVM type");
    auto llvmVecVLType = createVLVectorType(elemType);
    UnrealizedConversionCastOp srcCasted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecVLType, src);
    Operation *vpackOp = buildVpackOp(loc, part, srcCasted->getResult(0),
                                      llvmVecVLType, rewriter);
    if (!vpackOp)
      return rewriter.notifyMatchFailure(op, "failed to create VpackInstrOp");
    UnrealizedConversionCastOp resCasted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecType,
                                                    vpackOp->getResult(0));
    rewriter.replaceOp(op, resCasted->getResult(0));
    return success();
  }
};

struct HIVMVunpackOpLowering
    : public ConvertOpToLLVMPattern<hivmave::VFVunpackOp> {
  explicit HIVMVunpackOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<hivmave::VFVunpackOp>(converter) {}

  LogicalResult
  matchAndRewrite(hivmave::VFVunpackOp op,
                  hivmave::VFVunpackOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value part = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(adaptor.getPart()));
    VectorType vecType = cast<VectorType>(op.getRes().getType());
    Type elemType = vecType.getElementType();
    Type llvmVecType = getTypeConverter()->convertType(vecType);
    if (!llvmVecType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert vector type to LLVM type");
    auto llvmVecVLType = createVLVectorType(elemType);
    UnrealizedConversionCastOp srcCasted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecVLType, src);
    Operation *vunpackOp = buildVunpackOp(loc, part, srcCasted->getResult(0),
                                          llvmVecVLType, rewriter);
    if (!vunpackOp)
      return rewriter.notifyMatchFailure(op, "failed to create Vzunpack");
    UnrealizedConversionCastOp resCasted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecType,
                                                    vunpackOp->getResult(0));
    rewriter.replaceOp(op, resCasted->getResult(0));
    return success();
  }
};

struct HIVMPregCastOpLowering
    : public ConvertOpToLLVMPattern<hivmave::VFPregTypeCastOp> {
  using ConvertOpToLLVMPattern<
      hivmave::VFPregTypeCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hivmave::VFPregTypeCastOp op,
                  VFPregTypeCastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto castMode = op.getPregCastMode();

    Value part = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(0));
    auto srcVLVectorTy = createVLVectorType(rewriter.getI1Type());
    UnrealizedConversionCastOp castOp =
        rewriter.create<UnrealizedConversionCastOp>(loc, srcVLVectorTy, op.getSrc());
    if (castMode == hivmave::PregCastMode::PK4_B32) {
      Operation *ppackOp = buildPpackOp(
          loc, part, castOp->getResult(0), rewriter);
      ppackOp = buildPpackOp(
          loc, part, ppackOp->getResult(0), rewriter);
      rewriter.replaceOp(op, ppackOp->getResult(0));
      return success();
    } else if (castMode == hivmave::PregCastMode::PK_B32 ||
               castMode == hivmave::PregCastMode::PK_B16) {
      Operation *ppackOp = buildPpackOp(
          loc, part, castOp->getResult(0), rewriter);
      rewriter.replaceOp(op, ppackOp->getResult(0));
      return success();
    } else if (castMode == hivmave::PregCastMode::UNPK4_B8) {
      Operation *punpackOp = buildPunpackOp(
          loc, part, castOp->getResult(0), rewriter);
      punpackOp = buildPunpackOp(
          loc, part, punpackOp->getResult(0), rewriter);
      rewriter.replaceOp(op, punpackOp->getResult(0));
      return success();
    } else if (castMode == hivmave::PregCastMode::UNPK_B16 ||
               castMode == hivmave::PregCastMode::UNPK_B8) {
      Operation *punpackOp = buildPunpackOp(
          loc, part, castOp->getResult(0), rewriter);
      rewriter.replaceOp(op, punpackOp->getResult(0));
      return success();
    }
    return failure();
  }
};

/// An entry associating the "main" BinaryOp with its instantiations for
/// vectors.
template <typename OpTy, typename IntrV128F16OpTy, typename IntrV128S16OpTy,
          typename IntrV128U16OpTy, typename IntrV256S8OpTy,
          typename IntrV256U8OpTy, typename IntrV64F32OpTy,
          typename IntrV64S32OpTy, typename IntrV64U32OpTy,
          typename IntrV128BF16OpTy>
struct BinaryRegEntry {
  using MainOp = OpTy;
  using IntrV128F16Op = IntrV128F16OpTy;
  using IntrV128S16Op = IntrV128S16OpTy;
  using IntrV128U16Op = IntrV128U16OpTy;
  using IntrV256S8Op = IntrV256S8OpTy;
  using IntrV256U8Op = IntrV256U8OpTy;
  using IntrV64F32Op = IntrV64F32OpTy;
  using IntrV64S32Op = IntrV64S32OpTy;
  using IntrV64U32Op = IntrV64U32OpTy;
  using IntrV128BF16Op = IntrV128BF16OpTy;
};
/// A container for op association entries facilitating the configuration of
/// dialect conversion.
template <typename... Args> struct BinaryRegistryImpl {
  /// Registers the patterns specializing the "main" op to one of the
  /// "intrinsic" ops depending on vector length and elemental type.
  static void registerPatterns(LLVMTypeConverter &Converter,
                               RewritePatternSet &patterns) {
    patterns.add<BinaryLowerToIntrinsic<
        typename Args::MainOp, typename Args::IntrV128F16Op,
        typename Args::IntrV128S16Op, typename Args::IntrV128U16Op,
        typename Args::IntrV256S8Op, typename Args::IntrV256U8Op,
        typename Args::IntrV64F32Op, typename Args::IntrV64S32Op,
        typename Args::IntrV64U32Op, typename Args::IntrV128BF16Op>...>(
        Converter);
  }
};

using BinaryRegistry = BinaryRegistryImpl<
    BinaryRegEntry</*OpTy*/ VFAddOp, /*F16*/ VaddSXInstrOp,
                   /*S16*/ VaddSXInstrOp, /*U16*/ VaddUXInstrOp,
                   /*S8*/ VaddSXInstrOp, /*U8*/ VaddUXInstrOp,
                   /*F32*/ VaddSXInstrOp, /*S32*/ VaddSXInstrOp,
                   /*U32*/ VaddUXInstrOp, /*BF16*/ VaddSXInstrOp>,
    BinaryRegEntry</*OpTy*/ VFMulOp, /*F16*/ VmulSXInstrOp,
                   /*S16*/ VmulSXInstrOp, /*U16*/ VmulUXInstrOp,
                   /*S8*/ VmulSXInstrOp, /*U8*/ VmulUXInstrOp,
                   /*F32*/ VmulSXInstrOp, /*S32*/ VmulSXInstrOp,
                   /*U32*/ VmulUXInstrOp, /*BF16*/ VmulSXInstrOp>,
    BinaryRegEntry</*OpTy*/ VFSubOp, /*F16*/ VsubSXInstrOp,
                   /*S16*/ VsubSXInstrOp, /*U16*/ VsubUXInstrOp,
                   /*S8*/ VsubSXInstrOp, /*U8*/ VsubUXInstrOp,
                   /*F32*/ VsubSXInstrOp, /*S32*/ VsubSXInstrOp,
                   /*U32*/ VsubUXInstrOp, /*BF16*/ VsubSXInstrOp>,
    BinaryRegEntry</*OpTy*/ VFMaxOp, /*F16*/ VmaxSXInstrOp,
                   /*S16*/ VmaxSXInstrOp, /*U16*/ VmaxUXInstrOp,
                   /*S8*/ VmaxSXInstrOp, /*U8*/ VmaxUXInstrOp,
                   /*F32*/ VmaxSXInstrOp, /*S32*/ VmaxSXInstrOp,
                   /*U32*/ VmaxUXInstrOp, /*BF16*/ VmaxSXInstrOp>,
    BinaryRegEntry</*OpTy*/ VFMinOp, /*F16*/ VminSXInstrOp,
                   /*S16*/ VminSXInstrOp, /*U16*/ VminUXInstrOp,
                   /*S8*/ VminSXInstrOp, /*U8*/ VminUXInstrOp,
                   /*F32*/ VminSXInstrOp, /*S32*/ VminSXInstrOp,
                   /*U32*/ VminUXInstrOp, /*BF16*/ VminSXInstrOp>,
    BinaryRegEntry<
        /*OpTy*/ VMaxSIOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VmaxSXInstrOp, /*U16*/ VunsupportedBinaryInstrOp,
        /*S8*/ VmaxSXInstrOp, /*U8*/ VunsupportedBinaryInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VmaxSXInstrOp,
        /*U32*/ VunsupportedBinaryInstrOp, /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry<
        /*OpTy*/ VMinSIOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VminSXInstrOp, /*U16*/ VunsupportedBinaryInstrOp,
        /*S8*/ VminSXInstrOp, /*U8*/ VunsupportedBinaryInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VminSXInstrOp,
        /*U32*/ VunsupportedBinaryInstrOp, /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry<
        /*OpTy*/ VMaxUIOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VunsupportedBinaryInstrOp, /*U16*/ VmaxUXInstrOp,
        /*S8*/ VunsupportedBinaryInstrOp, /*U8*/ VmaxUXInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VunsupportedBinaryInstrOp,
        /*U32*/ VmaxUXInstrOp, /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry<
        /*OpTy*/ VMinUIOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VunsupportedBinaryInstrOp, /*U16*/ VminUXInstrOp,
        /*S8*/ VunsupportedBinaryInstrOp, /*U8*/ VminUXInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VunsupportedBinaryInstrOp,
        /*U32*/ VminUXInstrOp, /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry<
        /*OpTy*/ VFAbsDiffOp, /*F16*/ VabsdifSXInstrOp,
        /*S16*/ VunsupportedBinaryInstrOp,
        /*U16*/ VunsupportedBinaryInstrOp, /*S8*/ VunsupportedBinaryInstrOp,
        /*U8*/ VunsupportedBinaryInstrOp, /*F32*/ VabsdifSXInstrOp,
        /*S32*/ VunsupportedBinaryInstrOp, /*U32*/ VunsupportedBinaryInstrOp,
        /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry</*OpTy*/ VFAndOp, /*F16*/ VandXInstrOp,
                   /*S16*/ VandXInstrOp, /*U16*/ VandXInstrOp,
                   /*S8*/ VandXInstrOp, /*U8*/ VandXInstrOp,
                   /*F32*/ VandXInstrOp, /*S32*/ VandXInstrOp,
                   /*U32*/ VandXInstrOp, /*BF16*/ VandXInstrOp>,
    BinaryRegEntry</*OpTy*/ VFOrOp, /*F16*/ VorXInstrOp,
                   /*S16*/ VorXInstrOp, /*U16*/ VorXInstrOp,
                   /*S8*/ VorXInstrOp, /*U8*/ VorXInstrOp,
                   /*F32*/ VorXInstrOp, /*S32*/ VorXInstrOp,
                   /*U32*/ VorXInstrOp, /*BF16*/ VorXInstrOp>,
    BinaryRegEntry</*OpTy*/ VFXorOp, /*F16*/ VxorXInstrOp,
                   /*S16*/ VxorXInstrOp, /*U16*/ VxorXInstrOp,
                   /*S8*/ VxorXInstrOp, /*U8*/ VxorXInstrOp,
                   /*F32*/ VxorXInstrOp, /*S32*/ VxorXInstrOp,
                   /*U32*/ VxorXInstrOp, /*BF16*/ VxorXInstrOp>,
    BinaryRegEntry</*OpTy*/ VFDivOp, /*F16*/ VdivSXInstrOp,
                   /*S16*/ VdivSXInstrOp, /*U16*/ VdivUXInstrOp,
                   /*S8*/ VunsupportedBinaryInstrOp,
                   /*U8*/ VunsupportedBinaryInstrOp, /*F32*/ VdivSXInstrOp,
                   /*S32*/ VdivSXInstrOp, /*U32*/ VdivUXInstrOp,
                   /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry<
        /*OpTy*/ VFDivfOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VunsupportedBinaryInstrOp, /*U16*/ VdivfV128U16XInstrOp,
        /*S8*/ VunsupportedBinaryInstrOp, /*U8*/ VunsupportedBinaryInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VunsupportedBinaryInstrOp,
        /*U32*/ VunsupportedBinaryInstrOp, /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry<
        /*OpTy*/ VFSaddOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VsaddV128S16XInstrOp, /*U16*/ VunsupportedBinaryInstrOp,
        /*S8*/ VunsupportedBinaryInstrOp, /*U8*/ VunsupportedBinaryInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VunsupportedBinaryInstrOp,
        /*U32*/ VunsupportedBinaryInstrOp, /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry<
        /*OpTy*/ VFSsubOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VssubV128S16XInstrOp, /*U16*/ VunsupportedBinaryInstrOp,
        /*S8*/ VunsupportedBinaryInstrOp, /*U8*/ VunsupportedBinaryInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VunsupportedBinaryInstrOp,
        /*U32*/ VunsupportedBinaryInstrOp, /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry</*OpTy*/ VFShlOp, /*F16*/ VunsupportedBinaryInstrOp,
                   /*S16*/ VshlSXInstrOp, /*U16*/ VshlUXInstrOp,
                   /*S8*/ VshlSXInstrOp, /*U8*/ VshlUXInstrOp,
                   /*F32*/ VunsupportedBinaryInstrOp,
                   /*S32*/ VshlSXInstrOp, /*U32*/ VshlUXInstrOp,
                   /*BF16*/ VunsupportedBinaryInstrOp>,
    BinaryRegEntry<
        /*OpTy*/ VFPReluOp, /*F16*/ VpreluXInstrOp,
        /*S16*/ VunsupportedBinaryInstrOp, /*U16*/ VunsupportedBinaryInstrOp,
        /*S8*/ VunsupportedBinaryInstrOp, /*U8*/ VunsupportedBinaryInstrOp,
        /*F32*/ VpreluXInstrOp, /*S32*/ VunsupportedBinaryInstrOp,
        /*U32*/ VunsupportedBinaryInstrOp, /*BF16*/ VunsupportedBinaryInstrOp>>;

template <typename OpTy, typename IntrV256S8OpTy, typename IntrV128S16OpTy,
          typename IntrV128F16OpTy, typename IntrV64S32OpTy,
          typename IntrV64F32OpTy>
struct UnaryLowerToIntrinsic : public OpConversionPattern<OpTy> {
  explicit UnaryLowerToIntrinsic(LLVMTypeConverter &converter)
      : OpConversionPattern<OpTy>(converter, &converter.getContext()) {}

  const LLVMTypeConverter &getTypeConverter() const {
    return *static_cast<const LLVMTypeConverter *>(
        OpConversionPattern<OpTy>::getTypeConverter());
  }
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (IntrV128F16OpTy::getOperationName() == "hivm.intr.hivm.unsupported" ||
        IntrV128S16OpTy::getOperationName() == "hivm.intr.hivm.unsupported" ||
        IntrV256S8OpTy::getOperationName() == "hivm.intr.hivm.unsupported" ||
        IntrV64F32OpTy::getOperationName() == "hivm.intr.hivm.unsupported" ||
        IntrV64S32OpTy::getOperationName() == "hivm.intr.hivm.unsupported")
      return rewriter.notifyMatchFailure(op, "cannot legalize op");
    auto loc = op.getLoc();
    Value src = op.getSrc();
    auto vecType = cast<VectorType>(src.getType());
    uint64_t totalSize = static_cast<uint64_t>(vecType.getNumElements());
    Type elementType = vecType.getElementType();
    auto dataWidth = elementType.getIntOrFloatBitWidth();
    Value mask =
        getVLRegValueOrSelf(op.getMask(), rewriter);

    auto vlLength = util::VL_BITS / dataWidth;
    VectorType oriVecType = vecType;
    if (totalSize != vlLength) {
      vecType = VectorType::get(SmallVector<int64_t>{vlLength}, elementType);
      src = rewriter.create<UnrealizedConversionCastOp>(loc, vecType, src)
                .getResult(0);
    }
    Value res;
    if (elementType.isSignedInteger(8) || elementType.isSignlessInteger(8)) {
      res = rewriter.create<IntrV256S8OpTy>(loc, vecType, src, mask);
    } else if (elementType.isSignedInteger(16) ||
               elementType.isSignlessInteger(16)) {
      res = rewriter.create<IntrV128S16OpTy>(loc, vecType, src, mask);
    } else if (elementType.isF16()) {
      res = rewriter.create<IntrV128F16OpTy>(loc, vecType, src, mask);
    } else if (elementType.isSignedInteger(32) ||
               elementType.isSignlessInteger(32)) {
      res = rewriter.create<IntrV64S32OpTy>(loc, vecType, src, mask);
    } else if (elementType.isF32()) {
      res = rewriter.create<IntrV64F32OpTy>(loc, vecType, src, mask);
    } else {
      return rewriter.notifyMatchFailure(op, "cannot legalize op");
    }
    if (oriVecType != vecType) {
      Operation *ucc =
          rewriter.create<UnrealizedConversionCastOp>(loc, oriVecType, res);
      rewriter.replaceOp(op, ucc);
    } else {
      rewriter.replaceOp(op, res);
    }
    return success();
  }
};

template <typename OpTy, typename IntrV256S8OpTy, typename IntrV128S16OpTy,
          typename IntrV128F16OpTy, typename IntrV64S32OpTy,
          typename IntrV64F32OpTy>
struct UnaryRegEntry {
  using MainOp = OpTy;
  using IntrV256S8Op = IntrV256S8OpTy;
  using IntrV128S16Op = IntrV128S16OpTy;
  using IntrV128F16Op = IntrV128F16OpTy;
  using IntrV64S32Op = IntrV64S32OpTy;
  using IntrV64F32Op = IntrV64F32OpTy;
};

template <typename... Args> struct UnaryRegistryImpl {
  static void registerPatterns(LLVMTypeConverter &Converter,
                               RewritePatternSet &patterns) {
    patterns.add<UnaryLowerToIntrinsic<
        typename Args::MainOp, typename Args::IntrV256S8Op,
        typename Args::IntrV128S16Op, typename Args::IntrV128F16Op,
        typename Args::IntrV64S32Op, typename Args::IntrV64F32Op>...>(
        Converter);
  }
};

using UnaryRegistry = UnaryRegistryImpl<
    UnaryRegEntry<VFAbsOp, VabsXInstrOp, VabsXInstrOp, VabsXInstrOp,
                  VabsXInstrOp, VabsXInstrOp>,
    UnaryRegEntry<VFNegOp, VNegXInstrOp, VNegXInstrOp, VNegXInstrOp,
                  VNegXInstrOp, VNegXInstrOp>,
    UnaryRegEntry<VFNotOp, VNotXInstrOp, VNotXInstrOp, VNotXInstrOp,
                  VNotXInstrOp, VNotXInstrOp>,
    UnaryRegEntry<VFSqrtOp, VunsupportedUnaryInstrOp, VunsupportedUnaryInstrOp,
                  VSqrtXInstrOp, VunsupportedUnaryInstrOp, VSqrtXInstrOp>,
    UnaryRegEntry<VFExpOp, VunsupportedUnaryInstrOp, VunsupportedUnaryInstrOp,
                  VExpXInstrOp, VunsupportedUnaryInstrOp, VExpXInstrOp>,
    UnaryRegEntry<VFLnOp, VunsupportedUnaryInstrOp, VunsupportedUnaryInstrOp,
                  VLnXInstrOp, VunsupportedUnaryInstrOp, VLnXInstrOp>,
    UnaryRegEntry<VFReluOp, VunsupportedUnaryInstrOp, VunsupportedUnaryInstrOp,
                  VReluXInstrOp, VReluXInstrOp, VReluXInstrOp>>;

template <typename OpTy, typename IntrOpTy>
struct HIVMBroadCastScalarOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  explicit HIVMBroadCastScalarOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<OpTy>(converter) {}
  LogicalResult
  matchAndRewrite(OpTy convertOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = convertOp.getLoc();
    VectorType vecTy = cast<VectorType>(convertOp.getRes().getType());
    Type elementType = vecTy.getElementType();
    auto dataWidth = elementType.getIntOrFloatBitWidth();
    Value scalar = adaptor.getSrc();
    Value mask = getVLRegValueOrSelf(adaptor.getMask(), rewriter);
    Value mode = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

    uint64_t vecSize = static_cast<uint64_t>(vecTy.getNumElements());
    auto vlLength = util::VL_BITS / dataWidth;
    VectorType oriVecTy = vecTy;
    if (vecSize != vlLength) {
      vecTy = VectorType::get(SmallVector<int64_t>{vlLength}, elementType);
    }
    mlir::Value result;
    if (dataWidth == 16 || dataWidth == 32) {
      result = rewriter.create<IntrOpTy>(loc, vecTy, scalar, mask, mode);
    } else if (dataWidth == 8) {
      Value i8Value;
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto i8VecTy =
          VectorType::get(SmallVector<int64_t>{vecTy.getNumElements()}, i8Type);
      if (isa<Float8E4M3FNType>(elementType) || isa<Float8E5M2Type>(elementType) ||
          scalar.getType().isFloat8E4M3FN() || scalar.getType().isFloat8E5M2()) {
        i8Value = rewriter.create<LLVM::BitcastOp>(loc, i8Type, scalar);
      } else if (scalar.getType().getIntOrFloatBitWidth() == 16) {
        i8Value = rewriter.create<arith::TruncIOp>(loc, i8Type, scalar);
      } else {
        i8Value = scalar;
      }
      result = rewriter.create<IntrOpTy>(loc, i8VecTy, i8Value, mask, mode);
      result = rewriter.create<LLVM::BitcastOp>(loc, vecTy, result);
    }
    if (oriVecTy != vecTy) {
      Operation *ucc =
          rewriter.create<UnrealizedConversionCastOp>(loc, oriVecTy, result);
      rewriter.replaceOp(convertOp, ucc);
    } else {
      rewriter.replaceOp(convertOp, result);
    }
    return success();
  }
};

template <typename OpTy, typename IntrOpTy> struct BroadCastScalarRegEntry {
  using MainOp = OpTy;
  using IntrOp = IntrOpTy;
};

template <typename... Args> struct BroadCastScalarRegistryImpl {
  static void registerPatterns(LLVMTypeConverter &Converter,
                               RewritePatternSet &patterns) {
    patterns.add<HIVMBroadCastScalarOpLowering<typename Args::MainOp,
                                               typename Args::IntrOp>...>(

        Converter);
  }
};

using BroadCastScalarRegistry = BroadCastScalarRegistryImpl<
    BroadCastScalarRegEntry<VFBroadcastScalarMaskOp, VdupsZInstrOp>>;

static bool isBRCDist(Value dist) {
  // check load dist is brc
  auto constantOp = dyn_cast<LLVM::ConstantOp>(dist.getDefiningOp());
  IntegerAttr attr = dyn_cast<IntegerAttr>(constantOp.getValue());
  uint32_t distValue = (uint32_t)(attr.getValue().getZExtValue());
  return distValue == (uint32_t)hivmave::LoadDist::BRC_B8 ||
         distValue == (uint32_t)hivmave::LoadDist::BRC_B16 ||
         distValue == (uint32_t)hivmave::LoadDist::BRC_B32;
}

// Map HIVMAVE LoadDist for i1/predicate loads onto CCE MaskDist encodings for
// hivm.plds.b8. HIVMAVE enum values US=24 / DS=25 must not be passed through:
// AscendC / __clang_cce_vector_intrinsics Dist::DIST_US=1, DIST_DS=2.
static uint32_t mapLoadDistToPldsMaskDist(hivmave::LoadDist pattern) {
  switch (pattern) {
  case hivmave::LoadDist::US:
    return 1;
  case hivmave::LoadDist::DS:
    return 2;
  default:
    return 0; // NORM
  }
}

static bool isONEPTDist(Value dist) {
  // check store dist is onept
  auto constantOp = dyn_cast<LLVM::ConstantOp>(dist.getDefiningOp());
  IntegerAttr attr = dyn_cast<IntegerAttr>(constantOp.getValue());
  uint32_t distValue = (uint32_t)(attr.getValue().getZExtValue());
  return distValue == (uint32_t)hivmave::StoreDist::ONEPT_B8 ||
         distValue == (uint32_t)hivmave::StoreDist::ONEPT_B16 ||
         distValue == (uint32_t)hivmave::StoreDist::ONEPT_B32 ||
         distValue == (uint32_t)hivmave::StoreDist::ONEPT_B64;
}

static Value createMaskByPGE(PatternRewriter &rewriter, Location loc) {
  auto pgeType = VectorType::get(256, rewriter.getI1Type());
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                  rewriter.getI32IntegerAttr(0));
  Operation *newOp = rewriter.create<PgeB8>(loc, pgeType, zero, zero);
  Value pge = newOp->getResult(0);
  return pge;
}

static Value
preginterleaveDataLayoutForExtCast(ConversionPatternRewriter &rewriter,
                                   Location loc, Value src,
                                   int elementAlignment) {
  Value pge = createMaskByPGE(rewriter, loc);
  auto srcVLVectorTy =
      hivm_regbaseintrins::createVLVectorType(rewriter.getI1Type());
  auto intlvType = LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {srcVLVectorTy, srcVLVectorTy});
  Operation *pintlv;
  if (elementAlignment == 16) {
    pintlv = rewriter.create<PintlvB8InstrOp>(loc, intlvType, src, pge);
  } else {
    pintlv = rewriter.create<PintlvB8InstrOp>(loc, intlvType, src, pge);
    Value res =
        rewriter.create<LLVM::ExtractValueOp>(loc, pintlv->getResult(0), 0);
    pintlv = rewriter.create<PintlvB8InstrOp>(loc, intlvType, res, pge);
  }
  return rewriter.create<LLVM::ExtractValueOp>(loc, pintlv->getResult(0), 0);
}

static Operation *createPstuOp(Value data, Value dataPtr,
                               PatternRewriter &rewriter,
                               int elementAlignment) {
  auto loc = dataPtr.getLoc();
  VectorType vector32I8Type = VectorType::get(32, rewriter.getI8Type());
  auto alignData =
      rewriter.create<InitVectorAlignDataInstrOp>(loc, vector32I8Type);
  Operation *result = hivm_regbaseintrins::buildPstuOp(
      data, dataPtr, rewriter, elementAlignment, alignData);
  Value USTd =
      rewriter.create<LLVM::ExtractValueOp>(loc, result->getResult(0), 0);
  Value newDataPtr =
      rewriter.create<LLVM::ExtractValueOp>(loc, result->getResult(0), 1);
  Operation *asResult =
      hivm_regbaseintrins::buildVstasOp(USTd, newDataPtr, rewriter);
  return asResult;
}

static Operation *createLoadOpFori1Type(Value dataPtr,
                                        PatternRewriter &rewriter,
                                        int elementAlignment) {
  auto loc = dataPtr.getLoc();
  VectorType dstType = VectorType::get(256, rewriter.getI1Type());
  auto asResult = hivm_regbaseintrins::buildVldasOp(dataPtr, rewriter);
  Value USTD = asResult->getResults()[0];
  Value cstZero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  // Use vldu + movvp to support b16/b32 i1 unalign
  Type elementType =
      elementAlignment == 16 ? rewriter.getI16Type() : rewriter.getI32Type();
  Operation *usResult =
      buildVldusPostOp(dataPtr, USTD, cstZero, elementType, rewriter);
  Value extractOp =
      rewriter.create<LLVM::ExtractValueOp>(loc, usResult->getResult(0), 0);
  Operation *result =
      buildMovvpOp(loc, dstType, extractOp, rewriter, elementAlignment);
  return result;
}

struct HIVM2VLLoadOpLowering : public ConvertOpToLLVMPattern<VFLoadOp> {
  explicit HIVM2VLLoadOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFLoadOp>(converter) {}
  LogicalResult
  matchAndRewrite(VFLoadOp load, VFLoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (load.getPattern() != hivmave::LoadDist::DINTLV_B32)
      return failure();
    VectorType vectorTy = load.getVectorType();
    auto loc = load->getLoc();
    MemRefType memRefTy = load.getMemRefType();
    Value dataPtr = this->getStridedElementPtr(loc, memRefTy, adaptor.getBase(),
                                               adaptor.getIndices(), rewriter);
    Value offset = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    Value mode = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    Value dist = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), static_cast<uint32_t>(load.getPattern()));
    auto dstType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                    {vectorTy, vectorTy});
    auto result = rewriter.create<Vldsx2V64F32InstrOp>(loc, dstType, dataPtr,
                                                       offset, dist, mode);
    Value extractOp0 =
        rewriter.create<LLVM::ExtractValueOp>(loc, result->getResult(0), 0);
    Value extractOp1 =
        rewriter.create<LLVM::ExtractValueOp>(loc, result->getResult(0), 1);
    rewriter.replaceAllUsesWith(load.getRes(), extractOp0);
    rewriter.replaceAllUsesWith(load.getRes1(), extractOp1);
    rewriter.eraseOp(load);
    return success();
  }
};

static Value castToTypeIfNeeded(PatternRewriter &rewriter, Location loc,
                                Value v, Type dstType) {
  if (v.getType() == dstType)
    return v;
  return rewriter.create<UnrealizedConversionCastOp>(loc, dstType, v)
      .getResult(0);
}

struct HIVMLoadOpLowering : public ConvertOpToLLVMPattern<VFLoadOp> {
  explicit HIVMLoadOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFLoadOp>(converter) {}
  LogicalResult
  matchAndRewrite(VFLoadOp load, VFLoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (load.getPattern() == hivmave::LoadDist::DINTLV_B32)
      return failure();
    // Only 1-D vectors can be lowered to LLVM.
    VectorType vectorTy = load.getVectorType();
    if (vectorTy.getRank() > 1)
      return rewriter.notifyMatchFailure(load,
                                         "cannot convert vectors more than 1D");
    uint64_t vecSize = static_cast<uint64_t>(vectorTy.getNumElements());
    auto vecElemTy = vectorTy.getElementType();
    auto elemWidth = vecElemTy.getIntOrFloatBitWidth();
    if (vecElemTy != rewriter.getI1Type() &&
        vecSize * elemWidth != util::VL_BITS)
      vectorTy = VectorType::get(
          SmallVector<int64_t>{util::VL_BITS / elemWidth}, vecElemTy);
    else if (vecElemTy == rewriter.getI1Type() &&
             vecSize != util::PREDICATE_BITS)
      vectorTy = VectorType::get(SmallVector<int64_t>{util::PREDICATE_BITS},
                                 vecElemTy);
    auto loc = load->getLoc();
    MemRefType memRefTy = load.getMemRefType();
    auto vtype = cast<VectorType>(this->typeConverter->convertType(vectorTy));
    Value dataPtr = this->getStridedElementPtr(loc, memRefTy, adaptor.getBase(),
                                               adaptor.getIndices(), rewriter);
    auto memRefDesc = adaptor.getBase();
    MemRefDescriptor memRefDescriptor(memRefDesc);

    Value offset = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    Value mode = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    Value dist = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), static_cast<uint32_t>(load.getPattern()));
    auto moduleOp = load->getParentOfType<mlir::ModuleOp>();
    bool archIs910_95 = hacc::utils::isAscend950(moduleOp);

    if (archIs910_95 && load->hasAttr(UnalignedAttr::name) &&
        !isBRCDist(dist)) {
      Type elementType = memRefTy.getElementType();
      // use vldus + movvp to support i1 unaligned address
      if (elementType.isInteger(1)) {
        int pbMode = getBitWidthFromAttr(load);
        auto result = createLoadOpFori1Type(dataPtr, rewriter, pbMode);
        if (!result) {
          llvm_unreachable("unsupported pbMode!");
        }
        rewriter.replaceOp(load, result);
        return success();
      }
      // use vldas + vldus to access UB with unaligned address
      auto asResult = hivm_regbaseintrins::buildVldasOp(dataPtr, rewriter);
      Value USTD = asResult->getResults()[0];
      Value cstZero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(0));
      // The vldus instruction is unstable.
      // Bisheng compiler no longer exposes the vldus intrinsic.
      // Use vldus.post instead of vldus
      auto usResult = hivm_regbaseintrins::buildVldusPostOp(
          dataPtr, USTD, cstZero, elementType, rewriter);

      Value extractOp =
          rewriter.create<LLVM::ExtractValueOp>(loc, usResult->getResult(0), 0);
      rewriter.replaceOp(load, extractOp);
      return success();
    }

    Type elementType = memRefTy.getElementType();
    if (elementType.isUnsignedInteger(64)) {
      auto result = rewriter.create<Vldsx1V32U64InstrOp>(loc, vtype, dataPtr,
                                                         offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isSignedInteger(64) ||
               elementType.isSignlessInteger(64)) {
      auto result = rewriter.create<Vldsx1V32S64InstrOp>(loc, vtype, dataPtr,
                                                         offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isSignedInteger(32) ||
               elementType.isSignlessInteger(32)) {
      auto result = rewriter.create<Vldsx1V64S32InstrOp>(loc, vtype, dataPtr,
                                                         offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isUnsignedInteger(32)) {
      auto result = rewriter.create<Vldsx1V64U32InstrOp>(loc, vtype, dataPtr,
                                                         offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isSignedInteger(16) ||
               elementType.isSignlessInteger(16)) {
      auto result = rewriter.create<Vldsx1V128S16InstrOp>(loc, vtype, dataPtr,
                                                          offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isUnsignedInteger(16)) {
      auto result = rewriter.create<Vldsx1V128U16InstrOp>(loc, vtype, dataPtr,
                                                          offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isSignedInteger(8) ||
               elementType.isSignlessInteger(8)) {
      Value result = rewriter.create<Vldsx1V256S8InstrOp>(loc, vtype, dataPtr,
                                                          offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isUnsignedInteger(8)) {
      Value result = rewriter.create<Vldsx1V256U8InstrOp>(loc, vtype, dataPtr,
                                                          offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isF32()) {
      auto result = rewriter.create<Vldsx1V64F32InstrOp>(loc, vtype, dataPtr,
                                                         offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isF16()) {
      auto result = rewriter.create<Vldsx1V128F16InstrOp>(loc, vtype, dataPtr,
                                                          offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isBF16()) {
      auto result = rewriter.create<Vldsx1V128BF16InstrOp>(loc, vtype, dataPtr,
                                                           offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (isa<Float8E4M3FNType>(elementType)) {
      auto result = rewriter.create<Vldsx1V256F8E4InstrOp>(loc, vtype, dataPtr,
                                                           offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (isa<Float8E5M2Type>(elementType)) {
      auto result = rewriter.create<Vldsx1V256F8E5InstrOp>(loc, vtype, dataPtr,
                                                           offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isInteger(1)) {
      // Do not hardcode NORM: honor MaskDist US/DS for plds.
      dist = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(
              mapLoadDistToPldsMaskDist(load.getPattern())));
      auto pLoadOp = rewriter.create<PLoadB8InstOp>(loc, vtype, dataPtr, offset,
                                                    dist, mode);
      Value pLoadRes = pLoadOp->getResult(0);
      auto origLoadTy = load->getResult(0).getType();
      int pbMode = getBitWidthFromAttr(load);
      bool shouldSparsifyLayout = pbMode == 16 || pbMode == 32;
      if (archIs910_95 && shouldSparsifyLayout) {
        Value res = preginterleaveDataLayoutForExtCast(rewriter, loc, pLoadRes,
                                                       pbMode);
        res = castToTypeIfNeeded(rewriter, loc, res, origLoadTy);
        rewriter.replaceOp(load, res);
      } else {
        pLoadRes = castToTypeIfNeeded(rewriter, loc, pLoadRes, origLoadTy);
        rewriter.replaceOp(load, pLoadRes);
      }
    } else {
      return rewriter.notifyMatchFailure(load, "cannot legalize op");
    }
    return success();
  }
};

struct HIVMStoreOpLowering : public ConvertOpToLLVMPattern<VFMaskedStoreOp> {
  explicit HIVMStoreOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFMaskedStoreOp>(converter) {}
  LogicalResult
  matchAndRewrite(VFMaskedStoreOp store, VFMaskedStoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType vectorTy = store.getVectorType();
    if (vectorTy.getRank() > 1)
      return rewriter.notifyMatchFailure(store,
                                         "cannot convert vectors more than 1D");

    auto loc = store->getLoc();
    MemRefType memRefTy = store.getMemRefType();

    Value data = store.getVal();
    VectorType dtype = dyn_cast<VectorType>(data.getType());
    auto dElemType = dtype.getElementType();
    uint64_t vecSize = static_cast<uint64_t>(dtype.getNumElements());
    VectorType dcastType;
    if (dElemType != rewriter.getI1Type() &&
        vecSize * dElemType.getIntOrFloatBitWidth() != util::VL_BITS)
      dcastType = VectorType::get(
          SmallVector<int64_t>{util::VL_BITS /
                               dElemType.getIntOrFloatBitWidth()},
          dElemType);
    else if (dElemType == rewriter.getI1Type() &&
             vecSize != util::PREDICATE_BITS)
      dcastType = VectorType::get(SmallVector<int64_t>{util::PREDICATE_BITS},
                                  dElemType);
    if (dcastType)
      data = rewriter.create<UnrealizedConversionCastOp>(loc, dcastType, data)
                 ->getResult(0);
    Value dataPtr = this->getStridedElementPtr(loc, memRefTy, adaptor.getBase(),
                                               adaptor.getIndices(), rewriter);
    Value offset = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    Value mode = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    Value dist = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), static_cast<uint32_t>(store.getPattern()));
    auto moduleOp = store->getParentOfType<mlir::ModuleOp>();
    bool archIs910_95 = hacc::utils::isAscend950(moduleOp);
    Value mask = getVLRegValueOrSelf(store.getMask(), rewriter);

    if (archIs910_95 && store->hasAttr(UnalignedAttr::name) &&
        !isONEPTDist(dist)) {
      if (dElemType.isInteger(1)) {
        int pbMode = getBitWidthFromAttr(store);
        auto asResult = createPstuOp(data, dataPtr, rewriter, pbMode);
        rewriter.replaceOp(store, asResult);
        return success();
      }
      // use vstus + vstas to access UB with unaligned address
      // VSTUS Vd, [Sn], Sm, USTd
      // vstus have no mask operand. vstus will only store the least significant
      // Sm-byte of Vd. Method getElemSizeByStoreMask will calculate Sm by dist
      // and mask.
      Value elemSize =
          getElemSizeByStoreMask(store.getMask(), dElemType, loc, rewriter);
      VectorType vector32I8Type = VectorType::get(32, rewriter.getI8Type());
      auto alignData =
          rewriter.create<InitVectorAlignDataInstrOp>(loc, vector32I8Type);
      auto usResult = hivm_regbaseintrins::buildVstusPostOp(
          data, dataPtr, elemSize, alignData->getResult(0), rewriter);
      Value USTd =
          rewriter.create<LLVM::ExtractValueOp>(loc, usResult->getResult(0), 0);
      Value newDataPtr =
          rewriter.create<LLVM::ExtractValueOp>(loc, usResult->getResult(0), 1);
      auto asResult =
          hivm_regbaseintrins::buildVstasOp(USTd, newDataPtr, rewriter);
      rewriter.replaceOp(store, asResult);
      return success();
    }
    if (archIs910_95 && store->hasAttr("hivm.is_continuous") &&
        isONEPTDist(dist)) {
      // use vstus + vstas to access onept
      int32_t sizeInBytes =
          static_cast<int32_t>(dElemType.getIntOrFloatBitWidth() / 8);
      Value elemSize = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(sizeInBytes));
      VectorType vector32I8Type = VectorType::get(32, rewriter.getI8Type());
      auto alignData =
          rewriter.create<InitVectorAlignDataInstrOp>(loc, vector32I8Type);
      auto usResult = hivm_regbaseintrins::buildVstusPostOp(
          data, dataPtr, elemSize, alignData->getResult(0), rewriter);
      Value USTd =
          rewriter.create<LLVM::ExtractValueOp>(loc, usResult->getResult(0), 0);
      Value newDataPtr =
          rewriter.create<LLVM::ExtractValueOp>(loc, usResult->getResult(0), 1);
      auto asResult =
          hivm_regbaseintrins::buildVstasOp(USTd, newDataPtr, rewriter);
      usResult->setAttr("hivm.is_continuous", rewriter.getUnitAttr());
      asResult->setAttr("hivm.is_continuous", rewriter.getUnitAttr());
      rewriter.replaceOp(store, asResult);
      return success();
    }
    if (dElemType.isUnsignedInteger(64)) {
      auto result = rewriter.create<Vstsx1V32U64InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isSignedInteger(64) ||
               dElemType.isSignlessInteger(64)) {
      auto result = rewriter.create<Vstsx1V32S64InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isSignedInteger(32) ||
               dElemType.isSignlessInteger(32)) {
      auto result = rewriter.create<Vstsx1V64S32InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isUnsignedInteger(32)) {
      auto result = rewriter.create<Vstsx1V64U32InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isSignedInteger(16) ||
               dElemType.isSignlessInteger(16)) {
      auto result = rewriter.create<Vstsx1V128S16InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isUnsignedInteger(16)) {
      auto result = rewriter.create<Vstsx1V128U16InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isSignedInteger(8) || dElemType.isSignlessInteger(8)) {
      auto result = rewriter.create<Vstsx1V256S8InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isUnsignedInteger(8)) {
      auto result = rewriter.create<Vstsx1V256U8InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isF32()) {
      auto result = rewriter.create<Vstsx1V64F32InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isF16()) {
      auto result = rewriter.create<Vstsx1V128F16InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isBF16()) {
      auto result = rewriter.create<Vstsx1V128BF16InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (isa<Float8E4M3FNType>(dElemType)) {
      auto result = rewriter.create<Vstsx1V256F8E4InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (isa<Float8E5M2Type>(dElemType)) {
      auto result = rewriter.create<Vstsx1V256F8E5InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isInteger(1)) {
      // Select the store sequence for the predicate layout.
      int pbMode = getBitWidthFromAttr(store);
      if (archIs910_95 && pbMode == 16 &&
          store.getPattern() == StoreDist::NORM_B8) {
        // Unaligned NORM_B8 stores have already returned through PSTU above.
        // CCE MaskDist::DIST_PACK (1) stores the 128 even predicate bits as
        // 16 bytes. PPACK followed by DIST_NORM would also overwrite the next
        // 16 bytes, which need not be padding owned by this store.
        dist = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                 rewriter.getI32IntegerAttr(1));
        auto result = rewriter.create<PStoreB8InstOp>(loc, data, dataPtr,
                                                      offset, dist, mode);
        rewriter.replaceOp(store, result);
      } else if (archIs910_95 && (pbMode == 16 || pbMode == 32)) {
        // Keep PSTU/VSTAS for the remaining sparse predicate layouts.
        auto result = createPstuOp(data, dataPtr, rewriter, pbMode);
        rewriter.replaceOp(store, result);
      } else {
        // Preserve the direct PB8 store for all other cases.
        dist = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                 rewriter.getI32IntegerAttr(0));
        auto result = rewriter.create<PStoreB8InstOp>(loc, data, dataPtr,
                                                      offset, dist, mode);
        rewriter.replaceOp(store, result);
      }
    } else {
      return rewriter.notifyMatchFailure(store, "cannot legalize op");
    }
    return success();
  }
};

struct HIVMStoreStrideOpLowering
    : public ConvertOpToLLVMPattern<VFStoreWithStrideOp> {
  explicit HIVMStoreStrideOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFStoreWithStrideOp>(converter) {}
  LogicalResult
  matchAndRewrite(VFStoreWithStrideOp store,
                  VFStoreWithStrideOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType vectorTy = store.getVectorType();
    if (vectorTy.getRank() > 1)
      return rewriter.notifyMatchFailure(store,
                                         "cannot convert vectors more than 1D");

    auto loc = store->getLoc();
    MemRefType memRefTy = store.getMemRefType();
    Value data = store.getVal();
    VectorType dtype = dyn_cast<VectorType>(data.getType());
    auto dElemType = dtype.getElementType();
    auto elemWidth = dElemType.getIntOrFloatBitWidth();
    uint64_t vecSize = static_cast<uint64_t>(dtype.getNumElements());
    VectorType dcastType;
    if (vecSize * dElemType.getIntOrFloatBitWidth() != util::VL_BITS)
      dcastType = VectorType::get(
          SmallVector<int64_t>{util::VL_BITS /
                               dElemType.getIntOrFloatBitWidth()},
          dElemType);
    if (dcastType)
      data = rewriter.create<UnrealizedConversionCastOp>(loc, dcastType, data)
                 ->getResult(0);
    Value dataPtr = this->getStridedElementPtr(loc, memRefTy, adaptor.getBase(),
                                               adaptor.getIndices(), rewriter);
    auto strideOperand = store.getOperand(3).getDefiningOp<arith::ConstantOp>();
    auto strideValue = strideOperand.getValue();
    auto elementsStride =
        dyn_cast<IntegerAttr>(strideValue).getValue().getSExtValue();
    auto numElementsPerBlock = util::vectorBlockSizeBit / elemWidth;
    auto strideConfigValue =
        (static_cast<unsigned>(elementsStride) / numElementsPerBlock)
        << BLOCK_STRIDE_OFFSET;
    Value strideConfig = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(strideConfigValue));
    Value mode = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    Value mask = getVLRegValueOrSelf(store.getMask(), rewriter);

    if (dElemType.isF32()) {
      auto result = rewriter.create<VsstbV64F32InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isF16()) {
      auto result = rewriter.create<VsstbV128F16InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isBF16()) {
      auto result = rewriter.create<VsstbV128BF16InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (isa<Float8E5M2Type>(dElemType)) {
      auto result = rewriter.create<Vsstb1V256F8E5InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (isa<Float8E4M3FNType>(dElemType)) {
      auto result = rewriter.create<Vsstb1V256F8E4InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isUnsignedInteger(32)) {
      auto result = rewriter.create<VsstbV64U32InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isSignedInteger(32) ||
               dElemType.isSignlessInteger(32)) {
      auto result = rewriter.create<VsstbV64S32InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isUnsignedInteger(16)) {
      auto result = rewriter.create<VsstbV128U16InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isSignedInteger(16) ||
               dElemType.isSignlessInteger(16)) {
      auto result = rewriter.create<VsstbV128S16InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isUnsignedInteger(8)) {
      auto result = rewriter.create<VsstbV256U8InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isSignedInteger(8) || dElemType.isSignlessInteger(8)) {
      auto result = rewriter.create<VsstbV256S8InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else {
      return rewriter.notifyMatchFailure(store, "cannot legalize op");
    }
    return success();
  }
  // For configuration register. Sm[31:16] = block stride and Sm[15:0] == repeat
  // stride.
  static const int BLOCK_STRIDE_OFFSET = 16;
};

struct HIVMGatherOpLowering : public ConvertOpToLLVMPattern<VFGatherOp> {
  explicit HIVMGatherOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFGatherOp>(converter) {}

  LogicalResult dispatchGatherOp(VFGatherOp gather,
                                 ConversionPatternRewriter &rewriter,
                                 VectorType vtype, Value dataPtr,
                                 Value indexVec, Value mask) const {

    MemRefType memRefTy = cast<MemRefType>(gather.getBase().getType());
    auto loc = gather->getLoc();

    Type elementType = memRefTy.getElementType();
    if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32)) {
      auto result = rewriter.create<VGatherV64S32InstrOp>(loc, vtype, dataPtr,
                                                          indexVec, mask);
      rewriter.replaceOp(gather, result);
    } else if (elementType.isUnsignedInteger(32)) {
      auto result = rewriter.create<VGatherV64U32InstrOp>(loc, vtype, dataPtr,
                                                          indexVec, mask);
      rewriter.replaceOp(gather, result);
    } else if (elementType.isSignedInteger(16) ||
               elementType.isSignlessInteger(16)) {
      auto result = rewriter.create<VGatherV128S16InstrOp>(loc, vtype, dataPtr,
                                                           indexVec, mask);
      rewriter.replaceOp(gather, result);
    } else if (elementType.isUnsignedInteger(16)) {
      auto result = rewriter.create<VGatherV128U16InstrOp>(loc, vtype, dataPtr,
                                                           indexVec, mask);
      rewriter.replaceOp(gather, result);
    } else if (elementType.isSignedInteger(8) ||
               elementType.isSignlessInteger(8)) {
      VectorType gatherType =
          VectorType::get({128}, rewriter.getIntegerType(16));
      auto result = rewriter.create<VGatherV256S8InstrOp>(
          loc, gatherType, dataPtr, indexVec, mask);
      Value resI8 = rewriter.create<LLVM::BitcastOp>(loc, vtype, result);
      rewriter.replaceOp(gather, resI8);
    } else if (elementType.isUnsignedInteger(8)) {
      VectorType gatherType =
          VectorType::get({128}, rewriter.getIntegerType(16));
      auto result = rewriter.create<VGatherV256U8InstrOp>(
          loc, gatherType, dataPtr, indexVec, mask);
      Value resI8 = rewriter.create<LLVM::BitcastOp>(loc, vtype, result);
      rewriter.replaceOp(gather, resI8);
    } else if (isa<Float8E4M3FNType>(elementType)) {
      VectorType gatherType =
          VectorType::get({128}, rewriter.getIntegerType(16));
      Value result = rewriter.create<VGatherV256F8E4InstrOp>(
          loc, gatherType, dataPtr, indexVec, mask);
      Value resF8E4 = rewriter.create<LLVM::BitcastOp>(loc, vtype, result);
      rewriter.replaceOp(gather, resF8E4);
    } else if (isa<Float8E5M2Type>(elementType)) {
      VectorType gatherType =
          VectorType::get({128}, rewriter.getIntegerType(16));
      Value result = rewriter.create<VGatherV256F8E5InstrOp>(
          loc, gatherType, dataPtr, indexVec, mask);
      Value resF8E5 = rewriter.create<LLVM::BitcastOp>(loc, vtype, result);
      rewriter.replaceOp(gather, resF8E5);
    } else if (elementType.isF32()) {
      auto result = rewriter.create<VGatherV64F32InstrOp>(loc, vtype, dataPtr,
                                                          indexVec, mask);
      rewriter.replaceOp(gather, result);
    } else if (elementType.isF16()) {
      Value result = rewriter.create<VGatherV128F16InstrOp>(loc, vtype, dataPtr,
                                                            indexVec, mask);
      rewriter.replaceOp(gather, result);
    } else if (elementType.isBF16()) {
      Value result = rewriter.create<VGatherV128BF16InstrOp>(
          loc, vtype, dataPtr, indexVec, mask);
      rewriter.replaceOp(gather, result);
    } else {
      return failure();
    }

    return success();
  }

  LogicalResult
  matchAndRewrite(VFGatherOp gather, VFGatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only 1-D vectors can be lowered to LLVM.
    VectorType vectorTy = cast<VectorType>(gather.getType());
    if (vectorTy.getRank() > 1)
      return rewriter.notifyMatchFailure(gather,
                                         "cannot convert vectors more than 1D");
    uint64_t vecSize = static_cast<uint64_t>(vectorTy.getNumElements());
    auto vecElemTy = vectorTy.getElementType();
    auto elemWidth = vecElemTy.getIntOrFloatBitWidth();
    if (vecElemTy != rewriter.getI1Type() &&
        vecSize * elemWidth != util::VL_BITS)
      vectorTy = VectorType::get(
          SmallVector<int64_t>{util::VL_BITS / elemWidth}, vecElemTy);
    auto loc = gather->getLoc();
    MemRefType memRefTy = cast<MemRefType>(gather.getBase().getType());
    auto vtype = cast<VectorType>(this->typeConverter->convertType(vectorTy));
    Value dataPtr = this->getStridedElementPtr(loc, memRefTy, adaptor.getBase(),
                                               adaptor.getIndices(), rewriter);
    auto memRefDesc = adaptor.getBase();
    MemRefDescriptor memRefDescriptor(memRefDesc);
    Type intrinMaskType = VectorType::get(
        SmallVector<int64_t>{util::PREDICATE_BITS}, rewriter.getI1Type());
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, intrinMaskType, gather.getMask());
    Value mask = castOp.getOutputs()[0];
    Value indexVec = gather.getIndexVec();
    VectorType indexVecType = cast<VectorType>(indexVec.getType());
    auto indexVecElemType = indexVecType.getElementType();
    // for some reasons, currently all vgather2_v300 intrinsics use 64xi32 index
    // vec so we need to cast the index vector type to 64xi32
    if (indexVecElemType != rewriter.getI32Type() ||
        static_cast<unsigned>(indexVecType.getNumElements()) *
                indexVecElemType.getIntOrFloatBitWidth() !=
            util::VL_BITS) {
      auto desiredIndexElementType = rewriter.getI32Type();
      VectorType dcastType = VectorType::get(
          SmallVector<int64_t>{util::VL_BITS /
                               desiredIndexElementType.getIntOrFloatBitWidth()},
          desiredIndexElementType);
      auto srcVLVectorTy =
          hivm_regbaseintrins::createVLVectorType(indexVecElemType);
      Value srcCasted =
          rewriter
              .create<UnrealizedConversionCastOp>(loc, srcVLVectorTy, indexVec)
              ->getResult(0);
      indexVec = rewriter.create<LLVM::BitcastOp>(loc, dcastType, srcCasted);
    }

    if (failed(
            dispatchGatherOp(gather, rewriter, vtype, dataPtr, indexVec, mask)))
      return rewriter.notifyMatchFailure(gather, "cannot legalize op");

    return success();
  }
};

struct HIVMScatterOpLowering : public ConvertOpToLLVMPattern<VFScatterOp> {
  explicit HIVMScatterOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFScatterOp>(converter) {}
  LogicalResult
  matchAndRewrite(VFScatterOp scatter, VFScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(scatter,
                                       "scatter lowering not implemented yet");
  }
};

struct HIVMPgeOpLowering : public ConvertOpToLLVMPattern<VFPgeOp> {
  explicit HIVMPgeOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFPgeOp>(converter) {}
  LogicalResult
  matchAndRewrite(VFPgeOp pge, VFPgeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType dstType = cast<VectorType>(pge.getRes().getType());
    auto dstTyNumElems = cast<VectorType>(dstType).getNumElements();
    if (dstType.getRank() != 1)
      return failure();
    auto loc = pge->getLoc();
    PgePattern realPattern = pge.getPattern();
    Value pattern = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(static_cast<uint32_t>(realPattern)));

    int elementAlignment = getBitWidthFromAttr(pge);
    if (elementAlignment == -1)
      elementAlignment = util::VL_BITS / dstTyNumElems;
    Operation *newOp = buildPgeOp(loc, pattern, elementAlignment, rewriter);
    newOp->setAttr(utils::maskBitWidth,
                   rewriter.getI32IntegerAttr(elementAlignment));
    if (auto maskOpIdxAttr = pge->getAttr(utils::maskOpIdx))
      newOp->setAttr(utils::maskOpIdx, maskOpIdxAttr);
    rewriter.replaceOp(pge, newOp);
    return success();
  }
};

struct HIVMPltOpLowering : public ConvertOpToLLVMPattern<VFPltOp> {
  explicit HIVMPltOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFPltOp>(converter) {}

  LogicalResult
  matchAndRewrite(VFPltOp plt, VFPltOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstType = plt.getResult(0).getType();
    auto dstTyNumElems = cast<VectorType>(dstType).getNumElements();

    auto loc = plt->getLoc();
    Value true_shape = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), plt.getTrueShape());

    int elementAlignment = getBitWidthFromAttr(plt);
    if (elementAlignment == -1)
      elementAlignment = util::VL_BITS / dstTyNumElems;

    Value result =
        buildPltOp(loc, true_shape, elementAlignment, rewriter)->getResult(0);
    Operation *extractOp =
        rewriter.create<LLVM::ExtractValueOp>(loc, result, 0);
    extractOp->setAttr(
        utils::maskBitWidth,
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), elementAlignment));
    SmallVector<Value, 4> results;
    if (auto maskOpIdxAttr = plt->getAttr(utils::maskOpIdx))
      extractOp->setAttr(utils::maskOpIdx, maskOpIdxAttr);
    results.push_back(extractOp->getResult(0));
    results.push_back(rewriter.create<LLVM::ExtractValueOp>(loc, result, 1));
    rewriter.replaceOp(plt, results);
    return success();
  }
};

struct HIVMPltMOpLowering : public ConvertOpToLLVMPattern<VFPltMOp> {
  explicit HIVMPltMOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFPltMOp>(converter) {}

  LogicalResult
  matchAndRewrite(VFPltMOp pltm, VFPltMOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstType = pltm.getRes().getType();
    auto dstTyNumElems = cast<VectorType>(dstType).getNumElements();

    auto loc = pltm->getLoc();
    int elementAlignment = getBitWidthFromAttr(pltm);
    if (elementAlignment == -1)
      elementAlignment = util::VL_BITS / dstTyNumElems;

    Value offset = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), pltm.getOffset());
    Value ub = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                   pltm.getUb());

    // The VFPltM op's offset is defined differently from the intrinsic's,
    // adjust this difference here.
    Value vecLength = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(util::VL_BITS / elementAlignment));
    arith::DivSIOp divI64 =
        rewriter.create<arith::DivSIOp>(loc, offset, vecLength);
    auto div = rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(),
                                                divI64.getResult());

    Operation *pltmIntr = buildPltMOp(loc, div, ub, elementAlignment, rewriter);
    pltmIntr->setAttr(
        utils::maskBitWidth,
        rewriter.getIntegerAttr(rewriter.getI32Type(), elementAlignment));
    UnrealizedConversionCastOp cast =
        rewriter.create<UnrealizedConversionCastOp>(
            loc, pltm.getRes().getType(), pltmIntr->getResult(0));
    rewriter.replaceOp(pltm, cast);
    return success();
  }
};

struct HIVMSelectOpLowering : public ConvertOpToLLVMPattern<VFSelectOp> {
  explicit HIVMSelectOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFSelectOp>(converter) {}
  LogicalResult
  matchAndRewrite(VFSelectOp select, VFSelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = select.getLoc();
    VectorType vType = cast<VectorType>(select.getRes().getType());
    Type elementType = vType.getElementType();
    auto vlType = createVLVectorType(elementType);

    Value adaptMsk = adaptor.getMask();
    Value mask = adaptMsk;
    mask = getVLRegValueOrSelf(mask, rewriter);
    auto trueValue = select.getTrueValue();
    auto falseValue = select.getFalseValue();
    trueValue = getVLRegValueOrSelf(trueValue, rewriter);
    falseValue = getVLRegValueOrSelf(falseValue, rewriter);
    Value result;
    if (elementType.isUnsignedInteger(8) || elementType.isSignedInteger(8) ||
        elementType.isSignlessInteger(8) || elementType.isUnsignedInteger(16) ||
        elementType.isSignedInteger(16) || elementType.isSignlessInteger(16) ||
        elementType.isBF16() || elementType.isF16() ||
        elementType.isUnsignedInteger(32) || elementType.isSignedInteger(32) ||
        elementType.isSignlessInteger(32) || elementType.isF32() ||
        isa<Float8E4M3FNType>(elementType) || isa<Float8E5M2Type>(elementType)) {
      result = rewriter.create<VselInstrOp>(loc, vlType, trueValue, falseValue,
                                            mask);
    } else {
      return rewriter.notifyMatchFailure(select, "cannot legalize op");
    }
    if (vlType != vType) {
      result = rewriter.create<UnrealizedConversionCastOp>(loc, vType, result)
                   ->getResult(0);
    }
    rewriter.replaceOp(select, result);
    return success();
  }
};

template <typename IntrSOpTy, typename IntrUOpTy>
static Value cmpLowerToIntrin(VFCmpOp cmpOp,
                              ConversionPatternRewriter &rewriter) {
  VectorType vType = cast<VectorType>(cmpOp.getRes().getType());
  auto lhs = getVLRegValueOrSelf(cmpOp.getLhs(), rewriter);
  auto rhs = getVLRegValueOrSelf(cmpOp.getRhs(), rewriter);
  auto loc = cmpOp->getLoc();
  Type elementType = cast<VectorType>(lhs.getType()).getElementType();

  Value mask = getVLRegValueOrSelf(cmpOp.getMask(), rewriter);

  Value result;
  Type intrinMaskType = VectorType::get(
      SmallVector<int64_t>{util::PREDICATE_BITS}, rewriter.getI1Type());
  if (elementType.isUnsignedInteger(8) || elementType.isUnsignedInteger(16) ||
      elementType.isUnsignedInteger(32)) {
    result = rewriter.create<IntrUOpTy>(loc, intrinMaskType, lhs, rhs, mask);
  } else if (elementType.isSignedInteger(8) ||
             elementType.isSignlessInteger(8) ||
             elementType.isSignedInteger(16) ||
             elementType.isSignlessInteger(16) || elementType.isF16() ||
             elementType.isSignedInteger(32) ||
             elementType.isSignlessInteger(32) || elementType.isF32() ||
             elementType.isBF16()) {
    result = rewriter.create<IntrSOpTy>(loc, intrinMaskType, lhs, rhs, mask);
  }
  if (result && vType.getShape().back() != util::PREDICATE_BITS)
    result = rewriter.create<UnrealizedConversionCastOp>(loc, vType, result)
                 ->getResult(0);
  return result;
}

template <typename IntrOpTy>
static Value cmpLowerToIntrin(VFCmpOp cmpOp,
                              ConversionPatternRewriter &rewriter) {
  VectorType vType = cast<VectorType>(cmpOp.getRes().getType());
  auto lhs = getVLRegValueOrSelf(cmpOp.getLhs(), rewriter);
  auto rhs = getVLRegValueOrSelf(cmpOp.getRhs(), rewriter);
  auto loc = cmpOp->getLoc();
  Type elementType = cast<VectorType>(lhs.getType()).getElementType();

  Value mask = getVLRegValueOrSelf(cmpOp.getMask(), rewriter);

  Value result;
  Type intrinMaskType = VectorType::get(
      SmallVector<int64_t>{util::PREDICATE_BITS}, rewriter.getI1Type());

  if (elementType.isSignlessInteger(8) || elementType.isSignlessInteger(16) ||
      elementType.isSignlessInteger(32)) {
    result = rewriter.create<IntrOpTy>(loc, intrinMaskType, lhs, rhs, mask);
  } else {
    // Handle unsupported element types
    std::ignore = rewriter.notifyMatchFailure(
        cmpOp, "Unsupported element type in cmpLowerToIntrin");
    return Value();
  }
  if (result && vType.getShape().back() != util::PREDICATE_BITS)
    result = rewriter.create<UnrealizedConversionCastOp>(loc, vType, result)
                 ->getResult(0);
  return result;
}

struct HIVMCmpOpLowering : public ConvertOpToLLVMPattern<VFCmpOp> {
  explicit HIVMCmpOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFCmpOp>(converter) {}
  LogicalResult
  matchAndRewrite(VFCmpOp cmpOp, VFCmpOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value res;
    switch (cmpOp.getCmp()) {
    case CmpType::EQ:
      res = cmpLowerToIntrin<VCmpSEqInstrOp, VCmpUEqInstrOp>(cmpOp, rewriter);
      break;
    case CmpType::NE:
      res = cmpLowerToIntrin<VCmpSNeInstrOp, VCmpUNeInstrOp>(cmpOp, rewriter);
      break;
    case CmpType::GT:
      res = cmpLowerToIntrin<VCmpSGtInstrOp, VCmpUGtInstrOp>(cmpOp, rewriter);
      break;
    case CmpType::GE:
      res = cmpLowerToIntrin<VCmpSGeInstrOp, VCmpUGeInstrOp>(cmpOp, rewriter);
      break;
    case CmpType::LT:
      res = cmpLowerToIntrin<VCmpSLtInstrOp, VCmpULtInstrOp>(cmpOp, rewriter);
      break;
    case CmpType::LE:
      res = cmpLowerToIntrin<VCmpSLeInstrOp, VCmpULeInstrOp>(cmpOp, rewriter);
      break;
    case CmpType::ULE:
      res = cmpLowerToIntrin<VCmpULeInstrOp>(cmpOp, rewriter);
      break;
    case CmpType::UGE:
      res = cmpLowerToIntrin<VCmpUGeInstrOp>(cmpOp, rewriter);
      break;
    case CmpType::ULT:
      res = cmpLowerToIntrin<VCmpULtInstrOp>(cmpOp, rewriter);
      break;
    case CmpType::UGT:
      res = cmpLowerToIntrin<VCmpUGtInstrOp>(cmpOp, rewriter);
      break;
    }

    rewriter.replaceOp(cmpOp, res);
    return success();
  }
};



struct HIVMReductionOpLowering : public ConvertOpToLLVMPattern<ReductionOp> {
  explicit HIVMReductionOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<ReductionOp>(converter) {}
  LogicalResult
  matchAndRewrite(ReductionOp reduction, ReductionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = reduction.getLoc();
    // cast reduction source to VL
    auto vector = getVLRegValueOrSelf(reduction.getVector(), rewriter);
    auto resultType = cast<VectorType>(reduction.getResult().getType());
    Type elementType = resultType.getElementType();
    // cast reduction result to VL
    VectorType vType = hivm_regbaseintrins::createVLVectorType(elementType);

    auto kind = reduction.getKind();
    Value mask = getVLRegValueOrSelf(reduction.getMask(), rewriter);
    VectorType predType = VectorType::get(
        SmallVector<int64_t>{util::PREDICATE_BITS}, rewriter.getI1Type());
    switch (kind) {
    case hivmave::CombiningKind::ADD: {
      return matchAndRewriteVcadd(reduction, rewriter, loc, vector, vType,
                                  elementType, predType, mask);
    }
    case hivmave::CombiningKind::MAX: {
      return matchAndRewriteVcMinMax<hivmave::CombiningKind::MAX>(
          reduction, rewriter, loc, vector, vType, elementType, mask);
    }
    case hivmave::CombiningKind::MIN: {
      return matchAndRewriteVcMinMax<hivmave::CombiningKind::MIN>(
          reduction, rewriter, loc, vector, vType, elementType, mask);
    }
    case hivmave::CombiningKind::UMIN: {
      return matchAndRewriteVcMinMax<hivmave::CombiningKind::UMIN>(
          reduction, rewriter, loc, vector, vType, elementType, mask);
    }
    case hivmave::CombiningKind::UMAX: {
      return matchAndRewriteVcMinMax<hivmave::CombiningKind::UMAX>(
          reduction, rewriter, loc, vector, vType, elementType, mask);
    }
    default:
      return failure();
    }
  }

private:
  LogicalResult matchAndRewriteVcadd(ReductionOp reduction,
                                     ConversionPatternRewriter &rewriter,
                                     Location loc, Value vector,
                                     VectorType vType, Type elementType,
                                     VectorType predType, Value mask) const {
    Operation *vcaddOp;
    Value result;
    auto cstValue = [&](auto enumVal) -> Value {
      return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                               static_cast<uint32_t>(enumVal));
    };
    auto srcVLVectorTy = hivm_regbaseintrins::createVLVectorType(elementType);
    Value srcCasted =
        rewriter.create<UnrealizedConversionCastOp>(loc, srcVLVectorTy, vector)
            ->getResult(0);
    if (elementType.isUnsignedInteger(8)) {
      Type i16VecType =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI16Type());
      vcaddOp =
          rewriter.create<VcaddUXInstrOp>(loc, i16VecType, srcCasted, mask);
      Type truncType =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI8Type());
      result = rewriter
                   .create<VcvtiiU162U8InstrOp>(loc, truncType,
                                                vcaddOp->getResult(0), mask,
                                                cstValue(0), cstValue(0))
                   ->getResult(0);
    } else if (elementType.isSignedInteger(8) ||
               elementType.isSignlessInteger(8)) {
      Type i16VecType =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI16Type());
      vcaddOp =
          rewriter.create<VcaddSXInstrOp>(loc, i16VecType, srcCasted, mask);
      Type truncType =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI8Type());
      result = rewriter
                   .create<VcvtiiS162U8InstrOp>(loc, truncType,
                                                vcaddOp->getResult(0), mask,
                                                cstValue(0), cstValue(0))
                   ->getResult(0);
    } else if (elementType.isUnsignedInteger(16)) {
      Type i32VecType =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI32Type());
      vcaddOp =
          rewriter.create<VcaddUXInstrOp>(loc, i32VecType, srcCasted, mask);
      Type truncType =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI16Type());
      result = rewriter
                   .create<VcvtiiU322U16InstrOp>(loc, truncType,
                                                 vcaddOp->getResult(0), mask,
                                                 cstValue(0), cstValue(0))
                   ->getResult(0);
    } else if (elementType.isSignedInteger(16) ||
               elementType.isSignlessInteger(16)) {
      Type i32VecType =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI32Type());
      vcaddOp =
          rewriter.create<VcaddSXInstrOp>(loc, i32VecType, srcCasted, mask);
      Type truncType =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI16Type());
      result = rewriter
                   .create<VcvtiiS322S16InstrOp>(loc, truncType,
                                                 vcaddOp->getResult(0), mask,
                                                 cstValue(0), cstValue(0))
                   ->getResult(0);
    } else if (elementType.isF16()) {
      result = rewriter.create<VcaddSXInstrOp>(loc, vType, srcCasted, mask)
                   ->getResult(0);
    } else if (elementType.isUnsignedInteger(32)) {
      result = rewriter.create<VcaddUXInstrOp>(loc, vType, srcCasted, mask)
                   ->getResult(0);
    } else if (elementType.isSignedInteger(32) ||
               elementType.isSignlessInteger(32)) {
      result = rewriter.create<VcaddSXInstrOp>(loc, vType, srcCasted, mask)
                   ->getResult(0);
    } else if (elementType.isF32()) {
      result = rewriter.create<VcaddSXInstrOp>(loc, vType, srcCasted, mask)
                   ->getResult(0);
    } else {
      std::ignore = rewriter.notifyMatchFailure(
          reduction, "unsupported element type for vcadd");
      return failure();
    }

    rewriter.replaceOp(reduction, result);
    return success();
  }

  template <bool uOp, int width, typename OpS, typename OpU, typename OpF>
  FailureOr<Operation *> buildVcMinMaxOp(ConversionPatternRewriter &rewriter,
                                         Type elementType, Location loc,
                                         VectorType vType, Value vector,
                                         Value mask) const {
    if (elementType.getIntOrFloatBitWidth() != width) {
      return failure();
    }
    if (elementType.isInteger()) {
      return uOp ? static_cast<Operation *>(
                       rewriter.create<OpU>(loc, vType, vector, mask))
                 : static_cast<Operation *>(
                       rewriter.create<OpS>(loc, vType, vector, mask));
    }

    if constexpr (!std::is_same_v<OpF, std::nullptr_t>) {
      return static_cast<Operation *>(
          rewriter.create<OpF>(loc, vType, vector, mask));
    }
    llvm::report_fatal_error("avoid compilation of create<nulltype>");
  }

  template <hivmave::CombiningKind kind, int width, typename OpSMin,
            typename OpUMin, typename OpSMax, typename OpUMax, typename OpFMin,
            typename OpFMax>
  FailureOr<Operation *> buildVcMinMaxOp(ConversionPatternRewriter &rewriter,
                                         Type elementType, Location loc,
                                         VectorType vType, Value vector,
                                         Value mask) const {
    constexpr auto umin = kind == hivmave::CombiningKind::UMIN;
    constexpr auto umax = kind == hivmave::CombiningKind::UMAX;
    constexpr auto min = kind == hivmave::CombiningKind::MIN;
    constexpr auto max = kind == hivmave::CombiningKind::MAX;

    if constexpr (umin || min) {
      return buildVcMinMaxOp<umin, width, OpSMin, OpUMin, OpFMin>(
          rewriter, elementType, loc, vType, vector, mask);
    } else if constexpr (umax || max) {
      return buildVcMinMaxOp<umax, width, OpSMax, OpUMax, OpFMax>(
          rewriter, elementType, loc, vType, vector, mask);
    } else {
      llvm::report_fatal_error("unsupported CombiningKind (should be detected in "
                       "compile timeon newer clang)");
    }
  }

  template <hivmave::CombiningKind kind>
  LogicalResult matchAndRewriteVcMinMax(ReductionOp reduction,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc, Value vector,
                                        VectorType vType, Type elementType,
                                        Value mask) const {
    FailureOr<Operation *> op;

    op = buildVcMinMaxOp<kind,
                         /*Bitwidth*/ 8,
                         /*Vcmin signed*/ VcminSXInstrOp,
                         /*Vcmin unsigned*/ VcminUXInstrOp,
                         /*Vcmax signed*/ VcmaxSXInstrOp,
                         /*Vcmax unsigned*/ VcmaxUXInstrOp,
                         /*Vcmin float*/ std::nullptr_t,
                         /*Vcmax float*/ std::nullptr_t>(
        rewriter, elementType, loc, vType, vector, mask);
    if (!failed(op)) {
      rewriter.replaceOp(reduction, *op);
      return success();
    }

    op = buildVcMinMaxOp<kind,
                         /*Bitwidth*/ 16,
                         /*Vcmin signed*/ VcminSXInstrOp,
                         /*Vcmin unsigned*/ VcminUXInstrOp,
                         /*Vcmax signed*/ VcmaxSXInstrOp,
                         /*Vcmax unsigned*/ VcmaxUXInstrOp,
                         /*Vcmin float*/ VcminSXInstrOp,
                         /*Vcmax float*/ VcmaxSXInstrOp>(
        rewriter, elementType, loc, vType, vector, mask);
    if (!failed(op)) {
      rewriter.replaceOp(reduction, *op);
      return success();
    }

    op = buildVcMinMaxOp<kind,
                         /*Bitwidth*/ 32,
                         /*Vcmin signed*/ VcminSXInstrOp,
                         /*Vcmin unsigned*/ VcminUXInstrOp,
                         /*Vcmax signed*/ VcmaxSXInstrOp,
                         /*Vcmax unsigned*/ VcmaxUXInstrOp,
                         /*Vcmin float*/ VcminSXInstrOp,
                         /*Vcmax float*/ VcmaxSXInstrOp>(
        rewriter, elementType, loc, vType, vector, mask);
    if (!failed(op)) {
      rewriter.replaceOp(reduction, *op);
      return success();
    }

    return rewriter.notifyMatchFailure(reduction, "cannot legalize op");
  }
};

struct HIVMBroadcastScalarOpLowering
    : public ConvertOpToLLVMPattern<VFBroadcastScalarOp> {
  explicit HIVMBroadcastScalarOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFBroadcastScalarOp>(converter) {}

  LogicalResult
  matchAndRewrite(VFBroadcastScalarOp broadcast,
                  VFBroadcastScalarOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = broadcast.getLoc();
    Value src = adaptor.getSrc();
    VectorType vType = cast<VectorType>(broadcast.getResult().getType());
    Type srcType = src.getType();
    auto vlType = createVLVectorType(srcType);
    Value vbr;
    if (srcType.isSignlessInteger(1)) {
      // I. Convert I1 to I8
      // II. Broadcast I8 to vector<256*I8>
      // III. Create a zero vector<256*I8>
      // IV. The compare result of the two vector above is the broadcast result
      //     of src I1. The result type is vector<256*I1>, and saved in PReg.
      Value i8Value =
          rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI8Type(), src);
      auto v256Ty = VectorType::get({util::VL}, rewriter.getI8Type());
      auto vbrI8 = rewriter.create<VbrInstrOp>(loc, v256Ty, i8Value);
      Value zeroI8 =
          rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8IntegerAttr(0));
      auto vbrI8Zero = rewriter.create<VbrInstrOp>(loc, v256Ty, zeroI8);
      auto zeroI32 =
          rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
      auto vcmpTy = VectorType::get({util::VL}, rewriter.getI1Type());
      auto maskI32 = rewriter.create<hivm_regbaseintrins::PgeB8>(
          zeroI32.getLoc(), vcmpTy, zeroI32, zeroI32);
      auto vcmpOp = rewriter.create<VCmpSNeInstrOp>(loc, vcmpTy, vbrI8.getRes(),
                                                    vbrI8Zero.getRes(),
                                                    maskI32->getResults()[0]);
      vbr = rewriter
                .create<UnrealizedConversionCastOp>(loc, vType, vcmpOp.getRes())
                .getResult(0);
    } else if (srcType.isUnsignedInteger(8)) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isSignedInteger(8) || srcType.isSignlessInteger(8)) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isUnsignedInteger(16)) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isSignedInteger(16) || srcType.isSignlessInteger(16)) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isF16()) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isBF16()) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isUnsignedInteger(32)) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isSignedInteger(32) || srcType.isSignlessInteger(32)) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isF32()) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isFloat8E4M3FN() || srcType.isFloat8E5M2()) {
      Type i8Type = rewriter.getI8Type();
      VectorType i8VLType = createVLVectorType(i8Type);
      Value i8Src = rewriter.create<LLVM::BitcastOp>(loc, i8Type, src);
      Value i8Vbr = rewriter.create<VbrInstrOp>(loc, i8VLType, i8Src);
      vbr = rewriter.create<LLVM::BitcastOp>(loc, vlType, i8Vbr);
    } else {
      return rewriter.notifyMatchFailure(broadcast, "cannot legalize op");
    }
    if (!srcType.isSignlessInteger(1) && vlType != vType) {
      vbr = rewriter.create<UnrealizedConversionCastOp>(loc, vType, vbr)
                ->getResult(0);
    }
    rewriter.replaceOp(broadcast, vbr);
    return success();
  }
};

template <typename IntrOpTy>
static Value vdupLowerToIntrin(VFBroadcastVectorOp vecBrcOp,
                               ConversionPatternRewriter &rewriter) {
  Value vdup = nullptr;
  auto resType = vecBrcOp.getRes().getType();

  VectorType vecType = cast<VectorType>(resType);
  Type elemType = vecType.getElementType();
  VectorType newType = VectorType::get({getVectorSizeByElementType(elemType)}, elemType);

  auto src = vecBrcOp.getSrc();
  auto loc = vecBrcOp->getLoc();
  // TODO: predication mode merging
  auto pModeCst =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
  Type elementType = cast<VectorType>(src.getType()).getElementType();
  auto srcTy = createVLVectorType(elementType);
  src =
      rewriter.create<UnrealizedConversionCastOp>(loc, srcTy, src).getResult(0);
  Value mask = getVLRegValueOrSelf(vecBrcOp.getMask(), rewriter);
  if (elementType.isUnsignedInteger(8) || elementType.isSignedInteger(8) ||
      elementType.isSignlessInteger(8) || elementType.isUnsignedInteger(16) ||
      elementType.isSignedInteger(16) || elementType.isSignlessInteger(16) ||
      elementType.isF16() || elementType.isUnsignedInteger(32) ||
      elementType.isSignedInteger(32) || elementType.isSignlessInteger(32) ||
      elementType.isF32() || elementType.isBF16() ||
      isa<Float8E5M2Type>(elementType) || isa<Float8E4M3FNType>(elementType)) {
    vdup = rewriter.create<IntrOpTy>(loc, newType, src, mask, pModeCst);
  }
  return vdup;
}

struct HIVMVecBrcOpPattern
    : public ConvertOpToLLVMPattern<VFBroadcastVectorOp> {
  explicit HIVMVecBrcOpPattern(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFBroadcastVectorOp>(converter) {}

  LogicalResult
  matchAndRewrite(VFBroadcastVectorOp vecBrcOp,
                  VFBroadcastVectorOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value vdup;
    bool brcLowest = vecBrcOp.getLow();
    if (brcLowest)
      vdup = vdupLowerToIntrin<VdupZInstrOp>(vecBrcOp, rewriter);
    else
      vdup = vdupLowerToIntrin<VdupmZInstrOp>(vecBrcOp, rewriter);

    rewriter.replaceOp(vecBrcOp, vdup);
    return success();
  }
};

template <typename OpToBeConverted>
struct HIVMTypeConvertionOpLowering
    : public ConvertOpToLLVMPattern<OpToBeConverted> {
  explicit HIVMTypeConvertionOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<OpToBeConverted>(converter) {}

  LogicalResult
  matchAndRewrite(OpToBeConverted srcOp,
                  typename OpToBeConverted::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = srcOp.getLoc();
    Value src = srcOp.getSrc();
    VectorType resType = cast<VectorType>(srcOp.getResult().getType());
    VectorType srcType = cast<VectorType>(src.getType());
    Type outElemType = resType.getElementType();
    Type inElemType = srcType.getElementType();

    auto cstValue = [&](auto enumVal) -> Value {
      return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                               static_cast<uint32_t>(enumVal));
    };

    auto srcVLVectorTy = hivm_regbaseintrins::createVLVectorType(inElemType);
    auto dstVLVectorTy = hivm_regbaseintrins::createVLVectorType(outElemType);

    Value srcCasted =
        rewriter.create<UnrealizedConversionCastOp>(loc, srcVLVectorTy, src)
            ->getResult(0);

    Value zero = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    Value mask = getVLRegValueOrSelf(srcOp.getMask(), rewriter);
    // If the data have been arranged by ElementAlignment when load, do not need
    // to interleave/deinterleave when type cast, otherwise need.
    Operation *newOp = nullptr;
    if constexpr (std::is_same_v<OpToBeConverted, VFExtFOp>) {
      if (outElemType.isF32()) {
        if (inElemType.isBF16()) {
          newOp = rewriter.create<VcvtffBF162F32InstrOp>(
              loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getPart()));
        } else if (isa<Float8E4M3FNType>(inElemType)) {
          newOp = rewriter.create<VcvtffF8E4M32F32InstrOp>(
              loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getPart()));
        } else if (isa<Float8E5M2Type>(inElemType)) {
          newOp = rewriter.create<VcvtffF8E5M22F32InstrOp>(
              loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getPart()));
        } else if (inElemType.isF16()) {
          newOp = rewriter.create<VcvtffF162F32InstrOp>(
              loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getPart()));
        }
      }
    } else if constexpr (std::is_same_v<OpToBeConverted, VFExtSIOp>) {
      if (inElemType.isSignlessInteger(8) && outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VcvtiiS82S32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getPp()));
      else if (inElemType.isSignlessInteger(8) &&
               outElemType.isSignlessInteger(16))
        newOp = rewriter.create<VcvtiiS82S16InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getPart()));
      else if (inElemType.isSignlessInteger(16) &&
               outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VcvtiiS162S32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getPart()));
      else if (inElemType.isSignlessInteger(32) &&
               outElemType.isSignlessInteger(64))
        newOp = rewriter.create<VcvtiiS322S64InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getPart()));
    } else if constexpr (std::is_same_v<OpToBeConverted, VFExtUIOp>) {
      if (inElemType.isSignlessInteger(8) && outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VcvtiiU82U32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getPp()));
      else if (inElemType.isSignlessInteger(8) &&
               outElemType.isSignlessInteger(16))
        newOp = rewriter.create<VcvtiiU82U16InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getPart()));
      else if (inElemType.isSignlessInteger(16) &&
               outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VcvtiiU162U32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getPart()));
    } else if constexpr (std::is_same_v<OpToBeConverted, VFTruncIOp>) {
      if (inElemType.isSignlessInteger(64) &&
          outElemType.isSignlessInteger(32)) {
        newOp = rewriter.create<VcvtiiS642S32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
            cstValue(*srcOp.getPart()));
      } else if (inElemType.isSignlessInteger(32) &&
                 outElemType.isSignlessInteger(16)) {
        auto uniAttr = srcOp->getAttr("uni");
        if (!uniAttr) {
          newOp = rewriter.create<VcvtiiS322S16InstrOp>(
              loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
              cstValue(*srcOp.getPart()));
        } else {
          if (auto unsignedModeAttr =
                  dyn_cast<hivm::UnsignedModeAttr>(uniAttr)) {
            auto mode = unsignedModeAttr.getValue();
            switch (mode) {
            case hivm::UnsignedMode::SI2SI: {
              newOp = rewriter.create<VcvtiiS322S16InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPart()));
              break;
            }
            case hivm::UnsignedMode::SI2UI: {
              newOp = rewriter.create<VcvtiiS322U16InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPart()));
              break;
            }
            case hivm::UnsignedMode::UI2SI: {
              newOp = rewriter.create<VcvtiiU322S16InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPart()));
              break;
            }
            case hivm::UnsignedMode::UI2UI: {
              newOp = rewriter.create<VcvtiiU322U16InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPart()));
              break;
            }
            }
          } else {
            llvm::errs() << "Error: 'uni' attribute exists but is not a "
                            "hivm::UnsignedModeAttr!\n";
            return failure();
          }
        }
      } else if (inElemType.isSignlessInteger(16) &&
                 outElemType.isSignlessInteger(8)) {
        auto uniAttr = srcOp->getAttr("uni");
        if (!uniAttr) {
          newOp = rewriter.create<VcvtiiS162U8InstrOp>(
              loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
              cstValue(*srcOp.getPart()));
        } else {
          if (auto unsignedModeAttr =
                  dyn_cast<hivm::UnsignedModeAttr>(uniAttr)) {
            auto mode = unsignedModeAttr.getValue();
            switch (mode) {
            case hivm::UnsignedMode::SI2UI: {
              newOp = rewriter.create<VcvtiiS162U8InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPart()));
              break;
            }
            case hivm::UnsignedMode::UI2UI: {
              newOp = rewriter.create<VcvtiiU162U8InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPart()));
              break;
            }
            case hivm::UnsignedMode::SI2SI: {
              newOp = rewriter.create<VcvtiiS162U8InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPart()));
              break;
            }
            case hivm::UnsignedMode::UI2SI: {
              newOp = rewriter.create<VcvtiiS162U8InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPart()));
              break;
            }
            }
          } else {
            llvm::errs() << "Error: 'uni' attribute exists but is not a "
                            "hivm::UnsignedModeAttr!\n";
            return failure();
          }
        }
      } else if (inElemType.isSignlessInteger(32) &&
                 outElemType.isSignlessInteger(8)) {
        auto uniAttr = srcOp->getAttr("uni");
        if (!uniAttr) {
          newOp = rewriter.create<VcvtiiS322U8InstrOp>(
              loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
              cstValue(*srcOp.getPp()));
        } else {
          if (auto unsignedModeAttr =
                  dyn_cast<hivm::UnsignedModeAttr>(uniAttr)) {
            auto mode = unsignedModeAttr.getValue();
            switch (mode) {
            case hivm::UnsignedMode::SI2UI: {
              newOp = rewriter.create<VcvtiiS322U8InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPp()));
              break;
            }
            case hivm::UnsignedMode::UI2UI: {
              newOp = rewriter.create<VcvtiiU322U8InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPp()));
              break;
            }
            case hivm::UnsignedMode::SI2SI: {
              newOp = rewriter.create<VcvtiiS322U8InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPp()));
              break;
            }
            case hivm::UnsignedMode::UI2SI: {
              newOp = rewriter.create<VcvtiiS322U8InstrOp>(
                  loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getSat()),
                  cstValue(*srcOp.getPp()));
              break;
            }
            }
          } else {
            llvm::errs() << "Error: 'uni' attribute exists but is not a "
                            "hivm::UnsignedModeAttr!\n";
            return failure();
          }
        }
      }
    } else if constexpr (std::is_same_v<OpToBeConverted, VFTruncFOp>) {
      if (outElemType.isBF16())
        newOp = rewriter.create<VcvtffF322BF16InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(srcOp.getSat()), cstValue(srcOp.getPart()));
      else if (outElemType.isF16())
        newOp = rewriter.create<VcvtffF322F16InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(srcOp.getSat()), cstValue(srcOp.getPart()));
      else if (isa<Float8E4M3FNType>(outElemType))
        newOp = rewriter.create<VcvtffF322F8E4M3InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(srcOp.getSat()), cstValue(srcOp.getPart()));
      else if (isa<Float8E5M2Type>(outElemType))
        newOp = rewriter.create<VcvtffF322F8E5M2InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(srcOp.getSat()), cstValue(srcOp.getPart()));
    } else if constexpr (std::is_same_v<OpToBeConverted, VFFpToSIntOp>) {
      if (srcOp.getRnd() == hivm::RoundMode::TRUNCWITHOVERFLOW) {
        auto rintMode = hivm::RoundMode::TRUNC;
        srcOp.setRnd(rintMode);
      }
      if (inElemType.isF32() && outElemType.isSignlessInteger(64))
        newOp = rewriter.create<VcvtfiF322S64InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(*srcOp.getSat()), cstValue(*srcOp.getPart()));
      else if (inElemType.isF32() && outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VcvtfiF322S32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(*srcOp.getSat()));
      else if (inElemType.isF32() && outElemType.isSignlessInteger(16)) {
        newOp = rewriter.create<VcvtfiF322S16InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(*srcOp.getSat()), cstValue(*srcOp.getPart()));
      } else if (inElemType.isF16() && outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VcvtfiF162S32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(*srcOp.getPart()));
      else if (inElemType.isF16() && outElemType.isSignlessInteger(16))
        newOp = rewriter.create<VcvtfiF162S16InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(*srcOp.getSat()));
      else if (inElemType.isF16() && outElemType.isSignlessInteger(8)) {
        newOp = rewriter.create<VcvtfiF162S8InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(*srcOp.getSat()), cstValue(*srcOp.getPart()));
      } else if (inElemType.isBF16() && outElemType.isSignlessInteger(32))
        newOp = rewriter.create<VcvtfiBF162S32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(*srcOp.getSat()), cstValue(*srcOp.getPart()));
    } else if constexpr (std::is_same_v<OpToBeConverted, VFFpToUIntOp>) {
      if (srcOp.getRnd() == hivm::RoundMode::TRUNCWITHOVERFLOW) {
        auto rintMode = hivm::RoundMode::TRUNC;
        srcOp.setRnd(rintMode);
      }
      if (inElemType.isF16() && outElemType.isSignlessInteger(8)) {
        newOp = rewriter.create<VcvtfiF162U8InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(srcOp.getSat()), cstValue(srcOp.getPart()));
      }
    } else if constexpr (std::is_same_v<OpToBeConverted, VFSIntToFpOp>) {
      if (inElemType.isSignlessInteger(8) && outElemType.isF16())
        newOp = rewriter.create<VcvtifS82F16InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getPart()));
      else if (inElemType.isSignlessInteger(16) && outElemType.isF16())
        newOp = rewriter.create<VcvtifS162F16InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getRnd()));
      else if (inElemType.isSignlessInteger(16) && outElemType.isF32())
        newOp = rewriter.create<VcvtifS162F32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getPart()));
      else if (inElemType.isSignlessInteger(32) && outElemType.isF32())
        newOp = rewriter.create<VcvtifS322F32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getRnd()));
      else if (inElemType.isSignlessInteger(64) && outElemType.isF32()) {
        newOp = rewriter.create<VcvtifS642F32InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(*srcOp.getRnd()),
            cstValue(*srcOp.getPart()));
      }
    } else if constexpr (std::is_same_v<OpToBeConverted, VFUIntToFpOp>) {
      if (inElemType.isSignlessInteger(8) && outElemType.isF16())
        newOp = rewriter.create<VcvtifU82F16InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getPart()));
    }
    if (!newOp)
      llvm::report_fatal_error("Unsupported type conversion op.");

    UnrealizedConversionCastOp resultCasted =
        rewriter.create<UnrealizedConversionCastOp>(loc, resType,
                                                    newOp->getResult(0));
    rewriter.replaceOp(srcOp, resultCasted);
    return success();
  }
};

struct HIVMVtrcOpLowering : public ConvertOpToLLVMPattern<VFVtrcOp> {
  explicit HIVMVtrcOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFVtrcOp>(converter) {}

  LogicalResult
  matchAndRewrite(VFVtrcOp op, VFVtrcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op->getOperand(0);
    VectorType resType = cast<VectorType>(op.getResult().getType());
    Type elemResType = resType.getElementType();

    if (!elemResType.isF16() && !elemResType.isBF16() && !elemResType.isF32()) {
      return failure();
    }

    VectorType srcType = cast<VectorType>(src.getType());
    Type srcElemType = srcType.getElementType();
    auto srcVLVectorTy = hivm_regbaseintrins::createVLVectorType(srcElemType);
    Value srcCasted =
        rewriter.create<UnrealizedConversionCastOp>(loc, srcVLVectorTy, src)
            ->getResult(0);
    auto resVLVectorTy = hivm_regbaseintrins::createVLVectorType(elemResType);
    Value roundingMode = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), static_cast<uint32_t>(op.getRnd()));
    Operation *vtrc = nullptr;
    Value mask = getVLRegValueOrSelf(op.getMask(), rewriter);
    if (elemResType.isF16()) {
      vtrc = rewriter.create<VtrcV128F16InstrOp>(loc, resVLVectorTy, srcCasted,
                                                 roundingMode, mask);
    } else if (elemResType.isBF16()) {
      vtrc = rewriter.create<VtrcV128BF16InstrOp>(loc, resVLVectorTy, srcCasted,
                                                  roundingMode, mask);
    } else if (elemResType.isF32()) {
      vtrc = rewriter.create<VtrcV64F32InstrOp>(loc, resVLVectorTy, srcCasted,
                                                roundingMode, mask);
    } else {
      return failure();
    }
    UnrealizedConversionCastOp resCasted =
        rewriter.create<UnrealizedConversionCastOp>(loc, resType,
                                                    vtrc->getResult(0));
    rewriter.replaceOp(op, resCasted);
    return success();
  }
};

static Value ConvertScalarToI16(Location loc,
                                ConversionPatternRewriter &rewriter,
                                Value scalar, Type elementType) {
  if (elementType.isInteger(16))
    return scalar;
  if (elementType.getIntOrFloatBitWidth() > 16) {
    return rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), scalar)
        .getResult();
  }
  if (auto constOp = scalar.getDefiningOp<arith::ConstantOp>()) {
    auto attr = constOp.getValue();
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
      int64_t value = intAttr.getValue().getSExtValue();
      auto i16Type = rewriter.getI16Type();
      return rewriter
          .create<arith::ConstantOp>(loc, i16Type,
                                     rewriter.getIntegerAttr(i16Type, value))
          .getResult();
    }
  }

  // If not a constant
  if (elementType.isSignedInteger() || elementType.isSignlessInteger()) {
    auto i16Type = rewriter.getI16Type();
    return rewriter.create<arith::ExtSIOp>(loc, i16Type, scalar).getResult();
  }

  if (elementType.isUnsignedInteger()) {
    auto i16Type = rewriter.getI16Type();
    return rewriter.create<arith::ExtUIOp>(loc, i16Type, scalar).getResult();
  }

  return scalar;
}

template <typename SourceOp>
struct HIVMVShiftOpLowering : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src1 = op.getOperand(0);
    Value src2 = op.getOperand(1);
    VectorType resType = cast<VectorType>(op.getResult().getType());
    Type elemType = resType.getElementType();
    Value sign = op.getOperand(3);
    bool isSigned = false;

    if (auto constantOp = sign.getDefiningOp<arith::ConstantOp>()) {
      Attribute valueAttr = constantOp.getValue();
      if (auto boolAttr = dyn_cast<BoolAttr>(valueAttr)) {
        isSigned = boolAttr.getValue();
      } else if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
        isSigned = intAttr.getValue().getBoolValue();
      }
    }
    Value mask = getVLRegValueOrSelf(adaptor.getMask(), rewriter);

    uint64_t totalSize = static_cast<uint64_t>(resType.getNumElements());
    auto dataWidth = elemType.getIntOrFloatBitWidth();
    auto vlLength = util::VL_BITS / dataWidth;

    VectorType oriResType = resType;
    resType = VectorType::get(SmallVector<int64_t>{vlLength}, elemType);

    if (totalSize != vlLength) {
      src1 = rewriter.create<UnrealizedConversionCastOp>(loc, resType, src1)
                 .getResult(0);
      if constexpr (!isScalar) // isVectorShift
        src2 = rewriter.create<UnrealizedConversionCastOp>(loc, resType, src2)
                   .getResult(0);
    }

    // Scalar VShiftOp literal has to be I16
    if constexpr (isScalar)
      src2 = ConvertScalarToI16(loc, rewriter, src2, elemType);

    auto createShiftOp =
        [&rewriter, op, loc, resType, src1, src2, mask, dataWidth, vlLength,
         isSigned](auto s8, auto u8, auto s16, auto u16, auto s32,
                   auto u32) -> FailureOr<Operation *> {
      using S8Op = typename decltype(s8)::type;
      using U8Op = typename decltype(u8)::type;
      using S16Op = typename decltype(s16)::type;
      using U16Op = typename decltype(u16)::type;
      using S32Op = typename decltype(s32)::type;
      using U32Op = typename decltype(u32)::type;

      if (dataWidth == 8 && vlLength == 256) {
        return isSigned ? rewriter.create<S8Op>(loc, resType, src1, src2, mask)
                        : rewriter.create<U8Op>(loc, resType, src1, src2, mask);
      } else if (dataWidth == 16 && vlLength == 128) {
        return isSigned
                   ? rewriter.create<S16Op>(loc, resType, src1, src2, mask)
                   : rewriter.create<U16Op>(loc, resType, src1, src2, mask);
      } else if (dataWidth == 32 && vlLength == 64) {
        return isSigned
                   ? rewriter.create<S32Op>(loc, resType, src1, src2, mask)
                   : rewriter.create<U32Op>(loc, resType, src1, src2, mask);
      }
      return rewriter.notifyMatchFailure(
          op, "Unsupported vector type/width combination");
    };

    FailureOr<Operation *> vshiftOrFailure;
    if constexpr (isLeft) {
      if constexpr (isScalar) {
        // Left Shift Scalar (VFShlsOp)
        vshiftOrFailure =
            createShiftOp(OpTag<VshlsSXInstrOp>{}, OpTag<VshlsUXInstrOp>{},
                          OpTag<VshlsSXInstrOp>{}, OpTag<VshlsUXInstrOp>{},
                          OpTag<VshlsSXInstrOp>{}, OpTag<VshlsUXInstrOp>{});
      } else {
        // Left Shift Vector (VFShlOp)
        vshiftOrFailure =
            createShiftOp(OpTag<VshlSXInstrOp>{}, OpTag<VshlUXInstrOp>{},
                          OpTag<VshlSXInstrOp>{}, OpTag<VshlUXInstrOp>{},
                          OpTag<VshlSXInstrOp>{}, OpTag<VshlUXInstrOp>{});
      }
    } else {
      if constexpr (isScalar) {
        // Right Shift Scalar (VFShrsOp)
        vshiftOrFailure =
            createShiftOp(OpTag<VshrsSXInstrOp>{}, OpTag<VshrsUXInstrOp>{},
                          OpTag<VshrsSXInstrOp>{}, OpTag<VshrsUXInstrOp>{},
                          OpTag<VshrsSXInstrOp>{}, OpTag<VshrsUXInstrOp>{});
      } else {
        // Right Shift Vector (VFShrOp)
        vshiftOrFailure =
            createShiftOp(OpTag<VshrSXInstrOp>{}, OpTag<VshrUXInstrOp>{},
                          OpTag<VshrSXInstrOp>{}, OpTag<VshrUXInstrOp>{},
                          OpTag<VshrSXInstrOp>{}, OpTag<VshrUXInstrOp>{});
      }
    }

    if (failed(vshiftOrFailure)) {
      return failure();
    }

    Operation *vshift = *vshiftOrFailure;

    if (oriResType != resType) {
      Operation *ucc = rewriter.create<UnrealizedConversionCastOp>(
          loc, oriResType, vshift->getResult(0));
      rewriter.replaceOp(op, ucc);
    } else {
      rewriter.replaceOp(op, vshift);
    }
    return success();
  }

private:
  static constexpr bool isScalar =
      std::is_same_v<SourceOp, hivmave::VFShrsOp> ||
      std::is_same_v<SourceOp, hivmave::VFShlsOp>;

  static constexpr bool isLeft = std::is_same_v<SourceOp, hivmave::VFShlOp> ||
                                 std::is_same_v<SourceOp, hivmave::VFShlsOp>;
};

struct HIVMInterleaveOpLowering
    : public ConvertOpToLLVMPattern<hivmave::VFInterleaveOp> {
  explicit HIVMInterleaveOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<hivmave::VFInterleaveOp>(converter) {}

  LogicalResult
  matchAndRewrite(hivmave::VFInterleaveOp op,
                  hivmave::VFInterleaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto [vecType, elemType, numElements] = getVectorInfo(op);
    if (!vecType)
      return rewriter.notifyMatchFailure(op, "failed to get vector type");

    Value src1 = adaptor.getSrc1();
    Value src2 = adaptor.getSrc2();

    Type llvmVecType = getTypeConverter()->convertType(vecType);
    if (!llvmVecType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert vector type to LLVM type");
    auto llvmVecVLType = createVLVectorType(elemType);
    UnrealizedConversionCastOp src1Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecVLType, src1);
    UnrealizedConversionCastOp src2Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecVLType, src2);
    Type intlvType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {llvmVecVLType, llvmVecVLType});

    Operation *intrinOp = createInterleaveIntrinsic(
        rewriter, loc, intlvType, src1Casted->getResult(0),
        src2Casted->getResult(0), elemType);
    if (!intrinOp)
      return rewriter.notifyMatchFailure(
          op, "Unsupported vector type for interleave");

    Value agg = intrinOp->getResult(0);
    auto [res0, res1] = extractResults(rewriter, loc, llvmVecVLType, agg);
    UnrealizedConversionCastOp res0Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecType, res0);
    UnrealizedConversionCastOp res1Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecType, res1);
    rewriter.replaceOp(op,
                       {res0Casted->getResult(0), res1Casted->getResult(0)});
    return success();
  }

private:
  std::tuple<VectorType, Type, int64_t>
  getVectorInfo(hivmave::VFInterleaveOp op) const {
    VectorType vecType = cast<VectorType>(op.getSrc1().getType());
    Type elemType = vecType.getElementType();
    int64_t numElements = vecType.getNumElements();
    return {vecType, elemType, numElements};
  }

  Operation *createInterleaveIntrinsic(ConversionPatternRewriter &rewriter,
                                       Location loc, Type intlvType, Value src1,
                                       Value src2, Type elemType) const {
    if (elemType.isInteger(32) || elemType.isInteger(16) ||
        elemType.isInteger(8) || elemType.isF32() || elemType.isF16() ||
        elemType.isBF16() || isa<Float8E4M3FNType>(elemType) ||
        isa<Float8E5M2Type>(elemType)) {
      return rewriter.create<VintlvInstrOp>(loc, intlvType, src1, src2);
    }
    return nullptr;
  }

  std::pair<Value, Value> extractResults(ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmVecType,
                                         Value agg) const {
    SmallVector<int64_t, 1> idx0{0};
    SmallVector<int64_t, 1> idx1{1};
    Value res0 =
        rewriter.create<LLVM::ExtractValueOp>(loc, llvmVecType, agg, idx0);
    Value res1 =
        rewriter.create<LLVM::ExtractValueOp>(loc, llvmVecType, agg, idx1);
    return {res0, res1};
  }
};

struct HIVMDeInterleaveOpLowering
    : public ConvertOpToLLVMPattern<hivmave::VFDeInterleaveOp> {
  explicit HIVMDeInterleaveOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<hivmave::VFDeInterleaveOp>(converter) {}

  LogicalResult
  matchAndRewrite(hivmave::VFDeInterleaveOp op,
                  hivmave::VFDeInterleaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto [vecType, elemType, numElements] = getVectorInfo(op);
    if (!vecType)
      return rewriter.notifyMatchFailure(op, "failed to get vector type");

    Value src1 = adaptor.getSrc1();
    Value src2 = adaptor.getSrc2();

    Type llvmVecType = getTypeConverter()->convertType(vecType);
    if (!llvmVecType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert vector type to LLVM type");

    auto llvmVecVLType = createVLVectorType(elemType);
    UnrealizedConversionCastOp src1Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecVLType, src1);
    UnrealizedConversionCastOp src2Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecVLType, src2);

    Type deintlvType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {llvmVecVLType, llvmVecVLType});

    Operation *deintrinOp = createDeinterleaveIntrinsic(
        rewriter, loc, deintlvType, src1Casted->getResult(0),
        src2Casted->getResult(0), elemType);
    if (!deintrinOp)
      return rewriter.notifyMatchFailure(
          op, "Unsupported vector type for interleave");

    Value agg = deintrinOp->getResult(0);
    auto [res0, res1] = extractResults(rewriter, loc, llvmVecVLType, agg);

    UnrealizedConversionCastOp res0Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecType, res0);
    UnrealizedConversionCastOp res1Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecType, res1);
    rewriter.replaceOp(op,
                       {res0Casted->getResult(0), res1Casted->getResult(0)});
    return success();
  }

private:
  std::tuple<VectorType, Type, int64_t>
  getVectorInfo(hivmave::VFDeInterleaveOp op) const {
    VectorType vecType = cast<VectorType>(op.getSrc1().getType());
    Type elemType = vecType.getElementType();
    int64_t numElements = vecType.getNumElements();
    return {vecType, elemType, numElements};
  }

  Operation *createDeinterleaveIntrinsic(ConversionPatternRewriter &rewriter,
                                         Location loc, Type deintlvType,
                                         Value src1, Value src2,
                                         Type elemType) const {
    if (elemType.isInteger(32) || elemType.isInteger(16) ||
        elemType.isInteger(8) || elemType.isF32() || elemType.isF16() ||
        elemType.isBF16() || isa<Float8E4M3FNType>(elemType) ||
        isa<Float8E5M2Type>(elemType)) {
      return rewriter.create<VDintlvInstrOp>(loc, deintlvType, src1, src2);
    }
    return nullptr;
  }

  std::pair<Value, Value> extractResults(ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmVecType,
                                         Value agg) const {
    SmallVector<int64_t, 1> idx0{0};
    SmallVector<int64_t, 1> idx1{1};
    Value res0 =
        rewriter.create<LLVM::ExtractValueOp>(loc, llvmVecType, agg, idx0);
    Value res1 =
        rewriter.create<LLVM::ExtractValueOp>(loc, llvmVecType, agg, idx1);
    return {res0, res1};
  }
};

struct HIVMVmullOpLowering : public ConvertOpToLLVMPattern<hivmave::VFVMULLOp> {
  explicit HIVMVmullOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<hivmave::VFVMULLOp>(converter) {}

  LogicalResult
  matchAndRewrite(hivmave::VFVMULLOp op, hivmave::VFVMULLOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto [vecType, elemType, numElements] = getVectorInfo(op);
    if (!vecType)
      return rewriter.notifyMatchFailure(op, "failed to get vector type");

    Value src1 = adaptor.getSrc1();
    Value src2 = adaptor.getSrc2();

    Type llvmVecType = getTypeConverter()->convertType(vecType);
    if (!llvmVecType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert vector type to LLVM type");

    auto llvmVecVLType = createVLVectorType(elemType);
    UnrealizedConversionCastOp src1Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecVLType, src1);
    UnrealizedConversionCastOp src2Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecVLType, src2);

    Value mask = getVLRegValueOrSelf(op.getMask(), rewriter);
    Type vmullType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {llvmVecVLType, llvmVecVLType});

    Operation *vmullOp = createVmullIntrinsic(
        rewriter, loc, vmullType, src1Casted->getResult(0),
        src2Casted->getResult(0), elemType, mask, op.getCast());
    if (!vmullOp)
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported vector type for vmull");

    Value agg = vmullOp->getResult(0);
    auto [res0, res1] = extractResults(rewriter, loc, llvmVecVLType, agg);
    UnrealizedConversionCastOp res0Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecType, res0);
    UnrealizedConversionCastOp res1Casted =
        rewriter.create<UnrealizedConversionCastOp>(loc, llvmVecType, res1);
    rewriter.replaceOp(op,
                       {res0Casted->getResult(0), res1Casted->getResult(0)});
    return success();
  }

private:
  std::tuple<VectorType, Type, int64_t>
  getVectorInfo(hivmave::VFVMULLOp op) const {
    VectorType vecType = cast<VectorType>(op.getSrc1().getType());
    Type elemType = vecType.getElementType();
    int64_t numElements = vecType.getNumElements();
    return {vecType, elemType, numElements};
  }

  Operation *createVmullIntrinsic(ConversionPatternRewriter &rewriter,
                                  Location loc, Type vecType, Value src1,
                                  Value src2, Type elemType, Value mask,
                                  TypeFn cast) const {
    if (!elemType.isInteger(32))
      return nullptr;

    if (cast == TypeFn::cast_signed) {
      return rewriter.create<VmullV64S32InstrOp>(loc, vecType, src1, src2,
                                                 mask);
    }
    if (cast == TypeFn::cast_unsigned) {
      return rewriter.create<VmullV64U32InstrOp>(loc, vecType, src1, src2,
                                                 mask);
    }
    return nullptr;
  }

  std::pair<Value, Value> extractResults(ConversionPatternRewriter &rewriter,
                                         Location loc, Type llvmVecType,
                                         Value agg) const {
    SmallVector<int64_t, 1> idx0{0};
    SmallVector<int64_t, 1> idx1{1};
    Value res0 =
        rewriter.create<LLVM::ExtractValueOp>(loc, llvmVecType, agg, idx0);
    Value res1 =
        rewriter.create<LLVM::ExtractValueOp>(loc, llvmVecType, agg, idx1);
    return {res0, res1};
  }
};

struct VectorBitCastLowering
    : public ConvertOpToLLVMPattern<vector::BitCastOp> {
  explicit VectorBitCastLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<vector::BitCastOp>(converter) {}

  LogicalResult
  matchAndRewrite(vector::BitCastOp op, vector::BitCastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSource();
    auto dst = op.getResult();
    VectorType srcVecTy = dyn_cast<VectorType>(src.getType());
    VectorType dstVecTy = dyn_cast<VectorType>(dst.getType());
    if (!srcVecTy || !dstVecTy)
      return failure();
    VectorType srcVecRegTy = createVLVectorType(srcVecTy.getElementType());
    VectorType dstVecRegTy = createVLVectorType(dstVecTy.getElementType());
    Value castedSrc = src;
    if (srcVecRegTy != srcVecTy) {
      castedSrc =
          rewriter.create<UnrealizedConversionCastOp>(loc, srcVecRegTy, src)
              ->getResult(0);
    }
    Value llvmBitCast =
        rewriter.create<LLVM::BitcastOp>(loc, dstVecRegTy, castedSrc);

    if (dstVecRegTy != dstVecTy) {
      llvmBitCast =
          rewriter
              .create<UnrealizedConversionCastOp>(loc, dstVecTy, llvmBitCast)
              ->getResult(0);
    }
    rewriter.replaceOp(op, llvmBitCast);
    return success();
  }
};
struct HIVMVSlideOpLowering : public ConvertOpToLLVMPattern<VFSlideOp> {
  explicit HIVMVSlideOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFSlideOp>(converter) {}

  LogicalResult
  matchAndRewrite(VFSlideOp op, VFSlideOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src1 = adaptor.getSrc1();
    Value src2 = adaptor.getSrc2();
    auto slideAmount = adaptor.getSlideAmount();
    VectorType resType = cast<VectorType>(op.getResult().getType());
    Type elemType = resType.getElementType();

    Operation *vslide = nullptr;
    if (elemType.isSignedInteger(32) || elemType.isSignlessInteger(32) ||
        elemType.isUnsignedInteger(32) || elemType.isSignedInteger(16) ||
        elemType.isSignlessInteger(16) || elemType.isUnsignedInteger(16) ||
        elemType.isSignedInteger(8) || elemType.isSignlessInteger(8) ||
        elemType.isUnsignedInteger(8) || elemType.isF32() || elemType.isF16()) {
      vslide =
          rewriter.create<VSlideInstrOp>(loc, resType, src1, src2, slideAmount);
    } else {
      return rewriter.notifyMatchFailure(vslide, "cannot legalize op");
    }

    rewriter.replaceOp(op, vslide->getResults());
    return success();
  }
};

struct HIVMPnotOpLowering : public ConvertOpToLLVMPattern<PregNotOp> {
  explicit HIVMPnotOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<PregNotOp>(converter) {}
  LogicalResult
  matchAndRewrite(PregNotOp op, PregNotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src1 = adaptor.getSrc();
    VectorType resType = cast<VectorType>(op.getResult().getType());
    Operation *newOp = nullptr;
    Value newsrc1 = getVLRegValueOrSelf(src1, rewriter);
    auto newOpResTy = createVLVectorType(rewriter.getI1Type());

    Value mask = getVLRegValueOrSelf(adaptor.getMask(), rewriter);
    newOp = rewriter.create<PnotB8InstrOp>(loc, newOpResTy, newsrc1, mask);
    if (resType != newOpResTy) {
      newOp = rewriter.create<UnrealizedConversionCastOp>(loc, resType,
                                                          newOp->getResults());
    }
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct HIVMExpdifOpLowering
    : public ConvertOpToLLVMPattern<hivmave::VFExpdifOp> {
  explicit HIVMExpdifOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFExpdifOp>(converter) {}
  LogicalResult
  matchAndRewrite(VFExpdifOp op, VFExpdifOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    VectorType lhsVectorTy = cast<VectorType>(lhs.getType());
    VectorType rhsVectorTy = cast<VectorType>(rhs.getType());
    if (lhsVectorTy != rhsVectorTy)
      return rewriter.notifyMatchFailure(
          op, "the vector type of lhs and rhs must be equal");
    Type elementType = lhsVectorTy.getElementType();

    VectorType vecType = lhsVectorTy;
    auto totalSize = lhsVectorTy.getNumElements();
    auto dataWidth = elementType.getIntOrFloatBitWidth();
    auto vlLength = util::VL_BITS / dataWidth;
    VectorType oriVecType = vecType;
    if (totalSize != vlLength) {
      vecType = VectorType::get(SmallVector<int64_t>{vlLength}, elementType);
      lhs = rewriter.create<UnrealizedConversionCastOp>(loc, vecType, lhs)
                .getResult(0);
      rhs = rewriter.create<UnrealizedConversionCastOp>(loc, vecType, rhs)
                .getResult(0);
    }
    Value mask = getVLRegValueOrSelf(op.getMask(), rewriter);
    Value cstZero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    Value res;
    if (elementType.isF32()) {
      res = rewriter.create<VExpdifInstrOp>(loc, vecType, lhs, rhs, mask,
                                            cstZero);
    } else if (elementType.isF16()) {
      res = rewriter.create<VExpdifInstrOp>(loc, vecType, lhs, rhs, mask,
                                            cstZero);
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported element type in expdif");
    }
    if (oriVecType != vecType) {
      Operation *ucc =
          rewriter.create<UnrealizedConversionCastOp>(loc, oriVecType, res);
      rewriter.replaceOp(op, ucc);
    } else {
      rewriter.replaceOp(op, res);
    }
    return success();
  }
};

template <typename OpTy, typename IntrV256S8OpTy, typename IntrV256U8OpTy,
          typename IntrV128S16OpTy, typename IntrV128U16OpTy,
          typename IntrV64S32OpTy, typename IntrV64U32OpTy,
          typename IntrV128F16OpTy, typename IntrV64F32OpTy,
          typename IntrV128BF16OpTy>
struct TenaryLowerToIntrinsic : public ConvertOpToLLVMPattern<OpTy> {
  explicit TenaryLowerToIntrinsic(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<OpTy>(converter) {}
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value src1 = op.getSrc1();
    Value src2 = op.getSrc2();
    Value src3 = op.getSrc3();
    VectorType src1VectorTy = cast<VectorType>(src1.getType());
    VectorType src2VectorTy = cast<VectorType>(src2.getType());
    VectorType src3VectorTy = cast<VectorType>(src3.getType());

    if (src1VectorTy != src2VectorTy || src2VectorTy != src3VectorTy)
      return rewriter.notifyMatchFailure(
          op, "the vector type of all source must be equal.");

    Type elementType = src1VectorTy.getElementType();
    VectorType vecType = src1VectorTy;
    uint64_t totalSize = static_cast<uint64_t>(src1VectorTy.getNumElements());
    auto dataWidth = elementType.getIntOrFloatBitWidth();
    auto vlLength = util::VL_BITS / dataWidth;
    VectorType oriVecType = vecType;

    if (totalSize != vlLength) {
      vecType = VectorType::get(SmallVector<int64_t>{vlLength}, elementType);
      src1 = rewriter.create<UnrealizedConversionCastOp>(loc, vecType, src1)
                 .getResult(0);
      src2 = rewriter.create<UnrealizedConversionCastOp>(loc, vecType, src2)
                 .getResult(0);
      src3 = rewriter.create<UnrealizedConversionCastOp>(loc, vecType, src3)
                 .getResult(0);
    }

    Value mask = getVLRegValueOrSelf(op.getMask(), rewriter);
    Value res;
    if (elementType.isUnsignedInteger(8)) {
      res =
          rewriter.create<IntrV256U8OpTy>(loc, vecType, src1, src2, src3, mask);
    } else if (elementType.isSignedInteger(8) ||
               elementType.isSignlessInteger(8)) {
      res =
          rewriter.create<IntrV256S8OpTy>(loc, vecType, src1, src2, src3, mask);
    } else if (elementType.isUnsignedInteger(16)) {
      res = rewriter.create<IntrV128U16OpTy>(loc, vecType, src1, src2, src3,
                                             mask);
    } else if (elementType.isSignedInteger(16) ||
               elementType.isSignlessInteger(16)) {
      res = rewriter.create<IntrV128S16OpTy>(loc, vecType, src1, src2, src3,
                                             mask);
    } else if (elementType.isUnsignedInteger(32)) {
      res =
          rewriter.create<IntrV64U32OpTy>(loc, vecType, src1, src2, src3, mask);
    } else if (elementType.isSignedInteger(32) ||
               elementType.isSignlessInteger(32)) {
      res =
          rewriter.create<IntrV64S32OpTy>(loc, vecType, src1, src2, src3, mask);
    } else if (elementType.isF32()) {
      res =
          rewriter.create<IntrV64F32OpTy>(loc, vecType, src1, src2, src3, mask);
    } else if (elementType.isF16()) {
      res = rewriter.create<IntrV128F16OpTy>(loc, vecType, src1, src2, src3,
                                             mask);
    } else if (elementType.isBF16()) {
      res = rewriter.create<IntrV128BF16OpTy>(loc, vecType, src1, src2, src3,
                                              mask);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Unsupported element type in Tenaryop.");
    }

    if (oriVecType != vecType) {
      Operation *ucc =
          rewriter.create<UnrealizedConversionCastOp>(loc, oriVecType, res);
      rewriter.replaceOp(op, ucc);
    } else {
      rewriter.replaceOp(op, res);
    }
    return success();
  }
};

template <typename OpTy, typename IntrV256S8OpTy, typename IntrV256U8OpTy,
          typename IntrV128S16OpTy, typename IntrV128U16OpTy,
          typename IntrV64S32OpTy, typename IntrV64U32OpTy,
          typename IntrV128F16OpTy, typename IntrV64F32OpTy,
          typename IntrV128BF16OpTy>
struct TenaryRegEntry {
  using MainOp = OpTy;
  using IntrV256S8Op = IntrV256S8OpTy;
  using IntrV256U8Op = IntrV256U8OpTy;
  using IntrV128S16Op = IntrV128S16OpTy;
  using IntrV128U16Op = IntrV128U16OpTy;
  using IntrV64S32Op = IntrV64S32OpTy;
  using IntrV64U32Op = IntrV64U32OpTy;
  using IntrV128F16Op = IntrV128F16OpTy;
  using IntrV64F32Op = IntrV64F32OpTy;
  using IntrV128BF16Op = IntrV128BF16OpTy;
};

template <typename... Args> struct TenaryRegistryImpl {
  static void registerPatterns(LLVMTypeConverter &Converter,
                               RewritePatternSet &patterns) {
    patterns.add<TenaryLowerToIntrinsic<
        typename Args::MainOp, typename Args::IntrV256S8Op,
        typename Args::IntrV256U8Op, typename Args::IntrV128S16Op,
        typename Args::IntrV128U16Op, typename Args::IntrV64S32Op,
        typename Args::IntrV64U32Op, typename Args::IntrV128F16Op,
        typename Args::IntrV64F32Op, typename Args::IntrV128BF16Op>...>(
        Converter);
  }
};

using TenaryRegistry = TenaryRegistryImpl<TenaryRegEntry<
    hivmave::VFMulaOp, VunsupportedTenaryInstrOp, VunsupportedTenaryInstrOp,
    VmulaSMInstrOp, VmulaUMInstrOp, VmulaSMInstrOp, VmulaUMInstrOp,
    VmulaSMInstrOp, VmulaSMInstrOp, VmulaSMInstrOp>>;

template <typename OpToBeConverted>
struct HIVMPredicateBinaryLogicOpLowering
    : public ConvertOpToLLVMPattern<OpToBeConverted> {
  explicit HIVMPredicateBinaryLogicOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<OpToBeConverted>(converter) {}

  LogicalResult
  matchAndRewrite(OpToBeConverted op, typename OpToBeConverted::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType resType = cast<VectorType>(op.getResult().getType());
    Operation *newOp = nullptr;
    Value mask = getVLRegValueOrSelf(adaptor.getMask(), rewriter);

    Value oldLhs = op.getLhs();
    Value oldRhs = op.getRhs();
    Value adaptLhs = adaptor.getLhs();
    Value adaptRhs = adaptor.getRhs();
    Value lhs = getVLRegValueOrSelf(adaptLhs, rewriter);
    Value rhs = getVLRegValueOrSelf(adaptRhs, rewriter);
    auto newOpResTy = createVLVectorType(rewriter.getI1Type());
    // Handle VFPAndOp, VFPOrOp, VFPXorOp
    if constexpr (std::is_same_v<OpToBeConverted, PregAndOp> ||
                  std::is_same_v<OpToBeConverted, PregOrOp> ||
                  std::is_same_v<OpToBeConverted, PregXorOp>) {

      if constexpr (std::is_same_v<OpToBeConverted, PregAndOp>)
        newOp = rewriter.create<PandB8InstrOp>(loc, newOpResTy, lhs, rhs, mask);
      else if constexpr (std::is_same_v<OpToBeConverted, PregOrOp>)
        newOp = rewriter.create<PorB8InstrOp>(loc, newOpResTy, lhs, rhs, mask);
      else if constexpr (std::is_same_v<OpToBeConverted, PregXorOp>)
        newOp = rewriter.create<PxorB8InstrOp>(loc, newOpResTy, lhs, rhs, mask);
    } else {
      llvm::report_fatal_error("Unsupported operation type.");
      return failure();
    }
    if (resType != newOpResTy) {
      newOp = rewriter.create<UnrealizedConversionCastOp>(loc, resType,
                                                          newOp->getResults());
    }
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct HIVMVCIOpLowering : public ConvertOpToLLVMPattern<VFVCIOp> {
  explicit HIVMVCIOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<VFVCIOp>(converter) {}

  LogicalResult
  matchAndRewrite(VFVCIOp op, VFVCIOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // convert the starting index to i8/i16/i32/f16/f32 depending on output type
    VectorType resType = cast<VectorType>(op.getResult().getType());
    Type elemType = resType.getElementType();
    auto index = op.getIndex();
    Value src1;
    Operation *indexOp = index.getDefiningOp();
    if (auto constIndex = dyn_cast_or_null<arith::ConstantOp>(indexOp)) {
      if (dyn_cast_if_present<IntegerAttr>(constIndex.getValue())) {
        auto constIndexAttr =
            dyn_cast_if_present<IntegerAttr>(constIndex.getValue());
        src1 = rewriter.create<LLVM::ConstantOp>(loc, elemType,
                                                 constIndexAttr.getInt());
      } else {
        auto constIndexAttr =
            dyn_cast_if_present<FloatAttr>(constIndex.getValue());
        src1 = rewriter.create<LLVM::ConstantOp>(loc, elemType,
                                                 constIndexAttr.getValue());
      }
    } else {
      src1 = rewriter.create<UnrealizedConversionCastOp>(loc, elemType, index)
                 .getResult(0);
    }
    // convert increase_p to i32 (0: increase; 1: decrease)
    auto increaseDirection = op.getIncreaseP();
    uint32_t increaseDirectionI32 = 0;
    if (increaseDirection == mlir::hivmave::VCIType::DECREASE) {
      increaseDirectionI32 = 1;
    }
    Value src2 = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                   increaseDirectionI32);

    Value result;
    if (elemType.isInteger(32) && resType.getNumElements() <= 64) {
      auto targetType = VectorType::get(64, rewriter.getI32Type());
      result = rewriter.create<VciInstrOp>(loc, targetType, src1, src2);
    } else if (elemType.isInteger(16) && resType.getNumElements() <= 128) {
      // FIXME: Adapt for VciInstrOp but <= is WARNING
      auto targetType = VectorType::get(128, rewriter.getI16Type());
      result = rewriter.create<VciInstrOp>(loc, targetType, src1, src2);
    } else if (elemType.isInteger(8) && resType.getNumElements() <= 256) {
      // FIXME: Adapt for VciInstrOp but <= is WARNING
      auto targetType = VectorType::get(256, rewriter.getI8Type());
      result = rewriter.create<VciInstrOp>(loc, targetType, src1, src2);
    } else if (elemType.isF32() && resType.getNumElements() <= 64) {
      auto targetType = VectorType::get(64, rewriter.getF32Type());
      result = rewriter.create<VciInstrOp>(loc, targetType, src1, src2);
    } else if (elemType.isF16() && resType.getNumElements() <= 128) {
      auto targetType = VectorType::get(128, rewriter.getF16Type());
      result = rewriter.create<VciInstrOp>(loc, targetType, src1, src2);
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported vector type");
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};
template <typename OpTy, typename IntrV128F16OpTy, typename IntrV128S16OpTy,
          typename IntrV128U16OpTy, typename IntrV256S8OpTy,
          typename IntrV256U8OpTy, typename IntrV64F32OpTy,
          typename IntrV64S32OpTy, typename IntrV64U32OpTy>
struct BinaryVectorScalarLowerToIntrinsic : public OpConversionPattern<OpTy> {
  explicit BinaryVectorScalarLowerToIntrinsic(LLVMTypeConverter &converter)
      : OpConversionPattern<OpTy>(converter, &converter.getContext()) {}

  const LLVMTypeConverter &getTypeConverter() const {
    return *static_cast<const LLVMTypeConverter *>(
        OpConversionPattern<OpTy>::getTypeConverter());
  }
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.getVec();
    Value rhs = op.getScalar();
    auto vecType = cast<VectorType>(lhs.getType());
    uint64_t totalSize = static_cast<uint64_t>(vecType.getNumElements());
    Type elementType = vecType.getElementType();
    auto dataWidth = elementType.getIntOrFloatBitWidth();
    Value mask = getVLRegValueOrSelf(op.getMask(), rewriter);

    auto vlLength = util::VL_BITS / dataWidth;
    VectorType oriVecType = vecType;
    if (totalSize != vlLength) {
      vecType = VectorType::get(SmallVector<int64_t>{vlLength}, elementType);
      lhs = rewriter.create<UnrealizedConversionCastOp>(loc, vecType, lhs)
                .getResult(0);
    }
    Value res =
        createCorrespondingIntr<IntrV128F16OpTy, IntrV128S16OpTy,
                                IntrV128U16OpTy, IntrV256S8OpTy, IntrV256U8OpTy,
                                IntrV64F32OpTy, IntrV64S32OpTy, IntrV64U32OpTy,
                                /*BF16*/ VunsupportedBinaryInstrOp>(
            rewriter, loc, elementType, vecType, lhs, rhs, mask);
    if (!res) {
      return rewriter.notifyMatchFailure(op, "cannot legalize op");
    }
    if (oriVecType != vecType) {
      Operation *ucc =
          rewriter.create<UnrealizedConversionCastOp>(loc, oriVecType, res);
      rewriter.replaceOp(op, ucc);
    } else {
      rewriter.replaceOp(op, res);
    }
    return success();
  }
};
/// An entry associating the "main" BinaryOp with its instantiations for
/// vectors.
template <typename OpTy, typename IntrV128F16OpTy, typename IntrV128S16OpTy,
          typename IntrV128U16OpTy, typename IntrV256S8OpTy,
          typename IntrV256U8OpTy, typename IntrV64F32OpTy,
          typename IntrV64S32OpTy, typename IntrV64U32OpTy>
struct BinaryVectorScalarRegEntry {
  using MainOp = OpTy;
  using IntrV128F16Op = IntrV128F16OpTy;
  using IntrV128S16Op = IntrV128S16OpTy;
  using IntrV128U16Op = IntrV128U16OpTy;
  using IntrV256S8Op = IntrV256S8OpTy;
  using IntrV256U8Op = IntrV256U8OpTy;
  using IntrV64F32Op = IntrV64F32OpTy;
  using IntrV64S32Op = IntrV64S32OpTy;
  using IntrV64U32Op = IntrV64U32OpTy;
};
/// A container for op association entries facilitating the configuration of
/// dialect conversion.
template <typename... Args> struct BinaryVectorScalarRegistryImpl {
  static void registerPatterns(LLVMTypeConverter &Converter,
                               RewritePatternSet &patterns) {
    patterns.add<BinaryVectorScalarLowerToIntrinsic<
        typename Args::MainOp, typename Args::IntrV128F16Op,
        typename Args::IntrV128S16Op, typename Args::IntrV128U16Op,
        typename Args::IntrV256S8Op, typename Args::IntrV256U8Op,
        typename Args::IntrV64F32Op, typename Args::IntrV64S32Op,
        typename Args::IntrV64U32Op>...>(Converter);
  }
};

using BinaryVectorScalarRegistry = BinaryVectorScalarRegistryImpl<
    BinaryVectorScalarRegEntry<
        /*OpTy*/ VFAddsOp, /*F16*/ VaddsSXInstrOp,
        /*S16*/ VaddsSXInstrOp, /*U16*/ VaddsUXInstrOp,
        /*S8*/ VaddsSXInstrOp, /*U8*/ VaddsUXInstrOp,
        /*F32*/ VaddsSXInstrOp, /*S32*/ VaddsSXInstrOp,
        /*U32*/ VaddsUXInstrOp>,
    BinaryVectorScalarRegEntry<
        /*OpTy*/ VFMulsOp, /*F16*/ VmulsSXInstrOp,
        /*S16*/ VmulsSXInstrOp, /*U16*/ VmulsUXInstrOp,
        /*S8*/ VmulsSXInstrOp, /*U8*/ VmulsUXInstrOp,
        /*F32*/ VmulsSXInstrOp, /*S32*/ VmulsSXInstrOp,
        /*U32*/ VmulsUXInstrOp>,
    BinaryVectorScalarRegEntry<
        /*OpTy*/ VFMaxsOp, /*F16*/ VmaxsSXInstrOp,
        /*S16*/ VmaxsSXInstrOp, /*U16*/ VmaxsUXInstrOp,
        /*S8*/ VmaxsSXInstrOp, /*U8*/ VmaxsUXInstrOp,
        /*F32*/ VmaxsSXInstrOp, /*S32*/ VmaxsSXInstrOp,
        /*U32*/ VmaxsUXInstrOp>,
    BinaryVectorScalarRegEntry<
        /*OpTy*/ VFMinsOp, /*F16*/ VminsSXInstrOp,
        /*S16*/ VminsSXInstrOp, /*U16*/ VminsUXInstrOp,
        /*S8*/ VminsSXInstrOp, /*U8*/ VminsUXInstrOp,
        /*F32*/ VminsSXInstrOp, /*S32*/ VminsSXInstrOp,
        /*U32*/ VminsUXInstrOp>,
    BinaryVectorScalarRegEntry<
        /*OpTy*/ VMaxsSIOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VmaxsSXInstrOp, /*U16*/ VunsupportedBinaryInstrOp,
        /*S8*/ VmaxsSXInstrOp, /*U8*/ VunsupportedBinaryInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VmaxsSXInstrOp,
        /*U32*/ VunsupportedBinaryInstrOp>,
    BinaryVectorScalarRegEntry<
        /*OpTy*/ VMinsSIOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VminsSXInstrOp, /*U16*/ VunsupportedBinaryInstrOp,
        /*S8*/ VminsSXInstrOp, /*U8*/ VunsupportedBinaryInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VminsSXInstrOp,
        /*U32*/ VunsupportedBinaryInstrOp>,
    BinaryVectorScalarRegEntry<
        /*OpTy*/ VMaxsUIOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VunsupportedBinaryInstrOp, /*U16*/ VmaxsUXInstrOp,
        /*S8*/ VunsupportedBinaryInstrOp, /*U8*/ VmaxsUXInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VunsupportedBinaryInstrOp,
        /*U32*/ VmaxsUXInstrOp>,
    BinaryVectorScalarRegEntry<
        /*OpTy*/ VMinsUIOp, /*F16*/ VunsupportedBinaryInstrOp,
        /*S16*/ VunsupportedBinaryInstrOp, /*U16*/ VminsUXInstrOp,
        /*S8*/ VunsupportedBinaryInstrOp, /*U8*/ VminsUXInstrOp,
        /*F32*/ VunsupportedBinaryInstrOp, /*S32*/ VunsupportedBinaryInstrOp,
        /*U32*/ VminsUXInstrOp>,
    BinaryVectorScalarRegEntry<
        /*OpTy*/ VFLRelusOp, /*F16*/ VlreluXInstrOp,
        /*S16*/ VunsupportedBinaryInstrOp, /*U16*/ VunsupportedBinaryInstrOp,
        /*S8*/ VunsupportedBinaryInstrOp, /*U8*/ VunsupportedBinaryInstrOp,
        /*F32*/ VlreluXInstrOp, /*S32*/ VunsupportedBinaryInstrOp,
        /*U32*/ VunsupportedBinaryInstrOp>>;

/// Populate the given list with patterns that convert from HIVM to LLVM.
void populateHIVMAVEToAVEIntrinPatterns(LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns) {
  // Populate conversion patterns
  // clang-format off
  patterns.add<HIVMLoadOpLowering, HIVM2VLLoadOpLowering, HIVMStoreOpLowering, HIVMStoreStrideOpLowering,
               HIVMGatherOpLowering, HIVMScatterOpLowering,
               HIVMPgeOpLowering, HIVMPltOpLowering, HIVMPltMOpLowering,
               HIVMVtrcOpLowering, HIVMVSlideOpLowering,
               HIVMSelectOpLowering, HIVMReductionOpLowering,
               HIVMVCIOpLowering, HIVMBroadcastScalarOpLowering,
               HIVMVShiftOpLowering<hivmave::VFShrOp>,
               HIVMVShiftOpLowering<hivmave::VFShrsOp>,
               HIVMVShiftOpLowering<hivmave::VFShlOp>,
               HIVMVShiftOpLowering<hivmave::VFShlsOp>,
               HIVMTypeConvertionOpLowering<VFExtFOp>,
               HIVMTypeConvertionOpLowering<VFExtSIOp>,
               HIVMTypeConvertionOpLowering<VFExtUIOp>,
               HIVMTypeConvertionOpLowering<VFTruncIOp>,
               HIVMTypeConvertionOpLowering<VFTruncFOp>,
               HIVMTypeConvertionOpLowering<VFFpToSIntOp>,
               HIVMTypeConvertionOpLowering<VFFpToUIntOp>,
               HIVMTypeConvertionOpLowering<VFSIntToFpOp>,
               HIVMTypeConvertionOpLowering<VFUIntToFpOp>,
               HIVMCmpOpLowering, VectorBitCastLowering,
               HIVMVecBrcOpPattern, HIVMPnotOpLowering,
               HIVMPredicateBinaryLogicOpLowering<PregAndOp>,
               HIVMPredicateBinaryLogicOpLowering<PregOrOp>,
               HIVMPredicateBinaryLogicOpLowering<PregXorOp>,
               HIVMInterleaveOpLowering,HIVMDeInterleaveOpLowering,
               HIVMExpdifOpLowering, HIVMMemBarOpLowering,
               HIVMVmullOpLowering,
               HIVMVpackOpLowering, HIVMVunpackOpLowering,
               HIVMPregCastOpLowering
  >(converter);
  BinaryRegistry::registerPatterns(converter, patterns);
  UnaryRegistry::registerPatterns(converter, patterns);
  TenaryRegistry::registerPatterns(converter, patterns);
  BroadCastScalarRegistry::registerPatterns(converter, patterns);
  BinaryVectorScalarRegistry::registerPatterns(converter, patterns);
  // clang-format on
}

namespace {
// Describe the data actually written, independently of its producer. A sparse
// value can occupy the whole register while storing only a quarter of its bits.
// Ordinary vectors have 2048 physical bits; predicates have only 256.
struct StorePayload {
  VFMaskedStoreOp store;
  memref::ReinterpretCastOp view;
  int64_t registerBits;
  int64_t elementBits;
  int64_t elements;
  int64_t laneStride;

  bool isPredicate() const { return elementBits == 1; }
  int64_t sizeBits() const { return elements * elementBits; }
  bool isUnderfilled() const { return sizeBits() < registerBits; }
};

// Recognize the normalized addressing shared by payload and alias analysis:
// one contiguous UB slice, with the complete access offset in the view.
static memref::ReinterpretCastOp getContiguousUBView(Value base,
                                                     ValueRange indices) {
  auto view = base.getDefiningOp<memref::ReinterpretCastOp>();
  if (!view || view.getType().getRank() != 1 || indices.size() != 1 ||
      getConstantIntValue(indices.front()) != 0 ||
      getConstantIntValue(view.getMixedStrides().front()) != 1 ||
      hivm::getOptionalHIVMAddressSpace(view.getType()) !=
          hivm::AddressSpace::UB)
    return {};
  return view;
}

static std::optional<StorePayload> analyzeStorePayload(VFMaskedStoreOp store) {
  auto view = getContiguousUBView(store.getBase(), store.getIndices());
  if (!view)
    return std::nullopt;
  VectorType dataType = store.getVectorType();
  if (dataType.getRank() != 1 || dataType.isScalable())
    return std::nullopt;
  int64_t bits = view.getType().getElementTypeBitWidth();
  int64_t stride = 1;
  int64_t registerBits = bits == 1 ? util::PREDICATE_BITS : util::VL_BITS;
  if (bits == 1) {
    int pb = getBitWidthFromAttr(store);
    if (store.getPattern() != StoreDist::NORM_B8 ||
        (pb != 8 && pb != 16 && pb != 32) ||
        !dataType.getElementType().isInteger(1))
      return std::nullopt;
    stride = pb / 8;
  } else {
    switch (store.getPattern()) {
    case StoreDist::PK_B16:
      if (bits != 8 || dataType.getElementTypeBitWidth() != 16)
        return std::nullopt;
      stride = 2;
      break;
    case StoreDist::PK_B32:
      if (bits != 16 || dataType.getElementTypeBitWidth() != 32)
        return std::nullopt;
      stride = 2;
      break;
    case StoreDist::PK4_B32:
      if (bits != 8 || dataType.getElementTypeBitWidth() != 32)
        return std::nullopt;
      stride = 4;
      break;
    default:
      if (!((bits == 8 && store.getPattern() == StoreDist::NORM_B8) ||
            (bits == 16 && store.getPattern() == StoreDist::NORM_B16) ||
            (bits == 32 && store.getPattern() == StoreDist::NORM_B32)) ||
          dataType.getElementType() != view.getType().getElementType())
        return std::nullopt;
      break;
    }
    // Unaligned vector stores consume a dense prefix. Normalize their packing
    // distribution before considering them here.
    if (stride != 1 && store->hasAttr(UnalignedAttr::name))
      return std::nullopt;
  }
  int64_t capacity = registerBits / (bits * stride);
  // AveMaskType also admits the custom AVE mask type; only fixed vectors have
  // the lane counts used by this analysis.
  auto maskType = dyn_cast<VectorType>(store.getMask().getType());
  if (!maskType || maskType.getRank() != 1 || maskType.isScalable())
    return std::nullopt;
  Operation *maskOp = store.getMask().getDefiningOp();
  if (!maskOp)
    return std::nullopt;
  int64_t maskWidth = getBitWidthFromAttr(maskOp);
  if (maskWidth == -1)
    maskWidth = util::VL_BITS / maskType.getNumElements();
  int64_t expectedWidth = bits == 1 ? stride * 8 : bits * stride;
  if (maskWidth != expectedWidth)
    return std::nullopt;

  int64_t count;
  if (auto pge = dyn_cast<VFPgeOp>(maskOp)) {
    if (pge.getPattern() == PgePattern::ALL) {
      // Both aligned and unaligned lowering must agree on the full footprint.
      if (maskType.getNumElements() != capacity)
        return std::nullopt;
      count = capacity;
    } else {
      // Predicate stores do not have a hardware write mask. Do not turn a
      // partial logical mask into a full packed write.
      if (bits == 1 || stride != 1)
        return std::nullopt;
      count = pge.getPattern() == PgePattern::H ? capacity / 2
                                                : getNumfromPgePattern(pge);
    }
  } else if (auto plt = dyn_cast<VFPltOp>(maskOp)) {
    auto active = getConstantIntValue(plt.getTrueShape());
    if (bits == 1 || stride != 1 || !active)
      return std::nullopt;
    count = *active;
  } else {
    return std::nullopt;
  }
  // AveVector bounds ordinary vectors by 2048 bits. Predicate capacity also
  // depends on PB8/16/32, so only predicates need the layout-specific upper
  // bound. Together with count <= dataElements, this also bounds count and
  // makes sparse ALL payloads exactly capacity elements.
  // The mask must not read past the data or write past the view. Unknown view
  // sizes use the negative kDynamic sentinel and fail the size comparison.
  if (count <= 0 || dataType.getNumElements() < count ||
      (bits == 1 && dataType.getNumElements() > capacity) ||
      view.getType().getDimSize(0) < count)
    return std::nullopt;
  return StorePayload{store, view, registerBits, bits, count, stride};
}

// reinterpret_cast offsets are relative to the underlying aligned pointer,
// not the source view's offset. Only peel operations preserving that pointer.
static Value getStoreRoot(Value value) {
  while (true) {
    if (auto metadata = value.getDefiningOp<memref::ExtractStridedMetadataOp>())
      value = metadata.getSource();
    else if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      // This project-specific cast folds the source offset into the pointer.
      // It cannot be peeled like an ordinary descriptor-preserving cast.
      if (cast->hasAttr(mlir::utils::KFoldOffsetMarker))
        return value;
      value = cast.getSource();
    } else if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>())
      value = view.getSource();
    else
      return value;
  }
}

// Prove adjacency and retain the first offset's divisor from the same affine
// composition. The caller uses it to check alignment without recomputing this
// pair when a larger group requires a stronger alignment.
static bool areAdjacentStores(memref::ReinterpretCastOp first,
                              memref::ReinterpretCastOp second,
                              int64_t elements, uint64_t &offsetDivisor) {
  SmallVector<Value> operands;
  SmallVector<AffineExpr> offsets;
  auto *ctx = first.getContext();
  for (auto view : {first, second}) {
    OpFoldResult offset = view.getMixedOffsets().front();
    if (auto value = dyn_cast<Value>(offset)) {
      offsets.push_back(getAffineDimExpr(operands.size(), ctx));
      operands.push_back(value);
    } else {
      offsets.push_back(getAffineConstantExpr(
          cast<IntegerAttr>(cast<Attribute>(offset)).getInt(), ctx));
    }
  }
  auto map = AffineMap::get(operands.size(), 0, offsets, ctx);
  affine::fullyComposeAffineMapAndOperands(&map, &operands);
  affine::canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  auto delta = dyn_cast<AffineConstantExpr>(
      simplifyAffineExpr(map.getResult(1) - map.getResult(0), map.getNumDims(),
                         map.getNumSymbols()));
  if (!delta || delta.getValue() != elements)
    return false;
  offsetDivisor = map.getResult(0).getLargestKnownDivisor();
  return true;
}

// Access footprints are needed when two views may share a planned UB base,
// as with in-place i8-to-i1 compression.
struct StoreAccess {
  Value root;
  OpFoldResult offset;
  int64_t elementBits;
  int64_t sizeBits;
};

static StoreAccess getStoreAccess(StorePayload payload) {
  return {getStoreRoot(payload.view.getSource()),
          payload.view.getMixedOffsets().front(), payload.elementBits,
          payload.sizeBits()};
}

// Use the physical instruction footprint, not the logical vector length.
static std::optional<int64_t> getLoadAccessSizeBits(VFLoadOp load) {
  if (load.getRes1())
    return std::nullopt;
  bool unaligned = load->hasAttr(UnalignedAttr::name);
  switch (load.getPattern()) {
  case LoadDist::NORM:
    // Aligned predicate loads use PLDS. Unaligned loads, including predicate
    // loads followed by MOVVP, read a full ordinary register through VLDUS.
    return load.getMemRefType().getElementType().isInteger(1) && !unaligned
               ? util::PREDICATE_BITS
               : util::VL_BITS;
  case LoadDist::UNPK4_B8:
    if (load.getMemRefType().getElementTypeBitWidth() != 8)
      return std::nullopt;
    // VLDUS has no distribution operand, so its read is not reduced by UNPK4.
    return unaligned ? util::VL_BITS : util::VL_BITS / 4;
  default:
    return std::nullopt;
  }
}

// Fallback when payload analysis cannot determine the effective store data.
// Keep its existing scope: ordinary B16/B32 and unaligned PB32 stores. Other
// layouts still require successful payload analysis.
static std::optional<int64_t> getStoreAccessSizeBits(VFMaskedStoreOp store) {
  int64_t patternBits;
  switch (store.getPattern()) {
  case StoreDist::NORM_B8:
    patternBits = 8;
    break;
  case StoreDist::NORM_B16:
    patternBits = 16;
    break;
  case StoreDist::NORM_B32:
    patternBits = 32;
    break;
  default:
    return std::nullopt;
  }

  int64_t elementBits = store.getVectorType().getElementTypeBitWidth();
  if (elementBits == 1) {
    if (getBitWidthFromAttr(store) != 32 ||
        !store->hasAttr(UnalignedAttr::name))
      return std::nullopt;
    // PSTU.B32 writes 8 bytes regardless of the logical vector length or mask.
    return util::PREDICATE_BITS / 4;
  }
  if (patternBits == 8 || elementBits != patternBits)
    return std::nullopt;

  if (store->hasAttr(UnalignedAttr::name)) {
    // Mirror getElemSizeByStoreMask: VSTUS uses a byte count, not the store
    // distribution or logical vector length. An unknown PLT count cannot be
    // assumed to fit one register.
    Operation *maskOp = store.getMask().getDefiningOp();
    while (isa_and_nonnull<UnrealizedConversionCastOp, LLVM::ExtractValueOp>(
        maskOp))
      maskOp = maskOp->getOperand(0).getDefiningOp();
    uint32_t count;
    if (auto plt = dyn_cast_or_null<VFPltOp>(maskOp)) {
      auto active = getConstantIntValue(plt.getTrueShape());
      if (!active)
        return std::nullopt;
      count = *active;
    } else if (auto pge = dyn_cast_or_null<VFPgeOp>(maskOp)) {
      count = getNumfromPgePattern(pge);
    } else {
      // Lowering defaults to one register when it cannot recognize the mask.
      return util::VL_BITS;
    }
    // Match the i32 byte count used by VSTUS before converting it to bits.
    uint32_t bytes = count * (elementBits / 8);
    return int64_t(bytes) * 8;
  }

  // A full-register bound also covers stores with an unknown hardware mask.
  return util::VL_BITS;
}

static std::optional<StoreAccess> getStoreAccess(Operation *op) {
  Value base;
  ValueRange indices;
  VectorType vectorType;
  auto load = dyn_cast<VFLoadOp>(op);
  if (load) {
    base = load.getBase();
    indices = load.getIndices();
    vectorType = load.getVectorType();
  } else {
    // canSinkStorePast only asks about loads and masked stores.
    auto store = cast<VFMaskedStoreOp>(op);
    if (auto payload = analyzeStorePayload(store))
      return getStoreAccess(*payload);
    base = store.getBase();
    indices = store.getIndices();
    vectorType = store.getVectorType();
  }
  auto view = getContiguousUBView(base, indices);
  if (!view || vectorType.isScalable() ||
      vectorType.getElementType() != view.getType().getElementType())
    return std::nullopt;
  auto sizeBits = load ? getLoadAccessSizeBits(load)
                       : getStoreAccessSizeBits(cast<VFMaskedStoreOp>(op));
  if (!sizeBits)
    return std::nullopt;
  return StoreAccess{getStoreRoot(view.getSource()),
                     view.getMixedOffsets().front(),
                     view.getType().getElementTypeBitWidth(), *sizeBits};
}

// These operations preserve the aligned pointer. In particular, a
// reinterpret_cast replaces the descriptor offset; its source offset must not
// be added again. memref.view and unknown pointer arithmetic are not peeled.
static bool collectUBAddresses(Value value, SmallVectorImpl<int64_t> &addresses,
                               unsigned depth = 0) {
  // These are compile-time analysis budgets, not hardware limits. Bound both
  // view/select traversal and the address combinations checked at call sites.
  if (depth > 16 || addresses.size() >= 16)
    return false;
  Value root = getStoreRoot(value);
  if (root != value)
    return collectUBAddresses(root, addresses, depth + 1);
  if (auto subview = value.getDefiningOp<memref::SubViewOp>())
    return collectUBAddresses(subview.getSource(), addresses, depth + 1);
  if (auto expand = value.getDefiningOp<memref::ExpandShapeOp>())
    return collectUBAddresses(expand.getSrc(), addresses, depth + 1);
  if (auto collapse = value.getDefiningOp<memref::CollapseShapeOp>())
    return collectUBAddresses(collapse.getSrc(), addresses, depth + 1);
  if (auto select = value.getDefiningOp<arith::SelectOp>())
    return collectUBAddresses(select.getTrueValue(), addresses, depth + 1) &&
           collectUBAddresses(select.getFalseValue(), addresses, depth + 1);
  auto pointer = value.getDefiningOp<hivm::PointerCastOp>();
  // PointerCastOp permits multiple addresses; this proof models one UB base.
  if (!pointer || pointer.getAddrs().size() != 1)
    return false;
  auto address = getConstantIntValue(pointer.getAddrs().front());
  if (!address || *address < 0 ||
      *address > std::numeric_limits<uint32_t>::max())
    return false;
  addresses.push_back(*address);
  return true;
}

static bool getCallSiteUBAddresses(Value root, func::CallOp call,
                                   SmallVectorImpl<int64_t> &addresses) {
  auto arg = dyn_cast<BlockArgument>(root);
  if (!arg || !isa<func::FuncOp>(arg.getOwner()->getParentOp()))
    return false;
  // Calls are collected for this VF; func.call verification guarantees the
  // operand count and types match its UB parameters. The peeled view operations
  // preserve that memory space, so collectUBAddresses need not recheck it.
  return collectUBAddresses(call.getOperand(arg.getArgNumber()), addresses);
}

static bool hasAlignedCallSiteBase(Value root, ArrayRef<func::CallOp> calls,
                                   int64_t alignment) {
  return llvm::all_of(calls, [&](func::CallOp call) {
    SmallVector<int64_t> addresses;
    // Every call and every possible selected base must satisfy alignment.
    return getCallSiteUBAddresses(root, call, addresses) &&
           llvm::all_of(addresses,
                        [&](int64_t addr) { return addr % alignment == 0; });
  });
}

static bool hasAlignedStoreBase(Value root, int64_t alignment,
                                ArrayRef<func::CallOp> calls,
                                bool allUsesKnown) {
  if (auto alloc = root.getDefiningOp<memref::AllocOp>()) {
    auto baseAlignment = alloc.getAlignment();
    // Alloc alignment may be zero or not a power of two; require a positive
    // multiple of the target alignment.
    return baseAlignment && *baseAlignment != 0 &&
           *baseAlignment % alignment == 0;
  }
  auto argument = dyn_cast<BlockArgument>(root);
  if (argument && isa<func::FuncOp>(argument.getOwner()->getParentOp()) &&
      (!allUsesKnown || !hasAlignedCallSiteBase(root, calls, alignment)))
    return false;
  // The ABI helper assumes parameter base alignment. For outlined VFs the
  // actual caller bases must also satisfy it, as checked above. With no calls,
  // keep the ABI contract used when lowering a standalone VF.
  // Require a zero layout offset: this helper checks descriptor alignment,
  // whereas reinterpret_cast supplies its own absolute offset.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(cast<MemRefType>(root.getType()), strides,
                                 offset)) ||
      offset != 0)
    return false;
  return isLoadStoreIndexAligned(root, OperandRange(nullptr, 0));
}

// Resolve different roots at every call site, including all select
// alternatives. PlanMemory guarantees that accesses from distinct planned UB
// bases do not overlap. For equal bases, bound delta = bOffset -
// aOffset in bits:
//   LB(delta) >= a.sizeBits: A ends at or before B starts.
//   UB(delta) <= -b.sizeBits: B ends at or before A starts.
// If neither proof succeeds, preserve the original memory order.
static bool areDisjointAtCallSites(StorePayload first, Operation *other,
                                   Value otherRoot,
                                   ArrayRef<func::CallOp> calls) {
  auto a = getStoreAccess(first);
  if (a.root != otherRoot) {
    if (calls.empty())
      return false;
    bool mayShareBase = false;
    for (func::CallOp call : calls) {
      SmallVector<int64_t> aAddresses, bAddresses;
      if (!getCallSiteUBAddresses(a.root, call, aAddresses) ||
          !getCallSiteUBAddresses(otherRoot, call, bAddresses))
        return false;
      mayShareBase |= llvm::any_of(aAddresses, [&bAddresses](int64_t address) {
        return llvm::is_contained(bAddresses, address);
      });
    }
    if (!mayShareBase)
      return true;
  }

  auto b = getStoreAccess(other);
  if (!b)
    return false;
  SmallVector<Value> operands;
  auto *ctx = other->getContext();
  auto getOffset = [ctx, &operands](const StoreAccess &access) -> AffineExpr {
    if (auto offset = getConstantIntValue(access.offset)) {
      // Do not let conversion to bits wrap an affine constant and falsely
      // establish disjointness.
      int64_t bitOffset;
      if (llvm::MulOverflow(*offset, access.elementBits, bitOffset))
        return {};
      return getAffineConstantExpr(bitOffset, ctx);
    }
    // Keep dynamic offsets symbolic: their difference may be bounded even
    // when neither absolute offset has known bounds.
    auto expr = getAffineDimExpr(operands.size(), ctx);
    operands.push_back(cast<Value>(access.offset));
    return expr * access.elementBits;
  };
  AffineExpr aOffset = getOffset(a), bOffset = getOffset(*b);
  if (!aOffset || !bOffset)
    return false;
  auto map = AffineMap::get(operands.size(), 0, bOffset - aOffset);
  affine::fullyComposeAffineMapAndOperands(&map, &operands);
  affine::canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  auto bound = [map, &operands](presburger::BoundType type) {
    return ValueBoundsConstraintSet::computeConstantBound(
        type, ValueBoundsConstraintSet::Variable(map, operands),
        /*stopCondition=*/nullptr, /*closedUB=*/true);
  };
  auto deltaLo = bound(presburger::BoundType::LB);
  auto deltaHi = bound(presburger::BoundType::UB);
  return (succeeded(deltaLo) && *deltaLo >= a.sizeBits) ||
         (succeeded(deltaHi) && *deltaHi <= -b->sizeBits);
}

static bool canSinkStorePast(Operation *op, StorePayload first, Value root,
                             AliasAnalysis &aliasAnalysis,
                             ArrayRef<func::CallOp> calls) {
  // Do not cross control flow, barriers, calls, or unknown side effects. Loads
  // and stores to proven-disjoint allocations may occur between the compares.
  if (op->getNumRegions() != 0)
    return false;
  if (isMemoryEffectFree(op))
    return true;
  Value base;
  if (auto load = dyn_cast<VFLoadOp>(op))
    base = load.getBase();
  else if (auto store = dyn_cast<VFMaskedStoreOp>(op))
    base = store.getBase();
  if (!base)
    return false;
  Value otherRoot = getStoreRoot(base);
  if (aliasAnalysis.alias(root, otherRoot).isNo())
    return true;
  // The planned-allocation contract applies to UB accesses only.
  return hivm::getOptionalHIVMAddressSpace(otherRoot.getType()) ==
             hivm::AddressSpace::UB &&
         areDisjointAtCallSites(first, op, otherRoot, calls);
}

// Packing is a target decision, separate from payload discovery. Sparse stores
// use deinterleave; dense unaligned prefixes can share the alignment register.
// N sparse stores become N-1 deinterleaves and one store: no extra data
// instructions, with fewer memory instructions. A stream replaces 2N store/
// flush instructions with N stores and one flush, without shuffling data.
// A dense aligned prefix is still a candidate, but has no cheaper plan here.
static bool canPackStore(StorePayload payload) {
  // Predicate stores require ALL, so an underfilled predicate is necessarily
  // sparse. The dense case here only contains ordinary vector prefixes.
  return payload.isUnderfilled() &&
         (payload.laneStride > 1 ||
          payload.store->hasAttr(UnalignedAttr::name));
}

static Value packStoreValues(ArrayRef<StorePayload> stores,
                             PatternRewriter &rewriter) {
  StorePayload payload = stores.front();
  Location loc = stores.back().store->getLoc();
  SmallVector<Value> values;
  for (StorePayload part : stores) {
    auto store = part.store;
    Type element = store.getVectorType().getElementType();
    auto physical = VectorType::get(
        {payload.registerBits / element.getIntOrFloatBitWidth()}, element);
    values.push_back(
        castToTypeIfNeeded(rewriter, loc, store.getVal(), physical));
  }
  int64_t stride = payload.laneStride;
  while (values.size() > 1) {
    stride /= 2;
    SmallVector<Value> next;
    for (unsigned i = 0; i < values.size(); i += 2) {
      Value a = values[i], b = values[i + 1];
      Type element = rewriter.getIntegerType(
          payload.isPredicate() ? 1 : payload.elementBits * stride);
      auto physical = VectorType::get(
          {payload.registerBits / element.getIntOrFloatBitWidth()}, element);
      if (a.getType() != physical)
        a = rewriter.create<LLVM::BitcastOp>(loc, physical, a);
      if (b.getType() != physical)
        b = rewriter.create<LLVM::BitcastOp>(loc, physical, b);
      auto pairType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                       {physical, physical});
      Value pair;
      if (!payload.isPredicate())
        pair = rewriter.create<VDintlvInstrOp>(loc, pairType, a, b);
      else if (stride == 2)
        pair = rewriter.create<PDintlvB16InstrOp>(loc, pairType, a, b);
      else // stride == 1
        pair = rewriter.create<PDintlvB8InstrOp>(loc, pairType, a, b);
      next.push_back(rewriter.create<LLVM::ExtractValueOp>(loc, pair, 0));
    }
    values = std::move(next);
  }
  return values.front();
}

static void emitPackedStore(ArrayRef<StorePayload> stores,
                            PatternRewriter &rewriter) {
  StorePayload payload = stores.front();
  auto first = payload.store;
  Location loc = stores.back().store->getLoc();
  Value data = packStoreValues(stores, rewriter);
  int64_t count = payload.elements * stores.size();
  int64_t stride = payload.laneStride / stores.size();
  auto element = payload.view.getType().getElementType();
  auto dataType = VectorType::get({count}, element);
  FunctionDistType dist = FunctionDistType::NORM;
  StoreDist pattern = first.getPattern();
  if (payload.isPredicate()) {
    data = castToTypeIfNeeded(rewriter, loc, data, dataType);
    dist = stride == 1 ? FunctionDistType::PB8 : FunctionDistType::PB16;
  } else if (stride == 1) {
    if (data.getType() != dataType)
      data = rewriter.create<LLVM::BitcastOp>(loc, dataType, data);
    pattern =
        payload.elementBits == 8 ? StoreDist::NORM_B8 : StoreDist::NORM_B16;
  } else {
    // Two PK4_B32 stores form one PK_B16 store, still only 128 bytes.
    pattern = StoreDist::PK_B16;
    dist = FunctionDistType::PK;
  }
  auto viewType =
      MemRefType::get({count}, element, payload.view.getType().getLayout(),
                      payload.view.getType().getMemorySpace());
  Value view = rewriter.create<memref::ReinterpretCastOp>(
      loc, viewType, payload.view.getSource(),
      payload.view.getMixedOffsets()[0],
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(count)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
  auto maskType = VectorType::get({count}, rewriter.getI1Type());
  auto mask = rewriter.create<VFPgeOp>(loc, maskType, PgePattern::ALL);
  int64_t maskWidth =
      payload.isPredicate() ? stride * 8 : payload.elementBits * stride;
  FunctionDistType maskDist = maskWidth == 8    ? FunctionDistType::PB8
                              : maskWidth == 16 ? FunctionDistType::PB16
                                                : FunctionDistType::PB32;
  mask->setAttr("functionType",
                FunctionDistTypeAttr::get(rewriter.getContext(), maskDist));
  auto merged = rewriter.create<VFMaskedStoreOp>(
      loc, pattern, view, first.getIndices(), mask, data);
  // The builder sets the new pattern; copy only the remaining attributes.
  merged->setDiscardableAttrs(first->getDiscardableAttrDictionary());
  merged->setAttr("functionType",
                  FunctionDistTypeAttr::get(rewriter.getContext(), dist));
  merged->removeAttr(UnalignedAttr::name);
  merged->removeAttr("hivm.is_continuous");
}

static void emitStoreStream(ArrayRef<StorePayload> stores,
                            PatternRewriter &rewriter) {
  StorePayload payload = stores.front();
  Location loc = stores.back().store->getLoc();
  LLVMTypeConverter converter(rewriter.getContext());
  populateHIVMAddressSpaceAttributeConversions(converter);
  Value descriptor = rewriter
                         .create<UnrealizedConversionCastOp>(
                             loc, converter.convertType(payload.view.getType()),
                             payload.view.getResult())
                         .getResult(0);
  MemRefDescriptor memref(descriptor);
  Value base = memref.alignedPtr(rewriter, loc);
  Value offset = memref.offset(rewriter, loc);
  Type element = payload.view.getType().getElementType();
  Value pointer = rewriter.create<LLVM::GEPOp>(loc, base.getType(), element,
                                               base, ValueRange{offset});
  Value alignment = rewriter.create<InitVectorAlignDataInstrOp>(
      loc, VectorType::get({32}, rewriter.getI8Type()));
  auto physical =
      VectorType::get({util::VL_BITS / payload.elementBits}, element);
  for (StorePayload part : stores) {
    Value count = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(part.sizeBits() / 8));
    Value data =
        castToTypeIfNeeded(rewriter, loc, part.store.getVal(), physical);
    auto result = buildVstusPostOp(data, pointer, count, alignment, rewriter);
    alignment =
        rewriter.create<LLVM::ExtractValueOp>(loc, result->getResult(0), 0);
    pointer =
        rewriter.create<LLVM::ExtractValueOp>(loc, result->getResult(0), 1);
  }
  buildVstasOp(alignment, pointer, rewriter);
}

static void mergeUnderfilledStoresIn(Operation *scope, ModuleOp module,
                                     ArrayRef<func::CallOp> calls,
                                     bool allUsesKnown) {
  SmallVector<VFMaskedStoreOp> stores;
  scope->walk([&](VFMaskedStoreOp store) { stores.push_back(store); });
  llvm::SmallPtrSet<Operation *, 16> erased;
  AliasAnalysis aliasAnalysis(module);
  PatternRewriter rewriter(module.getContext());
  for (auto first : stores) {
    // The worklist predates rewrites; an earlier group may have erased this op.
    if (erased.contains(first))
      continue;
    auto payload = analyzeStorePayload(first);
    if (!payload || !canPackStore(*payload))
      continue;
    bool stream = payload->laneStride == 1;
    Value root = getStoreRoot(payload->view.getSource());
    SmallVector<StorePayload> group{*payload};
    unsigned bestSize = 0;
    int64_t elements = payload->elements;
    for (Operation *op = first->getNextNode(); op; op = op->getNextNode()) {
      auto next = dyn_cast<VFMaskedStoreOp>(op);
      auto nextPayload = next ? analyzeStorePayload(next) : std::nullopt;
      uint64_t offsetDivisor;
      // Preserve all store attributes until their semantic relevance is
      // modeled explicitly; matching payloads alone is not sufficient.
      // Equal element types and attributes imply the same register and lane
      // layout. Sparse ALL payloads then have equal sizes. A positive group
      // fitting one register also implies every member is underfilled.
      bool tag = nextPayload &&
                 nextPayload->view.getType().getElementType() ==
                     payload->view.getType().getElementType() &&
                 getStoreRoot(nextPayload->view.getSource()) == root &&
                 first->getAttrDictionary() == next->getAttrDictionary() &&
                 (elements + nextPayload->elements) * payload->elementBits <=
                     payload->registerBits &&
                 areAdjacentStores(payload->view, nextPayload->view, elements,
                                   offsetDivisor);
      if (tag) {
        group.push_back(*nextPayload);
        elements += nextPayload->elements;
        if (stream) {
          bestSize = group.size();
        } else if (llvm::isPowerOf2_64(group.size())) {
          // Pairwise deinterleave requires a power-of-two group. Retain the
          // largest proven group if a later candidate cannot be included.
          int64_t alignment = payload->isPredicate() ? elements / 8 : 32;
          // The offset and the actual UB base must each preserve alignment.
          if (offsetDivisor % (alignment * 8 / payload->elementBits) == 0 &&
              hasAlignedStoreBase(root, alignment, calls, allUsesKnown))
            bestSize = group.size();
        }
        // Bound each group to eight stores to limit compile-time analysis.
        // This is not a hardware limit; the total payload must also fit the
        // register capacity checked above.
        if (elements * payload->elementBits == payload->registerBits ||
            group.size() == 8)
          break;
        continue;
      }
      // All pending stores are delayed, so checking only the first would miss
      // a read or write that conflicts with another member of the group.
      if (!llvm::all_of(group, [&](StorePayload pending) {
            return canSinkStorePast(op, pending, root, aliasAnalysis, calls);
          }))
        break;
    }
    if (bestSize < 2)
      continue;
    group.resize(bestSize);
    rewriter.setInsertionPoint(group.back().store);
    if (stream)
      emitStoreStream(group, rewriter);
    else
      emitPackedStore(group, rewriter);
    for (StorePayload part : group) {
      erased.insert(part.store);
      rewriter.eraseOp(part.store);
    }
  }
}

// Optimize the original VF in place only when every call site satisfies the
// required proofs. Reject non-call symbol uses conservatively; access-specific
// proofs must never become blanket noalias metadata.
static void mergeUnderfilledStores(ModuleOp module) {
  if (!hacc::utils::isAscend950(module))
    return;
  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;
    SmallVector<func::CallOp> calls;
    bool allUsesKnown = true;
    if (func->hasAttr(hivm::VectorFunctionAttr::name) &&
        !hacc::utils::isDeviceEntry(func)) {
      auto uses = SymbolTable::getSymbolUses(func, module);
      // Kernel-only callers make module-local analysis sufficient, but do not
      // make arbitrary symbol references direct calls. Unknown use enumeration
      // or non-callee references must not yield a partial call-site proof.
      allUsesKnown =
          uses && llvm::all_of(*uses, [&](const SymbolTable::SymbolUse &use) {
            auto call = dyn_cast<func::CallOp>(use.getUser());
            // getSymbolUses already identifies this symbol in its scope; only
            // verify that this reference is the call's callee attribute.
            if (!call || call.getCalleeAttr() != use.getSymbolRef())
              return false;
            calls.push_back(call);
            return true;
          });
      if (!allUsesKnown)
        calls.clear();
    }
    // Gather call sites before any rewrite: even adjacent stores need their
    // actual base alignment checked before upgrading to an aligned store.
    // Local allocations do not depend on the caller's argument alignment.
    mergeUnderfilledStoresIn(func, module, calls, allUsesKnown);
  });
}

struct ForOpIVNarrowing final : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  template <typename... Args>
  explicit ForOpIVNarrowing(bool useUnsignedCmp, bool enableI16IV,
                            Args &&...args)
      : OpRewritePattern(std::forward<Args>(args)...),
        useUnsignedCmpFor(useUnsignedCmp), enableI16IndVar(enableI16IV) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const final {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    // Skip non-VF functions.
    if (!funcOp || !funcOp->hasAttr(mlir::hivm::VectorFunctionAttr::name))
      return failure();
    Type oldTy = op.getInductionVar().getType();
    unsigned oldWidth = 64; // By default, 64 index.
    if (auto intTy = llvm::dyn_cast<IntegerType>(oldTy)) {
      assert(intTy.isSignlessInteger() &&
             "'scf.for' must be signless integer or index");
      oldWidth = intTy.getWidth();
      // Skip if the IV is already narrow enough.
      if (oldWidth <= 16)
        return failure();
    }
    auto lb = getConstantIntValue(op.getLowerBound());
    auto ub = getConstantIntValue(op.getUpperBound());
    auto step = getConstantIntValue(op.getStep());
    // Skip if lb, ub, or step cannot be analyzed further.
    // TODO: Add more complicated analysis if one of them is not constant.
    if (!lb || !ub || !step)
      return failure();

    std::optional<unsigned> newWidth;
    bool asSigned = true; // Should be treated as signed integers.
    if (enableI16IndVar && llvm::isIntN(16, *lb) && llvm::isIntN(16, *ub) &&
        llvm::isIntN(16, *step)) {
      newWidth = 16;
      asSigned = true;
    } else if (enableI16IndVar && useUnsignedCmpFor && llvm::isUIntN(16, *lb) &&
               llvm::isUIntN(16, *ub) && llvm::isUIntN(16, *step)) {
      newWidth = 16;
      asSigned = false;
    } else if (llvm::isIntN(32, *lb) && llvm::isIntN(32, *ub) &&
               llvm::isIntN(32, *step)) {
      newWidth = 32;
      asSigned = true;
    } else if (useUnsignedCmpFor && llvm::isUIntN(32, *lb) &&
               llvm::isUIntN(32, *ub) && llvm::isUIntN(32, *step)) {
      newWidth = 32;
      asSigned = false;
    }

    // Skip if the new width is not narrower than the old one.
    if (!newWidth || *newWidth >= oldWidth)
      return failure();

    // FIXME: Port the unsigned comparison support for 'scf.for' so that more
    // cases could be narrowed.
    assert((asSigned || useUnsignedCmpFor) &&
           "unsigned is only supported if 'scf.for' has 'unsignedCmp' support");

    auto newTy = rewriter.getIntegerType(*newWidth);

    auto truncToNarrowFn = [](OpBuilder &builder, Location loc, Value source,
                              Type narrowTy) -> Value {
      if (source.getType().isIndex())
        return builder.create<arith::IndexCastOp>(loc, narrowTy, source);
      assert(source.getType().isSignlessInteger());
      return builder.create<arith::TruncIOp>(loc, narrowTy, source);
    };

    auto extFromNarrowFn = [](OpBuilder &builder, Location loc, Value source,
                              Type wideTy, bool asSigned) -> Value {
      if (wideTy.isIndex()) {
        if (asSigned)
          return builder.create<arith::IndexCastOp>(loc, wideTy, source);
        return builder.create<arith::IndexCastUIOp>(loc, wideTy, source);
      }
      assert(wideTy.isSignlessInteger());
      if (asSigned)
        return builder.create<arith::ExtSIOp>(loc, wideTy, source);
      return builder.create<arith::ExtUIOp>(loc, wideTy, source);
    };

    Location loc = op.getLoc();
    DictionaryAttr attrs = op->getAttrDictionary();
    auto newOp = rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, truncToNarrowFn(rewriter, loc, op.getLowerBound(), newTy),
        truncToNarrowFn(rewriter, loc, op.getUpperBound(), newTy),
        truncToNarrowFn(rewriter, loc, op.getStep(), newTy), op.getInitArgs(),
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          SmallVector<Value> newBlockTransferArgs;
          newBlockTransferArgs.emplace_back(
              extFromNarrowFn(b, loc, iv, oldTy, asSigned));
          newBlockTransferArgs.append(iterArgs.begin(), iterArgs.end());
          rewriter.inlineBlockBefore(op.getBody(), b.getInsertionBlock(),
                                     b.getInsertionPoint(),
                                     newBlockTransferArgs);
        },
        /*unsignedCmp=*/!asSigned);
    newOp->setAttrs(attrs);

    return success();
  }

protected:
  bool useUnsignedCmpFor;
  bool enableI16IndVar; // Enable i16 IV.
};
} // namespace

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct HIVMAVEToAVEIntrinPass
    : public impl::ConvertHIVMAVEToAVEIntrinBase<HIVMAVEToAVEIntrinPass> {

  /// Set VF target attr to the global entry func so that bisheng works on
  /// the RegBased arch.
  LogicalResult setVFTargetAttr(ModuleOp moduleOp) {
    if (!hacc::utils::isRegBasedArch(moduleOp)) {
      return success();
    }

    auto maybeSpecInterface = hacc::utils::getNPUTargetSpec(moduleOp);
    if (!maybeSpecInterface.has_value()) {
      return failure();
    }
    auto specInterface = maybeSpecInterface.value();
    auto aArch = specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::ARCH);
    auto archStr = cast<StringAttr>(aArch.getValue()).str();
    // Here we use regbaseintrins's TargetAttr so that
    // HIVMRegbaseIntrinsDialectLLVMIRTranslationInterface would convert
    // the target attr to target-cpu and target-feature in the LLVM IR.
    // In the future, we could use hacc's target attr to replace
    // hivm_regbaseintrins::SIMT_TargetAttr.
    auto targetAttr = hivm_regbaseintrins::SIMT_TargetAttr::get(
        moduleOp.getContext(), archStr);

    moduleOp->walk([&](func::FuncOp funcOp) {
      if (!hacc::utils::isDevice(funcOp))
        return WalkResult::skip();
      funcOp->setAttr(hivm_regbaseintrins::kDavinciTargetAttrName, targetAttr);
      return WalkResult::advance();
    });
    return success();
  }

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    mergeUnderfilledStores(getOperation());
    // If applicable, narrow loop ivs to help generate hardware loops.
    narrowSCFForIV();

    LLVMConversionTarget target(getContext());
    target.addLegalOp<LLVM::BitcastOp>();
    target.addLegalDialect<hivm_regbaseintrins::HIVMRegbaseIntrinsDialect>();

    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());
    populateHIVMAddressSpaceAttributeConversions(converter);
    mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
    populateHIVMAVEToAVEIntrinPatterns(converter, patterns);

    auto moduleOp = getOperation();
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();

    if (failed(setVFTargetAttr(moduleOp)))
      signalPassFailure();
  }

private:
  void narrowSCFForIV() {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForOpIVNarrowing>(useUnsignedCmpFor, enableI16IndVar, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertHIVMAVEToAVEIntrinPass() {
  return std::make_unique<HIVMAVEToAVEIntrinPass>();
}

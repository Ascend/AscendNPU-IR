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
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
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

static Operation *createPgeIntrinOp(PatternRewriter &rewriter, VFPgeOp pge,
                                    unsigned dataWidth) {
  auto loc = pge->getLoc();
  Type dstType = VectorType::get(SmallVector<int64_t>{util::PREDICATE_BITS},
                                 rewriter.getI1Type());
  auto numElems = cast<VectorType>(pge.getRes().getType()).getShape()[0];
  PgePattern realPattern = pge.getPattern();
  if (realPattern == PgePattern::ALL)
    realPattern =
        hivmave::getPgePatternAttr(rewriter, numElems, util::PREDICATE_BITS)
            .value()
            .getValue();
  Value pattern = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(static_cast<uint32_t>(realPattern)));

  Operation *newOp;
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                 rewriter.getI32IntegerAttr(0));
  switch (dataWidth) {
  case 1:
  case 8:
    newOp = rewriter.create<PgeB8>(loc, dstType, pattern, zero);
    newOp->setAttr(utils::maskBitWidth,
                   rewriter.getIntegerAttr(rewriter.getIntegerType(32), 8));
    break;
  case 16:
    newOp = rewriter.create<PgeB16>(loc, dstType, pattern, zero);
    newOp->setAttr(utils::maskBitWidth,
                   rewriter.getIntegerAttr(rewriter.getIntegerType(32), 16));
    break;
  default:
    newOp = rewriter.create<PgeB32>(loc, dstType, pattern, zero);
    newOp->setAttr(utils::maskBitWidth,
                   rewriter.getIntegerAttr(rewriter.getIntegerType(32), 32));
  }
  return newOp;
}

static Operation *createPltIntrinOp(PatternRewriter &rewriter, VFPltOp plt,
                                    unsigned dataWidth) {
  auto loc = plt->getLoc();
  Type dstType = VectorType::get(SmallVector<int64_t>{util::PREDICATE_BITS},
                                 rewriter.getI1Type());
  dstType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                             {dstType, rewriter.getI32Type()});
  Value true_shape = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI32Type(), plt.getTrueShape());
  Value newOp;
  Operation *extractOp;
  switch (dataWidth) {
  case 1:
  case 8:
    newOp = rewriter.create<PltB8>(loc, dstType, true_shape);
    extractOp = rewriter.create<LLVM::ExtractValueOp>(loc, newOp, 0);
    extractOp->setAttr(utils::maskBitWidth,
                       rewriter.getIntegerAttr(rewriter.getIntegerType(32), 8));
    break;
  case 16:
    newOp = rewriter.create<PltB16>(loc, dstType, true_shape);
    extractOp = rewriter.create<LLVM::ExtractValueOp>(loc, newOp, 0);
    extractOp->setAttr(
        utils::maskBitWidth,
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 16));
    break;
  default:
    newOp = rewriter.create<PltB32>(loc, dstType, true_shape);
    extractOp = rewriter.create<LLVM::ExtractValueOp>(loc, newOp, 0);
    extractOp->setAttr(
        utils::maskBitWidth,
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 32));
  }
  return extractOp;
}

static Value findProperMaskOrCreateOneAccordingToMaskOpIdxAttr(
    PatternRewriter &rewriter, Value mask, unsigned dataWidth) {
  Operation *maskOp = mask.getDefiningOp();
  assert(maskOp != nullptr);
  auto maskOpIdxAttr = maskOp->getAttr(utils::maskOpIdx);
  int maskOpIdx = dyn_cast<IntegerAttr>(maskOpIdxAttr).getInt();

  assert(mask.getDefiningOp() != nullptr);
  auto funcOp = mask.getDefiningOp()->getParentOfType<func::FuncOp>();
  bool found = false;
  funcOp->walk([&](Operation *op) {
    if (auto candidateMaskOpIdxAttr = op->getAttr(utils::maskOpIdx)) {
      if (auto candidateMaskBitWidthAttr = op->getAttr(utils::maskBitWidth)) {
        int candidateMaskOpIdx =
            dyn_cast<IntegerAttr>(candidateMaskOpIdxAttr).getInt();
        int candidateMaskBitWidth =
            dyn_cast<IntegerAttr>(candidateMaskBitWidthAttr).getInt();
        if (maskOpIdx == candidateMaskOpIdx &&
            int(dataWidth) == candidateMaskBitWidth) {
          found = true;
          mask = op->getResult(0);
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  if (!found) {
    Operation *newOp = nullptr;
    // Insert the instruction at the original mask position to ensure the
    // dominance relationship when the instruction is reused.
    auto originPos = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(maskOp);
    if (auto plt = dyn_cast<VFPltOp>(maskOp)) {
      newOp = createPltIntrinOp(rewriter, plt, dataWidth);
    } else if (auto pge = dyn_cast<VFPgeOp>(maskOp)) {
      newOp = createPgeIntrinOp(rewriter, pge, dataWidth);
    }
    rewriter.restoreInsertionPoint(originPos);
    if (newOp) {
      newOp->setAttr(utils::maskOpIdx, maskOpIdxAttr);
      mask = newOp->getResult(0);
    }
  }
  return getVLRegValueOrSelf(mask, rewriter);
}

static int getParentOpElementAlignmentBitWidth(Operation *op) {
  if (op != nullptr && op->getParentOp() != nullptr) {
    if (auto elementAlignmentAttr =
            op->getParentOp()->getAttr(utils::elementAlignmentBitWidth)) {
      return dyn_cast<mlir::IntegerAttr>(elementAlignmentAttr).getInt();
    }
  }
  return -1;
}

static int getOpElementAlignmentBitWidth(Operation *op) {
  if (op != nullptr) {
    if (auto elementAlignmentAttr =
            op->getAttr(utils::elementAlignmentBitWidth)) {
      return dyn_cast<mlir::IntegerAttr>(elementAlignmentAttr).getInt();
    }
  }
  return -1;
}

static int getElementAlignmentBitWidth(Operation *op) {
  int elementAlignment = getParentOpElementAlignmentBitWidth(op);
  if (elementAlignment != -1) {
    return elementAlignment;
  }
  return getOpElementAlignmentBitWidth(op);
}

static bool isAlignByElementAlignment(Operation *op) {
  if (op->getUsers().empty()) {
    return false;
  }
  if (isa<func::CallOp>(*(op->getUsers().begin()))) {
    // result is used by call op, assume it as aligned.
    return true;
  }
  auto srcAlignment = getOpElementAlignmentBitWidth(op);
  int dstAlignment = -1;

  // All users should be the same alignmentbitwidth,
  // excpet that in some case a user's alignmentbitwidth
  // may not be infered(eg. func.call)
  for (auto user : op->getUsers()) {
    dstAlignment = getOpElementAlignmentBitWidth(user);
    if (dstAlignment != -1) {
      break;
    }
  }
  return srcAlignment == dstAlignment && srcAlignment != -1;
}

static Value findProperMaskOrCreateOne(PatternRewriter &rewriter,
                                       Operation *maskedOp, Value mask,
                                       unsigned dataWidth) {
  int elementAlignment = getElementAlignmentBitWidth(maskedOp);
  if (mask.getDefiningOp() && mask.getDefiningOp()->getAttr(utils::maskOpIdx)) {
    mask = findProperMaskOrCreateOneAccordingToMaskOpIdxAttr(
        rewriter, mask, elementAlignment == -1 ? dataWidth : elementAlignment);
  } else {
    mask = getVLRegValueOrSelf(mask, rewriter);
  }
  return mask;
}

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
        findProperMaskOrCreateOne(rewriter, op, op.getMask(), dataWidth);
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
      llvm_unreachable("Invalid membar_type_value!");
    }
    rewriter.replaceOp(op, membar);
    return success();
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
        findProperMaskOrCreateOne(rewriter, op, op.getMask(), dataWidth);

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

// llvm.hivm.vdups.v256s8.z (i16, <256 x i1>, i32)
// llvm.hivm.vdups.z (i8, <256 x i1>, i32)
static Value I16ToI8(Value scalar, ConversionPatternRewriter &rewriter) {
  Type currentType = scalar.getType();
  Value i8Value = scalar;
  if (auto intType = mlir::dyn_cast<IntegerType>(currentType)) {
    auto i8Type = IntegerType::get(scalar.getContext(), 8);
    if (intType.getWidth() == 16) {
      i8Value =
          rewriter.create<arith::TruncIOp>(scalar.getLoc(), i8Type, scalar);
      return i8Value;
    }
  }
  return i8Value;
}

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
    Value mask = findProperMaskOrCreateOne(rewriter, convertOp,
                                           adaptor.getMask(), dataWidth);
    Value mode = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

    uint64_t vecSize = static_cast<uint64_t>(vecTy.getNumElements());
    auto vlLength = util::VL_BITS / dataWidth;
    VectorType oriVecTy = vecTy;
    if (vecSize != vlLength) {
      vecTy = VectorType::get(SmallVector<int64_t>{vlLength}, elementType);
    }
    mlir::Value result;
    if (elementType.isBF16() || elementType.isF16() ||
        elementType.isSignedInteger(16) || elementType.isSignlessInteger(16) ||
        elementType.isUnsignedInteger(16) || elementType.isF32() ||
        elementType.isSignedInteger(32) || elementType.isSignlessInteger(32) ||
        elementType.isUnsignedInteger(32) || elementType.isFloat8E4M3FN() ||
        elementType.isFloat8E5M2()) {
      result = rewriter.create<IntrOpTy>(loc, vecTy, scalar, mask, mode);
    } else if (elementType.isSignedInteger(8) ||
               elementType.isSignlessInteger(8) ||
               elementType.isUnsignedInteger(8)) {
      Value i8Value = I16ToI8(scalar, rewriter);
      result = rewriter.create<IntrOpTy>(loc, vecTy, i8Value, mask, mode);
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

static Value interleaveDataLayoutForExtCast(ConversionPatternRewriter &rewriter,
                                            Location loc, int unpackCoefficient,
                                            bool isUI, Value src) {
  Type srcElemType = cast<VectorType>(src.getType()).getElementType();
  auto srcVLVectorTy = hivm_regbaseintrins::createVLVectorType(srcElemType);

  unsigned bitWidth = srcElemType.getIntOrFloatBitWidth();
  bool shouldUnpack = (bitWidth == 8 && unpackCoefficient == 4);
  bool isB8 = bitWidth == 8;

  auto intlvType = LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {srcVLVectorTy, srcVLVectorTy});

  auto createIntlv = [&rewriter, loc, srcVLVectorTy, intlvType, src,
                      srcElemType, shouldUnpack,
                      isB8](auto vbrOpType, auto vintlvOpType) -> Value {
    using VbrOp = typename decltype(vbrOpType)::type;
    using VintlvOp = typename decltype(vintlvOpType)::type;
    Value vbr;
    if (srcElemType.isFloat8E4M3FN() || srcElemType.isFloat8E5M2()) {
      // Workaround: CCEC VbrOp does not support FP8 scalar literal.
      // Broadcast an i8 zero and bitcast the result to FP8 vector.

      // TODO: I16 to avoid the problem in #ISSUE#81, fix when cce adapted.
      Value cstZeroI16 = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getZeroAttr(rewriter.getI16Type()));
      auto i8VLVectorTy =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI8Type());
      cstZeroI16 = isB8 ? I16ToI8(cstZeroI16, rewriter) : cstZeroI16;
      Value vbrI8 = rewriter.create<VbrInstrOp>(loc, i8VLVectorTy, cstZeroI16);
      vbr = rewriter.create<LLVM::BitcastOp>(loc, srcVLVectorTy, vbrI8);
    } else {
      // TODO: I16 to avoid the problem in #ISSUE#81, fix when cce adapted.
      Type zeroType = isB8 ? rewriter.getI16Type() : srcElemType;
      Value cstZero = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getZeroAttr(zeroType));
      cstZero = isB8 ? I16ToI8(cstZero, rewriter) : cstZero;
      vbr = rewriter.create<VbrOp>(loc, srcVLVectorTy, cstZero);
    }

    Value result = rewriter.create<VintlvOp>(loc, intlvType, src, vbr);

    if (shouldUnpack) {
      Value extracted = rewriter.create<LLVM::ExtractValueOp>(loc, result, 0);
      result = rewriter.create<VintlvOp>(loc, intlvType, extracted, vbr);
    }
    return result;
  };
  if (srcElemType.isFloat8E4M3FN() || srcElemType.isFloat8E5M2() ||
      srcElemType.isBF16() || srcElemType.isF16() || srcElemType.isF32() ||
      srcElemType.isSignedInteger(8) || srcElemType.isSignlessInteger(8) ||
      srcElemType.isUnsignedInteger(8) || srcElemType.isSignedInteger(16) ||
      srcElemType.isSignlessInteger(16) || srcElemType.isUnsignedInteger(16) ||
      srcElemType.isSignedInteger(32) || srcElemType.isSignlessInteger(32) ||
      srcElemType.isUnsignedInteger(32)) {
    Value intlv = createIntlv(OpTag<VbrInstrOp>{}, OpTag<VintlvInstrOp>{});
    return rewriter.create<LLVM::ExtractValueOp>(loc, intlv, 0);
  }
  llvm_unreachable("Unexpected element type in interleaveDataLayoutForExtCast");
}

static bool isBRCDist(Value dist) {
  // check load dist is brc
  auto constantOp = dyn_cast<LLVM::ConstantOp>(dist.getDefiningOp());
  IntegerAttr attr = dyn_cast<IntegerAttr>(constantOp.getValue());
  uint32_t distValue = (uint32_t)(attr.getValue().getZExtValue());
  return distValue == (uint32_t)hivmave::LoadDist::BRC_B8 ||
         distValue == (uint32_t)hivmave::LoadDist::BRC_B16 ||
         distValue == (uint32_t)hivmave::LoadDist::BRC_B32;
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

// Use pintlv + pintlv + pstu(b32)  to support b8 i1 unalign for vector<64xi1>
// Use pintlv + pstu(b16)  to support b8 i1 unalign for vector<128xi1>
static Operation *createStoreOpFori1Type(ConversionPatternRewriter &rewriter,
                                         Location loc, Value src, Value dataPtr,
                                         uint64_t vecSize) {
  Value pge = createMaskByPGE(rewriter, loc);
  auto srcVLVectorTy =
      hivm_regbaseintrins::createVLVectorType(rewriter.getI1Type());
  auto intlvType = LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {srcVLVectorTy, srcVLVectorTy});
  if (vecSize == 128) {
    Operation *pintlv =
        rewriter.create<PintlvB8InstrOp>(loc, intlvType, src, pge);
    Value res =
        rewriter.create<LLVM::ExtractValueOp>(loc, pintlv->getResult(0), 0);
    Operation *pstuOp = createPstuOp(res, dataPtr, rewriter, 16);
    return pstuOp;
  } else if (vecSize == 64) {
    Operation *pintlv =
        rewriter.create<PintlvB8InstrOp>(loc, intlvType, src, pge);
    Value res =
        rewriter.create<LLVM::ExtractValueOp>(loc, pintlv->getResult(0), 0);
    pintlv = rewriter.create<PintlvB8InstrOp>(loc, intlvType, res, pge);
    Value lowData =
        rewriter.create<LLVM::ExtractValueOp>(loc, pintlv->getResult(0), 0);
    Operation *pstuOp = createPstuOp(lowData, dataPtr, rewriter, 32);
    return pstuOp;
  } else {
    llvm_unreachable("unsupported vecSize!");
  }
}

static Operation *createLoadOpFori1Type(Value dataPtr,
                                        PatternRewriter &rewriter,
                                        int elementAlignment,
                                        uint64_t vecSize) {
  auto loc = dataPtr.getLoc();
  VectorType dstType = VectorType::get(256, rewriter.getI1Type());
  auto asResult = hivm_regbaseintrins::buildVldasOp(dataPtr, rewriter);
  Value USTD = asResult->getResults()[0];
  Value cstZero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  if (elementAlignment == 8) {
    // Use vldu + movvp to support typei32 i1 unalign
    if (vecSize == 64) {
      Operation *usResult = buildVldusPostOp(dataPtr, USTD, cstZero,
                                             rewriter.getI32Type(), rewriter);
      Value extractOp =
          rewriter.create<LLVM::ExtractValueOp>(loc, usResult->getResult(0), 0);
      Operation *result = buildMovvpOp(loc, dstType, extractOp, rewriter, 32);
      return result;
    } else if (vecSize == 128) {
      // Use vldu + movvp(fp16) + pdintlv to support typei16 i1 unalign
      Type elementType = rewriter.getI8Type();
      Operation *usResult =
          buildVldusPostOp(dataPtr, USTD, cstZero, elementType, rewriter);
      Value extractOp =
          rewriter.create<LLVM::ExtractValueOp>(loc, usResult->getResult(0), 0);
      Operation *result = buildMovvpOp(loc, dstType, extractOp, rewriter, 16);
      Value pge = createMaskByPGE(rewriter, loc);
      auto srcVLVectorTy =
          hivm_regbaseintrins::createVLVectorType(rewriter.getI1Type());
      auto intlvType = LLVM::LLVMStructType::getLiteral(
          rewriter.getContext(), {srcVLVectorTy, srcVLVectorTy});
      Operation *pintlv = rewriter.create<PintlvB8InstrOp>(
          loc, intlvType, result->getResult(0), pge);
      Operation *res =
          rewriter.create<LLVM::ExtractValueOp>(loc, pintlv->getResult(0), 0);
      return res;
    }

  } else if (elementAlignment == 16 || elementAlignment == 32) {
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
  return nullptr;
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

unsigned getMaxDataTypeWidths(Operation *op, int elementAlignment) {
  unsigned elementWidth = 0;
  assert(op != nullptr);
  int opElementWidth = getOpElementAlignmentBitWidth(op);
  if (opElementWidth != -1 && opElementWidth != 1) {
    return static_cast<unsigned>(opElementWidth);
  }
  opElementWidth = getParentOpElementAlignmentBitWidth(op);
  if (opElementWidth != -1 && opElementWidth != 1) {
    return static_cast<unsigned>(opElementWidth);
  }
  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    Type optype = operand.getType();
    if (auto vectorType = mlir::dyn_cast<VectorType>(optype)) {
      Type elementType = vectorType.getElementType();
      elementWidth = elementWidth > elementType.getIntOrFloatBitWidth()
                         ? elementWidth
                         : elementType.getIntOrFloatBitWidth();
    }
  }
  elementWidth = (elementWidth == 8 || elementWidth == 16 || elementWidth == 32)
                     ? elementWidth
                     : static_cast<unsigned>(elementAlignment);
  return elementWidth;
}

static Value castToTypeIfNeeded(ConversionPatternRewriter &rewriter,
                                Location loc, Value v, Type dstType) {
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
    int elementAlignment = getElementAlignmentBitWidth(load);
    auto moduleOp = load->getParentOfType<mlir::ModuleOp>();
    bool archIs910_95 = hacc::utils::isAscend950(moduleOp);

    if (archIs910_95 && load->hasAttr(UnalignedAttr::name) &&
        !isBRCDist(dist)) {
      Type elementType = memRefTy.getElementType();
      // use vldus + movvp to support i1 unaligned address
      if (elementType.isInteger(1)) {
        unsigned elementWidth = 0;
        for (auto *userOp : load->getUsers()) {
          elementWidth = getMaxDataTypeWidths(userOp, elementAlignment);
          break;
        }
        auto result =
            createLoadOpFori1Type(dataPtr, rewriter, elementWidth, vecSize);
        if (!result) {
          llvm_unreachable("unsupported elementAlignment!");
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
    } else if (elementType.isFloat8E4M3FN()) {
      auto result = rewriter.create<Vldsx1V256F8E4InstrOp>(loc, vtype, dataPtr,
                                                           offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isFloat8E5M2()) {
      auto result = rewriter.create<Vldsx1V256F8E5InstrOp>(loc, vtype, dataPtr,
                                                           offset, dist, mode);
      rewriter.replaceOp(load, result);
    } else if (elementType.isInteger(1)) {
      dist = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                               rewriter.getI32IntegerAttr(0));
      auto pLoadOp = rewriter.create<PLoadB8InstOp>(loc, vtype, dataPtr, offset,
                                                    dist, mode);
      // loading is compacted by default,
      // if the use case is sparse,
      // it needs to be converted to sparse using pintlv.
      Value pLoadRes = pLoadOp->getResult(0);
      auto origLoadTy = load->getResult(0).getType();
      bool shouldSparsifyLayout =
          elementAlignment == 16 || elementAlignment == 32;
      if (archIs910_95 && shouldSparsifyLayout) {
        Value res = preginterleaveDataLayoutForExtCast(rewriter, loc, pLoadRes,
                                                       elementAlignment);
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

    int elementAlignment = -1;
    if (auto valOp = store.getVal().getDefiningOp()) {
      elementAlignment = getElementAlignmentBitWidth(valOp);
    }
    if (elementAlignment == -1) {
      elementAlignment = getElementAlignmentBitWidth(store);
    }
    auto elemWidth = dElemType.getIntOrFloatBitWidth();
    Value dist = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), static_cast<uint32_t>(store.getPattern()));
    auto moduleOp = store->getParentOfType<mlir::ModuleOp>();
    bool archIs910_95 = hacc::utils::isAscend950(moduleOp);
    Value mask = store.getMask();
    if (mask.getDefiningOp() &&
        mask.getDefiningOp()->getAttr(utils::maskOpIdx)) {
      if (elemWidth == 8 && elementAlignment == 32 && !archIs910_95)
        mask = findProperMaskOrCreateOneAccordingToMaskOpIdxAttr(rewriter, mask,
                                                                 8);
      else
        mask = findProperMaskOrCreateOneAccordingToMaskOpIdxAttr(
            rewriter, mask,
            elementAlignment == -1 ? elemWidth : elementAlignment);
    } else {
      mask = getVLRegValueOrSelf(mask, rewriter);
    }

    if (archIs910_95 && store->hasAttr(UnalignedAttr::name) &&
        !isONEPTDist(dist) && !dElemType.isInteger(1)) {
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
    } else if (dElemType.isFloat8E4M3FN()) {
      auto result = rewriter.create<Vstsx1V256F8E4InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isFloat8E5M2()) {
      auto result = rewriter.create<Vstsx1V256F8E5InstrOp>(
          loc, data, dataPtr, offset, dist, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isInteger(1)) {
      // if store data is sparse, need to convert compact
      if (archIs910_95 && (elementAlignment == 16 || elementAlignment == 32)) {
        auto asResult = createPstuOp(data, dataPtr, rewriter, elementAlignment);
        rewriter.replaceOp(store, asResult);
      } else if (archIs910_95 && elementAlignment == 8 &&
                 store->hasAttr(UnalignedAttr::name)) {
        auto asResult =
            createStoreOpFori1Type(rewriter, loc, data, dataPtr, vecSize);
        rewriter.replaceOp(store, asResult);
      } else {
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

    int elementAlignment = getElementAlignmentBitWidth(store);
    Value mask = store.getMask();
    if (mask.getDefiningOp() &&
        mask.getDefiningOp()->getAttr(utils::maskOpIdx)) {
      mask = findProperMaskOrCreateOneAccordingToMaskOpIdxAttr(
          rewriter, mask,
          elementAlignment == -1 ? elemWidth : elementAlignment);
    } else {
      mask = getVLRegValueOrSelf(mask, rewriter);
    }

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
    } else if (dElemType.isFloat8E5M2()) {
      auto result = rewriter.create<Vsstb1V256F8E5InstrOp>(
          loc, data, dataPtr, strideConfig, mode, mask);
      rewriter.replaceOp(store, result);
    } else if (dElemType.isFloat8E4M3FN()) {
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
    } else if (elementType.isFloat8E4M3FN()) {
      VectorType gatherType =
          VectorType::get({128}, rewriter.getIntegerType(16));
      Value result = rewriter.create<VGatherV256F8E4InstrOp>(
          loc, gatherType, dataPtr, indexVec, mask);
      Value resF8E4 = rewriter.create<LLVM::BitcastOp>(loc, vtype, result);
      rewriter.replaceOp(gather, resF8E4);
    } else if (elementType.isFloat8E5M2()) {
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
      if (!isAlignByElementAlignment(gather))
        result = interleaveDataLayoutForExtCast(rewriter, loc, 2, true, result);
      rewriter.replaceOp(gather, result);
    } else if (elementType.isBF16()) {
      Value result = rewriter.create<VGatherV128BF16InstrOp>(
          loc, vtype, dataPtr, indexVec, mask);
      if (!isAlignByElementAlignment(gather))
        result = interleaveDataLayoutForExtCast(rewriter, loc, 2, true, result);
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
    if (dstType.getRank() != 1)
      return failure();

    int elementAlignment = -1;
    // FIXME: if pge is inside a nested ForOp and this nested ForOp and its
    // parent ForOp may have different elementAlignmentBitWidth, we should use
    // elementAlignmentBitWidth of the parent ForOp. Here only handle user is
    // MaskedStoreOp, other op also should be handled in the future.
    if (auto store = dyn_cast<VFMaskedStoreOp>(*pge->getUsers().begin())) {
      if (auto valOp = store.getVal().getDefiningOp()) {
        elementAlignment = getElementAlignmentBitWidth(valOp);
      }
    }

    // FIXME: Using attribute of op first and then parent op,
    // this is not a best method to solve template maskStoreOp
    // mask bitwidth.
    if (elementAlignment == -1) {
      elementAlignment = getOpElementAlignmentBitWidth(pge);
    }
    if (elementAlignment == -1) {
      elementAlignment = getElementAlignmentBitWidth(pge);
    }
    if (elementAlignment == -1)
      elementAlignment = util::VL_BITS / dstType.getNumElements();
    Operation *newOp = createPgeIntrinOp(rewriter, pge, elementAlignment);
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
    if (dstTyNumElems != util::PREDICATE_BITS)
      dstType = VectorType::get(SmallVector<int64_t>{util::PREDICATE_BITS},
                                rewriter.getI1Type());
    dstType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {dstType, rewriter.getI32Type()});

    auto loc = plt->getLoc();
    Value true_shape = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), plt.getTrueShape());
    Value result;
    Operation *extractOp;

    int elementAlignment = getElementAlignmentBitWidth(plt);
    // when plt is in the top block of VF func, plt's elem align bit width
    // is obtained from the VF func's attr(maybe b8). However, plt is used for
    // like vsts 64xf32. Thus plt.b8 is used for vsts.b32 which is wrong.
    if (auto parFuncOp = plt->getParentOfType<func::FuncOp>()) {
      llvm::SmallSet<int, 4> userElemAlignSet;
      for (auto *u : plt.getResult(0).getUsers()) {
        int curUserElemAlign = getOpElementAlignmentBitWidth(u);
        if (curUserElemAlign != -1) {
          userElemAlignSet.insert(curUserElemAlign);
        }
      }
      if (userElemAlignSet.size() == 1) {
        // fix only the simple case.
        // TODO: refactor the data layout analysis to fix all the cases.
        //       or create new plt op to fix the case of various users.
        elementAlignment = *userElemAlignSet.begin();
      } else if (userElemAlignSet.size() > 1) {
        return plt.emitError("plt has multiple users of various "
                             "elementAlignmentBitWidth"),
               failure();
      }
    }
    if (elementAlignment == -1)
      elementAlignment = util::VL_BITS / dstTyNumElems;

    switch (elementAlignment) {
    case 8:
      result = rewriter.create<PltB8>(loc, dstType, true_shape);
      extractOp = rewriter.create<LLVM::ExtractValueOp>(loc, result, 0);
      extractOp->setAttr(
          utils::maskBitWidth,
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), 8));
      break;
    case 16:
      result = rewriter.create<PltB16>(loc, dstType, true_shape);
      extractOp = rewriter.create<LLVM::ExtractValueOp>(loc, result, 0);
      extractOp->setAttr(
          utils::maskBitWidth,
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), 16));
      break;
    default:
      result = rewriter.create<PltB32>(loc, dstType, true_shape);
      extractOp = rewriter.create<LLVM::ExtractValueOp>(loc, result, 0);
      extractOp->setAttr(
          utils::maskBitWidth,
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), 32));
    }
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
    Type resType = VectorType::get(SmallVector<int64_t>{util::PREDICATE_BITS},
                                   rewriter.getI1Type());
    Type dstType = pltm.getRes().getType();
    auto dstTyNumElems = cast<VectorType>(dstType).getNumElements();
    if (dstTyNumElems != util::PREDICATE_BITS)
      dstType = VectorType::get(SmallVector<int64_t>{util::PREDICATE_BITS},
                                rewriter.getI1Type());

    auto loc = pltm->getLoc();
    int elementAlignment = getElementAlignmentBitWidth(pltm);
    if (elementAlignment == -1)
      elementAlignment = util::VL_BITS / dstTyNumElems;
    if (elementAlignment > 32)
      elementAlignment = 32;

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

    Operation *pltmIntr;
    switch (elementAlignment) {
    case 8:
      pltmIntr = rewriter.create<PltMB8>(loc, resType, div, ub);
      break;
    case 16:
      pltmIntr = rewriter.create<PltMB16>(loc, resType, div, ub);
      break;
    default:
      pltmIntr = rewriter.create<PltMB32>(loc, resType, div, ub);
    }
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

    Value oldMsk = select.getMask();
    Value adaptMsk = adaptor.getMask();
    Value mask = adaptMsk;
    auto parAlign = getParentOpElementAlignmentBitWidth(select);
    auto *oldMskDefOp = oldMsk.getDefiningOp();
    bool unaligned = false;
    if (oldMskDefOp) {
      unaligned = oldMskDefOp->hasAttr(UnalignedAttr::name);
    }
    if (parAlign == -1 && !unaligned && oldMskDefOp) {
      // no global elem align bit width
      int align = getElementAlignmentBitWidth(select);
      int oldMskAlign = getElementAlignmentBitWidth(oldMskDefOp);
      if (oldMskAlign != -1 && oldMskAlign < align) {
        mask = getVLRegValueOrSelf(mask, rewriter);
        mask = preginterleaveDataLayoutForExtCast(rewriter, loc, mask, align);
      }
    }

    mask = findProperMaskOrCreateOne(rewriter, select, mask,
                                     elementType.getIntOrFloatBitWidth());
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
        elementType.isFloat8E4M3FN() || elementType.isFloat8E5M2()) {
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

  Value mask = findProperMaskOrCreateOne(rewriter, cmpOp, cmpOp.getMask(),
                                         elementType.getIntOrFloatBitWidth());

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

  Value mask = findProperMaskOrCreateOne(rewriter, cmpOp, cmpOp.getMask(),
                                         elementType.getIntOrFloatBitWidth());

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
    Value mask =
        findProperMaskOrCreateOne(rewriter, reduction, reduction.getMask(),
                                  elementType.getIntOrFloatBitWidth());
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
    llvm_unreachable("avoid compilation of create<nulltype>");
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
      llvm_unreachable("unsupported CombiningKind (should be detected in "
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
    Value src = broadcast.getSrc();
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
      // TODO: Avoid the problem in #ISSUE#81, fix util cce adapted.
      Value srcI16 =
          rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI16Type(), src);
      auto v256Ty = VectorType::get({util::VL}, rewriter.getI8Type());
      Value i8Value = I16ToI8(srcI16, rewriter);
      auto vbrI8 = rewriter.create<VbrInstrOp>(loc, v256Ty, i8Value);
      Value zeroI16 =
          rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI16IntegerAttr(0));
      Value zeroI8 = I16ToI8(zeroI16, rewriter);
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
      Value i8Value = I16ToI8(src, rewriter);
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, i8Value);
    } else if (srcType.isSignedInteger(8) || srcType.isSignlessInteger(8)) {
      // TODO: Avoid the problem in #ISSUE#81, fix util cce adapted.
      Type i16Type = rewriter.getI16Type();
      Value srcForI8 = rewriter.create<LLVM::ZExtOp>(loc, i16Type, src);
      Value i8Value = I16ToI8(srcForI8, rewriter);
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, i8Value);
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
    } else if (srcType.isFloat8E4M3FN()) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
    } else if (srcType.isFloat8E5M2()) {
      vbr = rewriter.create<VbrInstrOp>(loc, vlType, src);
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
  auto src = vecBrcOp.getSrc();
  auto loc = vecBrcOp->getLoc();
  // TODO: predication mode merging
  auto pModeCst =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
  Type elementType = cast<VectorType>(src.getType()).getElementType();
  auto srcTy = createVLVectorType(elementType);
  src =
      rewriter.create<UnrealizedConversionCastOp>(loc, srcTy, src).getResult(0);
  Value mask = findProperMaskOrCreateOne(rewriter, vecBrcOp, vecBrcOp.getMask(),
                                         elementType.getIntOrFloatBitWidth());
  if (elementType.isUnsignedInteger(8) || elementType.isSignedInteger(8) ||
      elementType.isSignlessInteger(8) || elementType.isUnsignedInteger(16) ||
      elementType.isSignedInteger(16) || elementType.isSignlessInteger(16) ||
      elementType.isF16() || elementType.isUnsignedInteger(32) ||
      elementType.isSignedInteger(32) || elementType.isSignlessInteger(32) ||
      elementType.isF32() || elementType.isBF16()) {
    vdup = rewriter.create<IntrOpTy>(loc, resType, src, mask, pModeCst);
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
    auto dataWidth =
        outElemType.getIntOrFloatBitWidth() > inElemType.getIntOrFloatBitWidth()
            ? outElemType.getIntOrFloatBitWidth()
            : inElemType.getIntOrFloatBitWidth();
    Value mask =
        findProperMaskOrCreateOne(rewriter, srcOp, srcOp.getMask(), dataWidth);
    // If the data have been arranged by ElementAlignment when load, do not need
    // to interleave/deinterleave when type cast, otherwise need.
    Operation *newOp = nullptr;
    if constexpr (std::is_same_v<OpToBeConverted, VFExtFOp>) {
      if (outElemType.isF32()) {
        if (inElemType.isBF16()) {
          newOp = rewriter.create<VcvtffBF162F32InstrOp>(
              loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getPart()));
        } else if (inElemType.isFloat8E4M3FN()) {
          newOp = rewriter.create<VcvtffF8E4M32F32InstrOp>(
              loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getPart()));
        } else if (inElemType.isFloat8E5M2()) {
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
      else if (outElemType.isFloat8E4M3FN())
        newOp = rewriter.create<VcvtffF322F8E4M3InstrOp>(
            loc, dstVLVectorTy, srcCasted, mask, cstValue(srcOp.getRnd()),
            cstValue(srcOp.getSat()), cstValue(srcOp.getPart()));
      else if (outElemType.isFloat8E5M2())
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
      llvm_unreachable("Unsupported type conversion op.");

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
    Value mask = findProperMaskOrCreateOne(rewriter, op, op.getMask(),
                                           elemResType.getIntOrFloatBitWidth());
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
    Value mask = findProperMaskOrCreateOne(rewriter, op, adaptor.getMask(),
                                           elemType.getIntOrFloatBitWidth());

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
        elemType.isBF16() || elemType.isFloat8E4M3FN() ||
        elemType.isFloat8E5M2()) {
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
        elemType.isBF16() || elemType.isFloat8E4M3FN() ||
        elemType.isFloat8E5M2()) {
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

    Type elementType = vecType.getElementType();
    auto dataWidth = elementType.getIntOrFloatBitWidth();
    Value mask =
        findProperMaskOrCreateOne(rewriter, op, op.getMask(), dataWidth);
    Type vmullType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {llvmVecVLType, llvmVecVLType});

    Operation *vmullOp =
        createVmullIntrinsic(rewriter, loc, vmullType, src1Casted->getResult(0),
                             src2Casted->getResult(0), elemType, mask);
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
                                  Value src2, Type elemType, Value mask) const {
    if (elemType.isSignedInteger(32) || elemType.isSignlessInteger(32)) {
      return rewriter.create<VmullV64S32InstrOp>(loc, vecType, src1, src2,
                                                 mask);
    } else if (elemType.isUnsignedInteger(32)) {
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

    Value mask = findProperMaskOrCreateOne(
        rewriter, op, adaptor.getMask(),
        resType.getElementType().getIntOrFloatBitWidth());
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
    Value mask =
        findProperMaskOrCreateOne(rewriter, op, op.getMask(), dataWidth);
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

    Value mask =
        findProperMaskOrCreateOne(rewriter, op, op.getMask(), dataWidth);
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
    Value mask = findProperMaskOrCreateOne(
        rewriter, op, adaptor.getMask(),
        resType.getElementType().getIntOrFloatBitWidth());

    Value oldLhs = op.getLhs();
    Value oldRhs = op.getRhs();
    Value adaptLhs = adaptor.getLhs();
    Value adaptRhs = adaptor.getRhs();
    Value lhs = getVLRegValueOrSelf(adaptLhs, rewriter);
    Value rhs = getVLRegValueOrSelf(adaptRhs, rewriter);
    auto parAlign = getParentOpElementAlignmentBitWidth(op);
    if (parAlign == -1) {
      // no global elem align bit width
      int align = getOpElementAlignmentBitWidth(op);
      auto *oldLhsDefOp = oldLhs.getDefiningOp();
      int oldLhsAlign = getOpElementAlignmentBitWidth(oldLhsDefOp);
      bool unaligned = oldLhsDefOp && oldLhsDefOp->hasAttr(UnalignedAttr::name);
      if (oldLhsAlign != -1 && oldLhsAlign < align && !unaligned) {
        lhs = preginterleaveDataLayoutForExtCast(rewriter, loc, lhs, align);
      }
      auto *oldRhsDefOp = oldRhs.getDefiningOp();
      int oldRhsAlign = getElementAlignmentBitWidth(oldRhsDefOp);
      unaligned = oldRhsDefOp && oldRhsDefOp->hasAttr(UnalignedAttr::name);
      if (oldRhsAlign != -1 && oldRhsAlign < align && !unaligned) {
        rhs = preginterleaveDataLayoutForExtCast(rewriter, loc, rhs, align);
      }
    }
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
      llvm_unreachable("Unsupported operation type.");
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
    // TODO: Avoid the problem in #ISSUE#81, fix util cce adapted.
    if (auto constIndex = dyn_cast_or_null<arith::ConstantOp>(indexOp)) {
      if (dyn_cast_if_present<IntegerAttr>(constIndex.getValue())) {
        auto constIndexAttr =
            dyn_cast_if_present<IntegerAttr>(constIndex.getValue());
        src1 = rewriter.create<LLVM::ConstantOp>(
            loc, elemType.isInteger(8) ? rewriter.getI16Type() : elemType,
            constIndexAttr.getInt());
      } else {
        auto constIndexAttr =
            dyn_cast_if_present<FloatAttr>(constIndex.getValue());
        src1 = rewriter.create<LLVM::ConstantOp>(loc, elemType,
                                                 constIndexAttr.getValue());
      }
    } else {
      src1 =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, elemType.isInteger(8) ? rewriter.getI16Type() : elemType,
                  index)
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
      Value i8Value = I16ToI8(src1, rewriter);
      result = rewriter.create<VciInstrOp>(loc, targetType, i8Value, src2);
    } else if (elemType.isF32() && resType.getNumElements() <= 64) {
      result = rewriter.create<VciInstrOp>(loc, resType, src1, src2);
    } else if (elemType.isF16() && resType.getNumElements() <= 128) {
      result = rewriter.create<VciInstrOp>(loc, resType, src1, src2);
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
    Value mask =
        findProperMaskOrCreateOne(rewriter, op, op.getMask(), dataWidth);

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
               HIVMVmullOpLowering
  >(converter);
  BinaryRegistry::registerPatterns(converter, patterns);
  UnaryRegistry::registerPatterns(converter, patterns);
  TenaryRegistry::registerPatterns(converter, patterns);
  BroadCastScalarRegistry::registerPatterns(converter, patterns);
  BinaryVectorScalarRegistry::registerPatterns(converter, patterns);
  // clang-format on
}

namespace {
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

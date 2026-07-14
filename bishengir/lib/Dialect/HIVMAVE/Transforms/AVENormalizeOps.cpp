//===---------------------------- AVENormalizeOps.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <mlir/IR/Attributes.h>

#define DEBUG_TYPE "ave-normalize-ops"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_AVENORMALIZEOPS
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivmave;

static std::optional<hivmave::VecMemType> getVecMemType(Type vector) {
  auto vecType = dyn_cast<VectorType>(vector);
  if (!vecType)
    return std::nullopt;
  auto layout = dyn_cast_or_null<hivmave::VectorLayoutAttr>(vecType.getLayout());
  if (!layout)
    return std::nullopt;
  auto memAttr = dyn_cast<hivmave::VecMemTypeAttr>(layout.getMem());
  if (!memAttr)
    return std::nullopt;
  return memAttr.getValue();
}

static bool breaksPB16AlignmentOnNextContinuousStore(VFMaskedStoreOp store) {
  if (!store->hasAttr("hivm.is_continuous") ||
      !store->getParentOfType<scf::ForOp>())
    return false;

  // The "hivm.is_continuous" marker is produced from
  // loopStep * memrefStride == vectorLength, so adjacent stores advance by one
  // stored vector.  PB16 predicate stores require the next start address to
  // preserve 16-byte granularity.
  constexpr int64_t pb16AlignmentBits = 16 * 8;
  const int64_t vectorElements = store.getVectorType().getNumElements();
  const int64_t elementBitWidth =
      static_cast<int64_t>(store.getMemRefType().getElementTypeBitWidth());
  const int64_t nextStoreDeltaBits = vectorElements * elementBitWidth;

  return nextStoreDeltaBits % pb16AlignmentBits != 0;
}

/// Before conversion:
/// ```mlir
///    %res = ave.hir.vload <NORM> %subview[%c0] {ave.unaligned_ub_access =
///     #ave.unaligned_ub_access,
///     functionType = #ave.func_dist_type<pb8>} : memref<64xi1, strided<[1],
///     offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
///    %6 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<64xi1>
///    %7 = ave.hir.preg.and <b8> %5, %res, %6 : vector<64xi1>
/// ```
/// After conversion:
/// ```mlir
///    %res = ave.hir.vload <NORM> %subview[%c0] {ave.unaligned_ub_access =
///     #ave.unaligned_ub_access,
///     functionType = #ave.func_dist_type<pb16>} : memref<64xi1,
///     strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
///    %6 = ave.hir.preg.cast %res <PK_B16> : vector<64xi1> -> vector<64xi1>  
///    %7 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<64xi1>
///    %10 = ave.hir.preg.and <b8> %5, %6, %7 : vector<64xi1>
/// ```
static LogicalResult insertPregCastAfterLoad(VFLoadOp load,
                                             VectorType vectorTy,
                                             PatternRewriter &rewriter) {
  auto vecMemTypeOpt = getVecMemType(load.getResult(0).getType());
  if (!vecMemTypeOpt)
    return failure();
  auto vecMemType = *vecMemTypeOpt;

  if (vecMemType == hivmave::VecMemType::B8) {
    load->setAttr("functionType", hivmave::FunctionDistTypeAttr::get(
                                      load.getContext(),
                                      hivmave::FunctionDistType::PB16));
  } else {
    return failure();
  }

  rewriter.setInsertionPointAfter(load);
  auto castOp = rewriter.create<hivmave::VFPregTypeCastOp>(
      load.getLoc(), vectorTy, load.getResult(0),
      hivmave::PregCastMode::PK_B16);

  for (auto &use : llvm::make_early_inc_range(load.getResult(0).getUses())) {
    if (use.getOwner() != castOp) {
      use.set(castOp.getResult());
    }
  }
  return success();
}

/// Before conversion:
/// ```mlir
///    ave.hir.masked_store <NORM_B8> %subview_0[%c0], %8, %7
///    {ave.unaligned_ub_access = #ave.unaligned_ub_access,
///    functionType = #ave.func_dist_type<pb8>,
///    hivm.is_continuous} : memref<64xi1,
///    strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>,
///    vector<64xi1>
/// ```
/// After conversion:
/// ```mlir
///   %12 = ave.hir.preg.cast %11 <UNPK4_B8> : vector<64xi1> -> vector<64xi1>
///       ave.hir.masked_store <NORM_B8> %subview_0[%c0], %10, %12
///       {ave.unaligned_ub_access = #ave.unaligned_ub_access,
///       functionType = #ave.func_dist_type<pb32>,
///       hivm.is_continuous} : memref<64xi1,
///       strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>,
///       vector<64xi1>
/// ```
static LogicalResult
insertPregCastBeforeStore(VFMaskedStoreOp store, VectorType maskType,
                          hivmave::FunctionDistType funcDist,
                          PatternRewriter &rewriter) {
  auto vecMemTypeOpt = getVecMemType(store.getVal().getType());
  if (!vecMemTypeOpt)
    return failure();
  auto vecMemType = *vecMemTypeOpt;
  Value originalValue = store.getVal();
  VectorType vectorTy = store.getVectorType();
  auto vecSize = vectorTy.getNumElements();
  hivmave::PregCastMode castMode;
  if (vecMemType == hivmave::VecMemType::B8) {
    if (funcDist != hivmave::FunctionDistType::PB8)
      return failure();
    if (vecSize == 64) {
      store->setAttr("functionType", hivmave::FunctionDistTypeAttr::get(
                                         store.getContext(),
                                         hivmave::FunctionDistType::PB32));
      castMode = hivmave::PregCastMode::UNPK4_B8;
    } else if (vecSize == 128) {
      store->setAttr("functionType", hivmave::FunctionDistTypeAttr::get(
                                         store.getContext(),
                                         hivmave::FunctionDistType::PB16));
      castMode = hivmave::PregCastMode::UNPK_B8;
    } else {
      return failure();
    }
  } else if (vecMemType == hivmave::VecMemType::B16) {
    if (funcDist != hivmave::FunctionDistType::PB16 || vecSize != 64 ||
        !breaksPB16AlignmentOnNextContinuousStore(store))
      return failure();
    store->setAttr("functionType", hivmave::FunctionDistTypeAttr::get(
                                       store.getContext(),
                                       hivmave::FunctionDistType::PB32));
    castMode = hivmave::PregCastMode::UNPK_B16;
  } else {
    return failure();
  }

  rewriter.setInsertionPoint(store);
  auto castOp = rewriter.create<hivmave::VFPregTypeCastOp>(
      store.getLoc(), maskType, originalValue, castMode);

  store->setOperand(3, castOp.getResult());
  return success();
}

static Value addDistForUnalignedLoad(Value srcVal, hivmave::LoadDist dist,
                                     Location loc, IRRewriter &rewriter) {
  // vldus have no dist operand
  // VLDS support BRC mode with 1Byte alignment.
  // So the BRC mode does not use unaligned instruction.
  // Implement UNPK_xxx by using vintlv
  Value dst = srcVal;
  switch (dist) {
  case hivmave::LoadDist::UNPK_B8:
  case hivmave::LoadDist::UNPK_B16:
  case hivmave::LoadDist::UNPK_B32: {
    dst = sparseByIntlv(srcVal, rewriter, loc);
    break;
  }
  case hivmave::LoadDist::UNPK4_B8: {
    dst = sparseByIntlv(srcVal, rewriter, loc);
    dst = sparseByIntlv(dst, rewriter, loc);
    break;
  }
  default:
    break;
  }
  return dst;
}

static Value addDistForUnalignedStore(Value srcVal, hivmave::StoreDist dist,
                                      Location loc, IRRewriter &rewriter) {
  // vstus have no dist operand
  // Implement ONEPT_xxx by using vsel
  // Implement PK_xxx by using vdintlv
  Value dst = srcVal;
  switch (dist) {
  case hivmave::StoreDist::PK_B16:
  case hivmave::StoreDist::PK_B32: {
    dst = denseByDIntlv(srcVal, rewriter, loc);
    break;
  }
  case hivmave::StoreDist::PK4_B32: {
    dst = denseByDIntlv(srcVal, rewriter, loc);
    dst = denseByDIntlv(dst, rewriter, loc);
    break;
  }
  default:
    break;
  }
  return dst;
}

/// Change NORM to UNPK_B8/UNPK_B16/UNPK4_B8 based on functionType
struct AVELoadPattern : public OpRewritePattern<VFLoadOp> {
  explicit AVELoadPattern(MLIRContext *context)
      : OpRewritePattern<VFLoadOp>(context) {}
  LogicalResult matchAndRewrite(VFLoadOp load,
                                PatternRewriter &rewriter) const override {
    LDBG("process operation : " << load);
    VectorType vectorTy = load.getVectorType();
    auto vecElemTy = vectorTy.getElementType();
    auto elemWidth = vecElemTy.getIntOrFloatBitWidth();
    auto funcDistAttr =
        load->getAttrOfType<FunctionDistTypeAttr>("functionType");
    if (load.getPattern() == hivmave::LoadDist::NORM && funcDistAttr) {
      switch (funcDistAttr.getValue()) {
      case hivmave::FunctionDistType::PK:
        if (elemWidth == 8) {
          load.setPattern(hivmave::LoadDist::UNPK_B8);
          LDBG("set load dist from NORM to UNPK_B8");
          return success();
        } else if (elemWidth == 16) {
          load.setPattern(hivmave::LoadDist::UNPK_B16);
          LDBG("set load dist from NORM to UNPK_B16");
          return success();
        }
        break;
      case hivmave::FunctionDistType::PK4:
        if (elemWidth == 8) {
          load.setPattern(hivmave::LoadDist::UNPK4_B8);
          LDBG("set load dist from NORM to UNPK4_B8");
          return success();
        }
        break;
      default:
        break;
      }
    }

    if (!vecElemTy.isInteger(1) || !load->hasAttr(UnalignedAttr::name)) {
      return failure();
    }
    if (funcDistAttr) {
      auto funcDist = funcDistAttr.getValue();
      if (funcDist == hivmave::FunctionDistType::PB8) {
        return insertPregCastAfterLoad(load, vectorTy, rewriter);
      }
    }
    return failure();
  }
};

/// Change NORM to PK_B16/PK_B32/PK4_B32 based on functionType
struct AVEStorePattern : public OpRewritePattern<VFMaskedStoreOp> {
  explicit AVEStorePattern(MLIRContext *context)
      : OpRewritePattern<VFMaskedStoreOp>(context) {}

  static void changeStoreVectorType(VFMaskedStoreOp store,
                                    PatternRewriter &rewriter, int factor,
                                    int targetWidth) {
    if (store->hasAttr(UnalignedAttr::name))
      return;
    Value data = store.getVal();
    Location loc = store->getLoc();
    VectorType orgVectorTy = store.getVectorType();
    Type dElemType = orgVectorTy.getElementType();
    int64_t vecSize = orgVectorTy.getNumElements();
    int64_t regSize = util::VL_BITS / dElemType.getIntOrFloatBitWidth();
    int64_t targetRegSize = regSize / factor;
    Type targetRegType = rewriter.getIntegerType(targetWidth);
    if (llvm::isa<FloatType>(dElemType))
      targetRegType =
          targetWidth == 16 ? rewriter.getF16Type() : rewriter.getF32Type();
    if (vecSize != regSize) {
      VectorType castRegType =
          VectorType::get(SmallVector<int64_t>{regSize}, dElemType);
      UnrealizedConversionCastOp ucc =
          rewriter.create<UnrealizedConversionCastOp>(loc, castRegType, data);
      data = ucc->getResult(0);
    }
    VectorType targetVectorTy = VectorType::get(targetRegSize, targetRegType);
    LLVM::BitcastOp bitcast =
        rewriter.create<LLVM::BitcastOp>(loc, targetVectorTy, data);
    store->setOperand(store->getNumOperands() - 1, bitcast);
  }

  LogicalResult matchAndRewrite(VFMaskedStoreOp store,
                                PatternRewriter &rewriter) const override {
    LDBG("process operation : " << store);
    auto moduleOp = store->getParentOfType<ModuleOp>();
    bool archIs910_95 = hacc::utils::isAscend950(moduleOp);
    VectorType vectorTy = store.getVectorType();
    auto vecElemTy = vectorTy.getElementType();
    auto elemWidth = vecElemTy.getIntOrFloatBitWidth();
    auto funcDistAttr =
        store->getAttrOfType<FunctionDistTypeAttr>("functionType");

    if ((store.getPattern() == hivmave::StoreDist::NORM_B8 ||
         store.getPattern() == hivmave::StoreDist::NORM_B16) && funcDistAttr) {
      switch (funcDistAttr.getValue()) {
      case hivmave::FunctionDistType::PK:
        if (elemWidth == 8) {
          store.setPattern(hivmave::StoreDist::PK_B16);
          LDBG("set store dist from NORM to PK_B16");
          changeStoreVectorType(store, rewriter, 2, 16);
          return success();
        } else if (elemWidth == 16) {
          store.setPattern(hivmave::StoreDist::PK_B32);
          LDBG("set store dist from NORM to PK_B32");
          changeStoreVectorType(store, rewriter, 2, 32);
          return success();
        }
        break;
      case hivmave::FunctionDistType::PK4:
        if (elemWidth == 8) {
          store.setPattern(hivmave::StoreDist::PK4_B32);
          LDBG("set store dist from NORM to PK4_B32");
        if (archIs910_95)
          changeStoreVectorType(store, rewriter, 4, 32);
        return success();
        }
        break;
      default:
        break;
      }
    }

    auto maskType = dyn_cast<VectorType>(store.getVal().getType());
    bool isMaskI1Type = maskType && maskType.getElementType().isInteger(1);
    if (!isMaskI1Type || !store->hasAttr(UnalignedAttr::name)) {
      return failure();
    }
    if (funcDistAttr) {
      auto funcDist = funcDistAttr.getValue();
      if (funcDist == hivmave::FunctionDistType::PB8 ||
          funcDist == hivmave::FunctionDistType::PB16) {
        return insertPregCastBeforeStore(store, maskType, funcDist, rewriter);
      }
    }
    return failure();
  }
};

/// Handle func_dist_type (DINTLV2/DINTLV4) on store_with_stride ops by
/// inserting dintlv ops before the store to densify the data layout.
struct AVEStoreWithStridePattern
    : public OpRewritePattern<VFStoreWithStrideOp> {
  explicit AVEStoreWithStridePattern(MLIRContext *context)
      : OpRewritePattern<VFStoreWithStrideOp>(context) {}

  LogicalResult matchAndRewrite(VFStoreWithStrideOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto funcDistAttr =
        storeOp->getAttrOfType<FunctionDistTypeAttr>("functionType");
    if (!funcDistAttr)
      return failure();

    int numDIntlv = 0;
    switch (funcDistAttr.getValue()) {
    case FunctionDistType::DINTLV2:
      numDIntlv = 1;
      break;
    case FunctionDistType::DINTLV4:
      numDIntlv = 2;
      break;
    default:
      return failure();
    }

    Location loc = storeOp.getLoc();
    Value srcVal = storeOp.getVal();

    rewriter.setInsertionPoint(storeOp);
    Value result = srcVal;
    for (int i = 0; i < numDIntlv; ++i)
      result = denseByDIntlv(result, rewriter, loc);

    storeOp->setOperand(storeOp->getNumOperands() - 1, result);
    storeOp->removeAttr("functionType");
    return success();
  }
};

/// Handle func_dist_type (INTLV2/INTLV4) on ops by inserting intlv ops
/// after the op to sparsify the data layout.
/// Supports single-result ops (VFGatherOp, VFVCIOp) via getRes()
/// and two-result ops (VFInterleaveOp, VFDeInterleaveOp) via getRes1()/getRes2().
template <typename IntlvOp>
struct AVEIntlvFuncDistPattern : public OpRewritePattern<IntlvOp> {
  AVEIntlvFuncDistPattern(MLIRContext *context)
      : OpRewritePattern<IntlvOp>(context) {}

  LogicalResult matchAndRewrite(IntlvOp intlvOp,
                                PatternRewriter &rewriter) const override {
    auto funcDistAttr =
        intlvOp->template getAttrOfType<FunctionDistTypeAttr>("functionType");
    if (!funcDistAttr)
      return failure();

    int numIntlv = 0;
    switch (funcDistAttr.getValue()) {
    case FunctionDistType::INTLV2:
      numIntlv = 1;
      break;
    case FunctionDistType::INTLV4:
      numIntlv = 2;
      break;
    default:
      return failure();
    }

    Location loc = intlvOp.getLoc();

    rewriter.setInsertionPointAfter(intlvOp);

    auto processResult = [&](Value res) {
      SmallVector<Operation *> oldUsers(res.getUsers());
      if (oldUsers.empty())
        return;

      Value result = res;
      for (int i = 0; i < numIntlv; ++i)
        result = sparseByIntlv(result, rewriter, loc);
      for (Operation *user : oldUsers)
        user->replaceUsesOfWith(res, result);
    };

    if constexpr (std::is_same_v<IntlvOp, VFInterleaveOp> ||
                  std::is_same_v<IntlvOp, VFDeInterleaveOp>) {
      processResult(intlvOp.getRes1());
      processResult(intlvOp.getRes2());
    } else {
      processResult(intlvOp.getRes());
    }

    intlvOp->removeAttr("functionType");
    return success();
  }
};

/// Instead of materializing intlv/dintlv for VectorLayoutCastOp's functionType,
/// move the functionType attribute to the defining op of srcVal.
/// The actual intlv/dintlv lowering will be handled in ConvertHIVMAVEToAVEIntrin.
struct AVEVectorLayoutCastPattern
    : public OpRewritePattern<hivmave::VectorLayoutCastOp> {
  explicit AVEVectorLayoutCastPattern(MLIRContext *context)
      : OpRewritePattern<hivmave::VectorLayoutCastOp>(context) {}

  LogicalResult matchAndRewrite(hivmave::VectorLayoutCastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto funcDistAttr =
        castOp->getAttrOfType<FunctionDistTypeAttr>("functionType");
    if (!funcDistAttr)
      return failure();

    Value srcVal = castOp.getSrc();
    Operation *definingOp = srcVal.getDefiningOp();
    if (!definingOp) {
      // src is a block argument; cannot move attribute.
      // Keep the cast as-is; RemoveVectorLayoutAttr will clean it up.
      LDBG("AVEVectorLayoutCastPattern: src is a block argument, "
           << "cannot move functionType from " << castOp);
      return failure();
    }

    // Move functionType from castOp to its defining op.
    LDBG("AVEVectorLayoutCastPattern: moving functionType from "
         << castOp << " to " << *definingOp);
    definingOp->setAttr("functionType", funcDistAttr);
    castOp->removeAttr("functionType");
    return success();
  }
};

/// Hardware does not support fp16-->u16
/// Use fp16-->s32 + s32-->u16 instead.
struct AVEFpToUIntPattern : public OpRewritePattern<VFFpToUIntOp> {
  explicit AVEFpToUIntPattern(MLIRContext *context)
      : OpRewritePattern<VFFpToUIntOp>(context) {}
  LogicalResult matchAndRewrite(VFFpToUIntOp cvtOp,
                                PatternRewriter &rewriter) const override {
    VectorType resType = cast<VectorType>(cvtOp.getResult().getType());
    VectorType srcType = cast<VectorType>(cvtOp.getSrc().getType());
    Type outElemType = resType.getElementType();
    Type inElemType = srcType.getElementType();
    if (inElemType.isF16() && outElemType.isSignlessInteger(16)) {
      LDBG("process operation : " << cvtOp);
      auto loc = cvtOp.getLoc();
      auto ctx = cvtOp->getContext();
      hivm::UnsignedModeAttr s2uAttr =
          hivm::UnsignedModeAttr::get(ctx, hivm::UnsignedMode::SI2UI);
      rewriter.setInsertionPointAfter(cvtOp);
      int64_t totalElements = srcType.getNumElements();
      LDBG("totalElements :" << totalElements);
      if (totalElements <= 64) {
        VectorType i32VecType =
            VectorType::get({totalElements}, rewriter.getI32Type());
        auto cvtF162I32 = rewriter.create<VFFpToSIntOp>(
            loc, i32VecType, cvtOp.getSrc(), cvtOp.getMask(),
            cvtOp.getRndAttr(), nullptr, cvtOp.getPartAttr());
        auto cvtI322U16 = rewriter.create<VFTruncIOp>(
            loc, resType, cvtF162I32, cvtOp.getMask(), cvtOp.getSat(),
            cvtOp.getPartAttr(), nullptr, s2uAttr);
        rewriter.replaceOp(cvtOp, cvtI322U16);
        LDBG("replace by : \n" << cvtF162I32 << "\n" << cvtI322U16);
      } else {
        assert(totalElements == 128 && "Only support 128 for now");
        auto innerVecTy = VectorType::get({util::VL_B32}, inElemType);
        auto srcStructTy =
            LLVM::LLVMStructType::getLiteral(ctx, {innerVecTy, innerVecTy});
        Value srcCastedStruct = rewriter
                                    .create<UnrealizedConversionCastOp>(
                                        loc, srcStructTy, cvtOp.getSrc())
                                    ->getResult(0);

        auto firstHalf = rewriter.create<LLVM::ExtractValueOp>(
            loc, innerVecTy, srcCastedStruct, ArrayRef<int64_t>{0});
        auto secondHalf = rewriter.create<LLVM::ExtractValueOp>(
            loc, innerVecTy, srcCastedStruct, ArrayRef<int64_t>{1});

        auto s32Ty = VectorType::get({util::VL_B32}, rewriter.getI32Type());
        auto i16Ty = VectorType::get({util::VL_B32}, rewriter.getI16Type());
        auto cvtF162I32Op1 = rewriter.create<VFFpToSIntOp>(
            loc, s32Ty, firstHalf, cvtOp.getMask(), cvtOp.getRndAttr(), nullptr,
            cvtOp.getPartAttr());
        auto cvtI322U16Op1 = rewriter.create<VFTruncIOp>(
            loc, i16Ty, cvtF162I32Op1, cvtOp.getMask(), cvtOp.getSat(),
            cvtOp.getPartAttr(), nullptr, s2uAttr);
        auto cvtF162I32Op2 = rewriter.create<VFFpToSIntOp>(
            loc, s32Ty, secondHalf, cvtOp.getMask(), cvtOp.getRndAttr(),
            nullptr, cvtOp.getPartAttr());
        auto cvtI322U16Op2 = rewriter.create<VFTruncIOp>(
            loc, i16Ty, cvtF162I32Op2, cvtOp.getMask(), cvtOp.getSat(),
            cvtOp.getPartAttr(), nullptr, s2uAttr);

        auto dstStructTy =
            LLVM::LLVMStructType::getLiteral(ctx, {i16Ty, i16Ty});
        auto zeroVector = rewriter.create<LLVM::UndefOp>(loc, dstStructTy);
        auto merged1 = rewriter.create<LLVM::InsertValueOp>(
            loc, dstStructTy, zeroVector, cvtI322U16Op1, ArrayRef<int64_t>{0});
        auto merged2 = rewriter.create<LLVM::InsertValueOp>(
            loc, dstStructTy, merged1, cvtI322U16Op2, ArrayRef<int64_t>{1});
        auto newOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, resType, ValueRange{merged2.getResult()});
        rewriter.replaceOp(cvtOp, newOp);
        LDBG("replace by : \n"
             << cvtF162I32Op1 << "\n"
             << cvtI322U16Op1 << "\n"
             << cvtF162I32Op2 << "\n"
             << cvtI322U16Op2);
      }
    }
    return failure();
  }
};

struct AVEPgePattern : public OpRewritePattern<VFPgeOp> {
  explicit AVEPgePattern(MLIRContext *context)
      : OpRewritePattern<VFPgeOp>(context) {}
  LogicalResult matchAndRewrite(VFPgeOp pge,
                                PatternRewriter &rewriter) const override {
    VectorType dstType = cast<VectorType>(pge.getRes().getType());
    auto dstTyNumElems = cast<VectorType>(dstType).getNumElements();
    int elementAlignment = getBitWidthFromAttr(pge);
    if (elementAlignment == -1)
      elementAlignment = util::VL_BITS / dstTyNumElems;
    PgePattern pattern = pge.getPattern();
    PgePattern normPattern = pattern;
    if (pattern == PgePattern::ALL &&
        dstTyNumElems != util::VL_BITS / elementAlignment)
      normPattern = hivmave::getPgePatternAttr(rewriter, dstTyNumElems,
                                           util::PREDICATE_BITS)
                    .value()
                    .getValue();
    // Use pattern all/half instead of const int
    switch (elementAlignment) {
    case 8:
      if (normPattern == PgePattern::VL128)
        normPattern = PgePattern::H;
      break;
      break;
    case 16:
      if (normPattern == PgePattern::VL128)
        normPattern = PgePattern::ALL;
      else if (pattern == PgePattern::VL64)
        normPattern = PgePattern::H;
      break;
    case 32:
      if (normPattern == PgePattern::VL64)
        normPattern = PgePattern::ALL;
      else if (normPattern == PgePattern::VL32)
        normPattern = PgePattern::H;
      break;
    default:
      llvm_unreachable("Invalid element bit width for a predicate vector.");
    }
    if (normPattern == pattern)
      return failure();
    pge.setPattern(normPattern);
    return success();
  }
};

static void adaptBitWidthForLoad(IRRewriter &rewriter,
                                 mlir::func::FuncOp &funcOp,
                                 bool archIs910_95) {
  funcOp->walk([&rewriter, archIs910_95](VFLoadOp load) {
    LDBG("process operation : " << load);
    hivmave::LoadDist dist = load.getPattern();
    Location loc = load->getLoc();
    auto dstVec = load.getRes();
    rewriter.setInsertionPointAfter(load);
    SmallVector<Operation *> oldUsers(dstVec.getUsers());
    // Add additional op to adpat dist mode for unaligned load
    if (archIs910_95 && load->hasAttr(UnalignedAttr::name)) {
      Value result =
          addDistForUnalignedLoad(dstVec, dist, loc, rewriter);
      if (result != dstVec) {
        LDBG("add intlv for unaligned load");
        for (Operation *u : oldUsers)
          u->replaceUsesOfWith(dstVec, result);
      }
    }
    // If data bits = 1/4 * ElementAlignment, because vlds/vsts
    // intrin of 310b4 doesn't support UNPK4 or PK4 dist, we have to
    // load/store data in compact manner and interleave data after load
    // and deinterleave data before store.
    if (!archIs910_95 && dist == hivmave::LoadDist::UNPK4_B8) {
      LDBG("add intlv for 310b4");
      Value sparseVec1 = sparseByIntlv(dstVec, rewriter, loc);
      Value sparseVec2 = sparseByIntlv(sparseVec1, rewriter, loc);
      for (Operation *u : oldUsers)
        u->replaceUsesOfWith(dstVec, sparseVec2);
      load.setPattern(hivmave::LoadDist::NORM);
    }
    return WalkResult::advance();
  });
}

static void adaptBitWidthForStore(IRRewriter &rewriter,
                                  mlir::func::FuncOp &funcOp,
                                  bool archIs910_95) {
  funcOp->walk([&rewriter, archIs910_95](VFMaskedStoreOp store) {
    LDBG("process operation : " << store);
    hivmave::StoreDist dist = store.getPattern();
    Location loc = store->getLoc();
    auto srcVec = store.getVal();
    rewriter.setInsertionPoint(store);
    // Add additional op to adpat dist mode for unaligned store
    if (archIs910_95 && store->hasAttr(UnalignedAttr::name)) {
      Value result =
          addDistForUnalignedStore(srcVec, dist, loc, rewriter);
      if (result != srcVec) {
        LDBG("add dintlv for unaligned store");
        // set new src vector of store
        store.setOperand(3, result);
      }
    }
    // If data bits = 1/4 * ElementAlignment, because vlds/vsts
    // intrin of 310b4 doesn't support UNPK4 or PK4 dist, we have to
    // load/store data in compact manner and interleave data after load
    // and deinterleave data before store.
    if (!archIs910_95 && dist == hivmave::StoreDist::PK4_B32) {
      LDBG("add dintlv for 310b4");
      Value denseVec1 = denseByDIntlv(srcVec, rewriter, loc);
      Value denseVec2 = denseByDIntlv(denseVec1, rewriter, loc);
      // set new src vector of store
      store.setOperand(3, denseVec2);
      store.setPattern(hivmave::StoreDist::NORM_B8);
    }
    return WalkResult::advance();
  });
}

namespace {
struct AVENormalizeOpsPass
    : public impl::AVENormalizeOpsBase<AVENormalizeOpsPass> {
  using Base::Base;

  void adaptBitWidth() {
    IRRewriter rewriter(&getContext());
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    bool archIs910_95 = hacc::utils::isAscend950(moduleOp);

    adaptBitWidthForLoad(rewriter, funcOp, archIs910_95);
    adaptBitWidthForStore(rewriter, funcOp, archIs910_95);
  }

public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();
    RewritePatternSet patterns(ctx);
    mlir::GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    // add dist for load/store op
    patterns.add<AVELoadPattern>(ctx);
    patterns.add<AVEStorePattern>(ctx);
    patterns.add<AVEStoreWithStridePattern>(ctx);
    patterns.add<AVEFpToUIntPattern>(ctx);
    patterns.add<AVEIntlvFuncDistPattern<VFInterleaveOp>>(ctx);
    patterns.add<AVEIntlvFuncDistPattern<VFDeInterleaveOp>>(ctx);
    patterns.add<AVEVectorLayoutCastPattern>(ctx);
    patterns.add<AVEIntlvFuncDistPattern<VFGatherOp>>(ctx);
    patterns.add<AVEIntlvFuncDistPattern<VFVCIOp>>(ctx);
    // norm pge pattern to all/half, used during bisheng memory analysis
    // TODO: After bisheng memory analysis is complete, it can be deleted
    patterns.add<AVEPgePattern>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }

    // add intlv/dintlv to adapt bitwidth
    adaptBitWidth();
  }
};

} // namespace

std::unique_ptr<Pass> hivmave::createAVENormalizeOpsPass() {
  return std::make_unique<AVENormalizeOpsPass>();
}

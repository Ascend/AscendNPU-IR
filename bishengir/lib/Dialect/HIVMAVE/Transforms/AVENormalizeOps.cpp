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
    // If user is scf.for, trace through iter_args to find the real consumer
    // alignment, since scf.for itself does not carry alignment attributes.
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto [idx, initVal] : llvm::enumerate(forOp.getInitArgs())) {
        for (auto res : op->getResults()) {
          if (initVal == res) {
            Value forResult = forOp.getResult(idx);
            for (auto forResultUser : forResult.getUsers()) {
              dstAlignment = getOpElementAlignmentBitWidth(forResultUser);
              if (dstAlignment != -1)
                break;
            }
            if (dstAlignment != -1)
              break;
          }
        }
        if (dstAlignment != -1)
          break;
      }
    }
  }
  return srcAlignment == dstAlignment && srcAlignment != -1;
}

static Value addDistForUnalignedLoad(Value srcVal, hivmave::LoadDist dist,
                                     Location loc, IRRewriter &rewriter,
                                     Attribute bitWidthAttr) {
  // vldus have no dist operand
  // VLDS support BRC mode with 1Byte alignment.
  // So the BRC mode does not use unaligned instruction.
  // Implement UNPK_xxx by using vintlv
  Value dst = srcVal;
  switch (dist) {
  case hivmave::LoadDist::UNPK_B8:
  case hivmave::LoadDist::UNPK_B16:
  case hivmave::LoadDist::UNPK_B32: {
    dst = sparseByIntlv(srcVal, rewriter, loc, bitWidthAttr);
    break;
  }
  case hivmave::LoadDist::UNPK4_B8: {
    dst = sparseByIntlv(srcVal, rewriter, loc, bitWidthAttr);
    dst = sparseByIntlv(dst, rewriter, loc, bitWidthAttr);
    break;
  }
  default:
    break;
  }
  return dst;
}

static Value addDistForUnalignedStore(Value srcVal, hivmave::StoreDist dist,
                                      Location loc, IRRewriter &rewriter,
                                      Attribute bitWidthAttr) {
  // vstus have no dist operand
  // Implement ONEPT_xxx by using vsel
  // Implement PK_xxx by using vdintlv
  Value dst = srcVal;
  switch (dist) {
  case hivmave::StoreDist::PK_B16:
  case hivmave::StoreDist::PK_B32: {
    dst = denseByDIntlv(srcVal, rewriter, loc, bitWidthAttr);
    break;
  }
  case hivmave::StoreDist::PK4_B32: {
    dst = denseByDIntlv(srcVal, rewriter, loc, bitWidthAttr);
    dst = denseByDIntlv(dst, rewriter, loc, bitWidthAttr);
    break;
  }
  default:
    break;
  }
  return dst;
}

/// Add load dist by element alignment bitwidth
/// Change NORM to UNPK_B8/UNPK_B16/UNPK4_B8
struct AVELoadPattern : public OpRewritePattern<VFLoadOp> {
  explicit AVELoadPattern(MLIRContext *context)
      : OpRewritePattern<VFLoadOp>(context) {}
  LogicalResult matchAndRewrite(VFLoadOp load,
                                PatternRewriter &rewriter) const override {
    LDBG("process operation : " << load);
    VectorType vectorTy = load.getVectorType();
    auto vecElemTy = vectorTy.getElementType();
    auto elemWidth = vecElemTy.getIntOrFloatBitWidth();
    int elementAlignment = getElementAlignmentBitWidth(load);
    if (load.getPattern() == hivmave::LoadDist::NORM) {
      if (elemWidth == 8 && elementAlignment == 16) {
        load.setPattern(hivmave::LoadDist::UNPK_B8);
        LDBG("set load dist from NORM to UNPK_B8");
        return success();
      } else if (elemWidth == 16 && elementAlignment == 32) {
        load.setPattern(hivmave::LoadDist::UNPK_B16);
        LDBG("set load dist from NORM to UNPK_B16");
        return success();
      } else if (elemWidth == 8 && elementAlignment == 32) {
        load.setPattern(hivmave::LoadDist::UNPK4_B8);
        LDBG("set load dist from NORM to UNPK4_B8");
        return success();
      }
    }
    return failure();
  }
};

/// Add store dist by element alignment bitwidth
/// Change NORM to PK_B16/PK_B32/PK4_B32
struct AVEStorePattern : public OpRewritePattern<VFMaskedStoreOp> {
  explicit AVEStorePattern(MLIRContext *context)
      : OpRewritePattern<VFMaskedStoreOp>(context) {}
  LogicalResult matchAndRewrite(VFMaskedStoreOp store,
                                PatternRewriter &rewriter) const override {
    LDBG("process operation : " << store);
    VectorType vectorTy = store.getVectorType();
    auto vecElemTy = vectorTy.getElementType();
    auto elemWidth = vecElemTy.getIntOrFloatBitWidth();
    int elementAlignment = -1;
    if (auto valOp = store.getVal().getDefiningOp()) {
      elementAlignment = getElementAlignmentBitWidth(valOp);
    }
    if (elementAlignment == -1) {
      elementAlignment = getElementAlignmentBitWidth(store);
    }

    if (store.getPattern() == hivmave::StoreDist::NORM_B8 ||
        store.getPattern() == hivmave::StoreDist::NORM_B16) {
      if (elemWidth == 8 && elementAlignment == 16) {
        store.setPattern(hivmave::StoreDist::PK_B16);
        LDBG("set store dist from NORM to PK_B16");
        return success();
      } else if (elemWidth == 16 && elementAlignment == 32) {
        store.setPattern(hivmave::StoreDist::PK_B32);
        LDBG("set store dist from NORM to PK_B32");
        return success();
      } else if (elemWidth == 8 && elementAlignment == 32) {
        store.setPattern(hivmave::StoreDist::PK4_B32);
        LDBG("set store dist from NORM to PK4_B32");
        return success();
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
    auto bitWidthAttr = storeOp->getAttr(utils::elementAlignmentBitWidth);
    Value srcVal = storeOp.getVal();

    rewriter.setInsertionPoint(storeOp);
    Value result = srcVal;
    for (int i = 0; i < numDIntlv; ++i)
      result = denseByDIntlv(result, rewriter, loc, bitWidthAttr);

    storeOp->setOperand(storeOp->getNumOperands() - 1, result);
    storeOp->removeAttr("functionType");
    return success();
  }
};

/// Handle func_dist_type (INTLV2/INTLV4) on interleave/deinterleave ops by
/// inserting intlv ops after the op to sparsify the data layout.
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
    auto bitWidthAttr =
        intlvOp->getAttr(utils::elementAlignmentBitWidth);

    // Both res1 and res2 share the same layout state; sparse each of them.
    rewriter.setInsertionPointAfter(intlvOp);
    for (Value res : {intlvOp.getRes1(), intlvOp.getRes2()}) {
      SmallVector<Operation *> oldUsers(res.getUsers());
      if (oldUsers.empty())
        continue;

      Value result = res;
      for (int i = 0; i < numIntlv; ++i)
        result = sparseByIntlv(result, rewriter, loc, bitWidthAttr);

      for (Operation *user : oldUsers)
        user->replaceUsesOfWith(res, result);
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

static void adaptBitWidthForLoad(IRRewriter &rewriter,
                                 mlir::func::FuncOp &funcOp,
                                 bool archIs910_95) {
  funcOp->walk([&rewriter, archIs910_95](VFLoadOp load) {
    LDBG("process operation : " << load);
    hivmave::LoadDist dist = load.getPattern();
    Location loc = load->getLoc();
    auto dstVec = load.getRes();
    auto bitWidthAttr = load->getAttr(mlir::utils::elementAlignmentBitWidth);
    rewriter.setInsertionPointAfter(load);
    SmallVector<Operation *> oldUsers(dstVec.getUsers());
    // Add additional op to adpat dist mode for unaligned load
    if (archIs910_95 && load->hasAttr(UnalignedAttr::name)) {
      Value result =
          addDistForUnalignedLoad(dstVec, dist, loc, rewriter, bitWidthAttr);
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
      Value sparseVec1 = sparseByIntlv(dstVec, rewriter, loc, bitWidthAttr);
      Value sparseVec2 = sparseByIntlv(sparseVec1, rewriter, loc, bitWidthAttr);
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
    auto bitWidthAttr = store->getAttr(mlir::utils::elementAlignmentBitWidth);
    rewriter.setInsertionPoint(store);
    // Add additional op to adpat dist mode for unaligned store
    if (archIs910_95 && store->hasAttr(UnalignedAttr::name)) {
      Value result =
          addDistForUnalignedStore(srcVec, dist, loc, rewriter, bitWidthAttr);
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
      Value denseVec1 = denseByDIntlv(srcVec, rewriter, loc, bitWidthAttr);
      Value denseVec2 = denseByDIntlv(denseVec1, rewriter, loc, bitWidthAttr);
      // set new src vector of store
      store.setOperand(3, denseVec2);
      store.setPattern(hivmave::StoreDist::NORM_B8);
    }
    return WalkResult::advance();
  });
}

template <typename OpToBeConverted>
static void adaptBitWidthForTypeConversion(IRRewriter &rewriter,
                                           mlir::func::FuncOp &funcOp) {
  funcOp->walk([&rewriter](Operation *op) {
    if (!isa<OpToBeConverted>(op))
      return WalkResult::skip();
    LDBG("process operation : " << *op);
    OpToBeConverted convOp = dyn_cast<OpToBeConverted>(op);
    VectorType resType = cast<VectorType>(convOp.getResult().getType());
    VectorType srcType = cast<VectorType>(convOp.getSrc().getType());
    unsigned int outBitWidth = resType.getElementTypeBitWidth();
    unsigned int inBitWidt = srcType.getElementTypeBitWidth();
    bool alignByElementAlignment = isAlignByElementAlignment(convOp);
    LDBG("inBitWidt : " << inBitWidt);
    LDBG("outBitWidth : " << outBitWidth);
    LDBG("alignByElementAlignment : " << alignByElementAlignment);
    if (alignByElementAlignment)
      return WalkResult::skip();
    if (inBitWidt == outBitWidth)
      return WalkResult::skip();

    auto loc = convOp.getLoc();
    auto bitWidthAttr = convOp->getAttr(mlir::utils::elementAlignmentBitWidth);
    if (inBitWidt > outBitWidth) {
      // VFTruncF adapte for trunc + vsstb
      if constexpr (std::is_same_v<OpToBeConverted, VFTruncFOp>)
        if (resType.getNumElements() != srcType.getNumElements())
          return WalkResult::skip();
      rewriter.setInsertionPointAfter(convOp);
      auto resVec = convOp.getResult();
      SmallVector<Operation *> oldUsers(resVec.getUsers());
      Value denseVec = denseByDIntlv(resVec, rewriter, loc, bitWidthAttr);
      if (inBitWidt / outBitWidth == 4)
        denseVec = denseByDIntlv(denseVec, rewriter, loc, bitWidthAttr);
      for (Operation *u : oldUsers)
        u->replaceUsesOfWith(resVec, denseVec);
    } else {
      rewriter.setInsertionPoint(convOp);
      auto srcVec = convOp.getSrc();
      Value sparseVec = sparseByIntlv(srcVec, rewriter, loc, bitWidthAttr);
      if (outBitWidth / inBitWidt == 4)
        sparseVec = sparseByIntlv(sparseVec, rewriter, loc, bitWidthAttr);
      convOp.setOperand(0, sparseVec);
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

    adaptBitWidthForTypeConversion<hivmave::VFExtFOp>(rewriter, funcOp);
    adaptBitWidthForTypeConversion<hivmave::VFExtSIOp>(rewriter, funcOp);
    adaptBitWidthForTypeConversion<hivmave::VFExtUIOp>(rewriter, funcOp);
    adaptBitWidthForTypeConversion<hivmave::VFTruncIOp>(rewriter, funcOp);
    adaptBitWidthForTypeConversion<hivmave::VFTruncFOp>(rewriter, funcOp);
    adaptBitWidthForTypeConversion<hivmave::VFFpToSIntOp>(rewriter, funcOp);
    adaptBitWidthForTypeConversion<hivmave::VFFpToUIntOp>(rewriter, funcOp);
    adaptBitWidthForTypeConversion<hivmave::VFSIntToFpOp>(rewriter, funcOp);
    adaptBitWidthForTypeConversion<hivmave::VFUIntToFpOp>(rewriter, funcOp);
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

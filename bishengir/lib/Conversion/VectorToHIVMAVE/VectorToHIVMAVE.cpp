//===- VectorToHIVMAVE.cpp - convert vector op to ave op-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/VectorToHIVMAVE/VectorToHIVMAVE.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstddef>

#define DEBUG_TYPE "convert-vector-to-hivmave"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOHIVMAVE
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;
using namespace mlir::hivmave;

template <typename SourceOp, typename VFAlignedOp, typename... Args>
static FailureOr<Value> buildHIVMVFLdOrSTOp(SourceOp op,
                                            PatternRewriter &rewriter,
                                            Location loc, Args &&...args) {
  auto moduleOp = op->template getParentOfType<mlir::ModuleOp>();
  bool archIs950 = hacc::utils::isAscend950(moduleOp);
  if constexpr (std::is_same_v<SourceOp, vector::LoadOp> ||
                std::is_same_v<SourceOp, vector::MaskedLoadOp>) {
    mlir::Type resTy = op.getResult().getType();
    auto loadRes =
        rewriter.create<VFAlignedOp>(loc, resTy, std::forward<Args>(args)...);
    if (archIs950 && !isLoadStoreIndexAligned(op.getBase(), op.getIndices())) {
      loadRes->setAttr(UnalignedAttr::name,
                       UnalignedAttr::get(rewriter.getContext()));
    }
    return loadRes.getResult(0);
  } else if constexpr (std::is_same_v<SourceOp, vector::MaskedStoreOp> ||
                       std::is_same_v<SourceOp, vector::StoreOp>) {
    auto storeRes =
        rewriter.create<VFAlignedOp>(loc, std::forward<Args>(args)...);
    if (archIs950 && !isLoadStoreIndexAligned(op.getBase(), op.getIndices())) {
      storeRes->setAttr(UnalignedAttr::name,
                        UnalignedAttr::get(rewriter.getContext()));
    }
    return storeRes.getVal();
  }
  llvm_unreachable("unknown SourceOp");
}

// Convert Vector::LoadOp to hivm::VFLoadOp
struct LoadOpRewritePattern : public OpRewritePattern<vector::LoadOp> {
public:
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto base = op.getBase();
    auto indices = op.getIndices();

    FailureOr<Value> loadRes =
        buildHIVMVFLdOrSTOp<vector::LoadOp, hivmave::VFLoadOp>(
            op, rewriter, loc, base, indices);
    if (failed(loadRes)) {
      return rewriter.notifyMatchFailure(op, "failed to get legalized op");
    }
    rewriter.replaceOp(op, loadRes.value());

    return success();
  }
};

// Convert Vector StoreOp into Hivm dialect
struct MaskedStoreOpPattern : public OpRewritePattern<vector::MaskedStoreOp> {
  using OpRewritePattern<vector::MaskedStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto base = op.getBase();
    auto indices = op.getIndices();
    auto mask = op.getMask();
    auto val = op.getValueToStore();

    FailureOr<Value> storeRes =
        buildHIVMVFLdOrSTOp<vector::MaskedStoreOp, hivmave::VFMaskedStoreOp>(
            op, rewriter, loc, base, indices, mask, val);
    if (failed(storeRes)) {
      return rewriter.notifyMatchFailure(op, "failed to get legalized op");
    }
    rewriter.eraseOp(op);

    return success();
  }
};

struct MaskedLoadOpPattern : public OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern<vector::MaskedLoadOp>::OpRewritePattern;

  // Convert vector MaskedLoad into load + select.
  LogicalResult matchAndRewrite(vector::MaskedLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto base = op.getBase();
    auto indices = op.getIndices();
    auto mask = op.getMask();
    auto passThru = op.getPassThru();
    auto resType = op.getResult().getType();

    FailureOr<Value> loadRes =
        buildHIVMVFLdOrSTOp<vector::MaskedLoadOp, hivmave::VFLoadOp>(
            op, rewriter, loc, base, indices);
    if (failed(loadRes)) {
      return rewriter.notifyMatchFailure(op, "failed to get legalized op");
    }

    // Analyze whether the VFSelectOp could be optimized, according to the
    // annotation of this MaskedLoadOp op is same with all its users
    auto loadResult = op.getResult();
    bool maskReuseable = false;
    if (utils::getAnnotateOpWithAttr(loadResult, utils::reachedMaskOpsIdx)) {
      Value mask = hivmave::findReuseableMask(op, rewriter);
      if (mask && dyn_cast<VectorType>(mask.getType()).getShape().size() ==
                      resType.getShape().size()) {
        maskReuseable = true;
      }
    }
    if (maskReuseable) {
      // If the mask in MaskedLoadOp could be reused to one masked op or more,
      // then load all data without mask to reduce the select instruction
      rewriter.replaceOp(op, loadRes.value());
    } else {
      auto selectOp = rewriter.create<hivmave::VFSelectOp>(
          loc, resType, mask, loadRes.value(), passThru);
      rewriter.replaceOp(op, selectOp.getResult());
    }

    return success();
  }
};

struct StoreOpPattern : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto base = op.getBase();
    auto indices = op.getIndices();
    auto val = op.getValueToStore();

    if (!mlir::isa<VectorType>(val.getType())) {
      emitError(loc, "value to store must be a vector");
      return failure();
    }

    // assert the mask is a 1-D vector.
    if (val.getType().getShape().size() != 1) {
      emitError(loc, "mask should be a 1-D vector.");
      return failure();
    }

    // TODO: Support more pattern.
    auto shapeSize = mlir::cast<VectorType>(val.getType()).getShape()[0];
    auto pgePatternAttr = getPgePatternAttr(rewriter, shapeSize, shapeSize);
    // TODO: Maybe use PLT?
    if (failed(pgePatternAttr)) {
      return rewriter.notifyMatchFailure(op,
                                         "failed  to legalize vector::StoreOp");
    }

    auto resultMaskType =
        VectorType::get(SmallVector<int64_t>{val.getType().getShape()[0]},
                        rewriter.getI1Type());

    auto pgeOp = rewriter.create<hivmave::VFPgeOp>(loc, resultMaskType,
                                                   pgePatternAttr.value());

    FailureOr<Value> storeRes =
        buildHIVMVFLdOrSTOp<vector::StoreOp, hivmave::VFMaskedStoreOp>(
            op, rewriter, loc, base, indices, pgeOp, val);
    if (failed(storeRes)) {
      return rewriter.notifyMatchFailure(op, "failed to get legalized op");
    }
    rewriter.eraseOp(op);

    return success();
  }
};

struct TransferReadOpPattern : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    auto loc = readOp.getLoc();
    auto source = readOp.getSource();
    MemRefType srcType = cast<MemRefType>(source.getType());
    Type elementType = srcType.getElementType();
    int64_t dataWidth = srcType.getElementTypeBitWidth();
    VectorType vecType = readOp.getVectorType();
    VectorType resVecType = vecType;
    if (vecType.getShape().size() != 1)
      resVecType = VectorType::get(
          SmallVector<int64_t>{vecType.getNumElements()}, elementType);

    hivmave::VFLoadOp loadOp = rewriter.create<hivmave::VFLoadOp>(
        loc, resVecType, source, readOp.getIndices());

    auto moduleOp = readOp->getParentOfType<mlir::ModuleOp>();
    bool archIs950 = hacc::utils::isAscend950(moduleOp);
    if (archIs950 &&
        !isLoadStoreIndexAligned(readOp.getSource(), readOp.getIndices())) {
      loadOp->setAttr(UnalignedAttr::name,
                      UnalignedAttr::get(rewriter.getContext()));
    }

    if (vecType.getNumElements() == 1 ||
        (srcType.hasStaticShape() && srcType.getNumElements() == 1)) {
      if (dataWidth == 8)
        loadOp.setPattern(hivmave::LoadDist::BRC_B8);
      else if (dataWidth == 16)
        loadOp.setPattern(hivmave::LoadDist::BRC_B16);
      else if (dataWidth == 32)
        loadOp.setPattern(hivmave::LoadDist::BRC_B32);
      else if (dataWidth == 64)
        loadOp.setPattern(hivmave::LoadDist::BRC_B64);
    }
    if (resVecType != vecType) {
      Operation *op = rewriter.create<vector::ShapeCastOp>(loc, vecType,
                                                           loadOp.getResult(0));
      rewriter.replaceOp(readOp, op);
      return success();
    }
    rewriter.replaceOp(readOp, loadOp);
    return success();
  }
};

struct TransferWriteOpPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  bool transferWriteWithStride(vector::TransferWriteOp writeOp,
                               PatternRewriter &rewriter, Value mask,
                               Value vector, Value source) const {
    if (!utils::isTransferWriteSuitForStoreWithStride(writeOp)) {
      return false;
    }
    auto memrefTy = mlir::dyn_cast<MemRefType>(writeOp.getSource().getType());
    auto [strides, offset] = getStridesAndOffset(memrefTy);
    auto loc = writeOp.getLoc();
    Value strideConst = rewriter.create<arith::ConstantOp>(
        writeOp.getLoc(), rewriter.getIndexAttr(strides[0]));
    hivmave::VFStoreWithStrideOp storeOp =
        rewriter.create<hivmave::VFStoreWithStrideOp>(
            loc, source, writeOp.getIndices(), strideConst, mask, vector);
    LLVM_DEBUG(DBGS() << "transfer write ave op: " << storeOp << "\n");
    rewriter.replaceOp(writeOp, storeOp);
    return true;
  }

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = writeOp.getLoc();
    VectorType vecType = writeOp.getVectorType();
    Value source = writeOp.getSource();
    MemRefType srcType = cast<MemRefType>(source.getType());
    Type elementType = vecType.getElementType();
    int64_t dataWidth = vecType.getElementTypeBitWidth();
    Value vector = writeOp.getVector();
    VectorType castVecType = vecType;
    if (vecType.getShape().size() != 1) {
      castVecType = VectorType::get(
          SmallVector<int64_t>{vecType.getNumElements()}, elementType);
      vector = rewriter.create<vector::ShapeCastOp>(loc, castVecType, vector)
                   ->getResult(0);
    }
    Value mask = writeOp.getMask();
    bool isSingleElementMask = false;
    if (!mask) {
      mask = hivmave::createMaskByPGE(castVecType, rewriter, loc);
    } else {
      Value unwrapped = mask;
      while (true) {
        Operation *def = unwrapped.getDefiningOp();
        if (!def)
          break;
        if (auto sc = dyn_cast<vector::ShapeCastOp>(def)) {
          unwrapped = sc.getSource();
          continue;
        }
        if (def->getName().getStringRef() == "annotation.mark" &&
            def->getNumOperands() == 1) {
          unwrapped = def->getOperand(0);
          continue;
        }
        break;
      }
      if (auto constMaskOp =
              unwrapped.getDefiningOp<vector::ConstantMaskOp>()) {
        auto maskDimSizes = constMaskOp.getMaskDimSizes();
        if (maskDimSizes.size() == 1 &&
            mlir::cast<IntegerAttr>(maskDimSizes[0]).getInt() == 1) {
          isSingleElementMask = true;
        }
      } else if (auto pgeOp = unwrapped.getDefiningOp<hivmave::VFPgeOp>()) {
        auto pattern = pgeOp.getPattern();
        if (pattern == hivmave::PgePattern::VL1) {
          isSingleElementMask = true;
        }
      }
      auto maskVecType = cast<VectorType>(mask.getType());
      if (maskVecType.getShape().size() != 1) {
        int64_t bound = maskVecType.getShape().back();
        VectorType newMaskVecType =
            VectorType::get({bound}, rewriter.getI1Type());
        mask = rewriter
                   .create<vector::ShapeCastOp>(loc, newMaskVecType, unwrapped)
                   ->getResult(0);
      }
    }
    hivmave::StoreDist storePattern = hivmave::StoreDist::NORM_B8;
    if (isSingleElementMask) {
      if (dataWidth == 8)
        storePattern = hivmave::StoreDist::ONEPT_B8;
      else if (dataWidth == 16)
        storePattern = hivmave::StoreDist::ONEPT_B16;
      else if (dataWidth == 32)
        storePattern = hivmave::StoreDist::ONEPT_B32;
      else if (dataWidth == 64)
        storePattern = hivmave::StoreDist::ONEPT_B64;
    } else if (dataWidth == 1) {
      storePattern = hivmave::StoreDist::NORM_B8;
    } else if (vecType.getNumElements() == 1 ||
               (srcType.hasStaticShape() && srcType.getNumElements() == 1)) {
      if (dataWidth == 8)
        storePattern = hivmave::StoreDist::ONEPT_B8;
      else if (dataWidth == 16)
        storePattern = hivmave::StoreDist::ONEPT_B16;
      else if (dataWidth == 32)
        storePattern = hivmave::StoreDist::ONEPT_B32;
      else if (dataWidth == 64)
        storePattern = hivmave::StoreDist::ONEPT_B64;
    } else {
      if (dataWidth == 16)
        storePattern = hivmave::StoreDist::NORM_B16;
      else if (dataWidth == 32)
        storePattern = hivmave::StoreDist::NORM_B32;
      else if (dataWidth == 64)
        storePattern = hivmave::StoreDist::NORM_B64;
    }
    if (transferWriteWithStride(writeOp, rewriter, mask, vector, source)) {
      return success();
    }
    hivmave::VFMaskedStoreOp storeOp =
        rewriter.create<hivmave::VFMaskedStoreOp>(
            loc, storePattern, source, writeOp.getIndices(), mask, vector);
    auto moduleOp = writeOp->getParentOfType<mlir::ModuleOp>();
    bool archIs950 = hacc::utils::isAscend950(moduleOp);
    if (archIs950 &&
        !isLoadStoreIndexAligned(writeOp.getSource(), writeOp.getIndices())) {
      storeOp->setAttr(UnalignedAttr::name,
                       UnalignedAttr::get(rewriter.getContext()));
    }
    if (archIs950 && checkVectorStoreContinuity(writeOp)) {
      storeOp->setAttr("hivm.is_continuous", rewriter.getUnitAttr());
    }
    rewriter.replaceOp(writeOp, storeOp);
    return success();
  }
};

struct BroadcastOpPattern : public OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    // Only a scalar value can be broadcast to a vector whose length is
    // util::VL_BITS.
    if (!isa<IntegerType, FloatType>(op.getSourceType()))
      return failure();
    VectorType vecType = op.getResultVectorType();

    auto originalShape = vecType.getShape();
    auto dimsGreaterThanOne =
        llvm::count_if(originalShape, [](int64_t dim) { return dim > 1; });
    if (dimsGreaterThanOne > 1) {
      emitError(op.getLoc(), "at most one dimension can greater than 1.");
      return failure();
    }

    // get new reshape type
    VectorType reshapedVecType =
        VectorType::get({vecType.getNumElements()}, vecType.getElementType());
    if (originalShape.empty()) {
      reshapedVecType = vecType;
    }
    Value src = op.getSource();
    if (auto srcExtractOp =
            dyn_cast_or_null<vector::ExtractOp>(src.getDefiningOp())) {
      // BEFORE:
      // ```mlir
      // %0 = vector.extract %src[0] : f32 from vector<64xf32>
      // %1 = vector.broadcast %0 : f32 to vector<64xf32>
      // ```
      // AFTER:
      // ```mlir
      // %mask = ave.hir.pge <ALL> : vector<64xi1>
      // %0 = ave.hir.vector_broadcast %src, %mask, true :
      //      vector<64xf32>, vector<64xi1> -> vector<64xf32>
      // ```
      if (llvm::all_of(srcExtractOp.getStaticPosition(),
                       [](int64_t pos) { return pos == 0; })) {
        Value mask =
            hivmave::createMaskByPGE(reshapedVecType, rewriter, op.getLoc());
        auto vecBrc = rewriter.create<hivmave::VFBroadcastVectorOp>(
            op.getLoc(), reshapedVecType, srcExtractOp.getVector(), mask,
            rewriter.getBoolAttr(true));
        auto shapeCastOp = rewriter.create<vector::ShapeCastOp>(
            op.getLoc(), vecType, vecBrc.getResult());
        rewriter.replaceOp(op, shapeCastOp);
        return success();
      }
    }
    auto broadcastOp = rewriter.create<hivmave::VFBroadcastScalarOp>(
        op.getLoc(), reshapedVecType, src);
    auto shapeCastOp = rewriter.create<vector::ShapeCastOp>(
        op.getLoc(), vecType, broadcastOp.getResult());

    rewriter.replaceOp(op, shapeCastOp);
    return success();
  }
};

struct VecBroadcastOpPattern : public OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<VectorType>(op.getSourceType()))
      return failure();

    VectorType srcType = cast<VectorType>(op.getSourceType());
    VectorType dstType = op.getResultVectorType();
    Value dupSrc = op.getSource();
    // only broadcast vector that contains one element
    if (srcType.getNumElements() != 1) {
      return failure();
    }
    Location loc = op->getLoc();
    VectorType dupDstTy =
        VectorType::get({dstType.getShape().back()}, srcType.getElementType());

    if (srcType.getRank() > 1) {
      VectorType dupSrcTy = VectorType::get({1}, srcType.getElementType());
      dupSrc = rewriter.create<vector::ShapeCastOp>(loc, dupSrcTy, dupSrc);
    }

    if (dupDstTy != dupSrc.getType()) {
      dupSrc = rewriter.create<vector::ShapeCastOp>(loc, dupDstTy, dupSrc);
    }

    Value mask = hivmave::findReuseableMaskOrCreateOne(op, dstType, rewriter);
    // TODO: highest element
    Value vecBrc = rewriter.create<hivmave::VFBroadcastVectorOp>(
        op.getLoc(), dupDstTy, dupSrc, mask, rewriter.getBoolAttr(true));
    auto vecBrcShapeCast =
        rewriter.create<vector::ShapeCastOp>(loc, dstType, vecBrc);

    rewriter.replaceOp(op, vecBrcShapeCast);
    return success();
  }
};

struct VecGatherOpPattern : public OpRewritePattern<vector::GatherOp> {
  using OpRewritePattern<vector::GatherOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto base = op.getBase();
    auto indices = op.getIndices();
    Value indexVec = op.getIndexVec();
    Value mask = op.getMask();
    auto loc = op.getLoc();
    VectorType resultType = op.getType();
    Type resultElementType = resultType.getElementType();

    if (!mask) {
      // If the TransferWrite op does not have a mask, create one
      mask = hivmave::createMaskByPGE(resultType, rewriter, loc);
    } else {
      if (utils::getAnnotateOpWithAttr(mask, utils::maskOpIdx)) {
        mask = hivmave::findReuseableMaskOrCreateOne(op, resultType, rewriter);
      }
    }

    // handle passthrough; if it is constant zero, do nothing
    // otherwise, we need a select between gathered value and passthrough
    auto passthrough = op.getPassThru();
    bool isRequireSelect = true;
    if (auto constValue =
            dyn_cast<arith::ConstantOp>(passthrough.getDefiningOp())) {
      auto valueAttr = cast<DenseElementsAttr>(constValue.getValue());
      if (valueAttr.isSplat()) {
        if (isa<FloatType>(resultElementType)) {
          if (valueAttr.getSplatValue<APFloat>().isZero()) {
            isRequireSelect = false;
          }
        } else if (isa<IntegerType>(resultElementType)) {
          if (valueAttr.getSplatValue<APInt>().isZero()) {
            isRequireSelect = false;
          }
        }
      }
    }
    if (isRequireSelect) {
      // TODO
      return failure();
    }
    auto secondaryIndexAttr =
        op->getAttrOfType<DenseElementsAttr>("secondary_index");
    Value result;
    if (secondaryIndexAttr) {
      result = createDualGather(rewriter, loc, resultType, base, indices,
                                indexVec, mask, secondaryIndexAttr);
    } else {
      auto gather = rewriter.create<hivmave::VFGatherOp>(
          loc, resultType, base, indices, indexVec, mask);
      result = gather.getResult();
    }
    // TODO handle passthrough requiring a select
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static Value createDualGather(PatternRewriter &rewriter, Location loc,
                                VectorType resultType, Value base,
                                ValueRange indices, Value indexVec, Value mask,
                                DenseElementsAttr secondaryIndexAttr) {
    auto secondaryIndexConstant =
        rewriter.create<arith::ConstantOp>(loc, secondaryIndexAttr);
    Value secondaryIndexVec = secondaryIndexConstant.getResult();

    auto gather1 = rewriter.create<hivmave::VFGatherOp>(
        loc, resultType, base, indices, indexVec, mask);
    auto gather2 = rewriter.create<hivmave::VFGatherOp>(
        loc, resultType, base, indices, secondaryIndexVec, mask);
    Value gather1Result = gather1.getResult();
    Value gather2Result = gather2.getResult();

    auto resDtype = resultType.getElementType();
    VectorType vecTy = VectorType::get({256}, resDtype);
    auto i8Type = rewriter.getI8Type();
    VectorType i8VecTy = VectorType::get({256}, i8Type);
    Value cstZero = rewriter.create<arith::ConstantOp>(
        loc, i8Type, rewriter.getIntegerAttr(i8Type, 0));
    Value cstZeroVec =
        rewriter.create<hivmave::VFBroadcastScalarOp>(loc, i8VecTy, cstZero);
    if (resDtype.isFloat8E4M3FN() || resDtype.isFloat8E5M2()) {
      cstZeroVec = rewriter.create<arith::BitcastOp>(loc, vecTy, cstZeroVec);
    }
    auto resEven = rewriter.create<hivmave::VFDeInterleaveOp>(
        loc, vecTy, vecTy, gather1Result, cstZeroVec);
    auto resOdd = rewriter.create<hivmave::VFDeInterleaveOp>(
        loc, vecTy, vecTy, gather2Result, cstZeroVec);

    auto res = rewriter.create<hivmave::VFInterleaveOp>(
        loc, vecTy, vecTy, resEven.getResult(0), resOdd.getResult(0));
    Value finalResult = res.getResult(0);

    return finalResult;
  }
};

struct VecScatterOpPattern : public OpRewritePattern<vector::ScatterOp> {
  using OpRewritePattern<vector::ScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    // TODO
    return failure();
  }
};

template <class HIVMReductionOp, class VectorAccOp>
static Value createReductionArithmeticOpLowering(
    PatternRewriter &rewriter, Location loc, hivmave::CombiningKind kind,
    Value v1, Value accmulator, Value mask, bool enableTritonKernelCompile,
    bool withoutInitMergeOp) {
  Value result = rewriter.create<HIVMReductionOp>(
      loc, v1.getType(),
      hivmave::CombiningKindAttr::get(rewriter.getContext(), kind), v1, mask);
  if (accmulator && withoutInitMergeOp && enableTritonKernelCompile)
    return result;
  if (accmulator)
    result =
        rewriter.create<VectorAccOp>(loc, v1.getType(), accmulator, result);
  return result;
}

static Value makeHivmReduction(PatternRewriter &b, Location loc,
                               vector::CombiningKind kind, Value v1, Value acc,
                               Value mask, Type vecEltType,
                               bool enableTritonKernelCompile,
                               bool withoutInitMergeOp) {
  Value result;
  if (kind == vector::CombiningKind::ADD) {
    if (mlir::isa<FloatType>(vecEltType))
      result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                   arith::AddFOp>(
          b, loc, hivmave::CombiningKind::ADD, v1, acc, mask,
          enableTritonKernelCompile, withoutInitMergeOp);
    else
      result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                   arith::AddIOp>(
          b, loc, hivmave::CombiningKind::ADD, v1, acc, mask,
          enableTritonKernelCompile, withoutInitMergeOp);
  } else if (kind == vector::CombiningKind::MAXSI)
    result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                 arith::MaxSIOp>(
        b, loc, hivmave::CombiningKind::MAX, v1, acc, mask,
        enableTritonKernelCompile, withoutInitMergeOp);
  else if (kind == vector::CombiningKind::MAXUI)
    result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                 arith::MaxUIOp>(
        b, loc, hivmave::CombiningKind::UMAX, v1, acc, mask,
        enableTritonKernelCompile, withoutInitMergeOp);
  else if (kind == vector::CombiningKind::MINSI)
    result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                 arith::MinSIOp>(
        b, loc, hivmave::CombiningKind::MIN, v1, acc, mask,
        enableTritonKernelCompile, withoutInitMergeOp);
  else if (kind == vector::CombiningKind::MINUI)
    result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                 arith::MinUIOp>(
        b, loc, hivmave::CombiningKind::UMIN, v1, acc, mask,
        enableTritonKernelCompile, withoutInitMergeOp);
  else if (kind == vector::CombiningKind::MAXNUMF)
    result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                 arith::MaxNumFOp>(
        b, loc, hivmave::CombiningKind::MAX, v1, acc, mask,
        enableTritonKernelCompile, withoutInitMergeOp);
  else if (kind == vector::CombiningKind::MAXIMUMF)
    result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                 arith::MaximumFOp>(
        b, loc, hivmave::CombiningKind::MAX, v1, acc, mask,
        enableTritonKernelCompile, withoutInitMergeOp);
  else if (kind == vector::CombiningKind::MINNUMF)
    result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                 arith::MinNumFOp>(
        b, loc, hivmave::CombiningKind::MIN, v1, acc, mask,
        enableTritonKernelCompile, withoutInitMergeOp);
  else if (kind == vector::CombiningKind::MINIMUMF)
    result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                 arith::MinimumFOp>(
        b, loc, hivmave::CombiningKind::MIN, v1, acc, mask,
        enableTritonKernelCompile, withoutInitMergeOp);
  else if (kind == vector::CombiningKind::XOR)
    result = createReductionArithmeticOpLowering<hivmave::ReductionOp,
                                                 arith::XOrIOp>(
        b, loc, hivmave::CombiningKind::XORI, v1, acc, mask,
        enableTritonKernelCompile, withoutInitMergeOp);
  else
    (void)emitOptionalError(loc, "Reduction operation type not supported");
  assert(result && "unknown CombiningKind");
  return result;
}

/// Conversion from vector.reduction to ave.reduction + arith.add/max/min
/// clang-format off
/// Before conversion:
/// ```mlir
///   %acc_scalar = builtin.unrealized_conversion_cast %acc : vector<1xf32> to
///   f32 %reduction = vector.mask %mask { vector.reduction <add>, %v,
///   %acc_scalar
///     : vector<64xf32> into f32 } : vector<64xi1> -> f32
///   %cast = builtin.unrealized_conversion_cast %reduction : f32 to
///   vector<1xf32>
/// ```
/// After conversion:
/// ```mlir
///   %acc_scalar = builtin.unrealized_conversion_cast %acc : vector<1xf32> to
///   f32 %acc_0 = builtin.unrealized_conversion_cast %acc_scalar : f32 to
///   vector<1xf32> %acc_1 = builtin.unrealized_conversion_cast %acc_0 :
///   vector<1xf32> to vector<64xf32> %reduction = ave.hir.reduction <add>, %v,
///   %mask : vector<64xf32>, vector<64xi1> -> vector<64xf32> %result =
///   arith.addf %reduction, %acc_1 : vector<64xf32> %cast_0 =
///   builtin.unrealized_conversion_cast %result : vector<64xf32> to
///   vector<1xf32> %cast_1 = builtin.unrealized_conversion_cast %cast_0 :
///   vector<1xf32> to f32 %cast = builtin.unrealized_conversion_cast %reduction
///   : f32 to vector<1xf32>
/// ```
/// clang-format on
struct ReductionOpPattern : public OpRewritePattern<vector::ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  explicit ReductionOpPattern(mlir::MLIRContext *ctx,
                              bool enableTritonKernelCompile = false)
      : OpRewritePattern<vector::ReductionOp>(ctx),
        enableTritonKernelCompile(enableTritonKernelCompile) {}

  LogicalResult matchAndRewrite(vector::ReductionOp reductionOp,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    auto maskableOp =
        cast<vector::MaskableOpInterface>(reductionOp.getOperation());
    Operation *rootOp;
    Value mask;
    if (maskableOp.isMasked()) {
      rewriter.setInsertionPoint(maskableOp.getMaskingOp());
      rootOp = maskableOp.getMaskingOp();
      mask = maskableOp.getMaskingOp().getMask();
    } else
      rootOp = reductionOp;
    auto srcVectorType =
        dyn_cast<VectorType>(reductionOp.getSourceVectorType());
    if (!srcVectorType || srcVectorType.getRank() != 1) {
      return rewriter.notifyMatchFailure(reductionOp,
                                         "not a 1-D vector source");
    }
    Location loc = reductionOp.getLoc();
    Value acc = reductionOp.getAcc();
    Value result;
    if (!mask) {
      // gen an all-true constant mask.
      SmallVector<int64_t> maskShape(srcVectorType.getShape());
      auto maskType = VectorType::get(maskShape, rewriter.getI1Type());
      mask = rewriter.create<vector::ConstantMaskOp>(
          loc, maskType, rewriter.getI64ArrayAttr(maskShape));
    }
    auto elementType = acc.getType();
    auto accVecType = VectorType::get({1}, elementType);
    if (acc) {
      auto acc0 =
          rewriter.create<UnrealizedConversionCastOp>(loc, accVecType, acc)
              .getResult(0);
      acc =
          rewriter.create<UnrealizedConversionCastOp>(loc, srcVectorType, acc0)
              .getResult(0);
    }
    result = makeHivmReduction(
        rewriter, loc, reductionOp.getKind(), reductionOp.getVector(), acc,
        mask, srcVectorType.getElementType(), enableTritonKernelCompile,
        reductionOp->hasAttr("withoutInitMergeOp"));
    auto acc0 =
        rewriter.create<UnrealizedConversionCastOp>(loc, accVecType, result)
            .getResult(0);
    result = rewriter.create<UnrealizedConversionCastOp>(loc, elementType, acc0)
                 .getResult(0);
    rewriter.replaceOp(rootOp, result);
    return success();
  }

private:
  bool enableTritonKernelCompile{false};
};

struct ConstantMaskOpConversionPattern
    : public OpRewritePattern<vector::ConstantMaskOp> {
  using OpRewritePattern<vector::ConstantMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ConstantMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto origMaskType = cast<VectorType>(op.getResult().getType());
    if (origMaskType.isScalable())
      return emitError(loc, "Do not support the mask");
    // we only support 1D vectors, if there is more than one
    // dimension, the leading dimension must be constant 1
    auto maskDimSizes = op.getMaskDimSizes();
    unsigned int rank = maskDimSizes.size();
    for (size_t dim = 0; dim < rank - 1; ++dim) {
      if (mlir::cast<IntegerAttr>(maskDimSizes[dim]).getInt() != 1)
        return emitError(loc, "Do not support the mask");
    }
    int origMaskNumElements = origMaskType.getNumElements();
    VectorType maskType = VectorType::get(
        SmallVector<int64_t>{origMaskNumElements}, rewriter.getI1Type());
    int boundCst = mlir::cast<IntegerAttr>(maskDimSizes[rank - 1]).getInt();
    auto pgePatternAttr =
        getPgePatternAttr(rewriter, boundCst, origMaskNumElements);
    IntegerAttr maskOpIdxAttr = nullptr;
    if (utils::getAnnotateOpWithAttr(op.getResult(), utils::maskOpIdx)) {
      annotation::MarkOp mark = dyn_cast<annotation::MarkOp>(
          utils::getAnnotateOpWithAttr(op.getResult(), utils::maskOpIdx)
              .value());
      maskOpIdxAttr =
          mark->template getAttrOfType<IntegerAttr>(utils::maskOpIdx);
    }
    if (!failed(pgePatternAttr)) {
      VFPgeOp pge = rewriter.create<hivmave::VFPgeOp>(loc, maskType,
                                                      pgePatternAttr.value());
      if (maskOpIdxAttr)
        pge->setAttr(utils::maskOpIdx, maskOpIdxAttr);
      op.replaceAllUsesWith(pge.getResult());
      rewriter.replaceOp(op, pge);
    } else if (boundCst < 256) {
      auto trueShape = rewriter.create<arith::ConstantIndexOp>(loc, boundCst);
      VFPltOp plt = rewriter.create<hivmave::VFPltOp>(
          loc, maskType, rewriter.getIndexType(), trueShape);
      if (maskOpIdxAttr)
        plt->setAttr(utils::maskOpIdx, maskOpIdxAttr);
      rewriter.replaceOp(op, plt.getResults()[0]);
    } else {
      return emitError(loc, "Do not support the mask");
    }

    return success();
  }
};

struct CreateMaskOpConversionPattern
    : public OpRewritePattern<vector::CreateMaskOp> {
  using OpRewritePattern<vector::CreateMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::CreateMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto maskDimSizes = op.getOperands();
    auto origMaskType = op.getResult().getType();

    // assert the mask is a 1-D vector.
    if (maskDimSizes.size() != 1) {
      return emitError(loc, "mask should be a 1-D vector.");
    }

    auto resultMaskType = VectorType::get(
        SmallVector<int64_t>{origMaskType.getShape()[0]}, rewriter.getI1Type());
    VFPltOp newOp = rewriter.create<hivmave::VFPltOp>(
        loc, resultMaskType, rewriter.getIndexType(), maskDimSizes[0]);
    if (utils::getAnnotateOpWithAttr(op.getResult(), utils::maskOpIdx)) {
      annotation::MarkOp mark = dyn_cast<annotation::MarkOp>(
          utils::getAnnotateOpWithAttr(op.getResult(), utils::maskOpIdx)
              .value());
      newOp->setAttr(
          utils::maskOpIdx,
          mark->template getAttrOfType<IntegerAttr>(utils::maskOpIdx));
    }
    rewriter.replaceOp(op, newOp.getResults()[0]);

    return success();
  }
};

/// BEFORE:
///   %3 = vector.interleave %1, %2 : vector<64xf32>, vector<64xf32> ->
///   vector<128xf32>
/// AFTER:
///   %31, %32 = ave.vintlv %1, %2
///   %3 = UCC %31 vector<128xf32> -> vector<64xf32>
struct VecInterleaveOpPattern : public OpRewritePattern<vector::InterleaveOp> {
  using OpRewritePattern<vector::InterleaveOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::InterleaveOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<VectorType>(op.getSourceVectorType()))
      return failure();

    VectorType srcType = cast<VectorType>(op.getSourceVectorType());
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Location loc = op->getLoc();

    auto aveIntlvOp = rewriter.create<hivmave::VFInterleaveOp>(
        loc, srcType, srcType, lhs, rhs);
    auto aveRes1 = aveIntlvOp.getRes1();

    auto vecRes = op.getResult();
    auto vecResTy = op.getResultVectorType();
    auto ucc =
        rewriter.create<UnrealizedConversionCastOp>(loc, vecResTy, aveRes1);
    aveRes1 = ucc->getResult(0);
    rewriter.replaceAllUsesWith(vecRes, aveRes1);
    rewriter.eraseOp(op);

    return success();
  }
};

// clang-format off
/// BEFORE:
///   %1 = builtin.unrealized_conversion_cast %low, %high : vector<64xf32>, vector<64xf32> to vector<128xf32>
///   %2, %3 = vector.deinterleave %1 : vector<128xf32> -> vector<64xf32>
/// AFTER:
///   %1 = builtin.unrealized_conversion_cast %low, %high : vector<64xf32>, vector<64xf32> to vector<128xf32>
///   %1_low, %1_high = builtin.unrealized_conversion_cast %1 : vector<128xf32> to vector<64xf32>, vector<64xf32>
///   %31, %32 = ave.vdeintlv %1_low, %2_high
// clang-format on
struct VecDeinterleaveOpPattern
    : public OpRewritePattern<vector::DeinterleaveOp> {
  using OpRewritePattern<vector::DeinterleaveOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::DeinterleaveOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto vecResTy = op.getResultVectorType();
    Value src = op.getSource();
    auto ucc = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{vecResTy, vecResTy}, src);

    Value lhs = ucc->getResult(0);
    Value rhs = ucc->getResult(1);
    auto aveDeintlvOp = rewriter.create<hivmave::VFDeInterleaveOp>(
        loc, vecResTy, vecResTy, lhs, rhs);
    auto aveRes1 = aveDeintlvOp.getRes1();
    auto aveRes2 = aveDeintlvOp.getRes2();

    auto vecRes1 = op.getRes1();
    auto vecRes2 = op.getRes2();

    rewriter.replaceAllUsesWith(vecRes1, aveRes1);
    rewriter.replaceAllUsesWith(vecRes2, aveRes2);
    rewriter.eraseOp(op);

    return success();
  }
};

struct ShapeCastOpPattern : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Operation *ucc = rewriter.create<UnrealizedConversionCastOp>(
        loc, op->getResult(0).getType(), op->getOperand(0));
    rewriter.replaceOp(op, ucc);
    return success();
  }
};

namespace {
struct VectorToHIVMAVEPass
    : public impl::ConvertVectorToHIVMAVEBase<VectorToHIVMAVEPass> {
  using Base::Base;
  void runOnOperation() override {
    ConversionTarget target(getContext());
    ModuleOp moduleOp = cast<ModuleOp>(getOperation());
    if (moduleOp) {
      tagConstantArguments(moduleOp);
    }
    target.addLegalDialect<hivmave::AVEDialect, arith::ArithDialect,
                           BuiltinDialect, annotation::AnnotationDialect>();
    target.addIllegalOp<vector::LoadOp, vector::StoreOp, vector::MaskedStoreOp,
                        vector::MaskedLoadOp, vector::CreateMaskOp,
                        vector::ConstantMaskOp, vector::BroadcastOp,
                        vector::TransferReadOp, vector::TransferWriteOp,
                        vector::GatherOp, vector::ScatterOp,
                        vector::ShapeCastOp, vector::ReductionOp>();
    target.addLegalOp<arith::ConstantOp>();

    moduleOp->walk([&](func::FuncOp funcOp) {
      {
        RewritePatternSet patterns(&getContext());
        patterns.add<ReductionOpPattern>(patterns.getContext(),
                                         enableTritonCompile);
        if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
          signalPassFailure();
        }
      }
      RewritePatternSet patterns(&getContext());
      hivmave::populateVectorToHIVMAVEConversionPatterns(patterns);
      if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
        signalPassFailure();
      }
    });
  }
};
} // namespace

void mlir::hivmave::populateVectorToHIVMAVEConversionPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<MaskedLoadOpPattern, LoadOpRewritePattern, StoreOpPattern,
           MaskedStoreOpPattern, ConstantMaskOpConversionPattern,
           CreateMaskOpConversionPattern, BroadcastOpPattern, VecInterleaveOpPattern,
           VecInterleaveOpPattern, VecDeinterleaveOpPattern,
           TransferReadOpPattern, TransferWriteOpPattern, VecGatherOpPattern,
           VecScatterOpPattern, VecBroadcastOpPattern, ShapeCastOpPattern>(
          patterns.getContext());
}

std::unique_ptr<Pass> mlir::createVectorToHIVMAVEConversionPass(
    const ConvertVectorToHIVMAVEOptions &options) {
  return std::make_unique<VectorToHIVMAVEPass>(options);
}

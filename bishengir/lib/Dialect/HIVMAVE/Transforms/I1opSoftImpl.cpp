//===------------- I1opSoftImpl.cpp - soft impl i1 op ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "ave-i1op-soft-impl"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_I1OPSOFTIMPL
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivmave;

static constexpr llvm::StringLiteral i1ProcessedAttr = "1xi1 processed";

namespace {
struct LinearizedI1Access {
  Value rootMemref;
  Value linearBitOffset;
};
} // namespace

static SmallVector<int64_t> getStaticStrides(MemRefType memRefTy) {
  SmallVector<int64_t> strides(memRefTy.getRank());
  int64_t offset = 0;
  if (succeeded(getStridesAndOffset(memRefTy, strides, offset)))
    return strides;

  int64_t runningStride = 1;
  for (int64_t i = memRefTy.getRank() - 1; i >= 0; --i) {
    strides[i] = runningStride;
    runningStride *= memRefTy.getDimSize(i);
  }
  return strides;
}

static FailureOr<LinearizedI1Access>
linearizeSingleElementI1Access(PatternRewriter &rewriter, Location loc,
                               Value baseMemref, ValueRange indices) {
  auto memRefTy = dyn_cast<MemRefType>(baseMemref.getType());
  if (!memRefTy || !memRefTy.hasStaticShape() ||
      indices.size() != static_cast<size_t>(memRefTy.getRank()))
    return failure();

  Operation *defOp = baseMemref.getDefiningOp();
  if (!defOp) {
    return LinearizedI1Access{baseMemref,
                              computeLinearMemRefOffset(
                                  rewriter, loc, baseMemref, indices,
                                  rewriter.getIndexType())};
  }

  if (auto subViewOp = dyn_cast<memref::SubViewOp>(defOp)) {
    SmallVector<Value> sourceIndices;
    sourceIndices.reserve(subViewOp.getSourceType().getRank());

    auto mixedOffsets = subViewOp.getMixedOffsets();
    auto mixedStrides = subViewOp.getMixedStrides();
    llvm::SmallBitVector droppedDims = subViewOp.getDroppedDims();
    size_t resultIdx = 0;
    for (int64_t dim = 0, e = subViewOp.getSourceType().getRank(); dim < e;
         ++dim) {
      Value offset =
          getValueOrCreateConstantIndexOp(rewriter, loc, mixedOffsets[dim]);
      if (droppedDims[dim]) {
        sourceIndices.push_back(offset);
        continue;
      }

      Value stride =
          getValueOrCreateConstantIndexOp(rewriter, loc, mixedStrides[dim]);
      Value scaledIndex =
          rewriter.create<arith::MulIOp>(loc, indices[resultIdx++], stride);
      sourceIndices.push_back(
          rewriter.create<arith::AddIOp>(loc, offset, scaledIndex));
    }
    return linearizeSingleElementI1Access(rewriter, loc, subViewOp.getSource(),
                                          sourceIndices);
  }

  if (auto castOp = dyn_cast<memref::CastOp>(defOp))
    return linearizeSingleElementI1Access(rewriter, loc, castOp.getSource(),
                                          indices);

  return LinearizedI1Access{baseMemref,
                            computeLinearMemRefOffset(
                                rewriter, loc, baseMemref, indices,
                                rewriter.getIndexType())};
}

static FailureOr<Value> buildRawLinearI1View(PatternRewriter &rewriter,
                                             Location loc, Value rootMemref) {
  auto rootTy = dyn_cast<MemRefType>(rootMemref.getType());
  if (!rootTy || !rootTy.hasStaticShape())
    return failure();

  SmallVector<int64_t> strides = getStaticStrides(rootTy);
  bool hasDynamicStride = false;
  for (int64_t stride : strides) {
    if (ShapedType::isDynamic(stride)) {
      hasDynamicStride = true;
      break;
    }
  }

  int64_t staticSpan = ShapedType::kDynamic;
  SmallVector<OpFoldResult> linearSizes;
  if (!hasDynamicStride) {
    staticSpan = 1;
    for (auto [dim, stride] : llvm::zip(rootTy.getShape(), strides))
      staticSpan += (dim - 1) * stride;
    linearSizes.push_back(rewriter.getIndexAttr(staticSpan));
  }
  auto linearTy = MemRefType::get(
      {staticSpan}, rootTy.getElementType(),
      StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic, {1}),
      rootTy.getMemorySpace());
  auto metadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, rootMemref);
  if (linearSizes.empty()) {
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value span = one;
    for (auto [size, stride] :
         llvm::zip(metadata.getSizes(), metadata.getStrides())) {
      Value dimMinusOne = rewriter.create<arith::SubIOp>(loc, size, one);
      Value dimSpan = rewriter.create<arith::MulIOp>(loc, dimMinusOne, stride);
      span = rewriter.create<arith::AddIOp>(loc, span, dimSpan);
    }
    linearSizes.push_back(span);
  }
  auto rawView = rewriter.create<memref::ReinterpretCastOp>(
      loc, linearTy, metadata.getBaseBuffer(), getAsOpFoldResult(metadata.getOffset()),
      linearSizes,
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)});
  return rawView.getResult();
}

// Decompose a memory offset into a VL-aligned base and an intra-VL offset.
//
// Given `currOffset`, computes:
//   base = floor(currOffset / VL) * VL   // VL-aligned base address
//   offsetInVL = currOffset - base           // remainder within [0, VL)
//
// This is used when loading i1 vectors: the hardware vector load must start
// at a VL-aligned boundary, and the actual data position within the loaded
// vector is tracked separately via `offsetInVL`.  For example, with VL=256
// and offset=300:
//   base = 256, offsetInVL = 44
//
// Returns {base, offsetInVL} both cast back to index type.
std::pair<Value, Value> getBaseAndOffetInVL(PatternRewriter &rewriter,
                                            Location &loc, Value currOffset) {
  Value constVL = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(util::VL));
  Value constByteSize = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(util::BITS_PER_BYTE));
  Value i32Offset = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI32Type(), currOffset);
  Value numVL = rewriter.create<arith::DivSIOp>(loc, i32Offset, constVL);
  Value newBase = rewriter.create<arith::MulIOp>(loc, numVL, constVL);
  Value newOffset = rewriter.create<arith::SubIOp>(loc, i32Offset, newBase);
  // Load indice will be convert to llvm.gep.
  // The offset of the gep instruction is measured in bytes.
  Value newBaseInByte =
      rewriter.create<arith::DivSIOp>(loc, newBase, constByteSize);
  Value indexBase = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), newBaseInByte);
  Value indexOffset = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), newOffset);
  return {indexBase, indexOffset};
}

/// Convert an i1 vector to a predicate register via i8 expansion:
///   i1 mask → select(i8 0/1) → broadcast → cmp(NE, 0) → constrain(B8)
/// If `offsetInVL` has value, inserts a PregXor + Reduction(XORI) before
/// the broadcast to shift the active element to the lowest position
/// (used for unaligned loads).
static Value convertI1ToPreg(VectorType orgVectorTy, Value i1Val,
                             Value offsetInVL, PatternRewriter &rewriter,
                             Location loc) {
  int64_t vecSize = orgVectorTy.getNumElements();
  VectorType i8VecTy = VectorType::get({util::VL}, rewriter.getI8Type());
  VectorType i8MaskTy = VectorType::get({util::VL}, rewriter.getI1Type());
  if (vecSize != util::VL)
    i1Val = rewriter.create<UnrealizedConversionCastOp>(loc, i8MaskTy, i1Val)
                .getResult(0);
  Value allI8Mask = rewriter.create<hivmave::VFPgeOp>(
      loc, i8MaskTy,
      PgePatternAttr::get(rewriter.getContext(), PgePattern::ALL));
  Value constZeroI8 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(rewriter.getI8Type()));
  Value constOneI8 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getOneAttr(rewriter.getI8Type()));

  // i1 → i8: select 1 for true lanes, 0 for false lanes.
  VFBroadcastScalarOp brcZero =
      rewriter.create<hivmave::VFBroadcastScalarOp>(loc, i8VecTy, constZeroI8);
  VFBroadcastScalarOp brcOne =
      rewriter.create<hivmave::VFBroadcastScalarOp>(loc, i8VecTy, constOneI8);
  Value selI8 = rewriter.create<hivmave::VFSelectOp>(loc, i8VecTy, i1Val,
                                                     brcOne, brcZero);

  // For unaligned loads, shift the active element to position 0 via
  // reduce-xori so that the broadcast propagates the correct value.
  Value shiftResult = selI8;
  if (offsetInVL) {
    Value indexOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    auto postOffsetInVL =
        rewriter.create<arith::AddIOp>(loc, offsetInVL, indexOne);
    VFPltOp p1 = rewriter.create<hivmave::VFPltOp>(
        loc, i8MaskTy, rewriter.getIndexType(), postOffsetInVL);
    VFPltOp p2 = rewriter.create<hivmave::VFPltOp>(
        loc, i8MaskTy, rewriter.getIndexType(), offsetInVL);
    PregXorOp pXor = rewriter.create<hivmave::PregXorOp>(
        loc, i8MaskTy, MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8),
        p1.getResults()[0], p2.getResults()[0], allI8Mask);
    shiftResult = rewriter.create<hivmave::ReductionOp>(
        loc, i8VecTy, hivmave::CombiningKind::XORI, selI8, pXor);
  }

  // Broadcast the (possibly shifted) i8 value, then compare NE with zero
  // to produce the final predicate register.
  VFBroadcastVectorOp brcI8 = rewriter.create<hivmave::VFBroadcastVectorOp>(
      loc, i8VecTy, shiftResult, allI8Mask, rewriter.getBoolAttr(true));
  Value newPreg = rewriter.create<hivmave::VFCmpOp>(
      loc, i8MaskTy, hivmave::CmpType::NE, brcI8, brcZero, allI8Mask);

  // Constrain the layout to B8 for VectorLayout analysis.
  newPreg = hivmave::constrainVectorLayout(newPreg, hivmave::VecMemType::B8,
                                           rewriter);
  if (vecSize != util::VL)
    newPreg =
        rewriter.create<UnrealizedConversionCastOp>(loc, orgVectorTy, newPreg)
            .getResult(0);
  return newPreg;
}

// process load + brc i1
struct loadBroadcastPattern : public OpRewritePattern<hivmave::VFLoadOp> {
  loadBroadcastPattern(MLIRContext *context)
      : OpRewritePattern<hivmave::VFLoadOp>(context, /*benefit=*/10) {}

  void rewriteLoadI1(mlir::hivmave::VFLoadOp loadOp,
                     PatternRewriter &rewriter) const {
    Location loc = loadOp.getLoc();
    rewriter.setInsertionPointAfter(loadOp);
    SmallVector<Operation *> oldUsers(loadOp.getRes().getUsers());
    Value newPreg = convertI1ToPreg(loadOp.getVectorType(), loadOp.getRes(),
                                    Value(), rewriter, loc);
    for (Operation *user : oldUsers)
      user->replaceUsesOfWith(loadOp.getRes(), newPreg);
    loadOp->setAttr(i1ProcessedAttr, rewriter.getUnitAttr());
  }

  LogicalResult rewriteLoadI1Linearized(mlir::hivmave::VFLoadOp loadOp,
                                        PatternRewriter &rewriter) const {
    Location loc = loadOp.getLoc();
    rewriter.setInsertionPointAfter(loadOp);
    auto linearized =
        linearizeSingleElementI1Access(rewriter, loc, loadOp.getBase(),
                                       loadOp.getIndices());
    if (failed(linearized))
      return failure();
    auto rawLinearView =
        buildRawLinearI1View(rewriter, loc, linearized->rootMemref);
    if (failed(rawLinearView))
      return failure();

    auto [baseIndices, offsetInVL] =
        getBaseAndOffetInVL(rewriter, loc, linearized->linearBitOffset);
    VectorType orgVectorTy = loadOp.getVectorType();
    VFLoadOp newLoad = rewriter.create<hivmave::VFLoadOp>(
        loc, orgVectorTy, *rawLinearView, baseIndices);
    Value newPreg = convertI1ToPreg(orgVectorTy, newLoad.getResult(0),
                                    offsetInVL, rewriter, loc);
    rewriter.replaceAllUsesWith(loadOp.getResult(0), newPreg);
    rewriter.eraseOp(loadOp);
    newLoad->setAttr(i1ProcessedAttr, rewriter.getUnitAttr());
    return success();
  }

  LogicalResult matchAndRewrite(mlir::hivmave::VFLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    VectorType orgVectorTy = loadOp.getVectorType();
    Type vecElemTy = orgVectorTy.getElementType();
    MemRefType memRefTy = loadOp.getMemRefType();
    if (loadOp->hasAttr(i1ProcessedAttr))
      return failure();
    if (!vecElemTy.isInteger(1) || !memRefTy.hasStaticShape() ||
        memRefTy.getNumElements() != 1)
      return failure();
    LDBG("Process operation : " << loadOp);

    if (!loadOp->hasAttr(UnalignedAttr::name) && memRefTy.getRank() == 1) {
      rewriteLoadI1(loadOp, rewriter);
    } else {
      if (failed(rewriteLoadI1Linearized(loadOp, rewriter)))
        return failure();
    }
    return success();
  }
};

namespace {
struct i1opSoftImplPass : public impl::I1opSoftImplBase<i1opSoftImplPass> {
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<loadBroadcastPattern>(context);
    mlir::GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> hivmave::createI1opSoftImplPass() {
  return std::make_unique<i1opSoftImplPass>();
}

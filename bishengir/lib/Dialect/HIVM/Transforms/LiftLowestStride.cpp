//===- LiftLowestStride.cpp ---- lift lowest stride of operands -----------===//
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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <type_traits>

#define DEBUG_TYPE "hivm-lift-lowest-stride"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_LIFTLOWESTSTRIDE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace util;

namespace {
/// Create lifted operand using ReinterpretCastOp.
/// Rank of the operand would increase by 1, sizes and strides would append 1.
static Value createLiftedOperand(PatternRewriter &rewriter, Location loc,
                                 Value operand) {
  if (operand.getType().isIntOrIndexOrFloat()) {
    return operand;
  }

  auto mem = cast<MemRefType>(operand.getType());
  auto [stridesLong, offsetLong] = getStridesAndOffset(mem);
  SmallVector<int64_t> sizesVec(mem.getShape());
  sizesVec.push_back(1);
  SmallVector<int64_t> stridesVec(stridesLong);
  stridesVec.push_back(1);

  auto castType = MemRefType::get(
      sizesVec, mem.getElementType(),
      StridedLayoutAttr::get(rewriter.getContext(), offsetLong, stridesVec),
      mem.getMemorySpace());

  // For dynamic shape, use ExtractStridedMetadataOp to get offset, sizes,
  // strides.
  auto stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, operand);
  OpFoldResult offsetOfr = getAsOpFoldResult(stridedMetadata.getOffset());
  SmallVector<OpFoldResult> sizesOfr =
      getAsOpFoldResult(stridedMetadata.getSizes());
  SmallVector<OpFoldResult> stridesOfr =
      getAsOpFoldResult(stridedMetadata.getStrides());

  sizesOfr.push_back(rewriter.getIndexAttr(1));
  stridesOfr.push_back(rewriter.getIndexAttr(1));

  return rewriter.create<memref::ReinterpretCastOp>(
      loc, castType, stridedMetadata.getBaseBuffer(), offsetOfr, sizesOfr,
      stridesOfr);
}

struct VBrcOpLiftLowestStridePattern : public OpRewritePattern<hivm::VBrcOp> {
  using OpRewritePattern<hivm::VBrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VBrcOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics");
    }

    auto src = op.getSrc();
    auto dst = op.getDst();

    auto srcMemRefType = dyn_cast_or_null<MemRefType>(src.getType());
    if (srcMemRefType && srcMemRefType.getRank() == 0) {
      return rewriter.notifyMatchFailure(op, " op should not have zero rank");
    }

    if (isLastDimContiguous(src) && isLastDimContiguous(dst)) {
      return rewriter.notifyMatchFailure(
          op, " last dim of all operands are contiguous or has size of 1,"
              " no need to lift stride.");
    }

    Value srcCast = createLiftedOperand(rewriter, op->getLoc(), src);
    Value dstCast = createLiftedOperand(rewriter, op->getLoc(), dst);
    rewriter.create<hivm::VBrcOp>(op->getLoc(), TypeRange(), srcCast, dstCast,
                                  op.getTempBuffer(),
                                  op.getBroadcastDimsAttr());

    // Erase old op
    rewriter.eraseOp(op);
    return success();
  }
};

struct VReduceOpLiftLowestStridePattern
    : public OpRewritePattern<hivm::VReduceOp> {
  using OpRewritePattern<hivm::VReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics");
    }

    auto src = op.getSrc();
    if (cast<MemRefType>(src.getType()).getRank() == 0) {
      return rewriter.notifyMatchFailure(op, " op should not have zero rank");
    }

    auto dstRange = op.getDst();
    bool isDstContiguous =
        std::all_of(dstRange.begin(), dstRange.end(), isLastDimContiguous);
    if (isLastDimContiguous(src) && isDstContiguous) {
      return rewriter.notifyMatchFailure(
          op, " last dim of all operands are contiguous or has size of 1,"
              " no need to lift stride.");
    }

    Value srcCast = createLiftedOperand(rewriter, op->getLoc(), src);
    Value indicesCast = nullptr;
    if (op.getIndices()) {
      indicesCast = createLiftedOperand(rewriter, op->getLoc(), op.getIndices());
    }


    SmallVector<Value> dstVec;
    for (Value dst : dstRange) {
      Value dstCast = createLiftedOperand(rewriter, op->getLoc(), dst);
      dstVec.push_back(dstCast);
    }

    rewriter.create<hivm::VReduceOp>(
        op->getLoc(), TypeRange(), srcCast, ValueRange(dstVec),
        op.getTempBuffer(), op.getArithAttr(), op.getReduceDimsAttr(), indicesCast);

    // Erase old op
    rewriter.eraseOp(op);
    return success();
  }
};

struct VTransposeOpLiftLowestStridePattern
    : public OpRewritePattern<hivm::VTransposeOp> {
  using OpRewritePattern<hivm::VTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VTransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics");
    }

    auto src = op.getSrc();
    auto dst = op.getDst();
    if (cast<MemRefType>(src.getType()).getRank() == 0) {
      return rewriter.notifyMatchFailure(op, " op should not have zero rank");
    }

    if (isLastDimContiguous(src) && isLastDimContiguous(dst)) {
      return rewriter.notifyMatchFailure(
          op, " last dim of all operands are contiguous or has size of 1,"
              " no need to lift stride.");
    }

    Value srcCast = createLiftedOperand(rewriter, op->getLoc(), src);
    Value dstCast = createLiftedOperand(rewriter, op->getLoc(), dst);
    SmallVector<int64_t> permVec(op.getPermutation());
    permVec.push_back(permVec.size());

    rewriter.create<hivm::VTransposeOp>(op->getLoc(), TypeRange(), srcCast,
                                        dstCast, op.getTempBuffer(), permVec);

    // Erase old op
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename CumOp>
struct CumulativeOpLiftLowestStridePattern : public OpRewritePattern<CumOp> {
  using OpRewritePattern<CumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CumOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics");
    }

    // zero rank
    auto src = op.getSrc();
    auto memref = dyn_cast_or_null<MemRefType>(src.getType());
    if (memref && memref.getRank() == 0) {
      return rewriter.notifyMatchFailure(op, " op should not have zero rank");
    }

    // last dim of cumsum/cumprod op shoule be contiguous or size 1
    bool isSrcContiguous = isLastDimContiguous(src);
    auto dst = op.getDst();
    bool isDstContiguous = isLastDimContiguous(dst);
    if (isSrcContiguous && isDstContiguous) {
      return rewriter.notifyMatchFailure(
          op, " last dim of all operands are contiguous or has size of 1,"
              " no need to lift stride.");
    }

    Value srcCast = createLiftedOperand(rewriter, op->getLoc(), src);
    Value dstCast = createLiftedOperand(rewriter, op->getLoc(), dst);

    rewriter.replaceOpWithNewOp<CumOp>(op, TypeRange(), srcCast, dstCast,
                                       op.getCumDims(), op.getReverse());
    return success();
  }
};

struct VMulExtendedOpLiftLowestStridePattern
    : public OpRewritePattern<hivm::VMulextendedOp> {
  using OpRewritePattern<hivm::VMulextendedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VMulextendedOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics");
    }

    // zero rank
    auto src0 = op.getSrc()[0];
    auto memref = dyn_cast_or_null<MemRefType>(src0.getType());
    if (memref && memref.getRank() == 0) {
      return rewriter.notifyMatchFailure(op, " op should not have zero rank");
    }

    // last dim of mulextended op shoule be contiguous or size 1
    bool isSrcContiguous = isLastDimContiguous(src0);
    auto dst0 = op.getDst()[0];
    bool isDstContiguous = isLastDimContiguous(dst0);
    if (isSrcContiguous && isDstContiguous) {
      return rewriter.notifyMatchFailure(
          op, " last dim of all operands are contiguous or has size of 1,"
              " no need to lift stride.");
    }
    auto src1 = op.getSrc()[1];
    auto dst1 = op.getDst()[1];
    Value src0Cast = createLiftedOperand(rewriter, op->getLoc(), src0);
    Value src1Cast = createLiftedOperand(rewriter, op->getLoc(), src1);
    Value dst0Cast = createLiftedOperand(rewriter, op->getLoc(), dst0);
    Value dst1Cast = createLiftedOperand(rewriter, op->getLoc(), dst1);

    rewriter.replaceOpWithNewOp<hivm::VMulextendedOp>(
        op, TypeRange(), ValueRange{src0Cast, src1Cast},
        ValueRange{dst0Cast, dst1Cast});
    return success();
  }
};

struct CopyOpLiftLowestStridePattern : public OpRewritePattern<hivm::CopyOp> {
  using OpRewritePattern<hivm::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::CopyOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics");
    }

    Value src = op.getSrc();
    Value dst = op.getDst();
    int64_t srcRank = cast<MemRefType>(src.getType()).getRank();
    if (srcRank == 0) {
      return rewriter.notifyMatchFailure(op, " op should not have zero rank");
    }

    if (isLastDimContiguous(src) && isLastDimContiguous(dst)) {
      return rewriter.notifyMatchFailure(
          op, " last dim of all operands are contiguous or has size of 1,"
              " no need to lift stride.");
    }

    Value srcCast = createLiftedOperand(rewriter, op->getLoc(), src);
    Value dstCast = createLiftedOperand(rewriter, op->getLoc(), dst);

    std::optional<ArrayAttr> maybeCollapseReassociation =
        op.getCollapseReassociation();
    SmallVector<ReassociationIndices> newReassociation;
    if (maybeCollapseReassociation.has_value()) {
      newReassociation = op.getReassociationIndices(/*isCollapse=*/true);
      // For example, before lift:
      //   memref<axbxf16, strided<[?, 8]>>
      //   collapse_reassociation = [[0, 1]]
      //
      // After lift:
      //   memref<axbx1xf16, strided<[?, 8, 1]>>
      //   collapse_reassociation [[0, 1], [2]]
      // The last dimension is not contiguous with the second-to-last dimension.
      newReassociation.push_back({srcRank});
    }
    rewriter.replaceOpWithNewOp<hivm::CopyOp>(
        op, TypeRange(), srcCast, dstCast, op.getPadModeAttr(),
        op.getPadValue(),
        /*collapse_reassociation=*/
        maybeCollapseReassociation.has_value()
            ? getReassociationIndicesAttribute(rewriter, newReassociation)
            : nullptr);
    return success();
  }
};

template <typename HIVMOp>
struct ElemwiseOpLiftLowestStridePattern : public OpRewritePattern<HIVMOp> {
  using OpRewritePattern<HIVMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HIVMOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics");
    }

    if (!mlir::hivm::detail::isElemwiseNaryOpImpl(op)) {
      return rewriter.notifyMatchFailure(op,
                                         " this pattern is for elemwiseOp.");
    }

    auto srcRange = op.getSrc();
    auto dstRange = op.getDst();

    bool hasZeroRank = llvm::any_of(srcRange, [](Value operand) {
      auto memref = dyn_cast_or_null<MemRefType>(operand.getType());
      return memref && memref.getRank() == 0;
    });
    if (hasZeroRank) {
      return rewriter.notifyMatchFailure(op, " op should not have zero rank");
    }

    bool isSrcContiguous =
        std::all_of(srcRange.begin(), srcRange.end(), isLastDimContiguous);
    bool isDstContiguous =
        std::all_of(dstRange.begin(), dstRange.end(), isLastDimContiguous);
    if (isSrcContiguous && isDstContiguous) {
      return rewriter.notifyMatchFailure(
          op, " last dim of all operands are contiguous or has size of 1,"
              " no need to lift stride.");
    }

    SmallVector<Value> srcVec;
    for (Value src : srcRange) {
      Value srcCast = createLiftedOperand(rewriter, op->getLoc(), src);
      srcVec.push_back(srcCast);
    }

    SmallVector<Value> dstVec;
    for (Value dst : dstRange) {
      Value dstCast = createLiftedOperand(rewriter, op->getLoc(), dst);
      dstVec.push_back(dstCast);
    }

    IRMapping mapper;
    mapper.map(op.getSrc(), ValueRange(srcVec));
    mapper.map(op.getDst(), ValueRange(dstVec));

    Operation *clonedOp = rewriter.clone(*op, mapper);
    HIVMOp clonedHIVMOp = cast<HIVMOp>(clonedOp);
    if (!op.getTranspose().empty()) {
      auto trans = op.getTranspose();
      SmallVector<int64_t> transposeVec(trans);
      transposeVec.push_back(trans.size());

      clonedHIVMOp.setTranspose(transposeVec);
    }

    rewriter.replaceOp(op, clonedHIVMOp);
    return success();
  }
};

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  if constexpr (!(std::is_same_v<OpType, hivm::VBrcOp> ||
                  std::is_same_v<OpType, hivm::VReduceOp> ||
                  std::is_same_v<OpType, hivm::VTransposeOp> ||
                  std::is_same_v<OpType, hivm::VArangeOp> ||
                  std::is_same_v<OpType, hivm::VMulExtOp> ||
                  std::is_same_v<OpType, hivm::VInterleaveOp> ||
                  std::is_same_v<OpType, hivm::VDeinterleaveOp> ||
                  std::is_same_v<OpType, hivm::VFlipOp> ||
                  std::is_same_v<OpType, hivm::VMulextendedOp> ||
                  std::is_same_v<OpType, hivm::VPadOp> ||
                  std::is_same_v<OpType, hivm::VConcatOp> ||
                  std::is_same_v<OpType, hivm::VGatherOp> ||
                  std::is_same_v<OpType, hivm::VGatherMaskOp> ||
                  std::is_same_v<OpType, hivm::VCumsumOp> ||
                  std::is_same_v<OpType, hivm::VCumprodOp> ||
                  std::is_same_v<OpType, hivm::VSortOp>)) {
    patterns.add<ElemwiseOpLiftLowestStridePattern<OpType>>(
        patterns.getContext());
  }
}

template <typename... OpTypes>
static void registerVectorOps(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

struct LiftLowestStridePass
    : public impl::LiftLowestStrideBase<LiftLowestStridePass> {
public:
  void runOnOperation() override;
};
} // namespace

void populateLiftLowestStridePatterns(RewritePatternSet &patterns) {
  // clang-format off
  (void)patterns.add<
    CopyOpLiftLowestStridePattern
  >(patterns.getContext());
  (void)patterns.add<
    VBrcOpLiftLowestStridePattern
  >(patterns.getContext());
  (void)patterns.add<
    VReduceOpLiftLowestStridePattern
  >(patterns.getContext());
  (void)patterns.add<
    VTransposeOpLiftLowestStridePattern
  >(patterns.getContext());
  (void)patterns.add<
    VMulExtendedOpLiftLowestStridePattern
  >(patterns.getContext());
  (void)patterns.add<CumulativeOpLiftLowestStridePattern<hivm::VCumsumOp>>(patterns.getContext());
  (void)patterns.add<CumulativeOpLiftLowestStridePattern<hivm::VCumprodOp>>(patterns.getContext());
  registerVectorOps<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >(patterns);
  // clang-format on
}

void LiftLowestStridePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLiftLowestStridePatterns(patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createLiftLowestStridePass() {
  return std::make_unique<LiftLowestStridePass>();
}

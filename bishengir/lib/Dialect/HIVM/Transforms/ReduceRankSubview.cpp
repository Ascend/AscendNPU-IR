//===- ReduceRankSubview.cpp ---- reduce rank by subview ------------------===//
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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-reduce-rank-subview"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_REDUCERANKSUBVIEW
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

// Returns dim indices that can be reduced, i.e. one-size dim and indice not
// included in stopDims
SmallVector<int64_t> getOneSizeDimsToDrop(MemRefType srcMem, MemRefType dstMem,
                                          ArrayRef<int64_t> stopDims) {
  ArrayRef<int64_t> srcShapes = srcMem.getShape();
  ArrayRef<int64_t> dstShapes = dstMem.getShape();
  assert(srcShapes.size() == dstShapes.size() &&
         "src and dst rank mismatch when reducing rank");
  DenseSet<int64_t> stopDimSet;
  llvm::for_each(stopDims, [&](int64_t dim) { stopDimSet.insert(dim); });
  SmallVector<int64_t> dropDims;
  int64_t rank = srcMem.getRank();
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (stopDimSet.contains(dim) || srcShapes[dim] != 1 ||
        dstShapes[dim] != 1) {
      continue;
    }
    dropDims.push_back(dim);
  }
  return dropDims;
}

SmallVector<int64_t>
getElemOneSizeDimsToDrop(llvm::SmallVectorImpl<Value> &operands) {
  auto memrefOpers = llvm::make_filter_range(operands, [](Value oper) {
    auto type = oper.getType();
    return isa<MemRefType>(type);
  });
  if (memrefOpers.empty()) {
    return {};
  }

  llvm::SmallVector<MemRefType> memrefTypes;
  llvm::transform(memrefOpers, std::back_inserter(memrefTypes),
                  [](Value v) { return cast<MemRefType>(v.getType()); });

  llvm::SmallVector<int64_t> dropDims;
  auto rank = cast<MemRefType>(memrefOpers.begin()->getType()).getRank();
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (llvm::all_of(memrefTypes,
                     [&](MemRefType t) {
                       auto shape = t.getShape();
                       return shape[dim] == 1;
                     }) &&
        (static_cast<int64_t>(dropDims.size()) != rank - 1)) {
      dropDims.push_back(dim);
    }
  }

  return dropDims;
}

static void debugShape(ArrayRef<int64_t> shape, std::string msg) {
  std::string shapeStr = ": [";
  for (auto s : shape) {
    shapeStr += std::to_string(s) + ", ";
  }
  shapeStr += "]";
  LDBG(msg << shapeStr);
}

MemRefType inferRankReducedResultType(ArrayRef<int64_t> resultShape,
                                      MemRefType sourceRankedTensorType,
                                      ArrayRef<int64_t> offsets,
                                      ArrayRef<int64_t> sizes,
                                      ArrayRef<int64_t> strides,
                                      DenseSet<int64_t> &dropDims) {
  auto inferredType = llvm::cast<MemRefType>(memref::SubViewOp::inferResultType(
      sourceRankedTensorType, offsets, sizes, strides));
  assert(inferredType.getRank() >= static_cast<int64_t>(resultShape.size()) &&
         "expected ");
  if (inferredType.getRank() == static_cast<int64_t>(resultShape.size()))
    return inferredType;

  // Debug which dimensions are dropped.
  LDBG("sourceRankedTensorType: " << sourceRankedTensorType);
  LDBG("inferredType: " << inferredType);
  debugShape(inferredType.getShape(), "inferredType.getShape(): ");
  debugShape(resultShape, "resultShape: ");
#ifndef NDEBUG
  for (unsigned dim : dropDims) {
    LDBG("dropDim: " << dim);
  }
#endif
  // Compute the layout and result type.
  auto inferredLayout = llvm::cast<StridedLayoutAttr>(inferredType.getLayout());
  SmallVector<int64_t> rankReducedStrides;
  rankReducedStrides.reserve(resultShape.size());
  for (auto [idx, value] : llvm::enumerate(inferredLayout.getStrides())) {
    if (!dropDims.contains(idx))
      rankReducedStrides.push_back(value);
  }
  return MemRefType::get(resultShape, inferredType.getElementType(),
                         StridedLayoutAttr::get(inferredLayout.getContext(),
                                                inferredLayout.getOffset(),
                                                rankReducedStrides),
                         inferredType.getMemorySpace());
}

MemRefType inferRankReducedResultType(ArrayRef<int64_t> resultShape,
                                      MemRefType sourceRankedTensorType,
                                      ArrayRef<OpFoldResult> offsets,
                                      ArrayRef<OpFoldResult> sizes,
                                      ArrayRef<OpFoldResult> strides,
                                      DenseSet<int64_t> &dropDims) {
  SmallVector<int64_t> staticOffsets;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicOffsets;
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return inferRankReducedResultType(resultShape, sourceRankedTensorType,
                                    staticOffsets, staticSizes, staticStrides,
                                    dropDims);
}

memref::SubViewOp getReducedSubviewOp(Value val, MemRefType mem, Location loc,
                                      DenseSet<int64_t> &dropDims,
                                      PatternRewriter &rewriter) {
  assert(!dropDims.empty() &&
         "dropDims must not be empty in order to get reduced subview op");

  SmallVector<int64_t> reducedSizes;
  SmallVector<OpFoldResult> viewOffsets;
  SmallVector<OpFoldResult> viewSizes;
  SmallVector<OpFoldResult> viewStrides;
  int64_t rank = mem.getRank();
  for (int64_t dim = 0; dim < rank; dim++) {
    viewOffsets.push_back(rewriter.getIndexAttr(0));
    viewStrides.push_back(rewriter.getIndexAttr(1));
    // subview has different sizes for dropped dim
    if (dropDims.contains(dim)) {
      viewSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      viewSizes.push_back(memref::getMixedSize(rewriter, loc, val, dim));
      reducedSizes.push_back(mem.getDimSize(dim));
    }
  }

  MemRefType reducedType = inferRankReducedResultType(
      reducedSizes, mem, viewOffsets, viewSizes, viewStrides, dropDims);

  return rewriter.create<memref::SubViewOp>(loc, reducedType, val, viewOffsets,
                                            viewSizes, viewStrides);
}

SmallVector<int64_t> adjustDropDims(int64_t rank, ArrayRef<int64_t> originDims,
                                    DenseSet<int64_t> &dropDims) {
  SmallVector<int64_t> resultDims;
  size_t originDimIdx = 0;
  int dropNum = 0;
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (originDimIdx >= originDims.size()) {
      break;
    }
    if (dropDims.contains(dim)) {
      dropNum++;
    } else if (dim == originDims[originDimIdx]) {
      originDimIdx++;
      resultDims.push_back(dim - dropNum);
    }
  }
  return resultDims;
}

struct HIVMElemReduceRankSuvbviewPattern
    : public OpInterfaceRewritePattern<hivm::HIVMStructuredOp> {
  using OpInterfaceRewritePattern<
      hivm::HIVMStructuredOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(hivm::HIVMStructuredOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics.");
    }

    if (!op.isElemwiseNaryOp() && !llvm::isa<hivm::CopyOp>(op.getOperation()) &&
        !llvm::isa<hivm::LoadOp>(op.getOperation()) &&
        !llvm::isa<hivm::StoreOp>(op.getOperation())) {
      return failure();
    }

    if (op.getDpsInits().empty()) {
      return failure();
    }

    auto output0Type = op.getDpsInits()[0].getType();
    if (!isa<MemRefType>(output0Type)) {
      return failure();
    }

    auto output0MemRefType = cast<MemRefType>(output0Type);
    if (output0MemRefType.getRank() == 1) {
      return failure();
    }

    auto operands = llvm::to_vector<4>(op->getOperands());
    auto dropDims = getElemOneSizeDimsToDrop(operands);
    if (dropDims.empty()) {
      return failure();
    }

    DenseSet<int64_t> dropDimSet;
    llvm::for_each(dropDims, [&](int64_t dim) { dropDimSet.insert(dim); });

    llvm::SmallVector<Value> reduceOperands;
    IRMapping map;
    for (auto oper : operands) {
      auto type = oper.getType();
      if (isa<MemRefType>(type)) {
        auto reduceOper = getReducedSubviewOp(
            oper, cast<MemRefType>(type), op->getLoc(), dropDimSet, rewriter);
        map.map(oper, reduceOper);
      }
    }

    rewriter.clone(*op.getOperation(), map);
    rewriter.eraseOp(op);
    return success();
  }
};

struct VBrcOpReduceRankSubviewPattern : public OpRewritePattern<hivm::VBrcOp> {
  using OpRewritePattern<hivm::VBrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VBrcOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics.");
    }

    auto src = op.getSrc();
    auto dst = op.getDst();
    auto srcTy = src.getType();
    auto dstTy = dst.getType();
    if (srcTy.isIntOrIndexOrFloat() || dstTy.isIntOrIndexOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, " no need to reduce rank for scalar brc.");
    }

    MemRefType srcMem = cast<MemRefType>(srcTy);
    MemRefType dstMem = cast<MemRefType>(dstTy);
    ArrayRef<int64_t> brcDims = op.getBroadcastDims();
    SmallVector<int64_t> dropDims =
        getOneSizeDimsToDrop(srcMem, dstMem, brcDims);
    if (dropDims.empty()) {
      return rewriter.notifyMatchFailure(op, " no dims to reduce.");
    }
    DenseSet<int64_t> dropDimSet;
    llvm::for_each(dropDims, [&](int64_t dim) { dropDimSet.insert(dim); });
    auto subviewSrcOp =
        getReducedSubviewOp(src, srcMem, op->getLoc(), dropDimSet, rewriter);
    auto subviewDstOp =
        getReducedSubviewOp(dst, dstMem, op->getLoc(), dropDimSet, rewriter);
    SmallVector<int64_t> newBrcDims =
        adjustDropDims(srcMem.getRank(), op.getBroadcastDims(), dropDimSet);

    rewriter.create<hivm::VBrcOp>(op->getLoc(), TypeRange(), subviewSrcOp,
                                  subviewDstOp, op.getTempBuffer(), newBrcDims);
    rewriter.eraseOp(op);
    return success();
  }
};

struct VReduceOpReduceRankSubviewPattern
    : public OpRewritePattern<hivm::VReduceOp> {
  using OpRewritePattern<hivm::VReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::VReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics.");
    }

    auto src = op.getSrc();
    auto dstRange = op.getDst();
    auto srcTy = src.getType();
    auto dstTy = op.getDstValue().getType();
    if (srcTy.isIntOrIndexOrFloat() || dstTy.isIntOrIndexOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, " no need to reduce rank for scalar brc.");
    }

    MemRefType srcMemType = cast<MemRefType>(srcTy);
    MemRefType dstMemType = cast<MemRefType>(dstTy);
    ArrayRef<int64_t> reduceDims = op.getReduceDims();
    SmallVector<int64_t> dropDims =
        getOneSizeDimsToDrop(srcMemType, dstMemType, reduceDims);
    if (dropDims.empty()) {
      return rewriter.notifyMatchFailure(op, " no dims to reduce.");
    }
    DenseSet<int64_t> dropDimSet;
    llvm::for_each(dropDims, [&](int64_t dim) { dropDimSet.insert(dim); });

    auto subviewSrcOp = getReducedSubviewOp(src, srcMemType, op->getLoc(),
                                            dropDimSet, rewriter);
    Value subviewIndicesOp = nullptr;
    if (op.getIndices()) {
      getReducedSubviewOp(op.getIndices(), srcMemType, op->getLoc(),
                                            dropDimSet, rewriter);
    }
    SmallVector<Value> subviewDstRange;
    for (Value dst : dstRange) {
      MemRefType dstType = cast<MemRefType>(dst.getType());
      subviewDstRange.push_back(getReducedSubviewOp(dst, dstType, op->getLoc(),
                                                    dropDimSet, rewriter));
    }
    SmallVector<int64_t> newReduceDims =
        adjustDropDims(srcMemType.getRank(), op.getReduceDims(), dropDimSet);

    rewriter.create<hivm::VReduceOp>(
        op->getLoc(), TypeRange(), subviewSrcOp, ValueRange(subviewDstRange),
        op.getTempBuffer(), op.getArithAttr(), newReduceDims, subviewIndicesOp);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReduceRankSubviewPass
    : public impl::ReduceRankSubviewBase<ReduceRankSubviewPass> {
public:
  void runOnOperation() override;
};
} // namespace

void populateReduceRankSubviewPatterns(RewritePatternSet &patterns) {
  // clang-format off
  (void)patterns.add<
      HIVMElemReduceRankSuvbviewPattern
  >(patterns.getContext());
  (void)patterns.add<
    VBrcOpReduceRankSubviewPattern
  >(patterns.getContext());
  (void)patterns.add<
    VReduceOpReduceRankSubviewPattern
  >(patterns.getContext());
  // clang-format on
}

void ReduceRankSubviewPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateReduceRankSubviewPatterns(patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createReduceRankSubviewPass() {
  return std::make_unique<ReduceRankSubviewPass>();
}

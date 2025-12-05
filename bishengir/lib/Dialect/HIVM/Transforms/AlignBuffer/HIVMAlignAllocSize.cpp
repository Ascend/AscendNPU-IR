//===- HIVMAlignAllocSize.cpp ---- Align Alloc Size Pass ------------------===//
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
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hivm-align-alloc-size"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_ALIGNALLOCSIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

Value createAlignedValue(
    PatternRewriter &rewriter, memref::AllocOp op,
    const SmallVectorImpl<OpFoldResult> &origShape,
    const llvm::SmallVectorImpl<OpFoldResult> &alignedShape) {
  SmallVector<Value> dynSizes;
  SmallVector<int64_t> staticSizes;
  dispatchIndexOpFoldResults(alignedShape, dynSizes, staticSizes);
  auto unAlignType = op.getType();
  MemRefType alignedTy = MemRefType::Builder(unAlignType)
                             .setShape(staticSizes)
                             .setLayout(MemRefLayoutAttrInterface());
  auto alignedAlloc =
      rewriter.create<memref::AllocOp>(op.getLoc(), alignedTy, dynSizes);

  SmallVector<OpFoldResult> offsets(alignedTy.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(alignedTy.getRank(),
                                    rewriter.getIndexAttr(1));
  MemRefType subviewResTy =
      cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          unAlignType.getShape(), alignedTy, offsets, origShape, strides));

  auto alignedMemRef = rewriter.create<memref::SubViewOp>(
      op.getLoc(), subviewResTy, alignedAlloc, offsets, origShape, strides);
  return alignedMemRef;
}

struct AlignAllocSizePattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    auto alignDimsAttr = op->getAttr(hivm::AllocAlignDimsAttr::name);
    auto alignBytesAttr = op->getAttr(hivm::AllocAlignValueInByteAttr::name);
    if (alignDimsAttr == nullptr || alignBytesAttr == nullptr)
      return failure();

    SmallVector<OpFoldResult> shape = op.getMixedSizes();

    auto alignBytes = cast<DenseI32ArrayAttr>(alignBytesAttr).asArrayRef();
    auto alignDims = cast<DenseI32ArrayAttr>(alignDimsAttr).asArrayRef();
    llvm::SmallVector<uint32_t> alignUnits(shape.size(), 1);
    auto elmentTypeBytes =
        getElementTypeOrSelf(op->getResultTypes()[0]).getIntOrFloatBitWidth() /
        mlir::utils::INTR_BITS_PER_BYTE;
    for (size_t i = 0; i < alignDims.size(); i++) {
      alignUnits[alignDims[i]] =
          (static_cast<uint64_t>(alignBytes[i]) / elmentTypeBytes);
    }
    rewriter.setInsertionPointAfter(op);
    llvm::SmallVector<OpFoldResult> alignedShape(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
      alignedShape[i] =
          AlignUpOFR(rewriter, op->getLoc(), shape[i], alignUnits[i]);
    }

    if (!isEqualConstantIntOrValueArray(alignedShape, shape)) {
      auto alignedMemRef =
          createAlignedValue(rewriter, op, shape, alignedShape);
      if (failed(replaceAndPropagateMemRefType(rewriter, op.getLoc(), op,
                                               alignedMemRef))) {
        LDBG("Cannot replace with aligned memref " << (Value)alignedMemRef);
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    } else {
      LDBG("no need to do alignment");
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op->removeAttr(hivm::AllocAlignDimsAttr::name);
      op->removeAttr(hivm::AllocAlignValueInByteAttr::name);
    });
    return success();
  }
};

struct AlignAllocSizePass
    : public impl::AlignAllocSizeBase<AlignAllocSizePass> {
public:
  void runOnOperation() override;
};
} // namespace

void populateAlignAllocAlignPattern(RewritePatternSet &patterns) {
  patterns.add<AlignAllocSizePattern>(patterns.getContext());
}

template <typename HIVMOP>
LogicalResult alignAllocSize(HIVMOP op, OpBuilder &builder) {
  if (!op.hasPureBufferSemantics()) {
    return failure();
  }

  std::vector<std::unique_ptr<util::OperAlignInfo>> operAlignInfoList;
  if (failed(getUnAlignSizeInfo(op, &operAlignInfoList))) {
    return failure();
  }

  for (auto &it : operAlignInfoList) {
    createAlignMarkOp(builder, op->getLoc(), it->operand, it->alignDims,
                      it->alignBytes, hivm::AllocAlignDimsAttr::name.str(),
                      hivm::AllocAlignValueInByteAttr::name.str());
  }
  return success();
}

LogicalResult markAllocAlign(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  WalkResult result = funcOp->walk([&builder](Operation *op) {
    if (auto transposeOp = dyn_cast<hivm::VTransposeOp>(op)) {
      if (!transposeOp.isLastDimTranspose()) {
        // un-last transpose, no need to do alloc size alignment, just do stride
        // alignment
        return WalkResult::skip();
      }

      if (transposeOp.getDisableAlign()) {
        return WalkResult::skip();
      }

      if (failed(alignAllocSize(transposeOp, builder))) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    } else if (auto castOp = dyn_cast<hivm::VCastOp>(op)) {
      auto srcType = getElementTypeOrSelf(castOp.getSrc()[0]);
      auto dstType = getElementTypeOrSelf(castOp.getDst()[0]);
      const bool isI32ToI8 = srcType.isInteger(32) && dstType.isInteger(8);
      const bool isI16ToI8 = srcType.isInteger(16) && dstType.isInteger(8);
      if (!isI32ToI8 && !isI16ToI8) {
        return WalkResult::skip();
      }

      if (failed(alignAllocSize(castOp, builder))) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    } else if (auto sortOp = dyn_cast<hivm::VSortOp>(op)) {
      if (failed(alignAllocSize(sortOp, builder))) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    return failure();
  }
  return success();
}

void AlignAllocSizePass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  // step 1: mark size align info
  if (failed(markAllocAlign(funcOp))) {
    return signalPassFailure();
  }

  LDBG("IR after marking alloc align");
  LDBG(funcOp);

  // step 2: propagate up align info to root memref.alloc
  RewritePatternSet patterns(&getContext());
  populatePropagateAlignUpToRootAllocationPattern(
      patterns, hivm::AllocAlignDimsAttr::name.str(),
      hivm::AllocAlignValueInByteAttr::name.str());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  LDBG("IR after propagating up alloc size to root memref.alloc");
  LDBG(funcOp);

  // step 3: modify the alloc and do size alignment
  patterns.clear();
  populateAlignAllocAlignPattern(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  LDBG("IR after modifying the alloc size");
  LDBG(funcOp);
}

std::unique_ptr<Pass> mlir::hivm::createAlignAllocSizePass() {
  return std::make_unique<AlignAllocSizePass>();
}

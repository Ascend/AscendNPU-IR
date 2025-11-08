//===--- HIVMInlineOTFLoadStore.cpp On the Fly Inline ---------------------===//
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
//
// This file implements a pass to inline the hivm load and store operations
//
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMINLINEOTFLOADSTORE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-inline-otf-load-store"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

bool isLastDimConcatAligned(hivm::VConcatOp concatOp, uint64_t bytesToAlign) {
  auto concatOut = concatOp.getDpsInitOperand(0)->get();
  auto tensorTypeOut = cast<RankedTensorType>(concatOut.getType());
  unsigned elemSizeInBytes = tensorTypeOut.getElementTypeBitWidth() / 8;
  uint64_t concatDim = concatOp.getDim();
  SmallVector<Value> inputs = concatOp.getDpsInputs();
  uint64_t accumOffset = 0;
  for (auto input : inputs) {
    auto shape = cast<ShapedType>(input.getType()).getShape();
    accumOffset = accumOffset + static_cast<uint64_t>(shape[concatDim]);
    if ((accumOffset * elemSizeInBytes) % bytesToAlign != 0)
      return false;
  }
  return true;
}

struct UnalignedLastDimConcatStorePattern
    : public OpRewritePattern<hivm::StoreOp> {
public:
  using OpRewritePattern<hivm::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto concatOp = storeOp.getDpsInputs()[0].getDefiningOp<hivm::VConcatOp>();
    if (!concatOp)
      return rewriter.notifyMatchFailure(storeOp, "not defined by concat");
    if (!concatOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(storeOp,
                                         "concat not pure tensor semantic");
    int64_t concatDim = static_cast<int64_t>(concatOp.getDim());
    auto concatOut = concatOp.getDpsInitOperand(0)->get();
    auto tensorTypeOut = cast<RankedTensorType>(concatOut.getType());
    auto rank = tensorTypeOut.getRank();
    if (concatDim != (rank - 1))
      return rewriter.notifyMatchFailure(storeOp, "concat is not last dim");
    if (!ShapedType::isDynamic(tensorTypeOut.getShape()[rank - 1]))
      if (isLastDimConcatAligned(concatOp, utils::INTR_BYTES_PER_BLOCK))
        return rewriter.notifyMatchFailure(storeOp, "concat dim is aligned");
    SmallVector<Value> inputs = concatOp.getDpsInputs();
    SmallVector<OpFoldResult> sliceOffsets(
        rank, getAsIndexOpFoldResult(rewriter.getContext(), 0));
    SmallVector<OpFoldResult> sliceStrides(
        rank, getAsIndexOpFoldResult(rewriter.getContext(), 1));
    Value insertSliceAccumulator = concatOp.getDst();
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      auto inputSizes = tensor::getMixedSizes(rewriter, input.getLoc(), input);
      LDBG("InsertSliceAccumulator " << insertSliceAccumulator << " for "
                                     << input);
      auto loc = insertSliceAccumulator.getLoc();
      auto insertSliceResults = rewriter.create<tensor::InsertSliceOp>(
          loc, input, insertSliceAccumulator, sliceOffsets, inputSizes,
          sliceStrides);

      if (idx + 1 < inputs.size()) {
        sliceOffsets[concatDim] = getValueOrCreateConstantIndexOp(
            rewriter, loc, sliceOffsets[concatDim]);
        inputSizes[concatDim] = getValueOrCreateConstantIndexOp(
            rewriter, loc, inputSizes[concatDim]);
        sliceOffsets[concatDim] =
            rewriter
                .create<arith::AddIOp>(loc, inputSizes[concatDim].get<Value>(),
                                       sliceOffsets[concatDim].get<Value>())
                .getResult();
      }
      insertSliceAccumulator = insertSliceResults.getResult();
    }
    rewriter.modifyOpInPlace(storeOp, [&]() {
      storeOp.getSrcMutable().assign(insertSliceAccumulator);
    });

    /// In many practical cases the vconcat op will also have a annotation.mark
    /// To specify Buffer Size, with this optimization we do not need that
    /// anymore, removing them will also let the vconcat op to be erased
    for (Operation *user : llvm::make_early_inc_range(concatOp->getUsers())) {
      if (utils::isAnnotationWithAttr(user, kBufferSizeInByteAttr)) {
        rewriter.eraseOp(user);
      }
    }
    return success();
  }
};
struct HIVMInlineOTFLoadStore
    : public impl::HIVMInlineOTFLoadStoreBase<HIVMInlineOTFLoadStore> {
  void runOnOperation() override;
};
} // namespace

void HIVMInlineOTFLoadStore::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<UnalignedLastDimConcatStorePattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createHIVMInlineOTFLoadStorePass() {
  return std::make_unique<HIVMInlineOTFLoadStore>();
}

//===- InferBufferSizeFromSlice.cpp - infer buffer sizes --------*- C++ -*-===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_INFERBUFFERSIZEFROMSLICE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

/// Mark the dst of a copy-like op (any CopyOpInterface, e.g. hivm.hir.copy,
/// hivm.hir.load, memref.copy) with a buffer_size_in_byte upper bound derived
/// from the op's src defining chain.
struct BufferSizeUpperBoundPattern
    : public OpInterfaceRewritePattern<CopyOpInterface> {
  using OpInterfaceRewritePattern<CopyOpInterface>::OpInterfaceRewritePattern;

  /// Trace `value`'s defining chain up to a statically-shaped ancestor and
  /// return its element count as a sound upper bound of `value`'s element
  /// count. Returns std::nullopt when no static ancestor can be reached.
  // TODO: Get rid of this ad-hoc tracing. Bind values to the dynamic dims of
  // extract_slice via the symbol dialect so sizes can be reused globally.
  std::optional<int64_t> getNumElemsUpperBound(Value value) const {
    auto shapedType = dyn_cast<ShapedType>(value.getType());
    if (shapedType && shapedType.hasStaticShape())
      return utils::getStaticTotalSize(shapedType.getShape());

    Operation *defOp = value.getDefiningOp();
    if (isa_and_nonnull<tensor::ExtractSliceOp, memref::SubViewOp>(defOp)) {
      return getNumElemsUpperBound(defOp->getOperand(0));
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(CopyOpInterface op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSource();
    Value dst = op.getTarget();
    auto loc = op.getLoc();
    // Idempotency: skip when dst already carries a buffer size mark, so the
    // clone below cannot re-trigger this pattern forever.
    if (utils::getAnnotateOpWithAttr(dst, utils::kBufferSizeInByteAttr))
      return failure();
    // Only direct results of memref.alloc are sized; other dsts (e.g. block
    // args, views) are left to their defining allocation.
    if (!isa_and_nonnull<memref::AllocOp, bufferization::ToTensorOp>(
            dst.getDefiningOp()))
      return failure();
    // dst must be dynamically-shaped to receive the buffer size mark.
    auto shapedType = dyn_cast<ShapedType>(dst.getType());
    if (!shapedType || shapedType.hasStaticShape())
      return failure();

    rewriter.setInsertionPoint(op);
    if (utils::cloneAnnotateOpWithAttr(rewriter, src, dst,
                                       utils::kBufferSizeInByteAttr))
      return success();
    auto numElems = getNumElemsUpperBound(src);
    if (!numElems.has_value())
      return failure();
    // The byte size uses the dst element type: quantizing ops (e.g. an
    // F322F16 fixpipe) must bound the narrower dst buffer, not the wider src.
    int64_t elemBits = getElementTypeOrSelf(dst).getIntOrFloatBitWidth();
    int64_t sizeInBytes =
        llvm::divideCeil(numElems.value() * elemBits, utils::kBitsToByte);
    auto markOp = rewriter.create<annotation::MarkOp>(loc, dst);
    markOp->setAttr(utils::kBufferSizeInByteAttr,
                    rewriter.getI64IntegerAttr(sizeInBytes));
    return success();
  }
};

struct InferBufferSizeFromSlicePass
    : public impl::InferBufferSizeFromSliceBase<InferBufferSizeFromSlicePass> {
  void runOnOperation() override;
};

} // namespace

void InferBufferSizeFromSlicePass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;
  RewritePatternSet patterns(&getContext());
  patterns.add<BufferSizeUpperBoundPattern>(&getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createInferBufferSizeFromSlicePass() {
  return std::make_unique<InferBufferSizeFromSlicePass>();
}

//===- ConstantizeBufferSize.cpp --------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONSTANTIZEBUFFERSIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct ConstantizeBufferPass
    : public impl::ConstantizeBufferSizeBase<ConstantizeBufferPass> {
  void runOnOperation() override;
};

template <typename AllocLikeOp>
struct ConstantizeAllocLikeOp : public OpRewritePattern<AllocLikeOp> {
public:
  using OpRewritePattern<AllocLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocLikeOp op,
                                PatternRewriter &rewriter) const override {
    auto dynSizes = op.getDynamicSizes();
    // No change if alloc is already fully static.
    if (dynSizes.empty())
      return failure();

    SmallVector<Value> newDynSizes;
    SmallVector<int64_t> newStaticSizes;
    MemRefType currentMemRefType = op.getType();
    auto currentSizes = currentMemRefType.getShape();
    int dynSizeIdx = 0;
    for (auto size : currentSizes) {
      if (!ShapedType::isDynamic(size)) {
        newStaticSizes.push_back(size);
        continue;
      }
      auto dynSize = dynSizes[dynSizeIdx];
      dynSizeIdx++;
      FailureOr<int64_t> upperBound =
          ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::UB, dynSize,
              /*stopCondition=*/nullptr, /*closedUB=*/true);
      // If failed to compute constant upper bound, do nothing.
      if (failed(upperBound)) {
        newStaticSizes.push_back(size);
        newDynSizes.push_back(dynSize);
        continue;
      }
      newStaticSizes.push_back(*upperBound);
    }
    if (newStaticSizes == currentSizes) {
      return rewriter.notifyMatchFailure(op, " already constantized");
    }

    // Construct new alloc with all dimensions constantized
    auto totalBits = utils::getStaticTotalSizeInBits(
        newStaticSizes, currentMemRefType.getElementType());
    if (!totalBits.has_value()) {
      return rewriter.notifyMatchFailure(
          op, " all dynamic dimensions should be constantized");
    }

    auto bufferSize = getAnnotateBufferSizeInBits(op.getResult());
    if (bufferSize.has_value()) {
      if (totalBits > bufferSize)
        return rewriter.notifyMatchFailure(
            op, " constantized buffer size should not exceed set buffer size");

      // remove all buffer_size_in_byte annotations if alloc can be constantized
      eraseAnnotateBufferSizeUsers(op.getResult(), rewriter);
    }

    int64_t totalBytes = static_cast<int64_t>(
        llvm::divideCeil(totalBits.value(), utils::kBitsToByte));
    auto newType =
        MemRefType::get({totalBytes}, rewriter.getI8Type(), mlir::AffineMap{},
                        currentMemRefType.getMemorySpace());
    Location loc = op->getLoc();
    auto newAlloc = rewriter.create<AllocLikeOp>(loc, newType);
    auto startOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.replaceOpWithNewOp<memref::ViewOp>(
        op, /*resultType=*/currentMemRefType, /*source=*/newAlloc.getResult(),
        /*byte_shift=*/startOffset,
        /*sizes=*/op->getOperands());
    return success();
  }

private:
  std::optional<int64_t> getAnnotateBufferSizeInBits(Value value) const {
    std::optional<Operation *> markMaybe =
        utils::getAnnotateOpWithAttr(value, kBufferSizeInByteAttr);
    if (!markMaybe.has_value()) {
      return std::nullopt;
    }
    auto markOp = cast<annotation::MarkOp>(markMaybe.value());
    return markOp->getAttrOfType<IntegerAttr>(kBufferSizeInByteAttr).getInt();
  }

  void eraseAnnotateBufferSizeUsers(Value value,
                                    PatternRewriter &rewriter) const {
    SmallVector<Operation *> annotateUsers;
    llvm::for_each(value.getUsers(), [&](Operation *user) {
      if (utils::isAnnotationWithAttr(user, kBufferSizeInByteAttr))
        annotateUsers.push_back(user);
    });

    for (Operation *user : annotateUsers)
      rewriter.eraseOp(user);
  }
};

} // namespace

void ConstantizeBufferPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.add<ConstantizeAllocLikeOp<memref::AllocOp>,
               ConstantizeAllocLikeOp<memref::AllocaOp>>(patterns.getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createConstantizeBufferSizePass() {
  return std::make_unique<ConstantizeBufferPass>();
}

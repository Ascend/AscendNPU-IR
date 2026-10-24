//===---------- HIVMAggregatedDecomposeOp.cpp - hivm op decompose----------===//
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
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Interfaces/AggregatedOpInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/RWMutex.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMAGGREGATEDDECOMPOSEOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-aggregated-decompose-op"

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct HIVMAggregatedDecomposeOpPass
    : public impl::HIVMAggregatedDecomposeOpBase<
          HIVMAggregatedDecomposeOpPass> {
  explicit HIVMAggregatedDecomposeOpPass(
      const HIVMAggregatedDecomposeOpOptions &options)
      : HIVMAggregatedDecomposeOpBase(options) {}

  void runOnOperation() override;
};

struct HIVMDecomposePattern : public OpInterfaceRewritePattern<
                                  bishengir::BiShengIRAggregatedOpInterface> {
  using OpInterfaceRewritePattern<
      bishengir::BiShengIRAggregatedOpInterface>::OpInterfaceRewritePattern;

  explicit HIVMDecomposePattern(MLIRContext *context,
                                bishengir::DecomposePhase d)
      : OpInterfaceRewritePattern<bishengir::BiShengIRAggregatedOpInterface>(
            context) {
    decomposePhase = d;
  }

  LogicalResult matchAndRewrite(bishengir::BiShengIRAggregatedOpInterface op,
                                PatternRewriter &rewriter) const override {
    bishengir::DecomposePhase phase = op.getDecomposePhase();
    if (phase != decomposePhase &&
        phase != bishengir::DecomposePhase::NO_CONSTRAINT) {
      return rewriter.notifyMatchFailure(op, "Not current phase");
    }

    FailureOr<SmallVector<Value>> maybeNewResults =
        op.decomposeOperation(rewriter);

    if (failed(maybeNewResults))
      return failure();

    if (maybeNewResults.value().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    rewriter.replaceOp(op, *maybeNewResults);
    return success();
  }

private:
  bishengir::DecomposePhase decomposePhase;
};

struct DecomposeUnalignedSubview : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSourceType();
    auto dstType = op.getType();
    if (srcType.getElementType().getIntOrFloatBitWidth() != 1)
      return failure();
    auto srcLastDimSize = srcType.getShape().back();
    auto dstLastDimSize = dstType.getShape().back();
    if (srcType.getRank() == 1)
      return failure();
    if (srcLastDimSize == dstLastDimSize)
      return failure();
    if (op.getDroppedDims().back())
      return failure();
    if (dstLastDimSize % utils::kUBAlignSizeInBits == 0)
      return failure();
    if (llvm::any_of(op->getUses(), [&](auto &use) {
          auto dstOp =
              dyn_cast_or_null<DestinationStyleOpInterface>(use.getOwner());
          if (!dstOp)
            return false;
          return dstOp.isDpsInit(&use);
        }))
      return failure();

    auto layout = dyn_cast<StridedLayoutAttr>(dstType.getLayout());
    if (!layout || layout.getStrides()[1] == srcLastDimSize)
      return failure();

    auto loc = op.getLoc();
    auto i1Type = rewriter.getI1Type();
    auto tmpType = rewriter.getF16Type();
    if ((dstLastDimSize * 16) % utils::kUBAlignSizeInBits != 0) {
      if ((dstLastDimSize * 32) % utils::kUBAlignSizeInBits != 0)
        return failure();
      tmpType = rewriter.getF32Type();
    }
    auto srcRoundAttr = rewriter.getAttr<hivm::RoundModeAttr>(
        utils::selectRoundMode<hivm::RoundMode>(i1Type, tmpType));

    auto srcCast = castTo(rewriter, loc, op.getSource(), srcRoundAttr, tmpType);
    auto newSubviewSrc = srcCast.getDst()[0];

    auto newSubviewType = memref::SubViewOp::inferRankReducedResultType(
        dstType.getShape(), cast<MemRefType>(newSubviewSrc.getType()),
        op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides());
    auto newOp = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(newSubviewType), newSubviewSrc,
        op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides());
    for (auto attr : op->getAttrs()) {
      if (!newOp->hasAttr(attr.getName()))
        newOp->setAttr(attr.getName(), attr.getValue());
    }
    auto dstBuffer =
        utils::createTmpBufferOrTensorWithTargetType(rewriter, loc, newOp);
    auto newValue =
        rewriter.create<hivm::CopyOp>(loc, TypeRange{}, newOp, dstBuffer)
            .getDst();
    auto oneValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(tmpType, 1));
    dstBuffer = utils::createTmpBufferOrTensorWithTargetType(rewriter, loc, op);
    auto dstCast = rewriter.create<hivm::VCmpOp>(
        loc, TypeRange{}, ValueRange{newValue, oneValue}, ValueRange{dstBuffer},
        hivm::CompareMode::EQ);
    rewriter.replaceOp(op, dstCast.getDst()[0]);
    return success();
  }
};

} // namespace

void HIVMAggregatedDecomposeOpPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;
  RewritePatternSet patterns(&getContext());
  patterns.add<HIVMDecomposePattern>(&getContext(), decomposePhase);
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (moduleOp && hacc::utils::isMemBasedArch(moduleOp) &&
      decomposePhase ==
          bishengir::DecomposePhase::AFTER_INFER_HIVM_DATA_LAYOUT) {
    LLVM_DEBUG(llvm::dbgs() << "Applying decompose unaligned subview\n";);
    patterns.add<DecomposeUnalignedSubview>(&getContext());
  }
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createHIVMAggregatedDecomposeOpPass(
    const HIVMAggregatedDecomposeOpOptions &options) {
  return std::make_unique<HIVMAggregatedDecomposeOpPass>(options);
}

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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

static std::optional<Type>
selectTmpElementTypeForUBAlign(int64_t lastDimElems,
                               PatternRewriter &rewriter) {
  for (Type type : {Type(rewriter.getF16Type()), Type(rewriter.getF32Type())}) {
    int64_t lastDimBits = lastDimElems * type.getIntOrFloatBitWidth();
    if (lastDimBits % utils::kUBAlignSizeInBits == 0)
      return type;
  }
  return std::nullopt;
}

static Value copyUnalignedSubviewViaLoadStore(PatternRewriter &rewriter,
                                              Location loc,
                                              memref::SubViewOp op) {
  Value compactBuffer =
      utils::createTmpBufferOrTensorWithTargetType(rewriter, loc, op);

  auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> srcIndices;
  SmallVector<Value> dstIndices;
  for (auto [size, offset] :
       llvm::zip(op.getMixedSizes(), op.getMixedOffsets())) {
    Value upper = getValueOrCreateConstantIndexOp(rewriter, loc, size);
    auto forOp = rewriter.create<scf::ForOp>(loc, c0, upper, c1);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value offsetVal = getValueOrCreateConstantIndexOp(rewriter, loc, offset);
    srcIndices.push_back(rewriter.create<arith::AddIOp>(loc, iv, offsetVal));
    dstIndices.push_back(iv);
  }

  Value loaded =
      rewriter.create<memref::LoadOp>(loc, op.getSource(), srcIndices);
  rewriter.create<memref::StoreOp>(loc, loaded, compactBuffer, dstIndices);

  return compactBuffer;
}

static void copyNamedAttrs(Operation *from, Operation *to) {
  for (auto attr : from->getAttrs()) {
    if (!to->hasAttr(attr.getName()))
      to->setAttr(attr.getName(), attr.getValue());
  }
}

/// After replacing a strided subview with a compact buffer, refresh derived
/// memref view ops whose result layout is computed from the source type.
static void refreshMemrefViewChain(PatternRewriter &rewriter, Value root) {
  SmallVector<Value> workList = {root};
  DenseSet<Value> visited;
  while (!workList.empty()) {
    Value val = workList.pop_back_val();
    if (!visited.insert(val).second)
      continue;

    SmallVector<Operation *> users(val.getUsers());
    for (Operation *user : users) {
      if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(user)) {
        auto srcType = cast<MemRefType>(expandOp.getSrc().getType());
        FailureOr<MemRefType> expectedType =
            memref::ExpandShapeOp::computeExpandedType(
                srcType, expandOp.getResultType().getShape(),
                expandOp.getReassociationIndices());
        if (succeeded(expectedType) &&
            *expectedType != expandOp.getResultType()) {
          rewriter.setInsertionPoint(expandOp);
          auto newOp = rewriter.create<memref::ExpandShapeOp>(
              expandOp.getLoc(), expandOp.getResultType().getShape(),
              expandOp.getSrc(), expandOp.getReassociationIndices());
          copyNamedAttrs(expandOp, newOp);
          rewriter.replaceOp(expandOp, newOp.getResult());
          workList.push_back(newOp.getResult());
        } else {
          workList.push_back(expandOp.getResult());
        }
        continue;
      }

      if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user)) {
        auto srcType = cast<MemRefType>(collapseOp.getSrc().getType());
        MemRefType expectedType = memref::CollapseShapeOp::computeCollapsedType(
            srcType, collapseOp.getReassociationIndices());
        if (expectedType != collapseOp.getResultType()) {
          rewriter.setInsertionPoint(collapseOp);
          auto newOp = rewriter.create<memref::CollapseShapeOp>(
              collapseOp.getLoc(), collapseOp.getSrc(),
              collapseOp.getReassociationIndices());
          copyNamedAttrs(collapseOp, newOp);
          rewriter.replaceOp(collapseOp, newOp.getResult());
          workList.push_back(newOp.getResult());
        } else {
          workList.push_back(collapseOp.getResult());
        }
        continue;
      }

      if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
        auto newType = cast<MemRefType>(memref::SubViewOp::inferResultType(
            cast<MemRefType>(subviewOp.getSource().getType()),
            subviewOp.getMixedOffsets(), subviewOp.getMixedSizes(),
            subviewOp.getMixedStrides()));
        if (newType != subviewOp.getType()) {
          rewriter.modifyOpInPlace(
              subviewOp, [&]() { subviewOp.getResult().setType(newType); });
        }
        workList.push_back(subviewOp.getResult());
        continue;
      }

      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        auto srcType = cast<MemRefType>(castOp.getSource().getType());
        auto dstType = cast<MemRefType>(castOp.getType());
        auto newDstType =
            MemRefType::get(dstType.getShape(), dstType.getElementType(),
                            srcType.getLayout(), dstType.getMemorySpace());
        if (newDstType != dstType) {
          rewriter.modifyOpInPlace(
              castOp, [&]() { castOp.getResult().setType(newDstType); });
        }
        workList.push_back(castOp.getResult());
        continue;
      }

      if (auto spaceCastOp = dyn_cast<memref::MemorySpaceCastOp>(user)) {
        workList.push_back(spaceCastOp.getResult());
      }
    }
  }
}

static void replaceSubviewAndRefreshMemrefViews(PatternRewriter &rewriter,
                                                memref::SubViewOp op,
                                                Value replacement) {
  rewriter.replaceOp(op, replacement);
  refreshMemrefViewChain(rewriter, replacement);
}

struct DecomposeUnalignedSubview : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSourceType();
    auto dstType = op.getType();
    if (!isLocalBuffer(GetBufferSpaceAttr(op.getSource())))
      return failure();
    if (!isMarkedExtractSliceOp(op))
      return failure();
    if (utils::isAlignedInUB(dstType))
      return failure();
    if (ShapedType::isDynamic(dstType.getShape().back()))
      return failure();
    if (op.getDroppedDims().back())
      return failure();
    auto slicedDims = getExtractOrInsertDim(op);
    if (slicedDims.size() != 1 || *slicedDims.begin() != srcType.getRank() - 1)
      return failure();
    if (llvm::any_of(op->getUses(), [&](auto &use) {
          auto dstOp =
              dyn_cast_or_null<DestinationStyleOpInterface>(use.getOwner());
          if (!dstOp)
            return false;
          return dstOp.isDpsInit(&use);
        }))
      return failure();

    auto elemType = srcType.getElementType();
    auto bitWidth = elemType.getIntOrFloatBitWidth();
    auto dstLastDimElems = dstType.getShape().back();
    auto tmpType = selectTmpElementTypeForUBAlign(dstLastDimElems, rewriter);
    auto loc = op.getLoc();

    if (!tmpType.has_value()) {
      Value result = copyUnalignedSubviewViaLoadStore(rewriter, loc, op);
      replaceSubviewAndRefreshMemrefViews(rewriter, op, result);
      return success();
    }

    auto srcRoundAttr = rewriter.getAttr<hivm::RoundModeAttr>(
        utils::selectRoundMode<hivm::RoundMode>(elemType, tmpType.value()));

    auto srcCast =
        castTo(rewriter, loc, op.getSource(), srcRoundAttr, tmpType.value());
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
    if (bitWidth == 1) {
      auto oneValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(tmpType.value(), 1));
      dstBuffer =
          utils::createTmpBufferOrTensorWithTargetType(rewriter, loc, op);
      auto dstCast = rewriter.create<hivm::VCmpOp>(
          loc, TypeRange{}, ValueRange{newValue, oneValue},
          ValueRange{dstBuffer}, hivm::CompareMode::EQ);
      replaceSubviewAndRefreshMemrefViews(rewriter, op, dstCast.getDst()[0]);
      return success();
    } else {
      auto dstRoundAttr = rewriter.getAttr<hivm::RoundModeAttr>(
          utils::selectRoundMode<hivm::RoundMode>(tmpType.value(), elemType));
      auto dstCast = castTo(rewriter, loc, newValue, dstRoundAttr, elemType);
      replaceSubviewAndRefreshMemrefViews(rewriter, op, dstCast.getDst()[0]);
      return success();
    }
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

//===--------- MapForToForall.cpp -  Map scf.for to scf.forall ops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to map scf.for op to scf.forall ops.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MAPFORTOFORALL
#include "bishengir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
static constexpr llvm::StringLiteral kMappingAttrName = "mapping";

struct ForToForallPass : public impl::MapForToForallBase<ForToForallPass> {
  explicit ForToForallPass(const MapForToForallOptions &other)
      : MapForToForallBase(other) {
    options.simtMode = other.simtMode;
  }
  void runOnOperation() override;
  MapForToForallOptions options;
};
} // namespace

struct ForToForallRewritePattern : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasAttrOfType<UnitAttr>(utils::kMapForToForallAttrName))
      return failure();

    std::optional<ArrayAttr> deviceMappingAttrs = std::nullopt;
    // If the mapping attribute exists beforehand, just use whatever's passed in
    if (op->hasAttrOfType<ArrayAttr>(kMappingAttrName))
      deviceMappingAttrs = op->getAttrOfType<ArrayAttr>(kMappingAttrName);
    // else if no mapping attribute exists, append a default one with no order
    else {
      deviceMappingAttrs = rewriter.getArrayAttr(
          {hivm::HIVMBlockMappingAttr::get(getContext())});
    }

    scf::ForallOp maybeResult = nullptr;
    DiagnosedSilenceableFailure diag = scf::utils::mapForToForallImpl(
        rewriter, op, deviceMappingAttrs, maybeResult, true);
    if (!diag.succeeded())
      return rewriter.notifyMatchFailure(op, diag.getMessage());

    rewriter.replaceOp(op, maybeResult);
    return success();
  }
};

struct InsertToInsertSliceRewritePattern
    : public OpRewritePattern<tensor::InsertOp> {
public:
  using OpRewritePattern<tensor::InsertOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(tensor::InsertOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Value splatOp = rewriter.create<tensor::SplatOp>(
        op.getLoc(), op.getScalar(), ArrayRef<int64_t>{1});
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    for (auto v : op.getIndices()) {
      offsets.push_back(v);
      sizes.push_back(rewriter.getI64IntegerAttr(1));
      strides.push_back(rewriter.getI64IntegerAttr(1));
    }
    auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        op.getLoc(), splatOp, op.getDest(), offsets, sizes, strides);
    rewriter.replaceOp(op, insertSliceOp);
    return success();
  }
};

void ForToForallPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  if (options.simtMode)
    patterns.insert<InsertToInsertSliceRewritePattern>(patterns.getContext());
  patterns.insert<ForToForallRewritePattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass>
scf::createMapForToForallPass(const MapForToForallOptions &options) {
  return std::make_unique<ForToForallPass>(options);
}

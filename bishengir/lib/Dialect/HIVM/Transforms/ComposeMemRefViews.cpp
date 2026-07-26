//===- ComposeMemRefViews.cpp --------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/ComposeMemRefViews.h"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::hivm {
namespace {

/// Compose a rank-preserving, unit-stride subview with its source
/// reinterpret_cast:
///
///   reinterpret_cast offset [B], sizes [...], strides [S0, S1, ...]
///     -> subview offsets [O0, O1, ...], sizes [N0, N1, ...],
///                strides [1, 1, ...]
///
/// becomes a reinterpret_cast of the original source with:
///
///   offset  = B + O0 * S0 + O1 * S1 + ...
///   sizes   = [N0, N1, ...]
///   strides = [S0, S1, ...]
///
/// Keeping the source strides is valid because the subview has unit strides;
/// keeping a one-to-one dimension mapping requires the subview to preserve
/// rank.
struct ComposeReinterpretCastSubviewPattern
    : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp subviewOp,
                                PatternRewriter &rewriter) const override {
    auto reinterpretCast =
        subviewOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCast)
      return failure();

    if (subviewOp.getSourceType().getRank() != subviewOp.getType().getRank() ||
        !subviewOp.hasUnitStride())
      return failure();

    rewriter.setInsertionPoint(subviewOp);
    Location loc = subviewOp.getLoc();
    using AV = affine::AffineValueExpr;
    affine::AffineBuilder affineBuilder(rewriter, loc);
    AffineExpr dim0, dim1, symbol0;
    bindDims(rewriter.getContext(), dim0, dim1);
    bindSymbols(rewriter.getContext(), symbol0);

    OpFoldResult composedOffset = reinterpretCast.getMixedOffsets().front();
    SmallVector<OpFoldResult> composedSizes;
    SmallVector<OpFoldResult> composedStrides;
    for (auto [offset, size, reinterpretStride] :
         llvm::zip(subviewOp.getMixedOffsets(), subviewOp.getMixedSizes(),
                   reinterpretCast.getMixedStrides())) {
      OpFoldResult scaledOffset =
          affineBuilder.mul(AV(dim0).bind(offset),
                            AV(symbol0).bind(reinterpretStride));
      composedOffset =
          affineBuilder.add(AV(dim0).bind(composedOffset),
                            AV(dim1).bind(scaledOffset));
      composedSizes.push_back(size);
      composedStrides.push_back(reinterpretStride);
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        subviewOp, subviewOp.getType(), reinterpretCast.getSource(),
        composedOffset, composedSizes, composedStrides);
    return success();
  }
};

} // namespace

void populateComposeMemRefViewPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<ComposeReinterpretCastSubviewPattern>(context);
}

} // namespace mlir::hivm

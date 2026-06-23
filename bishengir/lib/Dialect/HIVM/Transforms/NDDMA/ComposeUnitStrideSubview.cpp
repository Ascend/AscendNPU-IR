//===- ComposeUnitStrideSubview.cpp - Fuse unit-stride subviews ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Folds a unit-stride subview-of-subview pair into a single subview.
//
// Used as a companion to mlir::memref::populateComposeSubViewPatterns in the
// fuse-transpose-into-load pass. Upstream ComposeSubView rejects inner subviews
// with dynamic sizes; NDDMA tiled loads often produce exactly that shape, e.g.:
//
//   %cast = memref.reinterpret_cast %arg0
//       : memref<?xbf16> to memref<64x64xbf16, strided<[128, 1], offset: ?>>
//   %tile = memref.subview %cast[0, %col_off] [64, 32] [1, 1]
//       : memref<64x64xbf16, strided<[128, 1], offset: ?>>
//         to memref<64x32xbf16, strided<[128, 1], offset: ?>>
//   %win = memref.subview %tile[0, 0] [%m, %n] [1, 1]
//       : memref<64x32xbf16, strided<[128, 1], offset: ?>>
//         to memref<?x?xbf16, strided<[128, 1], offset: ?>>
//
// After one application of ComposeUnitStrideSubviewOpPattern:
//
//   %win = memref.subview %cast[0, %col_off] [%m, %n] [1, 1]
//       : memref<64x64xbf16, strided<[128, 1], offset: ?>>
//         to memref<?x?xbf16, strided<[128, 1], offset: ?>>
//
// Behavior
// -------
// * Matches the inner `memref.subview` when its source is another subview.
// * Requires both subviews to use static unit strides `[1, 1, ...]` on every
//   dimension.
// * Offsets compose additively (`inner + outer`) because strides are 1. Static
//   offsets are folded; mixed or dynamic offsets become `affine.apply`.
// * Sizes are taken from the inner subview unchanged, so dynamic sizes are
//   supported.
// * Merges one pair at a time; longer chains are folded by running the pattern
//   to fixpoint together with upstream compose-subview.
//
// Limitations
// -----------
// * The immediate source must be a `memref.subview`, not e.g. a
//   `memref.reinterpret_cast`. A chain `reinterpret_cast -> subview -> subview`
//   needs two greedy iterations to become a single subview on the cast.
// * The outer (source) subview must not be rank-reducing; rank-reducing views
//   are only supported as the innermost result type.
// * Non-unit or dynamic strides are not supported and are left unchanged.
// * Offset composition beyond a single static/dynamic add is not handled; the
//   pattern fails if offsets cannot be expressed that way.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/NDDMA/ComposeUnitStrideSubview.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::hivm::nddma;

namespace {

static bool hasUnitStrides(ArrayRef<OpFoldResult> strides) {
  for (OpFoldResult stride : strides) {
    auto strideVal = getConstantIntValue(stride);
    if (!strideVal || *strideVal != 1)
      return false;
  }
  return true;
}

static OpFoldResult composeUnitStrideOffset(OpFoldResult opOffset,
                                            OpFoldResult sourceOffset,
                                            PatternRewriter &rewriter,
                                            Location loc) {
  if (auto opConst = getConstantIntValue(opOffset); opConst && *opConst == 0)
    return sourceOffset;
  if (auto srcConst = getConstantIntValue(sourceOffset);
      srcConst && *srcConst == 0)
    return opOffset;

  Attribute opOffsetAttr = dyn_cast_if_present<Attribute>(opOffset);
  Attribute sourceOffsetAttr = dyn_cast_if_present<Attribute>(sourceOffset);
  if (opOffsetAttr && sourceOffsetAttr) {
    return rewriter.getIndexAttr(cast<IntegerAttr>(opOffsetAttr).getInt() +
                                 cast<IntegerAttr>(sourceOffsetAttr).getInt());
  }

  AffineExpr expr;
  SmallVector<Value> affineApplyOperands;
  if (sourceOffsetAttr) {
    expr =
        rewriter.getAffineConstantExpr(cast<IntegerAttr>(sourceOffsetAttr).getInt());
  } else {
    expr = rewriter.getAffineSymbolExpr(affineApplyOperands.size());
    affineApplyOperands.push_back(sourceOffset.get<Value>());
  }

  if (opOffsetAttr) {
    expr = expr + cast<IntegerAttr>(opOffsetAttr).getInt();
  } else {
    expr = expr + rewriter.getAffineSymbolExpr(affineApplyOperands.size());
    affineApplyOperands.push_back(opOffset.get<Value>());
  }

  AffineMap map =
      AffineMap::get(0, affineApplyOperands.size(), expr);
  return rewriter
      .create<affine::AffineApplyOp>(loc, map, affineApplyOperands)
      .getResult();
}

/// Greedy rewrite: `%inner = subview %outer` with `%outer = subview %root`.
/// See the file header for the full MLIR example, behavior, and limitations.
struct ComposeUnitStrideSubviewOpPattern
    : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceOp = op.getSource().getDefiningOp<memref::SubViewOp>();
    if (!sourceOp)
      return failure();

    if (sourceOp.getSourceType().getRank() != sourceOp.getType().getRank())
      return failure();

    if (!hasUnitStrides(op.getMixedStrides()) ||
        !hasUnitStrides(sourceOp.getMixedStrides()))
      return failure();

    SmallVector<OpFoldResult> offsets;
    offsets.reserve(op.getMixedOffsets().size());
    for (auto [opOffset, sourceOffset] :
         llvm::zip(op.getMixedOffsets(), sourceOp.getMixedOffsets())) {
      offsets.push_back(
          composeUnitStrideOffset(opOffset, sourceOffset, rewriter, op.getLoc()));
    }

    SmallVector<OpFoldResult> strides(op.getMixedStrides().size(),
                                      rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        op, op.getType(), sourceOp.getSource(), offsets, op.getMixedSizes(),
        strides);
    return success();
  }
};

} // namespace

void mlir::hivm::nddma::populateComposeUnitStrideSubviewPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ComposeUnitStrideSubviewOpPattern>(context);
}

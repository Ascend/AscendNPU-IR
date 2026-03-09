//===- ConvertLayoutUtils.h - Implementation of Utilities for Layouts -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_CONVERTLAYOUTUTILS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_CONVERTLAYOUTUTILS_H

#include "bishengir/Conversion/Passes.h"

namespace mlir::hivm {

/// Enum to specify fractal layout type for block size interpretation
/// God, I still consider whether this is an appropriate naming for this layout type 🥶
enum class FractalLayoutType {
  nZ,  // shape: (a ceildiv f[1], b ceildiv f[0], f[0], f[1])
  zN   // shape: (b ceildiv f[1], a ceildiv f[0], f[0], f[1])
};

using FractalSize = std::pair<int32_t, int32_t>;
bool isNDLayout(DataLayoutAttr layoutAttr);

FailureOr<FractalSize> extractBlockSizes(
    DataLayoutAttr layout);

FailureOr<SmallVector<OpFoldResult>> computeMixedNDToFractalShape(
    ArrayRef<OpFoldResult> currentShape,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout,
    OpBuilder &builder,
    Location loc);

FailureOr<SmallVector<OpFoldResult>> computeMixedFractalToNDShape(
    ArrayRef<OpFoldResult> currentShape,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout,
    OpBuilder &builder,
    Location loc);

FailureOr<SmallVector<int64_t>> computeNDToFractalShapeStatic(
    ArrayRef<int64_t> currentShape,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout);

FailureOr<SmallVector<OpFoldResult>> computeMixedTargetLayoutShape(
    ArrayRef<OpFoldResult> currentShape,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout,
    OpBuilder &builder,
    Location loc);

int computeBatchIndexBias(size_t rank);

Value createConvertLayoutLike(PatternRewriter &rewriter,
                              ConvertLayoutOp templateOp,
                              Value input);

Value createConvertLayoutOpposite(PatternRewriter &rewriter,
                              ConvertLayoutOp templateOp,
                              Value input);

bool isPropagatingUp(ConvertLayoutOp op);

bool isPropagatingDown(ConvertLayoutOp op);

bool isLayoutAgnosticOp(Operation *op);
void populateConvertLayoutElementwise(RewritePatternSet &patterns,
                                      MLIRContext *context);

void populateConvertLayoutScfFor(RewritePatternSet &patterns,
                                      MLIRContext *context);

void populateConvertLayoutExtractSlice(RewritePatternSet &patterns,
                                       MLIRContext *context);

void populateFixpipeExtractSlice(RewritePatternSet& patterns,
                                 MLIRContext* context);

FailureOr<SmallVector<OpFoldResult>> computeTargetLayoutOffset(
    ArrayRef<OpFoldResult> currentOffset,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout,
    PatternRewriter &rewriter,
    Location loc);
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_CONVERTLAYOUTUTILS_H
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

constexpr int32_t kFractalDims = 4;
using FractalSize = std::pair<int32_t, int32_t>;

struct PropagateConvertLayoutInternalOptions {
  bool allowAgnosticOps{false};
};

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

/// Compute the mixed (static/dynamic) shape that a tensor will have after a
/// layout conversion from srcLayout to dstLayout.
///
/// This is the single entry-point used by layout-propagation passes to derive
/// the result shape of a ConvertLayoutOp without duplicating the ND ↔ Fractal
/// shape arithmetic. The function dispatches to the appropriate direction
/// specific helper:
///
/// | Source  | Destination | Helper called                    |
/// |---------|-------------|----------------------------------|
/// | ND      | Fractal     |    computeMixedNDToFractalShape  |
/// | Fractal | ND          |    computeMixedFractalToNDShape  |
///
/// Any other combination (ND → ND, Fractal → Fractal) is considered invalid and
/// returns failure().
///
/// @param currentShape The operand's current mixed shape (static or dynamic
///                     OpFoldResult per dimension).
/// @param srcLayout    Data-layout attribute of the source tensor.
/// @param dstLayout    Data-layout attribute of the destination tensor.
/// @param builder      OpBuilder used to create any required affine
///                     expressions or constant attribute values.
/// @param loc          Location to attach to newly created operations.
/// @returns The mixed target shape on success, or failure() if the
///          conversion direction is unsupported or a sub-computation fails.
FailureOr<SmallVector<OpFoldResult>> computeMixedTargetLayoutShape(
    ArrayRef<OpFoldResult> currentShape,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout,
    OpBuilder &builder,
    Location loc);

int computeBatchIndexBias(size_t rank);

/// Create a new ConvertLayoutOp with the same source to destination layout
/// direction as templateOp but operating on input.
///
/// The helper clones templateOp and patches:
///   - the source operand to input,
///   - the result element type to match input's element type (preserving
///     the result shape from the clone).
///
/// This is used by propagation rewrites that need to insert a conversion on a
/// new value while preserving the layout pair established by an existing op.
///
/// @param rewriter      The pattern rewriter to use for cloning.
/// @param templateOp    The ConvertLayoutOp whose layout pair is reused.
/// @param input         The new source value for the cloned op.
/// @returns The result Value of the newly created ConvertLayoutOp.
Value createConvertLayoutLike(PatternRewriter &rewriter,
                              ConvertLayoutOp templateOp,
                              Value input);

/// Create a new ConvertLayoutOp that is the inverse of templateOp,
/// operating on input.
///
/// Where templateOp converts from layout A to layout B, the op created
/// here converts from layout B to layout A. The result type is derived by
/// cloning the source type of templateOp and substituting the element
/// type of input.
///
/// This is used by propagation rewrites that need to re-introduce the original
/// layout on a value after the conversion has been moved past an operation,
/// effectively "undoing" the conversion on the path back to consumers.
///
/// @param rewriter      The pattern rewriter used to create the new op.
/// @param templateOp    The ConvertLayoutOp whose layout pair is reversed.
/// @param input         The source value for the new inverse conversion op.
/// @returns The result Value of the newly created inverse ConvertLayoutOp.
Value createInverseConvertLayout(PatternRewriter& rewriter,
                                 ConvertLayoutOp templateOp,
                                 Value input);

bool isPropagatingUp(ConvertLayoutOp op);

bool isPropagatingDown(ConvertLayoutOp op);

/// Return true if op is a layout-agnostic operation through which a
/// ConvertLayoutOp may be propagated without changing semantics.
///
/// A layout-agnostic operation is one whose result is independent of the
/// memory layout of its operands, it merely forwards or transforms element
/// values. Layout-propagation passes use this predicate to determine whether
/// a ConvertLayoutOp can be moved past op.
bool isLayoutAgnosticOp(Operation *op);

FailureOr<SmallVector<OpFoldResult>> computeTargetLayoutOffset(
    ArrayRef<OpFoldResult> currentOffset,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout,
    PatternRewriter &rewriter,
    Location loc);

void populateConvertLayoutElementwise(RewritePatternSet &patterns,
                                      MLIRContext *context,
                  const PropagateConvertLayoutInternalOptions &options);

void populateConvertLayoutScfIf(RewritePatternSet &patterns,
                                      MLIRContext *context);

void populateConvertLayoutScfFor(RewritePatternSet &patterns,
                                      MLIRContext *context);

void populateConvertLayoutScfWhile(RewritePatternSet &patterns,
                                      MLIRContext *context);

void populateConvertLayoutExtractSlice(RewritePatternSet &patterns,
                                       MLIRContext *context);
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_CONVERTLAYOUTUTILS_H
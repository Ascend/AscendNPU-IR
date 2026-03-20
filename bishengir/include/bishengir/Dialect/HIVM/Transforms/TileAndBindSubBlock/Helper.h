//===- Helper.h --Helper functions for HIVMTileAndBindSubBlock pass--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TILEANDBINDSUBBLOCKHELPER_H
#define BISHENGIR_DIALECT_HIVM_TILEANDBINDSUBBLOCKHELPER_H

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mlir {
namespace hivm {

static constexpr llvm::StringLiteral toBeBubbleUpSlice = "to_be_bubbled_slice";
inline constexpr llvm::StringLiteral batchMatmulAttr = "batch_matmul";
static constexpr int kSubBlockDim = 2;
static constexpr int kMaxIterations = 50;

// Mark this op, so it will be considered as candidate to be bubbled up.
void markCreatedExtractSliceOp(RewriterBase &rewriter, Operation *op);

// Return true if this op is intented to be bubbled up.
bool isMarkedExtractSliceOp(Operation *op);

OpFoldResult calculateOffsetAtTilingDim(RewriterBase &rewriter, Location loc,
                                        scf::ForOp containingLoop,
                                        OpFoldResult singleTileSize);

/// This function calculates the tile size by dividing the dimension size
/// by kSubBlockDim (using ceiling division).
///
/// For static dimensions: tile_size = ceil(dim_size / kSubBlockDim)
/// For dynamic dimensions: creates affine operations to compute at runtime
///
/// @param input The input tensor to be tiled
/// @return The computed tile size as an OpFoldResult, or failure if the
///         static dimension size is less than kSubBlockDim
FailureOr<OpFoldResult> getSingleTileSize(OpBuilder &rewriter, Location loc,
                                          Value input, int64_t tileDimension,
                                          scf::ForOp containingLoop);

LogicalResult findCorrespondingSizesOffsetsStrides(
    RewriterBase &rewriter, ShapedType rankType, int64_t tilingDim,
    OpFoldResult offsetAtTileDim, OpFoldResult tileSize,
    SmallVector<OpFoldResult, 4> &mixedStrides,
    SmallVector<OpFoldResult, 4> &mixedOffsets,
    SmallVector<OpFoldResult, 4> &mixedSize, SmallVector<int64_t, 4> &newShape);

DenseSet<size_t> getExtractOrInsertDim(OffsetSizeAndStrideOpInterface op);
DenseSet<size_t> getIntersectionDims(DenseSet<size_t> dims1,
                                     const DenseSet<size_t> &dims2);

bool createdByTiling(OffsetSizeAndStrideOpInterface offsetSizeAndStrideOp);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TILEANDBINDSUBBLOCKHELPER_H
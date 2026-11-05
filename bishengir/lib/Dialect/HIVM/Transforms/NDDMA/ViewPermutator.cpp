//===- ViewPermutator.cpp - NDDMA tile view permutation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/NDDMA/ViewPermutator.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <numeric>

using namespace mlir;
using namespace mlir::hivm::nddma;

namespace {

bool isValidPermutation(ArrayRef<int64_t> permutation, int64_t rank) {
  if (static_cast<int64_t>(permutation.size()) != rank) {
    return false;
  }

  llvm::SmallBitVector seen(rank);
  for (int64_t dim : permutation) {
    if (dim < 0 || dim >= rank || seen.test(dim)) {
      return false;
    }
    seen.set(dim);
  }
  return true;
}

SmallVector<unsigned> getKeptRootDims(const TileView &tile) {
  SmallVector<unsigned> keptDims;
  for (unsigned dim = 0, e = tile.droppedDims.size(); dim < e; ++dim) {
    if (!tile.droppedDims.test(dim)) {
      keptDims.push_back(dim);
    }
  }
  return keptDims;
}

FailureOr<SmallVector<int64_t>> getRootPermutation(const TileView &tile,
                                                   ArrayRef<int64_t> viewPerm) {
  if (!isValidPermutation(viewPerm, tile.viewType.getRank()) ||
      tile.offsets.size() != tile.droppedDims.size()) {
    return failure();
  }

  SmallVector<unsigned> keptDims = getKeptRootDims(tile);
  if (keptDims.size() != viewPerm.size()) {
    return failure();
  }

  SmallVector<int64_t> rootPerm(tile.offsets.size());
  for (auto [index, value] : llvm::enumerate(rootPerm)) {
    value = static_cast<int64_t>(index);
  }
  for (auto [newDim, oldDim] : llvm::enumerate(viewPerm)) {
    rootPerm[keptDims[newDim]] = keptDims[oldDim];
  }
  return rootPerm;
}

Value permuteRoot(const TileView &tile, ArrayRef<int64_t> rootPerm,
                  OpBuilder &builder) {
  Value root = tile.root;
  if (auto allocOp = dyn_cast_or_null<memref::AllocOp>(root.getDefiningOp())) {
    SmallVector<OpFoldResult> newSizes =
        applyPermutation(allocOp.getMixedSizes(), rootPerm);
    SmallVector<Value> dynamicSizes;
    SmallVector<int64_t> newShape;
    dispatchIndexOpFoldResults(newSizes, dynamicSizes, newShape);

    auto newType = MemRefType::get(newShape, tile.rootType.getElementType(),
                                   MemRefLayoutAttrInterface{},
                                   tile.rootType.getMemorySpace());
    return builder
        .create<memref::AllocOp>(root.getLoc(), newType, dynamicSizes,
                                 allocOp.getAlignmentAttr())
        .getResult();
  }

  if (auto castOp =
          dyn_cast_or_null<memref::ReinterpretCastOp>(root.getDefiningOp())) {
    SmallVector<OpFoldResult> newSizes =
        applyPermutation(castOp.getMixedSizes(), rootPerm);
    SmallVector<OpFoldResult> newStrides =
        applyPermutation(castOp.getMixedStrides(), rootPerm);

    SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
    SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
    dispatchIndexOpFoldResults(castOp.getMixedOffsets(), dynamicOffsets,
                               staticOffsets);
    dispatchIndexOpFoldResults(newSizes, dynamicSizes, staticSizes);
    dispatchIndexOpFoldResults(newStrides, dynamicStrides, staticStrides);

    auto layout = StridedLayoutAttr::get(builder.getContext(),
                                         staticOffsets.front(), staticStrides);
    auto newType = MemRefType::get(staticSizes, tile.rootType.getElementType(),
                                   layout, tile.rootType.getMemorySpace());
    return builder
        .create<memref::ReinterpretCastOp>(
            root.getLoc(), newType, castOp.getSource(),
            castOp.getMixedOffsets().front(), newSizes, newStrides)
        .getResult();
  }

  llvm::report_fatal_error("unsupported root for NDDMA tile permutation");
}

TileView buildPermutedView(const TileView &tile, ArrayRef<int64_t> rootPerm,
                           ArrayRef<int64_t> viewPerm, Value newRoot,
                           OpBuilder &builder) {
  TileView result;
  result.root = newRoot;
  result.rootType = cast<MemRefType>(newRoot.getType());
  result.sizes = applyPermutation(tile.sizes, rootPerm);
  result.offsets = applyPermutation(tile.offsets, rootPerm);
  result.strides = applyPermutation(tile.strides, rootPerm);
  result.droppedDims = tile.droppedDims;

  if (tile.view == tile.root) {
    result.view = newRoot;
    result.viewType = result.rootType;
    return result;
  }

  auto newShape = applyPermutation(tile.viewType.getShape(), viewPerm);
  auto newType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      newShape, result.rootType, result.offsets, result.sizes, result.strides));
  auto subviewOp = builder.create<memref::SubViewOp>(
      tile.view.getLoc(), newType, newRoot, result.offsets, result.sizes,
      result.strides);
  result.view = subviewOp.getResult();
  result.viewType = subviewOp.getType();
  result.droppedDims = subviewOp.getDroppedDims();
  return result;
}

} // namespace

bool mlir::hivm::nddma::isLastTwoDimTranspose(ArrayRef<int64_t> permutation) {
  int64_t rank = static_cast<int64_t>(permutation.size());
  if (rank < 2) {
    return false;
  }

  for (int64_t dim = 0; dim < rank - 2; ++dim) {
    if (permutation[dim] != dim) {
      return false;
    }
  }
  return permutation[rank - 2] == rank - 1 && permutation[rank - 1] == rank - 2;
}

SmallVector<int64_t> mlir::hivm::nddma::getLastTwoDimPermutation(int64_t rank) {
  SmallVector<int64_t> permutation(rank);
  std::iota(permutation.begin(), permutation.end(), 0);
  if (rank >= 2) {
    std::swap(permutation[rank - 2], permutation[rank - 1]);
  }
  return permutation;
}

ViewPermutator::ViewPermutator(const TileView &tile,
                               ArrayRef<int64_t> permutation)
    : tile(tile), permutation(permutation) {}

bool ViewPermutator::canPermute() const {
  return succeeded(getRootPermutation(tile, permutation)) &&
         TileView::getRootSizes(tile.root).has_value();
}

TileView ViewPermutator::permute(OpBuilder &builder) const {
  auto rootPerm = getRootPermutation(tile, permutation);
  if (failed(rootPerm)) {
    llvm::report_fatal_error("invalid NDDMA tile permutation");
  }

  Value newRoot = permuteRoot(tile, *rootPerm, builder);
  return buildPermutedView(tile, *rootPerm, permutation, newRoot, builder);
}

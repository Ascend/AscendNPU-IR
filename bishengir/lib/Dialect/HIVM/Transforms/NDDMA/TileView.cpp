//===- TileView.cpp - NDDMA tile view utilities --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/NDDMA/TileView.h"

#include "bishengir/Dialect/Utils/IndexBoundAnalyzer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::hivm::nddma;

static Value getDominatorRoot(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return lhs;
  }

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef) {
    return lhs;
  }
  if (!rhsDef) {
    return rhs;
  }

  DominanceInfo dominance;
  if (dominance.dominates(lhsDef, rhsDef)) {
    return lhs;
  }
  if (dominance.dominates(rhsDef, lhsDef)) {
    return rhs;
  }
  llvm::report_fatal_error("cannot find a common higher NDDMA tile root");
}

static bool isDroppedDim(const TileView &tile, unsigned dim) {
  return dim < tile.droppedDims.size() && tile.droppedDims.test(dim);
}

static std::optional<unsigned> lastKeptDim(const TileView &tile) {
  for (unsigned dim = tile.offsets.size(); dim-- > 0;) {
    if (!isDroppedDim(tile, dim)) {
      return dim;
    }
  }
  return std::nullopt;
}

static bool coversRootDim(const TileView &tile, unsigned dim,
                          ArrayRef<OpFoldResult> rootSizes) {
  return isConstantIntValue(tile.offsets[dim], 0) &&
         isEqualConstantIntOrValue(tile.sizes[dim], rootSizes[dim]);
}

static OpFoldResult extentFromOffset(OpBuilder &builder, Location loc,
                                     OpFoldResult size, OpFoldResult offset) {
  if (isConstantIntValue(offset, 0)) {
    return size;
  }
  if (auto sizeCst = getConstantIntValue(size)) {
    if (auto offsetCst = getConstantIntValue(offset)) {
      return builder.getIndexAttr(*sizeCst - *offsetCst);
    }
  }
  Value sizeVal = getValueOrCreateConstantIndexOp(builder, loc, size);
  Value offsetVal = getValueOrCreateConstantIndexOp(builder, loc, offset);
  return builder.create<arith::SubIOp>(loc, sizeVal, offsetVal).getResult();
}

static void remapViewToRoot(TileView &tile, const TileView &rootTile,
                            OpBuilder &builder) {
  // Keep the tile's view window unchanged, but materialize it on the selected
  // root so users of both tiles read/write the same allocation.
  tile.root = rootTile.root;
  tile.rootType = rootTile.rootType;
  tile.rematerializeView(builder);
}

std::optional<SmallVector<OpFoldResult>> TileView::getRootSizes(Value root) {
  Operation *def = root.getDefiningOp();

  if (auto allocOp = dyn_cast_or_null<memref::AllocOp>(def)) {
    return allocOp.getMixedSizes();
  }

  if (auto castOp = dyn_cast_or_null<memref::ReinterpretCastOp>(def)) {
    if (isa<BlockArgument>(castOp.getSource())) {
      return castOp.getMixedSizes();
    }
  }

  return std::nullopt;
}

bool TileView::isContiguous() const {
  if (viewType.getLayout().isIdentity()) {
    return true;
  }

  SmallVector<int64_t> strides;
  int64_t offset = ShapedType::kDynamic;
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  if (failed(getStridesAndOffset(viewType, strides, offset))) {
#else
  if (failed(viewType.getStridesAndOffset(strides, offset))) {
#endif
    return false;
  }

  int64_t expectedStride = 1;
  for (int64_t dim = viewType.getRank() - 1; dim >= 0; --dim) {
    int64_t size = viewType.getDimSize(dim);
    if (size == 1) {
      continue;
    }
    if (strides[dim] != expectedStride) {
      return false;
    }
    if (ShapedType::isDynamic(size)) {
      return dim == 0;
    }
    expectedStride *= size;
  }
  return true;
}

void TileView::rematerializeView(OpBuilder &builder) {
  auto rootSizes = getRootSizes(root);
  bool isFullRoot = rootSizes.has_value();
  if (isFullRoot) {
    for (unsigned dim = 0, e = offsets.size(); dim < e; ++dim) {
      if (isDroppedDim(*this, dim) || !coversRootDim(*this, dim, *rootSizes)) {
        isFullRoot = false;
        break;
      }
    }
  }

  if (isFullRoot) {
    view = root;
    viewType = rootType;
    return;
  }

  SmallVector<int64_t> viewShape;
  for (unsigned dim = 0, e = offsets.size(); dim < e; ++dim) {
    if (isDroppedDim(*this, dim)) {
      continue;
    }
    viewShape.push_back(
        getConstantIntValue(sizes[dim]).value_or(ShapedType::kDynamic));
  }
  auto newType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      viewShape, rootType, offsets, sizes, strides));
  auto subview = builder.create<memref::SubViewOp>(view.getLoc(), newType, root,
                                                   offsets, sizes, strides);
  view = subview.getResult();
  viewType = subview.getType();
  droppedDims = subview.getDroppedDims();
}

void TileView::expandLastKeptDimToRoot(OpBuilder &builder) {
  auto rootSizes = getRootSizes(root);
  std::optional<unsigned> last = lastKeptDim(*this);
  if (!rootSizes || !last) {
    return;
  }

  OpFoldResult newSize = extentFromOffset(builder, view.getLoc(),
                                          (*rootSizes)[*last], offsets[*last]);
  if (isEqualConstantIntOrValue(sizes[*last], newSize)) {
    return;
  }
  sizes[*last] = newSize;
  rematerializeView(builder);
}

bool TileView::nonLastKeptDimsCoverRoot() const {
  auto rootSizes = getRootSizes(root);
  std::optional<unsigned> last = lastKeptDim(*this);
  if (!rootSizes || !last) {
    return false;
  }
  for (unsigned dim = 0; dim < *last; ++dim) {
    if (isDroppedDim(*this, dim)) {
      continue;
    }
    if (!coversRootDim(*this, dim, *rootSizes)) {
      return false;
    }
  }
  return true;
}

void TileView::unifyRoot(TileView &lhs, TileView &rhs, OpBuilder &builder) {
  // Only the root allocation shape must agree. The two tile views may describe
  // different windows over that allocation.
  if (lhs.rootType != rhs.rootType) {
    llvm::report_fatal_error("cannot unify NDDMA tile roots with different "
                             "types");
  }

  Value root = getDominatorRoot(lhs.root, rhs.root);
  if (lhs.root == root) {
    remapViewToRoot(rhs, lhs, builder);
    return;
  }
  remapViewToRoot(lhs, rhs, builder);
}

FailureOr<TileView> TileView::fromMemRef(Value memref, Builder &builder) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || type.getRank() < 2) {
    return failure();
  }

  if (auto subview = memref.getDefiningOp<memref::SubViewOp>()) {
    // Support one-level subview.
    Value root = subview.getSource();
    if (!getRootSizes(root).has_value()) {
      return failure();
    }
    return TileView{subview};
  }

  // Support `root == memref`.
  if (!getRootSizes(memref).has_value()) {
    return failure();
  }
  return TileView{memref, builder};
}

void TileView::print(raw_ostream &os) const {
  os << "TileView[\n"
     << "  view = " << view << "\n"
     << "  root = " << root << "\n"
     << "]\n";
}

TileView::TileView(memref::SubViewOp subview)
    : root(subview.getSource()), view(subview.getResult()),
      rootType(cast<MemRefType>(subview.getSource().getType())),
      viewType(cast<MemRefType>(subview.getResult().getType())),
      offsets(subview.getMixedOffsets()), sizes(subview.getMixedSizes()),
      strides(subview.getMixedStrides()),
      droppedDims(subview.getDroppedDims()) {}

TileView::TileView(Value memref, Builder &builder)
    : root(memref), view(memref), rootType(cast<MemRefType>(memref.getType())),
      viewType(cast<MemRefType>(memref.getType())) {
  offsets.assign(rootType.getRank(), builder.getIndexAttr(0));
  sizes = getRootSizes(memref).value();
  strides.assign(rootType.getRank(), builder.getIndexAttr(1));
  droppedDims.resize(rootType.getRank(), false);
}

raw_ostream &mlir::hivm::nddma::operator<<(raw_ostream &os,
                                           const TileView &tileView) {
  tileView.print(os);
  return os;
}

static LogicalResult verifyTileBounds(const TileView &permTile,
                                      const TileView &loadTile) {
  utils::IndexBoundAnalyzer boundAnalyzer;

  for (unsigned rootDim = 0, e = permTile.droppedDims.size(); rootDim < e;
       ++rootDim) {
    if (!isEqualConstantIntOrValue(loadTile.strides[rootDim],
                                   permTile.strides[rootDim])) {
      return failure();
    }

    if (permTile.view == permTile.root) {
      // permTile.view == permTile.root == loadTile.root, meaning that the whole
      // load dst buffer is being transposed, so load tile must be a part of
      // perm tile.
      // TODO: the current implementation is kind of aggressive, not sure
      // whether not-contiguous load tile is safely transposed, and whether we
      // should consider view size > root size cases.
      continue;
    }

    if (!isEqualConstantIntOrValue(loadTile.offsets[rootDim],
                                   permTile.offsets[rootDim])) {
      return failure();
    }

    if (boundAnalyzer.get(loadTile.sizes[rootDim]) <=
        boundAnalyzer.get(permTile.sizes[rootDim])) {
      continue;
    }

    return failure();
  }
  return success();
}

LogicalResult mlir::hivm::nddma::verifyLoadTileCompatibleWithPermTile(
    const TileView &loadTile, const TileView &permTile) {
  if (permTile.root != loadTile.root ||
      permTile.droppedDims != loadTile.droppedDims) {
    return failure();
  }

  if (!permTile.isContiguous() || !loadTile.isContiguous()) {
    return verifyTileBounds(permTile, loadTile);
  }

  return success();
}

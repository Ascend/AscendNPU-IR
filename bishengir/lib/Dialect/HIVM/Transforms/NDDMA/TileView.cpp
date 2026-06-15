//===- TileView.cpp - NDDMA tile view utilities --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/NDDMA/TileView.h"

#include "bishengir/Dialect/Utils/IndexBoundAnalyzer.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

static void remapViewToRoot(TileView &tile, const TileView &rootTile,
                            OpBuilder &builder) {
  // Keep the tile's view window unchanged, but materialize it on the selected
  // root so users of both tiles read/write the same allocation.
  tile.root = rootTile.root;
  tile.rootType = rootTile.rootType;

  auto newType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      tile.viewType.getShape(), tile.rootType, tile.offsets, tile.sizes,
      tile.strides));
  auto subview =
      builder.create<memref::SubViewOp>(tile.view.getLoc(), newType, tile.root,
                                        tile.offsets, tile.sizes, tile.strides);
  tile.view = subview.getResult();
  tile.viewType = subview.getType();
  tile.droppedDims = subview.getDroppedDims();
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
  if (failed(getStridesAndOffset(viewType, strides, offset))) {
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
    if (!isEqualConstantIntOrValue(loadTile.offsets[rootDim],
                                   permTile.offsets[rootDim]) ||
        !isEqualConstantIntOrValue(loadTile.strides[rootDim],
                                   permTile.strides[rootDim])) {
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

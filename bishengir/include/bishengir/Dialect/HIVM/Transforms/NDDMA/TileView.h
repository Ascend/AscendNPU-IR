//===- TileView.h - NDDMA tile view utilities ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines small helpers for describing memref tiles used by NDDMA
// transpose fusion.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_NDDMA_TILEVIEW_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_NDDMA_TILEVIEW_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallBitVector.h"

#include <optional>

namespace mlir {
namespace hivm {
namespace nddma {

/// Normalizes a memref or subview as a tile in one root allocation.
class TileView {
public:
  TileView() = default;

  /// Build a tile from a rank-reducing or normal subview.
  explicit TileView(memref::SubViewOp subview);

  /// Build a tile whose root and view are the same memref.
  explicit TileView(Value memref, Builder &builder);

  /// Build a tile from a root memref or one-level subview.
  static FailureOr<TileView> fromMemRef(Value memref, Builder &builder);

  /// Return root sizes for roots whose shape can be reconstructed.
  static std::optional<SmallVector<OpFoldResult>> getRootSizes(Value root);

  /// Return true when the current view has contiguous physical layout.
  bool isContiguous() const;

  /// Rebuild one tile view so both tiles use the dominating root.
  static void unifyRoot(TileView &lhs, TileView &rhs, OpBuilder &builder);

  /// Print the root/view pair for debug logging.
  void print(raw_ostream &os) const;

public:
  /// Tile state inspected by the current NDDMA transforms.

  /// Root allocation and current memref value represented by this tile.
  Value root;
  Value view;

  /// Types for `root` and `view`.
  MemRefType rootType;
  MemRefType viewType;

  /// Offsets/sizes/strides in root dimensions, including dropped dims.
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;

  /// Marks root dimensions removed by rank-reducing subviews.
  llvm::SmallBitVector droppedDims;
};

raw_ostream &operator<<(raw_ostream &os, const TileView &tileView);

/// Checks whether a load tile can replace the tile read by `to_tensor`.
LogicalResult verifyLoadTileCompatibleWithPermTile(const TileView &loadTile,
                                                   const TileView &permTile);

} // namespace nddma
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_NDDMA_TILEVIEW_H

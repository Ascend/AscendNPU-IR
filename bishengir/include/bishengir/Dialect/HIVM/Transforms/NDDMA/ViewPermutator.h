//===- ViewPermutator.h - NDDMA tile view permutation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_NDDMA_VIEWPERMUTATOR_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_NDDMA_VIEWPERMUTATOR_H

#include "bishengir/Dialect/HIVM/Transforms/NDDMA/TileView.h"

namespace mlir {
namespace hivm {
namespace nddma {

/// Helpers for the last-two-dim transpose pattern used by NDDMA load fusion.
bool isLastTwoDimTranspose(ArrayRef<int64_t> permutation);

SmallVector<int64_t> getLastTwoDimPermutation(int64_t rank);

/// Applies a visible-dimension permutation to a TileView.
class ViewPermutator {
public:
  ViewPermutator(const TileView &tile, ArrayRef<int64_t> permutation);

  /// Return true if `permute` can materialize this permutation.
  bool canPermute() const;

  /// Create a new root/view pair with the visible dimensions permuted.
  TileView permute(OpBuilder &builder) const;

private:
  TileView tile;
  SmallVector<int64_t> permutation;
};

} // namespace nddma
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_NDDMA_VIEWPERMUTATOR_H

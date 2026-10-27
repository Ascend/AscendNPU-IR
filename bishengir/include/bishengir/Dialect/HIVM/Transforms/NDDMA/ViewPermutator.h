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

/// NDDMA drives up to five nested loops (loop0 .. loop4), so a permutation
/// of at most this rank folds into a single descriptor. A higher rank still
/// reaches the descriptor for its innermost axes: the library call keeps five
/// axes and MultiDimOpToLibraryCallPattern peels the rest into loops.
constexpr int64_t kMaxNDDMARank = 5;

/// True for any non-identity permutation the NDDMA load path can carry.
bool isSupportedNDDMAPermutation(ArrayRef<int64_t> permutation);

/// True when the permutation maps the last axis onto itself. The load pad
/// counts address that axis, so they only survive such a permutation.
bool permutationKeepsLastAxis(ArrayRef<int64_t> permutation);

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

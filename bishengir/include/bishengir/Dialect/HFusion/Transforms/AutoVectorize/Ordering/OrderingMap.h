//===- OrderingMap.h - Ordering adjacency-map primitives ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_ORDERINGMAP_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_ORDERINGMAP_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace hfusion {

/// A pair of ordering-adjacency sets, keyed by a handle type `T`: `up` holds
/// the entities ordered before this one, `down` those ordered after.
template <typename T> struct OrderingSets {
  llvm::DenseSet<T> up;
  llvm::DenseSet<T> down;
};
template <typename T> using OrderingMap = llvm::DenseMap<T, OrderingSets<T>>;

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_ORDERINGMAP_H

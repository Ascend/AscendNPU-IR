//===- TensorViewAttrs.h - TensorView attributes ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TENSORVIEW_ATTRS_H
#define BISHENGIR_DIALECT_TENSORVIEW_ATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "bishengir/Dialect/TensorView/IR/TensorViewEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/TensorView/IR/TensorViewAttrs.h.inc"

namespace mlir {
namespace tv {

// Return true if `enc` is one of the view encodings (partition/strided/gs).
bool isViewEncoding(Attribute enc);

// Tile shape (== loaded/stored tensor shape) carried by a view encoding.
llvm::SmallVector<int64_t> getEncodingTileShape(Attribute enc);

// Per-dim traversal stride: the strided view's own field, or the tile shape
// itself for partition/gather-scatter views (which traverse tile-by-tile).
llvm::SmallVector<int64_t> getEncodingTraversal(Attribute enc);

// Gather/scatter dimensions of a view encoding (empty for non-gather-scatter).
llvm::SmallVector<int64_t> getEncodingSparseDims(Attribute enc);

// Padding value used by out-of-bounds view loads.
PaddingValue getEncodingPaddingValue(Attribute enc);

} // namespace tv
} // namespace mlir

#endif // BISHENGIR_DIALECT_TENSORVIEW_ATTRS_H

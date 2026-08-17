//===- MemRefDescriptor.h - the 1:N converted form of a MemRef ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HIVMTOTRITONGPU_MEMREFDESCRIPTOR_H
#define BISHENGIR_CONVERSION_HIVMTOTRITONGPU_MEMREFDESCRIPTOR_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hivm {

/// The converted form of a MemRef value.
///
///     memref<S...xT, layout, AS>
///       ->  (!tt.ptr<T, AS> allocPtr, !tt.ptr<T, AS> alignedPtr, i64 offset,
///            i64 size_0..size_{r-1}, i64 stride_0..stride_{r-1})
///
/// This is deliberately the SAME expansion LLVMTypeConverter::convertMemRefType
/// produces, and the same one FuncOpPattern must emit for the tt.func ABI. One
/// layout serves both, so the signature a function is given and the descriptor
/// its body reads can never drift apart.
///
/// The pointers are always the root allocation: no view offset is folded into
/// them. `offset` carries the fully composed view offset instead. `sizes` and
/// `strides` have exactly one entry per dimension of the converted MemRef's
/// rank, so a rank-reducing subview drops entries rather than zeroing them.
struct MemRefDescriptor {
  Value allocPtr;
  Value alignedPtr;
  Value offset;               // i64
  SmallVector<Value> sizes;   // i64, one per dim
  SmallVector<Value> strides; // i64, one per dim

  unsigned getRank() const { return strides.size(); }

  Value basePtr() const { return alignedPtr; }

  /// Flattens back into the ValueRange layout the type converter declares.
  SmallVector<Value> flatten() const {
    SmallVector<Value> flat{allocPtr, alignedPtr, offset};
    llvm::append_range(flat, sizes);
    llvm::append_range(flat, strides);
    return flat;
  }
};

/// Number of converted values a MemRef of this rank expands to:
/// two pointers, the offset, then one size and one stride per dimension.
inline unsigned getDescriptorSize(unsigned rank) { return 3 + 2 * rank; }

/// Reads a descriptor out of the values a 1:N adaptor supplied for one MemRef
/// operand. Returns failure when the range is not a well-formed descriptor,
/// which happens when the operand reached the pattern unconverted.
inline FailureOr<MemRefDescriptor> getMemRefDescriptor(ValueRange values) {
  // 3 + 2r values, so anything below the fixed prefix or with an odd number of
  // trailing entries is not a descriptor.
  if (values.size() < getDescriptorSize(0) || (values.size() - 3) % 2 != 0)
    return failure();

  unsigned rank = (values.size() - 3) / 2;
  MemRefDescriptor desc;
  desc.allocPtr = values[0];
  desc.alignedPtr = values[1];
  desc.offset = values[2];
  llvm::append_range(desc.sizes, values.slice(3, rank));
  llvm::append_range(desc.strides, values.slice(3 + rank, rank));
  return desc;
}

inline FailureOr<MemRefDescriptor> getMemRefDescriptor(OpBuilder &builder,
                                                       Location loc,
                                                       MemRefType memrefTy,
                                                       ValueRange values) {
  FailureOr<MemRefDescriptor> desc = getMemRefDescriptor(values);
  if (failed(desc) || desc->getRank() != memrefTy.getRank())
    return failure();

  SmallVector<int64_t> staticStrides;
  int64_t staticOffset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(memrefTy, staticStrides, staticOffset)))
    return failure();

  auto pin = [&](Value &field, int64_t known) {
    if (!ShapedType::isDynamic(known))
      field = builder.create<arith::ConstantIntOp>(loc, known, 64);
  };
  pin(desc->offset, staticOffset);
  for (auto [dim, stride] : llvm::enumerate(staticStrides))
    pin(desc->strides[dim], stride);
  return desc;
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMTOTRITONGPU_MEMREFDESCRIPTOR_H

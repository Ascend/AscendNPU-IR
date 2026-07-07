//===-----------------------------Utils.cpp--------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MemRef/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "memref-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace memref {

BitVector getContiguousAxes(ArrayRef<Type> shapedTypes) {
  BitVector ret;
  // Presume all same rank
  for (auto type : shapedTypes) {
    auto shapedType = dyn_cast<MemRefType>(type);
    if (!shapedType)
      continue;
    int rank = shapedType.getRank();
    if (ret.empty())
      ret.resize(shapedType.getRank(), true);

    auto stridedLayout = dyn_cast<StridedLayoutAttr>(shapedType.getLayout());
    if (!stridedLayout)
      continue;
    if (stridedLayout.isIdentity())
      continue;

    auto strides = stridedLayout.getStrides();
    auto shape = shapedType.getShape();
    for (int64_t axis = 1; axis < rank; axis++) {
      // If it's dynamic its undeterminable
      if (ShapedType::isDynamic(strides[axis]) ||
          ShapedType::isDynamic(strides[axis - 1]) ||
          ShapedType::isDynamic(shape[axis])) {
        ret[axis] = false;
        continue;
      }
      // Check if its contiguous
      if (strides[axis] * shape[axis] != strides[axis - 1])
        ret[axis] = false;
    }
  }
  // First dimension is always contiguous
  if (!ret.empty())
    ret[0] = true;
  return ret;
}
} // namespace memref
} // namespace mlir
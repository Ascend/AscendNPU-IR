//===- MemRefExtImpl.cpp.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MemRefExt/IR/MemRefExtImpl.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"

namespace mlir {
namespace memref_ext {

bool isDefiningOpAllocLike(Value operand) {
  if (!operand.getDefiningOp()) {
    return false;
  }

  if (isa<memref::AllocOp, bishengir::memref_ext::AllocWorkspaceOp>(
          operand.getDefiningOp())) {
    return true;
  }
  return false;
}

} // namespace memref_ext
} // namespace mlir

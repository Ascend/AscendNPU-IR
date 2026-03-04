//===- AllocToPointerCast.h --Convert memref.AllocOp to hivm.pointercastOp-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace hivm {
#ifndef LLVM_PROJECT_ALLOCTOPOINTERCAST_H
#define LLVM_PROJECT_ALLOCTOPOINTERCAST_H
class MemrefAllocaOpToPointerCastOpPattern
    : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  /// map from buffer to its allocated addresses
  /// note: the buffer which does multibuffer n optimization will be allocated n
  /// addresses.
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;

  explicit MemrefAllocaOpToPointerCastOpPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<memref::AllocOp>(context),
        buffer2Offsets(buffer2Offsets) {}
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final;
};

class UpdateWorkSpaceAllocaOpOffsetPattern
    : public OpRewritePattern<bishengir::memref_ext::AllocWorkspaceOp> {
public:
  using OpRewritePattern<
      bishengir::memref_ext::AllocWorkspaceOp>::OpRewritePattern;

  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;

  explicit UpdateWorkSpaceAllocaOpOffsetPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<bishengir::memref_ext::AllocWorkspaceOp>(context),
        buffer2Offsets(buffer2Offsets) {}
  LogicalResult matchAndRewrite(bishengir::memref_ext::AllocWorkspaceOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace hivm
} // namespace mlir

#endif // LLVM_PROJECT_ALLOCTOPOINTERCAST_H

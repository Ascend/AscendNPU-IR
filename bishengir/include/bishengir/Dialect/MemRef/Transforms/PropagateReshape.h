//===- PropagateReshape.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#ifndef BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PROPAGATERESHAPE_H
#define BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PROPAGATERESHAPE_H

namespace mlir {
namespace memref {

// Pattern to propagate collapse shape operations downward through the IR
class PropagateMemrefCollapseDown
    : public mlir::OpRewritePattern<memref::CollapseShapeOp> {
public:
  explicit PropagateMemrefCollapseDown(MLIRContext *context)
      : OpRewritePattern<memref::CollapseShapeOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(memref::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override;
};

// Pattern to propagate expand shape operations upward through the IR
class PropagateMemrefExpandUp
    : public mlir::OpRewritePattern<memref::ExpandShapeOp> {
public:
  explicit PropagateMemrefExpandUp(MLIRContext *context)
      : OpRewritePattern<memref::ExpandShapeOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(memref::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace memref
} // namespace mlir

#endif // BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PROPAGATERESHAPE_H
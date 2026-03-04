//===- PropagateCollapseDown.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#ifndef BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_COLLAPSEDOWN_H
#define BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_COLLAPSEDOWN_H

namespace mlir {
namespace tensor {

// Pattern to propagate collapse shape operations downward through the IR
class PropagateCollapseDown
    : public mlir::OpRewritePattern<tensor::CollapseShapeOp> {
public:
  explicit PropagateCollapseDown(MLIRContext *context,
                                 PropagateReshapeOptions opts)
    : OpRewritePattern<tensor::CollapseShapeOp>(context, /*benefit=*/1) {
    options = opts;
  }

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override;

private:
  PropagateReshapeOptions options;
};

} // namespace tensor
} // namespace mlir

#endif // BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_COLLAPSEDOWN_H
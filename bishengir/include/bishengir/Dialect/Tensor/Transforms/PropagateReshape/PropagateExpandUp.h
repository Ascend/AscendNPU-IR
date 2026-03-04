//===- PropagateExpandUp.h ------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#ifndef BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_EXPANDUP_H
#define BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_EXPANDUP_H

namespace mlir {
namespace tensor {

// Pattern to propagate expand shape operations upward through the IR
class PropagateExpandUp : public mlir::OpRewritePattern<tensor::ExpandShapeOp> {
public:
  explicit PropagateExpandUp(MLIRContext *context, PropagateReshapeOptions opts)
      : OpRewritePattern<tensor::ExpandShapeOp>(context, /*benefit=*/1) {
    options = opts;
  }
  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override;

private:
  PropagateReshapeOptions options;
};

} // namespace tensor
} // namespace mlir

#endif // BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_EXPANDUP_H
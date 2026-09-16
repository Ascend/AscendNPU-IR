//===- PropagatableOp.h - Header for propagatable ops ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#ifndef BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_PROPAGATABLEOP_H
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"

namespace mlir::tensor {
using namespace mlir::hfusion;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::hfusion::reshape_utils;

class PropagatableOp {
public:
  /// Matches and rewrites an expand shape operation by propagating it through
  /// the operation. This is called when an expand operation needs to be moved
  /// or transformed in relation to this operation.
  ///
  /// @param rewriter The pattern rewriter used for IR modifications
  /// @param op The operation that the expand shape is being propagated through
  /// @param expandOp The expand shape operation to be propagated
  /// @return LogicalResult indicating success or failure of the transformation
  virtual LogicalResult matchAndRewriteExpand(PatternRewriter &rewriter,
                                              Operation *op,
                                              tensor::ExpandShapeOp expandOp);

  /// Matches and rewrites a collapse shape operation by propagating it through
  /// the operation. This is called when a collapse operation needs to be moved
  /// or transformed in relation to this operation.
  ///
  /// @param rewriter The pattern rewriter used for IR modifications
  /// @param op The operation that the collapse shape is being propagated
  /// through
  /// @param collapseOp The collapse shape operation to be propagated
  /// @return LogicalResult indicating success or failure of the transformation
  virtual LogicalResult
  matchAndRewriteCollapse(PatternRewriter &rewriter, Operation *op,
                          tensor::CollapseShapeOp collapseOp);

  virtual ~PropagatableOp() = default;
};

class PropagatableScfFor : public PropagatableOp {
public:
  LogicalResult matchAndRewriteExpand(mlir::PatternRewriter &rewriter,
                                      mlir::Operation *op,
                                      tensor::ExpandShapeOp expandOp) override;
};

class PropagatableMulExt : public PropagatableOp {
public:
  LogicalResult matchAndRewriteExpand(PatternRewriter &rewriter,
                                      Operation *op,
                                      tensor::ExpandShapeOp expandOp) override;
  LogicalResult
  matchAndRewriteCollapse(PatternRewriter &rewriter, Operation *op,
                          tensor::CollapseShapeOp collapseOp) override;
};

class PropagatableAnnotationMark : public PropagatableOp {
public:
  LogicalResult
  matchAndRewriteCollapse(PatternRewriter &rewriter, Operation *op,
                          tensor::CollapseShapeOp collapseOp) override;
};
} // namespace mlir::tensor
#define BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_PROPAGATABLEOP_H

#endif // BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_PROPAGATABLEOP_H

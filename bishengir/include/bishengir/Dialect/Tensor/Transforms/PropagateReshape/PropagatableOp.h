//===- PropagatableOp.h - Header for propagatable ops ---------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

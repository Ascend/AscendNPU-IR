//===- PropagateReshape.h -------------------------------------------------===//
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

// Pattern to swap collapse and expand shape operations when beneficial
class SwapMemrefCollapseExpand
    : public mlir::OpRewritePattern<memref::ExpandShapeOp> {
public:
  explicit SwapMemrefCollapseExpand(MLIRContext *context)
      : OpRewritePattern<memref::ExpandShapeOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(memref::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace memref
} // namespace mlir

#endif // BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PROPAGATERESHAPE_H
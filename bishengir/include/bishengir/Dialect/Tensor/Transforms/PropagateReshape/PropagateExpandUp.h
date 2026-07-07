//===- PropagateExpandUp.h ------------------------------------------------===//
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
  explicit PropagateExpandUp(MLIRContext *context, bool forHIVM)
      : OpRewritePattern<tensor::ExpandShapeOp>(context, /*benefit=*/1) {
    options.forHIVM = forHIVM;
  }

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
//===- Transforms.cpp --------- Implementation of VFFusion Transforms --------------------------===//
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
#include "bishengir/Dialect/Analysis/VFFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir {
namespace analysis {
namespace {

/// Replace linalg.fill with tensor.empty when its user is linalg.reduce and
/// satisfies shouldUseTileReductionUsingForV2 pattern.
///
/// before:
///   %2 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>)
///        -> tensor<64xf32>
///   %reduced = linalg.reduce ins(%1 : tensor<64x128xf32>)
///        outs(%2 : tensor<64xf32>) dimensions = [1]
///
/// after:
///   %2 = tensor.empty() : tensor<64xf32>
///   %reduced = linalg.reduce ins(%1 : tensor<64x128xf32>)
///        outs(%2 : tensor<64xf32>) dimensions = [1]
class EmptifyReduceInitPattern : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (!hfusion::shouldUseTileReductionUsingForV2(reduceOp))
      return failure();

    if (reduceOp.getInits().empty())
      return failure();

    Value initValue = reduceOp.getInits()[0];

    Operation *initOp = initValue.getDefiningOp();
    if (!initOp || !hfusion::isFillOp(initOp))
      return failure();

    auto outType = dyn_cast<RankedTensorType>(initValue.getType());
    if (!outType)
      return failure();

    rewriter.setInsertionPoint(reduceOp);

    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        reduceOp.getLoc(), outType.getShape(), outType.getElementType());

    rewriter.modifyOpInPlace(reduceOp, [&]() {
      reduceOp.getInitsMutable()[0].assign(emptyOp.getResult());
    });

    if (initOp->use_empty())
      rewriter.eraseOp(initOp);

    return success();
  }
};

} // namespace

void populateEmptifyReduceInitPatterns(RewritePatternSet &patterns) {
  patterns.add<EmptifyReduceInitPattern>(patterns.getContext());
}

} // namespace analysis
} // namespace mlir
//===- FoldCollapseIntoAllocWithLoadPattern.h -----------------------------===//
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

#ifndef BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_FOLDCOLLAPSEINTOALLOCWITHLOADPATTERN_H
#define BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_FOLDCOLLAPSEINTOALLOCWITHLOADPATTERN_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensor {

// Folds a memref.collapse_shape into the alloc + load chain that feeds it.
//
// Matches an hivm load whose destination is a subview of an alloc that also
// feeds a collapse; the collapse among the alloc's users proves that every
// consumer wants the collapsed (matrix) form, so the chain — alloc, subview,
// and the load's source view — is rebuilt directly in that form and the
// collapse is erased. Chains without such a collapse (e.g. genuine batch
// loads) are never matched.
class FoldCollapseIntoAllocWithLoadPattern
    : public mlir::OpRewritePattern<hivm::LoadOp> {
public:
  explicit FoldCollapseIntoAllocWithLoadPattern(MLIRContext *context)
      : OpRewritePattern<hivm::LoadOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(hivm::LoadOp loadOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace tensor
} // namespace mlir

#endif // BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_FOLDCOLLAPSEINTOALLOCWITHLOADPATTERN_H

//===- NormalizeComparison.h -----------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
//
// This file defines templates for normalizing comparison operations.
// All compare op templates should be placed here.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECOMPARISON_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECOMPARISON_H

#include "bishengir/Transforms/Normalize/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir {

/// Template for normalizing cmp vne to not(cmp veq).
///
/// This template handles the common pattern where a "not equal" comparison
/// is transformed into a "not(equal)" comparison to handle NAN values correctly.
///
/// Example transformation:
///   y = cmp x, z {vne} -> i1
/// is normalized to:
///   tmp = cmp x, z {veq} -> i1
///   y = not tmp -> i1
///
/// The Traits class must provide:
///   - shouldNormalize(op): returns true if the op should be normalized
///   - createCmpOp(rewriter, loc, lhs, rhs, kind): creates comparison op
///   - createUnaryOp(rewriter, loc, input, output, kind): creates unary op
///
/// @tparam SourceOp The source operation type (e.g., hfusion::CompareOp or
/// hivm::VCmpOp)
/// @tparam Traits The traits class providing Dialect-specific implementations
template <typename SourceOp, typename Traits>
struct NormalizeCmpVneOpTemplate : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalize(op))
      return failure();

    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dpsOp) {
      op->emitError("NormalizeCmpVneOpTemplate: operation does not implement "
                    "DestinationStyleOpInterface");
      return failure();
    }

    auto inputs = dpsOp.getDpsInputs();
    Value output = dpsOp.getDpsInits()[0];
    Location loc = op->getLoc();

    Value veqResult = Traits::createCmpOp(rewriter, loc, inputs[0], inputs[1], CompareKind::EQ);

    Value vnotResult = Traits::createUnaryOp(rewriter, loc, veqResult, output, UnaryKind::Not);

    rewriter.replaceOp(op, vnotResult);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZECOMPARISON_H
//===- ResolvePropagation.h ---- Resolve conflict of scope propagation ----===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_RESOLVE_PROPAGATION_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_RESOLVE_PROPAGATION_H

#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace hivm {

/// Resolves a fully propagated boundary where an up-propagator consumes a
/// down-propagator directly. At that point propagation has identified two
/// incompatible requirements on the same value, so this pattern materializes
/// the required data movement and rewires the up-propagator to use the moved
/// value.
///
/// For example, after propagation a value may be marked as coming down from GM
/// and immediately needed by a local UB consumer:
///
/// ```
/// %gm_value = unrealized_conversion_cast %arg
///     {propagate_down, address_space = #hivm.address_space<gm>}
///     : tensor<16x16xf16> to tensor<16x16xf16>
/// %ub_value = unrealized_conversion_cast %gm_value
///     {propagate_up, address_space = #hivm.address_space<ub>}
///     : tensor<16x16xf16> to tensor<16x16xf16>
/// ```
///
/// This pattern turns such conflicts into `hivm.hir.load`, `hivm.hir.store`,
/// or a store/load pair for local-to-local transfers, then recreates the
/// remaining propagation markers around the inserted operation.
class ResolvePropagationPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
public:
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  explicit ResolvePropagationPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp upPropOp,
                                PatternRewriter &rewriter) const override;
};

/// Remove redundant propagation.
/// Remove the pattern such as
///
/// down
/// up
/// down
/// up
///
/// to
///
/// down
/// up
class RemoveRedundantPropagationPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
public:
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  explicit RemoveRedundantPropagationPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp upPropOp,
                                PatternRewriter &rewriter) const override;
};

class CloneMultipleAddressSpaceOperation : public RewritePattern {
public:
  explicit CloneMultipleAddressSpaceOperation(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_RESOLVE_PROPAGATION_H

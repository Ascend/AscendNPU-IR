//===- PropagateOp.h - Propagate pattern of InsertLoadStoreForMixCV -------===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_PROPAGATE_OP_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_PROPAGATE_OP_H

#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace hivm {

enum class PropagationStep {
  LOCAL,
  GM,
  UB,
  L1,
  ALL,
};

/// Propagation patterns push layout/address-space requirements through the IR
/// by rewriting temporary `unrealized_conversion_cast` marker ops. A typical
/// loop-carried case starts with the marker attached to only one side of the
/// loop:
///
/// ```
/// %arg_down = unrealized_conversion_cast %arg
///     {propagate_down, tcore_type = #hivm.tcore_type<VECTOR>,
///      address_space = #hivm.address_space<ub>}
///     : tensor<1x1x16x16xf16> to tensor<16x16xf16>
/// %r = scf.for %iv = %lb to %ub step %step
///     iter_args(%iter = %arg_down) -> (tensor<16x16xf16>) {
///   %mid = "hivm.hir.vexp"(%iter) : (tensor<16x16xf16>)
///       -> tensor<16x16xf16>
///   scf.yield %mid : tensor<16x16xf16>
/// }
/// ```
///
/// The propagation rewrite mirrors the marker across the region boundary so the
/// producer, block argument, yield operand, and loop result agree on where
/// load/store materialization or later layout conversion should happen.
///
/// `step` gates which core type is propagated in the current rewrite round:
/// step LOCAL handles mixed cube/vector markers, step UB handles vector markers, and
/// step L1 handles cube markers.
///
/// Rewrites a down-propagator from a value to each operation that consumes it.
/// For region-carrying ops this means walking from an operand into tied block
/// arguments, terminator operands, and results. For HIVM structured ops it also
/// propagates through DPS inputs/inits so later passes can insert the required
/// loads and stores at the operation boundary.
struct PropagateDownPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
public:
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  explicit PropagateDownPattern(MLIRContext *ctx, PropagationStep step)
      : OpRewritePattern(ctx, /*benefit=*/3), step(step) {}
      : OpRewritePattern(ctx, /*benefit=*/4), step(step) {}

private:
  LogicalResult matchAndRewrite(UnrealizedConversionCastOp propagateOp,
                                PatternRewriter &rewriter) const override;

  LogicalResult propagateDownForOp(scf::ForOp op, OpOperand &operand,
                                   UnrealizedConversionCastOp propagateOp,
                                   PatternRewriter &rewriter) const;

  LogicalResult propagateDownWhileOp(scf::WhileOp op, OpOperand &operand,
                                     UnrealizedConversionCastOp propagateOp,
                                     PatternRewriter &rewriter) const;

  LogicalResult propagateDownYieldOp(scf::YieldOp op, OpOperand &operand,
                                     UnrealizedConversionCastOp propagateOp,
                                     PatternRewriter &rewriter) const;

  LogicalResult propagateDownConditionOp(scf::ConditionOp op,
                                         OpOperand &operand,
                                         UnrealizedConversionCastOp propagateOp,
                                         PatternRewriter &rewriter) const;

  LogicalResult propagateDownReturnOp(scope::ReturnOp op, OpOperand &operand,
                                      UnrealizedConversionCastOp propagateOp,
                                      PatternRewriter &rewriter) const;

  LogicalResult propagateDownDmaOp(hivm::HIVMStructuredOp op,
                                   OpOperand &operand,
                                   UnrealizedConversionCastOp propagateOp,
                                   PatternRewriter &rewriter) const;

  PropagationStep step;
};

/// Rewrites an up-propagator from a value back to the operation or block
/// argument that produced it. This is the counterpart of
/// `PropagateDownPattern`: it walks from results into tied operands and region
/// terminators for `scf.for`, `scf.while`, `scf.if`, `scf.index_switch`, and
/// `scope.scope`, and mirrors requirements across HIVM structured op results.
struct PropagateUpPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
public:
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  explicit PropagateUpPattern(MLIRContext *ctx, PropagationStep step)
      : OpRewritePattern(ctx, /*benefit=*/2), step(step) {}
      : OpRewritePattern(ctx, /*benefit=*/3), step(step) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp propagateOp,
                                PatternRewriter &rewriter) const override;

private:
  LogicalResult propagateUpBlockArgument(BlockArgument blockArgument,
                                         UnrealizedConversionCastOp propagateOp,
                                         PatternRewriter &rewriter) const;

  LogicalResult propagateUpIfOp(scf::IfOp op, OpResult res,
                                UnrealizedConversionCastOp propagateOp,
                                PatternRewriter &rewriter) const;

  LogicalResult propagateUpIndexSwitchOp(scf::IndexSwitchOp op, OpResult res,
                                         UnrealizedConversionCastOp propagateOp,
                                         PatternRewriter &rewriter) const;

  LogicalResult propagateUpForOp(scf::ForOp op, OpResult res,
                                 UnrealizedConversionCastOp propagateOp,
                                 PatternRewriter &rewriter) const;

  LogicalResult propagateUpWhileOp(scf::WhileOp op, OpResult res,
                                   UnrealizedConversionCastOp propagateOp,
                                   PatternRewriter &rewriter) const;

  LogicalResult propagateUpScopeOp(scope::ScopeOp op, OpResult res,
                                   UnrealizedConversionCastOp propagateOp,
                                   PatternRewriter &rewriter) const;

  LogicalResult propagateUpDmaOp(hivm::HIVMStructuredOp op, OpResult res,
                                 UnrealizedConversionCastOp propagateOp,
                                 PatternRewriter &rewriter) const;

  PropagationStep step;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_PROPAGATE_OP_H

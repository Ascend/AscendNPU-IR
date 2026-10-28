//===- Utils.h - helpers for OptimizeTransposePlacement ---------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_OPTIMIZE_TRANSPOSE_PLACEMENT_UTILS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_OPTIMIZE_TRANSPOSE_PLACEMENT_UTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace hivm {
namespace otp {

/// The nearest ancestor module carrying `hacc.target` decides. An inner
/// module with no target inherits the outer one. No target at all is
/// membase.
bool isRegBaseScope(Operation *op);

/// annotation.mark and hivm.hir.debug do not consume a value. tensor.dim
/// does: its index names an axis, and sinking swaps that axis.
bool isIgnorableUser(Operation *user);
bool hasSingleRealUse(Value value);
bool isRank2Static(Type type);
FailureOr<RankedTensorType> swap2D(Type type);
bool isTransparentChainOp(Operation *op);
bool isSwapTranspose(VTransposeOp transpose);
bool isFreelyTransposableSource(Value value);
FailureOr<Value> getOnlyShapedInput(Operation *op);

Value createSwappedEmpty(IRRewriter &rewriter, Location loc, Value oldInit,
                         RankedTensorType newType);
/// Copy address-space and inserted-tensor attributes from `attrSrc`'s
/// defining op onto `dstEmpty`'s defining op when present.
void overlayBufferAttrs(Value dstEmpty, Value attrSrc);
/// Clone `op` at the swapped shape: `chainSrc` remaps to `newChainSrc`,
/// other shaped DPS inputs through `siblings`, which must cover them all.
Value cloneWithSwappedLayout(IRRewriter &rewriter, Operation *op,
                             Value chainSrc, Value newChainSrc,
                             const DenseMap<Value, Value> &siblings);
/// Drop `op` when nothing uses it. Ignorable users are erased, or, when
/// `replacement` is set, cloned onto that value so a mark or debug op stays
/// with the rewritten result and still dominates its operand.
void eraseIfDead(IRRewriter &rewriter, Operation *op, Value replacement = {});

/// Erase trivially dead ops under `root`, skipping `keep` and mark/debug
/// ops. `keep` carries the ops already dead before the pass ran, so only
/// pass-orphaned ops are swept — no blanket DCE of unrelated code.
void sweepDeadOps(Operation *root, const SmallPtrSetImpl<Operation *> &keep);

} // namespace otp
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_OPTIMIZE_TRANSPOSE_PLACEMENT_UTILS_H

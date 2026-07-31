//===- NormalizeRelu.cpp ----------------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::hivm {
namespace {

/// Check if a value is a constant zero (supports float and integer types).
static bool isConstZero(Value v) {
  if (!v)
    return false;

  auto type = getElementTypeOrSelf(v);
  if (isa<FloatType>(type)) {
    if (matchPattern(v, m_PosZeroFloat()) ||
        matchPattern(v, m_NegZeroFloat())) {
      return true;
    }
  } else if (type.isIntOrIndex()) {
    if (matchPattern(v, m_Zero())) {
      return true;
    }
  }

  // Check if it's produced by a fill operation with zero
  auto defineOp = v.getDefiningOp();
  if (!defineOp)
    return false;

  if (auto fillOp = dyn_cast<linalg::FillOp>(defineOp)) {
    // Check if the fill value is zero
    return isConstZero(fillOp.getInputs()[0]);
  }

  return false;
}

/// Check if a value is produced (directly or through scf.for) by an mmad op.
/// This traces through scf.for yield/init-arg chains to handle iterative
/// matmul accumulation patterns.
bool isProducedByMmad(Value value) {
  SmallPtrSet<Value, 16> visited;
  SmallVector<Value, 8> worklist;
  worklist.push_back(value);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;

    auto *defOp = v.getDefiningOp();
    if (!defOp)
      continue;

    if (auto matmulOp = dyn_cast_if_present<hivm::LocalMatmulLikeOpInterface>(defOp))
      return true;

    // Trace through scf.for: if value is a for-loop result, check the
    // corresponding yielded value and init arg.
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      auto result = cast<OpResult>(v);
      unsigned idx = result.getResultNumber();
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      worklist.push_back(yieldOp.getOperand(idx));
      if (idx < forOp.getInitArgs().size())
        worklist.push_back(forOp.getInitArgs()[idx]);
      continue;
    }

    // Trace through other ops (e.g., casts) by checking their operands
    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

/// Convert hivm.hir.vmax(C, zero) or hivm.hir.vmax(zero, C) to
/// hivm.hir.vrelu(C). This is limited to the mmad + vmax pattern to enable
/// downstream fixpipe fusion in hivm-inline-fixpipe pass.
struct VMaxToVReluPattern : public OpRewritePattern<hivm::VMaxOp> {
public:
  using OpRewritePattern<hivm::VMaxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::VMaxOp vmaxOp,
                                PatternRewriter &rewriter) const final {
    // Get the two input operands
    auto srcOperands = vmaxOp.getSrc();
    if (srcOperands.size() != 2)
      return failure();

    Value lhs = srcOperands[0];
    Value rhs = srcOperands[1];

    // Check element type is supported by vrelu (F16, F32, I32)
    Type elemType = getElementTypeOrSelf(lhs.getType());
    if (!elemType.isF16() && !elemType.isF32() &&
        !(elemType.isInteger(32) && vmaxOp.getIsSigned()))
      return failure();

    // Detect which operand is zero
    Value nonZeroOperand = nullptr;
    if (isConstZero(rhs)) {
      nonZeroOperand = lhs;
    } else if (isConstZero(lhs)) {
      nonZeroOperand = rhs;
    } else {
      return failure();
    }

    // Only convert when the non-zero operand is produced by an mmad op.
    // This targets the mmad + vmax + fixpipe pattern specifically.
    if (!isProducedByMmad(nonZeroOperand))
      return failure();

    // Propagate broadcast/transpose from the vmax op
    auto transpose = vmaxOp.getTransposeAttr();
    auto broadcast = vmaxOp.getBroadcastAttr();

    // Create vrelu op to replace vmax
    auto vreluOp = rewriter.create<hivm::VReluOp>(
        vmaxOp.getLoc(), vmaxOp.getResultTypes(),
        ValueRange{nonZeroOperand}, vmaxOp.getDst(),
        /*temp_buffer=*/Value(), transpose, broadcast);
    rewriter.replaceOp(vmaxOp, vreluOp.getResults());
    return success();
  }
};

} // namespace

void populateNormalizeReluPatterns(RewritePatternSet &patterns) {
  patterns.add<VMaxToVReluPattern>(patterns.getContext());
}

} // namespace mlir::hivm

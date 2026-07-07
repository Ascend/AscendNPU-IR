//===- FlattenOps.cpp ---- Flatten HIVM Ops -------------------------------===//
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
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <numeric>
#define DEBUG_TYPE "hivm-flatten-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir {
#define GEN_PASS_DEF_HIVMFLATTENOPS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::utils;

namespace {
class FlattenOpsRewritePattern
    : public OpInterfaceRewritePattern<HIVMStructuredOp> {
  using OpInterfaceRewritePattern<HIVMStructuredOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(HIVMStructuredOp op,
                                PatternRewriter &rewriter) const override {
    // Skip flattening for LoadOp with non-zero left_padding_num.
    // LoadOps with non-zero left_padding_num are padded loads whose
    // collapsed shape may lose the padding offset info.
    if (isa<hivm::LoadOp>(op.getOperation())) {
      auto loadOp = cast<hivm::LoadOp>(op.getOperation());
      auto leftPad = loadOp.getLeftPaddingNum();
      if (leftPad) {
        auto constIdx = leftPad.getDefiningOp<arith::ConstantIndexOp>();
        if (!constIdx || constIdx.value() != 0) {
          // Exception: if dst comes from a subview whose last offset is 0,
          // the padding offset is already captured by the subview, so
          // flattening is safe.
          bool shouldSkip = true;
          if (auto subviewOp =
                  loadOp.getDst().getDefiningOp<memref::SubViewOp>()) {
            auto offsets = subviewOp.getMixedOffsets();
            if (!offsets.empty()) {
              if (auto constOff = getConstantIntValue(offsets.back())) {
                shouldSkip = (*constOff != 0);
              }
            }
          }
          if (shouldSkip) {
            return rewriter.notifyMatchFailure(
                op, "LoadOp with non-zero left_padding_num");
          }
        }
      }
    }
    // Cast to flatten interface so it calls the implementation for it
    auto res = cast<FlattenInterface>(op.getOperation())
                   .getFlattened(FlattenOptions());
    if (failed(res))
      return rewriter.notifyMatchFailure(op, "Operation cannot be handled");
    if (res->isIdentityCollapse())
      return rewriter.notifyMatchFailure(op, "Identity reassociation");
    auto inputReassociation = res->getInputReassociation();
    auto initReassociation = res->getInitReassociation();
    IRMapping newValueMapping;
    for (OpOperand &operand : op->getOpOperands()) {
      const auto &reassociation =
          (op.isDpsInput(&operand) ? inputReassociation : initReassociation);
      if (isa<MemRefType>(operand.get().getType())) {
        auto collapsedOperand = rewriter.create<memref::CollapseShapeOp>(
            op.getLoc(), operand.get(), reassociation);
        newValueMapping.map(operand.get(), collapsedOperand);
      }
    }
    Operation *clonedOperation =
        rewriter.clone(*op.getOperation(), newValueMapping);
    // Collapse all of the operand based on the reassociation
    rewriter.modifyOpInPlace(clonedOperation, [&]() {
      cast<FlattenInterface>(clonedOperation)
          .adjustTargetDimensions(rewriter, res.value());
    });
    rewriter.replaceOp(op, clonedOperation);
    LDBG(*clonedOperation->getParentOp());
    LDBG((clonedOperation->getParentOp() ? *(clonedOperation->getParentOp()) : *clonedOperation));
    return failure();
  }
};
struct FlattenOpsPass : public impl::HIVMFlattenOpsBase<FlattenOpsPass> {
public:
  void runOnOperation() override;
};
} // namespace

void FlattenOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.add<FlattenOpsRewritePattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createFlattenOpsPass() {
  return std::make_unique<FlattenOpsPass>();
}

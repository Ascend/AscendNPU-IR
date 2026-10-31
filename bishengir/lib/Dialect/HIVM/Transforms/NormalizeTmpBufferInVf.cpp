//===-- NormalizeTmpBufferInVf.cpp - Normalize tmp buffers in VF -*- C++
//-*-===//
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
//
// This pass normalizes temporary tensor buffers inside vector functions.
//
// A nested `scf.for` that read-modify-writes a slice of an outer tensor (for
// example a `hivm.vector_function` argument) creates a bufferization conflict:
// the written slice is not part of the enclosing loop's iter_args, so
// one-shot-bufferize has to insert a `memref.copy`.
//
// When such an inner loop is found, the pass promotes the slice source into a
// fresh iter_arg of the parent loop. The parent loop gets the matching
// `tensor.insert_slice` at its tail, so the inner loop reads and writes the
// slice through the parent's iter_arg instead of an outer tensor. Enclosing
// loops further out are handled by re-matching the parent loop.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZETMPBUFFERINVF
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

Block *getDefineBlock(Value value) {
  if (auto bbArg = dyn_cast<BlockArgument>(value))
    return bbArg.getOwner();
  return value.getDefiningOp()->getBlock();
}

/// Promote the slice written by a loop nested in another loop into a new
/// iter_arg of the parent loop.
///
/// The pattern stops matching once the slice source is the parent's own
/// iter_arg: it is then defined in the same block as the nested loop.
struct PromoteTmpBufferInVfPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    bool modified = false;
    for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
      // Only tensors are promoted; skip vector/memref iter_args.
      if (!isa<TensorType>(initArg.getType()))
        continue;

      Value root = initArg;
      auto extractOp = root.getDefiningOp<tensor::ExtractSliceOp>();
      if (extractOp) {
        if (extractOp->getBlock() != forOp->getBlock()) {
          extractOp->dump();
          llvm::report_fatal_error(
              "tensor.extract_slice must be in the same block as scf.for");
        }
        root = extractOp.getSource();
      }

      if (getDefineBlock(root) == forOp->getBlock())
        continue;

      auto parentLoop = dyn_cast<scf::ForOp>(forOp->getParentOp());
      if (!parentLoop)
        llvm::report_fatal_error("if forOp's parentOp is func.func, root must "
                                 "be defined in same block as forOp");
      if (llvm::is_contained(parentLoop.getInitArgs(), root)) {
        llvm::dbgs() << "root : " << root << "\n";
        llvm::dbgs() << "parent loop: " << *parentLoop << "\n";
        llvm::report_fatal_error("root  must not be parent loop's iter_arg");
      }
      // Add a new iter_arg to the parent loop. It starts from the promoted
      // buffer and yields it updated with the result of `forOp`.
      Value forResult = forOp.getResult(idx);
      auto newYieldFn =
          [&](OpBuilder &b, Location loc,
              ArrayRef<BlockArgument> newIterArgs) -> SmallVector<Value> {
        if (!extractOp)
          return {forResult};
        auto insertOp = b.create<tensor::InsertSliceOp>(
            loc, forResult, newIterArgs.front(), extractOp.getMixedOffsets(),
            extractOp.getMixedSizes(), extractOp.getMixedStrides());
        return {insertOp.getResult()};
      };
      auto newLoop = parentLoop.replaceWithAdditionalYields(
          rewriter, /*newInitOperands=*/root,
          /*replaceInitOperandUsesInLoop=*/true, newYieldFn);
      if (failed(newLoop)) {
        parentLoop.dump();
        llvm::report_fatal_error("failed to add new iter_arg to parent loop");
      }
      modified = true;
    }
    return success(modified);
  }
};

struct NormalizeTmpBufferInVfPass
    : public mlir::impl::NormalizeTmpBufferInVfBase<
          NormalizeTmpBufferInVfPass> {
  void runOnOperation() override;
};

} // namespace

void NormalizeTmpBufferInVfPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!hivm::isVF(funcOp) || funcOp.getBody().empty())
    return;
  RewritePatternSet patterns(&getContext());
  patterns.add<PromoteTmpBufferInVfPattern>(&getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createNormalizeTmpBufferInVfPass() {
  return std::make_unique<NormalizeTmpBufferInVfPass>();
}

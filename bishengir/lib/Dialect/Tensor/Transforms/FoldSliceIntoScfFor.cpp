//===- FoldSliceIntoScfFor.cpp ----------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FOLDSLICEINTOSCFFOR
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Folds an extract_slice/insert_slice pair that surrounds an scf.for loop
/// into the loop body:
///
///   %slice = tensor.extract_slice %full[o] [s] [st]
///   %r = scf.for ... iter_args(%arg = %slice) -> (slice_ty) {
///     ... uses of %arg ...
///     scf.yield %yielded
///   }
///   %res = tensor.insert_slice %r into %full[o] [s] [st]
///
/// becomes:
///
///   %r = scf.for ... iter_args(%arg = %full) -> (full_ty) {
///     %slice = tensor.extract_slice %arg[o] [s] [st]
///     ... uses of %slice ...
///     %updated = tensor.insert_slice %yielded into %arg[o] [s] [st]
///     scf.yield %updated
///   }
///
/// and uses of %res are replaced with %r.
struct FoldSliceIntoScfForPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    for (unsigned idx = 0; idx < forOp.getRegionIterArgs().size(); idx++) {
      if (succeeded(foldOneIterArg(forOp, idx, rewriter)))
        return success();
    }
    return failure();
  }

  LogicalResult foldOneIterArg(scf::ForOp forOp, unsigned idx,
                               PatternRewriter &rewriter) const {
    // The init value must be an extract_slice with no other uses.
    auto outerExtract =
        forOp.getInitArgs()[idx].getDefiningOp<tensor::ExtractSliceOp>();
    if (!outerExtract || !outerExtract->hasOneUse())
      return failure();
    Value fullTensor = outerExtract.getSource();
    Type fullType = fullTensor.getType();

    // The loop result must be written back into the same tensor, with the
    // same offsets, sizes and strides as the extract.
    tensor::InsertSliceOp insertOp = nullptr;
    for (auto *user : forOp.getResult(idx).getUsers()) {
      auto candidate = dyn_cast<tensor::InsertSliceOp>(user);
      if (candidate && candidate.getDest() == fullTensor &&
          candidate.getSource() == forOp.getResult(idx) &&
          candidate.getMixedOffsets() == outerExtract.getMixedOffsets() &&
          candidate.getMixedSizes() == outerExtract.getMixedSizes() &&
          candidate.getMixedStrides() == outerExtract.getMixedStrides()) {
        insertOp = candidate;
        break;
      }
    }
    if (!insertOp)
      return failure();

    // Widen the iter arg, its init value and the loop result to the full
    // tensor type.
    rewriter.modifyOpInPlace(forOp, [&]() {
      forOp.getInitArgsMutable()[idx].assign(fullTensor);
      forOp.getRegionIterArg(idx).setType(fullType);
      forOp->getResult(idx).setType(fullType);
    });

    // Clone the extract_slice after the loop for the remaining users of the
    // loop result.
    auto *newExtract = rewriter.clone(*outerExtract);
    rewriter.moveOpAfter(newExtract, forOp);
    rewriter.modifyOpInPlace(newExtract, [&]() {
      newExtract->getOpOperand(0).set(forOp.getResult(idx));
    });
    rewriter.replaceAllUsesExcept(forOp.getResult(idx),
                                  newExtract->getResult(0), newExtract);

    // Replace all uses of the insert_slice with the loop result.
    rewriter.replaceAllUsesWith(insertOp.getResult(), forOp.getResult(idx));

    // Move the outer extract to the beginning of the loop body.
    Value iterArg = forOp.getRegionIterArg(idx);
    rewriter.moveOpBefore(outerExtract, &forOp.getBody()->front());
    rewriter.modifyOpInPlace(outerExtract, [&]() {
      outerExtract.getSourceMutable().assign(iterArg);
    });
    rewriter.replaceAllUsesExcept(iterArg, outerExtract.getResult(),
                                  outerExtract);

    // Move the outer insert_slice to the end of the loop body.
    scf::YieldOp yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    rewriter.moveOpBefore(insertOp, yieldOp);
    Value yielded = yieldOp->getOperand(idx);
    rewriter.modifyOpInPlace(insertOp, [&]() {
      insertOp.getSourceMutable().assign(yielded);
      insertOp.getDestMutable().assign(iterArg);
    });
    rewriter.modifyOpInPlace(
        yieldOp, [&]() { yieldOp.setOperand(idx, insertOp.getResult()); });
    return success();
  }
};

struct FoldSliceIntoScfForPass
    : public impl::FoldSliceIntoScfForBase<FoldSliceIntoScfForPass> {
  void runOnOperation() override;
};

} // namespace

void FoldSliceIntoScfForPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<FoldSliceIntoScfForPattern>(patterns.getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::tensor::createFoldSliceIntoScfForPass() {
  return std::make_unique<FoldSliceIntoScfForPass>();
}

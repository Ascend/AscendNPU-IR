//===- CloneTensorEmpty.cpp ---- Clone Tensor Empty Pass ------------------===//
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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CLONETENSOREMPTY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
void CloneNewTensorEmpty(HIVMStructuredOp op, PatternRewriter &rewriter) {
  for (Value dst : op.getDpsInits()) {
    auto DstDefiningOp = dst.getDefiningOp();
    if (!DstDefiningOp)
      continue;
    if (!isa<TensorType>(dst.getType()))
      continue;
    if (isa<tensor::EmptyOp>(DstDefiningOp)) {
      rewriter.setInsertionPoint(op);
      auto clonedOp = rewriter.clone(*DstDefiningOp);
      op->replaceUsesOfWith(dst, clonedOp->getResult(0));
    }
  }
}

template <typename OpTy>
struct CloneTensorEmptyHIVMStructuredOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isa<hivm::HIVMStructuredOp>(op.getOperation())) {
      return failure();
    }
    CloneNewTensorEmpty(op, rewriter);
    return success();
  }
};

struct CloneTensorEmptySCFForPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<unsigned> emptyInitIndex;
    for (auto [idx, init] : llvm::enumerate(op.getInitArgs())) {
      auto initDefOp = init.getDefiningOp();
      if (initDefOp && isa<tensor::EmptyOp>(initDefOp)) {
        emptyInitIndex.push_back(idx);
      }
    }

    if (emptyInitIndex.empty()) {
      return failure();
    }

    auto mutableInits = op.getInitArgsMutable();
    rewriter.setInsertionPoint(op);
    for (auto idx : emptyInitIndex) {
      auto &mtEmptyInit = mutableInits[idx];
      auto emptyDefOp = mtEmptyInit.get().getDefiningOp();
      if (emptyDefOp == nullptr)
        llvm::report_fatal_error("EmptyOp is not found");
      auto clonedOp = rewriter.clone(*emptyDefOp);
      mutableInits[idx].assign(clonedOp->getResult(0));
    }

    return success();
  }
};

struct CloneTensorInsert : public OpRewritePattern<tensor::InsertOp> {
  using OpRewritePattern<tensor::InsertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::InsertOp op,
                                PatternRewriter &rewriter) const override {

    Value dst = op.getDest();
    // run only dst with tensor.empty
    auto emptyOp = dst.getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp)
      return failure();

    // insert empty tensor just before use.
    rewriter.setInsertionPoint(op);
    Operation *clonedEmpty = rewriter.clone(*emptyOp);
    op->replaceUsesOfWith(dst, clonedEmpty->getResult(0));

    return success();
  }
};

/// This pass Output clones to different empty tensors based on hivmOp.
struct CloneTensorEmptyPass
    : public impl::CloneTensorEmptyBase<CloneTensorEmptyPass> {
public:
  void runOnOperation() override;
};

template <typename OpType>
void registerOne(RewritePatternSet &patterns) {
  patterns.add<CloneTensorEmptyHIVMStructuredOpPattern<OpType>>(
      patterns.getContext());
}

/// Variadic helper function.
template <typename... OpTypes>
void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populateCloneTensorEmptyPattern(RewritePatternSet &patterns) {
  patterns.add<CloneTensorEmptyHIVMStructuredOpPattern<hivm::CopyOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::LoadOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::StoreOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::FixpipeOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::MmadL1Op>,
               CloneTensorInsert, CloneTensorEmptySCFForPattern>(
      patterns.getContext());
  registerAll<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >(patterns);
}

void CloneTensorEmptyPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  populateCloneTensorEmptyPattern(patterns);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createCloneTensorEmptyPass() {
  return std::make_unique<CloneTensorEmptyPass>();
}

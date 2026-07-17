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
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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
void copyAnnotationMark(Value src, Value dst, PatternRewriter &rewriter) {
  auto *dstDefiningOp = dst.getDefiningOp();
  if (!dstDefiningOp)
    return;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(dstDefiningOp);
  for (Operation *user : llvm::make_early_inc_range(src.getUsers())) {
    auto markOp = dyn_cast<annotation::MarkOp>(user);
    if (!markOp || markOp.getSrc() != src)
      continue;
    // Only copy markOp that contains buffer_size_in_byte attribute
    if (!markOp->hasAttr(hivm::kBufferSizeInByteAttr))
      continue;
    auto clonedMarkOp = rewriter.create<annotation::MarkOp>(
        markOp.getLoc(), dst, markOp.getValues(), markOp.getKeysAttr());
    for (NamedAttribute attr : markOp->getAttrs())
      clonedMarkOp->setAttr(attr.getName(), attr.getValue());
  }
}

void cloneNewTensorEmpty(HIVMStructuredOp op, PatternRewriter &rewriter) {
  for (Value dst : op.getDpsInits()) {
    auto * dstDefiningOp = dst.getDefiningOp();
    if (!dstDefiningOp)
      continue;
    if (!isa<TensorType>(dst.getType()))
      continue;
    if (isa<tensor::EmptyOp>(dstDefiningOp)) {
      rewriter.setInsertionPoint(op);
      auto * clonedOp = rewriter.clone(*dstDefiningOp);
      copyAnnotationMark(dst, clonedOp->getResult(0), rewriter);
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
    cloneNewTensorEmpty(op, rewriter);
    return success();
  }
};

template <typename LoopOp>
struct CloneTensorEmptyLoopPattern : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<unsigned> emptyInitIndex;
    for (auto [idx, init] : llvm::enumerate(op.getInits())) {
      auto initDefOp = init.getDefiningOp();
      if (initDefOp && isa<tensor::EmptyOp>(initDefOp)) {
        emptyInitIndex.push_back(idx);
      }
    }

    if (emptyInitIndex.empty()) {
      return failure();
    }

    auto mutableInits = op.getInitsMutable();
    rewriter.setInsertionPoint(op);
    for (auto idx : emptyInitIndex) {
      auto &mtEmptyInit = mutableInits[idx];
      auto emptyDefOp = mtEmptyInit.get().getDefiningOp();
      if (emptyDefOp == nullptr)
        llvm::report_fatal_error("EmptyOp is not found");
      auto clonedOp = rewriter.clone(*emptyDefOp);
      copyAnnotationMark(emptyDefOp->getResult(0), clonedOp->getResult(0),
                         rewriter);
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

// Clone empty tensor operands for VF calls / indirect store so multi-buffer
// can assign distinct buffers per use site (Ascend950).
template <typename OpTy>
struct CloneTensorEmptyOperationPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    if (isa<mlir::func::CallOp>(op) && !isVFCall(operation))
      return failure();
    rewriter.setInsertionPoint(operation);
    bool clonedAny = false;
    for (size_t idx = 0; idx < operation->getNumOperands(); ++idx) {
      auto &operand = operation->getOpOperands()[idx];
      Value value = operand.get();
      auto definingOp = value.getDefiningOp();
      if (!isa_and_nonnull<mlir::tensor::EmptyOp>(definingOp))
        continue;
      if (!isa<TensorType>(value.getType()))
        continue;
      auto clonedProducer = rewriter.clone(*definingOp);
      copyAnnotationMark(definingOp->getResult(0), clonedProducer->getResult(0),
                         rewriter);
      rewriter.modifyOpInPlace(
          operation, [&]() { operand.set(clonedProducer->getResult(0)); });
      clonedAny = true;
    }
    if (!clonedAny)
      return failure();
    auto clonedConsumer = rewriter.clone(*operation);
    rewriter.replaceOp(operation, clonedConsumer->getResults());
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

void populateCloneTensorEmptyPattern(RewritePatternSet &patterns,
                                     bool isAscend950) {
  patterns.add<CloneTensorEmptyHIVMStructuredOpPattern<hivm::CopyOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::LoadOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::StoreOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::FixpipeOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::MmadL1Op>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::Conv1DL1Op>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::Conv2DL1Op>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::Conv3DL1Op>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::CustomOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::CustomMacroOp>,
               CloneTensorInsert, CloneTensorEmptyLoopPattern<scf::WhileOp>,
               CloneTensorEmptyLoopPattern<scf::ForOp>>(patterns.getContext());
  if (isAscend950) {
    patterns.add<CloneTensorEmptyOperationPattern<func::CallOp>,
                 CloneTensorEmptyOperationPattern<hivm::IndirectStoreOp>>(
        patterns.getContext());
  }
  registerAll<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >(patterns);
}

void CloneTensorEmptyPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  bool archIs950 = false;
  if (auto moduleOp = funcOp->getParentOfType<ModuleOp>())
    archIs950 = hacc::utils::isAscend950(moduleOp);

  RewritePatternSet patterns(&getContext());
  populateCloneTensorEmptyPattern(patterns, archIs950);
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  (void)applyPatternsGreedily(funcOp, std::move(patterns), config);
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createCloneTensorEmptyPass() {
  return std::make_unique<CloneTensorEmptyPass>();
}

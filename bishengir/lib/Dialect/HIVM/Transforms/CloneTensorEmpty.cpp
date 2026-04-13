//===- CloneTensorEmpty.cpp ---- Clone Tensor Empty Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
#define GEN_PASS_DEF_CLONETENSOREMPTY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
void CloneNewTensorEmpty(HIVMStructuredOp op, PatternRewriter &rewriter) {
  Operation *operation = op.getOperation();
  rewriter.setInsertionPoint(operation);
  for (size_t idx = 0; idx < operation->getNumOperands(); ++idx) {
    auto &operand = operation->getOpOperands()[idx];
    Value dst = operand.get();
    if (!llvm::is_contained(op.getDpsInits(), dst))
      continue;
    auto DstDefiningOp = dst.getDefiningOp();
    if (!isa_and_nonnull<mlir::tensor::EmptyOp>(DstDefiningOp))
      continue;
    if (!isa<TensorType>(dst.getType()))
      continue;
    if (!DstDefiningOp) {
      continue;
    }
    auto clonedProducer = rewriter.clone(*DstDefiningOp);
    rewriter.modifyOpInPlace(
        operation, [&]() { operand.set(clonedProducer->getResult(0)); });
  }
  auto clonedConsumer = rewriter.clone(*operation);
  rewriter.replaceOp(operation, clonedConsumer->getResults());
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

namespace {
// empty tensor sinking support simd-vf func call op & embedding gather op,
// inorder to help open multi-buffer
template <typename OpTy>
struct CloneTensorEmptyOperationPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    // Function call only support vf only if it is a function Call.
    // EmbeddingGatherOp does not need this check.
    if (isa<mlir::func::CallOp>(op) && !isVFCall(operation))
      return failure();
    rewriter.setInsertionPoint(operation);
    for (size_t idx = 0; idx < operation->getNumOperands(); ++idx) {
      auto &operand = operation->getOpOperands()[idx];
      Value value = operand.get();
      auto definingOp = value.getDefiningOp();
      if (!isa_and_nonnull<mlir::tensor::EmptyOp>(definingOp))
        continue;
      if (!isa<TensorType>(value.getType()))
        continue;
      // clone & modify the producer empty tensor operation
      auto clonedProducer = rewriter.clone(*definingOp);
      rewriter.modifyOpInPlace(
          operation, [&]() { operand.set(clonedProducer->getResult(0)); });
    }
    // clone & replace the consumer operation.
    auto clonedConsumer = rewriter.clone(*operation);
    rewriter.replaceOp(operation, clonedConsumer->getResults());
    return success();
  }
};
} // namespace

namespace {
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
} // namespace

/// This pass Output clones to different empty tensors based on hivmOp.
struct CloneTensorEmptyPass
    : public impl::CloneTensorEmptyBase<CloneTensorEmptyPass> {
  explicit CloneTensorEmptyPass() : CloneTensorEmptyBase() {}

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
                                     bool isSupportExtra) {
  patterns.add<CloneTensorEmptyHIVMStructuredOpPattern<hivm::CopyOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::LoadOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::StoreOp>,
               CloneTensorEmptyHIVMStructuredOpPattern<hivm::MmadL1Op>,
               CloneTensorEmptySCFForPattern>(patterns.getContext());
  // only 950 requires empty tensor of this two type of op sinking.
  if (isSupportExtra) {
    patterns.add<CloneTensorEmptyOperationPattern<hivm::EmbeddingGatherOp>,
                 CloneTensorEmptyOperationPattern<hivm::IndirectLoadOp>,
                 CloneTensorEmptyOperationPattern<hivm::IndirectStoreOp>,
                 CloneTensorEmptyOperationPattern<func::CallOp>>(
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

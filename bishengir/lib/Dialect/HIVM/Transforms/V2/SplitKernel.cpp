//===- V2SplitKernel.cpp - MIX split --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memref-IR split: clone the MIX into AIC/AIV and drop the other core's ops.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/SplitKernelUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_V2SPLITKERNEL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-v2-split-kernel"

using namespace mlir;

namespace {

static FailureOr<hivm::TCoreType> getSplitCoreType(Operation *op) {
  if (op->hasAttr(hivm::VectorFunctionAttr::name))
    return hivm::TCoreType::VECTOR;
  return hivm::getCoreType(op);
}

/// Freeze use-dependent inference before either clone loses consumers.
static LogicalResult resolveCoreTypes(func::FuncOp func) {
  WalkResult result = func.walk([&](Operation *op) {
    if (op == func || op->hasTrait<OpTrait::IsTerminator>())
      return WalkResult::advance();
    auto coreType = getSplitCoreType(op);
    if (failed(coreType))
      return WalkResult::interrupt();
    if (isa<hivm::LoadOp, hivm::DebugOp, hivm::CustomOp>(op) ||
        op->hasAttr(hivm::VectorFunctionAttr::name))
      op->setAttr(hivm::TCoreTypeAttr::name,
                  hivm::TCoreTypeAttr::get(func.getContext(), *coreType));
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

/// A shared region can still carry buffer descriptors and scalar loop state
/// after its core-local effects are erased. A vector function is indivisible.
static bool isSharedRegion(Operation *op) {
  return isa<RegionBranchOpInterface>(op) &&
         !op->hasAttr(hivm::VectorFunctionAttr::name);
}

static FailureOr<bool> isFiltered(Operation *op, hivm::TCoreType filteredCore) {
  auto coreType = getSplitCoreType(op);
  if (failed(coreType))
    return failure();
  if (*coreType != hivm::TCoreType::CUBE_OR_VECTOR)
    return *coreType == filteredCore;
  if (isa<RegionBranchOpInterface, annotation::MarkOp>(op) ||
      isMemoryEffectFree(op) || hasSingleEffect<MemoryEffects::Allocate>(op))
    return false;

  for (Operation *parent = op->getParentOp(); !isa<func::FuncOp>(parent);
       parent = parent->getParentOp()) {
    auto parentCore = getSplitCoreType(parent);
    if (failed(parentCore))
      return failure();
    if (*parentCore != hivm::TCoreType::CUBE_OR_VECTOR)
      return *parentCore == filteredCore;
  }
  return false;
}

/// Drop marks that would otherwise keep an unused buffer allocation alive.
struct EraseUnusedBufferMarks : OpRewritePattern<annotation::MarkOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(annotation::MarkOp mark,
                                PatternRewriter &rewriter) const override {
    Value src = mark.getSrc();
    Operation *def = src.getDefiningOp();
    if (!isa<BaseMemRefType>(src.getType()) || !def ||
        !wouldOpBeTriviallyDead(def) ||
        !llvm::all_of(src.getUsers(), [&](Operation *user) {
          auto otherMark = dyn_cast<annotation::MarkOp>(user);
          return otherMark && otherMark.getSrc() == src;
        }))
      return failure();
    rewriter.eraseOp(mark);
    return success();
  }
};

struct FilterCoreOps : RewritePattern {
  FilterCoreOps(MLIRContext *ctx, hivm::TCoreType filteredCore)
      : RewritePattern(MatchAnyOpTypeTag(), 1, ctx),
        filteredCore(filteredCore) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (isa<func::FuncOp>(op) || op->hasTrait<OpTrait::IsTerminator>())
      return failure();

    auto filtered = isFiltered(op, filteredCore);
    if (failed(filtered) || !*filtered || !op->use_empty() ||
        isSharedRegion(op))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }

  hivm::TCoreType filteredCore;
};

/// Remaining filtered ops must be shared control flow, without live vectors
/// or tensors. Clear stale ownership while preserving descriptor/scalar state.
static LogicalResult validateRemainingCoreOps(func::FuncOp func,
                                              hivm::TCoreType filteredCore) {
  WalkResult result = func.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (op == func || op->hasTrait<OpTrait::IsTerminator>())
      return WalkResult::advance();
    auto filtered = isFiltered(op, filteredCore);
    if (failed(filtered))
      return WalkResult::interrupt();
    if (!*filtered)
      return WalkResult::advance();
    if (!isSharedRegion(op) || llvm::any_of(op->getResults(), [](Value result) {
          return !result.use_empty() && isa<ShapedType>(result.getType()) &&
                 !isa<BaseMemRefType>(result.getType());
        })) {
      op->emitError("filtered operation has live results on the other core");
      return WalkResult::interrupt();
    }
    op->removeAttr(hivm::TCoreTypeAttr::name);
    op->removeAttr(hivm::kPipelinedLoopCoreTypeAttrName);
    op->removeAttr(hivm::kCubeOnlyAttrName);
    op->removeAttr(hivm::kVecOnlyAttrName);
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

static LogicalResult filterCoreOps(func::FuncOp func,
                                   hivm::TCoreType filteredCore) {
  MLIRContext *ctx = func.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<EraseUnusedBufferMarks>(ctx);
  patterns.add<FilterCoreOps>(ctx, filteredCore);
  scope::ScopeOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  scf::IfOp::getCanonicalizationPatterns(patterns, ctx);
  scf::WhileOp::getCanonicalizationPatterns(patterns, ctx);
  if (failed(applyPatternsGreedily(func, std::move(patterns))))
    return failure();
  return validateRemainingCoreOps(func, filteredCore);
}

static bool isMixFunction(func::FuncOp func) {
  if (hacc::utils::isHost(func) || func.isExternal())
    return false;
  auto coreType = hivm::queryFuncCoreType(func);
  return coreType && *coreType == hivm::TFuncCoreType::MIX &&
         !func->hasAttr(hivm::TPartOfMixAttr::name);
}

static LogicalResult splitMixKernel(func::FuncOp func) {
  if (failed(resolveCoreTypes(func)) || failed(hivm::verifyCoreRatio(func)))
    return failure();

  auto ratio = hivm::getCoreRatioAttr(func);
  if (hacc::utils::isAscend950(func->getParentOfType<ModuleOp>()) &&
      hivm::isSingleCoreRatio(ratio)) {
    bool vectorOnly = ratio.getCube() == 0;
    if (failed(filterCoreOps(func, vectorOnly ? hivm::TCoreType::CUBE
                                              : hivm::TCoreType::VECTOR)))
      return failure();
    func->setAttr(hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(func.getContext(),
                                               vectorOnly
                                                   ? hivm::TFuncCoreType::AIV
                                                   : hivm::TFuncCoreType::AIC));
    func->removeAttr(hivm::TCoreRatioAttr::name);
    return success();
  }

  auto splitFuncs = hivm::createSplitMixFunctions(func);
  if (failed(splitFuncs))
    return failure();
  auto [cubeFunc, vecFunc] = *splitFuncs;

  if (failed(filterCoreOps(cubeFunc, hivm::TCoreType::VECTOR)))
    return failure();
  if (failed(filterCoreOps(vecFunc, hivm::TCoreType::CUBE)))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "[hivm-v2-split-kernel] AIC:\n"
                          << cubeFunc << "\nAIV:\n"
                          << vecFunc << "\n");
  return success();
}

struct V2SplitKernelPass : public impl::V2SplitKernelBase<V2SplitKernelPass> {
  void runOnOperation() override {
    SmallVector<func::FuncOp> mixFunctions;
    getOperation().walk([&](func::FuncOp func) {
      if (isMixFunction(func))
        mixFunctions.push_back(func);
    });

    for (func::FuncOp func : mixFunctions)
      if (failed(splitMixKernel(func)))
        return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createV2SplitKernelPass() {
  return std::make_unique<V2SplitKernelPass>();
}

//===----------------------- MarkMultiBuffer.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "hivm-mark-multi-buffer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_MARKMULTIBUFFER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// MarkMultiBufferPass
//===----------------------------------------------------------------------===//
namespace {

struct MarkMultiBufferPass
    : public impl::MarkMultiBufferBase<MarkMultiBufferPass> {
  using MarkMultiBufferBase<MarkMultiBufferPass>::MarkMultiBufferBase;

  explicit MarkMultiBufferPass(const MarkMultiBufferOptions &options)
      : MarkMultiBufferBase(options) {}

public:
  void runOnOperation() override;
};

FailureOr<Operation *> tracebackForWorkspace(Value val) {
  // Workspace couldn't be any block argument currently
  if (isa<BlockArgument>(val))
    return failure();

  return TypeSwitch<Operation *, FailureOr<Operation *>>(val.getDefiningOp())
      .Case<bishengir::memref_ext::AllocWorkspaceOp>(
          [&](bishengir::memref_ext::AllocWorkspaceOp op) { return op; })
      .Case<bufferization::ToTensorOp>([&](bufferization::ToTensorOp op) {
        return tracebackForWorkspace(op.getMemref());
      })
      .Case<mlir::ViewLikeOpInterface>([&](ViewLikeOpInterface viewLikeOp) {
        return tracebackForWorkspace(viewLikeOp.getViewSource());
      })
      .Default([&](Operation *op) { return failure(); });
}

static Value traceToRootMemref(Value v) {
  auto isRoot = [](Value v) {
    Operation *op = v.getDefiningOp();
    if (!op)
      return true;

    return !isa<memref::MemorySpaceCastOp,
                bufferization::ToTensorOp,
                ViewLikeOpInterface>(op);
  };

  auto vals = utils::tracebackMemRefVecByTargetFn(v, isRoot);

  if (vals.size() == 1)
    return vals.front();

  return Value();
}

// 判断 op 是否对 targetMemref 有写入操作
bool hasWrite(Operation *op, Value targetMemref) {
  if (isa<scope::ScopeOp>(op)) {
    return false;
  }
  if (isa<annotation::MarkOp>(op)) {
    return false;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  utils::collectAllEffects(op, effects);

  for (auto &eff : effects) {
    Value v = eff.getValue();
    if (!v || !isa<MemRefType>(v.getType()))
      continue;
    // 判断：Effect 是 Write，且操作的 Value 就是目标 MemRef
    Value rootV = traceToRootMemref(v);
    Value rootT = traceToRootMemref(targetMemref);
    if (isa<MemoryEffects::Write>(eff.getEffect()) && rootV == rootT) {
      return true;
    }
  }
  return false;
}

// 判断 op 是否对 targetMemref 有写入操作
bool hasRead(Operation *op, Value targetMemref) {
  if (isa<scope::ScopeOp>(op)) {
    return false;
  }
  if (isa<annotation::MarkOp>(op)) {
    return false;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  utils::collectAllEffects(op, effects);

  for (auto &eff : effects) {
    Value v = eff.getValue();
    if (!v)
      continue;
    // 判断：Effect 是 Write，且操作的 Value 就是目标 MemRef
    Value rootV = traceToRootMemref(v);
    Value rootT = traceToRootMemref(targetMemref);
    if (isa<MemoryEffects::Read>(eff.getEffect()) && rootV == rootT) {
      return true;
    }
  }
  return false;
}

static bool isPassthroughOpForTrace(Operation *op) {
  return isa<memref::MemorySpaceCastOp>(op) ||
         isa<bufferization::ToTensorOp>(op) ||
         isa<ViewLikeOpInterface>(op);
}

void traceToScopes(Operation *op, SmallVectorImpl<scope::ScopeOp> &scopes, DenseSet<Operation *> visited) {
  if (!visited.insert(op).second)
    return;
  if (auto scopeOp = dyn_cast<scope::ScopeOp>(op->getParentOp())) {
    scopes.push_back(scopeOp);
    return;
  }
  for (OpResult res : op->getResults()) {
    for (Operation *user : res.getUsers()) {
      Operation *parent = user->getParentOp();
      if (auto scopeOp = dyn_cast<scope::ScopeOp>(parent)) {
        scopes.push_back(scopeOp);
        return;
      } else {
        traceToScopes(user, scopes, visited);
      }
    }
  }
}

void traceForwardToScopes(Value v,
                          SmallVectorImpl<scope::ScopeOp> &producerScopes,
                          SmallVectorImpl<scope::ScopeOp> &consumerScopes) {
  DenseSet<Operation *> visited;
  for (Operation *user : v.getUsers()) {
    if (hasWrite(user, v)) {
      traceToScopes(user, producerScopes, visited);
    } else if (hasRead(user, v)) {
      traceToScopes(user, consumerScopes, visited);
    } else {
      if (isPassthroughOpForTrace(user)) {
        for (OpResult res : user->getResults()) {
          traceForwardToScopes(res, producerScopes, consumerScopes);
        }
      }
    }
  }
}

/// Whether the op is already marked multi_buffer attr.
static bool isMarked(Operation *op) {
  bool marked = utils::getAnnotateOpWithAttr(op->getResult(0),
                                             hivm::MultiBufferAttr::name)
                    .has_value();
  if (marked)
    LLVM_DEBUG(DBGS() << "already marked, skip.\n");
  return marked;
}

static void mark(mlir::Operation *op, PatternRewriter &rewriter,
                 unsigned numBuffer = 2, bool isPreload = false) {
  // result of allocOp or memref_ext::AllocWorkspaceOp
  auto mem = op->getResult(0);

  annotation::MarkOp markOp;
  for (Operation *user : mem.getUsers()) {
    if (auto m = dyn_cast<annotation::MarkOp>(user)) {
      markOp = m;
      break;
    }
  }

  if (!markOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    markOp = rewriter.create<annotation::MarkOp>(op->getLoc(), mem);
  }

  markOp->setAttr(hivm::MultiBufferAttr::name,
                  rewriter.getI32IntegerAttr(numBuffer));
  if (isPreload) {
    markOp->setAttr(hivm::PreloadLocalBufferAttr::name,
                    rewriter.getI32IntegerAttr(1));
  }
}

static std::optional<int32_t> getConsumerPreloadNum(Value scopeResult,
                                                    int32_t producerPreloadNum) {
  std::optional<int32_t> consumerPreloadNum;

  for (Operation *user : scopeResult.getUsers()) {
    if (isa<annotation::MarkOp>(user))
      continue;

    auto consumerScope = dyn_cast<scope::ScopeOp>(user);
    if (!consumerScope)
      consumerScope = user->getParentOfType<scope::ScopeOp>();
    if (!consumerScope)
      continue;

    auto preloadNumAttr =
        consumerScope->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name);
    if (!preloadNumAttr)
      continue;

    int32_t preloadNum = preloadNumAttr.getInt();
    if (preloadNum >= producerPreloadNum)
      continue;

    consumerPreloadNum =
        consumerPreloadNum ? std::min(*consumerPreloadNum, preloadNum)
                           : std::optional<int32_t>(preloadNum);
  }

  return consumerPreloadNum;
}

struct MarkScopeTightlyMultiBuffer : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  
  explicit MarkScopeTightlyMultiBuffer(MLIRContext *ctx)
      : OpRewritePattern<memref::AllocOp>(ctx) {}
  
  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(allocOp->getParentOp())) {
      return failure();
    }

    Value targetMemref = allocOp.getResult();
    if (isMarked(allocOp)) {
      return failure();
    }
    if (!utils::getAnnotateOpWithAttr(allocOp.getResult(),
                                      hivm::HIVMTightlyCoupledBufferAttr::name)
             .has_value()) {
      return failure();
    }

    // Step 1: trace forward from alloc to find all reachable scopes
    SmallVector<scope::ScopeOp> producerScopes;
    SmallVector<scope::ScopeOp> consumerScopes;

    traceForwardToScopes(targetMemref, producerScopes, consumerScopes);
    
    int32_t producerPreloadNum = -1;
    int32_t consumerPreloadNum = INT32_MAX;

    for (auto scopeOp : producerScopes) {
      auto preloadNumAttr = scopeOp->template getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name);
      if (!preloadNumAttr) {
        continue;
      }
      int32_t preloadNum = preloadNumAttr.getInt();
      producerPreloadNum = std::max(producerPreloadNum, preloadNum);
    }

    for (auto scopeOp : consumerScopes) {
      auto preloadNumAttr = scopeOp->template getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name);
      if (!preloadNumAttr) {
        continue;
      }
      int32_t preloadNum = preloadNumAttr.getInt();
      consumerPreloadNum = std::min(consumerPreloadNum, preloadNum);
    }

    if (producerPreloadNum == -1 || consumerPreloadNum == INT32_MAX) {
      return failure();
    }

    int32_t numBuffer = producerPreloadNum - consumerPreloadNum + 1;
    if (numBuffer <= 0) {
      return failure();
    }

    // apply mark
    mark(allocOp, rewriter, numBuffer, true);
    return success();
  }
};

struct MarkScopeMultiBuffer : public OpRewritePattern<scope::ScopeOp> {
  using OpRewritePattern<scope::ScopeOp>::OpRewritePattern;

  explicit MarkScopeMultiBuffer(MLIRContext *ctx)
      : OpRewritePattern<scope::ScopeOp>(ctx) {}

  LogicalResult matchAndRewrite(scope::ScopeOp scopeOp,
                                PatternRewriter &rewriter) const override {
    // Filter preload_num == 0
    auto preloadNumAttr =
        scopeOp->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name);
    if (!preloadNumAttr || preloadNumAttr.getInt() == 0)
      return failure();
    int32_t producerPreloadNum = preloadNumAttr.getInt();

    Block &block = scopeOp.getRegion().front();
    auto returnOp = cast<scope::ReturnOp>(block.getTerminator());

    bool anyAllocHasBeenMarked = false;
    for (auto [returnOperand, scopeResult] :
         llvm::zip_equal(returnOp->getOperands(), scopeOp->getResults())) {
      auto maybeConsumerPreloadNum =
          getConsumerPreloadNum(scopeResult, producerPreloadNum);
      if (!maybeConsumerPreloadNum)
        continue;

      std::optional<memref::AllocOp> maybeAllocOp;
      if (auto toTensor =
              returnOperand.getDefiningOp<bufferization::ToTensorOp>()) {
        maybeAllocOp = utils::tracebackMemRefToAlloc(toTensor.getMemref());
      } else if (isa<BaseMemRefType>(returnOperand.getType())) {
        maybeAllocOp = utils::tracebackMemRefToAlloc(returnOperand);
      }

      if (!maybeAllocOp)
        continue;
      memref::AllocOp allocOp = *maybeAllocOp;
      auto maybeAddressSpace = getOptionalHIVMAddressSpace(allocOp.getType());
      if (maybeAddressSpace && *maybeAddressSpace == hivm::AddressSpace::GM)
        continue;

      int32_t requiredBufferNum =
          producerPreloadNum - *maybeConsumerPreloadNum + 1;
      std::optional<int32_t> existingBufferNum;
      if (auto maybeMarkOp = utils::getAnnotateOpWithAttr(
              allocOp.getResult(), hivm::MultiBufferAttr::name)) {
        if (auto attr = (*maybeMarkOp)
                            ->getAttrOfType<IntegerAttr>(
                                hivm::MultiBufferAttr::name)) {
          existingBufferNum = static_cast<int32_t>(attr.getInt());
        }
      }
      uint32_t numBuffer =
          std::max<uint32_t>(static_cast<uint32_t>(requiredBufferNum),
                             existingBufferNum.value_or(0));

      bool isPreloadLocalBufferMarked =
          utils::getAnnotateOpWithAttr(allocOp.getResult(),
                                       hivm::PreloadLocalBufferAttr::name)
              .has_value();
      if (existingBufferNum && *existingBufferNum >= requiredBufferNum &&
          isPreloadLocalBufferMarked)
        continue;

      mark(allocOp, rewriter, numBuffer, true);
      anyAllocHasBeenMarked = true;
    }

    return success(anyAllocHasBeenMarked);
  }
};

template <typename CopyOpType>
struct MarkMultiBuffer : public OpRewritePattern<CopyOpType> {
  using OpRewritePattern<CopyOpType>::OpRewritePattern;

  explicit MarkMultiBuffer(MLIRContext *ctx)
      : OpRewritePattern<CopyOpType>(ctx) {}

  LogicalResult matchAndRewrite(CopyOpType copyLikeOp,
                                PatternRewriter &rewriter) const override {
    auto markBufferFunc = [&](mlir::Value &v) -> LogicalResult {
      auto *allocOp = utils::tracebackMemRef(v).getDefiningOp();
      if (!utils::isAllocLikeOp(allocOp)) {
        return failure();
      }
      if (isMarked(allocOp)) {
        return failure();
      }
      auto parentLoop = mlir::hivm::getParentLoop(allocOp->getResult(0));
      if (!parentLoop) {
        LLVM_DEBUG(DBGS() << " allocOp has no proper parent loop.\n");
        return failure();
      }

      // Allow scf::ForOp and scf::WhileOp ancestors. scf.while is supported
      // via the alloca-based counter scheme implemented in
      // MultiBufferLoopAdapter; other LoopLike ops (scf.parallel,
      // scf.forall, ...) are not yet supported because their semantics break
      // the per-iteration slot rotation invariant.
      while (parentLoop) {
        if (!isa<scf::ForOp, scf::WhileOp>(parentLoop)) {
          LLVM_DEBUG(DBGS() << "Unsupported loop type for multi-buffer: "
                            << parentLoop->getName() << "\n");
          return failure();
        }
        parentLoop = parentLoop->getParentOfType<LoopLikeOpInterface>();
      }

      // Do mark operations
      mark(allocOp, rewriter);
      return success();
    };

    if (!copyLikeOp.hasPureBufferSemantics()) {
      LLVM_DEBUG(DBGS() << copyLikeOp
                        << "mark allocOp with multi-buffer is designed for "
                           "pure buffer state");

      return failure();
    }

    auto src = copyLikeOp.getSrc();
    auto dst = copyLikeOp.getDst();
    if (!dyn_cast<BaseMemRefType>(src.getType()).getMemorySpace() ||
        !dyn_cast<BaseMemRefType>(dst.getType()).getMemorySpace())
      return failure();

    if (getHIVMAddressSpace(src.getType()) != hivm::AddressSpace::GM) {
      return markBufferFunc(src);
    }

    if (getHIVMAddressSpace(dst.getType()) != hivm::AddressSpace::GM) {
      return markBufferFunc(dst);
    }

    return failure();
  }
};

// For workspace scene, it's distinct from marking ub multiple buffer that here
// only aims for `write workspace in loop`.
// Following pattern matches writing operations including storeOp and fixpipeOp,
// then it checks whether store dst is workspace and workspace is in loop
template <typename StoreOpType>
class MarkWorkspaceMultiBuffer : public OpRewritePattern<StoreOpType> {
  const unsigned multiBufferNum;

  LogicalResult matchAndRewrite(StoreOpType storeLikeOp,
                                PatternRewriter &rewriter) const override {
    if (!storeLikeOp.hasPureTensorSemantics()) {
      LLVM_DEBUG(DBGS() << storeLikeOp
                        << "mark allocWorkSpaceOp with "
                           "multi-buffer is designed for pure tensor state");
      return failure();
    }

    assert(storeLikeOp.getNumDpsInits() == 1);
    Value dst = storeLikeOp.getDpsInitOperand(0)->get();
    auto allocWorksapce = tracebackForWorkspace(dst);
    if (failed(allocWorksapce))
      return failure();

    // Already marked
    if (::isMarked(*allocWorksapce))
      return failure();

    // It cannot do multi buffer opt without parent loop
    if (!isa<LoopLikeOpInterface>((*allocWorksapce)->getParentOp()))
      return failure();

    ::mark(*allocWorksapce, rewriter, multiBufferNum);
    return success();
  }

public:
  explicit MarkWorkspaceMultiBuffer(MLIRContext *ctx, unsigned multiBufferNum)
      : OpRewritePattern<StoreOpType>(ctx), multiBufferNum(multiBufferNum) {}
};

void MarkMultiBufferPass::runOnOperation() {
  if (!enableAuto) {
    LLVM_DEBUG(
        DBGS() << "enableAuto is false, no need to mark automatically.\n");
    return;
  }

  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());

  auto funcCoreType = queryFuncCoreType(funcOp);
  const bool isMixFuncCore =
      funcCoreType.has_value() &&
      (funcCoreType.value() == TFuncCoreType::MIX ||
       funcOp->getAttrOfType<UnitAttr>(hivm::TPartOfMixAttr::name));
  patterns.insert<MarkScopeTightlyMultiBuffer>(patterns.getContext());
  patterns.insert<MarkScopeMultiBuffer>(patterns.getContext());
  // Per-buffer fine-grained gates (used by BiShengIRCompileMain's compile-
  // time fallback to surgically turn off the multi-buffer of just the
  // overflowing address space):
  //   ND2NZ   -> L1 (cbuf)         -> disableMultiBufferOnL1
  //   Fixpipe -> L0C (cube acc)    -> disableMultiBufferOnL0C
  //   Load    -> UB (Vector ingress)\
  //   Store   -> UB (Vector egress) -> disableMultiBufferOnUB
  // These AND-combine with the existing coarse Mix-core gates
  // (limitMixAutoMultiBufferBuffer == ONLY_VECTOR/ONLY_CUBE) and with
  // limitAutoMultiBufferOfLocalBuffer == CUBE_NO_L0C: any single switch can
  // disable a given group; we never re-enable.
  const bool allowCubeGroup =
      !isMixFuncCore ||
      !(limitMixAutoMultiBufferBuffer == MultiBufferStrategy::ONLY_VECTOR);
  if (allowCubeGroup) {
    if (!disableMultiBufferOnL1)
      patterns.insert<MarkMultiBuffer<hivm::ND2NZOp>>(patterns.getContext());
    // TODO: DN2NZ
    if (limitAutoMultiBufferOfLocalBuffer != MultiBufferStrategy::CUBE_NO_L0C &&
        !disableMultiBufferOnL0C) {
      patterns.insert<MarkMultiBuffer<hivm::FixpipeOp>>(patterns.getContext());
    }
  }
  const bool allowVectorGroup =
      !isMixFuncCore ||
      !(limitMixAutoMultiBufferBuffer == MultiBufferStrategy::ONLY_CUBE);
  if (allowVectorGroup && !disableMultiBufferOnUB) {
    patterns.insert<MarkMultiBuffer<hivm::LoadOp>>(patterns.getContext());
    patterns.insert<MarkMultiBuffer<hivm::StoreOp>>(patterns.getContext());
  }

  if (!limitAutoMultiBufferOnlyForLocalBuffer && isMixFuncCore)
    patterns.insert<MarkWorkspaceMultiBuffer<hivm::StoreOp>,
                    MarkWorkspaceMultiBuffer<hivm::FixpipeOp>>(
        patterns.getContext(), workspaceMultiBufferNum);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace

std::unique_ptr<Pass>
mlir::hivm::createMarkMultiBufferPass(const MarkMultiBufferOptions &options) {
  return std::make_unique<MarkMultiBufferPass>(options);
}

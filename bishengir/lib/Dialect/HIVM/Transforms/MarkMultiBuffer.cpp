//===----------------------- MarkMultiBuffer.cpp --------------------------===//
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
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

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

/// Whether the op is already marked multi_buffer attr.
static bool isMarked(Operation *op) {
  auto users = op->getUsers();
  // users has no rbegin iterator
  for (auto user : users) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
      auto attrDict = markOp->getAttrDictionary();
      if (!attrDict.empty() && attrDict.contains(hivm::MultiBufferAttr::name)) {
        LLVM_DEBUG(DBGS() << "already marked, skip.\n");
        return true;
      }
    }
  }

  return false;
}

static void mark(mlir::Operation *op, PatternRewriter &rewriter,
                 unsigned numBuffer = 2) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  // result of allocOp or memref_ext::AllocWorkspaceOp
  auto mem = op->getResult(0);
  auto markOp = rewriter.create<annotation::MarkOp>(op->getLoc(), mem);
  markOp->setAttr(hivm::MultiBufferAttr::name,
                  rewriter.getI32IntegerAttr(numBuffer));
}

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

      // Currently, only allow scf::ForOp to handle the multi buffering
      while (parentLoop) {
        if (!isa<scf::ForOp>(parentLoop)) {
          LLVM_DEBUG(DBGS()
                     << "Currently only scf.for is supported loop type.\n");
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
  if (!isMixFuncCore ||
      !(limitMixAutoMultiBufferBuffer == MultiBufferStrategy::ONLY_VECTOR)) {
    patterns.insert<MarkMultiBuffer<hivm::ND2NZOp>>(patterns.getContext());
    if (limitAutoMultiBufferOfLocalBuffer != MultiBufferStrategy::CUBE_NO_L0C) {
      patterns.insert<MarkMultiBuffer<hivm::FixpipeOp>>(patterns.getContext());
    }
  }
  if (!isMixFuncCore ||
      !(limitMixAutoMultiBufferBuffer == MultiBufferStrategy::ONLY_CUBE)) {
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

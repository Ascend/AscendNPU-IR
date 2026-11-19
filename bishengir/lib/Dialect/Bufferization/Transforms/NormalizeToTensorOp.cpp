//===-- NormalizeToTensorOp.cpp - Normalize bufferization.to_tensor ops --===//
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
// This pass normalizes `bufferization.to_tensor` ops before bufferization:
//
//  1. Memory-space isolation: insert a `memref.memory_space_cast` before a
//     `to_tensor` whose memref carries a memory space, so that the attribute
//     stays out of the tensor bufferization path.
//
//  2. Expose memref-level writes: One-Shot Bufferization only analyzes tensor
//     operands, so a write performed on the memref before the `to_tensor`
//     (e.g., a `hivm.hir.load` into a subview) is invisible to it. For
//     writable `to_tensor` ops whose memref may have been written (a DPS init
//     operand or a call, tracing through view-like and `arith.select`
//     aliases), rewrite
//
//       %alloc = memref.alloc()
//       hivm.hir.load outs(%subview_of_alloc)   // memref-level write
//       %t = bufferization.to_tensor %alloc restrict writable
//
//     into
//
//       %alloc = memref.alloc()
//       hivm.hir.load outs(%subview_of_alloc)   // memref-level write
//       %raw = bufferization.to_tensor %alloc restrict writable
//       %t = bufferization.alloc_tensor() copy(%raw) {to_be_replaced}
//
//     so that the write becomes visible at tensor level.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/ErrorHandling.h"

namespace bishengir::bufferization {
using namespace mlir;
#define GEN_PASS_DEF_NORMALIZETOTENSOROP
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bishengir::bufferization

using namespace mlir;
using namespace mlir::bufferization;

namespace {

static constexpr llvm::StringLiteral kToBeReplacedAttr = "to_be_replaced";

/// Isolate the memory space of a `to_tensor` operand.
struct IsolateToTensorMemspacePattern : public OpRewritePattern<ToTensorOp> {
  using OpRewritePattern<ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToTensorOp toTensorOp,
                                PatternRewriter &rewriter) const override {
    Value memref = toTensorOp.getMemref();
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType || !memrefType.getMemorySpace())
      return failure();

    rewriter.setInsertionPoint(toTensorOp);
    Type plainType =
        MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                        memrefType.getLayout());
    auto castOp = rewriter.create<memref::MemorySpaceCastOp>(
        toTensorOp.getLoc(), plainType, memref);
    rewriter.modifyOpInPlace(toTensorOp, [&]() {
      toTensorOp.getMemrefMutable().assign(castOp.getResult());
    });
    return success();
  }
};

/// Return true if `user` is known to execute before `toTensor`.
///
/// A `scope.scope` marked `hivm.load_preload` runs its body exactly once, so
/// a write in its direct body precedes a later `to_tensor` although it does
/// not structurally dominate it; writes nested deeper may not execute.
bool precedesToTensor(Operation *user, Operation *toTensor,
                      DominanceInfo &domInfo) {
  if (domInfo.properlyDominates(user, toTensor))
    return true;
  auto loadScope = dyn_cast<scope::ScopeOp>(user->getParentOp());
  if (!loadScope || !loadScope->hasAttr(hivm::LoadPreloadAttr::name))
    return false;
  return domInfo.properlyDominates(loadScope, toTensor);
}

/// Return true if `memref`, or an alias reached through view-like ops or
/// `arith.select`, may be written before `toTensor`.
///
/// A DPS init operand is a definite write; a call is conservatively treated
/// as one.
bool isMemrefWrittenBeforeToTensor(Value memref, Operation *toTensor) {
  SmallVector<Value> worklist = {memref};
  DenseSet<Value> visited;
  DominanceInfo domInfo(toTensor->getParentOfType<func::FuncOp>());

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (auto bbArg = dyn_cast<BlockArgument>(current)) {
      auto *ownerOp = bbArg.getOwner()->getParentOp();
      if (!isa<func::FuncOp>(ownerOp)) {
        llvm::dbgs() << current << "\n";
        llvm::report_fatal_error(
            "unexpected value when check memref written before to_tensor");
      }
    } else {
      auto *defOp = current.getDefiningOp();
      if (!llvm::isa<bishengir::memref_ext::AllocWorkspaceOp, memref::AllocOp,
                     arith::SelectOp, ViewLikeOpInterface>(defOp)) {
        defOp->dump();
        llvm::report_fatal_error(
            "unexpected defOp when check memref written before to_tensor");
      }
      if (auto viewLikeOp = llvm::dyn_cast<ViewLikeOpInterface>(defOp))
        worklist.push_back(viewLikeOp.getViewSource());
      else if (auto selectOp = llvm::dyn_cast<arith::SelectOp>(defOp)) {
        worklist.push_back(selectOp.getTrueValue());
        worklist.push_back(selectOp.getFalseValue());
      }
    }

    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (user == toTensor)
        continue;
      if (!llvm::isa<bishengir::memref_ext::AllocWorkspaceOp, memref::AllocOp,
                     DestinationStyleOpInterface, CopyOpInterface, func::CallOp,
                     annotation::MarkOp, arith::SelectOp, ViewLikeOpInterface,
                     memref::StoreOp>(user)) {
        user->dump();
        llvm::report_fatal_error(
            "unexpected user when check memref written before to_tensor");
      }
      if (!precedesToTensor(user, toTensor, domInfo))
        continue;
      // A DPS init operand is a definite write. A call may write through the
      // memref, so it is conservatively treated as a write.
      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(user)) {
        if (dpsOp.isDpsInit(&use))
          return true;
      } else if (auto callOp = dyn_cast<func::CallOp>(user)) {
        return true;
      } else if (auto copyOp = dyn_cast<CopyOpInterface>(user)) {
        if (copyOp.getTarget() == use.get())
          return true;
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
        if (storeOp.getMemRef() == use.get())
          return true;
      } else if (isa<ViewLikeOpInterface, arith::SelectOp>(user)) {
        worklist.push_back(user->getResult(0));
      }
    }
  }
  return false;
}

/// Copy the `to_tensor` result to expose a possible memref-level write at
/// tensor level; `to_be_replaced` lets bufferization fold the copy away later.
void exposeMemrefWriteToTensor(ToTensorOp toTensorOp, IRRewriter &rewriter) {
  rewriter.setInsertionPointAfter(toTensorOp);
  auto allocTensorCopy = utils::createAllocTensorOp(
      rewriter, toTensorOp.getLoc(), toTensorOp.getResult(), /*copy=*/true);
  allocTensorCopy->setAttr(kToBeReplacedAttr, rewriter.getUnitAttr());
  rewriter.replaceAllUsesExcept(toTensorOp.getResult(),
                                allocTensorCopy.getResult(), allocTensorCopy);
}

struct NormalizeToTensorOpPass
    : public bishengir::bufferization::impl::NormalizeToTensorOpBase<
          NormalizeToTensorOpPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    patterns.add<IsolateToTensorMemspacePattern>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    IRRewriter rewriter(context);
    funcOp.walk([&](ToTensorOp toTensorOp) {
      if (toTensorOp.getWritable() &&
          isMemrefWrittenBeforeToTensor(toTensorOp.getMemref(), toTensorOp))
        exposeMemrefWriteToTensor(toTensorOp, rewriter);
    });
  }
};

} // namespace

std::unique_ptr<Pass>
bishengir::bufferization::createNormalizeToTensorOpPass() {
  return std::make_unique<NormalizeToTensorOpPass>();
}

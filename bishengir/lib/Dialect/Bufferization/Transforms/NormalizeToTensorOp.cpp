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

/// Verify that `current` is rooted in an allocation or a function argument, and
/// enqueue the values it aliases (the source of a view-like op, or both arms of
/// a select). Abort on any other defining op, since the write trace cannot
/// reason about it.
void verifyAndCollectDefChain(Value current, SmallVector<Value> &worklist) {
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
}

/// Return true if `memref`, or an alias reached through view-like ops or
/// `arith.select`, is written before `toTensor` in memory view order.
///
/// A DPS init operand, a copy target and a store are definite writes; a call is
/// conservatively treated as one because it may write through the memref. Also
/// reports an error when the memref feeds more than one `to_tensor`, which
/// breaks the `restrict` contract.
bool isMemrefWrittenBeforeToTensor(Value memref, Operation *toTensor) {
  SmallVector<Value> worklist = {memref};
  DenseSet<Value> visited;
  bool needExpose = false;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    verifyAndCollectDefChain(current, worklist);
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (user == toTensor)
        continue;
      if (isa<bufferization::ToTensorOp>(user)) {
        user->emitOpError()
            << "illegal usage of to_tensor op and restrict attr : a memref "
               "should not be converted to multiple tensors, which may cause "
               "precision error. See more details in definition of to_tensor "
               "op and restrict attr";
        continue;
      }
      if (!llvm::isa<DestinationStyleOpInterface, CopyOpInterface, func::CallOp,
                     annotation::MarkOp, arith::SelectOp, ViewLikeOpInterface,
                     memref::StoreOp>(user)) {
        user->dump();
        llvm::report_fatal_error(
            "unexpected user when check memref written before to_tensor");
      }

      bool memrefWritten = false;
      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(user)) {
        if (dpsOp.isDpsInit(&use))
          memrefWritten = true;
      } else if (auto callOp = dyn_cast<func::CallOp>(user)) {
        memrefWritten = true;
      } else if (auto copyOp = dyn_cast<CopyOpInterface>(user)) {
        if (copyOp.getTarget() == use.get())
          memrefWritten = true;
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
        if (storeOp.getMemRef() == use.get())
          memrefWritten = true;
      } else if (isa<ViewLikeOpInterface, arith::SelectOp>(user)) {
        worklist.push_back(user->getResult(0));
      }

      if (!memrefWritten)
        // Read-only use; keep checking the other users.
        continue;

      if (utils::isBeforeInMemoryView(user, toTensor))
        needExpose = true;
      // TODO: A memref-level write after `to_tensor` may invalidate later reads
      // of the tensor and break tensor value semantics, e.g.:
      //
      //   %alloc = memref.alloc()
      //   load outs(%alloc)                         // Memref-level write.
      //   %t = bufferization.to_tensor %alloc restrict writable
      //   read %t
      //   write outs(%alloc)                        // Overwrites the buffer.
      //   read %t                                   // May observe new data.
      //
      // The write is safe if %t and its aliases are no longer read afterwards.
      // Ordering tensor reads against memref writes needs extra analysis, so no
      // error is reported here to avoid noisy diagnostics.
    }
  }
  return needExpose;
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

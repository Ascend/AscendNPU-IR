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
//  1. Memory-space isolation: when the memref operand carries a memory
//     space, a `memref.memory_space_cast` is inserted before the `to_tensor`
//     so that it operates on a memory-space-free memref. This keeps the
//     memory-space attr out of the tensor bufferization path.
//
//  2. Expose memref-level writes: for `to_tensor` ops with the `writable`
//     attribute whose source memref was written by a DPS op (e.g.,
//     hivm.hir.load) before the `to_tensor`, such memref-level writes are
//     invisible to One-Shot Bufferization analysis (it only analyzes
//     tensor-type operands). The transform replaces:
//
//       %alloc = memref.alloc()
//       hivm.hir.load outs(%subview_of_alloc)   // memref-level write
//       %t = bufferization.to_tensor %alloc restrict writable
//
//     with:
//       %alloc = memref.alloc()
//       hivm.hir.load outs(%subview_of_alloc)   // memref-level write
//       %raw = bufferization.to_tensor %alloc restrict writable
//       %t = bufferization.alloc_tensor() copy(%raw) {to_be_replaced}
//
//     This makes the writes of %alloc visible to One-Shot Bufferization
//     analysis, enabling it to see the tensor-level write and analyze
//     conflicts on %t.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Bufferization/Transforms/Passes.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
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

static constexpr llvm::StringLiteral kWriteExposedAttr = "write_exposed";
static constexpr llvm::StringLiteral kToBeReplacedAttr = "to_be_replaced";

/// Isolate the memory-space info from a `to_tensor` by inserting a
/// `memref.memory_space_cast` before it when the memref operand carries a
/// memory space.
struct IsolateToTensorMemspacePattern : public OpRewritePattern<ToTensorOp> {
  using OpRewritePattern<ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToTensorOp toTensorOp,
                                PatternRewriter &rewriter) const override {
    Value memref = toTensorOp.getMemref();
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    // The memref must carry a memory space; memrefs without one need no
    // isolation.
    if (!memrefType || !memrefType.getMemorySpace())
      return failure();

    rewriter.setInsertionPoint(toTensorOp);
    // Build a type without the memory-space attribute.
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

/// Check if a memref value (or its aliases via subview/reinterpret_cast/cast)
/// is written by a DPS op before the given `toTensor` operation. The memref
/// is traced up to its root, which must be an allocation (memref.alloc /
/// alloc_workspace) or a function-entry argument.
static bool isMemrefWrittenBeforeToTensor(Value memref, Operation *toTensor) {
  SmallVector<Value> worklist = {memref};
  DenseSet<Value> visited;
  DominanceInfo domInfo(toTensor->getParentOfType<func::FuncOp>());

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    // Trace the value up to its root: follow view-like defining ops (e.g. a
    // memory_space_cast inserted by the memspace isolation) to their source,
    // and accept allocations and function-entry arguments as roots. Anything
    // else (e.g. a block argument of a nested region) is unexpected.
    if (auto bbArg = dyn_cast<BlockArgument>(current)) {
      auto *ownerOp = bbArg.getOwner()->getParentOp();
      if (!isa<func::FuncOp>(ownerOp)) {
        llvm::dbgs() << current << "\n";
        llvm::report_fatal_error(
            "unexpected value when check memref written before to_tensor");
      }
    } else {
      auto *defOp = current.getDefiningOp();
      if (auto viewLikeOp = llvm::dyn_cast<ViewLikeOpInterface>(defOp)) {
        worklist.push_back(viewLikeOp.getViewSource());
      } else if (!llvm::isa<bishengir::memref_ext::AllocWorkspaceOp,
                            memref::AllocOp>(defOp)) {
        llvm::dbgs() << current << "\n";
        llvm::report_fatal_error(
            "unexpected value when check memref written before to_tensor");
      }
    }

    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      // Skip the to_tensor itself.
      if (user == toTensor)
        continue;
      // Only consider users that are properly before to_tensor.
      // dominanceInfo.properlyDominates returns true if `user` comes before
      // `to_tensor` in execution order (and they are not the same op).
      if (!domInfo.properlyDominates(user, toTensor))
        continue;
      // If user is a DPS op that writes to current (as a dpsInit operand),
      // the memref has been written.
      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(user)) {
        if (dpsOp.isDpsInit(&use))
          return true;
      }
      // Trace through alias-creating ops to find writes to derived memrefs.
      if (isa<ViewLikeOpInterface>(user)) {
        worklist.push_back(user->getResult(0));
      }
    }
  }
  return false;
}

/// Expose a prior memref-level DPS write to tensor-level analysis.
static void exposeMemrefWriteToTensor(ToTensorOp toTensorOp,
                                      PatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(toTensorOp);
  auto allocTensorCopy = utils::createAllocTensorOp(
      rewriter, toTensorOp.getLoc(), toTensorOp.getResult(), /*copy=*/true);
  allocTensorCopy->setAttr(kToBeReplacedAttr, rewriter.getUnitAttr());
  rewriter.replaceAllUsesExcept(toTensorOp.getResult(),
                                allocTensorCopy.getResult(), allocTensorCopy);
  rewriter.modifyOpInPlace(toTensorOp, [&]() {
    toTensorOp->setAttr(kWriteExposedAttr, rewriter.getUnitAttr());
  });
}

struct ExposeMemrefWriteToTensorPattern : public OpRewritePattern<ToTensorOp> {
  using OpRewritePattern<ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToTensorOp toTensorOp,
                                PatternRewriter &rewriter) const override {
    if (!toTensorOp.getWritable() || toTensorOp->hasAttr(kWriteExposedAttr) ||
        !isMemrefWrittenBeforeToTensor(toTensorOp.getMemref(), toTensorOp))
      return failure();

    exposeMemrefWriteToTensor(toTensorOp, rewriter);
    return success();
  }
};

struct NormalizeToTensorOpPass
    : public bishengir::bufferization::impl::NormalizeToTensorOpBase<
          NormalizeToTensorOpPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    patterns
        .add<IsolateToTensorMemspacePattern, ExposeMemrefWriteToTensorPattern>(
            context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
bishengir::bufferization::createNormalizeToTensorOpPass() {
  return std::make_unique<NormalizeToTensorOpPass>();
}

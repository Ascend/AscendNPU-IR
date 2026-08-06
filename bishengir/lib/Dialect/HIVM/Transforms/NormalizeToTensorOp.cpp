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
//  1. Memory-space isolation: when the memref operand carries an hivm
//     address-space attribute, a `memref.memory_space_cast` is inserted before
//     the `to_tensor` so that it operates on a memory-space-free memref. This
//     keeps the hivm address-space attr out of the tensor bufferization path.
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
//       %raw = bufferization.to_tensor %alloc restrict  writable
//       %dest = bufferization.alloc_tensor() : tensor<...>
//       %t = hivm.hir.copy in(%raw) out(%dest) {to_be_replaced}: tensor<...>
//
//     This makes the writes of %alloc visible to One-Shot Bufferization
//     analysis, enabling it to see the tensor-level write and analyze conflicts
//     on %t. The copyOp is later replaced with the original to_tensor op during
//     resolve-RAW-conflicts.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZETOTENSOROP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

namespace {

static constexpr llvm::StringLiteral kWriteExposedAttr = "write_exposed";

/// Isolate the hivm memory-space info from a `to_tensor` by inserting a
/// `memref.memory_space_cast` before it when the memref operand carries an
/// hivm address-space attribute.
struct IsolateToTensorMemspacePattern : public OpRewritePattern<ToTensorOp> {
  using OpRewritePattern<ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToTensorOp toTensorOp,
                                PatternRewriter &rewriter) const override {
    Value memref = toTensorOp.getMemref();
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType ||
        !hivm::getOptionalHIVMAddressSpace(memrefType).has_value())
      return failure();

    rewriter.setInsertionPoint(toTensorOp);
    // Build a type without the hivm memory-space attribute.
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
/// is written by a DPS op before the given `toTensor` operation.
static bool isMemrefWrittenBeforeToTensor(Value memref, Operation *toTensor) {
  DominanceInfo domInfo(toTensor->getParentOfType<func::FuncOp>());
  SmallVector<Value> worklist = {memref};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      // Skip the to_tensor itself.
      if (user == toTensor)
        continue;
      // Only consider users that are properly before to_tensor.
      // dominanceInfo.properlyDominates returns true if `user` comes before
      // `toTensor` in execution order (and they are not the same op).
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
  Location loc = toTensorOp.getLoc();
  Type tensorType = toTensorOp.getType();
  // Retarget only the uses existing now; the helpers created below add new
  // uses of the to_tensor result that must stay (dominance).
  auto toTensorValue = toTensorOp.getResult();
  SmallVector<OpOperand *> usesToReplace;
  for (OpOperand &use : toTensorValue.getUses()) {
    usesToReplace.push_back(&use);
  }
  // createAllocTensorOp derives dim sizes; to_tensor may be dynamic-typed (e.g.
  // sub-tiled AIV functions), where ValueRange{} is invalid.
  auto newValue = utils::createAllocTensorOp(rewriter, loc, toTensorValue);
  auto copyOp =
      rewriter.create<hivm::CopyOp>(loc, tensorType, toTensorValue, newValue);
  copyOp->setAttr("to_be_replaced", rewriter.getUnitAttr());
  for (OpOperand *use : usesToReplace) {
    Operation *user = use->getOwner();
    rewriter.modifyOpInPlace(user, [&]() { use->set(copyOp.getResult(0)); });
  }
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
    : public impl::NormalizeToTensorOpBase<NormalizeToTensorOpPass> {

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

std::unique_ptr<Pass> mlir::hivm::createNormalizeToTensorOpPass() {
  return std::make_unique<NormalizeToTensorOpPass>();
}

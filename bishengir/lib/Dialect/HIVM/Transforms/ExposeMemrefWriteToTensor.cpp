//===- ExposeMemrefWriteToTensor.cpp - Pre-bufferization transform --------===//
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
// This pass finds `bufferization.to_tensor` ops with the `writable` attribute
// whose source memref was written by a DPS op (e.g., hivm.hir.load) before
// the `to_tensor`. Such memref-level writes are invisible to One-Shot
// Bufferization analysis because it only analyzes tensor-type operands.
//
// The transform replaces:
//   %alloc = memref.alloc()
//   hivm.hir.load outs(%subview_of_alloc)   // memref-level write
//   %t = bufferization.to_tensor %alloc restrict writable
//
// With:
//   %alloc = memref.alloc()
//   hivm.hir.load outs(%subview_of_alloc)   // memref-level write
//   %raw = bufferization.to_tensor %alloc restrict  writable
//   %dest = tensor.empty() : tensor<...>
//   %t = hivm.hir.copy in(%raw) out(%dest) {to_be_replaced}: tensor<...>
//
// This makes the writes of %alloc visible to One-Shot Bufferization analysis,
// allowing it to see the tensor-level write and analyze conflicts on %t. And
// copyOp will be replaced with the original to_tensor op during resolve RaW
// conflicts.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_EXPOSEMEMREFWRITETOTENSOR
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

namespace {

/// Check if a memref value (or its aliases via subview/reinterpret_cast/cast)
/// is written by a DPS op before the given `toTensor` operation.
static bool isMemrefWrittenBeforeToTensor(Value memref, Operation *toTensor,
                                          DominanceInfo &domInfo) {
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

struct ExposeMemrefWriteToTensorPass
    : public impl::ExposeMemrefWriteToTensorBase<
          ExposeMemrefWriteToTensorPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(&getContext());
    DominanceInfo &domInfo = getAnalysis<DominanceInfo>();

    // Collect all to_tensor ops that need transformation.
    SmallVector<ToTensorOp> toTransform;
    funcOp.walk([&](ToTensorOp toTensorOp) {
      // Only transform to_tensor ops with writable attribute.
      if (!toTensorOp.getWritable())
        return;

      Value memref = toTensorOp.getMemref();

      // Check if the source memref was written by a DPS op before to_tensor.
      if (isMemrefWrittenBeforeToTensor(memref, toTensorOp, domInfo))
        toTransform.push_back(toTensorOp);
    });

    for (ToTensorOp toTensorOp : toTransform) {
      rewriter.setInsertionPointAfter(toTensorOp);
      Location loc = toTensorOp.getLoc();
      Type tensorType = toTensorOp.getType();

      // Retarget only the uses existing now; the helpers created below add
      // new uses of the to_tensor result that must stay (dominance).
      SmallVector<OpOperand *> usesToReplace;
      for (OpOperand &use : toTensorOp.getResult().getUses())
        usesToReplace.push_back(&use);

      // createEmptyOp derives dim sizes; to_tensor may be dynamic-typed
      // (e.g. sub-tiled AIV functions), where ValueRange{} is invalid.
      Value emptyOp =
          utils::createEmptyOp(rewriter, loc, toTensorOp.getResult());
      auto copyOp = rewriter.create<hivm::CopyOp>(
          loc, tensorType, toTensorOp.getResult(), emptyOp);
      copyOp->setAttr("to_be_replaced", rewriter.getUnitAttr());
      for (OpOperand *use : usesToReplace) {
        Operation *user = use->getOwner();
        rewriter.modifyOpInPlace(user,
                                 [&]() { use->set(copyOp.getResult(0)); });
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createExposeMemrefWriteToTensorPass() {
  return std::make_unique<ExposeMemrefWriteToTensorPass>();
}

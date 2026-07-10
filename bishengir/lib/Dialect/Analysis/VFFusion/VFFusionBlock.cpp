//===- VFFusionBlock.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/VFFusion/VFFusionBlock.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vf-fusion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace llvm;

namespace mlir {
namespace analysis {

bool VFFusionBlock::hasFusedUser(Operation *const op) const {
  if (ops.contains(op))
    return true;
  return any_of(op->getUsers(),
                [this](Operation *const user) { return hasFusedUser(user); });
}

ArrayRef<Operation *> VFFusionBlock::getOps() const { return ops.getArrayRef(); }

// insert operand to `inputs` if it uses an operand that is defined from
// outside.
SetVector<Value> VFFusionBlock::recomputeInputs() {
  inputs.clear();
  DenseSet<Value> definedInside;
  for (Operation *const outerOp : ops) {
    outerOp->walk<WalkOrder::PreOrder>([&definedInside](Operation *const op) {
      for (auto res : op->getResults())
        definedInside.insert(res);
      for (auto &reg : op->getRegions())
        for (auto &block : reg.getBlocks())
          for (auto blockArg : block.getArguments())
            definedInside.insert(blockArg);
    });
  }
  for (Operation *const outerOp : ops) {
    outerOp->walk<WalkOrder::PreOrder>([&](Operation *const op) {
      for (auto opr : op->getOperands())
        if (!definedInside.contains(opr))
          inputs.insert(opr);
    });
  }
  return inputs;
}

// might need to recompute inputs, consider how inputs looks like when outline
// two consecutive blocks.
const SetVector<Value> &VFFusionBlock::getInputs() const { return inputs; }

// insert result to `outputs` if it has a user that is not fused on the same
// block.
SetVector<Value> VFFusionBlock::recomputeOutputs() {
  outputs.clear();
  SetVector<Operation *> opsInside;
  // insert all operations that are fused.
  for (Operation *outerOp : ops) {
    outerOp->walk([&opsInside](Operation *const op) { opsInside.insert(op); });
    opsInside.insert(outerOp);
  }
  for (Operation *outerOp : ops) {
    for (auto res : outerOp->getResults()) {
      // all users are in the fused block.
      if (all_of(res.getUsers(), [&opsInside](Operation *userOp) {
            return opsInside.contains(userOp);
          }))
        continue;
      outputs.insert(res);
    }
  }
  return outputs;
}

// might need to recompute outputs, consider how outputs looks like when outline
// two consecutive blocks.
SetVector<Value> VFFusionBlock::getOutputs() const { return outputs; }

void VFFusionBlock::fuseOp(Operation *const op) {
  if (ops.contains(op))
    return;
  // Incrementally maintain inputs.
  DenseSet<Value> definedInOp;
  op->walk<WalkOrder::PreOrder>([&definedInOp](Operation *const inner) {
    for (auto res : inner->getResults())
      definedInOp.insert(res);
    for (auto &region : inner->getRegions())
      for (auto &block : region.getBlocks())
        for (auto blockArg : block.getArguments())
          definedInOp.insert(blockArg);
  });
  op->walk<WalkOrder::PreOrder>([&](Operation *const inner) {
    for (auto opr : inner->getOperands()) {
      if (definedInOp.contains(opr))
        continue; // defined within op itself
      Operation *def = opr.getDefiningOp();
      if (def && ops.contains(def))
        continue; // defined by an op already in the block
      inputs.insert(opr);
    }
  });
  ops.insert(op);
}

void VFFusionBlock::unfuseOp(Operation *const op) {
  if (!ops.contains(op))
    return;
  ops.remove(op);
  // The asymptotic time complexity is O(n), maybe it can be optimized to O(1).
  recomputeInputs();
}

} // namespace analysis
} // namespace mlir

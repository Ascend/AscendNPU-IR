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

SmallVector<Operation *> VFFusionBlock::getOps() const {
  return SmallVector<Operation *>(ops.begin(), ops.end());
}

// insert operand to `inputs` if it uses an operand that is defined from
// outside.
SetVector<Value> VFFusionBlock::recomputeInputs() {
  inputs.clear();
  // insert all operands.
  for (Operation *const outerOp : ops) {
    outerOp->walk<WalkOrder::PreOrder>([this](Operation *const op) -> void {
      for (auto opr : op->getOperands()) {
        inputs.insert(opr);
      }
    });
  }
  // remove operands that are defined from the FusedBlock.
  for (Operation *const outerOp : ops) {
    outerOp->walk<WalkOrder::PreOrder>([this](Operation *const op) -> void {
      for (auto opr : op->getResults()) {
        inputs.remove(opr);
      }
      for (auto &reg : op->getRegions()) {
        for (auto &block : reg.getBlocks()) {
          for (auto &blockArg : block.getArguments()) {
            inputs.remove(blockArg);
          }
        }
      }
    });
  }
  return inputs;
}

// might need to recompute inputs, consider how inputs looks like when outline
// two consecutive blocks.
SetVector<Value> VFFusionBlock::getInputs() const { return inputs; }

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
      if (all_of(res.getUsers(), [opsInside](Operation *userOp) {
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
  // NOTE: can be optimized by precompute all operations
  assert(!hasFusedUser(op) && "need to fuse operation in topological order");
  ops.insert(op);
}

} // namespace analysis
} // namespace mlir

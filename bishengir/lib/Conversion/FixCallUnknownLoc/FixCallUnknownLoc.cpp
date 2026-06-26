//===- FixCallUnknownLoc.cpp --- Fix UnknownLoc on call ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace UnknownLoc on call ops (func::CallOp / LLVM::CallOp) by inheriting
// location from the call's result users or parent ops. This fixes the LLVM IR
// verifier error: "inlinable function call in a function with debug info must
// have a !dbg location".
//
// The strategy:
//   1. Check if the function has any non-UnknownLoc op. If not, skip (no debug
//      info in this function, so no error will occur).
//   2. For each call op with UnknownLoc:
//      a. Try to find a non-UnknownLoc from the call's result users.
//      b. If not found, traverse up parent ops until a non-UnknownLoc is found.
//      c. If a valid location is found, set it and emit a warning.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/FixCallUnknownLoc/FixCallUnknownLoc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_FIXCALLUNKNOWNLOC
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Find a non-UnknownLoc location for the given operation.
/// Priority 1: from the result users (the ops that consume the call's results).
/// Priority 2: traverse up parent ops until a non-UnknownLoc is found.
static Location findNonUnknownLoc(Operation *op) {
  // Priority 1: from result users.
  for (Value result : op->getResults()) {
    for (OpOperand &use : result.getUses()) {
      Operation *userOp = use.getOwner();
      Location userLoc = userOp->getLoc();
      if (!isa<UnknownLoc>(userLoc))
        return userLoc;
    }
  }

  // Priority 2: traverse parent ops.
  Operation *parent = op->getParentOp();
  while (parent) {
    Location parentLoc = parent->getLoc();
    if (!isa<UnknownLoc>(parentLoc))
      return parentLoc;
    parent = parent->getParentOp();
  }

  return op->getLoc();
}

struct FixCallUnknownLocPass
    : public impl::FixCallUnknownLocBase<FixCallUnknownLocPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Check if the function has any non-UnknownLoc op (excluding call ops).
    // If all ops are UnknownLoc, the function won't have a DISubprogram in
    // LLVM IR, so no verifier error will occur.
    bool hasNonUnknownLoc = false;
    funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<func::CallOp, LLVM::CallOp>(op))
        return WalkResult::advance();
      if (!isa<UnknownLoc>(op->getLoc())) {
        hasNonUnknownLoc = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasNonUnknownLoc)
      return;

    // Fix all call ops with UnknownLoc.
    funcOp->walk([&](Operation *op) {
      if (!isa<func::CallOp, LLVM::CallOp>(op))
        return;
      if (!isa<UnknownLoc>(op->getLoc()))
        return;

      Location loc = findNonUnknownLoc(op);
      if (isa<UnknownLoc>(loc))
        return;

      op->setLoc(loc);
      op->emitWarning("has UnknownLoc, fixed by inheriting location from ")
          << loc;
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createFixCallUnknownLocPass() {
  return std::make_unique<FixCallUnknownLocPass>();
}

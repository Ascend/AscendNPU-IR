//===- FixCallUnknownLoc.cpp --- Fix UnknownLoc on call ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace effectively-unknown locations on func::FuncOp declarations and call
// ops (func::CallOp / LLVM::CallOp) by inheriting a location from callers,
// result users, or parent ops. This fixes the LLVM IR verifier error:
// "inlinable function call in a function with debug info must have a !dbg
// location".
//
// A location is "effectively unknown" if it is UnknownLoc, or a wrapper
// location (NameLoc / CallSiteLoc / FusedLoc) whose nested ocations are all
// unknown. This mirrors how LLVMDIScope's extractFileLoc unwraps wrapper
// locations, so a NameLoc wrapping UnknownLoc is treated as unknown and fixed
// here.
//
// The strategy:
//   1. Fix the FuncOp's own location first (from callers / body / module).
//   2. Check if the function has any non-unknown-loc op. If not, skip (no
//      debug info in this function, so no error will occur).
//   3. For each call op with an effectively-unknown location:
//      a. Try to find a non-unknown loc from the call's result users.
//      b. If not found, traverse up parent ops until a non-unknown loc is
//         found.
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

/// Check whether a location is effectively unknown: UnknownLoc, or a
/// NameLoc/CallSiteLoc/FusedLoc whose nested locations are all
/// unknown. This mirrors how LLVMDIScope's extractFileLoc unwraps wrapper
/// locations, so a NameLoc wrapping UnknownLoc is treated as unknown here.
static bool isLocUnknown(Location loc) {
  if (isa<UnknownLoc>(loc))
    return true;
  if (auto nl = dyn_cast<NameLoc>(loc))
    return isLocUnknown(nl.getChildLoc());
  if (auto cl = dyn_cast<CallSiteLoc>(loc))
    return isLocUnknown(cl.getCallee()) && isLocUnknown(cl.getCaller());
  if (auto fl = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fl.getLocations())
      if (!isLocUnknown(child))
        return false;
    return true;
  }
  return false;
}

/// Find a non-unknown-loc location for the given operation.
/// Priority 1: from the result users (the ops that consume the call's results).
/// Priority 2: traverse up parent ops until a non-unknown-loc is found.
static Location findNonUnknownLoc(Operation *op) {
  // Priority 1: from result users.
  for (Value result : op->getResults()) {
    for (OpOperand &use : result.getUses()) {
      Operation *userOp = use.getOwner();
      if (!isLocUnknown(userOp->getLoc()))
        return userOp->getLoc();
    }
  }

  // Priority 2: traverse parent ops.
  Operation *parent = op->getParentOp();
  while (parent) {
    if (!isLocUnknown(parent->getLoc()))
      return parent->getLoc();
    parent = parent->getParentOp();
  }

  return op->getLoc();
}

/// Find a non-unknown-loc location for a FuncOp declaration/function.
/// Priority 1: from its call sites (callers). For each caller, prefer the
///             call op's own loc, then walk up the caller's parent ops.
/// Priority 2: from a non-unknown-loc op inside the function body.
/// Priority 3: from the parent module's loc.
static Location findNonUnknownLocForFunc(func::FuncOp funcOp) {
  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  if (!module)
    return funcOp.getLoc();

  // Priority 1: from call sites.
  std::optional<SymbolTable::UseRange> symbolUses =
      funcOp.getSymbolUses(module);
  if (symbolUses) {
    for (const SymbolTable::SymbolUse &use : *symbolUses) {
      Operation *user = use.getUser();
      if (!isa<func::CallOp, LLVM::CallOp>(user))
        continue;
      if (!isLocUnknown(user->getLoc()))
        return user->getLoc();
      // The call op's loc is also unknown; walk up its parents.
      Operation *parent = user->getParentOp();
      while (parent) {
        if (!isLocUnknown(parent->getLoc()))
          return parent->getLoc();
        parent = parent->getParentOp();
      }
    }
  }

  // Priority 2: from a non-unknown-loc op inside the body.
  if (!funcOp.isExternal()) {
    Location bodyLoc = findNonUnknownLoc(funcOp);
    if (!isLocUnknown(bodyLoc))
      return bodyLoc;
  }

  // Priority 3: from the parent module.
  if (!isLocUnknown(module.getLoc()))
    return module.getLoc();

  return funcOp.getLoc();
}

struct FixCallUnknownLocPass
    : public impl::FixCallUnknownLocBase<FixCallUnknownLocPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Step 1: Fix the function's own location first. This matters for
    // external library function declarations (func.func private @ciface_...)
    // whose loc is NameLoc("name", UnknownLoc): they have no body so the
    // walk below would otherwise skip them, leaving a malformed DISubprogram
    // that triggers the LLVM IR verifier error.
    if (isLocUnknown(funcOp.getLoc())) {
      Location newLoc = findNonUnknownLocForFunc(funcOp);
      if (!isLocUnknown(newLoc)) {
        funcOp->setLoc(newLoc);
      }
    }

    // Step 2: Check if the function has any non-unknown-loc op (excluding
    // call ops). If all ops are unknown-loc, the function won't have a
    // DISubprogram in LLVM IR, so no verifier error will occur.
    bool hasNonUnknownLoc = false;
    funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<func::CallOp, LLVM::CallOp>(op))
        return WalkResult::advance();
      if (!isLocUnknown(op->getLoc())) {
        hasNonUnknownLoc = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasNonUnknownLoc)
      return;

    // Step 3: Fix all call ops with an effectively-unknown location.
    funcOp->walk([&](Operation *op) {
      if (!isa<func::CallOp, LLVM::CallOp>(op))
        return;
      if (!isLocUnknown(op->getLoc()))
        return;

      Location loc = findNonUnknownLoc(op);
      if (isLocUnknown(loc))
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

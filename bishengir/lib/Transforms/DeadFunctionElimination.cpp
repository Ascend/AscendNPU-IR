//===-- DeadFunctionElimination.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dead-function-elimination"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace bishengir {
#define GEN_PASS_DEF_DEADFUNCTIONELIMINATION
#include "bishengir/Transforms/Passes.h.inc"
} // namespace bishengir

using namespace mlir;

void bishengir::eliminateDeadFunctions(
    ModuleOp module, const bishengir::DeadFunctionEliminationOptions &options) {
  module->walk([&](FunctionOpInterface funcLikeOp) {
    if (!SymbolTable::symbolKnownUseEmpty(funcLikeOp, module)) {
      LDBG("Symbol: @" << funcLikeOp.getName() << " is still in use.");
      return;
    }

    if (!options.filterFn(funcLikeOp)) {
      LDBG("Symbol: @" << funcLikeOp.getName()
                       << " doesn't satisfy requirement.");
      return;
    }
    funcLikeOp.erase();
  });
}

namespace {

struct DeadFunctionEliminationPass
    : public bishengir::impl::DeadFunctionEliminationBase<
          DeadFunctionEliminationPass> {
  explicit DeadFunctionEliminationPass(
      const bishengir::DeadFunctionEliminationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    bishengir::eliminateDeadFunctions(getOperation(), options);
  }

private:
  bishengir::DeadFunctionEliminationOptions options;
};

} // namespace

std::unique_ptr<mlir::Pass> bishengir::createDeadFunctionEliminationPass(
    const bishengir::DeadFunctionEliminationOptions &options) {
  return std::make_unique<DeadFunctionEliminationPass>(options);
}

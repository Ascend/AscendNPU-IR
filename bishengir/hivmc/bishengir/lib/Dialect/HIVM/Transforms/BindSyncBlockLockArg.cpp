//===---------------------- BindSyncBlockLockArg.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "hivm-bind-sync-block-lock-arg"

namespace mlir {
#define GEN_PASS_DEF_BINDSYNCBLOCKLOCKARG
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace mlir::hivm {
class BindSyncBlockLockArgPass
    : public impl::BindSyncBlockLockArgBase<BindSyncBlockLockArgPass> {
public:
  using BindSyncBlockLockArgBase<
      BindSyncBlockLockArgPass>::BindSyncBlockLockArgBase;
  void runOnOperation() override;
};

void BindSyncBlockLockArgPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  std::optional<BlockArgument> syncBlockLockArg = hacc::utils::getBlockArgument(
      funcOp, hacc::KernelArgType::kSyncBlockLock);

  if (!syncBlockLockArg.has_value()) {
    return;
  }

  auto bindResult =
      funcOp.walk([&syncBlockLockArg](hivm::CreateSyncBlockLockOp op) {
        if (!op.getLockArg()) {
          op.getLockArgMutable().assign(syncBlockLockArg.value());
        }

        return WalkResult::advance();
      });
  if (bindResult == WalkResult::interrupt())
    return signalPassFailure();
}
} // namespace mlir::hivm

std::unique_ptr<Pass> mlir::hivm::createBindSyncBlockLockArgPass() {
  return std::make_unique<BindSyncBlockLockArgPass>();
}

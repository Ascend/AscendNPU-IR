//===---------------------- BindSyncBlockLockArg.cpp ----------------------===//
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

//===------------------------ BindWorkSpaceArg.cpp ------------------------===//
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
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "hivm-bind-workspace-arg"

namespace mlir {
#define GEN_PASS_DEF_BINDWORKSPACEARG
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

class BindWorkSpaceArgPass
    : public impl::BindWorkSpaceArgBase<BindWorkSpaceArgPass> {
public:
  using BindWorkSpaceArgBase<BindWorkSpaceArgPass>::BindWorkSpaceArgBase;
  void runOnOperation() override;
};

void BindWorkSpaceArgPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  std::optional<BlockArgument> workspaceArg =
      hacc::utils::getBlockArgument(funcOp, hacc::KernelArgType::kWorkspace);

  auto bindResult =
      funcOp.walk([&](bishengir::memref_ext::AllocWorkspaceOp op) {
        if (!op.getWorkspaceArg()) {
          if (!workspaceArg.has_value()) {
            op->emitOpError("failed to bind workspace argument");
            return WalkResult::interrupt();
          }
          op.getWorkspaceArgMutable().assign(workspaceArg.value());
        }

        return WalkResult::advance();
      });
  if (bindResult == WalkResult::interrupt())
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createBindWorkSpaceArgPass() {
  return std::make_unique<BindWorkSpaceArgPass>();
}

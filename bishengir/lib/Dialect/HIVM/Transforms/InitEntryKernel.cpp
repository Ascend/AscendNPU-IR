//===----------------------- InitEntryKernel.cpp --------------------------===//
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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_INITENTRYKERNEL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct InitEntryKernelPass
    : public impl::InitEntryKernelBase<InitEntryKernelPass> {
  using InitEntryKernelBase<InitEntryKernelPass>::InitEntryKernelBase;

public:
  void runOnOperation() override;
};
} // namespace

void InitEntryKernelPass::runOnOperation() {
  auto funcOp = getOperation();
  if (!hacc::utils::isDeviceEntry(funcOp))
    return;

  OpBuilder builder(&getContext());
  builder.setInsertionPointToStart(&funcOp.getBlocks().front());
  builder.create<hivm::SetMaskNormOp>(funcOp->getLoc());
}

std::unique_ptr<Pass> mlir::hivm::createInitEntryKernelPass() {
  return std::make_unique<InitEntryKernelPass>();
}

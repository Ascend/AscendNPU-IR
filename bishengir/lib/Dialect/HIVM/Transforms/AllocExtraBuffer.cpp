//===- AllocExtraBuffer.cpp -----------------------------------------------===//
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
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "hivm-alloc-extra-buffer"

namespace mlir {
#define GEN_PASS_DEF_ALLOCEXTRABUFFER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

struct AllocExtraBufferPass
    : public mlir::impl::AllocExtraBufferBase<AllocExtraBufferPass> {
public:
  void runOnOperation() override;
};
} // namespace

void AllocExtraBufferPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  auto walkResult = funcOp.walk([](ExtraBufferOpInterface op) {
    if (failed(op.allocExtraBuffersIfPossible()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createAllocExtraBufferPass() {
  return std::make_unique<AllocExtraBufferPass>();
}

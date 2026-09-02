//===- AllocExtraBuffer.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  auto mod = funcOp->getParentOfType<ModuleOp>();
  if (!mod)
    return;
  auto walkResult = funcOp.walk([](ExtraBufferOpInterface op) {
    if (failed(op.allocExtraBuffersIfPossible())) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createAllocExtraBufferPass() {
  return std::make_unique<AllocExtraBufferPass>();
}

//===- HoistConstantMasks.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_HOISTCONSTANTMASKS
#include "bishengir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct HoistConstantMasksPass
    : public impl::HoistConstantMasksBase<HoistConstantMasksPass> {
  using HoistConstantMasksBase<HoistConstantMasksPass>::HoistConstantMasksBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    SmallVector<vector::ConstantMaskOp> masks;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Nested isolated operations cannot capture a mask from this function.
      if (op != func.getOperation() &&
          op->hasTrait<OpTrait::IsIsolatedFromAbove>())
        return WalkResult::skip();
      if (auto mask = dyn_cast<vector::ConstantMaskOp>(op))
        masks.push_back(mask);
      return WalkResult::advance();
    });

    // Constant masks are pure and have no operands. Place them before the
    // arithmetic that can reuse them, and let a subsequent CSE merge
    // duplicates. Reverse iteration preserves the original walk order at the
    // block start.
    IRRewriter rewriter(&getContext());
    Block &entry = func.getBody().front();
    for (vector::ConstantMaskOp mask : llvm::reverse(masks))
      if (mask.getOperation() != &entry.front())
        rewriter.moveOpBefore(mask, &entry.front());
  }
};
} // namespace

std::unique_ptr<Pass> vector::createHoistConstantMasksPass() {
  return std::make_unique<HoistConstantMasksPass>();
}

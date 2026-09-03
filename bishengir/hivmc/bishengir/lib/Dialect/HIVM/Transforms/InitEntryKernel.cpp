//===----------------------- InitEntryKernel.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
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

void configureEntryForRegbaseArch(OpBuilder &b, Location loc) {
  // SPR.CTRL[60] affects the following two types of Convert instructions:
  // 1. `VCVTFI` and `VCVTII` from wider dynamic range to narrower dynamic
  // range, e.g. VCVTFI.f322s32
  // 2. `VCVTFF` from wider dynamic range to narrower dynamic range where fdst
  // != f32, e.g. VCVTFF.f322f16
  //
  // For these cases:
  // - When SPR.CTRL[60] is b1: Saturation mode is determined by other CTRL
  // register bits
  // - When SPR.CTRL[60] is b0: Saturation mode is determined by instruction's
  // sat params
  // // We choose to adopt the latter behavior. Since hardware initializes
  // SPR.CTRL[60] to b1 by default, we explicitly set it to b0.
  b.create<hivm::SetCtrlOp>(loc,
                            /**enable**/ false,
                            /**idx**/ hivm::OverrideSaturationBit);

  // For `VCVTFF` from narrower to wider dynamic range where fdst != f32 (e.g.
  // VCVTFF.f162bf16), and `VTRC` includes VTRC.f16/VTRC.bf16, the saturation
  // mode is controlled solely by SPR.CTRL[48]. We set SPR.CTRL[48] to b1 to
  // enable non-saturation mode by default.
  b.create<hivm::SetCtrlOp>(loc,
                            /**enable**/ true,
                            /**idx**/ hivm::SaturationControlBit);
}

void configureEntryForMembaseArch(OpBuilder &b, Location loc) {
  // initialize with "set_mask_norm()" in V220(910B)
  b.create<hivm::SetMaskNormOp>(loc);
}

} // namespace

void InitEntryKernelPass::runOnOperation() {
  auto funcOp = getOperation();
  if (!hacc::utils::isDeviceEntry(funcOp))
    return;

  auto mod = funcOp->getParentOfType<ModuleOp>();

  OpBuilder builder(&getContext());
  builder.setInsertionPointToStart(&funcOp.getBlocks().front());
  Location loc = funcOp->getLoc();
  // TODO: distinguish between 310b4 and david if needed
  if (hacc::utils::isRegBasedArch(mod)) {
    configureEntryForRegbaseArch(builder, loc);
  } else {
    configureEntryForMembaseArch(builder, loc);
  }
}

std::unique_ptr<Pass> mlir::hivm::createInitEntryKernelPass() {
  return std::make_unique<InitEntryKernelPass>();
}

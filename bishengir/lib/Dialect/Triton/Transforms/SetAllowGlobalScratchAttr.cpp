//===- SetAllowGlobalScratchAttr.cpp ------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that set the allow global scratch Attr from
// --enable-global-scratch-allocation
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_SETALLOWGLOBALSCRATCHATTR
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

class SetAllowGlobalScratchAttrPass
    : public impl::SetAllowGlobalScratchAttrBase<SetAllowGlobalScratchAttrPass> {
public:
  explicit SetAllowGlobalScratchAttrPass(
      const SetAllowGlobalScratchAttrOptions &options)
      : impl::SetAllowGlobalScratchAttrBase<SetAllowGlobalScratchAttrPass>(
            options) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    mod->setAttr(AttrEnableGlobalScratchAllocationName,
                 builder.getBoolAttr(enableGlobalScratchAllocation));
  }
};

} // namespace

std::unique_ptr<Pass> createSetAllowGlobalScratchAttrPass(
    const SetAllowGlobalScratchAttrOptions &options) {
  return std::make_unique<SetAllowGlobalScratchAttrPass>(options);
}

} // namespace triton
} // namespace bishengir

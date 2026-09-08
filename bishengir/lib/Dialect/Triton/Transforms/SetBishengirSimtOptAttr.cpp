//===- SetBishengirSimtOptAttr.cpp ------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that sets the SIMT optimization attribute from
// --simt-optimization-mode.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_SETBISHENGIRSIMTOPTATTR
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

class SetBishengirSimtOptAttrPass
    : public impl::SetBishengirSimtOptAttrBase<SetBishengirSimtOptAttrPass> {
public:
  explicit SetBishengirSimtOptAttrPass(
      const SetBishengirSimtOptAttrOptions &options)
      : impl::SetBishengirSimtOptAttrBase<SetBishengirSimtOptAttrPass>(
            options) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    mod->setAttr(AttrSimtOptimizationModeName,
                 builder.getI32IntegerAttr(simtOptimizationMode));
  }
};

} // namespace

std::unique_ptr<Pass> createSetBishengirSimtOptAttrPass(
    const SetBishengirSimtOptAttrOptions &options) {
  return std::make_unique<SetBishengirSimtOptAttrPass>(options);
}

} // namespace triton
} // namespace bishengir

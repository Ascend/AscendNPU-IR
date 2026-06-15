//===- RemoveAnnotationMark.cpp - Erase annotation.mark ops -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass removes every `annotation.mark` op in the module.  `annotation.mark`
// carries only non-semantic metadata about a value and has no results, so each
// op can simply be erased once its metadata is no longer needed.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace bishengir::triton {
#define GEN_PASS_DEF_REMOVEANNOTATIONMARK
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

class RemoveAnnotationMarkPass
    : public impl::RemoveAnnotationMarkBase<RemoveAnnotationMarkPass> {
public:
  using RemoveAnnotationMarkBase::RemoveAnnotationMarkBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<annotation::MarkOp> marks;
    mod.walk([&](annotation::MarkOp markOp) { marks.push_back(markOp); });
    for (annotation::MarkOp markOp : marks)
      markOp.erase();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createRemoveAnnotationMarkPass() {
  return std::make_unique<RemoveAnnotationMarkPass>();
}

} // namespace bishengir::triton

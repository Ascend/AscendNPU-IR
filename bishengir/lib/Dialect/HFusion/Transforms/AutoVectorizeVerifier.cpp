//===- AutoVectorizeVerifier.cpp - Verify auto vectorization result -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"

#include <memory>

namespace mlir {
#define GEN_PASS_DEF_AUTOVECTORIZEVERIFIER
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
    
bool hasVectorOperandOrResult(Operation *op) {
  auto isVectorType = [](Type type) { return isa<VectorType>(type); };
  return llvm::any_of(op->getOperandTypes(), isVectorType) ||
         llvm::any_of(op->getResultTypes(), isVectorType);
}

class AutoVectorizeVerifier
    : public impl::AutoVectorizeVerifierBase<AutoVectorizeVerifier> {
public:
  using AutoVectorizeVerifierBase::AutoVectorizeVerifierBase;

  void runOnOperation() override {
    WalkResult result = getOperation()->walk<WalkOrder::PreOrder>(
        [](Operation *op) {
          if (auto func = dyn_cast<func::FuncOp>(op)) {
            if (hivm::isVF(func)) {
              return WalkResult::skip();
            }
            return WalkResult::advance();
          }

          if (!hasVectorOperandOrResult(op)) {
            return WalkResult::advance();
          }

          op->emitError("unexpected vector operation outside vector function");
          return WalkResult::interrupt();
        });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createAutoVectorizeVerifierPass() {
  return std::make_unique<AutoVectorizeVerifier>();
}
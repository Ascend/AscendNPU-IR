//===- AutoVectorizeVerifier.cpp - Verify auto vectorization result -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Verify.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include <memory>

namespace mlir {
#define GEN_PASS_DEF_AUTOVECTORIZEVERIFIER
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class AutoVectorizeVerifierPass
    : public impl::AutoVectorizeVerifierBase<AutoVectorizeVerifierPass> {
public:
  using AutoVectorizeVerifierBase::AutoVectorizeVerifierBase;

  void runOnOperation() override {
    hfusion::AutoVectorizeVerifier verifier;
    if (failed(verifier.verifyFreeVectorRegion(verifyFreeVectorRegion)
                   .verifyFreeVectorFunc(verifyFreeVectorFunc)
                   .check(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createAutoVectorizeVerifierPass() {
  return std::make_unique<AutoVectorizeVerifierPass>();
}

//===- EnableAscendDPXMMA.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Validates that DotOps do not exceed size limits for the Ascend DPX MMA path.
// DotOps with M > 32, N > 32, or K > 64 will emit an error since the
// tt.dot lowering is still experimental.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_ENABLEASCENDDPXMMA
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace triton
} // namespace bishengir

using namespace mlir;
using namespace mlir::triton;

namespace bishengir {
namespace triton {

namespace {

class EnableAscendDPXMMAPass
    : public impl::EnableAscendDPXMMABase<EnableAscendDPXMMAPass> {
public:
  void runOnOperation() override {
    // Validate all DotOps - fail if any exceed size limits
    WalkResult result = getOperation()->walk([](DotOp dotOp) -> WalkResult {
      auto aType = cast<RankedTensorType>(dotOp.getA().getType());
      auto bType = cast<RankedTensorType>(dotOp.getB().getType());

      int64_t m = aType.getShape()[0];
      int64_t k = aType.getShape()[1];
      int64_t n = bType.getShape()[1];

      int64_t maxM = 64;
      int64_t maxN = 64;
      int64_t maxK = 32;
      int64_t total = maxM * maxN * maxK; 

      int64_t product = m * n * k;
      if (product > total) {
        return dotOp.emitError()
               << "tt.dot lowering is still experimental and in the works. "
               << "The current dot is too large (MxNxK=" << m << "x" << n << "x"
               << k << "). "
               << "Try to minimize it such that M <= " << maxM
               << ", N <= " << maxN << ", K <= " << maxK
               << " and preferably M,N,K <= 32";
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createEnableAscendDPXMMAPass() {
  return std::make_unique<EnableAscendDPXMMAPass>();
}

} // namespace triton
} // namespace bishengir
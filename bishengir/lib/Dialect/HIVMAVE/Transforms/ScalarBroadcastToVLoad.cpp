//===---- ScalarBroadcastToVLoad.cpp - Scalar Broadcast to VLoad Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-scalar-broadcast-to-vload"

namespace mlir {
#define GEN_PASS_DEF_SCALARBROADCASTTOVLOAD
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivmave;

/// Scalar + broadcast to vload
/// from:
/// %scalar = memref.load %memref_base[%idx] :
///        memref<?xf32, #hivm.address_space<ub>> -> f32
/// %broadcast = ave.hir.scalar_broadcast %scalar :
///        f32 -> vector<64xf32>
/// to:
/// %vload = ave.hir.vload <BRC_B32> %memref_base[%idx] :
///        memref<?xf32, #hivm.address_space<ub>> into vector<64xf32>
struct ScalarBroadcastToVLoadPattern
    : public OpRewritePattern<VFBroadcastScalarOp> {
  ScalarBroadcastToVLoadPattern(MLIRContext *context)
      : OpRewritePattern<VFBroadcastScalarOp>(context) {}

  LogicalResult matchAndRewrite(VFBroadcastScalarOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    Value scalarInput = broadcastOp.getSrc();
    if (!scalarInput)
      return failure();

    auto loadOp = dyn_cast_or_null<memref::LoadOp>(scalarInput.getDefiningOp());
    if (!loadOp)
      return failure();

    VectorType outputVecType =
        dyn_cast_or_null<VectorType>(broadcastOp.getRes().getType());
    if (!outputVecType)
      return failure();

    LoadDistAttr loadPatternAttr;
    Type elemType = outputVecType.getElementType();
    if (elemType.isF32() || elemType.isInteger(32)) {
      auto pattern = symbolizeLoadDist("BRC_B32");
      if (!pattern)
        return failure();
      loadPatternAttr = LoadDistAttr::get(rewriter.getContext(), *pattern);
    } else if (elemType.isF16() || elemType.isInteger(16) ||
               elemType.isBF16()) {
      auto pattern = symbolizeLoadDist("BRC_B16");
      if (!pattern)
        return failure();
      loadPatternAttr = LoadDistAttr::get(rewriter.getContext(), *pattern);
    } else if (elemType.isInteger(8) || elemType.isFloat8E4M3FN() ||
               elemType.isFloat8E5M2()) {
      auto pattern = symbolizeLoadDist("BRC_B8");
      if (!pattern)
        return failure();
      loadPatternAttr = LoadDistAttr::get(rewriter.getContext(), *pattern);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Unsupported element type for scalar_broadcast: "
                 << elemType << "\n");
      return failure();
    }

    Value memrefBase = loadOp.getMemRef();
    ValueRange loadIndices = loadOp.getIndices();

    Location loc = broadcastOp.getLoc();
    auto vloadOp = rewriter.create<VFLoadOp>(
        loc, outputVecType, loadPatternAttr, memrefBase, loadIndices);

    broadcastOp.getRes().replaceAllUsesWith(vloadOp.getRes());

    return success();
  }
};

namespace {
struct ScalarBroadcastToVLoadPass
    : public impl::ScalarBroadcastToVLoadBase<ScalarBroadcastToVLoadPass> {
  using ScalarBroadcastToVLoadBase<
      ScalarBroadcastToVLoadPass>::ScalarBroadcastToVLoadBase;

public:
  void runOnOperation() override;
};
} // namespace

void ScalarBroadcastToVLoadPass::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<ScalarBroadcastToVLoadPattern>(context);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> hivmave::createScalarBroadcastToVLoadPass() {
  return std::make_unique<ScalarBroadcastToVLoadPass>();
}
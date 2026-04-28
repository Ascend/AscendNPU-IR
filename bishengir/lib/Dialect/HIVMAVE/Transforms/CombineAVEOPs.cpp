//===------------------------- CombineAVEOPs.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_COMBINEAVEOPS
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "combine-ave-ops"

using namespace mlir;

namespace {

struct CombineAVEOPsPass
    : public impl::CombineAVEOPsBase<CombineAVEOPsPass> {
public:
  explicit CombineAVEOPsPass(const CombineAVEOPsOptions &options)
      : CombineAVEOPsBase(options) {}
  void runOnOperation() override;
};

template <typename HivmVFTenaryOp, typename HivmVFPreOp, typename HivmVFSucOp>
struct TenaryOpPattern : public OpRewritePattern<HivmVFSucOp> {
  using OpRewritePattern<HivmVFSucOp>::OpRewritePattern;
  /// Combine mul and add to vmula
  /// from:
  /// %0 = ave.hir.vload <NORM> %arg0[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %1 = ave.hir.vload <NORM> %arg1[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %2 = ave.hir.pge <ALL> : vector<64xi1>
  /// %3 = ave.hir.vmul %0, %1, %2 : vector<64xf32>, vector<64xi1>
  /// %4 = ave.hir.vload <NORM> %arg2[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %5 = ave.hir.pge <ALL> : vector<64xi1>
  /// %6 = ave.hir.vadd %3, %4, %5 : vector<64xf32>, vector<64xi1>
  /// to:
  /// %0 = ave.hir.vload <NORM> %arg0[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %1 = ave.hir.vload <NORM> %arg1[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %2 = ave.hir.vload <NORM> %arg2[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %3 = ave.hir.pge <ALL> : vector<64xi1>
  /// %4 = ave.hir.vmula %2, %0, %1, %3 : vector<64xf32>, vector<64xi1>
  LogicalResult matchAndRewrite(HivmVFSucOp op,
                                PatternRewriter &rewriter) const override {
    // The left operand comes from PreOp.
    if (HivmVFPreOp preOp =
            dyn_cast_or_null<HivmVFPreOp>(op.getLhs().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<HivmVFTenaryOp>(
          op, op.getResult().getType(), op.getRhs(), preOp.getLhs(),
          preOp.getRhs(), op.getMask(), nullptr);
      return success();
    }
    // The right operand comes from PreOp.
    if (HivmVFPreOp preOp =
            dyn_cast_or_null<HivmVFPreOp>(op.getRhs().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<HivmVFTenaryOp>(
          op, op.getResult().getType(), op.getLhs(), preOp.getLhs(),
          preOp.getRhs(), op.getMask(), nullptr);
      return success();
    }
    return failure();
  }
};

struct SubExpToExpdifOpPattern : public OpRewritePattern<hivmave::VFExpOp> {
public:
  using OpRewritePattern<hivmave::VFExpOp>::OpRewritePattern;

  /// Combine sub and exp to expdif
  /// from:
  /// %0 = ave.hir.vload <NORM> %arg0[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %1 = ave.hir.vload <NORM> %arg1[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %2 = ave.hir.pge <ALL> : vector<64xi1>
  /// %3 = ave.hir.vsub %0, %1, %2 : vector<64xf32>, vector<64xi1>
  /// %4 = ave.hir.pge <ALL> : vector<64xi1>
  /// %5 = ave.hir.vexp %3, %4 : vector<64xf32>, vector<64xi1>
  /// to:
  /// %0 = ave.hir.vload <NORM> %arg0[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %1 = ave.hir.vload <NORM> %arg1[%c0] :
  ///        memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  /// %2 = ave.hir.pge <ALL> : vector<64xi1>
  /// %3 = ave.hir.vexpdif %0, %1, %2 : vector<64xf32>, vector<64xi1>
  LogicalResult matchAndRewrite(hivmave::VFExpOp op,
                                PatternRewriter &rewriter) const override {
    auto expSrc = op.getSrc().getDefiningOp();
    if (hivmave::VFSubOp subOp = dyn_cast<hivmave::VFSubOp>(expSrc)) {
      rewriter.replaceOpWithNewOp<hivmave::VFExpdifOp>(
          op, op.getResult().getType(), subOp.getLhs(), subOp.getRhs(),
          op.getMask(), nullptr);
      return success();
    }
    return failure();
  }
};
} // namespace

void CombineAVEOPsPass::runOnOperation() {
  auto funcOp = getOperation();
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  bool archIs950 = hacc::utils::isAscend950(moduleOp);
  if (!archIs950)
    return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  if (enableFusedMultiplyAdd) {
    patterns.add<TenaryOpPattern<hivmave::VFMulaOp, hivmave::VFMulOp, hivmave::VFAddOp>>(ctx);
  }
  patterns.add<SubExpToExpdifOpPattern>(ctx);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> hivmave::createCombineAVEOPsPass(const CombineAVEOPsOptions &options) {
  return std::make_unique<CombineAVEOPsPass>(options);
}

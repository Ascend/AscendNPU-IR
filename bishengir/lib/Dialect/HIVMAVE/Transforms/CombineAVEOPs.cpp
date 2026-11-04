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

// Which element types a mul+add fusion pattern applies to. `Any` keeps the
// historical behaviour (fuse every element type into the same ternary op) and
// is used for every target except Ascend960.
enum class FuseTypePolicy { Any, FloatOnly, NonFloatOnly };

template <typename HivmVFTenaryOp, typename HivmVFPreOp, typename HivmVFSucOp,
          FuseTypePolicy TypePolicy = FuseTypePolicy::Any>
struct TenaryOpPattern : public OpRewritePattern<HivmVFSucOp> {
  using OpRewritePattern<HivmVFSucOp>::OpRewritePattern;
  /// Combine mul and add to a tenary fused multiply-add op
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
    // Only fuse when the result element type matches the requested class.
    Type elementType =
        cast<VectorType>(op.getResult().getType()).getElementType();
    bool isFloat =
        elementType.isF16() || elementType.isF32() || elementType.isBF16();
    if (TypePolicy == FuseTypePolicy::FloatOnly && !isFloat)
      return failure();
    if (TypePolicy == FuseTypePolicy::NonFloatOnly && isFloat)
      return failure();

    // The left operand comes from PreOp.
    if (HivmVFPreOp preOp =
            dyn_cast_or_null<HivmVFPreOp>(op.getLhs().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<HivmVFTenaryOp>(op, op.getResult().getType(),
                                                  op.getRhs(), preOp.getLhs(),
                                                  preOp.getRhs(), op.getMask());
      return success();
    }
    // The right operand comes from PreOp.
    if (HivmVFPreOp preOp =
            dyn_cast_or_null<HivmVFPreOp>(op.getRhs().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<HivmVFTenaryOp>(op, op.getResult().getType(),
                                                  op.getLhs(), preOp.getLhs(),
                                                  preOp.getRhs(), op.getMask());
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
          op.getMask());
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
  bool archIs960 = hacc::utils::isAscend960(moduleOp);
  if (!archIs950 && !archIs960)
    return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  if (enableFusedMultiplyAdd) {
    if (archIs960) {
      // Ascend960 (V121) has vfma: non-float mul+add keeps fusing to vmula,
      // while floats fuse to vfma.
      patterns
          .add<TenaryOpPattern<hivmave::VFMulaOp, hivmave::VFMulOp,
                               hivmave::VFAddOp, FuseTypePolicy::NonFloatOnly>>(
              ctx);
      patterns
          .add<TenaryOpPattern<hivmave::VFmaOp, hivmave::VFMulOp,
                               hivmave::VFAddOp, FuseTypePolicy::FloatOnly>>(
              ctx);
    } else {
      // Every other regbased target keeps the original behaviour: one vmula
      // fusion pattern for all element types.
      patterns.add<TenaryOpPattern<hivmave::VFMulaOp, hivmave::VFMulOp,
                                   hivmave::VFAddOp>>(ctx);
    }
  }
  patterns.add<SubExpToExpdifOpPattern>(ctx);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> hivmave::createCombineAVEOPsPass(const CombineAVEOPsOptions &options) {
  return std::make_unique<CombineAVEOPsPass>(options);
}

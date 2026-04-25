//===-------------------- LegalizeOptHIVMAVE.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "legalize-opt-hivmave"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_LEGALIZEOPTHIVMAVE
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivmave;

/// Legalize AVE i1 type vcmp NE -> pxor
/// Legalize AVE i1 type vcmp EQ -> pxor and pnot
/// clang-format off
/// vcmp NE Before transform:
/// ```mlir
/// %res_ne = ave.hir.vcmp <NE> %0, %1, %mask : vector<64xi1>, vector<64xi1> ->
///                                             vector<64xi1>
/// ```
/// vcmp NE After transform:
/// ```mlir
/// %res_ne = ave.hir.preg.xor <b8> %0, %1, %mask : vector<64xi1>
/// ```
/// vcmp EQ Before transform:
/// ```mlir
/// %res_eq = ave.hir.vcmp <EQ> %2, %3, %mask: vector<64xi1>, vector<64xi1> ->
///                                            vector<64xi1>
/// ```
/// vcmp EQ After transform:
/// ```mlir
/// %res_xor = ave.hir.preg.xor <b8> %2, %3, %mask : vector<64xi1>
/// %res_eq = ave.hir.preg.not <b8> %res_xor, %mask : vector<64xi1>
/// ```
/// clang-format on
struct LegalizeHIVMAVECmpPattern : public OpRewritePattern<VFCmpOp> {
  using OpRewritePattern<VFCmpOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(VFCmpOp cmpOp,
                                PatternRewriter &rewriter) const override {
    Type elemType =
        dyn_cast<VectorType>(cmpOp.getLhs().getType()).getElementType();
    if (elemType != rewriter.getI1Type()) {
      return failure();
    }
    hivmave::CmpType cmpType = cmpOp.getCmp();
    if (cmpType != CmpType::EQ && cmpType != CmpType::NE) {
      return failure();
    }
    Location loc = cmpOp.getLoc();
    VectorType resType = dyn_cast<VectorType>(cmpOp.getRes().getType());
    auto elemBitWidthAttr =
        MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8);
    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    Value mask = cmpOp.getMask();
    hivmave::PregXorOp pXorOp = rewriter.create<hivmave::PregXorOp>(
        loc, resType, elemBitWidthAttr, lhs, rhs, mask);
    if (cmpType == CmpType::NE) {
      rewriter.replaceOp(cmpOp, pXorOp.getRes());
    } else if (cmpType == CmpType::EQ) {
      hivmave::PregNotOp pNotOp = rewriter.create<hivmave::PregNotOp>(
          loc, resType, elemBitWidthAttr, pXorOp.getRes(), mask);
      rewriter.replaceOp(cmpOp, pNotOp.getRes());
    }
    return success();
  }
};

namespace {
struct LegalizeOptHIVMAVEPass
    : public impl::LegalizeOptHIVMAVEBase<LegalizeOptHIVMAVEPass> {
  using LegalizeOptHIVMAVEBase<LegalizeOptHIVMAVEPass>::LegalizeOptHIVMAVEBase;

public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LegalizeHIVMAVECmpPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> hivmave::createLegalizeOptHIVMAVEPass() {
  return std::make_unique<LegalizeOptHIVMAVEPass>();
}
//===------------- ProcessVsstb.cpp - process vsstb op --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ave-process-membar"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_PROCESSMEMBAR
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::annotation;
using namespace mlir::hivmave;

namespace {
class MarkOpLowering : public OpRewritePattern<MarkOp> {
public:
  explicit MarkOpLowering(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(MarkOp op,
                                PatternRewriter &rewriter) const override {
    /// Before conversion:
    /// ```mlir
    ///   annotation.mark %c0_i64 {SYNC_IN_VF = "VV_ALL"} : i64
    /// ```
    /// After conversion:
    /// ```mlir
    //    %c0_i32_0 = arith.constant 0 : i32
    ///   ave.hir.membar %c0_i32_0
    /// ```
    auto loc = op->getLoc();
    auto attrDict = op->getAttrDictionary();
    if (!attrDict.empty() && attrDict.contains("SYNC_IN_VF")) {
      auto syncAttr = op->getAttrOfType<StringAttr>("SYNC_IN_VF");
      if (syncAttr) {
        StringRef name = syncAttr.getValue();
        std::string str = name.str();
        Value cstValue = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(
                     mlir::hivm::membarType.find(str)->second));
        rewriter.replaceOpWithNewOp<mlir::hivmave::VFMemBarOp>(op, cstValue);
        return success();
      }
    }
    return failure();
  }
};

struct processMembarPass : public impl::ProcessMembarBase<processMembarPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<MarkOpLowering>(context);

    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<Pass> hivmave::createProcessMembarPass() {
  return std::make_unique<processMembarPass>();
}
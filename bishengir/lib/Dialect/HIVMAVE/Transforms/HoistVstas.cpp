//===-------------------- HoistVstas.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
 
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>
 
namespace mlir {
#define GEN_PASS_DEF_HOISTVSTAS
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir
 
using namespace mlir;
using namespace mlir::hivmave;
 
#define DEBUG_TYPE "hoist-vstas"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
 
namespace {
 
// The hardware spec limit is 4, leaving one for use inside a for loop
// TODO: Need to do registers pressure analysis.
static constexpr int64_t kMaxInitAlignDataForHoist = 3;

struct HoistVstasPattern : public OpRewritePattern<hivm_regbaseintrins::VstasInstrOp> {
  using OpRewritePattern::OpRewritePattern;

  // Per-func hoist count. Keyed by FuncOp's underlying Operation*.
  // Fresh DenseMap is created each time runOnOperation constructs a new pattern.
  mutable DenseMap<Operation *, int64_t> funcHoistedCount;

  LogicalResult matchAndRewrite(hivm_regbaseintrins::VstasInstrOp vstasOp,
                                PatternRewriter &rewriter) const override {
    // 1. Validate Scope: Ensure operation is within an SCF loop.
    auto forOp = dyn_cast<scf::ForOp>(vstasOp->getParentOp());
    if (!forOp) return failure();
 
    // 2. Validate Operands
    if (vstasOp.getNumOperands() != 4) return failure();
 
    Value alignedVec = vstasOp.getOperand(0);
    Value basePtr = vstasOp.getOperand(1);
 
    // 3. Trace Def-Use Chain: VSTAS -> ExtractValue -> VSTUS
    auto extractOp = basePtr.getDefiningOp<LLVM::ExtractValueOp>();
    if (!extractOp || extractOp.getPosition().size() != 1 || 
        extractOp.getPosition()[0] != 1) return failure();
 
    auto vstusResult = extractOp.getOperand().getDefiningOp();
    if (!vstusResult || 
        vstusResult->getName().getStringRef() != "hivm_regbaseintrins.intr.hivm.vstus.post.f32") {
      return failure();
    }
    auto vstusOp = cast<Operation*>(vstusResult);
 
    // 4. Only hoist if the "hivm.is_continuous" attribute is present.
    if (!vstasOp->hasAttr("hivm.is_continuous")) {
        return failure(); 
    }
 
    LLVM_DEBUG(DBGS() << "Found is_continuous marker. Initiating hoist.\n");
 
    // 5.  Ensure the vector data also originates from the same VSTUS op.
    auto extractVecOp = alignedVec.getDefiningOp<LLVM::ExtractValueOp>();
    if (!extractVecOp || extractVecOp.getPosition().size() != 1 || 
        extractVecOp.getPosition()[0] != 0) return failure();
 
    if (extractVecOp.getOperand().getDefiningOp() != vstusResult) return failure();
 
    // Count hoisted init.vector.align.data ops per-func. If the count
    // reaches the threshold, skip hoisting to avoid excessive loop-carried
    // variables.
    auto funcOp = vstasOp->getParentOfType<func::FuncOp>();
    if (funcHoistedCount[funcOp] >= kMaxInitAlignDataForHoist)
      return failure();
    ++funcHoistedCount[funcOp];

    // 6. Perform Hoisting Transformation
    rewriter.setInsertionPoint(forOp);
    Value initAlignedVec = vstusOp->getOperand(3);
    Operation *initOp = initAlignedVec.getDefiningOp();
    if (initOp && forOp->isAncestor(initOp)) {
        initOp->moveBefore(forOp);
    }
    Value initPtr = rewriter.create<LLVM::UndefOp>(vstasOp.getLoc(), basePtr.getType());

    SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(), forOp.getInitArgs().end());
    newInitArgs.push_back(initAlignedVec);
    newInitArgs.push_back(initPtr);

    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        newInitArgs,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) { /* Empty Body */ });

    SmallVector<Value> mergeArgs;
    mergeArgs.push_back(newForOp.getInductionVar());
    auto newRegionArgs = newForOp.getRegionIterArgs();
    mergeArgs.append(newRegionArgs.begin(), newRegionArgs.end() - 2);
    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(), mergeArgs);

    vstusOp->setOperand(3, newRegionArgs[newRegionArgs.size() - 2]);
    Operation *oldTerminator = newForOp.getBody()->getTerminator();
    rewriter.setInsertionPoint(oldTerminator);
    
    SmallVector<Value> yieldValues(oldTerminator->getOperands().begin(), oldTerminator->getOperands().end());
    yieldValues.push_back(vstasOp.getOperand(0));
    yieldValues.push_back(vstasOp.getOperand(1));
    rewriter.create<scf::YieldOp>(
        vstasOp.getLoc(), 
        yieldValues
    );
    rewriter.eraseOp(oldTerminator);

    int numOldResults = static_cast<int>(forOp.getNumResults());
    for (int i = 0; i < numOldResults; ++i) {
        forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
    }
    rewriter.setInsertionPointAfter(newForOp);
    rewriter.create<hivm_regbaseintrins::VstasInstrOp>(
        vstasOp.getLoc(),
        newForOp.getResult(numOldResults),
        newForOp.getResult(numOldResults + 1),
        vstasOp.getOperand(2), 
        vstasOp.getOperand(3)
    );
    rewriter.eraseOp(vstasOp);
    rewriter.eraseOp(forOp);

    LLVM_DEBUG(DBGS() << "Hoisting successful.\n");
    return success();
  }
};
 
struct HoistVstasPass : public impl::HoistVstasBase<HoistVstasPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    ModuleOp moduleOp = isa<ModuleOp>(op) ? cast<ModuleOp>(op) : op->getParentOfType<ModuleOp>();
    if (!hacc::utils::isAscend950(moduleOp)) {
      return;
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<HoistVstasPattern>(&getContext());
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
 
std::unique_ptr<mlir::Pass> mlir::hivmave::createHoistVstasPass() {
  return std::make_unique<HoistVstasPass>();
}
//===--------------------- OptimizeReductionLoop.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "optimize-reduction-loop"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_OPTIMIZEREDUCTIONLOOPHIVMAVE
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivmave;

constexpr auto REDUCTION_LOOP_ATTR = "reductionLoop";
constexpr auto REDUCTION_OP_ATTR = "reductionOp";

/// Remove Redunant vsel in loop
/// clang-format off
/// Before transform:
/// ```mlir
/// %init = scf.for %arg1 = %c0 to %c80 step %c64
///   iter_args(%arg2 = %1) -> (vector<1xf32>) {
///   ...
///   %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : ...
///   %14 = ave.hir.pge <ALL> : vector<64xi1>
///   %15 = ave.hir.vextf %res_4, <part_even>, %14 :
///     vector<64xf16>, vector<64xf32>, vector<64xi1>
///   %16 = ave.hir.vsel %13, %15, %1 :
///     vector<64xi1>, vector<64xf32>
///   %17 = ave.hir.pge <ALL> : vector<64xi1>
///   %18 = ave.hir.vadd %16, %arg5, %17 {reductionOp} :
///     vector<64xf32>, vector<64xi1>
///   ...
/// }
///
/// ```
/// After transform:
/// ```mlir
/// %init = scf.for %arg1 = %c0 to %c64 step %c64 iter_args(%arg2 = %1) ->
/// (vector<1xf32>) {
///   ...
///   %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : ...
///   %14 = ave.hir.pge <ALL> : vector<64xi1>
///   %15 = ave.hir.vextf %res_4, <part_even>, %14 :
///     vector<64xf16>, vector<64xf32>, vector<64xi1>
///   %17 = ave.hir.pge <ALL> : vector<64xi1>
///   %18 = ave.hir.vadd %16, %arg5, %17 {reductionOp} :
///     vector<64xf32>, vector<64xi1>
///   ...
/// }
/// %subview_1 = memref.subview %arg1[%arg3, %c64] [1, %c16] [1, 1] : ...
/// %10 = ave.hir.pltm %c64, %c80 : vector<64xi1>
/// %subview_2 = memref.subview %subview_1[0, 0] [1, %c16] [1, 1] : ...
/// %res_3 = ave.hir.vload <NORM> %subview_2[%c0] : ...
/// %11 = ave.hir.pge <ALL> : vector<64xi1>
/// %12 = ave.hir.vextf %res_3, <part_even>, %11 :
///   vector<64xf16>, vector<64xf32>, vector<64xi1>
/// %13 = ave.hir.vsel %10, %12, %1 :
///   vector<64xi1>, vector<64xf32>
/// %14 = ave.hir.pge <ALL> : vector<64xi1>
/// %15 = ave.hir.vadd %13, %9, %14 {reductionOp} :
///   vector<64xf32>, vector<64xi1>
/// ```
/// clang-format on
struct RemoveRedundantVselPattern : public OpRewritePattern<scf::ForOp> {
  RemoveRedundantVselPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!(forOp->hasAttr(REDUCTION_LOOP_ATTR))) {
      return failure();
    }
    Operation *reductionOp = nullptr;
    for (auto &op : forOp.getBody()->getOperations()) {
      if (op.hasAttr(REDUCTION_OP_ATTR)) {
        reductionOp = &op;
      }
    }
    if (!reductionOp) {
      return failure();
    }
    Location loc = forOp.getLoc();
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();
    int64_t lbCst = getConstantIntValue(lowerBound).value();
    int64_t ubCst = getConstantIntValue(upperBound).value();
    int64_t stepCst = getConstantIntValue(step).value();
    bool hasTail = (ubCst - lbCst) % stepCst != 0;
    if (!(hasTail)) {
      return failure();
    }
    // set truncated loop upperbound
    int64_t tripCount = (ubCst - lbCst) / stepCst;
    int64_t newUpperBoundInt = lbCst + (tripCount * stepCst);
    Value newUpperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, newUpperBoundInt);
    forOp.setUpperBound(newUpperBound);

    // handle tail block
    rewriter.setInsertionPointAfter(forOp);
    SmallPtrSet<Operation *, 8> tailOps;
    IRMapping mapper;
    mapper.map(forOp.getInductionVar(), newUpperBound);
    for (auto it : llvm::zip(forOp.getRegionIterArgs(), forOp.getResults())) {
      mapper.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &op : forOp.getBody()->without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, mapper);
      // The new mask operation does not use the old maskOpIdx to prevent mask
      // reuse error.
      if (op.hasAttr(utils::maskOpIdx))
        clonedOp->removeAttr(utils::maskOpIdx);
      tailOps.insert(clonedOp);
    }
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    SmallVector<Value> finalResults;
    for (Value result : yieldOp.getOperands()) {
      finalResults.push_back(mapper.lookup(result));
    }

    // remove vsel in truncated loop
    forOp.getBody()->walk([&](Operation *op) {
      if (op->hasAttr(REDUCTION_OP_ATTR)) {
        Value currentInput = op->getOperand(0);
        if (auto vselOp = currentInput.getDefiningOp<hivmave::VFSelectOp>()) {
          Value vselInput = vselOp.getTrueValue();
          op->setOperand(0, vselInput);
        }
      }
    });

    // combine results
    for (auto it : llvm::zip(forOp.getResults(), finalResults)) {
      Value forOpRes = std::get<0>(it);
      Value tailRes = std::get<1>(it);

      forOpRes.replaceUsesWithIf(tailRes, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return tailOps.find(user) == tailOps.end();
      });
    }
    return success();
  }
};

static void cloneForOpBody(scf::ForOp oldForOp, scf::ForOp newForOp,
                           Value offset, unsigned argOffset,
                           PatternRewriter &rewriter,
                           SmallVectorImpl<Value> &yieldValues) {
  ValueRange newRegionArgs = newForOp.getRegionIterArgs();
  ValueRange oldRegionArgs = oldForOp.getRegionIterArgs();
  IRMapping mapper;
  mapper.map(oldForOp.getInductionVar(), offset);
  for (unsigned k = 0; k < oldRegionArgs.size(); ++k) {
    mapper.map(oldRegionArgs[k], newRegionArgs[2 * k + argOffset]);
  }
  for (auto &op : oldForOp.getBody()->without_terminator()) {
    rewriter.clone(op, mapper);
  }
  for (unsigned k = 0; k < oldRegionArgs.size(); ++k) {
    Value oldYieldOperand = oldForOp.getBody()->getTerminator()->getOperand(k);
    Value newYieldOperand = mapper.lookup(oldYieldOperand);
    yieldValues[2 * k + argOffset] = newYieldOperand;
  }
}

static void combineLoopResults(scf::ForOp oldForOp, scf::ForOp &newForOp,
                               PatternRewriter &rewriter, Location loc,
                               Operation *reductionOp,
                               SmallVector<Value> &finalCombinedValues) {
  llvm::StringRef opName = reductionOp->getName().getStringRef();
  for (unsigned k = 0; k < oldForOp.getRegionIterArgs().size(); ++k) {
    Value resLow = newForOp.getResult(2 * k);
    Value resHigh = newForOp.getResult(2 * k + 1);
    SmallVector<Value, 4> newOperands;
    newOperands.push_back(resLow);
    newOperands.push_back(resHigh);
    IRMapping operandMapper;
    for (unsigned i = 2; i < reductionOp->getNumOperands(); ++i) {
      Value opd = reductionOp->getOperand(i);
      if (oldForOp.getRegion().isAncestor(opd.getParentRegion())) {
        Operation *defOp = opd.getDefiningOp();
        if (defOp) {
          Operation *clonedOp = rewriter.clone(*defOp, operandMapper);
          newOperands.push_back(clonedOp->getResult(0));
        }
      } else {
        newOperands.push_back(opd);
      }
    }
    OperationState state(loc, opName);
    state.addOperands(newOperands);
    state.addTypes(resLow.getType());
    Operation *combineOp = rewriter.create(state);
    auto combined = combineOp->getResult(0);
    finalCombinedValues.push_back(combined);
  }
}

static void cloneSingleLoop(scf::ForOp forOp, PatternRewriter &rewriter,
                            Value index,
                            SmallVectorImpl<Value> &finalCombinedValues) {
  IRMapping lastMapper;
  lastMapper.map(forOp.getInductionVar(), index);
  ValueRange oldRegionArgs = forOp.getRegionIterArgs();
  for (unsigned k = 0; k < oldRegionArgs.size(); ++k) {
    lastMapper.map(oldRegionArgs[k], finalCombinedValues[k]);
  }
  for (auto &op : forOp.getBody()->without_terminator()) {
    Operation *newOp = rewriter.clone(op, lastMapper);
    for (auto pair : llvm::zip(op.getResults(), newOp->getResults()))
      lastMapper.map(std::get<0>(pair), std::get<1>(pair));
  }

  SmallVector<Value, 4> actualFinalResults;
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (Value operand : yieldOp.getOperands()) {
    actualFinalResults.push_back(lastMapper.lookup(operand));
  }

  finalCombinedValues.assign(actualFinalResults.begin(),
                             actualFinalResults.end());
}

/// Split redcutionLoop
/// clang-format off
/// redcutionLoop Before transform:
/// ```mlir
/// %init = scf.for %arg1 = %c0 to %c768 step %c64 iter_args(%arg2 = %1) ->
/// (vector<1xf32>) {
///   ...
///   %res = ave.hir.vadd %17, %18, %19 {reductionOp} : vector<64xf32>,
///   vector<64xi1>
///   ...
///   scf.yield %23 : vector<1xf32>
/// } {reductionLoop}
///
/// ```
/// redcutionLoop After transform:
/// ```mlir
/// %init1, %init2 = scf.for %arg1 = %c0 to %c384 step %c64 iter_args(%arg2 =
/// %1, %arg3 = %2) -> (vector<1xf32>) {
///   ...
///   %res1 = ave.hir.vadd %17, %18, %19 {reductionOp} : vector<64xf32>,
///   vector<64xi1>
///   ...
///   %res2 = ave.hir.vadd %25, %26, %27 {reductionOp} : vector<64xf32>,
///   vector<64xi1>
///   ...
///   scf.yield %23, %31 : vector<1xf32>, vector<1xf32>
/// } {reductionLoop}
/// %res = ave.hir.vadd %init1, %init2, %mask : vector<1xf32>, vector<1xi1>
/// ```
/// clang-format on
struct ReduceSplitPattern : public OpRewritePattern<scf::ForOp> {
  static constexpr auto SPLIT_DEPTH_ATTR = "splitDepth";

  int maxSplitThreshold;

  ReduceSplitPattern(MLIRContext *context, int maxSplit)
      : OpRewritePattern<scf::ForOp>(context), maxSplitThreshold(maxSplit) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!(forOp->hasAttr(REDUCTION_LOOP_ATTR))) {
      return failure();
    }
    Operation *reductionOp = nullptr;
    for (auto &op : forOp.getBody()->getOperations()) {
      if (op.hasAttr(REDUCTION_OP_ATTR)) {
        reductionOp = &op;
      }
    }
    if (!reductionOp) {
      return failure();
    }
    int currentDepth = 0;
    if (auto attr = forOp->getAttrOfType<IntegerAttr>(SPLIT_DEPTH_ATTR)) {
      currentDepth = attr.getInt();
    }
    if (currentDepth >= maxSplitThreshold)
      return failure();

    // get reductionLoop & reductionOp info
    Location loc = forOp.getLoc();
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();
    int64_t lbCst = getConstantIntValue(lowerBound).value();
    int64_t ubCst = getConstantIntValue(upperBound).value();
    int64_t stepCst = getConstantIntValue(step).value();
    bool hasTail = (ubCst - lbCst) % stepCst != 0;
    int64_t tripCount = hasTail ? (ubCst - lbCst) / stepCst
                                : (ubCst - lbCst + stepCst - 1) / stepCst;
    // no need to reduce if trip count < 4
    if (tripCount < 4)
      return failure();
    ValueRange currentInits = forOp.getInitArgs();
    // In reductionLoop with no inits case, exists ub vl/st ops. Thus, split
    // loops can not be executed in parallel, so no need to split.
    if (currentInits.empty()) {
      return failure();
    }
    const bool isOdd = ((unsigned int)tripCount & 0x1);
    int64_t halfCount = tripCount / 2;
    int64_t newUpperBoundInt = lbCst + (halfCount * stepCst);
    Value newUpperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, newUpperBoundInt);
    // create new forOp
    SmallVector<Value, 4> newInits;
    for (Value init : currentInits) {
      newInits.push_back(init);
      newInits.push_back(init);
    }
    auto newForOp = rewriter.create<scf::ForOp>(loc, lowerBound, newUpperBound,
                                                step, newInits);
    newForOp->setAttr(SPLIT_DEPTH_ATTR,
                      rewriter.getI64IntegerAttr(currentDepth + 1));
    newForOp->setAttr(REDUCTION_LOOP_ATTR, rewriter.getUnitAttr());
    rewriter.setInsertionPointToStart(newForOp.getBody());
    int64_t indexOffsetInt = halfCount * stepCst;
    Value indexOffset =
        rewriter.create<arith::ConstantIndexOp>(loc, indexOffsetInt);
    Value i = newForOp.getInductionVar();
    Value i_plus_half = rewriter.create<arith::AddIOp>(loc, i, indexOffset);
    // unsigned numOldArgs = forOp.getRegionIterArgs().size();
    SmallVector<Value, 4> yieldValues(newInits.size());
    cloneForOpBody(forOp, newForOp, i, 0, rewriter, yieldValues);
    cloneForOpBody(forOp, newForOp, i_plus_half, 1, rewriter, yieldValues);
    rewriter.create<scf::YieldOp>(loc, yieldValues);

    // combine result
    rewriter.setInsertionPointAfter(newForOp);
    SmallVector<Value> finalCombinedValues;
    combineLoopResults(forOp, newForOp, rewriter, loc, reductionOp,
                       finalCombinedValues);
    // handle last calculation if trip count is odd
    if (isOdd) {
      int64_t lastOffset = lbCst + (tripCount - 1) * stepCst;
      Value lastIdx = rewriter.create<arith::ConstantIndexOp>(loc, lastOffset);
      cloneSingleLoop(forOp, rewriter, lastIdx, finalCombinedValues);
    }
    rewriter.replaceOp(forOp, finalCombinedValues);
    return success();
  }
};

bool isFromRegionArg(Value v, Block &regionBlock) {
  // if trivial
  for (auto arg : regionBlock.getArguments()) {
    if (v == arg) {
      return true;
    }
  }

  // find def
  if (auto defOp = v.getDefiningOp()) {
    if (defOp->getNumOperands() == 1 && defOp->getNumResults() == 1) {
      return isFromRegionArg(defOp->getOperand(0), regionBlock);
    }
  }

  return false;
}

struct EliminateReductionLoopInitPattern : public OpRewritePattern<scf::ForOp> {
  EliminateReductionLoopInitPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // stage-1: peel first iteration
    if (!forOp->hasAttr(REDUCTION_LOOP_ATTR)) {
      return failure();
    }
    scf::ForOp firstIteration;
    if (failed(
            scf::peelForLoopFirstIteration(rewriter, forOp, firstIteration))) {
      return failure();
    }
    forOp->removeAttr(REDUCTION_LOOP_ATTR);
    firstIteration->removeAttr(REDUCTION_LOOP_ATTR);

    // stage-2: remove the operand of which come from for-loop argument
    firstIteration.walk([&](Operation *op) {
      if (op->hasAttr(REDUCTION_OP_ATTR)) {
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        Block &body = firstIteration.getRegion().front();
        bool isLhsFromArg = isFromRegionArg(lhs, body);
        bool isRhsFromArg = isFromRegionArg(rhs, body);
        if (!isLhsFromArg && isRhsFromArg) {
          rewriter.replaceOp(op, lhs);
        } else if (isLhsFromArg && !isRhsFromArg) {
          rewriter.replaceOp(op, rhs);
        }
      }
    });
    return success();
  }
};

namespace {
struct OptimizeReductionLoopHIVMAVEPass
    : public impl::OptimizeReductionLoopHIVMAVEBase<
          OptimizeReductionLoopHIVMAVEPass> {
  using OptimizeReductionLoopHIVMAVEBase<
      OptimizeReductionLoopHIVMAVEPass>::OptimizeReductionLoopHIVMAVEBase;

public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveRedundantVselPattern>(patterns.getContext());
    patterns.add<ReduceSplitPattern>(patterns.getContext(), this->maxSplit);
    patterns.add<EliminateReductionLoopInitPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> hivmave::createOptimizeReductionLoopHIVMAVEPass(
    const OptimizeReductionLoopHIVMAVEOptions &options) {
  return std::make_unique<OptimizeReductionLoopHIVMAVEPass>(options);
}

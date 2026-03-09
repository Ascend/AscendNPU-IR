//===-------------------- PropagateConvertLayoutScfFor.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hivm-propagate-convert-layout"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Verify that both source and result of convertOp are shaped types.
LogicalResult verifyShapedTypes(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) {
  auto sourceType = dyn_cast<ShapedType>(convertOp.getSource().getType());
  auto resultType = dyn_cast<ShapedType>(convertOp.getResult().getType());
  if (!sourceType || !resultType)
    return rewriter.notifyMatchFailure(
        convertOp, "source or result is not a shaped type");
  return success();
}

/// Create a new scf.for with modified init arg at the specified index.
/// Removes the automatically created yield op from the new for loop.
scf::ForOp createForOpWithModifiedInit(PatternRewriter &rewriter,
                                       scf::ForOp forOp,
                                       unsigned modifiedIdx,
                                       Value newInitValue) {
  SmallVector<Value> newInitArgs(forOp.getInitArgs());
  newInitArgs[modifiedIdx] = newInitValue;

  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(),
      forOp.getLowerBound(),
      forOp.getUpperBound(),
      forOp.getStep(),
      newInitArgs);

  // Remove the automatically created yield
  if (newForOp.getBody()->mightHaveTerminator()) {
    rewriter.eraseOp(newForOp.getBody()->getTerminator());
  }
  return newForOp;
}

/// Set up basic IR mapping between old and new ForOp for induction variable
/// and all iter args.
void setupForOpIRMapping(IRMapping &mapping,
                         scf::ForOp oldForOp,
                         scf::ForOp newForOp) {
  mapping.map(oldForOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < oldForOp.getNumRegionIterArgs(); ++i) {
    mapping.map(oldForOp.getRegionIterArg(i), newForOp.getRegionIterArg(i));
  }
}

/// Clone all operations from forOp's body (except terminator and skipOp)
/// using the provided mapping.
void cloneForBodyOperations(PatternRewriter &rewriter,
                            scf::ForOp forOp,
                            IRMapping &mapping,
                            Operation *skipOp = nullptr) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (&op == skipOp)
      continue;
    rewriter.clone(op, mapping);
  }
}

/// Build yield operands by mapping values from old yield, with optional
/// override at a specific index.
SmallVector<Value> buildYieldOperands(scf::YieldOp oldYield,
                                      IRMapping &mapping,
                                      int overrideIdx,
                                      Value overrideValue) {
  SmallVector<Value> newYieldOperands;
  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    if (static_cast<int>(i) == overrideIdx && overrideValue) {
      newYieldOperands.push_back(overrideValue);
    } else {
      Value mapped = mapping.lookupOrDefault(oldYield.getOperand(i));
      newYieldOperands.push_back(mapped);
    }
  }
  return newYieldOperands;
}

/// Replace old forOp with results from newForOp, with an override at the
/// specified index.
void replaceForOpResults(PatternRewriter &rewriter,
                         scf::ForOp oldForOp,
                         scf::ForOp newForOp,
                         unsigned overrideIdx,
                         Value overrideValue) {
  SmallVector<Value> replacements(newForOp.getResults());
  replacements[overrideIdx] = overrideValue;
  rewriter.replaceOp(oldForOp, replacements);
}

//===----------------------------------------------------------------------===//
// Propagate UP into scf.for Loop (Convert on iter_arg)
//===----------------------------------------------------------------------===//

/// Pattern: Push convert_layout INTO scf.for loop when applied to iter_arg
/// Before:
///   %result = scf.for iter_args(%arg = %init) {
///     %conv = hivm.hir.convert_layout %arg {up}
///     ... use %conv ...
///     scf.yield %yieldVal
///   }
/// After:
///   %expandedInit = hivm.hir.convert_layout %init {up}
///   %result = scf.for iter_args(%newArg = %expandedInit) {
///     %collapsed = hivm.hir.convert_layout %newArg {down}
///     ... use %newArg (replacing %conv) ...
///     %expandedYield = hivm.hir.convert_layout %yieldVal {up}
///     scf.yield %expandedYield
///   }
///   %collapsed_result = hivm.hir.convert_layout %result {down}
struct PropagateConvertLayoutScfForIterArgs
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertOp))
      return rewriter.notifyMatchFailure(convertOp,
                                         "not a propagating-up conversion");

    // Check if source is an iter_arg of a scf.for
    auto blockArg = dyn_cast<BlockArgument>(convertOp.getSource());
    if (!blockArg)
      return rewriter.notifyMatchFailure(convertOp,
                                         "source is not a block argument");

    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return rewriter.notifyMatchFailure(
          convertOp, "block argument is not from scf.for");

    // Check it's an iter_arg (arg 0 is induction variable)
    if (blockArg.getArgNumber() < 1)
      return rewriter.notifyMatchFailure(
          convertOp, "block argument is the induction variable, not iter_arg");

    auto iterArgIdx = llvm::find(forOp.getRegionIterArgs(), blockArg) -
                      forOp.getRegionIterArgs().begin();

    // Verify shapes are compatible
    if (auto verifyResult = verifyShapedTypes(convertOp, rewriter); failed(
        verifyResult))
      return verifyResult;
    LDBG("ShapedType is verified");
    // Get corresponding init and yield values
    Value initArg = forOp.getInitArgs()[iterArgIdx];
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

    // Create expanded init before the loop
    rewriter.setInsertionPointAfterValue(initArg);
    Value expandedInit = createConvertLayoutLike(rewriter, convertOp, initArg);

    // Create new ForOp with expanded init
    rewriter.setInsertionPoint(forOp);
    LDBG(*forOp->getParentOp());
    auto newForOp = createForOpWithModifiedInit(rewriter, forOp, iterArgIdx,
                                                expandedInit);
    // Set up IR mapping for cloning
    IRMapping mapping;
    setupForOpIRMapping(mapping, forOp, newForOp);

    // At the start of new body, add collapse for the modified iter_arg
    rewriter.setInsertionPointToStart(newForOp.getBody());
    Value collapsedIterArg = createConvertLayoutOpposite(
        rewriter, convertOp, newForOp.getRegionIterArg(iterArgIdx));

    // Old iter_arg maps to collapsed value (for users expecting original layout)
    mapping.map(forOp.getRegionIterArg(iterArgIdx), collapsedIterArg);
    // ConvertOp result maps to new iter_arg (already in target layout)
    mapping.map(convertOp.getResult(), newForOp.getRegionIterArg(iterArgIdx));

    // Clone operations (skip convertOp)
    cloneForBodyOperations(rewriter, forOp, mapping, convertOp.getOperation());

    // Create new yield with expanded value for the target position
    Value mappedYieldValue =
        mapping.lookupOrDefault(yieldOp.getOperand(iterArgIdx));
    Value expandedYield =
        createConvertLayoutLike(rewriter, convertOp, mappedYieldValue);
    auto yieldOperands =
        buildYieldOperands(yieldOp, mapping, iterArgIdx, expandedYield);
    rewriter.create<scf::YieldOp>(yieldOp.getLoc(), yieldOperands);

    // After the loop, add collapse for the result
    rewriter.setInsertionPointAfter(newForOp);
    Value collapsedResult = createConvertLayoutOpposite(
        rewriter, convertOp, newForOp.getResult(iterArgIdx));

    // Replace old forOp results
    replaceForOpResults(rewriter, forOp, newForOp, iterArgIdx, collapsedResult);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Propagate DOWN through scf.for Loop (Convert at yield)
//===----------------------------------------------------------------------===//

/// Pattern: Hoist convert_layout OUT of scf.for loop when at yield
/// Before:
///   %result = scf.for iter_args(%arg = %init) {
///     ... use %arg ...
///     %conv = hivm.hir.convert_layout %val {down}
///     scf.yield %conv
///   }
/// After:
///   %expandedInit = hivm.hir.convert_layout %init {down}
///   %result = scf.for iter_args(%newArg = %expandedInit) {
///     %collapsed = hivm.hir.convert_layout %newArg {up}
///     ... use %collapsed (replacing %arg) ...
///     scf.yield %val
///   }
///   %collapsed_result = hivm.hir.convert_layout %result {up}
struct PropagateConvertLayoutScfForYield
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingDown(convertOp))
      return rewriter.notifyMatchFailure(convertOp,
                                         "not a propagating-down conversion");

    if (convertOp->use_empty())
      return rewriter.notifyMatchFailure(convertOp, "convert has no uses");

    // Find if this convertOp feeds into a yield of a scf.for
    scf::YieldOp yieldOp = nullptr;
    unsigned yieldOperandIdx = 0;

    for (OpOperand &use : convertOp->getUses()) {
      if (auto yield = dyn_cast<scf::YieldOp>(use.getOwner())) {
        if (isa<scf::ForOp>(yield->getParentOp())) {
          yieldOp = yield;
          yieldOperandIdx = use.getOperandNumber();
          break;
        }
      }
    }

    if (!yieldOp)
      return rewriter.notifyMatchFailure(
          convertOp, "convert does not feed into scf.for yield");

    auto forOp = cast<scf::ForOp>(yieldOp->getParentOp());

    // Verify shapes are compatible
    if (failed(verifyShapedTypes(convertOp, rewriter)))
      return failure();

    // Get corresponding init value
    Value initArg = forOp.getInitArgs()[yieldOperandIdx];

    // Create converted init before the loop
    rewriter.setInsertionPoint(forOp);
    Value convertedInit =
        createConvertLayoutLike(rewriter, convertOp, initArg);

    // Create new ForOp with converted init
    auto newForOp = createForOpWithModifiedInit(rewriter, forOp,
                                                yieldOperandIdx, convertedInit);

    // Set up IR mapping
    IRMapping mapping;
    setupForOpIRMapping(mapping, forOp, newForOp);

    // At the start, add inverse conversion for the modified iter_arg
    rewriter.setInsertionPointToStart(newForOp.getBody());
    Value inverseIterArg = createConvertLayoutOpposite(
        rewriter, convertOp, newForOp.getRegionIterArg(yieldOperandIdx));

    // Old iter_arg maps to inverse-converted value (original layout)
    mapping.map(forOp.getRegionIterArg(yieldOperandIdx), inverseIterArg);

    // Clone operations (skip convertOp)
    cloneForBodyOperations(rewriter, forOp, mapping, convertOp.getOperation());

    // Create new yield - use unconverted value for target position
    Value unconvertedValue = mapping.lookupOrDefault(convertOp.getSource());
    auto yieldOperands = buildYieldOperands(yieldOp, mapping, yieldOperandIdx,
                                            unconvertedValue);
    rewriter.create<scf::YieldOp>(yieldOp.getLoc(), yieldOperands);

    // After loop, add inverse conversion for the result
    rewriter.setInsertionPointAfter(newForOp);
    Value inverseResult = createConvertLayoutOpposite(
        rewriter, convertOp, newForOp.getResult(yieldOperandIdx));

    // Replace old forOp results
    replaceForOpResults(rewriter, forOp, newForOp, yieldOperandIdx,
                        inverseResult);

    return success();
  }
};

} // namespace

void mlir::hivm::populateConvertLayoutScfFor(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<PropagateConvertLayoutScfForIterArgs>(context);
  // patterns.add<PropagateConvertLayoutScfForYield>(context);
}
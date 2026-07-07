//===-------------------- SplitMixedIfConditionals.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Splits mixed-core scf::IfOps into per-core scf.ifs in two stages:
//
//   Stage A — branch-split. A mixed scf.if with both `then` and `else`
//             becomes two clones (thenIf preserves then, dummies else;
//             elseIf preserves else, dummies then), combined per-yield via
//             `arith.select` (or take-then-side for memrefs).
//
//   Stage B — core-split. Each clone's live branch is partitioned by core
//             type using the shared `WorklistBuilder`, then emitted as a
//             chain of per-core scf.ifs sharing the original condition.
//
//   Stage C — uniform-core attribute attachment. Already-uniform scf.ifs
//             get tagged with `hivm.cube_only` / `hivm.vec_only` so the
//             pattern doesn't re-visit them.
//
// Greedy driver handles recursion: Stage A's clones are re-visited and flow
// into Stage B; Stage B's outputs are tagged uniform-core and flow into
// Stage C.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/WorklistBuilder.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_SPLITMIXEDIFCONDITIONALS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "split-mixed-if-conditionals"

using hivm::detail::queryCoreTypeHelper;

//===----------------------------------------------------------------------===//
// Marker attribute names
//===----------------------------------------------------------------------===//

namespace {
constexpr llvm::StringLiteral kCubeOnlyAttr = "hivm.cube_only";
constexpr llvm::StringLiteral kVecOnlyAttr = "hivm.vec_only";
constexpr llvm::StringLiteral kBranchSplitDoneAttr = "hivm.branch_split_done";
constexpr llvm::StringLiteral kCoreSplitDoneAttr = "hivm.core_split_done";

struct SplitMixedIfConditionalsPass
    : public impl::SplitMixedIfConditionalsBase<SplitMixedIfConditionalsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Small helpers
//===----------------------------------------------------------------------===//

/// Returns the union of core types reachable from the op's regions.
static std::pair<bool, bool> coreTypesIn(scf::IfOp op) {
  auto [hasC, hasV] = analyzeCoreTypes(op.thenBlock());
  if (!op.getElseRegion().empty()) {
    auto [hasC2, hasV2] = analyzeCoreTypes(op.elseBlock());
    hasC |= hasC2;
    hasV |= hasV2;
  }
  return {hasC, hasV};
}

/// Returns CUBE if the op's body is uniformly CUBE, VECTOR if uniformly
/// VECTOR, or CUBE_OR_VECTOR if mixed/empty.
static TCoreType uniformCoreOf(scf::IfOp op) {
  auto [hasC, hasV] = coreTypesIn(op);
  if (hasC && !hasV)
    return TCoreType::CUBE;
  if (hasV && !hasC)
    return TCoreType::VECTOR;
  return TCoreType::CUBE_OR_VECTOR;
}

/// Recursively clone the SSA producer chain of `original` into the current
/// insertion point. Values defined outside `boundary`'s region (function
/// args, parent-scope SSA) are returned as-is. `mapping` is updated so
/// repeated calls don't redundantly re-clone the same producers.
static Value cloneProducerInto(OpBuilder &builder, Operation *boundary,
                               Value original, IRMapping &mapping) {
  if (Value mapped = mapping.lookupOrNull(original))
    return mapped;
  Operation *defining = original.getDefiningOp();
  if (!defining || !boundary->isAncestor(defining))
    return original;
  for (Value operand : defining->getOperands())
    cloneProducerInto(builder, boundary, operand, mapping);
  builder.clone(*defining, mapping);
  return mapping.lookup(original);
}

/// Materialize a value for the inactive branch's yield slot.
///   - For tensor yields, build a `tensor.empty` placeholder of matching
///     shape/element type. This is the only fabricated dummy we emit.
///   - For non-tensor yields (memref, scalar), clone the producer chain of
///     `source` (which originates in either the adjacent original branch in
///     Stage A, or the live block in Stage B) so the inactive branch carries
///     a real value rather than a fabricated zero or fresh alloc.
static Value materializeInactiveYield(OpBuilder &builder, Location loc,
                                      Operation *boundary, Value typeLike,
                                      Value source, IRMapping &chainMap) {
  if (isa<RankedTensorType>(typeLike.getType()))
    return mlir::utils::createEmptyOp(builder, loc, typeLike);
  return cloneProducerInto(builder, boundary, source, chainMap);
}

//===----------------------------------------------------------------------===//
// SplitMixedIfConditionalsPattern
//===----------------------------------------------------------------------===//

namespace {

struct SplitMixedIfConditionalsPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    // ===== Universal short-circuits =====
    if (op->hasAttr(kCubeOnlyAttr) || op->hasAttr(kVecOnlyAttr))
      return failure();
    if (op->hasAttr(kBranchSplitDoneAttr) && op->hasAttr(kCoreSplitDoneAttr))
      return failure();

    auto [hasC, hasV] = coreTypesIn(op);
    if (!hasC && !hasV)
      return failure(); // No CUBE/VECTOR ops → nothing to do.

    if (!hasOnlySplittableRegions(op.thenBlock()))
      return failure();
    if (!op.getElseRegion().empty() &&
        !hasOnlySplittableRegions(op.elseBlock()))
      return failure();

    // ===== Stage C — uniform-core =====
    TCoreType uniform = uniformCoreOf(op);
    if (uniform == TCoreType::CUBE) {
      rewriter.modifyOpInPlace(op, [&]() {
        op->setAttr(kCubeOnlyAttr, rewriter.getUnitAttr());
      });
      return success();
    }
    if (uniform == TCoreType::VECTOR) {
      rewriter.modifyOpInPlace(op, [&]() {
        op->setAttr(kVecOnlyAttr, rewriter.getUnitAttr());
      });
      return success();
    }

    // ===== Stage A — branch-split =====
    // Fires only on a still-fused mixed-core scf.if that has both branches.
    // Single-branch (no-else) ifs skip Stage A and fall through to Stage B.
    if (!op->hasAttr(kBranchSplitDoneAttr) && !op.getElseRegion().empty()) {
      Location loc = op.getLoc();
      Value cond = op.getCondition();
      auto thenYield = cast<scf::YieldOp>(op.thenBlock()->getTerminator());
      auto elseYield = cast<scf::YieldOp>(op.elseBlock()->getTerminator());
      if (thenYield.getNumOperands() != elseYield.getNumOperands())
        return failure();

      auto buildClone = [&](bool keepThen, Operation *insertAfter) -> scf::IfOp {
        rewriter.setInsertionPointAfter(insertAfter);
        auto newIf = rewriter.create<scf::IfOp>(
            loc, op.getResultTypes(), cond, /*addThenBlock=*/false,
            /*addElseBlock=*/false);
        Region &liveSrc = keepThen ? op.getThenRegion() : op.getElseRegion();
        Region &liveDst =
            keepThen ? newIf.getThenRegion() : newIf.getElseRegion();
        Region &dummyDst =
            keepThen ? newIf.getElseRegion() : newIf.getThenRegion();
        ValueRange liveYields =
            keepThen ? thenYield.getOperands() : elseYield.getOperands();
        ValueRange adjYields =
            keepThen ? elseYield.getOperands() : thenYield.getOperands();

        rewriter.cloneRegionBefore(liveSrc, liveDst, liveDst.end());

        Block *dummyBlock = rewriter.createBlock(&dummyDst);
        rewriter.setInsertionPointToStart(dummyBlock);
        IRMapping chainMap;
        SmallVector<Value> dummies;
        dummies.reserve(liveYields.size());
        for (size_t i = 0; i < liveYields.size(); ++i)
          dummies.push_back(materializeInactiveYield(
              rewriter, loc, op, /*typeLike=*/liveYields[i],
              /*source=*/adjYields[i], chainMap));
        rewriter.create<scf::YieldOp>(loc, dummies);

        newIf->setAttr(kBranchSplitDoneAttr, rewriter.getUnitAttr());
        return newIf;
      };

      scf::IfOp thenIf = buildClone(/*keepThen=*/true, op);
      scf::IfOp elseIf = buildClone(/*keepThen=*/false, thenIf);

      rewriter.setInsertionPointAfter(elseIf);
      SmallVector<Value> combined;
      combined.reserve(op.getNumResults());
      for (unsigned i = 0; i < op.getNumResults(); ++i) {
        Value t = thenIf.getResult(i);
        Value e = elseIf.getResult(i);
        if (isa<MemRefType>(t.getType())) {
          // arith.select doesn't support memrefs — take the then-side. The
          // inactive branches that produce these memrefs are dead code, so DCE
          // will clean them up after the live core executes.
          combined.push_back(t);
        } else {
          combined.push_back(
              rewriter.create<arith::SelectOp>(loc, cond, t, e));
        }
      }

      rewriter.replaceOp(op, combined);
      return success();
    }

    // ===== Stage B — core-split =====
    // Fires on a still-mixed scf.if that either skipped Stage A (no else,
    // single live branch) or completed Stage A and now has a live branch with
    // mixed core types. Partitions the live branch into per-core WorkItems
    // using the shared WorklistBuilder and emits a chain of per-core scf.ifs
    // sharing the original condition. Cross-WorkItem values flow through each
    // per-core scf.if's result.
    if (!op->hasAttr(kCoreSplitDoneAttr)) {
      Location loc = op.getLoc();
      Value cond = op.getCondition();

      // Live branch detection: the side with core ops. With no else, the live
      // side is then by definition; otherwise probe both sides via
      // analyzeCoreTypes.
      bool liveIsThen;
      if (op.getElseRegion().empty()) {
        liveIsThen = true;
      } else {
        auto [tC, tV] = analyzeCoreTypes(op.thenBlock());
        liveIsThen = (tC || tV);
      }
      Block *liveBlock = liveIsThen ? op.thenBlock() : op.elseBlock();
      auto liveYield = cast<scf::YieldOp>(liveBlock->getTerminator());

      hivm::WorklistBuilder wb(liveBlock);
      auto built = wb.build();
      if (failed(built) || built->worklist.empty())
        return failure();

      // globalMap: live-block values → their externally-visible representative.
      // After cloning a WI's ops, this maps each cloned op's result to the
      // value inside the per-core if; once the per-core if is finalised, the
      // map is updated again to point at the if's externally-visible result so
      // subsequent WIs and the final replacement chain see the right value.
      IRMapping globalMap;

      rewriter.setInsertionPointAfter(op);

      for (auto &wiPtr : built->worklist) {
        WorkItem &wi = *wiPtr;

        // Collect this WI's escaping values from localOutputs (already filters
        // for cross-WI / live-yield uses in block mode).
        SmallVector<Value> outputs;
        outputs.reserve(wi.localOutputs.size());
        for (auto [orig, _] : wi.localOutputs)
          outputs.push_back(orig);

        SmallVector<Type> outputTypes;
        outputTypes.reserve(outputs.size());
        for (Value v : outputs)
          outputTypes.push_back(v.getType());

        auto coreIf = rewriter.create<scf::IfOp>(
            loc, outputTypes, cond, /*addThenBlock=*/false,
            /*addElseBlock=*/false);

        Region &liveDst =
            liveIsThen ? coreIf.getThenRegion() : coreIf.getElseRegion();
        Region &dummyDst =
            liveIsThen ? coreIf.getElseRegion() : coreIf.getThenRegion();

        // Live side: clone WI ops in block order, then yield mapped outputs.
        Block *liveNewBlock = rewriter.createBlock(&liveDst);
        rewriter.setInsertionPointToStart(liveNewBlock);
        for (Operation *innerOp : wi.ops)
          rewriter.clone(*innerOp, globalMap);
        SmallVector<Value> mappedOutputs;
        mappedOutputs.reserve(outputs.size());
        for (Value v : outputs)
          mappedOutputs.push_back(globalMap.lookupOrDefault(v));
        rewriter.create<scf::YieldOp>(loc, mappedOutputs);

        // Dummy side: always create the block. scf.if's thenRegion is a
        // SizedRegion<1>, so when liveIsThen=false the dummy region (then)
        // must contain exactly one block even if there are no SSA outputs to
        // materialize. With outputs.empty() the dummy is just `scf.yield`.
        Block *dummyNewBlock = rewriter.createBlock(&dummyDst);
        rewriter.setInsertionPointToStart(dummyNewBlock);
        SmallVector<Value> dummies;
        if (!outputs.empty()) {
          IRMapping chainMap;
          dummies.reserve(outputs.size());
          for (Value v : outputs)
            // Stage B has no "adjacent original branch" — for non-tensor
            // yields we clone the live-block producer chain so the inactive
            // side carries a real value.
            dummies.push_back(materializeInactiveYield(
                rewriter, loc, op, /*typeLike=*/v, /*source=*/v, chainMap));
        }
        rewriter.create<scf::YieldOp>(loc, dummies);

        // Tag attributes. Always mark both branch_split_done and
        // core_split_done so the universal short-circuit catches re-visits.
        // Add a uniform-core marker when the WI's core is unambiguous.
        coreIf->setAttr(kBranchSplitDoneAttr, rewriter.getUnitAttr());
        coreIf->setAttr(kCoreSplitDoneAttr, rewriter.getUnitAttr());
        if (wi.core == TCoreType::CUBE)
          coreIf->setAttr(kCubeOnlyAttr, rewriter.getUnitAttr());
        else if (wi.core == TCoreType::VECTOR)
          coreIf->setAttr(kVecOnlyAttr, rewriter.getUnitAttr());

        // Re-route subsequent uses: each output's externally-visible form is
        // the corresponding per-core if result.
        for (auto [orig, ifResult] :
             llvm::zip(outputs, coreIf.getResults())) {
          globalMap.map(orig, ifResult);
        }

        rewriter.setInsertionPointAfter(coreIf);
      }

      // Build replacements for the original op's results from the live yield's
      // operands, looked up through the now-fully-populated globalMap.
      SmallVector<Value> finalReplacements;
      finalReplacements.reserve(liveYield.getNumOperands());
      for (Value v : liveYield.getOperands())
        finalReplacements.push_back(globalMap.lookupOrDefault(v));

      rewriter.replaceOp(op, finalReplacements);
      return success();
    }

    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void SplitMixedIfConditionalsPass::runOnOperation() {
  auto context = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(context);
  patterns.insert<SplitMixedIfConditionalsPattern>(context);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createSplitMixedIfConditionalsPass() {
  return std::make_unique<SplitMixedIfConditionalsPass>();
}

//===--------- FuseReductionIntoLoop.cpp -------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass fuses post-loop linalg.reduce operations into scf.for loop bodies.
// When a loop accumulates 2D tensors that are subsequently reduced to 1D
// outside the loop, this pass transforms the loop to accumulate reduced 1D
// tensors directly, lowering memory pressure and improving data locality.
//
// In zero-only mode, fusion requires a known-zero %init_2d. Enabling this pass
// allows reassociation even without fast-math flags on the original additions.
// Reassociation can change rounding or overflow intermediate sums.
//
// Before:
//   %result = scf.for ... iter_args(%acc = %init_2d) {
//     %val = ... : tensor<MxNxf32>
//     %new_acc = arith.addf %acc, %val : tensor<MxNxf32>
//     scf.yield %new_acc
//   }
//   %reduced = linalg.reduce ins(%result) outs(%zero_1d) dimensions = [0] {
//     ^bb0(%in: f32, %init: f32):
//       %s = arith.addf %in, %init : f32
//       linalg.yield %s : f32
//   }
//
// After, for a known-zero %init_2d:
//   %result = scf.for ... iter_args(%acc = %zero_1d) {
//     %val = ... : tensor<MxNxf32>
//     %reduced_val = linalg.reduce ins(%val) outs(%zero_1d) dimensions = [0] {
//       ^bb0(%in: f32, %init: f32):
//         %s = arith.addf %in, %init : f32
//         linalg.yield %s : f32
//     }
//     %new_acc = arith.addf %acc, %reduced_val : tensor<Nxf32>
//     scf.yield %new_acc
//   }
//
// A nonzero or unknown final reduction initializer is added once after the
// loop, including when the loop runs zero times.
// Late mode also accepts nonzero or unknown %init_2d. After the loop, reduce
// it with the original reduction initializer and add the loop result.
// Keep the original tensor value: reloading its buffer could read changed data.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fuse-reduction-into-loop"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace bishengir {
using namespace mlir;

#define GEN_PASS_DEF_FUSEREDUCTIONINTOLOOP
#include "bishengir/Transforms/Passes.h.inc"

namespace {

/// Describes a reducible accumulator inside a for loop.
struct ReduceCandidate {
  unsigned resultIdx;         // scf.for result index
  linalg::ReduceOp reduceOp;  // the post-loop reduce
  arith::AddFOp addfOp;       // the in-loop accumulation addf
  Value value2D;              // the 2D value being accumulated (not iter_arg)
  bool zeroInit;              // loop initializer is known to be zero
};

// New partial sums can overflow or produce NaNs. Do not copy nnan/ninf flags.
constexpr arith::FastMathFlags fusionFastMath =
    arith::FastMathFlags::reassoc | arith::FastMathFlags::nsz;

/// Check constants and fills, not runtime data. tensor.empty is not zero.
bool isZeroTensor(Value value) {
  if (matchPattern(value, m_AnyZeroFloat()))
    return true;
  auto fill = value.getDefiningOp<linalg::FillOp>();
  return fill && matchPattern(fill.getInputs().front(), m_AnyZeroFloat());
}

/// Return true if the linalg.reduce combiner is a simple additive reduction:
///   ^bb(%in: fT, %init: fT):
///     %s = arith.addf %in, %init
///     linalg.yield %s
bool isAdditiveReduction(linalg::ReduceOp reduceOp) {
  Region &combiner = reduceOp.getCombiner();
  if (!combiner.hasOneBlock())
    return false;
  Block &block = combiner.front();
  // Exactly two operations: arith.addf + linalg.yield.
  if (block.getOperations().size() != 2)
    return false;
  auto addfOp = dyn_cast<arith::AddFOp>(&block.front());
  if (!addfOp)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;
  Value input = block.getArgument(0);
  Value init = block.getArgument(1);
  return yieldOp.getOperand(0) == addfOp.getResult() &&
         ((addfOp.getLhs() == input && addfOp.getRhs() == init) ||
          (addfOp.getLhs() == init && addfOp.getRhs() == input));
}

/// Check whether the yield value at `iterArgIdx` follows the pattern:
///   scf.yield ..., arith.addf(%iter_arg, %val), ...
/// Returns the addf op and the non-iter-arg operand on success.
std::optional<std::pair<arith::AddFOp, Value>>
matchAccumulationPattern(scf::ForOp forOp, unsigned iterArgIdx) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value yieldedValue = yieldOp.getOperand(iterArgIdx);

  auto addfOp = yieldedValue.getDefiningOp<arith::AddFOp>();
  if (!addfOp)
    return std::nullopt;

  BlockArgument iterArg = forOp.getRegionIterArg(iterArgIdx);
  Value other;
  if (addfOp.getLhs() == iterArg)
    other = addfOp.getRhs();
  else if (addfOp.getRhs() == iterArg)
    other = addfOp.getLhs();
  else
    return std::nullopt;

  return std::make_pair(addfOp, other);
}

/// Scan all results of `forOp` and collect those that:
///   1. Have a single user which is an additive linalg.reduce.
///   2. Are accumulated inside the loop via arith.addf with the iter_arg.
///   3. The iter_arg has only that single addf use.
///   4. The reduce output has a static shape.
///   5. The loop initializer is known to be zero, or late mode is selected.
SmallVector<ReduceCandidate>
findReduceCandidates(scf::ForOp forOp, FuseReductionIntoLoopMode mode) {
  SmallVector<ReduceCandidate> candidates;

  for (OpResult result : forOp.getResults()) {
    unsigned idx = result.getResultNumber();

    // Result must feed into exactly one user (the reduce).
    if (!result.hasOneUse())
      continue;

    auto reduceOp = dyn_cast<linalg::ReduceOp>(*result.getUsers().begin());
    if (!reduceOp)
      continue;

    // Single-input additive reduction only.
    if (reduceOp.getInputs().size() != 1 || reduceOp.getInputs()[0] != result)
      continue;
    if (!isAdditiveReduction(reduceOp))
      continue;

    // In-loop pattern: yield arith.addf(iter_arg, val_2d).
    auto match = matchAccumulationPattern(forOp, idx);
    if (!match)
      continue;
    auto [addfOp, value2D] = *match;
    if (!addfOp->hasOneUse())
      continue;

    // iter_arg must only be used by the addf.
    BlockArgument iterArg = forOp.getRegionIterArg(idx);
    if (!iterArg.hasOneUse())
      continue;

    // Require static shapes for the reduced output type.
    auto outType =
        dyn_cast<RankedTensorType>(reduceOp.getResults()[0].getType());
    if (!outType || !outType.hasStaticShape() || outType.getEncoding())
      continue;

    bool zeroInit = isZeroTensor(forOp.getInitArgs()[idx]);
    if (!zeroInit && mode != FuseReductionIntoLoopMode::Late)
      continue;

    candidates.push_back({idx, reduceOp, addfOp, value2D, zeroInit});
  }

  return candidates;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct FuseReductionIntoLoop
    : public impl::FuseReductionIntoLoopBase<FuseReductionIntoLoop> {
  explicit FuseReductionIntoLoop(const FuseReductionIntoLoopOptions &options)
      : FuseReductionIntoLoopBase(options) {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Collect for ops first to avoid iterator invalidation.
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (scf::ForOp forOp : forOps)
      transformForOp(forOp);
  }

private:
  void transformForOp(scf::ForOp forOp) {
    auto candidates = findReduceCandidates(forOp, mode);
    if (candidates.empty())
      return;

    LDBG("Found " << candidates.size()
                   << " reduction candidate(s) to fuse into loop");

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    // Map candidate addf ops to their candidates for inline replacement.
    DenseMap<Operation *, ReduceCandidate *> addfToCand;
    for (auto &c : candidates)
      addfToCand[c.addfOp.getOperation()] = &c;

    // --- Step 1: Create 1D zero-initialized init args for candidates. ---
    SmallVector<Value> newInitArgs(forOp.getInitArgs());
    DenseMap<unsigned, RankedTensorType> candOutTypes;
    DenseMap<unsigned, Value> candZeroInits;

    for (auto &cand : candidates) {
      auto outType =
          cast<RankedTensorType>(cand.reduceOp.getResults()[0].getType());
      candOutTypes[cand.resultIdx] = outType;

      Value emptyTensor = builder.create<tensor::EmptyOp>(
          loc, outType.getShape(), outType.getElementType());
      Value zero = builder.create<arith::ConstantOp>(
          loc, builder.getZeroAttr(outType.getElementType()));
      Value zeroFilled =
          builder.create<linalg::FillOp>(loc, zero, emptyTensor)
              .getResult(0);

      newInitArgs[cand.resultIdx] = zeroFilled;
      candZeroInits[cand.resultIdx] = zeroFilled;
    }

    // --- Step 2: Create new scf.for with modified init args. ---
    // Note: ForOp::build does NOT create a terminator when iterArgs is
    // non-empty and no bodyBuilder is provided.  Supply a body builder
    // that creates a placeholder yield so the block is always valid.
    auto newForOp = builder.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        newInitArgs,
        [](OpBuilder &b, Location loc, Value /*iv*/, ValueRange iterArgs) {
          b.create<scf::YieldOp>(loc, iterArgs);
        });
    newForOp->setAttrs(forOp->getAttrs());

    Block *newBody = newForOp.getBody();
    Block *oldBody = forOp.getBody();

    // The body builder created a placeholder yield.  We insert all new
    // ops before it and update its operands at the end.
    Operation *autoYield = newBody->getTerminator();

    // --- Step 3: Build IRMapping for block arguments. ---
    IRMapping mapping;
    // Induction variable.
    mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
    // Iter args -- skip candidates (their only use is the addf we handle
    // separately).
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
      if (!candOutTypes.count(i))
        mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));
    }

    // --- Step 4: Clone body ops, replacing candidate addfs inline with
    //             reduce + 1D addf. ---
    builder.setInsertionPoint(autoYield);
    for (Operation &op : oldBody->without_terminator()) {
      auto it = addfToCand.find(&op);
      if (it != addfToCand.end()) {
        // Replace this candidate addf with: reduce(value2D) + addf(iter_arg_1d, reduced).
        auto &cand = *it->second;
        Value mapped2D = mapping.lookupOrDefault(cand.value2D);
        Value reduceInit = candZeroInits[cand.resultIdx];

        SmallVector<int64_t> dims(cand.reduceOp.getDimensions());
        auto newReduce = builder.create<linalg::ReduceOp>(
            cand.reduceOp.getLoc(),
            /*inputs=*/ValueRange{mapped2D},
            /*inits=*/ValueRange{reduceInit}, dims,
            [](OpBuilder &b, Location bodyLoc, ValueRange args) {
              Value sum = b.create<arith::AddFOp>(bodyLoc, args[0], args[1],
                                                  fusionFastMath);
              b.create<linalg::YieldOp>(bodyLoc, sum);
            });

        Value iterArg1D = newBody->getArgument(cand.resultIdx + 1);
        Value newAddf = builder.create<arith::AddFOp>(
            cand.addfOp.getLoc(), iterArg1D, newReduce.getResults()[0],
            fusionFastMath);

        mapping.map(cand.addfOp.getResult(), newAddf);
      } else {
        builder.clone(op, mapping);
      }
    }

    // --- Step 6: Rewrite the yield operands using the mapping. ---
    auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
    SmallVector<Value> newYieldOperands;
    for (unsigned i = 0; i < oldYield.getNumOperands(); ++i)
      newYieldOperands.push_back(
          mapping.lookupOrDefault(oldYield.getOperand(i)));
    autoYield->setOperands(newYieldOperands);

    // --- Step 7: Rewire external uses and clean up. ---
    // Non-candidate for results: direct replacement.
    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      if (!candOutTypes.count(i))
        forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
    }
    // Insert the final add at the original reduction, after its initializer
    // is defined.
    for (auto &cand : candidates) {
      Value accumulated = newForOp.getResult(cand.resultIdx);
      if (!cand.zeroInit) {
        cand.reduceOp.getInputsMutable().assign(
            ValueRange{forOp.getInitArgs()[cand.resultIdx]});
        auto addf =
            cast<arith::AddFOp>(&cand.reduceOp.getCombiner().front().front());
        addf.setFastmath(fusionFastMath);
        builder.setInsertionPointAfter(cand.reduceOp);
        Value reduced = cand.reduceOp.getResults()[0];
        auto merged = builder.create<arith::AddFOp>(
            cand.reduceOp.getLoc(), reduced, accumulated, fusionFastMath);
        reduced.replaceAllUsesExcept(merged.getResult(), merged);
        continue;
      }
      Value reductionInit = cand.reduceOp.getInits().front();
      builder.setInsertionPoint(cand.reduceOp);
      if (!isZeroTensor(reductionInit))
        accumulated = builder.create<arith::AddFOp>(
            cand.reduceOp.getLoc(), accumulated, reductionInit, fusionFastMath);
      cand.reduceOp.getResults()[0].replaceAllUsesWith(accumulated);
      cand.reduceOp->erase();
    }

    forOp->erase();
    LDBG("Successfully fused reductions into loop");
  }
};

} // namespace

/// Create FuseReductionIntoLoop pass.
std::unique_ptr<mlir::Pass>
createFuseReductionIntoLoopPass(const FuseReductionIntoLoopOptions &options) {
  return std::make_unique<FuseReductionIntoLoop>(options);
}

} // namespace bishengir

//===- SimplifyOps.cpp ------- Simplify operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SIMPLIFYOPS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {
struct SimplifyOpsPass : public impl::SimplifyOpsBase<SimplifyOpsPass> {
public:
  void runOnOperation() final;
};

bool involvesLinalgMatmulImpl(
    mlir::Value value,
    llvm::SmallPtrSetImpl<mlir::Value> &seenValues,
    llvm::SmallPtrSetImpl<mlir::Operation *> &seenOps) {
  if (!seenValues.insert(value).second)
    return false;

  auto defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  if (!seenOps.insert(defOp).second)
    return false;

  if (llvm::isa<mlir::linalg::MatmulOp>(defOp))
    return true;

  if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(defOp)) {
    auto result = llvm::cast<mlir::OpResult>(value);
    unsigned resultNumber = result.getResultNumber();

    auto yieldOp = llvm::cast<mlir::scf::YieldOp>(
        forOp.getBody()->getTerminator());

    if (involvesLinalgMatmulImpl(yieldOp.getOperand(resultNumber),
                                 seenValues, seenOps))
      return true;

    if (resultNumber < forOp.getInitArgs().size() &&
        involvesLinalgMatmulImpl(forOp.getInitArgs()[resultNumber],
                                 seenValues, seenOps))
      return true;

    return false;
  }

  for (mlir::Value operand : defOp->getOperands()) {
    if (involvesLinalgMatmulImpl(operand, seenValues, seenOps))
      return true;
  }

  return false;
}

bool involvesLinalgMatmul(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Value, 32> seenValues;
  llvm::SmallPtrSet<mlir::Operation *, 32> seenOps;
  return involvesLinalgMatmulImpl(value, seenValues, seenOps);
}

// TODO: Optimize to solve it generally by canonicalize
bool isSafeToSimplifyCast(CastOp castOp) {
  auto inputType = getElementTypeOrSelf(castOp.getInputs().front());
  auto outputType = getElementTypeOrSelf(castOp.getOutputs().front());
  // Blacklist for now:
  // 1. Not allowing cross data type cast simplify
  return !((isa<FloatType>(inputType) && outputType.isInteger()) ||
           (inputType.isInteger() && isa<FloatType>(outputType)));
}

struct CastOpPattern : public OpRewritePattern<CastOp> {
public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp castOp,
                                PatternRewriter &rewriter) const final {
    if (isOpTriviallyDead(castOp)) {
      rewriter.eraseOp(castOp);
      return success();
    }

    if (!isSafeToSimplifyCast(castOp)) {
      return failure();
    }

    // Helper function that return the cast op that
    // defines all inputs of the given op (in the same order). Return "nullptr"
    // if there is no such op.
    auto getInputCast = [](CastOp castOp) -> CastOp {
      auto inputCastOp = castOp.getInputs().front().getDefiningOp<CastOp>();
      if (!inputCastOp)
        return {};
      if (inputCastOp.getResults() != castOp.getInputs())
        return {};
      return inputCastOp;
    };

    // Helper to test for fastmath<contract> attribute.
    auto hasFastMathContract = [](CastOp op) -> bool {
      auto fastMathAttr = op->getAttrOfType<mlir::arith::FastMathFlagsAttr>(
          mlir::arith::FastMathFlagsAttr::name);
      if (!fastMathAttr)
        return false;

      using arith::FastMathFlags;
      FastMathFlags flags = fastMathAttr.getValue();

      return mlir::arith::bitEnumContainsAll(flags, FastMathFlags::contract);
    };

    // Process ops bottom-to-top.

    // Helper to get precision rank for float types (higher = more precision)
    auto getPrecisionRank = [](Type type) -> int {
      if (type.isBF16())
        return 0; // lowest precision
      if (type.isF16())
        return 1;
      if (type.isF32())
        return 2;
      if (type.isF64())
        return 3; // highest precision
      return -1;  // non-float type
    };

    // Helper to check if chain has precision down-then-up pattern
    auto hasPrecisionDownUpPattern =
        [&getPrecisionRank](SmallVector<CastOp> &chain) -> bool {
      if (chain.size() < 2)
        return false;

      // Track if we've seen a precision decrease
      bool hasDecreased = false;

      for (int i = static_cast<int>(chain.size()) - 1; i >= 0; --i) {
        Type inType = getElementTypeOrSelf(chain[i].getInputs()[0].getType());
        Type outType = getElementTypeOrSelf(chain[i].getOutputs()[0].getType());

        int inRank = getPrecisionRank(inType);
        int outRank = getPrecisionRank(outType);

        // Skip if not float-to-float cast
        if (inRank < 0 || outRank < 0)
          continue;

        if (outRank < inRank) {
          // Precision decrease
          hasDecreased = true;
        } else if (outRank > inRank && hasDecreased) {
          // Precision increase after a decrease - this is the pattern we're
          // looking for
          return true;
        }
      }
      return false;
    };

    // Helper to check if a cast is precision upcast (safe, no precision loss)
    auto isPrecisionUpcast = [&getPrecisionRank](CastOp op) -> bool {
      Type inType = getElementTypeOrSelf(op.getInputs()[0].getType());
      Type outType = getElementTypeOrSelf(op.getOutputs()[0].getType());
      int inRank = getPrecisionRank(inType);
      int outRank = getPrecisionRank(outType);
      // Precision upcast means output has higher or equal precision
      return inRank >= 0 && outRank >= 0 && outRank >= inRank;
    };

    // Helper to check if all casts in chain have fastmath contract
    // Precision upcasts are allowed without fastmath contract since they are
    // safe and don't cause precision loss
    auto allHaveFastMathContract = [&isPrecisionUpcast, &hasFastMathContract](
                                       SmallVector<CastOp> &chain) -> bool {
      for (CastOp op : chain) {
        // Skip precision upcasts - they are safe
        if (isPrecisionUpcast(op))
          continue;
        if (!hasFastMathContract(op))
          return false;
      }
      return true;
    };

    // Traverse the chain of input cast ops to see if an op with the same
    // input types can be found.
    SmallVector<CastOp> castChain;
    CastOp nextCast = castOp;
    while (nextCast) {
      // In total cast chain, if one cast of chain has the same type input and
      // output, it should always represent an intended change in value, which
      // means it couldn't be erased. So cast chain must be split at this point
      if (nextCast.getInputs().getTypes() == nextCast.getResultTypes())
        break;

      castChain.push_back(nextCast);

      if (nextCast.getInputs().getTypes() == castOp.getResultTypes()) {
        // Found a cast where the input types match the output types of the
        // matched op. We can directly use those inputs and the matched op can
        // be removed.
        // Check if chain has precision down-then-up pattern, if so all casts
        // must have fastmath contract to allow optimization.
        if (hasPrecisionDownUpPattern(castChain) &&
            !allHaveFastMathContract(castChain)) {
          if (!involvesLinalgMatmul(castChain[0].getOperand(0))) {
            break;
          }
        }

        rewriter.replaceOp(castOp, nextCast.getInputs());
        return success();
      }
      nextCast = getInputCast(nextCast);
    }

    return failure();
  }
};

inline bool isSimpleCastOp(CastOp op) {
  return op.getOutputs()[0].getDefiningOp<mlir::tensor::EmptyOp>();
}

struct LoopedCastOpPattern : public OpRewritePattern<CastOp> {
public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp castOp,
                                PatternRewriter &rewriter) const final {
    if (!isSafeToSimplifyCast(castOp) || !isSimpleCastOp(castOp)) {
      return failure();
    }

    // Verify that castop is in for loop
    if (!isa<scf::ForOp>(castOp->getParentOp()))
      return failure();
    scf::ForOp parentFor = llvm::cast<scf::ForOp>(castOp->getParentOp());
    assert(parentFor.getRegion().hasOneBlock());
    unsigned iterArgIdx = 0;

    // Verify that an iter_arg is only used in this cast
    Value castSrc = castOp.getInputs().front();
    if (!castSrc.hasOneUse())
      return failure();
    if (auto *iter = llvm::find(parentFor.getRegionIterArgs(), castSrc);
        iter != parentFor.getRegionIterArgs().end())
      iterArgIdx = static_cast<unsigned>(
          std::distance(iter, parentFor.getRegionIterArgs().begin()));
    else
      return failure();

    // Get the yield op and yielded value of this for loop
    assert(
        isa<scf::YieldOp>(parentFor.getRegion().getBlocks().front().back()) &&
        "For loop doesn't terminates with yieldOp");
    scf::YieldOp yieldOp =
        cast<scf::YieldOp>(parentFor.getRegion().getBlocks().front().back());
    assert(yieldOp.getNumOperands() > iterArgIdx &&
           "Yielded value num doesn't match loop's iter_arg num");
    Value yieldedValue = yieldOp.getResults()[iterArgIdx];
    assert(yieldedValue.getType() == castSrc.getType() &&
           "Yielded value type doesn't match loop's iter_arg type");

    // Verify that the yielded value is produced with another cast with source
    // types equal to current target types
    CastOp lastCast = yieldedValue.getDefiningOp<CastOp>();
    if (!lastCast || lastCast == castOp || !isSimpleCastOp(lastCast))
      return failure();
    if (castOp->getResultTypes() != lastCast.getInputs().getTypes())
      return failure();

    // Check that the exit cast's result is only used by the yield op inside
    // the loop. If there are other uses (e.g., tensor.extract_slice), moving
    // it after the loop would violate dominance.
    for (auto &use : lastCast->getResult(0).getUses()) {
      if (use.getOwner() != yieldOp)
        return failure();
    }

    // do moving

    // move current cast before the for loop, using init value of iter_arg as
    // src
    rewriter.moveOpBefore(castOp, parentFor);
    rewriter.setInsertionPoint(castOp);
    Operation *castOutput = castOp.getOutputs()[0].getDefiningOp();
    assert(castOutput && "Cast op output is null!");
    mlir::tensor::EmptyOp buffer =
        cast<mlir::tensor::EmptyOp>(rewriter.clone(*castOutput));
    rewriter.replaceAllUsesWith(castOp->getResults(),
                                parentFor.getRegionIterArg(iterArgIdx));
    rewriter.modifyOpInPlace(castOp, [&]() {
      castOp.getOutputsMutable()[0].set(buffer.getResult());
      castOp.getInputsMutable()[0].set(parentFor.getInitArgs()[iterArgIdx]);
    });

    // update yieldop to yield src of last cast
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp.getResultsMutable()[iterArgIdx].set(lastCast.getInputs()[0]);
    });

    // move last cast after the for loop. using forOp result as src
    rewriter.moveOpAfter(lastCast, parentFor);
    rewriter.setInsertionPoint(lastCast);
    Operation *lastCastOp = lastCast.getOutputs()[0].getDefiningOp();
    assert(lastCastOp && "Last cast op is null!");
    buffer = cast<mlir::tensor::EmptyOp>(rewriter.clone(*lastCastOp));
    rewriter.replaceAllUsesWith(parentFor.getResult(iterArgIdx),
                                lastCast.getResults());
    rewriter.modifyOpInPlace(lastCast, [&]() {
      lastCast.getOutputsMutable()[0].set(buffer.getResult());
      lastCast.getInputsMutable()[0].set(parentFor.getResult(iterArgIdx));
    });

    // update for op with new iter_arg type equal to casted type
    rewriter.modifyOpInPlace(parentFor, [&]() {
      parentFor.getRegionIterArg(iterArgIdx)
          .setType(castOp->getResultTypes()[0]);
      parentFor.getInitArgsMutable()[iterArgIdx].set(castOp->getResults()[0]);
      parentFor.getResult(iterArgIdx).setType(castOp->getResultTypes()[0]);
    });

    return llvm::success();
  }
};

struct TransposeOpPattern : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::TransposeOp transOp,
                                PatternRewriter &rewriter) const final {
    if (isOpTriviallyDead(transOp)) {
      rewriter.eraseOp(transOp);
      return success();
    }

    // Initialize baseline as golden for simulation result comparison
    SmallVector<int64_t> base(transOp.getPermutation().size());
    for (size_t i = 0; i < base.size(); ++i)
      base[i] = static_cast<int>(i);

    // Helper function simulating permutations on input data
    auto simulate = [](const ArrayRef<int64_t> input,
                       const ArrayRef<int64_t> permutation) {
      assert(input.size() == permutation.size());
      SmallVector<int64_t> res(permutation.size());
      for (const auto [i, v] : llvm::enumerate(permutation))
        res[v] = input[i];
      return res;
    };

    SmallVector<int64_t> sim(base);
    // Bottom-up searching the chain of transpose ops
    linalg::TransposeOp nextTransOp = transOp;
    while (nextTransOp) {
      sim = simulate(sim, nextTransOp.getPermutation());
      if (sim == base) {
        rewriter.replaceOp(transOp, {nextTransOp.getInput()});
        return success();
      }
      nextTransOp = nextTransOp.getInput().getDefiningOp<linalg::TransposeOp>();
    }

    return failure();
  }
};

bool isConstOne(Value v) {
  auto type = getElementTypeOrSelf(v);
  if (isa<FloatType>(type)) {
    if (matchPattern(v, m_OneFloat())) {
      return true;
    }
  } else if (type.isIntOrIndex()) {
    if (matchPattern(v, m_One())) {
      return true;
    }
  }

  auto defineOp = v.getDefiningOp();
  if (!defineOp) {
    return false;
  }

  auto resIndx = cast<OpResult>(v).getResultNumber();
  if (auto fillOp = dyn_cast<linalg::FillOp>(defineOp)) {
    return isConstOne(fillOp.getOperand(resIndx));
  } else if (auto castOp = dyn_cast<hfusion::CastOp>(defineOp)) {
    return isConstOne(castOp.getOperand(resIndx));
  }

  return false;
}

bool isConstZero(Value v) {
  auto type = getElementTypeOrSelf(v);
  if (isa<FloatType>(type)) {
    if (matchPattern(v, m_PosZeroFloat()) ||
        matchPattern(v, m_NegZeroFloat())) {
      return true;
    }
  } else if (type.isIntOrIndex()) {
    if (matchPattern(v, m_Zero())) {
      return true;
    }
  }

  auto defineOp = v.getDefiningOp();
  if (!defineOp) {
    return false;
  }

  auto resIndx = cast<OpResult>(v).getResultNumber();
  if (auto fillOp = dyn_cast<linalg::FillOp>(defineOp)) {
    return isConstZero(fillOp.getOperand(resIndx));
  } else if (auto castOp = dyn_cast<hfusion::CastOp>(defineOp)) {
    return isConstZero(castOp.getOperand(resIndx));
  }

  return false;
}

template <typename AddOP>
LogicalResult simplifyAdd(PatternRewriter &rewriter, AddOP addOp) {
  if (isConstZero(addOp.getOperand(0))) {
    rewriter.replaceOp(addOp, addOp.getOperand(1));
    return success();
  }
  if (isConstZero(addOp.getOperand(1))) {
    rewriter.replaceOp(addOp, addOp.getOperand(0));
    return success();
  }
  return failure();
}

template <typename SubOP>
LogicalResult simplifySub(PatternRewriter &rewriter, SubOP subOp) {
  if (isConstZero(subOp.getOperand(1))) {
    rewriter.replaceOp(subOp, subOp.getOperand(0));
    return success();
  }
  return failure();
}

template <typename DivOP>
LogicalResult simplifyDiv(PatternRewriter &rewriter, DivOP divOp) {
  if (isConstOne(divOp.getOperand(1))) {
    rewriter.replaceOp(divOp, divOp.getOperand(0));
    return success();
  }
  return failure();
}

template <typename MulOP>
LogicalResult simplifyMul(PatternRewriter &rewriter, MulOP mulOp) {
  if (isConstOne(mulOp.getOperand(0))) {
    rewriter.replaceOp(mulOp, mulOp.getOperand(1));
    return success();
  }

  if (isConstOne(mulOp.getOperand(1))) {
    rewriter.replaceOp(mulOp, mulOp.getOperand(0));
    return success();
  }
  return failure();
}

template <typename BINOP>
struct ElemBinaryPattern : public OpRewritePattern<BINOP> {
public:
  using OpRewritePattern<BINOP>::OpRewritePattern;
  LogicalResult matchAndRewrite(BINOP binaryOp,
                                PatternRewriter &rewriter) const final {
    auto binaryFunc = binaryOp.getFun();
    if constexpr (std::is_same_v<BINOP, linalg::ElemwiseBinaryOp>) {
      if (binaryFunc == linalg::BinaryFn::add) {
        return simplifyAdd<BINOP>(rewriter, binaryOp);
      }

      if (binaryFunc == linalg::BinaryFn::mul) {
        return simplifyMul<BINOP>(rewriter, binaryOp);
      }

      if (binaryFunc == linalg::BinaryFn::sub) {
        return simplifySub<BINOP>(rewriter, binaryOp);
      }

      if (binaryFunc == linalg::BinaryFn::div) {
        return simplifyDiv<BINOP>(rewriter, binaryOp);
      }
    }

    return failure();
  }
};

void populateSimplifyOpsPattern(RewritePatternSet &patterns) {
  patterns.add<LoopedCastOpPattern>(patterns.getContext());
  patterns.add<CastOpPattern>(patterns.getContext());
  patterns.add<TransposeOpPattern>(patterns.getContext());
  patterns.add<ElemBinaryPattern<hfusion::ElemwiseBinaryOp>>(
      patterns.getContext());
  patterns.add<ElemBinaryPattern<linalg::ElemwiseBinaryOp>>(
      patterns.getContext());
}

void SimplifyOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateSimplifyOpsPattern(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createSimplifyOpsPass() {
  return std::make_unique<SimplifyOpsPass>();
}

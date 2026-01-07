//===- SimplifyOps.cpp ------- Simplify operations ------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

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

    // Process ops bottom-to-top.

    // Traverse the chain of input cast ops to see if an op with the same
    // input types can be found.
    CastOp nextCast = castOp;
    while (nextCast) {
      // In total cast chain, if one cast of chain has the same type input and
      // output, it should always represent an intended change in value, which
      // means it couldn't be erased. So cast chain must be split at this point
      if (nextCast.getInputs().getTypes() == nextCast.getResultTypes())
        break;

      if (nextCast.getInputs().getTypes() == castOp.getResultTypes()) {
        // Found a cast where the input types match the output types of the
        // matched op. We can directly use those inputs and the matched op can
        // be removed.
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

    // Verify that an iter_arg is only used in this cast
    BlockArgument castSrc = dyn_cast<BlockArgument>(castOp.getInputs().front());
    if (!castSrc || !castSrc.hasOneUse() ||
        llvm::find(parentFor.getRegionIterArgs(), castSrc) ==
            parentFor.getRegionIterArgs().end())
      return failure();

    // Get the yield op and yielded value of this for loop
    assert(
        isa<scf::YieldOp>(parentFor.getRegion().getBlocks().front().back()) &&
        "For loop doesn't terminates with yieldOp");
    scf::YieldOp yieldOp =
        cast<scf::YieldOp>(parentFor.getRegion().getBlocks().front().back());
    OpOperand *yieldedValue = parentFor.getTiedLoopYieldedValue(castSrc);
    assert(yieldedValue);
    assert(yieldedValue->get().getType() == castSrc.getType() &&
           "Yielded value type doesn't match loop's iter_arg type");
    OpOperand *loopInit = parentFor.getTiedLoopInit(castSrc);
    assert(loopInit);
    OpResult loopResult = parentFor.getTiedLoopResult(castSrc);
    assert(loopResult);

    // Verify that the yielded value is produced with another cast with source
    // types equal to current target types
    CastOp lastCast = yieldedValue->get().getDefiningOp<CastOp>();
    if (!lastCast || lastCast == castOp || !isSimpleCastOp(lastCast))
      return failure();
    if (castOp->getResultTypes() != lastCast.getInputs().getTypes())
      return failure();

    // do moving

    // move current cast before the for loop, using init value of iter_arg as
    // src
    rewriter.moveOpBefore(castOp, parentFor);
    rewriter.setInsertionPoint(castOp);
    Operation *castOutput = castOp.getOutputs()[0].getDefiningOp();
    assert(castOutput && "Cast op output is null!");
    mlir::tensor::EmptyOp buffer =
        cast<mlir::tensor::EmptyOp>(rewriter.clone(*castOutput));
    rewriter.replaceAllUsesWith(castOp->getResults(), castSrc);
    rewriter.modifyOpInPlace(castOp, [&]() {
      castOp.getOutputsMutable()[0].set(buffer.getResult());
      castOp.getInputsMutable()[0].set(loopInit->get());
    });

    // update yieldop to yield src of last cast
    rewriter.modifyOpInPlace(
        yieldOp, [&]() { yieldedValue->assign(lastCast.getInputs()[0]); });

    // move last cast after the for loop. using forOp result as src
    rewriter.moveOpAfter(lastCast, parentFor);
    rewriter.setInsertionPoint(lastCast);
    Operation *lastCastOp = lastCast.getOutputs()[0].getDefiningOp();
    assert(lastCastOp && "Last cast op is null!");
    buffer = cast<mlir::tensor::EmptyOp>(rewriter.clone(*lastCastOp));
    rewriter.replaceAllUsesWith(loopResult, lastCast.getResults());
    rewriter.modifyOpInPlace(lastCast, [&]() {
      lastCast.getOutputsMutable()[0].set(buffer.getResult());
      lastCast.getInputsMutable()[0].set(loopResult);
    });

    // update for op with new iter_arg type equal to casted type
    rewriter.modifyOpInPlace(parentFor, [&]() {
      castSrc.setType(castOp->getResultTypes()[0]);
      loopInit->set(castOp->getResults()[0]);
      loopResult.setType(castOp->getResultTypes()[0]);
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

    if constexpr (std::is_same_v<BINOP, hfusion::ElemwiseBinaryOp>) {
      if (binaryFunc == hfusion::BinaryFn::vxor) {
        // only deel with x vxor 0xff...
        return simplifyVxorToVnot(rewriter, binaryOp);
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

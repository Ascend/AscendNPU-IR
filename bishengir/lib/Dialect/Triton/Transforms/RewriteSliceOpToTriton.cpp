//===- RewriteSliceOpToTriton.cpp --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites a restricted form of `tensor.extract_slice` and
// `tensor.insert_slice` into sequences of Triton dialect ops (`tt.trans`,
// `tt.reshape`, `tt.split`, `tt.join`).  Both patterns share a common shape
// constraint: indexing a single contiguous power-of-two block along *one*
// axis `a` of an N-D tensor; every other axis must be passed through in
// full.  With N := large.dim[a], S := small.dim[a], block offset r:
//
//     offsets[a] = r       offsets[i!=a] = 0
//     sizes[a]   = S       sizes[i!=a]   = D_i
//     strides    = [1, 1, ..., 1]
//     large : tensor<D0 x ... x D_a (= N) x ... x D_{R-1} x T>
//     small : tensor<D0 x ... x        S  x ... x D_{R-1} x T>
//
// where every dim of the large tensor and S itself are powers of two, and
// r is static, in [0, N - S], and a multiple of S.  For `extract_slice`,
// large is the source and small is the result; for `insert_slice`, large
// is the dest (and the result type) and small is the source.
//
// Common pipeline (with m := r/S, k := log2(N/S)):
//
//   1. tt.reshape splits axis `a` from N into (N/S, S):
//        <..., N, ...>  ->  <..., N/S, S, ...>
//   2. tt.trans with order [0, ..., a-1, a+1, ..., R, a] moves the (N/S)
//      dim from position a to the innermost slot required by tt.split.
//   3. tt.reshape replaces the trailing (N/S) with log2(N/S) trailing 2's:
//        <..., S, ..., 2, 2, ..., 2>
//   4. log2(N/S) x tt.split, each peeling the innermost 2 and picking
//      lhs/rhs by bit `(m >> i) & 1`, LSB first.  After k splits the leaf
//      chunk has shape == small.
//
// `extract_slice` then yields the leaf chunk directly.  `insert_slice`
// instead replaces the leaf with its source operand and walks the chain
// back up:
//
//   5. (insert_slice only) k x tt.join (inverse of step 4) using the
//      "other halves" recorded during the splits, picking the side from
//      the same bit of m so the join puts the new chunk back where the
//      split took it from.
//   6. (insert_slice only) inverse of step 3, then inverse of step 2,
//      then inverse of step 1 -- recovers the dest's original shape.
//
// When the small and large shapes are identical (no axis differs) the
// slice is a no-op and the pattern just forwards the source / dest.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace bishengir::triton {
#define GEN_PASS_DEF_REWRITESLICEOPTOTRITON
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

static bool isPow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

static int log2Pow2(int64_t v) {
  int r = 0;
  while ((int64_t(1) << r) < v)
    ++r;
  return r;
}

// Description of how to break a single-axis power-of-two block out of the
// large tensor.  `axis == -1` indicates the slice is a no-op (large and
// small shapes are identical).
struct SlicePlan {
  int axis = -1;
  int64_t N = 0; // large.dim[axis]
  int64_t S = 0; // small.dim[axis]
  int64_t r = 0; // offset[axis]
  int64_t m = 0; // r / S, the block index
  int k = 0;     // log2(N / S)
};

// Validates the slice shape and offsets/sizes/strides arrays shared by
// both `tensor.extract_slice` and `tensor.insert_slice`.  `large` is the
// larger tensor (extract source / insert dest), `small` is the smaller
// tensor (extract result / insert source).  Emits a diagnostic on `op`
// and returns failure on any violation.
static FailureOr<SlicePlan>
planSliceRewrite(Operation *op, RankedTensorType large, RankedTensorType small,
                 ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides,
                 ArrayRef<int64_t> offsets) {
  SlicePlan plan;

  if (large.getRank() < 1 || large.getRank() != small.getRank()) {
    op->emitError("large and small tensors must have matching rank >= 1; "
                  "got large ")
        << large << " vs small " << small;
    return failure();
  }
  const int rank = large.getRank();

  for (int i = 0; i < rank; ++i) {
    if (!isPow2(large.getDimSize(i))) {
      op->emitError("large tensor dim ")
          << i << " must be a power of two; got " << large.getDimSize(i);
      return failure();
    }
  }

  // Find the unique differing axis.
  for (int i = 0; i < rank; ++i) {
    if (large.getDimSize(i) != small.getDimSize(i)) {
      if (plan.axis != -1) {
        op->emitError("only single-axis slicing is supported; both axis ")
            << plan.axis << " and axis " << i
            << " differ between large and small";
        return failure();
      }
      plan.axis = i;
    }
  }

  if (plan.axis == -1)
    return plan; // no-op: shapes match.

  plan.N = large.getDimSize(plan.axis);
  plan.S = small.getDimSize(plan.axis);
  if (!isPow2(plan.S)) {
    op->emitError("indexed-axis size on the small tensor must be a power "
                  "of two; got ")
        << plan.S;
    return failure();
  }
  // S < N is guaranteed because `axis` was detected as differing; both are
  // powers of two, so S divides N.

  if (static_cast<int>(sizes.size()) != rank ||
      static_cast<int>(strides.size()) != rank ||
      static_cast<int>(offsets.size()) != rank) {
    op->emitError("offsets/sizes/strides arity must match rank ") << rank;
    return failure();
  }

  for (int i = 0; i < rank; ++i) {
    const int64_t expected = (i == plan.axis) ? plan.S : large.getDimSize(i);
    if (sizes[i] != expected) {
      op->emitError("size[")
          << i << "] must be " << expected << "; got " << sizes[i];
      return failure();
    }
  }

  for (int i = 0; i < rank; ++i) {
    if (strides[i] != 1) {
      op->emitError("strides must all be 1; got non-unit stride at axis ") << i;
      return failure();
    }
  }

  for (int i = 0; i < rank; ++i) {
    if (ShapedType::isDynamic(offsets[i])) {
      op->emitError("offsets must be static");
      return failure();
    }
    if (i != plan.axis && offsets[i] != 0) {
      op->emitError("offset[") << i << "] must be 0; got " << offsets[i];
      return failure();
    }
  }
  plan.r = offsets[plan.axis];
  if (plan.r < 0 || plan.r > plan.N - plan.S) {
    op->emitError("offset at indexed axis ")
        << plan.axis << " must be in [0, " << (plan.N - plan.S) << "] for size "
        << plan.S << " on a dim of " << plan.N << "; got " << plan.r;
    return failure();
  }
  if (plan.r % plan.S != 0) {
    op->emitError("offset at indexed axis ")
        << plan.axis << " must be a multiple of size " << plan.S << "; got "
        << plan.r;
    return failure();
  }

  plan.m = plan.r / plan.S;
  plan.k = log2Pow2(plan.N / plan.S);
  return plan;
}

// Builds steps 1-4: reshape splits N -> (N/S, S), trans moves (N/S) inner,
// reshape replaces trailing (N/S) with k 2's, then k splits peel the bits
// of `m`.  Returns the leaf chunk (shape == small type) and pushes the
// "other half" of each split into `outOtherHalves` (innermost first).
static Value buildSplitChain(OpBuilder &builder, Location loc, Value large,
                             const SlicePlan &plan,
                             SmallVectorImpl<Value> &outOtherHalves) {
  auto largeType = cast<RankedTensorType>(large.getType());
  const int rank = largeType.getRank();
  const int64_t N = plan.N, S = plan.S, m = plan.m;
  const int axis = plan.axis;
  const int k = plan.k;

  // Step 1: split N into (N/S, S) at position `axis`.
  SmallVector<int64_t> step1Shape;
  step1Shape.reserve(rank + 1);
  for (int i = 0; i < rank; ++i) {
    if (i == axis) {
      step1Shape.push_back(N / S);
      step1Shape.push_back(S);
    } else {
      step1Shape.push_back(largeType.getDimSize(i));
    }
  }
  Value cur = builder.create<ReshapeOp>(loc, step1Shape, large,
                                        /*allowReorder=*/false);

  // Step 2: trans (N/S) at position `axis` to innermost.
  SmallVector<int> transOrder;
  transOrder.reserve(rank + 1);
  for (int i = 0; i < rank + 1; ++i)
    if (i != axis)
      transOrder.push_back(i);
  transOrder.push_back(axis);
  cur = builder.create<TransOp>(loc, cur, transOrder);

  // Step 3: replace trailing (N/S) with k trailing 2's.
  SmallVector<int64_t> step3Shape;
  step3Shape.reserve(rank + k);
  for (int i = 0; i < rank; ++i)
    step3Shape.push_back(i == axis ? S : largeType.getDimSize(i));
  for (int i = 0; i < k; ++i)
    step3Shape.push_back(2);
  cur = builder.create<ReshapeOp>(loc, step3Shape, cur,
                                  /*allowReorder=*/false);

  // Step 4: walk down k splits, recording the not-picked half each time.
  outOtherHalves.clear();
  outOtherHalves.reserve(k);
  for (int i = 0; i < k; ++i) {
    auto split = builder.create<SplitOp>(loc, cur);
    if (((m >> i) & 1) == 0) {
      cur = split.getOutLHS();
      outOtherHalves.push_back(split.getOutRHS());
    } else {
      cur = split.getOutRHS();
      outOtherHalves.push_back(split.getOutLHS());
    }
  }
  return cur;
}

// Builds the inverse of the split chain for `insert_slice`: starts from a
// replacement leaf, walks `k` joins back up using `otherHalves`, then
// applies the inverse of step 3 / step 2 / step 1 to recover `largeType`.
static Value buildJoinChain(OpBuilder &builder, Location loc,
                            RankedTensorType largeType, Value leaf,
                            const SlicePlan &plan,
                            ArrayRef<Value> otherHalves) {
  const int rank = largeType.getRank();
  const int64_t N = plan.N, S = plan.S, m = plan.m;
  const int axis = plan.axis;
  const int k = plan.k;

  Value cur = leaf;
  // Inverse of step 4: k joins.  The bit of `m` at level i tells us which
  // side `cur` was at when we split, so we put it back there.
  for (int i = k - 1; i >= 0; --i) {
    Value other = otherHalves[i];
    if (((m >> i) & 1) == 0)
      cur = builder.create<JoinOp>(loc, cur, other);
    else
      cur = builder.create<JoinOp>(loc, other, cur);
  }

  // Inverse of step 3: collapse the trailing k 2's back into (N/S).
  SmallVector<int64_t> invStep3Shape;
  invStep3Shape.reserve(rank + 1);
  for (int i = 0; i < rank; ++i)
    invStep3Shape.push_back(i == axis ? S : largeType.getDimSize(i));
  invStep3Shape.push_back(N / S);
  cur = builder.create<ReshapeOp>(loc, invStep3Shape, cur,
                                  /*allowReorder=*/false);

  // Inverse of step 2: move (N/S) from innermost back to position `axis`.
  // Inverse permutation of [0..a-1, a+1, ..., R, a] is
  // [0..a-1, R, a, a+1, ..., R-1].
  SmallVector<int> invTransOrder;
  invTransOrder.reserve(rank + 1);
  for (int i = 0; i < axis; ++i)
    invTransOrder.push_back(i);
  invTransOrder.push_back(rank);
  for (int i = axis; i < rank; ++i)
    invTransOrder.push_back(i);
  cur = builder.create<TransOp>(loc, cur, invTransOrder);

  // Inverse of step 1: combine (N/S, S) at positions (axis, axis+1) back
  // into N at position `axis`.
  cur = builder.create<ReshapeOp>(loc, largeType.getShape(), cur,
                                  /*allowReorder=*/false);
  return cur;
}

struct ExtractSliceToTritonPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<RankedTensorType>(op.getSourceType());
    auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!srcType || !resType) {
      op.emitError("expected ranked tensor source and result");
      return failure();
    }

    auto planOrFail =
        planSliceRewrite(op, srcType, resType, op.getStaticSizes(),
                         op.getStaticStrides(), op.getStaticOffsets());
    if (failed(planOrFail))
      return failure();
    SlicePlan plan = *planOrFail;

    if (plan.axis == -1) {
      // No-op: forward source.
      rewriter.replaceOp(op, op.getSource());
      return success();
    }

    SmallVector<Value> otherHalves;
    Value leaf = buildSplitChain(rewriter, op.getLoc(), op.getSource(), plan,
                                 otherHalves);
    rewriter.replaceOp(op, leaf);
    return success();
  }
};

struct InsertSliceToTritonPattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<RankedTensorType>(op.getSourceType());
    auto destType = dyn_cast<RankedTensorType>(op.getDestType());
    if (!srcType || !destType) {
      op.emitError("expected ranked tensor source and dest");
      return failure();
    }

    auto planOrFail =
        planSliceRewrite(op, destType, srcType, op.getStaticSizes(),
                         op.getStaticStrides(), op.getStaticOffsets());
    if (failed(planOrFail))
      return failure();
    SlicePlan plan = *planOrFail;

    if (plan.axis == -1) {
      // No-op: source fills the entire dest; result == source.
      rewriter.replaceOp(op, op.getSource());
      return success();
    }

    // Break dest into chunks and discard the chunk at position m (the
    // splits are still emitted because we need the other-halves to
    // rejoin around the new source).
    SmallVector<Value> otherHalves;
    (void)buildSplitChain(rewriter, op.getLoc(), op.getDest(), plan,
                          otherHalves);

    // Substitute the source operand for the discarded chunk and rejoin.
    Value reassembled = buildJoinChain(rewriter, op.getLoc(), destType,
                                       op.getSource(), plan, otherHalves);
    rewriter.replaceOp(op, reassembled);
    return success();
  }
};

class RewriteSliceOpToTritonPass
    : public impl::RewriteSliceOpToTritonBase<RewriteSliceOpToTritonPass> {
public:
  using RewriteSliceOpToTritonBase::RewriteSliceOpToTritonBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ExtractSliceToTritonPattern, InsertSliceToTritonPattern>(ctx);

    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
      mod.emitError("Unsupported tensor slicing operations found in the "
                    "SIMT kernel");
      signalPassFailure();
      return;
    }

    // Anything left over is an unsupported slice op; the pattern has
    // already emitted a specific diagnostic on the offending op, so attach
    // the high-level pass-failure message to the same op.
    Operation *firstUnhandled = nullptr;
    mod.walk([&](Operation *op) {
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op)) {
        firstUnhandled = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (firstUnhandled) {
      firstUnhandled->emitError("Unsupported tensor slicing operations found "
                                "in the SIMT kernel");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createRewriteSliceOpToTritonPass() {
  return std::make_unique<RewriteSliceOpToTritonPass>();
}

} // namespace bishengir::triton

//===- RewriteSliceOpToTriton.cpp --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites `tensor.insert_slice`,  `tensor.extract_slice`, and `tensor.extract`
// ops using operations from the arith and triton dialects
//
// For `tensor.insert_slice`/`tensor.extract_slice` operations with static
// offsets: (offsets known at compile time)
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
// For `tensor.extract_slice` with a dynamic offset:
// Lower an `extract_slice` with a *static* result shape and a *dynamic*
// offset on a single axis to `tt.gather`:
//
//   %r = tensor.extract_slice %src[%off] [N] [1]
//         : tensor<MxT> to tensor<NxT>
//   ==>
//   %range   = tt.make_range {start=0, end=N} : tensor<Nxi32>
//   %offI32  = arith.index_cast %off : index to i32
//   %splat   = tt.splat %offI32 : i32 -> tensor<Nxi32>
//   %indices = arith.addi %range, %splat : tensor<Nxi32>
//   %r       = tt.gather %src[%indices] {axis = K : i32}
//                : (tensor<MxT>, tensor<Nxi32>) -> tensor<NxT>
//
// Generalises to higher-rank tensors where exactly one axis is sliced and
// all other axes are pass-through (offset 0, size matching source dim).
// The indices tensor is broadcast across the non-sliced dims to satisfy
// `tt.gather`'s shape requirements (rank == source rank, shape == result
// shape).
//
//
// For `tensor.insert_slice` with a dynamic offset:
//
// Rewrites `tensor.insert_slice` operations using
// a sequence of many different arith and triton dialect ops
//
//     offsets[i!=a] = 0
//     sizes[a]   = S       sizes[i!=a]   = D_i
//     strides    = [1, 1, ..., 1]
//     large : tensor<D0 x ... x D_a (= N) x ... x D_{R-1} x T>
//     small : tensor<D0 x ... x        S  x ... x D_{R-1} x T>
//
// Requirements:
// - S | N (N is a multiple of S)
// - r is in [0, N - S] (but can be dynamic)
//
// Building mask:
// 1. Get the lower and upper offset bounds (lower bound = r, upper bound = r +
// S),
// 2. Use triton::MakeRangeOp to make a tensor with values 0, 1, ..., N
// 3. Use triton::ExpandDimsOp to expand the tensor from 2. to get a
// 1x1x...xNx1x...x1 tensor
// 4. Use triton::BroadcastOp to broadcast the tensor from 3. to get a
// D0xD1x...xNx...xD_{R-1} tensor
//      By this point the tensor entry values are what values would be taken if
//      we had r=value, S=1 Ex: entries equal to 0 would be True in the mask if
//      r=0, S=1
// 5. Use triton::SplatOp to create 2 D0xD1x...xNx...xD_{R-1} tensors,
//      1 filled with the lower bound, 1 filled with the upper bound from
//      step 1.
// 6. Use arith::CmpIOp with arith::CmpIPredicate::sge with the lower bound
// splat from 5. and 4.
// 7. Use arith::CmpIOp with arith::CmpIPredicate::slt with the upper bound
// splat from 5. and 4.
// 8. Calculate the final mask as arith::AndIOp of tensors from 6. and 7.
//      This takes the intersection of entries where their value from the tensor
//      from 4. satisfy r <= value < r + S
//
// For `tt.insert_slice` based on the mask:
// First expand the tensor we want to insert to large tensor shape (required to
// do arith.select) Have to be careful to make sure the expanded tensor elements
// where mask=true remain the same as insert tensor Then use arith.select and
// mask to select the input tensor when true, original tensor when false
//
// 1. Use triton::ExpandDimsOp to expand the dimension of insert tensor to
// D0xD1x...x1xSx...xD_{R-1}
//       since we cannot use tt.broadcast to broadcast the dimension S to N
//       (unless S=1)
// 2. Then use tt.broadcast to broadcast this tensor to
// D0xD1x...xN/SxSx...xD_{R-1}
//       As we broadcasted along an the dimension before S, we have not messed
//       around with the placement of the insert tensor
// 3. Use tt.reshape to reshape the tensor from 2. to D0xD1x...xNx...xD_{R-1}
// 4. Use arith.select to use mask to select elements from 3. when true,
// original tensor when false
// 5. Return tensor from 4.
//
//
// For `tensor.extract` we flatten to a 1D tensor, get the index of where the
// element should be, then use tt.gather + tt.unsplat to get the element. If the
// input tensor has exactly 1 element, we replace `tensor.extract` with
// tt.unsplat
//
// For `tensor.insert` we flatten to a 1D tensor, get the index of where the
// element should be, then use tt.select with a mask to insert the element If
// the input tensor has exactly 1 element, we replace `tensor.insert` with
// tt.splat
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
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

  bool isOffsetStatic = true;
  int64_t r = 0; // offset[axis]
  int64_t m = 0; // r / S, the block index

  Value dynamicR; // offset[axis] if the offset is dynamic
  int k = 0;      // log2(N / S)
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
      if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
        plan.isOffsetStatic = false;
        plan.dynamicR = insertSliceOp.getMixedOffsets()[i].get<Value>();
      } else {
        auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
        plan.isOffsetStatic = false;
        plan.dynamicR = extractSliceOp.getMixedOffsets()[i].get<Value>();
      }
    }
    if (i != plan.axis && offsets[i] != 0) {
      if (ShapedType::isDynamic(offsets[i])) {
        op->emitError("offset[")
            << i << "] must be 0; got a dynamic offset instead";
        return failure();
      } else {
        op->emitError("offset[") << i << "] must be 0; got " << offsets[i];
        return failure();
      }
    }
  }

  // If the offset (r) is not static, we can't do bound checks at compile time
  // or calculate m
  if (plan.isOffsetStatic) {
    for (int i = 0; i < rank; ++i) {
      if (!isPow2(large.getDimSize(i))) {
        op->emitError("large tensor dim ")
            << i << " must be a power of two; got " << large.getDimSize(i);
        return failure();
      }
    }

    if (!isPow2(plan.S)) {
      op->emitError("indexed-axis size on the small tensor must be a power "
                    "of two; got ")
          << plan.S;
      return failure();
    }

    plan.r = offsets[plan.axis];
    if (plan.r < 0 || plan.r > plan.N - plan.S) {
      op->emitError("offset at indexed axis ")
          << plan.axis << " must be in [0, " << (plan.N - plan.S)
          << "] for size " << plan.S << " on a dim of " << plan.N << "; got "
          << plan.r;
      return failure();
    }

    if (plan.r % plan.S != 0) {
      op->emitError("offset at indexed axis ")
          << plan.axis << " must be a multiple of size " << plan.S << "; got "
          << plan.r;
      return failure();
    }

    plan.m = plan.r / plan.S;
  }

  // If we have a `tensor.insert_slice` op, we must check that N is divisible by
  // S as we are not guaranteed N and S are multiples of 2
  bool isInsertSliceOp = isa<tensor::InsertSliceOp>(op);
  if (isInsertSliceOp && plan.N % plan.S != 0) {
    op->emitError(
        "dynamic insert_slice rewrite requires slice size of small tensor ")
        << plan.S << " to divide the dest dim " << plan.N;
    return failure();
  }

  plan.k = log2Pow2(plan.N / plan.S);
  return plan;
}

static Value getAxisIndices(OpBuilder &builder, Location loc,
                            RankedTensorType resType, const SlicePlan &plan) {
  const int axis = plan.axis;
  const int S = plan.S;
  const int64_t rank = resType.getRank();

  Type i32Type = builder.getI32Type();
  auto axisI32Ty = RankedTensorType::get({S}, i32Type);

  // make_range(0, N) + splat(offset_i32).
  Value range = builder.create<MakeRangeOp>(loc, axisI32Ty, /*start=*/0,
                                            /*end=*/static_cast<int32_t>(S));
  Value offI32 = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                                    plan.dynamicR);
  Value splat = builder.create<SplatOp>(loc, axisI32Ty, offI32);
  Value axisIndices = builder.create<arith::AddIOp>(loc, range, splat);

  // For >1-D tensors, reshape the 1-D index tensor to size 1 on every dim
  // except the sliced one, then broadcast to the full result shape.
  Value indices = axisIndices;
  if (rank != 1) {
    SmallVector<int64_t, 4> expandedShape(rank, 1);
    expandedShape[axis] = S;
    auto expandedTy =
        RankedTensorType::get(expandedShape, builder.getI32Type());
    Value expanded = builder.create<ReshapeOp>(loc, expandedTy, axisIndices,
                                               /*allowReorder=*/false);
    auto fullIdxTy =
        RankedTensorType::get(resType.getShape(), builder.getI32Type());
    indices = builder.create<BroadcastOp>(loc, fullIdxTy, expanded);
  }

  return indices;
}

static Value useIndicesToExtract(OpBuilder &builder, Location loc,
                                 Value indices, Value large,
                                 RankedTensorType resType, const SlicePlan &plan) {
  auto res = builder.create<GatherOp>(loc, resType, large, indices, plan.axis);
  return res.getResult();
}

// Used in the case we have a dynamic offset
// Creates a mask tensor the size of the large tensor, which has True values
// where we want to insert into, and False everywhere else.
static Value buildMask(OpBuilder &builder, Location loc, Value large,
                       const SlicePlan &plan) {
  RankedTensorType largeType = cast<RankedTensorType>(large.getType());
  const int rank = largeType.getRank();
  const int64_t S = plan.S;
  const int64_t N = plan.N;
  const int axis = plan.axis;
  ArrayRef<int64_t> largeShape = largeType.getShape();
  SmallVector<int64_t> curShape = {N};
  Type i32Type = builder.getI32Type();
  Type i1Type = builder.getI1Type();
  RankedTensorType i1LargeType = RankedTensorType::get(largeShape, i1Type);

  auto indexToIntCast =
      builder.create<arith::IndexCastOp>(loc, i32Type, plan.dynamicR);
  RankedTensorType oneDShape = RankedTensorType::get(curShape, i32Type);
  Value cur = builder.create<triton::MakeRangeOp>(loc, oneDShape, 0, N);

  if (S > 1) {
    auto constantS = builder.create<arith::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(S));
    auto upperBound =
        builder.create<arith::AddIOp>(loc, i32Type, constantS, indexToIntCast);

    // creating 1D mask
    auto offsetLowerSplat =
        builder.create<triton::SplatOp>(loc, oneDShape, indexToIntCast);
    auto offsetUpperSplat =
        builder.create<triton::SplatOp>(loc, oneDShape, upperBound);

    auto lowerMask = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, cur, offsetLowerSplat);
    auto upperMask = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, cur, offsetUpperSplat);

    cur = builder.create<arith::AndIOp>(
        loc, RankedTensorType::get(curShape, i1Type), lowerMask, upperMask);
  } else {
    // Don't need upper/lower masks when S is 1
    auto offsetSplat =
        builder.create<triton::SplatOp>(loc, oneDShape, indexToIntCast);
    cur = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, cur,
                                        offsetSplat);
  }

  // We now expand our tensor cur to be tensor<1x1x...xNx1x...x1xT>
  int dimensionInsertLoc = 0;
  for (int i = 0; i < rank; i++) {
    if (i == axis) {
      dimensionInsertLoc = i + 1;
      continue;
    }
    if (dimensionInsertLoc == 0) {
      curShape.insert(curShape.begin(), 1);
    } else {
      curShape.push_back(1);
    }

    RankedTensorType curType = RankedTensorType::get(curShape, i1Type);
    cur = builder.create<triton::ExpandDimsOp>(loc, curType, cur,
                                               dimensionInsertLoc);
  }

  // broadcast it to the size of the large tensor
  cur = builder.create<triton::BroadcastOp>(loc, i1LargeType, cur);

  return cur;
}

// Used in the case we have a dynamic offset
// Uses the mask calculated earlier to insert the tensorToInsert where mask=true
// Requires S | N (N is a multiple of S)
static Value useMaskToInsert(OpBuilder &builder, Location loc, Value large,
                             Value mask, Value tensorToInsert,
                             const SlicePlan &plan) {
  RankedTensorType largeType = cast<RankedTensorType>(large.getType());
  ArrayRef<int64_t> smallShape =
      cast<RankedTensorType>(tensorToInsert.getType()).getShape();
  Type elementType = largeType.getElementType();
  const int64_t S = plan.S;
  const int64_t N = plan.N;
  const int axis = plan.axis;

  // Calculate shape after expanding by inserting 1 before dimension S
  SmallVector<int64_t> curShape(smallShape.begin(), smallShape.end());
  curShape.insert(curShape.begin() + axis, 1);

  // Expand
  auto expandedTensor = builder.create<triton::ExpandDimsOp>(
      loc, RankedTensorType::get(curShape, elementType), tensorToInsert, axis);

  // Calculate shape after broadcasting (the 1 we inserted will turn into N/S)
  curShape[axis] = N / S;

  // Perform broadcast
  auto broadcastedTensor = builder.create<triton::BroadcastOp>(
      loc, RankedTensorType::get(curShape, elementType), expandedTensor);

  // Reshape to collapse the N/S and S dimensions to a dimension of size N
  auto reshapedTensor =
      builder.create<triton::ReshapeOp>(loc, largeType, broadcastedTensor);

  // Select reshaped insert tensor when true, original tensor when false
  Value res = builder.create<arith::SelectOp>(loc, largeType, mask,
                                              reshapedTensor, large);
  return res;
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

// Given indices to a multi dimension tensor, finds the 1D index when flattened
static Value get1DIndex(OpBuilder &builder, Location loc, Value source,
                        const SmallVector<Value> &indices) {
  RankedTensorType sourceType = cast<RankedTensorType>(source.getType());
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  Type i32Type = builder.getI32Type();

  const int rank = sourceType.getRank();
  SmallVector<Value> dynamicStrides;
  dynamicStrides.reserve(rank);

  int64_t accumulatedIndex = 0;
  int64_t curStride = 1;

  for (int i = 0; i < rank; i++) {
    Value curIndex = indices[rank - 1 - i];
    int64_t dimSize = sourceShape[rank - 1 - i];

    if (auto staticVal = getConstantIntValue(curIndex)) {
      // static
      int64_t val = *staticVal;
      accumulatedIndex += val * curStride;
    } else {
      // dynamic
      auto indexToIntCast =
          builder.create<arith::IndexCastOp>(loc, i32Type, curIndex);
      auto constantStride = builder.create<arith::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(curStride));
      auto indexOffset = builder.create<arith::MulIOp>(
          loc, i32Type, indexToIntCast, constantStride);
      dynamicStrides.push_back(indexOffset);
    }

    curStride *= dimSize;
  }

  Value resIndex = builder.create<arith::ConstantOp>(
      loc, i32Type, builder.getI32IntegerAttr(accumulatedIndex));
  for (size_t i = 0; i < dynamicStrides.size(); i++) {
    resIndex = builder.create<arith::AddIOp>(loc, i32Type, resIndex,
                                             dynamicStrides[i]);
  }

  return resIndex;
}

struct ExtractToTritonPattern : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = op.getTensor();
    RankedTensorType sourceType = cast<RankedTensorType>(source.getType());

    // If there is just one element we can directly use unsplat op
    if (sourceType.getNumElements() == 1) {
      auto res = rewriter.create<triton::UnsplatOp>(loc, op.getTensor());
      rewriter.replaceOp(op, res->getResults());
      return success();
    }

    Type i32Type = rewriter.getI32Type();
    Type elementType = sourceType.getElementType();

    int64_t numElements = sourceType.getNumElements();
    RankedTensorType flattenedType =
        RankedTensorType::get({numElements}, elementType);
    auto flattenedTensor =
        rewriter.create<ReshapeOp>(loc, flattenedType, source);

    SmallVector<Value> indices = llvm::to_vector(op.getIndices());
    Value resIndex = get1DIndex(rewriter, loc, source, indices);
    resIndex = rewriter.create<SplatOp>(
        loc, RankedTensorType::get({1}, i32Type), resIndex);

    auto gatherOp =
        rewriter.create<GatherOp>(loc, flattenedTensor, resIndex, 0);
    auto res = rewriter.create<UnsplatOp>(loc, elementType, gatherOp);
    rewriter.replaceOp(op, res->getResults());
    return success();
  }
};

// Rewrites tensor.insert patterns by reshaping to a 1D tensor and using
// arith.select and reshaping back
struct InsertToTritonPattern : public OpRewritePattern<tensor::InsertOp> {
  using OpRewritePattern<tensor::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value dest = op.getDest();

    RankedTensorType destType = cast<RankedTensorType>(dest.getType());

    // If size is 1 just splat the scalar to the tensor shape wanted
    if (destType.getNumElements() == 1) {
      auto res =
          rewriter.create<triton::SplatOp>(loc, destType, op.getScalar());
      rewriter.replaceOp(op, res->getResults());
      return success();
    }

    Type i32Type = rewriter.getI32Type();
    Type elementType = destType.getElementType();

    int64_t numElements = destType.getNumElements();
    RankedTensorType flattenedType =
        RankedTensorType::get({numElements}, elementType);
    auto flattenedTensor =
        rewriter.create<ReshapeOp>(loc, flattenedType, dest);

    SmallVector<Value> indices = llvm::to_vector(op.getIndices());
    Value resIndex = get1DIndex(rewriter, loc, dest, indices);
    resIndex = rewriter.create<SplatOp>(
        loc, RankedTensorType::get({numElements}, i32Type), resIndex);

    Value scalar = op.getScalar();
    auto splattedScalar =
        rewriter.create<triton::SplatOp>(loc, flattenedType, scalar);

    auto range = rewriter.create<triton::MakeRangeOp>(
        loc, RankedTensorType::get({numElements}, i32Type), 0, numElements);
    auto mask = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               range, resIndex);

    auto flattenedRes = rewriter.create<arith::SelectOp>(
        loc, mask, splattedScalar, flattenedTensor);
    auto res =
        rewriter.create<triton::ReshapeOp>(loc, destType, flattenedRes);

    rewriter.replaceOp(op, res->getResults());
    return success();
  }
};

struct ExtractSliceToTritonPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSourceType());
    auto resType = cast<RankedTensorType>(op.getResult().getType());

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

    Value leaf;

    if (plan.isOffsetStatic) {
      SmallVector<Value> otherHalves;
      leaf = buildSplitChain(rewriter, op.getLoc(), op.getSource(), plan,
                             otherHalves);
    } else {
      Value indices = getAxisIndices(rewriter, op.getLoc(), resType, plan);
      leaf = useIndicesToExtract(rewriter, op.getLoc(), indices, op.getSource(),
                                 resType, plan);
    }
    rewriter.replaceOp(op, leaf);
    return success();
  }
};

struct InsertSliceToTritonPattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSourceType());
    auto destType = cast<RankedTensorType>(op.getDestType());

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

    Value reassembled;

    if (plan.isOffsetStatic) {
      // Break dest into chunks and discard the chunk at position m (the
      // splits are still emitted because we need the other-halves to
      // rejoin around the new source).
      SmallVector<Value> otherHalves;
      (void)buildSplitChain(rewriter, op.getLoc(), op.getDest(), plan,
                            otherHalves);

      // Substitute the source operand for the discarded chunk and rejoin.
      reassembled = buildJoinChain(rewriter, op.getLoc(), destType,
                                   op.getSource(), plan, otherHalves);
    } else {
      Value mask = buildMask(rewriter, op.getLoc(), op.getDest(), plan);
      reassembled = useMaskToInsert(rewriter, op.getLoc(), op.getDest(), mask,
                                    op.getSource(), plan);
    }
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
    patterns.add<ExtractSliceToTritonPattern, InsertSliceToTritonPattern,
                 ExtractToTritonPattern, InsertToTritonPattern>(ctx);

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
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp, tensor::ExtractOp, tensor::InsertOp>(
              op)) {
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

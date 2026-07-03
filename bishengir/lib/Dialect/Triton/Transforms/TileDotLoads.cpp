//===-- TileDotLoads.cpp - Tile tt.dot load inputs to reduce reg spill ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass overview
// -------------
// Tiles over-budget tt.dot ops (M*K*N > 32^3) so peak live registers fit:
//
//   Both operands are tt.loads -> K-tiling: scf.for over K/kTile, with
//     acc [M,N] carried as an iter_arg. Repeated smaller loads share the
//     same base ptr so they tend to hit DCache.
//
//   One operand is a register chain -> StageNonLoadOperandPattern: route
//     the non-load operand through scratch SHM and emit an unrolled
//     K-tile chain.
//
// Pointer styles supported
//   Block pointer   (!tt.ptr<tensor<...>>)  via tt.make_tensor_ptr/tt.advance
//   Tensor-of-ptrs  (tensor<...x!tt.ptr<T>>) via splat/addptr/make_range
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Triton/Transforms/SharedMemConflictModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace bishengir {
namespace triton {
/// Cycles per spilled element charged in `decideStaging`. Mirrors Ascend's
/// GM-roundtrip cost per element; calibration constant, not a tunable.
static constexpr unsigned kStageSpillCyclesPerElement = 200;
} // namespace triton
} // namespace bishengir

#define DEBUG_TYPE "tile-dot-loads"
#define DBGS() (llvm::dbgs() << "[TileDotLoads] ")

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_TILEDOTLOADS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
using namespace mlir;
using namespace mlir::triton;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static uint64_t floorPow2(uint64_t v) {
  if (v == 0)
    return 1;
  return 1ull << llvm::Log2_64(v);
}

/// Is the result type of `op` a ranked tensor of pointers
/// (tensor<...x!tt.ptr<T>>)?
static bool isTensorOfPtrs(Type ty) {
  auto rt = dyn_cast<RankedTensorType>(ty);
  return rt && isa<triton::PointerType>(rt.getElementType());
}

/// Is `ty` a block pointer  (!tt.ptr<tensor<...>>)?
static bool isBlockPtr(Type ty) {
  auto pt = dyn_cast<triton::PointerType>(ty);
  return pt && isa<RankedTensorType>(pt.getPointeeType());
}

enum class PtrStyle { BlockPtr, TensorOfPtrs, Unknown };

static PtrStyle classifyPtr(Value v) {
  if (isBlockPtr(v.getType()))
    return PtrStyle::BlockPtr;
  if (isTensorOfPtrs(v.getType()))
    return PtrStyle::TensorOfPtrs;
  return PtrStyle::Unknown;
}

// Wrapper for an op in a load chain, annotated with whether it's on the K-Line,
// and whether the value/data it holds is "transposed" (i.e. whether it has been
// transposed an odd number of times).
struct OpWrapper {
  Operation *op;
  bool K_Line = true;
  bool notTransposed = true;
};
/// Operand handle that lets us K-tile through a single `tt.trans` between
/// a load and a dot.  `load` is the underlying GMEM tt.load (null if the
/// operand isn't a load chain).  `trans` is the `tt.trans` between the
/// load and the dot operand (null if the operand is a direct load).  We
/// only handle a single 2-D transpose because that's the only shape Triton
/// emits between a GMEM load and a dot in flash-attention-style kernels.
struct DotLoadInfo {
  triton::LoadOp load;
  triton::TransOp trans;
};
static DotLoadInfo getDotLoadInfo(Value v) {
  DotLoadInfo info{nullptr, nullptr};
  if (auto ld = v.getDefiningOp<triton::LoadOp>()) {
    info.load = ld;
    return info;
  }
  if (auto tr = v.getDefiningOp<triton::TransOp>()) {
    if (auto ld = tr.getSrc().getDefiningOp<triton::LoadOp>()) {
      info.load = ld;
      info.trans = tr;
    }
  }
  return info;
}

/// Attribute key that prevents re-processing.
static constexpr llvm::StringLiteral kTiledAttr = "bishengir.dot.tiled";

//===----------------------------------------------------------------------===//
// Type substitution helper
//===----------------------------------------------------------------------===//

/// Replace `oldDim` with `newDim` in every tensor shape inside `ty`.
/// Handles RankedTensorType, PointerType(tensor); leaves scalar ptrs alone.
/// Also knows whether it's tiling the A operand (checkA=true) or the B operand (checkA=false)
/// and decides which dimension to tile accordingly (because of the transposes in the load chain,
/// the K dimension may be in dim 0 or dim 1, this decision is made prior to function call).
static Type tiledType(Type ty, int64_t oldDim, int64_t newDim, bool checkA) {
  if (auto rt = dyn_cast<RankedTensorType>(ty)) {
    SmallVector<int64_t> shape(rt.getShape());
    if (shape.size() == 2) {
      if (checkA && shape[1] == oldDim) {
        shape[1] = newDim;
      } else if (!checkA && shape[0] == oldDim) {
        shape[0] = newDim;
      }
    }
    return rt.clone(shape);
  }
  if (auto pt = dyn_cast<triton::PointerType>(ty)) {
    Type inner = tiledType(pt.getPointeeType(), oldDim, newDim, checkA);
    return triton::PointerType::get(inner, pt.getAddressSpace());
  }
  return ty;
}

/// A copy of tiledType that only handles the one-dimensional tiled type used for tile outputs.
/// This is needed to tile 1D vectors in the chain that are on the K_Line and are expanded later
/// to multiple dims.
static Type oneDimTiledType(Type ty, int64_t oldDim, int64_t newDim) {
  if (auto rt = dyn_cast<RankedTensorType>(ty)) {
    SmallVector<int64_t> shape(rt.getShape());
    if (shape.size() == 1 && shape[0] == oldDim) {
      shape[0] = newDim;
    }
    return rt.clone(shape);
  }
  if (auto pt = dyn_cast<triton::PointerType>(ty)) {
    Type inner = oneDimTiledType(pt.getPointeeType(), oldDim, newDim);
    return triton::PointerType::get(inner, pt.getAddressSpace());
  }
  return ty;
}
//===----------------------------------------------------------------------===//
// Tensor-of-pointers chain walking
//===----------------------------------------------------------------------===//

/// Collect ops defining the chain from `root` to `end` in topo order.
/// Returns false if `end` isn't reachable from `root`.
static bool collectChain(Value root, Value end,
                         SmallVectorImpl<OpWrapper> &chain, bool checkA) {
  // BFS backward from end until we hit root or run out of in-block defs.
  if (root == end)
    return true;
  SmallVector<Value, 8> worklist{end};
  llvm::SmallPtrSet<Operation *, 16> seen;
  SmallVector<OpWrapper> revOrder;
  bool retval = false;
  // Track if you're on the K_Line and if the current value is "transposed"
  SmallVector<bool> K_Line{true};
  SmallVector<bool> notTransposed{true};
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    bool cur_K_Line = K_Line.pop_back_val();
    bool cur_notTransposed = notTransposed.pop_back_val();
    if (cur == root)
      continue;
    Operation *def = cur.getDefiningOp();
    if (!def)
      continue;
    if (!seen.insert(def).second)
      continue;
    
    // If we see and expand_dims on the K_Line, we are no longer on the K_Line (K-Line is only 2D).
    // NOTE: checkA == cur_notTransposed is functionally the same as checkA XOR cur_notTransposed
    // checkA = true and not being transposed vs checkA = false and being transposed both mean that
    // the original K dimension is still in dim 1, so an expand_dims on either of those cases would
    // take you off the K-Line. Similar logic for other dim.
    if (auto expDimOp = dyn_cast<triton::ExpandDimsOp>(def)) {
      if (((checkA == cur_notTransposed) && expDimOp.getAxis() == 1) || (!(checkA == cur_notTransposed) && expDimOp.getAxis() == 0))
        cur_K_Line = false;
    } else if (auto tr = dyn_cast<triton::TransOp>(def)) {
      cur_notTransposed = !cur_notTransposed;
    }
    revOrder.push_back(OpWrapper{def, cur_K_Line, cur_notTransposed});
    for (Value operand : def->getOperands())
      if (operand != root) {
        worklist.push_back(operand);
        K_Line.push_back(cur_K_Line);
        notTransposed.push_back(cur_notTransposed);
      } else {
        retval = true;
      }
  }
  chain.append(revOrder.rbegin(), revOrder.rend());
  return retval;
}

/// Find the tt.make_range whose size equals `targetDim` along the pointer
/// axis of a tensor-of-pointers load. Returns nullptr if not found.
static triton::MakeRangeOp findMakeRange(Value loadPtr, int64_t targetDim) {
  SmallVector<Value, 8> worklist{loadPtr};
  llvm::SmallPtrSet<Value, 32> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    Operation *def = v.getDefiningOp();
    if (!def)
      continue;
    if (auto mr = dyn_cast<triton::MakeRangeOp>(def)) {
      int64_t rangeSize = static_cast<int64_t>(mr.getEnd()) - static_cast<int64_t>(mr.getStart());
      if (rangeSize == targetDim)
        return mr;
    }
    for (Value op : def->getOperands())
      worklist.push_back(op);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Helper to find the original MakeTensorPtrOp behind a load's pointer
//===----------------------------------------------------------------------===//

/// Walk a block-ptr load back to its originating tt.make_tensor_ptr,
/// following scf.for iter_args through their init values.
static triton::MakeTensorPtrOp getSourceMakeTensorPtr(triton::LoadOp load) {
  Value ptr = load.getPtr();
  while (true) {
    if (auto mk = ptr.getDefiningOp<triton::MakeTensorPtrOp>())
      return mk;

    auto blockArg = dyn_cast<BlockArgument>(ptr);
    if (!blockArg)
      return nullptr;
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return nullptr;
    unsigned idx = blockArg.getArgNumber() - forOp.getNumInductionVars();
    if (idx >= forOp.getInitArgs().size())
      return nullptr;
    ptr = forOp.getInitArgs()[idx];
  }
}

//===----------------------------------------------------------------------===//
// K-tiling: block pointer style
//===----------------------------------------------------------------------===//

static LogicalResult emitKTilingBlockPtr(triton::DotOp dot,
                                         triton::LoadOp loadA,
                                         triton::LoadOp loadB, int64_t kTile,
                                         PatternRewriter &rewriter) {
  Location loc = dot.getLoc();

  auto dTy = cast<RankedTensorType>(dot.getResult().getType());
  int64_t M = dTy.getDimSize(0);
  int64_t N = dTy.getDimSize(1);

  auto origMkA = getSourceMakeTensorPtr(loadA);
  auto origMkB = getSourceMakeTensorPtr(loadB);
  if (!origMkA || !origMkB)
    return failure();

  auto aTy = cast<RankedTensorType>(loadA.getResult().getType());
  int64_t K = aTy.getDimSize(1); // A is [M, K]

  // Build new ptr types: A -> [M, kTile], B -> [kTile, N].
  SmallVector<int32_t> tileShapeA{static_cast<int32_t>(M), static_cast<int32_t>(kTile)};
  SmallVector<int32_t> tileShapeB{static_cast<int32_t>(kTile), static_cast<int32_t>(N)};

  // Reconstruct from the original make_tensor_ptr even if outer loops
  // advanced the pointer; matches the desired lowering pattern.
  Value pA0 = rewriter.create<triton::MakeTensorPtrOp>(
      loc, origMkA.getBase(), origMkA.getShape(), origMkA.getStrides(),
      origMkA.getOffsets(), tileShapeA,
      SmallVector<int32_t>(origMkA.getOrder().begin(),
                           origMkA.getOrder().end()));
  Value pB0 = rewriter.create<triton::MakeTensorPtrOp>(
      loc, origMkB.getBase(), origMkB.getShape(), origMkB.getStrides(),
      origMkB.getOffsets(), tileShapeB,
      SmallVector<int32_t>(origMkB.getOrder().begin(),
                           origMkB.getOrder().end()));

  Value accInit = dot.getC();

  // scf.for %step = 0 to (K/kTile) step 1 iter_args(pA, pB, acc)
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value cNumTiles = rewriter.create<arith::ConstantIndexOp>(loc, K / kTile);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  auto forOp = rewriter.create<scf::ForOp>(loc, c0, cNumTiles, c1,
                                           ValueRange{pA0, pB0, accInit});

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());

    Value pA = forOp.getRegionIterArgs()[0];
    Value pB = forOp.getRegionIterArgs()[1];
    Value acc = forOp.getRegionIterArgs()[2];

    Value tA = rewriter.create<triton::LoadOp>(
        loc, pA, loadA.getBoundaryCheck(), loadA.getPadding(), loadA.getCache(),
        loadA.getEvict(), loadA.getIsVolatile());
    Value tB = rewriter.create<triton::LoadOp>(
        loc, pB, loadB.getBoundaryCheck(), loadB.getPadding(), loadB.getCache(),
        loadB.getEvict(), loadB.getIsVolatile());

    Value newAcc = rewriter.create<triton::DotOp>(loc, dTy, tA, tB, acc,
                                                  dot.getInputPrecision(),
                                                  dot.getMaxNumImpreciseAcc());

    // Advance: A by [0, kTile], B by [kTile, 0].
    auto i32 = rewriter.getI32Type();
    auto makeI32 = [&](int32_t v) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32,
                                                rewriter.getI32IntegerAttr(v));
    };

    Value advA = rewriter.create<triton::AdvanceOp>(
        loc, pA.getType(), pA, ValueRange{makeI32(0), makeI32(static_cast<int32_t>(kTile))});
    Value advB = rewriter.create<triton::AdvanceOp>(
        loc, pB.getType(), pB, ValueRange{makeI32(static_cast<int32_t>(kTile)), makeI32(0)});

    rewriter.create<scf::YieldOp>(loc, ValueRange{advA, advB, newAcc});
  }

  rewriter.replaceOp(dot, forOp.getResult(2));
  return success();
}

//===----------------------------------------------------------------------===//
// K-tiling: block-ptr style, outer-base reconstruction.
//
// Walks each load's pointer back to its outermost tt.make_tensor_ptr,
// accumulating advance contributions from any enclosing scf.for, then
// re-emits a fresh per-iter make_tensor_ptr with the K-axis offset set
// to (effective_outer_offset + inner_iv).
//
// vs. `emitKTilingBlockPtr`: one iter_arg (the accumulator) instead of
// three, and per-iter pointers share a hoistable dominator chain.
// Tile-size budget is enforced by `chooseTile`.
//===----------------------------------------------------------------------===//

namespace {
struct TracedPtr {
  triton::MakeTensorPtrOp rootMk;
  SmallVector<Value> effectiveOffsets;
};
} // namespace

static bool isZeroConstantI32(Value v) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
    return attr.getInt() == 0;
  return false;
}

/// SSA equality, or both arith.constant with the same integer value.
static bool areI32Equal(Value a, Value b) {
  if (a == b)
    return true;
  auto ca = a.getDefiningOp<arith::ConstantOp>();
  auto cb = b.getDefiningOp<arith::ConstantOp>();
  if (!ca || !cb)
    return false;
  auto aa = dyn_cast<IntegerAttr>(ca.getValue());
  auto ab = dyn_cast<IntegerAttr>(cb.getValue());
  return aa && ab && aa.getInt() == ab.getInt();
}

/// Walk a block-ptr back to its outermost tt.make_tensor_ptr, accumulating
/// per-axis offsets contributed by tt.advance through any number of nested
/// scf.for iter_args. Per-iter contribution is `(iv - lb)/step * advOff`,
/// short-circuiting when advOff == step.
/// Emits arithmetic at the rewriter's current insertion point. Returns
/// nullopt for unsupported chain shapes.
static std::optional<TracedPtr>
tracePtrToBase(Value ptr, PatternRewriter &rewriter, Location loc) {
  if (auto mk = ptr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    SmallVector<Value> effOffs(mk.getOffsets().begin(), mk.getOffsets().end());
    return TracedPtr{mk, std::move(effOffs)};
  }
  auto ba = dyn_cast<BlockArgument>(ptr);
  if (!ba)
    return std::nullopt;
  auto forOp = dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp());
  if (!forOp)
    return std::nullopt;
  unsigned idx = ba.getArgNumber() - forOp.getNumInductionVars();
  if (idx >= forOp.getInitArgs().size())
    return std::nullopt;

  auto innerInfo = tracePtrToBase(forOp.getInitArgs()[idx], rewriter, loc);
  if (!innerInfo)
    return std::nullopt;

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value yielded = yieldOp.getOperand(idx);
  auto adv = yielded.getDefiningOp<triton::AdvanceOp>();
  if (!adv || adv.getPtr() != ptr)
    return std::nullopt;

  Value iv = forOp.getInductionVar();
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();
  Type i32 = rewriter.getI32Type();

  Value diff;
  auto getDiff = [&]() -> Value {
    if (diff)
      return diff;
    Value ivI32 = iv;
    Value lbI32 = lb;
    if (iv.getType() != i32) {
      ivI32 = rewriter.create<arith::IndexCastOp>(loc, i32, iv);
      lbI32 = rewriter.create<arith::IndexCastOp>(loc, i32, lb);
    }
    diff = isZeroConstantI32(lbI32)
               ? ivI32
               : static_cast<Value>(rewriter.create<arith::SubIOp>(loc, ivI32, lbI32));
    return diff;
  };

  for (unsigned axis = 0; axis < adv.getOffsets().size(); ++axis) {
    Value advOff = adv.getOffsets()[axis];
    if (isZeroConstantI32(advOff))
      continue;
    Value contrib;
    if (areI32Equal(advOff, step)) {
      contrib = getDiff();
    } else {
      Value stepI32 = step;
      if (step.getType() != i32)
        stepI32 = rewriter.create<arith::IndexCastOp>(loc, i32, step);
      Value nIters = rewriter.create<arith::DivSIOp>(loc, getDiff(), stepI32);
      contrib = rewriter.create<arith::MulIOp>(loc, nIters, advOff);
    }
    Value &eff = innerInfo->effectiveOffsets[axis];
    eff = isZeroConstantI32(eff)
              ? contrib
              : static_cast<Value>(rewriter.create<arith::AddIOp>(loc, eff, contrib));
  }
  return innerInfo;
}

static LogicalResult
emitKTilingBlockPtrFromBase(triton::DotOp dot, triton::LoadOp loadA,
                            triton::LoadOp loadB, int64_t kTile,
                            PatternRewriter &rewriter,
                            StringRef tagAttrName = kTiledAttr) {
  Location loc = dot.getLoc();
  auto dTy = cast<RankedTensorType>(dot.getResult().getType());
  int64_t M = dTy.getDimSize(0);
  int64_t N = dTy.getDimSize(1);
  auto aTy = cast<RankedTensorType>(loadA.getResult().getType());
  int64_t K = aTy.getDimSize(1);

  rewriter.setInsertionPoint(dot);

  auto aChain = tracePtrToBase(loadA.getPtr(), rewriter, loc);
  auto bChain = tracePtrToBase(loadB.getPtr(), rewriter, loc);
  if (!aChain || !bChain)
    return failure();
  if (aChain->effectiveOffsets.size() != 2 ||
      bChain->effectiveOffsets.size() != 2)
    return failure();

  Type i32 = rewriter.getI32Type();

  // scf.for [0, K, kTile): load A/B slice, dot into acc, yield.
  Value c0 = rewriter.create<arith::ConstantOp>(loc, i32,
                                                rewriter.getI32IntegerAttr(0));
  Value cK = rewriter.create<arith::ConstantOp>(
      loc, i32, rewriter.getI32IntegerAttr(static_cast<int32_t>(K)));
  Value cKTile = rewriter.create<arith::ConstantOp>(
      loc, i32, rewriter.getI32IntegerAttr(static_cast<int32_t>(kTile)));

  auto forOp =
      rewriter.create<scf::ForOp>(loc, c0, cK, cKTile, ValueRange{dot.getC()});
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value acc = forOp.getRegionIterArgs()[0];

    auto buildPerTilePtr = [&](TracedPtr &chain, int kAxis, int64_t outShape0,
                               int64_t outShape1) -> Value {
      auto mk = chain.rootMk;
      SmallVector<Value> offs(chain.effectiveOffsets);
      offs[kAxis] =
          isZeroConstantI32(offs[kAxis])
              ? iv
              : static_cast<Value>(rewriter.create<arith::AddIOp>(loc, offs[kAxis], iv));
    SmallVector<int32_t> shape{static_cast<int32_t>(outShape0), static_cast<int32_t>(outShape1)};
      SmallVector<int32_t> orderVec(mk.getOrder().begin(), mk.getOrder().end());
      return rewriter.create<triton::MakeTensorPtrOp>(
          loc, mk.getBase(), mk.getShape(), mk.getStrides(), offs, shape,
          orderVec);
    };

    Value aPtr = buildPerTilePtr(*aChain, /*kAxis=*/1, M, kTile);
    Value bPtr = buildPerTilePtr(*bChain, /*kAxis=*/0, kTile, N);

    Value tA = rewriter.create<triton::LoadOp>(
        loc, aPtr, loadA.getBoundaryCheck(), loadA.getPadding(),
        loadA.getCache(), loadA.getEvict(), loadA.getIsVolatile());
    Value tB = rewriter.create<triton::LoadOp>(
        loc, bPtr, loadB.getBoundaryCheck(), loadB.getPadding(),
        loadB.getCache(), loadB.getEvict(), loadB.getIsVolatile());

    auto newAcc = rewriter.create<triton::DotOp>(loc, dTy, tA, tB, acc,
                                                 dot.getInputPrecision(),
                                                 dot.getMaxNumImpreciseAcc());
    newAcc->setAttr(tagAttrName, rewriter.getUnitAttr());

    rewriter.create<scf::YieldOp>(loc, ValueRange{newAcc.getResult()});
  }

  rewriter.replaceOp(dot, forOp.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// K-tiling: tensor-of-pointers style (chain-walking emitter).
//===----------------------------------------------------------------------===//

static Value cloneChainWithTiledRange(Value loadPtr,
                                      triton::MakeRangeOp origRange,
                                      Value tiledRange,
                                      PatternRewriter &rewriter, Location loc,
                                      int64_t K, int64_t kTile, bool checkA) {
  SmallVector<OpWrapper> chain;
  if (!collectChain(origRange.getResult(), loadPtr, chain, checkA))
    return nullptr;

  IRMapping mapping;
  mapping.map(origRange.getResult(), tiledRange);

  for (OpWrapper opW : chain) {
    Operation *op = opW.op;
    Operation *cloned;
    // If the op is on the K_Line we need to clone it with the mapping to update the range,
    // otherwise we can just clone it as is and update the mapping since the types shoudln't
    // change for ops that don't contribute to K dim later on
    if (opW.K_Line) {
      cloned = rewriter.clone(*op, mapping);
    } else {
      cloned = rewriter.clone(*op);
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), cloned->getResult(i));
      continue;
    }

    // constant ops are handled as a special case due to cloning limitations
    // NOTE: Again checkA == opW.notTransposed is functionally the same as
    // checkA XOR opW.notTransposed and the same logic as before applies for
    // deciding which dim to tile
    if (auto cstOp = dyn_cast<arith::ConstantOp>(*cloned)) {
      Type orig = cstOp.getType();
      Type tiled = tiledType(orig, K, kTile, checkA == opW.notTransposed);
      if (tiled != orig) {
        auto newAttr = cstOp.getValue();
        auto newAttrType = RankedTensorType::get(cast<RankedTensorType>(tiled).getShape(),
                                                 cast<RankedTensorType>(newAttr.getType()).getElementType());
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(newAttr)) {
          if (denseAttr.isSplat()) {
            newAttr = denseAttr.resizeSplat(newAttrType);
          } else {
            return nullptr;
          }
          } else {
            LLVM_DEBUG(DBGS() << "Unsupported constant type in tiled chain" << '\n');
          }
          
        cloned = rewriter.create<arith::ConstantOp>(cloned->getLoc(), tiled, newAttr);
      } else {
        tiled = oneDimTiledType(orig, K, kTile);
        if (tiled != orig) {
          auto newAttr = cstOp.getValue();
          auto newAttrType = RankedTensorType::get(cast<RankedTensorType>(tiled).getShape(),
                                                   cast<RankedTensorType>(newAttr.getType()).getElementType());
          if (auto denseAttr = dyn_cast<DenseElementsAttr>(newAttr)) {
            if (denseAttr.isSplat()) {
              newAttr = denseAttr.resizeSplat(newAttrType);
            } else {
              return nullptr;
            }
          } else {
            LLVM_DEBUG(DBGS() << "Unsupported constant type in tiled chain" << '\n');
          }
          cloned = rewriter.create<arith::ConstantOp>(cloned->getLoc(), tiled, newAttr);
        }
      }
    } else {
      for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
        Type orig = cloned->getResult(i).getType();
        Type tiled = tiledType(orig, K, kTile, checkA == opW.notTransposed);
        if (tiled != orig) {
          cloned->getResult(i).setType(tiled);
        } else {
          tiled = oneDimTiledType(orig, K, kTile);
          if (tiled != orig) {
            cloned->getResult(0).setType(tiled);
          }
        }
      }
    }
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      mapping.map(op->getResult(i), cloned->getResult(i));
  }
  return mapping.lookupOrNull(loadPtr);
}

static LogicalResult emitKTilingTensorOfPtrs(triton::DotOp dot,
                                             triton::LoadOp loadA,
                                             triton::LoadOp loadB,
                                             int64_t kTile,
                                             PatternRewriter &rewriter) {
  Location loc = dot.getLoc();

  auto dTy = cast<RankedTensorType>(dot.getResult().getType());

  auto aTy = cast<RankedTensorType>(loadA.getResult().getType());
  int64_t K = aTy.getDimSize(1);

  triton::MakeRangeOp mrA = findMakeRange(loadA.getPtr(), K);
  triton::MakeRangeOp mrB = findMakeRange(loadB.getPtr(), K);
  if (!mrA || !mrB)
    return failure();

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value cNumTiles = rewriter.create<arith::ConstantIndexOp>(loc, K / kTile);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  auto forOp = rewriter.create<scf::ForOp>(loc, c0, cNumTiles, c1,
                                           ValueRange{dot.getC()});
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());

    Value iv = forOp.getInductionVar();
    Value acc = forOp.getRegionIterArgs()[0];

  Value kTileVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(static_cast<int32_t>(kTile)));
    Value ivI32 =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), iv);
    Value kBase = rewriter.create<arith::MulIOp>(loc, ivI32, kTileVal);

    Value tileRange = rewriter.create<triton::MakeRangeOp>(
        loc, RankedTensorType::get({kTile}, rewriter.getI32Type()),
        /*start=*/0, /*end=*/static_cast<int32_t>(kTile));

    Value kBaseSplat = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get({kTile}, rewriter.getI32Type()), kBase);
    Value tiledRangeA =
        rewriter.create<arith::AddIOp>(loc, kBaseSplat, tileRange);
    Value tiledRangeB =
        rewriter.create<arith::AddIOp>(loc, kBaseSplat, tileRange);

    Value tiledPtrA = cloneChainWithTiledRange(loadA.getPtr(), mrA, tiledRangeA,
                                               rewriter, loc, K, kTile, true);
    Value tiledPtrB = cloneChainWithTiledRange(loadB.getPtr(), mrB, tiledRangeB,
                                               rewriter, loc, K, kTile, false);
    if (!tiledPtrA || !tiledPtrB)
      return failure();

    Value tA = rewriter.create<triton::LoadOp>(loc, tiledPtrA, loadA.getCache(),
                                               loadA.getEvict(),
                                               loadA.getIsVolatile());
    Value tB = rewriter.create<triton::LoadOp>(loc, tiledPtrB, loadB.getCache(),
                                               loadB.getEvict(),
                                               loadB.getIsVolatile());

    Value newAcc = rewriter.create<triton::DotOp>(loc, dTy, tA, tB, acc,
                                                  dot.getInputPrecision(),
                                                  dot.getMaxNumImpreciseAcc());
    rewriter.create<scf::YieldOp>(loc, ValueRange{newAcc});
  }

  rewriter.replaceOp(dot, forOp.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// Canonical K-tiling for tensor-of-pointers loads (with masks/other).
//
// Structural pattern-match on the IR `triton.language.load(...)` emits for
// a 2-D tile of a row-major matrix. Reuses K-invariant pieces (P_lo,
// M_mask_lo, kernel-arg scalar, stride splat, cmpi rhs) from outside the
// loop; rebuilds K-variant pieces inside around a fresh per-tile K range.
//
// Recognised shape:
//   A:[M,K] (K=axis 1): ptr = addptr(broadcast(P_lo:[M,1]Ptr),
//                                    broadcast(expand_dims(K_idx,0)))
//                        mask = andi(broadcast(M_mask_lo),
//                                    broadcast(cmpi(expand_dims(_,0), rhs)))
//   B:[K,N] (K=axis 0): ptr = addptr(broadcast(P_loK:[K,1]Ptr),
//                                    broadcast(O_2d:[1,N]))
//                        P_loK = addptr(splat(scalar),
//                                       [muli(]expand_dims(K_idx,1)[,stride])
//===----------------------------------------------------------------------===//

/// Re-emit a tt.splat or splat-constant at a different shape; nullptr
/// for non-splat values.
static Value resplatToShape(Value v, ArrayRef<int64_t> newShape,
                            PatternRewriter &rewriter, Location loc) {
  auto rt = dyn_cast<RankedTensorType>(v.getType());
  if (!rt)
    return nullptr;
  auto newTy = RankedTensorType::get(newShape, rt.getElementType());
  if (auto splat = v.getDefiningOp<triton::SplatOp>())
    return rewriter.create<triton::SplatOp>(loc, newTy, splat.getSrc());
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto da = dyn_cast<DenseElementsAttr>(cst.getValue())) {
      if (da.isSplat())
        return rewriter.create<arith::ConstantOp>(loc, newTy,
                                                  da.resizeSplat(newTy));
    }
  }
  return nullptr;
}

/// Decompose `andi(broadcast(maskLo), broadcast(maskLo))` (commutative) into
/// the K-axis half and the other-axis half.  Also accepts a single broadcast
/// (i.e. only one of the two halves present): if the broadcast's unit dim is
/// on the K axis it's a K-invariant ("other") mask; otherwise it's a K-axis
/// mask and the other half stays null.  Returns false on structure mismatch.
static bool splitMaskKvsOther(Value mask, int kAxis, Value &otherMaskLo,
                              Value &kMaskLo) {
  if (!mask)
    return true;
  auto unitDim = [](Value bv) -> int {
    auto t = dyn_cast<RankedTensorType>(bv.getType());
    if (!t || t.getRank() != 2)
      return -1;
    if (t.getDimSize(0) == 1 && t.getDimSize(1) > 1)
      return 0;
    if (t.getDimSize(1) == 1 && t.getDimSize(0) > 1)
      return 1;
    return -1;
  };
  if (auto andi = mask.getDefiningOp<arith::AndIOp>()) {
    auto bcL = andi.getLhs().getDefiningOp<triton::BroadcastOp>();
    auto bcR = andi.getRhs().getDefiningOp<triton::BroadcastOp>();
    if (!bcL || !bcR)
      return false;
    int axL = unitDim(bcL.getSrc());
    int axR = unitDim(bcR.getSrc());
    if (axL < 0 || axR < 0 || axL == axR)
      return false;
    // The mask whose unit dim sits at the K axis is K-invariant ("the
    // other mask"); its complement varies along K.
    if (axL == kAxis) {
      otherMaskLo = bcL.getSrc();
      kMaskLo = bcR.getSrc();
    } else {
      otherMaskLo = bcR.getSrc();
      kMaskLo = bcL.getSrc();
    }
    return true;
  }
  if (auto bc = mask.getDefiningOp<triton::BroadcastOp>()) {
    int ax = unitDim(bc.getSrc());
    if (ax < 0)
      return false;
    if (ax == kAxis)
      otherMaskLo = bc.getSrc(); // K-invariant
    else
      kMaskLo = bc.getSrc(); // K-axis
    return true;
  }
  return false;
}

/// Extract pred + rhs from `cmpi(pred, expand_dims(_, kExpandAxis), splat)`.
/// Caller re-emits the cmpi with a tiled lhs and resplatted rhs.
static bool extractKMaskCmpInfo(Value kMaskLo, int kExpandAxis,
                                arith::CmpIPredicate &pred,
                                Value &rhsSplatLike) {
  if (!kMaskLo)
    return true;
  auto cmp = kMaskLo.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return false;
  auto e = cmp.getLhs().getDefiningOp<triton::ExpandDimsOp>();
  if (!e || static_cast<int>(e.getAxis()) != kExpandAxis)
    return false;
  pred = cmp.getPredicate();
  rhsSplatLike = cmp.getRhs();
  return true;
}

/// K-tile canonical tensor-of-ptrs matmul loads (optionally masked).
/// `aInfo`/`bInfo` may carry an optional `tt.trans` between the underlying
/// load and the dot operand; in that case we tile the underlying load along
/// its K axis (which lives on the opposite tensor axis than the dot operand
/// sees) and re-apply the `tt.trans` to the per-tile result.  Only trans on
/// the B side is currently supported.
/// Leaves IR untouched and returns failure on any structural mismatch.
static LogicalResult
emitKTilingTensorOfPtrsCanonical(triton::DotOp dot, DotLoadInfo aInfo,
                                 DotLoadInfo bInfo, int64_t kTile,
                                 PatternRewriter &rewriter) {
  if (aInfo.trans)
    return failure(); // trans-on-A not yet supported here.
  triton::LoadOp loadA = aInfo.load;
  triton::LoadOp loadB = bInfo.load;
  Location loc = dot.getLoc();
  auto dTy = cast<RankedTensorType>(dot.getResult().getType());
  auto aTy = cast<RankedTensorType>(loadA.getResult().getType());
  // For trans-on-B, the underlying load has shape [N, K] (K on axis 1, like
  // A); the dot operand (post-trans) has shape [K, N] (K on axis 0).
  auto bLoadTy = cast<RankedTensorType>(loadB.getResult().getType());
  int64_t M = aTy.getDimSize(0);
  int64_t K = aTy.getDimSize(1);
  int64_t N = bInfo.trans ? bLoadTy.getDimSize(0) : bLoadTy.getDimSize(1);
  // Sanity: B underlying must have K on the expected axis.
  int64_t bLoadK = bInfo.trans ? bLoadTy.getDimSize(1) : bLoadTy.getDimSize(0);
  if (bLoadK != K || dTy.getDimSize(0) != M || dTy.getDimSize(1) != N)
    return failure();

  // -- A pointer decomposition ------------------------------------------------
  auto aTopAdd = loadA.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!aTopAdd)
    return failure();
  auto aPBcast = aTopAdd.getPtr().getDefiningOp<triton::BroadcastOp>();
  auto aOBcast = aTopAdd.getOffset().getDefiningOp<triton::BroadcastOp>();
  if (!aPBcast || !aOBcast)
    return failure();
  Value aPLo = aPBcast.getSrc(); // expect tensor<Mx1xPtr>
  Value aOLo = aOBcast.getSrc(); // expect tensor<1xKxi32>
  auto aPLoTy = dyn_cast<RankedTensorType>(aPLo.getType());
  auto aOLoTy = dyn_cast<RankedTensorType>(aOLo.getType());
  if (!aPLoTy || !aOLoTy || aPLoTy.getRank() != 2 || aOLoTy.getRank() != 2)
    return failure();
  if (aPLoTy.getDimSize(0) != M || aPLoTy.getDimSize(1) != 1)
    return failure();
  if (aOLoTy.getDimSize(0) != 1 || aOLoTy.getDimSize(1) != K)
    return failure();
  if (auto e = aOLo.getDefiningOp<triton::ExpandDimsOp>()) {
    if (e.getAxis() != 0)
      return failure();
  } else {
    return failure();
  }

  // -- B pointer decomposition ------------------------------------------------
  // Direct-B (no trans): underlying load is [K, N], K-variant ptr broadcast
  //   has shape [K, 1], K-invariant offset broadcast has shape [1, N].
  // Trans-B: underlying load is [N, K], same shape/layout as A's load.  We
  //   decompose it A-style: K-invariant ptr broadcast [N, 1], K-variant
  //   offset broadcast [1, K].  Inside the loop we then apply tt.trans to
  //   each tile so the dot gets [kTile, N].
  auto bTopAdd = loadB.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!bTopAdd)
    return failure();
  auto bPBcast = bTopAdd.getPtr().getDefiningOp<triton::BroadcastOp>();
  auto bOBcast = bTopAdd.getOffset().getDefiningOp<triton::BroadcastOp>();
  if (!bPBcast || !bOBcast)
    return failure();

  // Direct-B-only state (left null in trans-B path).
  Value bPLoK, bOLo;
  Value bBaseScalar, bStrideSplat;
  RankedTensorType bPLoTy;
  // Trans-B-only state (left null in direct-B path).
  Value bPLo_trans, bOLo_trans;
  RankedTensorType bPLoTy_trans;
  if (!bInfo.trans) {
    bPLoK = bPBcast.getSrc(); // expect tensor<Kx1xPtr>
    bOLo = bOBcast.getSrc();  // expect tensor<1xNxi32>
    bPLoTy = dyn_cast<RankedTensorType>(bPLoK.getType());
    auto bOLoTy = dyn_cast<RankedTensorType>(bOLo.getType());
    if (!bPLoTy || !bOLoTy || bPLoTy.getRank() != 2 || bOLoTy.getRank() != 2)
      return failure();
    if (bPLoTy.getDimSize(0) != K || bPLoTy.getDimSize(1) != 1)
      return failure();
    if (bOLoTy.getDimSize(0) != 1 || bOLoTy.getDimSize(1) != N)
      return failure();

    auto bPLoAdd = bPLoK.getDefiningOp<triton::AddPtrOp>();
    if (!bPLoAdd)
      return failure();
    auto bPSplat = bPLoAdd.getPtr().getDefiningOp<triton::SplatOp>();
    if (!bPSplat)
      return failure();
    bBaseScalar = bPSplat.getSrc();
    Value bRowOff = bPLoAdd.getOffset();

    // bRowOff is either expand_dims (stride 1) or muli(expand_dims,
    // splat-stride).
    if (auto mul = bRowOff.getDefiningOp<arith::MulIOp>()) {
      Value lhs = mul.getLhs(), rhs = mul.getRhs();
      if (auto e = lhs.getDefiningOp<triton::ExpandDimsOp>()) {
        if (e.getAxis() != 1)
          return failure();
        bStrideSplat = rhs;
      } else if (auto e = rhs.getDefiningOp<triton::ExpandDimsOp>()) {
        if (e.getAxis() != 1)
          return failure();
        bStrideSplat = lhs;
      } else {
        return failure();
      }
    } else if (auto e = bRowOff.getDefiningOp<triton::ExpandDimsOp>()) {
      if (e.getAxis() != 1)
        return failure();
    } else {
      return failure();
    }
  } else {
    // Trans-B: A-style decomposition on the underlying load [N, K].
    bPLo_trans = bPBcast.getSrc(); // expect tensor<Nx1xPtr>
    bOLo_trans = bOBcast.getSrc(); // expect tensor<1xKxi32>
    bPLoTy_trans = dyn_cast<RankedTensorType>(bPLo_trans.getType());
    auto bOLoTy_t = dyn_cast<RankedTensorType>(bOLo_trans.getType());
    if (!bPLoTy_trans || !bOLoTy_t || bPLoTy_trans.getRank() != 2 ||
        bOLoTy_t.getRank() != 2)
      return failure();
    if (bPLoTy_trans.getDimSize(0) != N || bPLoTy_trans.getDimSize(1) != 1)
      return failure();
    if (bOLoTy_t.getDimSize(0) != 1 || bOLoTy_t.getDimSize(1) != K)
      return failure();
    if (auto e = bOLo_trans.getDefiningOp<triton::ExpandDimsOp>()) {
      if (e.getAxis() != 0)
        return failure();
    } else {
      return failure();
    }
  }

  // -- Mask decomposition (optional) -----------------------------------------
  Value aOtherMaskLo, aKMaskLo;
  if (loadA.getMask() &&
      !splitMaskKvsOther(loadA.getMask(), /*kAxis=*/1, aOtherMaskLo, aKMaskLo))
    return failure();
  // For trans-B, the underlying load mask has K on axis 1 (A-style); for
  // direct-B, K is on axis 0.
  int bKAxis = bInfo.trans ? 1 : 0;
  int bKExpandAxis = bInfo.trans ? 0 : 1;
  Value bOtherMaskLo, bKMaskLo;
  if (loadB.getMask() && !splitMaskKvsOther(loadB.getMask(), bKAxis,
                                             bOtherMaskLo, bKMaskLo))
    return failure();

  arith::CmpIPredicate aKMaskPred{}, bKMaskPred{};
  Value aKMaskRhs, bKMaskRhs;
  if (!extractKMaskCmpInfo(aKMaskLo, /*kExpandAxis=*/0, aKMaskPred, aKMaskRhs))
    return failure();
  if (!extractKMaskCmpInfo(bKMaskLo, bKExpandAxis, bKMaskPred, bKMaskRhs))
    return failure();

  // -- Validation done; emit the K-tiling loop -------------------------------
  LLVM_DEBUG(DBGS() << "  -> canonical tensor-of-ptrs K-tiling: M="
                    << M << " K=" << K << " N=" << N << " kTile=" << kTile
                    << '\n');

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value cNumTiles = rewriter.create<arith::ConstantIndexOp>(loc, K / kTile);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Hoist loop-invariant pieces (the static `make_range`, the kTile
  // constant, and the K-invariant pointer broadcasts) out of the K-tile
  // loop body so PrefetchLoopLoads' prologue chain-clone doesn't reach
  // into the loop body for them 
  auto i32 = rewriter.getI32Type();
  auto i1 = rewriter.getI1Type();
  auto kRange1DTy = RankedTensorType::get({kTile}, i32);
  Value kRangeBase = rewriter.create<triton::MakeRangeOp>(
      loc, kRange1DTy, /*start=*/0, /*end=*/static_cast<int32_t>(kTile));
  Value kTileC = rewriter.create<arith::ConstantOp>(
      loc, i32, rewriter.getI32IntegerAttr(static_cast<int32_t>(kTile)));

  // A-side K-invariant ptr broadcast: [M, kTile] = broadcast(aPLo [M, 1]).
  auto aFullPtrTy = RankedTensorType::get({M, kTile}, aPLoTy.getElementType());
  Value hoistedTiledAPFull =
      rewriter.create<triton::BroadcastOp>(loc, aFullPtrTy, aPLo);

  // B-side K-invariant ptr broadcast (direct path only; trans path
  // hoists its own).
  Value hoistedTiledBPFull;
  RankedTensorType bFullPtrTyDirect, bUnderPtrTyTrans;
  if (!bInfo.trans) {
    bFullPtrTyDirect =
        RankedTensorType::get({kTile, N}, bPLoTy.getElementType());
  } else {
    bUnderPtrTyTrans =
        RankedTensorType::get({N, kTile}, bPLoTy_trans.getElementType());
    hoistedTiledBPFull =
        rewriter.create<triton::BroadcastOp>(loc, bUnderPtrTyTrans, bPLo_trans);
  }

  auto forOp = rewriter.create<scf::ForOp>(loc, c0, cNumTiles, c1,
                                           ValueRange{dot.getC()});
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value acc = forOp.getRegionIterArgs()[0];

    // tiledKRange1D = make_range(0, kTile) + splat(iv * kTile)
    Value ivI32 = rewriter.create<arith::IndexCastOp>(loc, i32, iv);
    Value kBase = rewriter.create<arith::MulIOp>(loc, ivI32, kTileC);
    Value kBaseSplat = rewriter.create<triton::SplatOp>(loc, kRange1DTy, kBase);
    Value tiledKRange1D =
        rewriter.create<arith::AddIOp>(loc, kBaseSplat, kRangeBase);

    // ---------------- A: tile load along K (axis 1) ------------------------
    auto aOLoTiledTy = RankedTensorType::get({1, kTile}, i32);
    Value tiledAOLo = rewriter.create<triton::ExpandDimsOp>(
        loc, aOLoTiledTy, tiledKRange1D, /*axis=*/0);
    auto aFullOffTy = RankedTensorType::get({M, kTile}, i32);
    Value tiledAOFull =
        rewriter.create<triton::BroadcastOp>(loc, aFullOffTy, tiledAOLo);
    Value tiledAPtr = rewriter.create<triton::AddPtrOp>(
        loc, aFullPtrTy, hoistedTiledAPFull, tiledAOFull);

    Value tiledAMask;
    if (loadA.getMask()) {
      auto aMaskTy = RankedTensorType::get({M, kTile}, i1);
      Value otherFull;
      if (aOtherMaskLo)
        otherFull =
            rewriter.create<triton::BroadcastOp>(loc, aMaskTy, aOtherMaskLo);
      Value kFull;
      if (aKMaskLo) {
        Value tiledRhs = resplatToShape(aKMaskRhs, {1, kTile}, rewriter, loc);
        if (!tiledRhs)
          return failure();
        Value tiledKMask2d = rewriter.create<arith::CmpIOp>(
            loc, aKMaskPred, tiledAOLo, tiledRhs);
        kFull = rewriter.create<triton::BroadcastOp>(loc, aMaskTy, tiledKMask2d);
      }
      if (otherFull && kFull)
        tiledAMask = rewriter.create<arith::AndIOp>(loc, otherFull, kFull);
      else
        tiledAMask = otherFull ? otherFull : kFull;
    }
    Value tiledAOther;
    if (loadA.getOther()) {
      tiledAOther = resplatToShape(loadA.getOther(), {M, kTile}, rewriter, loc);
      if (!tiledAOther)
        return failure();
    }

    Value tA;
    if (tiledAMask && tiledAOther)
      tA = rewriter.create<triton::LoadOp>(
          loc, tiledAPtr, tiledAMask, tiledAOther, loadA.getCache(),
          loadA.getEvict(), loadA.getIsVolatile());
    else if (tiledAMask)
      tA = rewriter.create<triton::LoadOp>(loc, tiledAPtr, tiledAMask,
                                           loadA.getCache(), loadA.getEvict(),
                                           loadA.getIsVolatile());
    else
      tA = rewriter.create<triton::LoadOp>(loc, tiledAPtr, loadA.getCache(),
                                           loadA.getEvict(),
                                           loadA.getIsVolatile());

    // ---------------- B: tile load along K --------------------------------
    Value tB;
    if (!bInfo.trans) {
      // Direct-B: K on axis 0, broadcast N on axis 1.
      auto bKIdx2DTiledTy = RankedTensorType::get({kTile, 1}, i32);
      Value tiledBKIdx2D = rewriter.create<triton::ExpandDimsOp>(
          loc, bKIdx2DTiledTy, tiledKRange1D, /*axis=*/1);
      Value tiledBRowOff = tiledBKIdx2D;
      if (bStrideSplat) {
        Value tiledStride =
            resplatToShape(bStrideSplat, {kTile, 1}, rewriter, loc);
        if (!tiledStride)
          return failure();
        tiledBRowOff =
            rewriter.create<arith::MulIOp>(loc, tiledBKIdx2D, tiledStride);
      }
      auto bPLoKTiledTy =
          RankedTensorType::get({kTile, 1}, bPLoTy.getElementType());
      Value tiledBSplat =
          rewriter.create<triton::SplatOp>(loc, bPLoKTiledTy, bBaseScalar);
      Value tiledBPLoK = rewriter.create<triton::AddPtrOp>(
          loc, bPLoKTiledTy, tiledBSplat, tiledBRowOff);
      auto bFullOffTy = RankedTensorType::get({kTile, N}, i32);
      Value tiledBPFull = rewriter.create<triton::BroadcastOp>(
          loc, bFullPtrTyDirect, tiledBPLoK);
      Value tiledBOFull =
          rewriter.create<triton::BroadcastOp>(loc, bFullOffTy, bOLo);
      Value tiledBPtr = rewriter.create<triton::AddPtrOp>(
          loc, bFullPtrTyDirect, tiledBPFull, tiledBOFull);

      Value tiledBMask;
      if (loadB.getMask()) {
        auto bMaskTy = RankedTensorType::get({kTile, N}, i1);
        Value otherFull;
        if (bOtherMaskLo)
          otherFull =
              rewriter.create<triton::BroadcastOp>(loc, bMaskTy, bOtherMaskLo);
        Value kFull;
        if (bKMaskLo) {
          Value tiledRhs = resplatToShape(bKMaskRhs, {kTile, 1}, rewriter, loc);
          if (!tiledRhs)
            return failure();
          Value tiledKMask2d = rewriter.create<arith::CmpIOp>(
              loc, bKMaskPred, tiledBKIdx2D, tiledRhs);
          kFull = rewriter.create<triton::BroadcastOp>(loc, bMaskTy,
                                                       tiledKMask2d);
        }
        if (otherFull && kFull)
          tiledBMask = rewriter.create<arith::AndIOp>(loc, kFull, otherFull);
        else
          tiledBMask = otherFull ? otherFull : kFull;
      }
      Value tiledBOther;
      if (loadB.getOther()) {
        tiledBOther =
            resplatToShape(loadB.getOther(), {kTile, N}, rewriter, loc);
        if (!tiledBOther)
          return failure();
      }

      if (tiledBMask && tiledBOther)
        tB = rewriter.create<triton::LoadOp>(
            loc, tiledBPtr, tiledBMask, tiledBOther, loadB.getCache(),
            loadB.getEvict(), loadB.getIsVolatile());
      else if (tiledBMask)
        tB = rewriter.create<triton::LoadOp>(loc, tiledBPtr, tiledBMask,
                                             loadB.getCache(), loadB.getEvict(),
                                             loadB.getIsVolatile());
      else
        tB = rewriter.create<triton::LoadOp>(loc, tiledBPtr, loadB.getCache(),
                                             loadB.getEvict(),
                                             loadB.getIsVolatile());
    } else {
      // Trans-B: A-style tile of underlying load [N, kTile], then apply
      // tt.trans to obtain [kTile, N] for the dot.  The tt.trans is a
      // layout/reorder op and is cheap on a tile-sized tensor.
      auto bOLoTiledTy = RankedTensorType::get({1, kTile}, i32);
      Value tiledBOLo = rewriter.create<triton::ExpandDimsOp>(
          loc, bOLoTiledTy, tiledKRange1D, /*axis=*/0);
      auto bUnderOffTy = RankedTensorType::get({N, kTile}, i32);
      Value tiledBOFull =
          rewriter.create<triton::BroadcastOp>(loc, bUnderOffTy, tiledBOLo);
      Value tiledBPtr = rewriter.create<triton::AddPtrOp>(
          loc, bUnderPtrTyTrans, hoistedTiledBPFull, tiledBOFull);

      Value tiledBMask;
      if (loadB.getMask()) {
        auto bMaskTy = RankedTensorType::get({N, kTile}, i1);
        Value otherFull;
        if (bOtherMaskLo)
          otherFull =
              rewriter.create<triton::BroadcastOp>(loc, bMaskTy, bOtherMaskLo);
        Value kFull;
        if (bKMaskLo) {
          Value tiledRhs = resplatToShape(bKMaskRhs, {1, kTile}, rewriter, loc);
          if (!tiledRhs)
            return failure();
          Value tiledKMask2d = rewriter.create<arith::CmpIOp>(
              loc, bKMaskPred, tiledBOLo, tiledRhs);
          kFull = rewriter.create<triton::BroadcastOp>(loc, bMaskTy,
                                                      tiledKMask2d);
        }
        if (otherFull && kFull)
          tiledBMask = rewriter.create<arith::AndIOp>(loc, otherFull, kFull);
        else
          tiledBMask = otherFull ? otherFull : kFull;
      }
      Value tiledBOther;
      if (loadB.getOther()) {
        tiledBOther =
            resplatToShape(loadB.getOther(), {N, kTile}, rewriter, loc);
        if (!tiledBOther)
          return failure();
      }

      Value tBUnder;
      if (tiledBMask && tiledBOther)
        tBUnder = rewriter.create<triton::LoadOp>(
            loc, tiledBPtr, tiledBMask, tiledBOther, loadB.getCache(),
            loadB.getEvict(), loadB.getIsVolatile());
      else if (tiledBMask)
        tBUnder = rewriter.create<triton::LoadOp>(
            loc, tiledBPtr, tiledBMask, loadB.getCache(), loadB.getEvict(),
            loadB.getIsVolatile());
      else
        tBUnder = rewriter.create<triton::LoadOp>(
            loc, tiledBPtr, loadB.getCache(), loadB.getEvict(),
            loadB.getIsVolatile());

      auto bLoadElemTy =
          cast<RankedTensorType>(loadB.getResult().getType()).getElementType();
      auto bDotTileTy = RankedTensorType::get({kTile, N}, bLoadElemTy);
      SmallVector<int32_t> order(bInfo.trans.getOrder().begin(),
                                 bInfo.trans.getOrder().end());
      tB = rewriter.create<triton::TransOp>(
          loc, bDotTileTy, tBUnder, rewriter.getDenseI32ArrayAttr(order));
    }

    Value newAcc = rewriter.create<triton::DotOp>(loc, dTy, tA, tB, acc,
                                                  dot.getInputPrecision(),
                                                  dot.getMaxNumImpreciseAcc());
    rewriter.create<scf::YieldOp>(loc, ValueRange{newAcc});
  }
  rewriter.replaceOp(dot, forOp.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// Tile-size selection
//===----------------------------------------------------------------------===//

static constexpr int64_t kMKNBudget = 32LL * 32 * 32; // 32768

// `TileDotPattern` only K-tiles dots whose A and B are both loads.  The
// "exactly one operand is a load" case (formerly handled by M/N-tiling
// emitters) is now handled structurally by `SplitDotOnLoadDimPattern`.
enum class TileStrategy { None, K };

struct TileInfo {
  TileStrategy strategy = TileStrategy::None;
  int64_t tileSize = 0;
  int64_t numTiles = 0;
};

static TileInfo chooseTile(triton::DotOp dot) {
  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto dTy = cast<RankedTensorType>(dot.getResult().getType());
  int64_t M = dTy.getDimSize(0);
  int64_t N = dTy.getDimSize(1);
  int64_t K = aTy.getDimSize(1);

  if (M * K * N <= kMKNBudget)
    return {};

  // Look through a single `tt.trans` between load and dot operand.
  DotLoadInfo aInfo = getDotLoadInfo(dot.getA());
  DotLoadInfo bInfo = getDotLoadInfo(dot.getB());

  auto isTileableLoad = [&](triton::LoadOp load, int64_t searchDim) -> bool {
    if (!load)
      return false;
    PtrStyle ps = classifyPtr(load.getPtr());
    if (ps == PtrStyle::BlockPtr)
      return getSourceMakeTensorPtr(load) != nullptr;
    if (ps == PtrStyle::TensorOfPtrs)
      return findMakeRange(load.getPtr(), searchDim) != nullptr;
    return false;
  };

  // Only fire when both sides are loads; the one-load case is owned by
  // `SplitDotOnLoadDimPattern` / `StageNonLoadOperandPattern`.
  if (!isTileableLoad(aInfo.load, K) || !isTileableLoad(bInfo.load, K))
    return {};

  int64_t maxKTile = kMKNBudget / std::max(int64_t(1), M * N);
  int64_t kTile = static_cast<int64_t>(floorPow2(static_cast<uint64_t>(maxKTile > 0 ? maxKTile : 1)));
  kTile = std::min(kTile, K);
  while (kTile > 1 && K % kTile != 0)
    kTile /= 2;
  if (kTile <= 0 || kTile >= K)
    return {};
  return {TileStrategy::K, kTile, K / kTile};
}

//===----------------------------------------------------------------------===//
// Rewrite pattern
//===----------------------------------------------------------------------===//

struct TileDotPattern : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern<triton::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOp dot,
                                PatternRewriter &rewriter) const override {
    if (dot->hasAttr(kTiledAttr))
      return failure();

    auto aTy = cast<RankedTensorType>(dot.getA().getType());
    auto dTy = cast<RankedTensorType>(dot.getResult().getType());
    int64_t M = dTy.getDimSize(0), N = dTy.getDimSize(1);
    int64_t K = aTy.getDimSize(1);

    LLVM_DEBUG(DBGS() << "examining tt.dot [M=" << M << " K=" << K << " N=" << N
                      << "] MKN=" << M * K * N << " budget=" << kMKNBudget
                      << '\n');

    TileInfo info = chooseTile(dot);
    if (info.strategy == TileStrategy::None) {
      LLVM_DEBUG(DBGS() << "  -> skip (MKN within budget, only one operand is a load, "
                        << "or no tileable load)" << '\n');
      return failure();
    }

    DotLoadInfo aInfo = getDotLoadInfo(dot.getA());
    DotLoadInfo bInfo = getDotLoadInfo(dot.getB());
    triton::LoadOp aLoad = aInfo.load;
    triton::LoadOp bLoad = bInfo.load;
    PtrStyle pA = classifyPtr(aLoad.getPtr());
    PtrStyle pB = classifyPtr(bLoad.getPtr());

    LLVM_DEBUG(DBGS() << "  -> K-tiling tileSize="
                      << info.tileSize << " numTiles=" << info.numTiles
                      << (aInfo.trans ? " (A has tt.trans)" : "")
                      << (bInfo.trans ? " (B has tt.trans)" : "") << '\n');

    LogicalResult res = failure();
    if (pA == PtrStyle::BlockPtr && pB == PtrStyle::BlockPtr) {
      // Block-ptr emitters don't yet handle tt.trans look-through; bail
      // and let other patterns deal with it.
      if (aInfo.trans || bInfo.trans) {
        LLVM_DEBUG(DBGS() << "  -> block-ptr emitters don't handle tt.trans yet, "
                          << "skipping" << '\n');
      } else {
        // Prefer outer-base emitter; fall back to chained-advance form.
        res = emitKTilingBlockPtrFromBase(dot, aLoad, bLoad, info.tileSize,
                                          rewriter);
        if (failed(res))
          res = emitKTilingBlockPtr(dot, aLoad, bLoad, info.tileSize, rewriter);
      }
    } else if (pA == PtrStyle::TensorOfPtrs && pB == PtrStyle::TensorOfPtrs) {
      // Canonical matmul/mask-aware emitter first; chain-walking fallback.
      res = emitKTilingTensorOfPtrsCanonical(dot, aInfo, bInfo, info.tileSize,
                                             rewriter);
      // Fallback only when neither operand has a tt.trans (the fallback
      // emitter doesn't yet support trans look-through).
      if (failed(res) && !aInfo.trans && !bInfo.trans)
        res =
            emitKTilingTensorOfPtrs(dot, aLoad, bLoad, info.tileSize, rewriter);
    } else {
      LLVM_DEBUG(DBGS() << "  -> mixed ptr styles, skipping" << '\n');
    }

    LLVM_DEBUG({
      if (succeeded(res))
        DBGS() << "  -> tiling succeeded" << '\n';
      else
        DBGS() << "  -> tiling failed (emitter returned failure)" << '\n';
    });

    return res;
  }
};

//===----------------------------------------------------------------------===//
// StageNonLoadOperandPattern — handles over-budget dots where at least one
// operand comes from a register chain (e.g. truncf(exp(...))) instead of
// a tt.load. Routes the non-load operand through scratch SHM so the dot
// can still be K-tiled.
//
// Steps:
//   1. Append a `!tt.ptr<elemTy, 6>` function arg tagged
//      `bishengir.scratch_shm`. `ConvertSharedPtrToMemDesc` lowers this
//      into `ttg.local_alloc` + per-tile `memdesc_subslice` accesses.
//   2. Emit one envelope-shape `tt.store` of the non-load operand into
//      the scratch arg. Pointer chain must match `matchScratchAccess`:
//        addptr(splat(scratch_arg), addi(rowSide, colSide))
//      where colSide = broadcast(expand_dims(make_range(0, K), axis=0)).
//   3. Replace the dot with an unrolled K-tile chain of (load A_tile,
//      load B_tile, dot into acc). Unrolled because `memdesc_subslice`
//      takes static offsets only.
//===----------------------------------------------------------------------===//

/// Append a `!tt.ptr<elemTy, 6>` function arg tagged `bishengir.scratch_shm`.
static BlockArgument appendScratchShmArg(triton::FuncOp func, Type elemTy,
                                         Location loc) {
  MLIRContext *ctx = func.getContext();
  unsigned newIdx = func.getNumArguments();
  auto sharedPtrTy = triton::PointerType::get(elemTy, /*addressSpace=*/6);
  auto unitAttr = UnitAttr::get(ctx);
  auto attrs = DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, "bishengir.scratch_shm"),
                           unitAttr)});
  func.insertArgument(newIdx, sharedPtrTy, attrs, loc);
  return func.getArgument(newIdx);
}

/// Build the scratch-shm pointer chain for one access.
static Value emitScratchShmAccessPtr(OpBuilder &b, Location loc,
                                     Value scratchArg, int64_t dimOther,
                                     int64_t tileSize, int64_t envSize,
                                     int kAxis, Value tileIdxI32,
                                     int64_t startConst, Type elemTy,
                                     int64_t otherStart = 0) {
  Type i32 = b.getI32Type();

  // Tile-axis side: broadcast(expand_dims(tile_1d, axis=otherAxis)).
  int otherAxis = 1 - kAxis;

  auto tile1DTy = RankedTensorType::get({tileSize}, i32);
  Value tileRange = b.create<triton::MakeRangeOp>(loc, tile1DTy, /*start=*/0,
                                                  /*end=*/static_cast<int32_t>(tileSize));
  Value tile1D = tileRange;
  if (tileIdxI32) {
    Value tileSizeC = b.create<arith::ConstantOp>(
        loc, i32, b.getI32IntegerAttr(static_cast<int32_t>(tileSize)));
    Value baseScalar = b.create<arith::MulIOp>(loc, tileIdxI32, tileSizeC);
    Value baseSplat = b.create<triton::SplatOp>(loc, tile1DTy, baseScalar);
    tile1D = b.create<arith::AddIOp>(loc, baseSplat, tileRange);
  } else if (startConst != 0) {
    Value baseScalar = b.create<arith::ConstantOp>(
        loc, i32, b.getI32IntegerAttr(static_cast<int32_t>(startConst)));
    Value baseSplat = b.create<triton::SplatOp>(loc, tile1DTy, baseScalar);
    tile1D = b.create<arith::AddIOp>(loc, baseSplat, tileRange);
  }
  // Expand to 2D with size 1 on otherAxis, then broadcast to full shape.
  SmallVector<int64_t, 2> tile2DShape =
      (kAxis == 1) ? SmallVector<int64_t, 2>{1, tileSize}
                   : SmallVector<int64_t, 2>{tileSize, 1};
  auto tile2DTy = RankedTensorType::get(tile2DShape, i32);
  Value tile2D = b.create<triton::ExpandDimsOp>(loc, tile2DTy, tile1D,
                                                /*axis=*/static_cast<uint32_t>(otherAxis));
  SmallVector<int64_t, 2> fullShape =
      (kAxis == 1) ? SmallVector<int64_t, 2>{dimOther, tileSize}
                   : SmallVector<int64_t, 2>{tileSize, dimOther};
  auto fullOffTy = RankedTensorType::get(fullShape, i32);
  Value tileFull = b.create<triton::BroadcastOp>(loc, fullOffTy, tile2D);

  // Non-tile side: strided 1D range so canonicalisation can't fold
  // the upcoming `addi`.  For the M-stripe path, `otherStart` shifts
  // the row window into the larger envelope without changing envSize.
  auto other1DTy = RankedTensorType::get({dimOther}, i32);
  Value otherRange = b.create<triton::MakeRangeOp>(
      loc, other1DTy, /*start=*/static_cast<int32_t>(otherStart),
      /*end=*/static_cast<int32_t>(otherStart + dimOther));
  Value envSizeScalar = b.create<arith::ConstantOp>(
      loc, i32, b.getI32IntegerAttr(static_cast<int32_t>(envSize)));
  Value envSizeSplat = b.create<triton::SplatOp>(loc, other1DTy, envSizeScalar);
  Value otherStrided = b.create<arith::MulIOp>(loc, otherRange, envSizeSplat);
  // Expand on `kAxis` so the "other" varies on the dimOther axis.
  SmallVector<int64_t, 2> other2DShape =
      (kAxis == 1) ? SmallVector<int64_t, 2>{dimOther, 1}
                   : SmallVector<int64_t, 2>{1, dimOther};
  auto other2DTy = RankedTensorType::get(other2DShape, i32);
  Value other2D = b.create<triton::ExpandDimsOp>(loc, other2DTy, otherStrided,
                                                 /*axis=*/static_cast<uint32_t>(kAxis));
  Value otherFull = b.create<triton::BroadcastOp>(loc, fullOffTy, other2D);

  // ---- offsets = addi(otherFull, tileFull) ------------------------------
  Value offsets = b.create<arith::AddIOp>(loc, otherFull, tileFull);

  // ---- splat(scratch_arg) + addptr --------------------------------------
  auto sharedPtrTy = triton::PointerType::get(elemTy, /*addressSpace=*/6);
  auto fullPtrTy = RankedTensorType::get(fullShape, sharedPtrTy);
  Value baseSplat = b.create<triton::SplatOp>(loc, fullPtrTy, scratchArg);
  return b.create<triton::AddPtrOp>(loc, fullPtrTy, baseSplat, offsets);
}

//===----------------------------------------------------------------------===//
// Tensor-of-ptrs per-tile load helpers (shared between StageNonLoadOperand-
// Pattern and the canonical TileDotPattern emitter).  Both extract the same
// K-invariant / K-variant pieces; the helpers below also re-emit each piece
// at the per-tile shape (K narrowed to `kTile`) using the loop's tile range.
//===----------------------------------------------------------------------===//

/// Decomposed pieces of a tensor-of-ptrs B-side load whose underlying shape
/// is [K, N] with K on axis 0 (i.e. K-variant).  Populated by
/// `decomposeTensorOfPtrsBKAxis0`.
struct TensorOfPtrsBInfo {
  Value bPLoK;        // tensor<K x 1 x Ptr>, K-variant
  Value bOLo;         // tensor<1 x N x i32>, K-invariant
  Value bBaseScalar;  // scalar Ptr (splat src for bPLoK)
  Value bStrideSplat; // optional splat<i32> stride; null = stride 1
  // Mask pieces (filled when load has a mask); see splitMaskKvsOther.
  Value bOtherMaskLo, bKMaskLo;
  arith::CmpIPredicate bKMaskPred{};
  Value bKMaskRhs;
};

/// Canonical decomposition of a tensor-of-ptrs `tt.load` of shape [K, N]
/// where K is on axis 0.  Returns nullopt on structural mismatch.  Mirrors
/// the B-direct decomposition path in `emitKTilingTensorOfPtrsCanonical`.
static std::optional<TensorOfPtrsBInfo>
decomposeTensorOfPtrsBKAxis0(triton::LoadOp loadB, int64_t K, int64_t N) {
  auto bTopAdd = loadB.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!bTopAdd)
    return std::nullopt;
  auto bPBcast = bTopAdd.getPtr().getDefiningOp<triton::BroadcastOp>();
  auto bOBcast = bTopAdd.getOffset().getDefiningOp<triton::BroadcastOp>();
  if (!bPBcast || !bOBcast)
    return std::nullopt;
  Value bPLoK = bPBcast.getSrc();
  Value bOLo = bOBcast.getSrc();
  auto bPLoTy = dyn_cast<RankedTensorType>(bPLoK.getType());
  auto bOLoTy = dyn_cast<RankedTensorType>(bOLo.getType());
  if (!bPLoTy || !bOLoTy || bPLoTy.getRank() != 2 || bOLoTy.getRank() != 2)
    return std::nullopt;
  if (bPLoTy.getDimSize(0) != K || bPLoTy.getDimSize(1) != 1)
    return std::nullopt;
  if (bOLoTy.getDimSize(0) != 1 || bOLoTy.getDimSize(1) != N)
    return std::nullopt;

  auto bPLoAdd = bPLoK.getDefiningOp<triton::AddPtrOp>();
  if (!bPLoAdd)
    return std::nullopt;
  auto bPSplat = bPLoAdd.getPtr().getDefiningOp<triton::SplatOp>();
  if (!bPSplat)
    return std::nullopt;
  TensorOfPtrsBInfo info;
  info.bPLoK = bPLoK;
  info.bOLo = bOLo;
  info.bBaseScalar = bPSplat.getSrc();
  Value bRowOff = bPLoAdd.getOffset();
  if (auto mul = bRowOff.getDefiningOp<arith::MulIOp>()) {
    Value lhs = mul.getLhs(), rhs = mul.getRhs();
    if (auto e = lhs.getDefiningOp<triton::ExpandDimsOp>()) {
      if (e.getAxis() != 1)
        return std::nullopt;
      info.bStrideSplat = rhs;
    } else if (auto e = rhs.getDefiningOp<triton::ExpandDimsOp>()) {
      if (e.getAxis() != 1)
        return std::nullopt;
      info.bStrideSplat = lhs;
    } else {
      return std::nullopt;
    }
  } else if (auto e = bRowOff.getDefiningOp<triton::ExpandDimsOp>()) {
    if (e.getAxis() != 1)
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  if (loadB.getMask() &&
      !splitMaskKvsOther(loadB.getMask(), /*kAxis=*/0, info.bOtherMaskLo,
                         info.bKMaskLo))
    return std::nullopt;
  if (!extractKMaskCmpInfo(info.bKMaskLo, /*kExpandAxis=*/1, info.bKMaskPred,
                            info.bKMaskRhs))
    return std::nullopt;
  return info;
}

/// Loop-invariant pieces of a tensor-of-ptrs K-axis-0 per-tile B load.
/// Built once outside the K-tile loop; passed to
/// `emitTensorOfPtrsBKAxis0Tile` per iter.  Hoisting these out prevents
/// PrefetchLoopLoads' prologue chain-clone from reaching into the loop
/// body for what is structurally a loop-invariant.
struct TensorOfPtrsBKAxis0Hoisted {
  Value tiledBSplat; // tensor<kTile x 1 x Ptr>, splat(bBaseScalar)
  Value tiledBOFull; // tensor<kTile x N x i32>, broadcast(bOLo)
  Value otherFull;   // tensor<kTile x N x i1>, broadcast(bOtherMaskLo) | null
  Value tiledRhs;    // tensor<kTile x 1 x i32>, resplat(bKMaskRhs) | null
  Value tiledStride; // tensor<kTile x 1 x i32>, resplat(bStrideSplat) | null
  Value tiledBOther; // tensor<kTile x N x bElem>, resplat(loadB.getOther()) | null
};

/// Pre-build the loop-invariant pieces for the per-tile B load.  Returns
/// nullopt if any required `resplatToShape` fails.
static std::optional<TensorOfPtrsBKAxis0Hoisted>
hoistTensorOfPtrsBKAxis0(PatternRewriter &b, Location loc, triton::LoadOp loadB,
                          const TensorOfPtrsBInfo &info, int64_t N,
                          int64_t kTile) {
  Type i32 = b.getI32Type();
  Type i1 = b.getI1Type();
  TensorOfPtrsBKAxis0Hoisted h;
  auto bPLoTy = cast<RankedTensorType>(info.bPLoK.getType());
  auto bPLoKTiledTy = RankedTensorType::get({kTile, 1}, bPLoTy.getElementType());
  h.tiledBSplat =
      b.create<triton::SplatOp>(loc, bPLoKTiledTy, info.bBaseScalar);
  auto bFullOffTy = RankedTensorType::get({kTile, N}, i32);
  h.tiledBOFull = b.create<triton::BroadcastOp>(loc, bFullOffTy, info.bOLo);
  if (info.bStrideSplat) {
    h.tiledStride = resplatToShape(info.bStrideSplat, {kTile, 1}, b, loc);
    if (!h.tiledStride)
      return std::nullopt;
  }
  if (loadB.getMask() && info.bOtherMaskLo) {
    auto bMaskTy = RankedTensorType::get({kTile, N}, i1);
    h.otherFull = b.create<triton::BroadcastOp>(loc, bMaskTy, info.bOtherMaskLo);
  }
  if (loadB.getMask() && info.bKMaskLo) {
    h.tiledRhs = resplatToShape(info.bKMaskRhs, {kTile, 1}, b, loc);
    if (!h.tiledRhs)
      return std::nullopt;
  }
  if (loadB.getOther()) {
    h.tiledBOther = resplatToShape(loadB.getOther(), {kTile, N}, b, loc);
    if (!h.tiledBOther)
      return std::nullopt;
  }
  return h;
}

/// Emit a per-tile load of B [kTile, N] given the hoisted invariants and
/// the loop's tiled K-range value `tiledKRange1D` (shape <kTile x i32>,
/// value = iv*kTile + [0..kTile)).
static Value emitTensorOfPtrsBKAxis0Tile(PatternRewriter &b, Location loc,
                                          triton::LoadOp loadB,
                                          const TensorOfPtrsBInfo &info,
                                          const TensorOfPtrsBKAxis0Hoisted &h,
                                          int64_t N, int64_t kTile,
                                          Value tiledKRange1D) {
  Type i32 = b.getI32Type();
  Type i1 = b.getI1Type();
  auto bPLoTy = cast<RankedTensorType>(info.bPLoK.getType());
  auto bKIdx2DTiledTy = RankedTensorType::get({kTile, 1}, i32);
  Value tiledBKIdx2D =
      b.create<triton::ExpandDimsOp>(loc, bKIdx2DTiledTy, tiledKRange1D,
                                     /*axis=*/1);
  Value tiledBRowOff = tiledBKIdx2D;
  if (h.tiledStride)
    tiledBRowOff = b.create<arith::MulIOp>(loc, tiledBKIdx2D, h.tiledStride);
  auto bPLoKTiledTy = RankedTensorType::get({kTile, 1}, bPLoTy.getElementType());
  Value tiledBPLoK = b.create<triton::AddPtrOp>(loc, bPLoKTiledTy, h.tiledBSplat,
                                                 tiledBRowOff);
  auto bFullPtrTy = RankedTensorType::get({kTile, N}, bPLoTy.getElementType());
  Value tiledBPFull =
      b.create<triton::BroadcastOp>(loc, bFullPtrTy, tiledBPLoK);
  Value tiledBPtr =
      b.create<triton::AddPtrOp>(loc, bFullPtrTy, tiledBPFull, h.tiledBOFull);

  Value tiledBMask;
  if (loadB.getMask()) {
    auto bMaskTy = RankedTensorType::get({kTile, N}, i1);
    Value kFull;
    if (h.tiledRhs) {
      Value tiledKMask2d = b.create<arith::CmpIOp>(loc, info.bKMaskPred,
                                                    tiledBKIdx2D, h.tiledRhs);
      kFull = b.create<triton::BroadcastOp>(loc, bMaskTy, tiledKMask2d);
    }
    if (h.otherFull && kFull)
      tiledBMask = b.create<arith::AndIOp>(loc, kFull, h.otherFull);
    else
      tiledBMask = h.otherFull ? h.otherFull : kFull;
  }
  Value tB;
  if (tiledBMask && h.tiledBOther)
    tB = b.create<triton::LoadOp>(loc, tiledBPtr, tiledBMask, h.tiledBOther,
                                   loadB.getCache(), loadB.getEvict(),
                                   loadB.getIsVolatile());
  else if (tiledBMask)
    tB = b.create<triton::LoadOp>(loc, tiledBPtr, tiledBMask, loadB.getCache(),
                                   loadB.getEvict(), loadB.getIsVolatile());
  else
    tB = b.create<triton::LoadOp>(loc, tiledBPtr, loadB.getCache(),
                                   loadB.getEvict(), loadB.getIsVolatile());
  return tB;
}

struct StageNonLoadOperandPattern : public OpRewritePattern<triton::DotOp> {
  /// Per-dot SHM budget in bytes; 0 disables the budget check (cost model
  /// still gates). Wired from the pass's `smem-budget-bytes` option.
  int64_t smemBudgetBytes;

  StageNonLoadOperandPattern(MLIRContext *ctx, int64_t smemBudgetBytes)
      : OpRewritePattern<triton::DotOp>(ctx), smemBudgetBytes(smemBudgetBytes) {
  }

  LogicalResult matchAndRewrite(triton::DotOp dot,
                                PatternRewriter &rewriter) const override {
    if (dot->hasAttr(kTiledAttr))
      return failure();

    auto aTy = cast<RankedTensorType>(dot.getA().getType());
    auto bTy = cast<RankedTensorType>(dot.getB().getType());
    auto dTy = cast<RankedTensorType>(dot.getResult().getType());
    int64_t M = dTy.getDimSize(0);
    int64_t N = dTy.getDimSize(1);
    int64_t K = aTy.getDimSize(1);
    if (bTy.getDimSize(0) != K)
      return failure();
    if (M * K * N <= kMKNBudget)
      return failure();

    auto aLoad = dot.getA().getDefiningOp<triton::LoadOp>();
    auto bLoad = dot.getB().getDefiningOp<triton::LoadOp>();
    // TileDotPattern owns the both-loads case.
    if (aLoad && bLoad)
      return failure();

    // Largest power-of-two divisor of K with per-tile M*kTile*N <= budget.
    int64_t maxKTile = kMKNBudget / std::max<int64_t>(1, M * N);
    int64_t kTile =
        static_cast<int64_t>(floorPow2(static_cast<uint64_t>(std::max<int64_t>(1, maxKTile))));
    kTile = std::min(kTile, K);
    while (kTile > 1 && K % kTile != 0)
      kTile /= 2;
    if (kTile <= 0 || kTile >= K) {
      LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> couldn't pick a kTile that "
                        << "divides K and stays in budget" << '\n');
      return failure();
    }

    // Load-side operands must be block-ptr style (chunking uses tt.advance).
    auto getBlockPtrSource =
        [](triton::LoadOp load) -> triton::MakeTensorPtrOp {
      if (!load)
        return nullptr;
      if (auto mk = load.getPtr().getDefiningOp<triton::MakeTensorPtrOp>())
        return mk;
      Value ptr = load.getPtr();
      while (true) {
        if (auto mk = ptr.getDefiningOp<triton::MakeTensorPtrOp>())
          return mk;
        if (auto adv = ptr.getDefiningOp<triton::AdvanceOp>()) {
          ptr = adv.getPtr();
          continue;
        }
        auto ba = dyn_cast<BlockArgument>(ptr);
        if (!ba)
          return nullptr;
        auto forOp = dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp());
        if (!forOp)
          return nullptr;
        unsigned idx = ba.getArgNumber() - forOp.getNumInductionVars();
        if (idx >= forOp.getInitArgs().size())
          return nullptr;
        ptr = forOp.getInitArgs()[idx];
      }
    };
    triton::MakeTensorPtrOp aMk = aLoad ? getBlockPtrSource(aLoad) : nullptr;
    triton::MakeTensorPtrOp bMk = bLoad ? getBlockPtrSource(bLoad) : nullptr;
    // Fallback: if the load isn't block-ptr style, try canonical
    // tensor-of-ptrs decomposition.  Currently supported: B-side K-axis-0
    // load (the V GMEM load in flash-attention-style PV dots).
    std::optional<TensorOfPtrsBInfo> bTopInfo;
    if (bLoad && !bMk) {
      bTopInfo = decomposeTensorOfPtrsBKAxis0(bLoad, K, N);
    }
    if ((aLoad && !aMk) || (bLoad && !bMk && !bTopInfo)) {
      LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> a load operand isn't block-ptr "
                        << "or canonical tensor-of-ptrs style; skipping" << '\n');
      return failure();
    }

    auto func = dot->getParentOfType<triton::FuncOp>();
    if (!func) {
      LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> dot is not inside a tt.func"
                        << '\n');
      return failure();
    }

    Location loc = dot.getLoc();
    Type aElemTy = aTy.getElementType();
    Type bElemTy = bTy.getElementType();
    int64_t numTiles = K / kTile;

    LLVM_DEBUG(DBGS() << "[StageNonLoadOperand] examining tt.dot [M="
                      << M << " K=" << K << " N=" << N << "] MKN=" << M * K * N
                      << " budget=" << kMKNBudget << "  kTile=" << kTile
                      << " numTiles=" << numTiles << " A=" << (aLoad ? "load" : "stage")
                      << " B=" << (bLoad ? "load" : "stage") << '\n');

    // ---- SHM-budget + cost-model gate (no IR mutation yet) --------------
    // Round-trip bytes ~= 2 * envelope (write + read), conservative.
    auto elemBytes = [](Type ty) -> unsigned {
      return static_cast<unsigned>((ty.getIntOrFloatBitWidth() + 7) / 8);
    };
    int64_t aEnvBytes = aLoad ? 0 : static_cast<unsigned>(M * K) * elemBytes(aElemTy);
    int64_t bEnvBytes = bLoad ? 0 : static_cast<unsigned>(N * K) * elemBytes(bElemTy);
    int64_t totalStageBytes = aEnvBytes + bEnvBytes;

    if (smemBudgetBytes > 0 && totalStageBytes > smemBudgetBytes) {
      LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> staging would need "
                        << totalStageBytes << " B; SRAM budget is " << smemBudgetBytes
                        << " B; skipping" << '\n');
      return failure();
    }

    // Conflict factor: build the post-swizzle 32-thread access pattern
    // `ttg.local_load` will issue, then feed it to the shared model.
    auto conflictFactor = [&](Type ty) -> unsigned {
      ::bishengir::triton::AscendSmemGeometry geom;
      unsigned eb = elemBytes(ty);
      unsigned bankCycle = geom.numBanks * geom.bankWidthBytes; // 128
      unsigned vec = std::max<unsigned>(
          1u, bankCycle / std::max<unsigned>(1u, geom.numThreadsPerWarp * eb));
      vec = std::min<unsigned>(vec, static_cast<unsigned>(kTile));
      unsigned chunks = static_cast<unsigned>(std::max<int64_t>(1, kTile / vec));
      unsigned cap = std::min<unsigned>(geom.numBanks, chunks);
      unsigned maxPhase = 1u;
      while ((maxPhase << 1) <= cap)
        maxPhase <<= 1;

      // Lane i: row i, col-chunk 0; XOR-swizzled to physical col
      // `(i % maxPhase) * vec` elements; each lane reads vec*eb bytes.
      SmallVector<::bishengir::triton::ThreadAccess> accesses(
          geom.numThreadsPerWarp);
      uint64_t rowStrideBytes = static_cast<uint64_t>(kTile) * eb;
      for (unsigned i = 0; i < geom.numThreadsPerWarp; ++i) {
        uint64_t physColEls = static_cast<uint64_t>(i % maxPhase) * vec;
        accesses[i].byteOffset = static_cast<uint64_t>(i) * rowStrideBytes + physColEls * eb;
        accesses[i].accessBytes = vec * eb;
      }
      return ::bishengir::triton::analyzeWarpAccessCycle(accesses, geom)
          .conflictFactor;
    };
    unsigned cfA = aLoad ? 1u : conflictFactor(aElemTy);
    unsigned cfB = bLoad ? 1u : conflictFactor(bElemTy);
    unsigned cfMax = std::max(cfA, cfB);

    // Spill-element estimate when NOT staging: MKN excess / warp size.
    int64_t excess = M * K * N - kMKNBudget;
    unsigned spillEst = excess > 0 ? static_cast<unsigned>((excess + 31) / 32) : 0;

    ::bishengir::triton::StagingDecisionInputs decInputs;
    decInputs.spillElementsIfNoStaging = spillEst;
    decInputs.roundTripBytes = static_cast<uint64_t>(2 * totalStageBytes);
    decInputs.conflictFactor = cfMax;
    decInputs.cyclesPerSpillElement =
        ::bishengir::triton::kStageSpillCyclesPerElement;
    auto decision = ::bishengir::triton::decideStaging(decInputs);
    LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> cost model: spill_elems="
                      << spillEst << " smem_bytes=" << totalStageBytes << " conflict="
                      << cfMax << " direct=" << static_cast<int64_t>(decision.directCostCycles)
                      << "c staged=" << static_cast<int64_t>(decision.stagedCostCycles) << "c"
                      << '\n');
    if (!decision.stageThroughSmem) {
      LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> cost model says staging is not "
                        << "profitable; skipping" << '\n');
      return failure();
    }

    
    // ---- Stage non-load operands via a single full-envelope store ------
    BlockArgument scratchArgA;
    if (!aLoad) {
      scratchArgA = appendScratchShmArg(func, aElemTy, loc);
      rewriter.setInsertionPoint(dot);
      // A:[M, K], K on innermost axis -> col-tile envelope [M, K].
      Value envPtrs = emitScratchShmAccessPtr(
          rewriter, loc, scratchArgA, /*dimOther=*/M, /*tileSize=*/K,
          /*envSize=*/K, /*kAxis=*/1, /*tileIdxI32=*/Value(),
          /*startConst=*/0, aElemTy);
      rewriter.create<triton::StoreOp>(
          loc, envPtrs, dot.getA(), /*mask=*/Value(),
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    }

    BlockArgument scratchArgB;
    if (!bLoad) {
      scratchArgB = appendScratchShmArg(func, bElemTy, loc);
      rewriter.setInsertionPoint(dot);
      // B is [K, N] with K on axis 0 (outermost).  row-tile (kAxis=0):
      // store envelope [K, N] (tileSize=K, startConst=0).
      Value envPtrs = emitScratchShmAccessPtr(
          rewriter, loc, scratchArgB, /*dimOther=*/N, /*tileSize=*/K,
          /*envSize=*/K, /*kAxis=*/0, /*tileIdxI32=*/Value(),
          /*startConst=*/0, bElemTy);
      rewriter.create<triton::StoreOp>(
          loc, envPtrs, dot.getB(), /*mask=*/Value(),
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    }

    // ---- Trace base of load-side operands (pre-loop) --------------------
    // tracePtrToBase folds enclosing scf.for tt.advance deltas into the
    // offsets so each outer iter sees its own K-window.
    rewriter.setInsertionPoint(dot);
    std::optional<TracedPtr> aChain, bChain;
    if (aLoad) {
      aChain = tracePtrToBase(aLoad.getPtr(), rewriter, loc);
      if (!aChain) {
        LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> couldn't trace A block-ptr "
                          << "to base; skipping" << '\n');
        return failure();
      }
    }
    if (bLoad && !bTopInfo) {
      bChain = tracePtrToBase(bLoad.getPtr(), rewriter, loc);
      if (!bChain) {
        LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> couldn't trace B block-ptr "
                          << "to base; skipping" << '\n');
        return failure();
      }
    }

    // ---- K-tile scf.for with DYNAMIC tile-idx ---------------------------
    // Single scf.for over [0, numTiles).  Inside the body, each operand
    // is loaded for the dynamic tile index `tI32`.  This serialises the
    // staged loads (only one tile's worth of bf16/elemTy data is live at
    // a time, vs. the C++-unrolled chain which kept all `numTiles` tile
    // results live simultaneously and spilled).
    Type i32 = rewriter.getI32Type();

    Value c0Idx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cNumTiles = rewriter.create<arith::ConstantIndexOp>(loc, numTiles);
    Value c1Idx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value accInit = dot.getC();

    // Hoist the static K-tile range AND every loop-invariant piece of the
    // tensor-of-ptrs per-tile B load outside the K-tile loop 
    Value hoistedKRangeBase;
    std::optional<TensorOfPtrsBKAxis0Hoisted> bHoisted;
    if (bTopInfo) {
      auto kRange1DTy = RankedTensorType::get({kTile}, i32);
      hoistedKRangeBase = rewriter.create<triton::MakeRangeOp>(
          loc, kRange1DTy, /*start=*/0, /*end=*/static_cast<int32_t>(kTile));
      bHoisted =
          hoistTensorOfPtrsBKAxis0(rewriter, loc, bLoad, *bTopInfo, N, kTile);
      if (!bHoisted) {
        LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> hoisting tensor-of-ptrs B "
                          << "invariants failed; bailing" << '\n');
        return failure();
      }
    }

    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0Idx, cNumTiles, c1Idx, ValueRange{accInit});

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());

    Value iv = forOp.getInductionVar();
    Value accIter = forOp.getRegionIterArgs()[0];
    Value tI32 = rewriter.create<arith::IndexCastOp>(loc, i32, iv);
    auto makeI32Const = [&](int32_t v) {
      return rewriter.create<arith::ConstantOp>(loc, i32,
                                                rewriter.getI32IntegerAttr(v));
    };
    Value kTileC = makeI32Const(static_cast<int32_t>(kTile));
    Value kBaseDyn = rewriter.create<arith::MulIOp>(loc, tI32, kTileC);

    // Per-tile block-ptr builder for the load-side operand: fresh
    // tt.make_tensor_ptr per iter at K-offset `iv * kTile`.
    auto buildPerTilePtr = [&](TracedPtr &chain, int kAxis, int64_t outShape0,
                               int64_t outShape1, Value kBaseDyn) -> Value {
      auto mk = chain.rootMk;
      SmallVector<Value> offs(chain.effectiveOffsets);
      offs[kAxis] = isZeroConstantI32(offs[kAxis])
                        ? kBaseDyn
                        : static_cast<Value>(rewriter.create<arith::AddIOp>(
                              loc, offs[kAxis], kBaseDyn));
      SmallVector<int32_t> shape{static_cast<int32_t>(outShape0), static_cast<int32_t>(outShape1)};
      SmallVector<int32_t> orderVec(mk.getOrder().begin(), mk.getOrder().end());
      return rewriter.create<triton::MakeTensorPtrOp>(
          loc, mk.getBase(), mk.getShape(), mk.getStrides(), offs, shape,
          orderVec);
    };

    auto emitStagedTile = [&](BlockArgument scratchArg, int64_t dimOther,
                              int kAxis, Type elemTy) -> Value {
      Value ptrs = emitScratchShmAccessPtr(
          rewriter, loc, scratchArg, dimOther, /*tileSize=*/kTile,
          /*envSize=*/K, kAxis, /*tileIdxI32=*/tI32,
          /*startConst=*/0, elemTy);
      return rewriter.create<triton::LoadOp>(
          loc, ptrs, /*mask=*/Value(), /*other=*/Value(),
          /*boundaryCheck=*/ArrayRef<int32_t>{},
          /*padding=*/std::optional<triton::PaddingOption>(),
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
          /*isVolatile=*/false);
    };

    Value aTile, bTile;
    if (aLoad) {
      Value aPtr = buildPerTilePtr(*aChain, /*kAxis=*/1, M, kTile, kBaseDyn);
      aTile = rewriter.create<triton::LoadOp>(
          loc, aPtr, aLoad.getBoundaryCheck(), aLoad.getPadding(),
          aLoad.getCache(), aLoad.getEvict(), aLoad.getIsVolatile());
    } else {
      aTile = emitStagedTile(scratchArgA, /*dimOther=*/M, /*kAxis=*/1, aElemTy);
    }
    if (bLoad) {
      if (bTopInfo) {
        // Tensor-of-ptrs path: rebuild the K-variant ptr column at the
        // per-tile K range and re-broadcast.  tiledKRange1D below has shape
        // <kTile x i32> and value [iv*kTile, iv*kTile + kTile).  The static
        // make_range is hoisted outside the loop (see above).
        auto kRange1DTy = RankedTensorType::get({kTile}, i32);
        Value kBaseSplat =
            rewriter.create<triton::SplatOp>(loc, kRange1DTy, kBaseDyn);
        Value tiledKRange1D = rewriter.create<arith::AddIOp>(loc, kBaseSplat,
                                                              hoistedKRangeBase);
        bTile = emitTensorOfPtrsBKAxis0Tile(rewriter, loc, bLoad, *bTopInfo,
                                             *bHoisted, N, kTile, tiledKRange1D);
        if (!bTile) {
          LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> tensor-of-ptrs per-tile B "
                            << "load emit failed; bailing" << '\n');
          return failure();
        }
      } else {
        Value bPtr = buildPerTilePtr(*bChain, /*kAxis=*/0, kTile, N, kBaseDyn);
        bTile = rewriter.create<triton::LoadOp>(
            loc, bPtr, bLoad.getBoundaryCheck(), bLoad.getPadding(),
            bLoad.getCache(), bLoad.getEvict(), bLoad.getIsVolatile());
      }
    } else {
      bTile = emitStagedTile(scratchArgB, /*dimOther=*/N, /*kAxis=*/0, bElemTy);
    }

    auto innerDot = rewriter.create<triton::DotOp>(
        loc, dTy, aTile, bTile, accIter, dot.getInputPrecision(),
        dot.getMaxNumImpreciseAcc());
    innerDot->setAttr(kTiledAttr, rewriter.getUnitAttr());

    rewriter.create<scf::YieldOp>(loc, ValueRange{innerDot.getResult()});

    rewriter.replaceOp(dot, forOp.getResult(0));
    LLVM_DEBUG(DBGS() << "[StageNonLoadOperand]   -> staged via scratch_shm + scf.for "
                      << "K-tile (numTiles=" << numTiles << ", kTile=" << kTile << ")"
                      << '\n');
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AlignCausalLoopStartPattern — skip over entirely-masked iterations of an
// FA-style causal loop. When the body has a causal mask
// `cmpi sle, bcast(expand_dims(addi(splat(%P), mr))), <iv-derived>` with %P
// loop-invariant, every iter with columns `< %P` is fully masked out.
//
// Rewrite: lb = (%P / step) * step, and pre-bake the per-iter advance
// (`advOffs[k] * (P / step)`) into each ptr iter_arg's init offset[k] so
// the body at the new lb matches the original at that iteration.
// Tagged `kCausalAlignedAttr` to prevent re-firing.
//===----------------------------------------------------------------------===//

static constexpr llvm::StringLiteral kCausalAlignedAttr =
    "bishengir.causal.aligned";

/// Return P from `splat(P)` on either side of `addi`, where P is defined
/// outside `body`; nullptr otherwise.
static Value findLoopInvariantSplatSrc(arith::AddIOp addi, const Block *body) {
  for (Value v : {addi.getLhs(), addi.getRhs()}) {
    auto splat = v.getDefiningOp<triton::SplatOp>();
    if (!splat)
      continue;
    Value src = splat.getSrc();
    if (Operation *def = src.getDefiningOp()) {
      if (def->getBlock() == body)
        continue;
      return src;
    }
    if (auto ba = dyn_cast<BlockArgument>(src)) {
      if (ba.getOwner() == body)
        continue;
      return src;
    }
  }
  return nullptr;
}

struct AlignCausalLoopStartPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOp->hasAttr(kCausalAlignedAttr))
      return failure();

    // Loop bounds: lb == 0, step > 0.
    auto lbCst = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
    if (!lbCst)
      return failure();
    auto lbAttr = dyn_cast<IntegerAttr>(lbCst.getValue());
    if (!lbAttr || lbAttr.getInt() != 0)
      return failure();
    auto stepCst = forOp.getStep().getDefiningOp<arith::ConstantOp>();
    if (!stepCst)
      return failure();
    auto stepAttr = dyn_cast<IntegerAttr>(stepCst.getValue());
    if (!stepAttr || stepAttr.getInt() <= 0)
      return failure();
    Type ivType = forOp.getInductionVar().getType();
    if (!isa<IntegerType>(ivType))
      return failure();

    // Find causal mask: cmpi sle, BCAST(EXPAND(ADDI(SPLAT(%P), MR))), ?
    Value rowStart;
    Block *body = forOp.getBody();
    for (Operation &op : body->without_terminator()) {
      auto cmp = dyn_cast<arith::CmpIOp>(&op);
      if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::sle)
        continue;
      auto bcast = cmp.getLhs().getDefiningOp<triton::BroadcastOp>();
      if (!bcast)
        continue;
      auto expand = bcast.getSrc().getDefiningOp<triton::ExpandDimsOp>();
      if (!expand)
        continue;
      auto addi = expand.getSrc().getDefiningOp<arith::AddIOp>();
      if (!addi)
        continue;
      Value found = findLoopInvariantSplatSrc(addi, body);
      if (found) {
        rowStart = found;
        break;
      }
    }
    if (!rowStart) {
      return failure();
    }
    if (rowStart.getType() != ivType)
      return failure(); // must match for divsi/muli

    Location loc = forOp.getLoc();

    // Aligned start = (P / step) * step, with div = P / step.
    rewriter.setInsertionPoint(forOp);
    Value div = rewriter.create<arith::DivSIOp>(loc, rowStart, forOp.getStep());
    Value aligned = rewriter.create<arith::MulIOp>(loc, div, forOp.getStep());

    // For each ptr iter_arg: bake `advOffs[k] * div` into its init offset[k].
    SmallVector<Value> newInits(forOp.getInitArgs().begin(),
                                forOp.getInitArgs().end());
    bool anyChange = false;
    auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

    for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
      Value init = forOp.getInitArgs()[i];
      if (!isa<triton::PointerType>(init.getType()))
        continue;
      BlockArgument iterArg = forOp.getRegionIterArgs()[i];

      // The yielded value into slot i must be a tt.advance of iterArg.
      Value yielded = yieldOp.getOperand(i);
      auto adv = yielded.getDefiningOp<triton::AdvanceOp>();
      if (!adv || adv.getPtr() != iterArg)
        continue;

      // Init must be a make_tensor_ptr we can rewrite.
      auto mk = init.getDefiningOp<triton::MakeTensorPtrOp>();
      if (!mk)
        continue;

      OperandRange origOffsets = mk.getOffsets();
      OperandRange advOffs = adv.getOffsets();
      if (origOffsets.size() != advOffs.size())
        continue;

      // Build new offsets per axis.
      SmallVector<Value> newOffsets;
      for (auto [origO, advO] : llvm::zip(origOffsets, advOffs)) {
        // Skip if the per-iter advance on this axis is constant 0.
        if (auto c = advO.getDefiningOp<arith::ConstantOp>()) {
          if (auto ai = dyn_cast<IntegerAttr>(c.getValue())) {
            if (ai.getInt() == 0) {
              newOffsets.push_back(origO);
              continue;
            }
          }
        }
        Value scaled = rewriter.create<arith::MulIOp>(loc, advO, div);
        // Skip the addi when origO is 0.
        if (auto c = origO.getDefiningOp<arith::ConstantOp>()) {
          if (auto ai = dyn_cast<IntegerAttr>(c.getValue())) {
            if (ai.getInt() == 0) {
              newOffsets.push_back(scaled);
              continue;
            }
          }
        }
        newOffsets.push_back(
            rewriter.create<arith::AddIOp>(loc, origO, scaled));
      }

      // Reuse the original mk's pointee tensor shape.
      SmallVector<int32_t> tileShape;
      {
        auto pt = cast<triton::PointerType>(mk.getType());
        auto inner = cast<RankedTensorType>(pt.getPointeeType());
        for (int64_t d : inner.getShape())
          tileShape.push_back(static_cast<int32_t>(d));
      }
      Value newMk = rewriter.create<triton::MakeTensorPtrOp>(
          loc, mk.getBase(), mk.getShape(), mk.getStrides(), newOffsets,
          tileShape,
          SmallVector<int32_t>(mk.getOrder().begin(), mk.getOrder().end()));
      newInits[i] = newMk;
      anyChange = true;
    }

    if (!anyChange) {
      // No ptr iter_args to rewrite -> not an FA-style loop; bail.
      return failure();
    }

    // Build a new scf.for with the aligned lb and rebuilt inits.
    auto newFor = rewriter.create<scf::ForOp>(
        loc, aligned, forOp.getUpperBound(), forOp.getStep(), newInits);
    newFor->setAttrs(forOp->getAttrs());
    newFor->setAttr(kCausalAlignedAttr, rewriter.getUnitAttr());

    // Move the body block over (new for has matching block args by ctor).
    if (newFor.getBody()->mightHaveTerminator())
      newFor.getBody()->getTerminator()->erase();
    rewriter.mergeBlocks(forOp.getBody(), newFor.getBody(),
                         newFor.getBody()->getArguments());

    rewriter.replaceOp(forOp, newFor.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TileDotLoadsPass : public impl::TileDotLoadsBase<TileDotLoadsPass> {
  using impl::TileDotLoadsBase<TileDotLoadsPass>::TileDotLoadsBase;
  void runOnOperation() override {
    auto fn = getOperation();
    
    static constexpr llvm::StringLiteral kSharedMemDynamicSizeAttr =
        "bishengir.shared-mem-dynamic-size";

    // Preserve the original launch-time shared memory size before later
    // passes overwrite "ttg.shared" with actual kernel allocation usage.
    if (auto module = fn->getParentOfType<ModuleOp>()) {
      auto b = Builder(&getContext());
      module->setAttr(kSharedMemDynamicSizeAttr,
                      b.getI32IntegerAttr(this->smemBudgetBytes));
    }
                      
    LLVM_DEBUG(DBGS() << "=== TileDotLoads running on function: " << fn.getName()
                      << " ===" << '\n');

    // K-tile the now-zero-accumulator dots.
    {
      RewritePatternSet p(&getContext());
      p.add<TileDotPattern>(&getContext());
      (void)applyPatternsGreedily(getOperation(), std::move(p));
    }

    // Skip entirely-masked iterations of FA-style causal loops.
    {
      RewritePatternSet p(&getContext());
      p.add<AlignCausalLoopStartPattern>(&getContext());
      (void)applyPatternsGreedily(getOperation(), std::move(p));
    }

    // Stage non-load operands of over-budget dots through scratch SHM.
    {
      RewritePatternSet p(&getContext());
      p.add<StageNonLoadOperandPattern>(&getContext(), this->smemBudgetBytes);
      (void)applyPatternsGreedily(getOperation(), std::move(p));
    }

    LLVM_DEBUG(DBGS() << "=== TileDotLoads done ===" << '\n');
  }
};

} // namespace


std::unique_ptr<mlir::Pass>
createTileDotLoadsPass(const TileDotLoadsOptions &options) {
  return std::make_unique<TileDotLoadsPass>(options);
}

} // namespace triton
} // namespace bishengir
//===- OptimizeRowLoads.cpp - Optimize scalar row loads
//-----------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_OPTIMIZEROWLOADS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

namespace mlir::hivm {
namespace {

// Recognize each current AIV row loop once, optionally plan its row mask, then
// merge adjacent original iterations. All matching precedes IR construction;
// neither ownership nor loop bounds are supplied by another pass.
constexpr unsigned kMaxRecursionDepth = 16;
constexpr int64_t kMaxBurstBytes = (int64_t(1) << 21) - 32;
constexpr int64_t kDmaAlignmentBytes = 32;

struct RowLoopCore {
  scf::ForOp loop;
  hivm::LoadOp copy;  // the per-row load
  scf::IfOp rowGuard; // null when the load is unconditional
  SmallVector<tensor::ExtractOp> rowReads;
  memref::SubViewOp slice;          // destination row subview inside the loop
  memref::CollapseShapeOp collapse; // optional descriptor-only collapse
  Value dst;                        // logical destination tile (slice source)
  int64_t rows = 0;
  int64_t cols = 0;
  memref::ReinterpretCastOp source;
  // Exact structural nodes, including source/destination row views and
  // the optional fixed-address pointer descriptor and pure GM metadata.
  SmallVector<Operation *> setupViews;
};

static bool isDenseRowTile(MemRefType type) {
  if (type.getRank() != 2 || type.getDimSize(1) <= 0 ||
      (!type.isDynamicDim(0) && type.getDimSize(0) <= 0))
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(getStridesAndOffset(type, strides, offset)) &&
         strides[0] == type.getDimSize(1) && strides[1] == 1;
}
static Value stripMemRefCasts(Value value, RowLoopCore &core) {
  while (auto cast = value.getDefiningOp<memref::CastOp>()) {
    if (cast->getBlock() == core.loop.getBody())
      core.setupViews.push_back(cast);
    else if (!core.loop.isDefinedOutsideOfLoop(value))
      break;
    value = cast.getSource();
  }
  return value;
}

// The gathered row must come from external (GM) memory. An explicit GM space is
// enough; a missing space is the frontend convention for GM only when the base
// is a function entry argument. A space-less allocation is a local buffer, so a
// row copy from it is not a GM transfer and must not be classified as one.
// Classify the normalized base once. Type evidence and metadata must agree;
// only this exact fixed-address descriptor and its pure GM marks may replay.
static bool matchGMRowBase(RowLoopCore &core) {
  Value base = stripMemRefCasts(core.source.getSource(), core);
  // reinterpret_cast addresses the underlying storage, independently of the
  // source descriptor's rank, shape and strides.
  auto type = dyn_cast<MemRefType>(base.getType());
  if (!type)
    return false;
  auto gm = AddressSpaceAttr::get(base.getContext(), AddressSpace::GM);
  Attribute space = type.getMemorySpace();
  if (space && space != gm)
    return false;
  if (auto argument = dyn_cast<BlockArgument>(base))
    return argument.getOwner()->isEntryBlock() &&
           isa<func::FuncOp>(argument.getOwner()->getParentOp());
  auto pointer = base.getDefiningOp<PointerCastOp>();
  if (!pointer || pointer.getAddrs().size() != 1 ||
      !core.loop.isDefinedOutsideOfLoop(pointer.getAddrs().front()))
    return false;
  bool local = pointer->getBlock() == core.loop.getBody();
  if (!local && !core.loop.isDefinedOutsideOfLoop(base))
    return false;
  bool markedGM = false;
  for (Operation *user : base.getUsers()) {
    auto mark = dyn_cast<annotation::MarkOp>(user);
    if (!mark)
      continue;
    if (mark.getAttrNum() != 1 || mark->getNumOperands() != 1 ||
        mark->getAttr(AddressSpaceAttr::getMnemonic()) != gm ||
        !isMemoryEffectFree(mark))
      return false;
    bool localMark = mark->getBlock() == core.loop.getBody();
    if (localMark != local || (!localMark && core.loop->isAncestor(mark)))
      return false;
    markedGM = true;
    if (localMark)
      core.setupViews.push_back(mark);
  }
  if (!space && !markedGM)
    return false;
  if (local)
    core.setupViews.push_back(pointer);
  return true;
}

static FailureOr<RowLoopCore> recoverRowLoopCore(scf::ForOp loop,
                                                 StringRef &reason) {
  // Body: exactly one row load, optionally wrapped in the row guard,
  // plus scalar row reads. Other regions are not replayable; the matcher
  // validates the remaining operations for its own transformation.
  RowLoopCore core;
  core.loop = loop;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (isOpTriviallyDead(&op))
      continue;
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (core.copy) {
        reason = "load: expected exactly one direct row load";
        return failure();
      }
      core.copy = load;
      continue;
    }
    if (auto guard = dyn_cast<scf::IfOp>(op); guard) {
      if (core.copy || guard.getNumResults() || !guard.elseBlock()) {
        reason = "load: invalid row guard";
        return failure();
      }
      auto &thenOps = guard.thenBlock()->getOperations();
      auto &elseOps = guard.elseBlock()->getOperations();
      if (thenOps.size() != 2 || elseOps.size() != 2) {
        reason = "load: invalid row branches";
        return failure();
      }
      core.copy = dyn_cast<LoadOp>(thenOps.front());
      auto fill = dyn_cast<VBrcOp>(elseOps.front());
      if (!core.copy || !fill || fill.getDst() != core.copy.getDst() ||
          isa<ShapedType>(fill.getSrc().getType())) {
        reason = "load: invalid row fallback";
        return failure();
      }
      core.rowGuard = guard;
      continue;
    }
    if (auto read = dyn_cast<tensor::ExtractOp>(op)) {
      core.rowReads.push_back(read);
      continue;
    }
    if (op.getNumRegions()) {
      reason = "extra users: unsupported region in row loop";
      return failure();
    }
  }
  if (!core.copy) {
    reason = "load: missing row load";
    return failure();
  }
  // Destination view: the load writes one loop-local row subview of an
  // invariant tile, optionally through a descriptor-only collapse. The row
  // target may pass through a descriptor-only collapse from subblock binding.
  Value rowTarget = stripMemRefCasts(core.copy.getDst(), core);
  core.collapse = rowTarget.getDefiningOp<memref::CollapseShapeOp>();
  if (core.collapse)
    rowTarget = stripMemRefCasts(core.collapse.getSrc(), core);
  core.slice = rowTarget.getDefiningOp<memref::SubViewOp>();
  if (!core.slice || core.slice->getBlock() != loop.getBody() ||
      (core.collapse && core.collapse->getBlock() != loop.getBody())) {
    reason = "view: expected a loop-local destination row subview";
    return failure();
  }
  core.dst = core.slice.getSource();
  if (!loop.isDefinedOutsideOfLoop(core.dst)) {
    reason = "view: expected an invariant destination tile";
    return failure();
  }
  core.setupViews.push_back(core.slice);
  if (core.collapse)
    core.setupViews.push_back(core.collapse);
  core.source = stripMemRefCasts(core.copy.getSrc(), core)
                    .getDefiningOp<memref::ReinterpretCastOp>();
  auto dstType = cast<MemRefType>(core.dst.getType());
  auto ub = AddressSpaceAttr::get(loop.getContext(), AddressSpace::UB);
  if ((dstType.getMemorySpace() && dstType.getMemorySpace() != ub) ||
      dstType.getRank() != 2 || dstType.getDimSize(1) <= 0 ||
      (!dstType.isDynamicDim(0) && dstType.getDimSize(0) <= 0)) {
    reason = "destination: requires an invariant row tile";
    return failure();
  }
  core.rows = dstType.getDimSize(0);
  core.cols = dstType.getDimSize(1);
  return core;
}

static bool hasInlineLayout(Operation *op) {
  for (StringRef name : {"transpose", "broadcast", "broadcast_axes"})
    if (auto layout = op->getAttrOfType<DenseI64ArrayAttr>(name);
        layout && !layout.empty())
      return true;
  return false;
}

static bool isPlainRowLoad(LoadOp load) {
  if (hasInlineLayout(load) || load->getNumResults() || load.getPadModeAttr() ||
      load.getInitOutBuffer() || load.getInitCondition() ||
      load.getMayImplicitTransposeWithLastAxis())
    return false;
  auto isAbsentOrZero = [](Value value) {
    return !value || getConstantIntValue(value) == 0;
  };
  return isAbsentOrZero(load.getLeftPaddingNum()) &&
         isAbsentOrZero(load.getRightPaddingNum());
}

// Prove descriptor geometry in the current loop domain, not a DMA strategy.
// Pairing needs consecutive destination rows; removing a tile-wide select also
// needs complete coverage. Arith min/max from subblock clipping have no bounds
// models in the pinned MLIR, so add only those constraints locally.
class RowGeometryBounds : public ValueBoundsConstraintSet {
  struct MinMax {
    Value result, lhs, rhs;
    bool isMin, resolved = false;
  };
  SmallVector<MinMax> minMax;

  bool collect(Value value, std::optional<int64_t> dim) {
    if (dim)
      return false;
    Operation *op = value.getDefiningOp();
    if (!op || !isa<arith::MinSIOp, arith::MaxSIOp>(op))
      return false;
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    bool isMin = isa<arith::MinSIOp>(op);
    // Enqueue both operands before processing any further constraints.
    AffineExpr a = getExpr(lhs);
    AffineExpr b = getExpr(rhs);
    if (isMin) {
      bound(value) <= a;
      bound(value) <= b;
    } else {
      bound(value) >= a;
      bound(value) >= b;
    }
    minMax.push_back({value, lhs, rhs, isMin});
    return true;
  }

  void saturate() {
    processWorklist();
    bool changed;
    do {
      changed = false;
      // All min/max operands were populated by the worklist. Compare column
      // positions directly: do not call static compare(), which starts over.
      for (MinMax &op : minMax) {
        if (op.resolved)
          continue;
        Value chosen;
        if (comparePos(getPos(op.lhs), LE, getPos(op.rhs)))
          chosen = op.isMin ? op.lhs : op.rhs;
        else if (comparePos(getPos(op.lhs), GE, getPos(op.rhs)))
          chosen = op.isMin ? op.rhs : op.lhs;
        if (!chosen)
          continue;
        bound(op.result) == getExpr(chosen);
        op.resolved = true;
        changed = true;
      }
    } while (changed);
  }

public:
  explicit RowGeometryBounds(MLIRContext *ctx)
      : ValueBoundsConstraintSet(ctx, [this](Value v, std::optional<int64_t> d,
                                             ValueBoundsConstraintSet &) {
          return collect(v, d);
        }) {}

  static Variable difference(const Variable &lhs, const Variable &rhs) {
    MLIRContext *ctx = lhs.getContext();
    auto map = AffineMap::get(
        2, 0, getAffineDimExpr(0, ctx) - getAffineDimExpr(1, ctx));
    return Variable(map, SmallVector<Variable>{lhs, rhs});
  }

  static bool proveEqual(const Variable &lhs, const Variable &rhs) {
    RowGeometryBounds proof(lhs.getContext());
    // Append query columns as symbols: inserting a second dimension would
    // shift the first query's operand columns without updating the worklist.
    int64_t l = proof.insert(lhs, /*isSymbol=*/true);
    int64_t r = proof.insert(rhs, /*isSymbol=*/true);
    proof.saturate();
    return proof.comparePos(l, EQ, r);
  }

  static bool proveEqual(OpFoldResult lhs, OpFoldResult rhs) {
    if (lhs == rhs)
      return true;
    auto lhsConstant = getConstantIntValue(lhs);
    auto rhsConstant = getConstantIntValue(rhs);
    if (lhsConstant && rhsConstant)
      return *lhsConstant == *rhsConstant;
    return proveEqual(Variable(lhs), Variable(rhs));
  }

  static bool proveEqual(OpFoldResult lhs, int64_t rhs) {
    if (auto constant = getConstantIntValue(lhs))
      return *constant == rhs;
    Builder builder(lhs.get<Value>().getContext());
    return proveEqual(lhs, builder.getIndexAttr(rhs));
  }

  static std::optional<int64_t> constantDifference(Value lhs, Value rhs) {
    if (lhs == rhs)
      return 0;
    auto lhsConstant = getConstantIntValue(lhs);
    auto rhsConstant = getConstantIntValue(rhs);
    if (lhsConstant && rhsConstant) {
      int64_t delta;
      if (llvm::SubOverflow(*lhsConstant, *rhsConstant, delta))
        return std::nullopt;
      return delta;
    }
    RowGeometryBounds proof(lhs.getContext());
    int64_t pos = proof.insert(difference(lhs, rhs), /*isSymbol=*/false);
    proof.saturate();
    auto delta = proof.cstr.getConstantBound(presburger::BoundType::EQ, pos);
    if (!delta || *delta < std::numeric_limits<int64_t>::min() ||
        *delta > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    return static_cast<int64_t>(*delta);
  }

  static bool proveUnitRowOffset(OpFoldResult offset, scf::ForOp loop) {
    if (offset == OpFoldResult(loop.getInductionVar()))
      return true;
    RowGeometryBounds proof(loop.getContext());
    int64_t pos = proof.insert(
        difference(Variable(offset), Variable(loop.getInductionVar())),
        /*isSymbol=*/false);
    assert(pos == 0 && "query must be first column");
    proof.saturate();
    // Preserve only loop-invariant SSA variables. The query is anonymous.
    proof.projectOut(
        [&](ValueDim p) { return !loop.isDefinedOutsideOfLoop(p.first); });
    proof.projectOutAnonymous(/*except=*/pos);
    SmallVector<AffineMap> lower(1), upper(1);
    proof.cstr.getSliceBounds(pos, 1, loop.getContext(), &lower, &upper,
                              /*closedUB=*/true);
    return lower[0] && lower[0].getNumResults() == 1 && lower[0] == upper[0];
  }
};

// Local row-mask analysis and emission. Requires RowLoopCore and
// kMaxRecursionDepth.

struct RowSelection {
  Value result;
  Value replacement;
  Value mask;
  Value fallback;
  TypedAttr fallbackConstant;
  Operation *select = nullptr;
  SmallVector<VCastOp> prefix;
  bool inverted = false;
};

// Keep the boolean represented by broadcast(select(mask, +/-1, +0)) != 0
// rather than extracting the legalized, two-dimensional vector mask.
static bool isPositiveZero(Value value) {
  if (matchPattern(value, m_Zero()) || matchPattern(value, m_PosZeroFloat()))
    return true;
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    return isPositiveZero(fill.getInputs().front());
  if (auto broadcast = value.getDefiningOp<VBrcOp>())
    return isPositiveZero(broadcast.getSrc());
  return false;
}

static bool isMaskTrue(Value value) {
  if (matchPattern(value, m_One()))
    return true;
  APFloat constant(0.0);
  if (matchPattern(value, m_ConstantFloat(&constant))) {
    constant.clearSign();
    return constant == APFloat(constant.getSemantics(), 1);
  }
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    return isMaskTrue(fill.getInputs().front());
  return false;
}

// True when a reshape or cast only adds or removes unit dimensions while
// keeping the single non-unit dimension in place. A row mask must keep
// indexing the outermost axis, so a view that moves the non-unit dimension
// (such as {rows} -> {1, rows}) is not transparent.
static bool isUnitRowView(ShapedType src, ShapedType dst) {
  if (!src || !dst || !src.hasStaticShape() || !dst.hasStaticShape() ||
      src.getNumElements() != dst.getNumElements())
    return false;
  auto nonUnitDim = [](ShapedType type) -> std::pair<int64_t, int64_t> {
    int64_t count = 0;
    int64_t position = -1;
    for (auto [index, dim] : llvm::enumerate(type.getShape()))
      if (dim != 1) {
        ++count;
        position = static_cast<int64_t>(index);
      }
    return {count, position};
  };
  auto [srcCount, srcPosition] = nonUnitDim(src);
  auto [dstCount, dstPosition] = nonUnitDim(dst);
  return srcCount <= 1 && dstCount <= 1 && srcPosition == dstPosition;
}

// Strip shape-only views that add or remove unit dimensions, so the row mask
// is recognized without a separate shape check per view op.
static Value stripUnitViews(Value value) {
  for (unsigned depth = 0; depth < kMaxRecursionDepth; ++depth) {
    Operation *def = value.getDefiningOp();
    if (auto expand = dyn_cast_or_null<tensor::ExpandShapeOp>(def)) {
      if (!isUnitRowView(expand.getSrcType(), expand.getResultType()))
        break;
      value = expand.getSrc();
    } else if (auto collapse = dyn_cast_or_null<tensor::CollapseShapeOp>(def)) {
      if (!isUnitRowView(collapse.getSrcType(), collapse.getResultType()))
        break;
      value = collapse.getSrc();
    } else if (auto cast = dyn_cast_or_null<tensor::CastOp>(def)) {
      auto src = dyn_cast<ShapedType>(cast.getSource().getType());
      auto dst = dyn_cast<ShapedType>(cast.getResult().getType());
      if (!isUnitRowView(src, dst))
        break;
      value = cast.getSource();
    } else {
      break;
    }
  }
  return value;
}

// Match a broadcast of a row vector across axis 1 of a `{rows, *}` tile. The
// source must be `{rows}`, or `{rows, 1}` when `allowUnitDim` because
// hivm.brc also accepts the explicit unit dimension.
static Value getRowBroadcastSource(Value value, Value source, int64_t rows,
                                   ArrayRef<int64_t> broadcastDims,
                                   bool allowUnitDim) {
  auto sourceType = dyn_cast<ShapedType>(source.getType());
  auto target = dyn_cast<RankedTensorType>(value.getType());
  if (!sourceType || !sourceType.hasRank() || sourceType.getRank() < 1 ||
      !target || target.getRank() != 2 || target.getDimSize(0) != rows ||
      broadcastDims != ArrayRef<int64_t>{1} ||
      sourceType.getDimSize(0) != rows ||
      (sourceType.getRank() != 1 &&
       (!allowUnitDim || sourceType.getRank() != 2 ||
        sourceType.getDimSize(1) != 1)))
    return {};
  return source;
}

static Value getRowMask(Value value, int64_t rows, unsigned depth = 0) {
  if (depth == kMaxRecursionDepth)
    return {};
  value = stripUnitViews(value);
  if (auto broadcast = value.getDefiningOp<linalg::BroadcastOp>()) {
    Value source = getRowBroadcastSource(value, broadcast.getInput(), rows,
                                         broadcast.getDimensions(),
                                         /*allowUnitDim=*/false);
    return source ? getRowMask(source, rows, depth + 1) : Value();
  }
  if (auto broadcast = value.getDefiningOp<VBrcOp>()) {
    Value source = getRowBroadcastSource(value, broadcast.getSrc(), rows,
                                         broadcast.getBroadcastDims(),
                                         /*allowUnitDim=*/true);
    return source ? getRowMask(source, rows, depth + 1) : Value();
  }
  if (auto compare = value.getDefiningOp<VCmpOp>()) {
    if (!hasInlineLayout(compare) &&
        compare.getCompareMode() == CompareMode::NE &&
        isPositiveZero(compare.getDpsInputs()[1]))
      return getRowMask(compare.getDpsInputs()[0], rows, depth + 1);
  }
  if (auto select = value.getDefiningOp<VSelOp>())
    if (!hasInlineLayout(select) && isMaskTrue(select.getDpsInputs()[1]) &&
        isPositiveZero(select.getDpsInputs()[2]))
      return getRowMask(select.getDpsInputs()[0], rows, depth + 1);
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getShape() == ArrayRef<int64_t>{rows} &&
                 type.getElementType().isInteger(1)
             ? value
             : Value();
}

// Shape queries, annotations and overwritten transpose inits do not consume
// elements or constrain a mask rewrite. They cannot establish Vector ownership.
static bool isMetadataOnly(OpOperand &use) {
  Operation *op = use.getOwner();
  return isa<tensor::DimOp, tensor::RankOp>(op) ||
         (isa<annotation::MarkOp>(op) && isMemoryEffectFree(op)) ||
         (isa<linalg::TransposeOp, VTransposeOp>(op) &&
          use.getOperandNumber() != 0);
}

// Follow only descriptor/shape views when deciding whether an observation
// reads elements. Unknown operations, including unknown effects, are data uses.
static bool hasOnlyMetadataUses(Value value, unsigned depth = 0) {
  if (depth == kMaxRecursionDepth)
    return false;
  return llvm::all_of(value.getUses(), [&](OpOperand &use) {
    Operation *user = use.getOwner();
    if (isa<memref::DimOp, memref::RankOp>(user) ||
        (isa<annotation::MarkOp>(user) && isMemoryEffectFree(user)) ||
        (isa<TensorType>(value.getType()) && isMetadataOnly(use)))
      return true;
    if (!isa<memref::CastOp, memref::MemorySpaceCastOp, memref::SubViewOp,
             memref::CollapseShapeOp, tensor::CastOp, tensor::ExpandShapeOp,
             tensor::CollapseShapeOp, bufferization::ToTensorOp>(user) ||
        user->getNumResults() != 1 || use.getOperandNumber() != 0)
      return false;
    return hasOnlyMetadataUses(user->getResult(0), depth + 1);
  });
}

// The load may reach its row through casts and a collapse. Ignore only safe
// metadata branches; every other data use must be the next node on that path.
static bool hasPrivateRowTarget(RowLoopCore &core) {
  Value target = core.copy.getDst();
  Operation *next = core.copy;
  while (true) {
    if (target == core.copy.getSrc())
      return false;
    for (OpOperand &use : target.getUses()) {
      Operation *user = use.getOwner();
      if (user == next || isa<memref::DimOp, memref::RankOp>(user) ||
          (isa<annotation::MarkOp>(user) && isMemoryEffectFree(user)))
        continue;
      if (user->getNumResults() != 1 ||
          !isa<memref::CastOp, memref::MemorySpaceCastOp, memref::SubViewOp,
               memref::CollapseShapeOp, bufferization::ToTensorOp>(user) ||
          use.getOperandNumber() != 0 ||
          !hasOnlyMetadataUses(user->getResult(0)))
        return false;
    }
    if (target == core.slice.getResult())
      return true;
    next = target.getDefiningOp();
    if (auto cast = dyn_cast_or_null<memref::CastOp>(next))
      target = cast.getSource();
    else if (auto collapse = dyn_cast_or_null<memref::CollapseShapeOp>(next))
      target = collapse.getSrc();
    else
      return false;
  }
}

static Operation *getOnlyDataUser(Value value) {
  Operation *result = nullptr;
  for (OpOperand &use : value.getUses()) {
    if (isMetadataOnly(use) ||
        (isa<tensor::CastOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
             use.getOwner()) &&
         hasOnlyMetadataUses(use.getOwner()->getResult(0))))
      continue;
    Operation *user = use.getOwner();
    if (result && result != user)
      return nullptr;
    result = user;
  }
  return result;
}

// Numerical casts keep their full semantics and remain in the IR. Only their
// unchanged element indexing lets ownership analysis look through a mask chain.
static bool isRowSelectionCast(VCastOp conversion, Value source) {
  if (hasInlineLayout(conversion) || conversion->getNumResults() != 1 ||
      conversion.getSingleSrc() != source)
    return false;
  auto from = dyn_cast<RankedTensorType>(source.getType());
  auto to = dyn_cast<RankedTensorType>(conversion->getResult(0).getType());
  return from && to && !from.getEncoding() && !to.getEncoding() &&
         from.getShape() == to.getShape() &&
         from.getElementType().isIntOrFloat() &&
         to.getElementType().isIntOrFloat();
}

// Match just the select, independently of the data's other uses. Classification
// uses this structural fact even when the tile is not private enough to mutate.
static std::optional<RowSelection> matchRowSelection(Value data,
                                                     Operation *user) {
  auto type = dyn_cast<RankedTensorType>(data.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape())
    return std::nullopt;
  RowSelection selection;
  selection.replacement = data;
  // The select that applies the row mask; the gathered data may be either the
  // selected or the fallback operand.
  Value condition, selected, other;
  if (auto select = dyn_cast_or_null<VSelOp>(user)) {
    if (hasInlineLayout(select))
      return std::nullopt;
    condition = select.getDpsInputs()[0];
    selected = select.getDpsInputs()[1];
    other = select.getDpsInputs()[2];
  } else if (auto select = dyn_cast_or_null<arith::SelectOp>(user)) {
    condition = select.getCondition();
    selected = select.getTrueValue();
    other = select.getFalseValue();
  } else {
    return std::nullopt;
  }
  if (user->getNumResults() != 1 ||
      user->getResult(0).getType() != data.getType())
    return std::nullopt;
  if (selected != data) {
    if (other != data)
      return std::nullopt;
    std::swap(selected, other);
    selection.inverted = true;
  }
  // Recover the row mask through broadcasts and boolean wrappers.
  selection.mask = getRowMask(condition, type.getDimSize(0));
  if (!selection.mask)
    return std::nullopt;
  selection.select = user;
  selection.result = user->getResult(0);
  // Unwrap a broadcast or filled fallback down to its scalar constant.
  while (isa<ShapedType>(other.getType())) {
    if (auto fill = other.getDefiningOp<linalg::FillOp>())
      other = fill.getInputs().front();
    else if (auto broadcast = other.getDefiningOp<VBrcOp>())
      other = broadcast.getSrc();
    else {
      if (auto constant = other.getDefiningOp<arith::ConstantOp>())
        if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
            dense && dense.isSplat())
          selection.fallbackConstant =
              cast<TypedAttr>(dense.getSplatValue<Attribute>());
      // Keep the fallback value for the exact-preimage and dominance proofs;
      // an unsupported fallback leaves the select in place.
      break;
    }
  }
  selection.fallback = other;
  return selection;
}

static std::optional<RowSelection> getRowSelection(Value gathered) {
  SmallVector<VCastOp> prefix;
  Value data = gathered;
  for (unsigned depth = 0; depth < kMaxRecursionDepth; ++depth) {
    Operation *user = getOnlyDataUser(data);
    if (auto conversion = dyn_cast_or_null<VCastOp>(user)) {
      if (!isRowSelectionCast(conversion, data))
        return std::nullopt;
      prefix.push_back(conversion);
      data = conversion->getResult(0);
      continue;
    }
    auto selection = matchRowSelection(data, user);
    if (selection)
      selection->prefix = std::move(prefix);
    return selection;
  }
  return std::nullopt;
}

static arith::CmpIPredicate scalarPredicate(VCmpOp compare) {
  switch (compare.getCompareMode()) {
  case CompareMode::EQ:
    return arith::CmpIPredicate::eq;
  case CompareMode::NE:
    return arith::CmpIPredicate::ne;
  case CompareMode::LT:
    return compare.getIsSigned() ? arith::CmpIPredicate::slt
                                 : arith::CmpIPredicate::ult;
  case CompareMode::LE:
    return compare.getIsSigned() ? arith::CmpIPredicate::sle
                                 : arith::CmpIPredicate::ule;
  case CompareMode::GT:
    return compare.getIsSigned() ? arith::CmpIPredicate::sgt
                                 : arith::CmpIPredicate::ugt;
  case CompareMode::GE:
    return compare.getIsSigned() ? arith::CmpIPredicate::sge
                                 : arith::CmpIPredicate::uge;
  }
  llvm_unreachable("unknown comparison");
}

// One classification table drives both support analysis and scalar emission.
enum class ScalarRecipe { Add, Sub, Mul, And, Or, Xor, Not, Compare };
static std::optional<ScalarRecipe> getScalarRecipe(Operation *op) {
  if (isa<VAddOp>(op))
    return ScalarRecipe::Add;
  if (isa<VSubOp>(op))
    return ScalarRecipe::Sub;
  if (isa<VMulOp>(op))
    return ScalarRecipe::Mul;
  if (isa<VAndOp>(op))
    return ScalarRecipe::And;
  if (isa<VOrOp>(op))
    return ScalarRecipe::Or;
  if (isa<VXorOp>(op))
    return ScalarRecipe::Xor;
  if (isa<VNotOp>(op))
    return ScalarRecipe::Not;
  if (isa<VCmpOp>(op))
    return ScalarRecipe::Compare;
  return std::nullopt;
}

// Rebuild a masked expression as per-row scalar arithmetic. `supports()` and
// `emit()` are memoized dual recursions over the same DAG: the first proves the
// shape and dominance invariants, the second translates each operation. Both
// caches are owned by the instance, so callers never thread a visited set or a
// value mapping through the recursion.
class MaskScalarizer {
public:
  MaskScalarizer(scf::ForOp loop, int64_t rows, DominanceInfo &dominance,
                 DenseMap<Value, Value> rowReads)
      : loop(loop), rows(rows), dominance(dominance),
        emitted(std::move(rowReads)) {
    for (const auto &entry : emitted)
      rowValues.insert(entry.first);
  }

  bool supports(Value value) {
    auto cached = supported.find(value);
    if (cached != supported.end())
      return cached->second;
    // computeSupports() recurses and inserts into the same map; insert only
    // after it returns so no iterator is held across a possible rehash.
    bool result = computeSupports(value);
    supported[value] = result;
    return result;
  }

  Value emit(Value value, Value row, OpBuilder &builder) {
    if (!isa<ShapedType>(value.getType()))
      return value;
    if (Value cached = emitted.lookup(value))
      return cached;
    Operation *op = value.getDefiningOp();
    Location loc = op->getLoc();
    Type element = cast<ShapedType>(value.getType()).getElementType();
    Value result;
    // Shape-only producers are reduced to their scalar payload; elementwise
    // operations are rebuilt as scalar arith.
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto dense = cast<DenseElementsAttr>(constant.getValue());
      result = builder.create<arith::ConstantOp>(
          loc, element, cast<TypedAttr>(dense.getSplatValue<Attribute>()));
    } else if (auto range = dyn_cast<VArangeOp>(op)) {
      Value index =
          builder.create<arith::MulIOp>(loc, row, range.getStrides().front());
      if (range.getOffset())
        index = builder.create<arith::AddIOp>(loc, index, range.getOffset());
      result = element.isIndex()
                   ? index
                   : builder.create<arith::IndexCastOp>(loc, element, index);
    } else if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      result = fill.getInputs().front();
    } else if (auto broadcast = dyn_cast<VBrcOp>(op)) {
      result = broadcast.getSrc();
    } else if (auto cast = dyn_cast<tensor::CastOp>(op)) {
      result = emit(cast.getSource(), row, builder);
    } else {
      SmallVector<Value> inputs;
      if (auto structured = dyn_cast<HIVMStructuredOp>(op))
        inputs = structured.getDpsInputs();
      else
        llvm::append_range(inputs, op->getOperands());
      SmallVector<Value> operands;
      for (Value input : inputs)
        operands.push_back(emit(input, row, builder));
      auto recipe = getScalarRecipe(op);
      if (recipe == ScalarRecipe::Compare) {
        auto compare = mlir::cast<VCmpOp>(op);
        result = builder.create<arith::CmpIOp>(loc, scalarPredicate(compare),
                                               operands[0], operands[1]);
      } else if (recipe == ScalarRecipe::Not) {
        auto intType = mlir::cast<IntegerType>(element);
        Value ones = builder.create<arith::ConstantOp>(
            loc, element,
            builder.getIntegerAttr(intType,
                                   APInt::getAllOnes(intType.getWidth())));
        result = builder.create<arith::XOrIOp>(loc, operands[0], ones);
      } else if (recipe == ScalarRecipe::Add) {
        result = builder.create<arith::AddIOp>(loc, operands[0], operands[1]);
      } else if (recipe == ScalarRecipe::Sub) {
        result = builder.create<arith::SubIOp>(loc, operands[0], operands[1]);
      } else if (recipe == ScalarRecipe::Mul) {
        result = builder.create<arith::MulIOp>(loc, operands[0], operands[1]);
      } else if (recipe == ScalarRecipe::And) {
        result = builder.create<arith::AndIOp>(loc, operands[0], operands[1]);
      } else if (recipe == ScalarRecipe::Or) {
        result = builder.create<arith::OrIOp>(loc, operands[0], operands[1]);
      } else if (recipe == ScalarRecipe::Xor) {
        result = builder.create<arith::XOrIOp>(loc, operands[0], operands[1]);
      } else {
        // The remaining supported operations are already scalar arith. Rebuild
        // them at the scalar element type, keeping flags (such as a comparison
        // predicate) that type inference cannot recover.
        OperationState state(loc, op->getName());
        state.addOperands(operands);
        state.addTypes(element);
        state.addAttributes(op->getAttrs());
        result = builder.create(state)->getResult(0);
      }
    }
    emitted[value] = result;
    return result;
  }

private:
  // A value is supported when it is scalar and loop invariant, a memoized row
  // read, or one row of integer data produced by a supported elementwise op.
  bool computeSupports(Value value) {
    if (!isa<ShapedType>(value.getType()))
      return value.getType().isIntOrIndex() && dominance.dominates(value, loop);
    if (rowValues.contains(value))
      return true;
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getEncoding() ||
        type.getShape() != ArrayRef<int64_t>{rows} ||
        !type.getElementType().isIntOrIndex())
      return false;
    Operation *op = value.getDefiningOp();
    if (!op || op->getNumResults() != 1 || hasInlineLayout(op))
      return false;
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      return dense && dense.isSplat();
    }
    if (auto range = dyn_cast<VArangeOp>(op))
      return range.getStrides().size() == 1 &&
             dominance.dominates(range.getStrides().front(), loop) &&
             (!range.getOffset() ||
              dominance.dominates(range.getOffset(), loop));
    SmallVector<Value> inputs;
    if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      llvm::append_range(inputs, fill.getInputs());
    } else if (auto broadcast = dyn_cast<VBrcOp>(op)) {
      if (isa<ShapedType>(broadcast.getSrc().getType()))
        return false;
      inputs.push_back(broadcast.getSrc());
    } else if (auto recipe = getScalarRecipe(op)) {
      if (*recipe == ScalarRecipe::Not && !type.getElementType().isInteger())
        return false;
      inputs = cast<HIVMStructuredOp>(op).getDpsInputs();
    } else if (isa<arith::ArithDialect>(op->getDialect()) &&
               op->hasTrait<OpTrait::Scalarizable>() &&
               isMemoryEffectFree(op) && isSpeculatable(op)) {
      // Scalarizable guarantees that replacing operands and results by their
      // element types preserves semantics; Elementwise alone is insufficient.
      llvm::append_range(inputs, op->getOperands());
    } else if (auto cast = dyn_cast<tensor::CastOp>(op)) {
      return supports(cast.getSource());
    } else {
      return false;
    }
    return llvm::all_of(inputs, [&](Value input) { return supports(input); });
  }

  scf::ForOp loop;
  int64_t rows;
  DominanceInfo &dominance;
  llvm::DenseSet<Value> rowValues;
  llvm::DenseMap<Value, bool> supported;
  llvm::DenseMap<Value, Value> emitted;
};

static FailureOr<SmallVector<bufferization::ToTensorOp>>
collectObservers(RowLoopCore &core) {
  SmallVector<Value> aliases;
  // Follow descriptor-only casts and tile subviews to a private allocation.
  // Only the independent-allocation proof looks through these casts.
  auto ub = AddressSpaceAttr::get(core.loop->getContext(), AddressSpace::UB);
  Value root = core.dst;
  aliases.push_back(root);
  while (true) {
    Value input;
    if (auto cast = root.getDefiningOp<memref::MemorySpaceCastOp>())
      input = cast.getSource();
    else if (auto cast = root.getDefiningOp<memref::CastOp>())
      input = cast.getSource();
    else if (auto view = root.getDefiningOp<memref::SubViewOp>())
      input = view.getSource();
    else
      break;
    auto from = dyn_cast<MemRefType>(input.getType());
    auto to = dyn_cast<MemRefType>(root.getType());
    // Cast verifiers guarantee storage identity, including dynamic-to-static
    // refinement. Subviews establish ancestry, not equality of observed tiles.
    if (!from || !to)
      return failure();
    if (auto space = to.getMemorySpace(); space && space != ub)
      return failure();
    root = input;
    aliases.push_back(root);
  }
  if (!root.getDefiningOp<memref::AllocOp>())
    return failure();
  auto rootType = cast<MemRefType>(root.getType());
  if (rootType.getMemorySpace() && rootType.getMemorySpace() != ub)
    return failure();

  SmallVector<bufferization::ToTensorOp> observations;
  bool exactTile = true;
  for (auto [index, alias] : llvm::enumerate(aliases)) {
    Operation *chainUser =
        index == 0 ? nullptr : aliases[index - 1].getDefiningOp();
    // Subviews may cover only part of their source allocation. Continue the
    // complete privacy walk, but never treat a wider observation as this tile.
    if (isa_and_nonnull<memref::SubViewOp>(chainUser))
      exactTile = false;
    for (Operation *user : alias.getUsers()) {
      if (user == chainUser || (alias == core.dst && user == core.slice))
        continue;
      // In-place masking requires every alias and observation to be accounted
      // for; device-core classification is unrelated to this privacy proof.
      if (isa<memref::DimOp, memref::RankOp>(user) ||
          (isa<annotation::MarkOp>(user) && isMemoryEffectFree(user)))
        continue;
      auto observation = dyn_cast<bufferization::ToTensorOp>(user);
      if (observation && hasOnlyMetadataUses(observation.getResult()))
        continue;
      if (!observation || !exactTile ||
          user->getBlock() != core.loop->getBlock() ||
          !core.loop->isBeforeInBlock(user)) {
        return failure();
      }
      observations.push_back(observation);
    }
  }
  return observations;
}

static bool isReusableRowRead(tensor::ExtractOp read, scf::ForOp loop,
                              int64_t rows, Value localRow) {
  auto type = cast<RankedTensorType>(read.getTensor().getType());
  return !type.getEncoding() && type.getShape() == ArrayRef<int64_t>{rows} &&
         read.getType().isIntOrIndex() &&
         loop.isDefinedOutsideOfLoop(read.getTensor()) && localRow &&
         read.getIndices().size() == 1 &&
         RowGeometryBounds::proveEqual(OpFoldResult(read.getIndices().front()),
                                       OpFoldResult(localRow));
}

// A mask that cannot be rebuilt as scalar arithmetic may still be a plain
// per-row predicate: it only has to be defined before the loop and carry one
// i1 per row. Reading it one element at a time replaces translating its
// formula, which covers external masks and arbitrary pre-loop producers.
static bool isByteReadableRowMask(const RowSelection &selection,
                                  scf::ForOp loop, DominanceInfo &dominance) {
  auto type = cast<RankedTensorType>(selection.mask.getType());
  return !type.getEncoding() && dominance.dominates(selection.mask, loop);
}

// Find an exact preimage for one numerical cast. Restrict this proof to integer
// extensions, exact integer-to-float conversions and float extensions. Exact
// values are independent of rounding, overflow and saturation choices, so the
// original cast and all its attributes can remain unchanged.
static FailureOr<TypedAttr> getExactCastPreimage(VCastOp conversion,
                                                 TypedAttr constant,
                                                 OpBuilder &builder) {
  Type source =
      cast<ShapedType>(conversion.getSingleSrc().getType()).getElementType();
  Type target =
      cast<ShapedType>(conversion->getResult(0).getType()).getElementType();
  if (constant.getType() != target)
    return failure();
  auto sourceInt = dyn_cast<IntegerType>(source);
  auto targetInt = dyn_cast<IntegerType>(target);
  bool sourceUnsigned = conversion.getCast() == TypeFn::cast_unsigned;
  bool targetUnsigned = sourceUnsigned;
  if (sourceInt && targetInt)
    if (auto mode =
            conversion->getAttrOfType<UnsignedModeAttr>(UnsignedModeAttr::name);
        mode && mode.getValue() != UnsignedMode::SI2SI) {
      sourceUnsigned = mode.getValue() == UnsignedMode::UI2SI ||
                       mode.getValue() == UnsignedMode::UI2UI;
      targetUnsigned = mode.getValue() == UnsignedMode::SI2UI ||
                       mode.getValue() == UnsignedMode::UI2UI;
    }
  // getCastName() first applies UnsignedMode, then util::getTypeName() also
  // treats explicitly unsigned integer types as unsigned regardless of TypeFn.
  sourceUnsigned |= sourceInt && sourceInt.isUnsigned();
  targetUnsigned |= targetInt && targetInt.isUnsigned();
  if (sourceInt && targetInt) {
    if (sourceInt.getWidth() <= 1 ||
        sourceInt.getWidth() >= targetInt.getWidth() ||
        sourceUnsigned != targetUnsigned)
      return failure();
    auto integer = dyn_cast<IntegerAttr>(constant);
    if (!integer)
      return failure();
    APInt candidate = integer.getValue().trunc(sourceInt.getWidth());
    APInt restored = sourceUnsigned ? candidate.zext(targetInt.getWidth())
                                    : candidate.sext(targetInt.getWidth());
    if (restored != integer.getValue())
      return failure();
    return TypedAttr(builder.getIntegerAttr(sourceInt, candidate));
  }
  auto number = dyn_cast<FloatAttr>(constant);
  auto targetFloat = dyn_cast<FloatType>(target);
  auto supportedFloat = [](FloatType type) {
    return type &&
           (type.isF16() || type.isBF16() || type.isF32() || type.isF64());
  };
  if (!number || !supportedFloat(targetFloat) || !number.getValue().isFinite())
    return failure();
  if (sourceInt) {
    if (sourceInt.getWidth() <= 1)
      return failure();
    llvm::APSInt candidate(sourceInt.getWidth(), sourceUnsigned);
    bool exact = false;
    if (number.getValue().convertToInteger(candidate, APFloat::rmTowardZero,
                                           &exact) != APFloat::opOK ||
        !exact)
      return failure();
    APFloat restored(targetFloat.getFloatSemantics());
    if (restored.convertFromAPInt(candidate, !sourceUnsigned,
                                  APFloat::rmNearestTiesToEven) !=
            APFloat::opOK ||
        !restored.bitwiseIsEqual(number.getValue()))
      return failure();
    return TypedAttr(builder.getIntegerAttr(sourceInt, candidate));
  }
  auto sourceFloat = dyn_cast<FloatType>(source);
  if (!supportedFloat(sourceFloat) ||
      sourceFloat.getWidth() >= targetFloat.getWidth())
    return failure();
  APFloat candidate = number.getValue();
  bool losesInfo = false;
  if (candidate.convert(sourceFloat.getFloatSemantics(),
                        APFloat::rmNearestTiesToEven,
                        &losesInfo) != APFloat::opOK ||
      losesInfo)
    return failure();
  APFloat restored = candidate;
  if (restored.convert(targetFloat.getFloatSemantics(),
                       APFloat::rmNearestTiesToEven,
                       &losesInfo) != APFloat::opOK ||
      losesInfo || !restored.bitwiseIsEqual(number.getValue()))
    return failure();
  return TypedAttr(builder.getFloatAttr(sourceFloat, candidate));
}

// Mask sinking and fill hoisting share the same pre-mutation alias proof.
class RowObservers {
public:
  explicit RowObservers(RowLoopCore &core) : core(core) {}

  const FailureOr<SmallVector<bufferization::ToTensorOp>> &get() {
    if (!result)
      result = collectObservers(core);
    return *result;
  }

private:
  RowLoopCore &core;
  std::optional<FailureOr<SmallVector<bufferization::ToTensorOp>>> result;
};

// Analysis owns no IR mutations. A rejected plan leaves all operations intact.
struct RowMaskPlan {
  RowSelection selection;
  DenseMap<Value, Value> rowReads;
  bool scalarMask = false;
  TypedAttr fallbackConstant;
  Value fallback;
};

static std::optional<RowMaskPlan> matchRowMask(RowLoopCore &core,
                                               bool coversTile, Value localRow,
                                               RowObservers &observers,
                                               DominanceInfo &dominance,
                                               OpBuilder &builder) {
  if (!coversTile || core.rowGuard || !hasPrivateRowTarget(core))
    return std::nullopt;
  const auto &observations = observers.get();
  if (failed(observations) || observations->size() != 1)
    return std::nullopt;
  auto observation = observations->front();
  auto selection = getRowSelection(observation.getResult());
  if (!selection)
    return std::nullopt;
  RowMaskPlan plan;
  plan.selection = *selection;
  for (tensor::ExtractOp read : core.rowReads)
    if (read->isBeforeInBlock(core.copy) &&
        isReusableRowRead(read, core.loop, core.rows, localRow))
      plan.rowReads.try_emplace(read.getTensor(), read.getResult());
  MaskScalarizer scalarizer(core.loop, core.rows, dominance, plan.rowReads);
  plan.scalarMask = scalarizer.supports(selection->mask);
  if (!plan.scalarMask &&
      !isByteReadableRowMask(*selection, core.loop, dominance))
    return std::nullopt;

  Type element = cast<MemRefType>(core.dst.getType()).getElementType();
  TypedAttr constant = selection->fallbackConstant;
  if (!selection->prefix.empty()) {
    if (!constant)
      if (auto op = selection->fallback.getDefiningOp<arith::ConstantOp>())
        constant = dyn_cast<TypedAttr>(op.getValue());
    if (!constant)
      return std::nullopt;
    for (VCastOp conversion : llvm::reverse(selection->prefix)) {
      auto preimage = getExactCastPreimage(conversion, constant, builder);
      if (failed(preimage))
        return std::nullopt;
      constant = *preimage;
    }
  }
  if (!constant) {
    if (selection->fallback.getType() != element ||
        !dominance.dominates(selection->fallback, core.loop))
      return std::nullopt;
    plan.fallback = selection->fallback;
  } else {
    auto integer = dyn_cast<IntegerType>(element);
    if (constant.getType() != element || (integer && !integer.isSignless()))
      return std::nullopt;
    plan.fallbackConstant = constant;
  }
  return plan;
}

struct PreparedRowMask {
  Value fallback;
  Value byteMask;
};

static PreparedRowMask materializeRowMask(const RowMaskPlan &plan,
                                          RowLoopCore &core,
                                          OpBuilder &builder) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPoint(core.loop);
  PreparedRowMask result{plan.fallback, {}};
  if (plan.fallbackConstant)
    result.fallback = builder.create<arith::ConstantOp>(
        core.loop.getLoc(), plan.fallbackConstant.getType(),
        plan.fallbackConstant);
  if (!plan.scalarMask) {
    auto roundMode = builder.getAttr<RoundModeAttr>(RoundMode::RINT);
    result.byteMask = castTo(builder, core.loop.getLoc(), plan.selection.mask,
                             roundMode, builder.getI8Type())
                          ->getResult(0);
  }
  return result;
}

// Each replay needs its own scalar cache, indexed by destination row, not IV.
static Value emitRowPredicate(const RowMaskPlan &plan,
                              const PreparedRowMask &prepared,
                              RowLoopCore &core, Value row, IRMapping &mapping,
                              DominanceInfo &dominance, OpBuilder &builder) {
  Value valid;
  if (plan.scalarMask) {
    DenseMap<Value, Value> rowReads;
    for (auto [tensor, read] : plan.rowReads)
      rowReads.try_emplace(tensor, mapping.lookup(read));
    MaskScalarizer scalarizer(core.loop, core.rows, dominance,
                              std::move(rowReads));
    valid = scalarizer.emit(plan.selection.mask, row, builder);
  } else {
    Value byte = builder.create<tensor::ExtractOp>(
        core.loop.getLoc(), prepared.byteMask, ValueRange{row});
    Value zero = builder.create<arith::ConstantIntOp>(core.loop.getLoc(), 0, 8);
    valid = builder.create<arith::CmpIOp>(core.loop.getLoc(),
                                          arith::CmpIPredicate::ne, byte, zero);
  }
  if (plan.selection.inverted) {
    Value one = builder.create<arith::ConstantIntOp>(core.loop.getLoc(), 1, 1);
    valid = builder.create<arith::XOrIOp>(core.loop.getLoc(), valid, one);
  }
  return valid;
}

static void eraseRowSelection(RowMaskPlan &plan, RewriterBase &rewriter) {
  plan.selection.result.replaceAllUsesWith(plan.selection.replacement);
  if (plan.selection.select->use_empty())
    rewriter.eraseOp(plan.selection.select);
}
struct RowTransfer {
  RowLoopCore core;
  OpFoldResult offset;
  SmallVector<Operation *> replayOps;
  SmallVector<Operation *> pairReplayOps;
  bool coversTile = false;
  std::optional<int64_t> tripCount;
};

struct RowAddressAnalysis {
  explicit RowAddressAnalysis(RowTransfer &transfer) : transfer(transfer) {}

  RowTransfer &transfer;
  llvm::DenseSet<Operation *> addressOps;
  bool visitRowRead(tensor::ExtractOp read) {
    if (!transfer.core.loop.isDefinedOutsideOfLoop(read.getTensor()) ||
        !read.getType().isIntOrIndex())
      return false;
    return llvm::all_of(read.getIndices(),
                        [&](Value index) { return visit(index); });
  }

  bool visit(Value value) {
    scf::ForOp loop = transfer.core.loop;
    if (value == loop.getInductionVar())
      return true;
    if (loop.isDefinedOutsideOfLoop(value))
      return value.getType().isIntOrIndex();
    Operation *def = value.getDefiningOp();
    // The matcher rejects iter_args and
    // nested regions other than the result-free load/fill guard.
    if (!def || def->getBlock() != loop.getBody())
      return false;
    if (!addressOps.insert(def).second)
      return true;
    if (auto read = dyn_cast<tensor::ExtractOp>(def))
      return visitRowRead(read);
    if (!isa<affine::AffineApplyOp>(def)) {
      auto scalarInteger = [](Type type) { return type.isIntOrIndex(); };
      if (!isa<arith::ArithDialect>(def->getDialect()) ||
          !llvm::all_of(def->getOperandTypes(), scalarInteger) ||
          !llvm::all_of(def->getResultTypes(), scalarInteger))
        return false;
    }
    return llvm::all_of(def->getOperands(),
                        [&](Value operand) { return visit(operand); });
  }
};

// Source geometry and replay are independent of optional mask ownership.
static LogicalResult matchRowGather(scf::ForOp loop, RowTransfer &transfer,
                                    PatternRewriter &rewriter) {
  if (!loop->hasAttr(ParallelLoopAttr::name) || loop.getNumRegionIterArgs() ||
      getConstantIntValue(loop.getStep()) != 1)
    return failure();
  StringRef reason;
  auto core = recoverRowLoopCore(loop, reason);
  if (failed(core))
    return rewriter.notifyMatchFailure(loop, reason);
  if (!isPlainRowLoad(core->copy))
    return failure();
  auto type = cast<MemRefType>(core->dst.getType());
  Type element = type.getElementType();
  if (!element.isInteger(8) && !element.isInteger(16) &&
      !element.isInteger(32) && !element.isInteger(64) && !element.isF16() &&
      !element.isBF16() && !element.isF32())
    return failure();
  auto slice = core->slice;
  using Variable = ValueBoundsConstraintSet::Variable;
  auto equal = [&](OpFoldResult lhs, int64_t rhs) {
    return RowGeometryBounds::proveEqual(lhs, rhs);
  };
  if (!equal(slice.getMixedSizes()[0], 1) ||
      !equal(slice.getMixedSizes()[1], core->cols) ||
      !equal(slice.getMixedOffsets()[1], 0) ||
      !llvm::all_of(slice.getMixedStrides(),
                    [&](OpFoldResult stride) { return equal(stride, 1); }))
    return failure();
  transfer.tripCount = RowGeometryBounds::constantDifference(
      loop.getUpperBound(), loop.getLowerBound());
  transfer.coversTile =
      transfer.tripCount == std::optional<int64_t>(core->rows) &&
      RowGeometryBounds::proveEqual(
          Variable(slice.getMixedOffsets()[0]),
          RowGeometryBounds::difference(loop.getInductionVar(),
                                        loop.getLowerBound()));
  // Full coverage already proves unit row progression. Partial windows need
  // the weaker invariant-offset proof before adjacent rows can be paired.
  if (!transfer.coversTile &&
      !RowGeometryBounds::proveUnitRowOffset(slice.getMixedOffsets()[0], loop))
    return failure();
  transfer.core = std::move(*core);
  return success();
}

// Unmasked pairs need only reconstruction inputs, not single-row descriptors.
static void collectPairReplayOps(RowTransfer &transfer) {
  auto &core = transfer.core;
  SmallVector<Value> needed{core.source.getSource()};
  if (auto value = transfer.offset.dyn_cast<Value>())
    needed.push_back(value);
  if (auto value = core.slice.getMixedOffsets()[0].dyn_cast<Value>())
    needed.push_back(value);
  for (Value operand : core.copy->getOperands())
    if (!isa<MemRefType>(operand.getType()))
      needed.push_back(operand);
  llvm::DenseSet<Operation *> pairOps;
  for (Operation *op : transfer.replayOps)
    if (isa<annotation::MarkOp>(op)) {
      pairOps.insert(op);
      llvm::append_range(needed, op->getOperands());
    }
  while (!needed.empty()) {
    Value value = needed.pop_back_val();
    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != core.loop.getBody() ||
        !pairOps.insert(op).second)
      continue;
    llvm::append_range(needed, op->getOperands());
  }
  for (Operation *op : transfer.replayOps)
    if (pairOps.contains(op))
      transfer.pairReplayOps.push_back(op);
}

static bool hasUBStorage(Value value) {
  // A descriptor cast must not hide an explicitly non-UB allocation.
  auto ub = AddressSpaceAttr::get(value.getContext(), AddressSpace::UB);
  while (auto type = dyn_cast<MemRefType>(value.getType())) {
    if (auto space = type.getMemorySpace(); space && space != ub)
      return false;
    Operation *def = value.getDefiningOp();
    if (auto cast = dyn_cast_or_null<memref::CastOp>(def))
      value = cast.getSource();
    else if (auto cast = dyn_cast_or_null<memref::MemorySpaceCastOp>(def))
      value = cast.getSource();
    else if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(def))
      value = view.getViewSource();
    else
      break;
  }
  return true;
}

static LogicalResult verifyRowTransfer(RowTransfer &transfer,
                                       PatternRewriter &rewriter) {
  auto &core = transfer.core;
  scf::ForOp loop = core.loop;
  // Only pairing reconstructs DMA descriptors. Mask sinking keeps the original
  // load and therefore does not require a fixed base or dense storage.
  if (!core.source || (core.source->getBlock() != loop.getBody() &&
                       !loop.isDefinedOutsideOfLoop(core.source.getResult())))
    return rewriter.notifyMatchFailure(
        loop, "source: expected a local or invariant row descriptor");
  if (core.source->getBlock() == loop.getBody())
    core.setupViews.push_back(core.source);
  if (!matchGMRowBase(core))
    return rewriter.notifyMatchFailure(loop,
                                       "source: requires a fixed GM base");
  if (!hasUBStorage(core.dst))
    return rewriter.notifyMatchFailure(
        loop, "destination: descriptor aliases non-UB storage");
  if (!isDenseRowTile(cast<MemRefType>(core.dst.getType())))
    return rewriter.notifyMatchFailure(
        loop, "destination: requires an invariant dense tile");
  auto sourceSizes = core.source.getMixedSizes();
  if (sourceSizes.empty() || sourceSizes.size() > 2 ||
      !RowGeometryBounds::proveEqual(sourceSizes.back(), core.cols) ||
      (sourceSizes.size() == 2 &&
       !RowGeometryBounds::proveEqual(sourceSizes.front(), 1)) ||
      !RowGeometryBounds::proveEqual(core.source.getMixedStrides().back(), 1))
    return rewriter.notifyMatchFailure(
        loop, "source: expected a contiguous full-row descriptor");
  transfer.offset = core.source.getMixedOffsets()[0];
  // Proof 1 -- address replay: prove the scalar address DAG, including the row
  // guard condition. The emitter reuses invariant leaves and clones the
  // effect-free loop-local setup verbatim.
  RowAddressAnalysis address(transfer);
  Value offset = transfer.offset.dyn_cast<Value>();
  if ((offset && !address.visit(offset)) ||
      (transfer.core.rowGuard &&
       !address.visit(transfer.core.rowGuard.getCondition())))
    return rewriter.notifyMatchFailure(
        loop, "address expression: requires invariant row tensors and a "
              "bounded scalar integer DAG");
  // Collect descriptor and load auxiliary operands using the same scalar DAG.
  // Known views are structural nodes; their memref operands are not scalar
  // DAGs.
  for (Operation *view : transfer.core.setupViews) {
    for (Value operand : view->getOperands())
      if (!isa<MemRefType>(operand.getType()) && !address.visit(operand))
        return rewriter.notifyMatchFailure(
            loop, "view: unsupported descriptor dependency");
  }
  for (Value operand : transfer.core.copy->getOperands())
    if (!isa<MemRefType>(operand.getType()) && !address.visit(operand))
      return rewriter.notifyMatchFailure(
          loop, "load: unsupported auxiliary dependency");
  // Cover every body operation exactly once, in original order, except the
  // direct load or top-level guard. Dead operations may be omitted, but
  // unknown effects and live work outside the address DAG remain unsupported.
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (&op == transfer.core.copy || &op == transfer.core.rowGuard)
      continue;
    if (!llvm::is_contained(transfer.core.setupViews, &op) &&
        !address.addressOps.contains(&op)) {
      if (isOpTriviallyDead(&op))
        continue;
      return rewriter.notifyMatchFailure(
          loop, "extra users: unsupported operation in row loop");
    }
    transfer.replayOps.push_back(&op);
  }

  collectPairReplayOps(transfer);

  // Both a pair and its single-row fallback must fit the DMA burst.
  auto dstType = cast<MemRefType>(transfer.core.dst.getType());
  int64_t bytes = dstType.getElementTypeBitWidth() / 8;
  if (transfer.core.cols > kMaxBurstBytes / bytes ||
      (transfer.core.cols * bytes) % kDmaAlignmentBytes != 0)
    return rewriter.notifyMatchFailure(
        loop, "DMA limits: row length must fit the burst limit and be "
              "32-byte aligned");

  return success();
}

// Keep a row's predicate with the replayed descriptors that produced it.
struct ReplayedRow {
  IRMapping mapping;
  Value valid;
};

static Value replayIndex(OpFoldResult value, const IRMapping &mapping,
                         Location loc, OpBuilder &builder) {
  if (auto index = value.dyn_cast<Value>())
    return mapping.lookupOrDefault(index);
  return getValueOrCreateConstantIndexOp(builder, loc, value);
}

struct HoistedFillPlan {
  Value fallback;
  TypedAttr constant;
};

// A complete private tile can be filled before replay without exposing writes.
static std::optional<HoistedFillPlan>
analyzeHoistedFill(RowTransfer &transfer, const RowMaskPlan *mask,
                   RowObservers &observers, DominanceInfo &dominance) {
  auto &core = transfer.core;
  if (!transfer.coversTile || (!mask && !core.rowGuard) ||
      failed(observers.get()))
    return std::nullopt;
  if (mask)
    return HoistedFillPlan{mask->fallback, mask->fallbackConstant};
  auto fill = cast<VBrcOp>(core.rowGuard.elseBlock()->front());
  if (!fill.getBroadcastDims().empty() || hasInlineLayout(fill))
    return std::nullopt;
  Value fallback = fill.getSrc();
  if (dominance.dominates(fallback, core.loop))
    return HoistedFillPlan{fallback, {}};
  Attribute constant;
  if (matchPattern(fallback, m_Constant(&constant)))
    return HoistedFillPlan{{}, cast<TypedAttr>(constant)};
  return std::nullopt;
}

class RowEmitter {
public:
  RowEmitter(RowTransfer &transfer, const RowMaskPlan *mask,
             DominanceInfo &dominance, PatternRewriter &rewriter)
      : transfer(transfer), core(transfer.core), mask(mask),
        dominance(dominance), rewriter(rewriter) {
    if (mask)
      prepared = materializeRowMask(*mask, core, rewriter);
  }

  bool isMasked() const { return mask || core.rowGuard; }

  void emitHoistedFill(const HoistedFillPlan &plan) {
    Value fallback = mask ? prepared.fallback : plan.fallback;
    if (!fallback)
      fallback = rewriter.create<arith::ConstantOp>(
          core.loop.getLoc(), plan.constant.getType(), plan.constant);
    emitFill({}, core.dst, fallback);
    hoistedFill = true;
  }

  enum class ReplayMode { SingleRow, PairOnly };

  ReplayedRow replay(Value iv, ReplayMode mode = ReplayMode::SingleRow) {
    assert((mode == ReplayMode::SingleRow || !isMasked()) &&
           "masked fallback requires single-row descriptors");
    ReplayedRow row;
    row.mapping.map(core.loop.getInductionVar(), iv);
    auto &operations = mode == ReplayMode::SingleRow ? transfer.replayOps
                                                     : transfer.pairReplayOps;
    for (Operation *op : operations)
      rewriter.clone(*op, row.mapping);
    row.valid = emitPredicate(row.mapping);
    return row;
  }

  void emitLoad(Value src, Value dst, const IRMapping &address) {
    IRMapping mapping(address);
    mapping.map(core.copy.getSrc(), src);
    mapping.map(core.copy.getDst(), dst);
    rewriter.clone(*core.copy, mapping);
  }

  void emitSingle(const ReplayedRow &row) {
    auto load = [&] {
      IRMapping mapping(row.mapping);
      rewriter.clone(*core.copy, mapping);
    };
    if (!isMasked()) {
      load();
      return;
    }
    auto guard =
        rewriter.create<scf::IfOp>(core.loop.getLoc(), row.valid, !hoistedFill);
    OpBuilder::InsertionGuard insertion(rewriter);
    rewriter.setInsertionPointToStart(guard.thenBlock());
    load();
    if (!hoistedFill) {
      rewriter.setInsertionPointToStart(guard.elseBlock());
      emitFill(row.mapping);
    }
  }

  ReplayedRow select(Value isFirst, const ReplayedRow &first,
                     const ReplayedRow &second) {
    auto choose = [&](Value lhs, Value rhs) -> Value {
      return rewriter.createOrFold<arith::SelectOp>(core.loop.getLoc(), isFirst,
                                                    lhs, rhs);
    };
    ReplayedRow row;
    row.mapping = first.mapping;
    row.valid = choose(first.valid, second.valid);
    auto selectOperand = [&](Value value) {
      row.mapping.map(value, choose(first.mapping.lookupOrDefault(value),
                                    second.mapping.lookupOrDefault(value)));
    };
    selectOperand(core.copy.getSrc());
    selectOperand(core.copy.getDst());
    if (core.rowGuard)
      selectOperand(cast<VBrcOp>(core.rowGuard.elseBlock()->front()).getSrc());
    return row;
  }

private:
  Value emitPredicate(IRMapping &mapping) {
    if (mask) {
      Value row = replayIndex(core.slice.getMixedOffsets()[0], mapping,
                              core.loop.getLoc(), rewriter);
      return emitRowPredicate(*mask, prepared, core, row, mapping, dominance,
                              rewriter);
    }
    return core.rowGuard ? mapping.lookupOrDefault(core.rowGuard.getCondition())
                         : Value{};
  }

  void emitFill(const IRMapping &address, Value dst = {}, Value fallback = {}) {
    if (core.rowGuard) {
      auto fill = cast<VBrcOp>(core.rowGuard.elseBlock()->front());
      IRMapping mapping(address);
      if (dst)
        mapping.map(fill.getDst(), dst);
      if (fallback)
        mapping.map(fill.getSrc(), fallback);
      rewriter.clone(*fill, mapping);
      return;
    }
    rewriter.create<VBrcOp>(core.loop.getLoc(), TypeRange{},
                            fallback ? fallback : prepared.fallback,
                            dst ? dst
                                : address.lookupOrDefault(core.copy.getDst()),
                            rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
  }

  RowTransfer &transfer;
  RowLoopCore &core;
  const RowMaskPlan *mask;
  DominanceInfo &dominance;
  PatternRewriter &rewriter;
  PreparedRowMask prepared;
  bool hoistedFill = false;
};

// Group original iterations, rather than using the allocation size as a trip
// count. An incomplete destination window is legal for pairing, but cannot
// justify removing a select over the complete tile.
static void emitRowDMA(RowTransfer &transfer, RowEmitter &rows,
                       bool fallbackLoop, PatternRewriter &rewriter) {
  auto &core = transfer.core;
  scf::ForOp loop = core.loop;
  auto source = core.source;
  Location loc = loop.getLoc();
  int64_t cols = core.cols;
  auto attr = [&](int64_t value) -> OpFoldResult {
    return rewriter.getIndexAttr(value);
  };
  auto index = [&](int64_t value) -> Value {
    return rewriter.create<arith::ConstantIndexOp>(loc, value);
  };
  auto add = [&](Value lhs, Value rhs) -> Value {
    return rewriter.createOrFold<arith::AddIOp>(loc, lhs, rhs);
  };
  auto sub = [&](Value lhs, Value rhs) -> Value {
    return rewriter.createOrFold<arith::SubIOp>(loc, lhs, rhs);
  };
  auto emitPairFallback = [&](const ReplayedRow &first,
                              const ReplayedRow &second) {
    auto lanes = rewriter.create<scf::ForOp>(loc, index(0), index(2), index(1));
    lanes->setAttr(ParallelLoopAttr::name, rewriter.getUnitAttr());
    OpBuilder::InsertionGuard insertion(rewriter);
    rewriter.setInsertionPointToStart(lanes.getBody());
    Value isFirst = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lanes.getInductionVar(), index(0));
    rows.emitSingle(rows.select(isFirst, first, second));
  };
  auto makeSource = [&](const ReplayedRow &first,
                        const ReplayedRow &second) -> Value {
    Value start = replayIndex(transfer.offset, first.mapping, loc, rewriter);
    Value end = replayIndex(transfer.offset, second.mapping, loc, rewriter);
    Value distance = sub(end, start);
    auto baseType = cast<MemRefType>(source.getSource().getType());
    auto type = MemRefType::get(
        {2, cols}, baseType.getElementType(),
        StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                               {ShapedType::kDynamic, 1}),
        baseType.getMemorySpace());
    return rewriter.create<memref::ReinterpretCastOp>(
        loc, type, first.mapping.lookupOrDefault(source.getSource()),
        OpFoldResult(start), ArrayRef<OpFoldResult>{attr(2), attr(cols)},
        ArrayRef<OpFoldResult>{OpFoldResult(distance), attr(1)});
  };

  Value zero = index(0), two = index(2);
  Value remainder, pairEnd;
  if (transfer.tripCount) {
    int64_t count = std::max<int64_t>(0, *transfer.tripCount);
    remainder = index(count % 2);
    pairEnd = add(loop.getLowerBound(), index(count - count % 2));
  } else {
    // The signed difference may overflow even for an empty loop. Test the
    // original bounds first; unsigned parity remains valid across wraparound.
    Value active = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, loop.getLowerBound(),
        loop.getUpperBound());
    Value difference = sub(loop.getUpperBound(), loop.getLowerBound());
    Value odd = rewriter.createOrFold<arith::RemUIOp>(loc, difference, two);
    remainder = rewriter.createOrFold<arith::SelectOp>(loc, active, odd, zero);
    pairEnd = rewriter.createOrFold<arith::SelectOp>(
        loc, active, sub(loop.getUpperBound(), odd), loop.getLowerBound());
  }
  auto paired =
      rewriter.create<scf::ForOp>(loc, loop.getLowerBound(), pairEnd, two);
  paired->setAttr(ParallelLoopAttr::name, rewriter.getUnitAttr());
  {
    OpBuilder::InsertionGuard insertion(rewriter);
    rewriter.setInsertionPointToStart(paired.getBody());
    Value iv = paired.getInductionVar();
    auto mode = rows.isMasked() ? RowEmitter::ReplayMode::SingleRow
                                : RowEmitter::ReplayMode::PairOnly;
    auto first = rows.replay(iv, mode);
    auto second = rows.replay(add(iv, index(1)), mode);
    scf::IfOp guard;
    if (rows.isMasked()) {
      Value both =
          rewriter.create<arith::AndIOp>(loc, first.valid, second.valid);
      guard = rewriter.create<scf::IfOp>(loc, both, true);
      rewriter.setInsertionPointToStart(guard.thenBlock());
    }
    Value src = makeSource(first, second);
    Value row = replayIndex(core.slice.getMixedOffsets()[0], first.mapping, loc,
                            rewriter);
    Value target = rewriter.create<memref::SubViewOp>(
        loc, core.dst, ArrayRef<OpFoldResult>{OpFoldResult(row), attr(0)},
        ArrayRef<OpFoldResult>{attr(2), attr(cols)},
        ArrayRef<OpFoldResult>{attr(1), attr(1)});
    rows.emitLoad(src, target, first.mapping);
    if (guard) {
      rewriter.setInsertionPointToStart(guard.elseBlock());
      if (fallbackLoop)
        emitPairFallback(first, second);
      else {
        rows.emitSingle(first);
        rows.emitSingle(second);
      }
    }
  }
  rewriter.setInsertionPointAfter(paired);
  if (auto odd = getConstantIntValue(remainder)) {
    if (*odd)
      rows.emitSingle(rows.replay(pairEnd));
  } else {
    Value isOdd = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                 remainder, zero);
    auto tail = rewriter.create<scf::IfOp>(loc, isOdd, false);
    OpBuilder::InsertionGuard insertion(rewriter);
    rewriter.setInsertionPointToStart(tail.thenBlock());
    rows.emitSingle(rows.replay(pairEnd));
  }
  rewriter.eraseOp(loop);
}

static bool optimizeRowGather(scf::ForOp loop, bool fallbackLoop,
                              DominanceInfo &dominance,
                              PatternRewriter &rewriter) {
  RowTransfer transfer;
  if (failed(matchRowGather(loop, transfer, rewriter)))
    return false;
  // Fill hoisting and mask sinking are only performed as part of pairing, so a
  // loop whose rows cannot be paired is left untouched.
  if ((transfer.tripCount && *transfer.tripCount < 2) ||
      failed(verifyRowTransfer(transfer, rewriter)))
    return false;
  RowObservers observers(transfer.core);
  auto mask =
      matchRowMask(transfer.core, transfer.coversTile,
                   transfer.core.slice.getMixedOffsets()[0].dyn_cast<Value>(),
                   observers, dominance, rewriter);
  auto *maskPlan = mask ? &*mask : nullptr;
  auto fill = analyzeHoistedFill(transfer, maskPlan, observers, dominance);

  RowEmitter rows(transfer, maskPlan, dominance, rewriter);
  rewriter.setInsertionPoint(loop);
  if (fill)
    rows.emitHoistedFill(*fill);
  emitRowDMA(transfer, rows, fallbackLoop, rewriter);
  if (mask)
    eraseRowSelection(*mask, rewriter);
  return true;
}

} // namespace

struct OptimizeRowLoadsPass : impl::OptimizeRowLoadsBase<OptimizeRowLoadsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    arith::registerValueBoundsOpInterfaceExternalModels(registry);
    affine::registerValueBoundsOpInterfaceExternalModels(registry);
    scf::registerValueBoundsOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto core = queryFuncCoreType(func);
    // VF helpers are outlined into non-entry functions before this pass, so
    // checking the AIV device entry alone excludes them.
    if (!enabled || !hacc::utils::isDeviceEntry(func) || !core ||
        *core != TFuncCoreType::AIV || func->hasAttr("hivm.backup_function") ||
        !hacc::utils::isAscend910_95(func->getParentOfType<ModuleOp>()))
      return;
    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    DominanceInfo dominance(func);
    PatternRewriter rewriter(&getContext());
    for (scf::ForOp loop : loops) {
      rewriter.setInsertionPoint(loop);
      if (optimizeRowGather(loop, fallbackLoop, dominance, rewriter))
        dominance.invalidate();
    }
  }
};

std::unique_ptr<Pass>
createOptimizeRowLoadsPass(const OptimizeRowLoadsOptions &options) {
  return std::make_unique<OptimizeRowLoadsPass>(options);
}
} // namespace mlir::hivm

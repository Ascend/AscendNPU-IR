//===- HIVMAliasAnalysis.cpp - BiShengIR memory region queries ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/HIVMAliasAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;
using namespace mlir::utils;

namespace {

/// Build and simplify address arithmetic in a private block. All expressions
/// are MLIR values; the input IR is never rewritten. Widening to i64 preserves
/// signed address arithmetic under the no-overflow contract.
class OffsetAnalysis {
public:
  OffsetAnalysis(Operation *scope, Operation *context)
      : builder(scope->getContext()), loc(scope->getLoc()),
        indexWidth(DataLayout::closest(scope)
                       .getTypeSizeInBits(builder.getIndexType())
                       .getFixedValue()) {
    builder.setInsertionPointToEnd(&scratch.emplaceBlock());
    if (!context || !scope->isAncestor(context))
      return;
    DominanceInfo dominance(scope);
    scope->walk([&](LLVM::AssumeOp assume) {
      if (auto cmp = assume.getCond().getDefiningOp<arith::CmpIOp>();
          cmp && dominance.properlyDominates(assume, context))
        assumptions.push_back(cmp);
    });
  }

  Value constant(int64_t value) {
    return create<arith::ConstantIntOp>(value, 64);
  }

  Value getValue(OpFoldResult value, unsigned depth = 0) {
    if (auto attr = dyn_cast<Attribute>(value))
      return constant(cast<IntegerAttr>(attr).getInt());
    Value original = cast<Value>(value);
    if (auto it = values.find(original); it != values.end())
      return it->second;
    if (depth > 32 || !original.getType().isIntOrIndex() ||
        getWidth(original.getType()) > 64)
      return {};
    Value result = expand(original, depth);
    if (!result) {
      // Unsupported operations remain distinct SSA values. A cloned operation
      // in a future iteration cannot accidentally reuse this opaque value.
      if (original.getType().isIndex())
        result = create<arith::IndexCastOp>(builder.getI64Type(), original);
      else if (getWidth(original.getType()) < 64)
        result = create<arith::ExtSIOp>(builder.getI64Type(), original);
      else
        result = original;
    }
    values[original] = result;
    return result;
  }

  /// A positive signed lower bound, from a constant or a dominating assume.
  /// Normalize casts and arithmetic exactly as for the queried stride.
  int64_t getPositiveLowerBound(Value value) {
    int64_t bound = getConstantIntValue(value).value_or(0);
    for (arith::CmpIOp cmp : assumptions) {
      Value lhs = getValue(cmp.getLhs()), rhs = getValue(cmp.getRhs());
      if (!lhs || !rhs)
        continue;
      auto predicate = cmp.getPredicate();
      std::optional<int64_t> constant;
      bool strict = false;
      if (lhs == value && (predicate == arith::CmpIPredicate::sge ||
                           predicate == arith::CmpIPredicate::sgt)) {
        constant = getConstantIntValue(rhs);
        strict = predicate == arith::CmpIPredicate::sgt;
      } else if (rhs == value && (predicate == arith::CmpIPredicate::sle ||
                                  predicate == arith::CmpIPredicate::slt)) {
        constant = getConstantIntValue(lhs);
        strict = predicate == arith::CmpIPredicate::slt;
      }
      if (!constant)
        continue;
      int64_t lower = *constant;
      if (strict && llvm::AddOverflow(lower, int64_t(1), lower))
        continue;
      bound = std::max(bound, lower);
    }
    return std::max(int64_t(0), bound);
  }

  Value add(Value a, Value b) {
    if (!a || !b)
      return {};
    auto ca = getConstantIntValue(a), cb = getConstantIntValue(b);
    if (ca && cb) {
      int64_t sum;
      return llvm::AddOverflow(*ca, *cb, sum) ? Value() : constant(sum);
    }
    if (ca)
      std::swap(a, b), std::swap(ca, cb);
    if (cb == 0)
      return a;
    if (auto sum = a.getDefiningOp<arith::AddIOp>(); sum && cb) {
      if (auto c = getConstantIntValue(sum.getRhs())) {
        int64_t combined;
        if (llvm::AddOverflow(*c, *cb, combined))
          return {};
        return add(sum.getLhs(), constant(combined));
      }
    }
    if (!cb && std::less<const void *>()(b.getAsOpaquePointer(),
                                         a.getAsOpaquePointer()))
      std::swap(a, b);
    return create<arith::AddIOp>(a, b);
  }

  Value multiply(Value a, Value b) {
    if (!a || !b)
      return {};
    auto ca = getConstantIntValue(a), cb = getConstantIntValue(b);
    if (ca && cb) {
      int64_t product;
      return llvm::MulOverflow(*ca, *cb, product) ? Value() : constant(product);
    }
    if (ca)
      std::swap(a, b), std::swap(ca, cb);
    if (cb == 0)
      return b;
    if (cb == 1)
      return a;
    if (!cb && std::less<const void *>()(b.getAsOpaquePointer(),
                                         a.getAsOpaquePointer()))
      std::swap(a, b);
    return create<arith::MulIOp>(a, b);
  }

  /// Cancel shared bases and factor shared strides:
  /// (B + (i + 64)*S) - (B + i*S) -> ((i + 64) - i)*S -> 64*S.
  Value subtract(Value a, Value b, unsigned depth = 0) {
    if (!a || !b || depth > 32)
      return {};
    if (a == b)
      return constant(0);
    auto ca = getConstantIntValue(a), cb = getConstantIntValue(b);
    if (ca && cb) {
      int64_t difference;
      return llvm::SubOverflow(*ca, *cb, difference) ? Value()
                                                     : constant(difference);
    }
    auto x = a.getDefiningOp<arith::AddIOp>();
    auto y = b.getDefiningOp<arith::AddIOp>();
    if (x && y) {
      for (unsigned i = 0; i != 2; ++i)
        for (unsigned j = 0; j != 2; ++j)
          if (x->getOperand(i) == y->getOperand(j))
            return subtract(x->getOperand(1 - i), y->getOperand(1 - j),
                            depth + 1);
    }
    if (x && (x.getLhs() == b || x.getRhs() == b))
      return x.getLhs() == b ? x.getRhs() : x.getLhs();
    auto ma = a.getDefiningOp<arith::MulIOp>();
    auto mb = b.getDefiningOp<arith::MulIOp>();
    if (ma && mb) {
      for (unsigned i = 0; i != 2; ++i)
        for (unsigned j = 0; j != 2; ++j)
          if (ma->getOperand(i) == mb->getOperand(j))
            return multiply(subtract(ma->getOperand(1 - i),
                                     mb->getOperand(1 - j), depth + 1),
                            ma->getOperand(i));
    }
    if (cb) {
      int64_t negated;
      if (llvm::SubOverflow(int64_t(0), *cb, negated))
        return {};
      return add(a, constant(negated));
    }
    if (x)
      return add(subtract(x.getLhs(), b, depth + 1), x.getRhs());
    if (y)
      return subtract(subtract(a, y.getLhs(), depth + 1), y.getRhs(),
                      depth + 1);
    if (auto sub = a.getDefiningOp<arith::SubIOp>())
      return subtract(subtract(sub.getLhs(), b, depth + 1), sub.getRhs(),
                      depth + 1);
    if (auto sub = b.getDefiningOp<arith::SubIOp>())
      return add(subtract(a, sub.getLhs(), depth + 1), sub.getRhs());
    return create<arith::SubIOp>(a, b);
  }

private:
  // Reuse equivalent scratch operations, so identical calculations share SSA
  // values. Bound the work even when an address has a large dependency graph.
  template <typename Op, typename... Args> Value create(Args &&...args) {
    if (++numOperations > 512)
      return {};
    Operation *op = builder.create<Op>(loc, std::forward<Args>(args)...);
    auto hash = OperationEquivalence::computeHash(
        op, OperationEquivalence::directHashValue,
        OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
    auto &bucket = operations[hash];
    for (Operation *existing : bucket)
      if (OperationEquivalence::isEquivalentTo(
              op, existing, OperationEquivalence::IgnoreLocations)) {
        op->erase();
        return existing->getResult(0);
      }
    bucket.push_back(op);
    return op->getResult(0);
  }

  unsigned getWidth(Type type) const {
    return type.isIndex() ? indexWidth : cast<IntegerType>(type).getWidth();
  }

  Value expand(Value value, unsigned depth) {
    if (auto c = getConstantIntValue(value))
      return constant(*c);
    if (auto apply = value.getDefiningOp<affine::AffineApplyOp>())
      return expandAffine(apply.getAffineMap().getResult(0),
                          apply.getMapOperands(),
                          apply.getAffineMap().getNumDims(), depth + 1);
    if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
      if (getWidth(castOp.getIn().getType()) <= getWidth(value.getType()))
        return getValue(castOp.getIn(), depth + 1);
      return {};
    }
    if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
      return getValue(ext.getIn(), depth + 1);
    Operation *op = value.getDefiningOp();
    if (!isa_and_present<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op))
      return {};
    Value a = getValue(op->getOperand(0), depth + 1);
    Value b = getValue(op->getOperand(1), depth + 1);
    if (isa<arith::AddIOp>(op))
      return add(a, b);
    if (isa<arith::SubIOp>(op))
      return subtract(a, b);
    return multiply(a, b);
  }

  Value expandAffine(AffineExpr expr, ValueRange operands, unsigned numDims,
                     unsigned depth) {
    if (depth > 32)
      return {};
    if (auto c = dyn_cast<AffineConstantExpr>(expr))
      return constant(c.getValue());
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
      return getValue(operands[dim.getPosition()], depth + 1);
    if (auto symbol = dyn_cast<AffineSymbolExpr>(expr))
      return getValue(operands[numDims + symbol.getPosition()], depth + 1);
    auto binary = cast<AffineBinaryOpExpr>(expr);
    Value a = expandAffine(binary.getLHS(), operands, numDims, depth + 1);
    Value b = expandAffine(binary.getRHS(), operands, numDims, depth + 1);
    if (expr.getKind() == AffineExprKind::Add)
      return add(a, b);
    if (expr.getKind() == AffineExprKind::Mul)
      return multiply(a, b);
    return {};
  }

  Region scratch;
  OpBuilder builder;
  Location loc;
  unsigned indexWidth;
  unsigned numOperations = 0;
  DenseMap<Value, Value> values;
  DenseMap<llvm::hash_code, SmallVector<Operation *>> operations;
  SmallVector<arith::CmpIOp> assumptions;
};

/// Half-open interval [start, start + extent). A bounding interval may contain
/// holes, which only makes the independence proof more conservative.
struct Interval {
  Value start;
  int64_t extent = 0;
};

static bool areDisjoint(const Interval &lhs, const Interval &rhs,
                        OffsetAnalysis &analysis) {
  if (!lhs.extent || !rhs.extent)
    return true;
  Value difference = analysis.subtract(rhs.start, lhs.start);
  auto distance = difference ? getConstantIntValue(difference) : std::nullopt;
  // Compare the separation with the full bounding extents, not just starts.
  return distance && (*distance >= lhs.extent || *distance <= -rhs.extent);
}

static std::optional<int64_t> getSizeBound(OpFoldResult size) {
  if (auto constant = getConstantIntValue(size))
    return *constant >= 0 ? constant : std::nullopt;
  auto bound = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, ValueBoundsConstraintSet::Variable(size),
      /*stopCondition=*/nullptr, /*closedUB=*/true);
  if (succeeded(bound) && *bound >= 0)
    return *bound;
  return std::nullopt;
}

static Value stripCasts(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<memref::CastOp, memref::MemorySpaceCastOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

static Value getStorage(Value value) {
  while (true) {
    value = stripCasts(value);
    auto view = value.getDefiningOp<ViewLikeOpInterface>();
    if (!view)
      return value;
    value = view.getViewSource();
  }
}

struct AccessRegion {
  Value base;
  Value offset;
  SmallVector<std::optional<int64_t>> sizes;
  SmallVector<int64_t> strides;
  // reinterpret_cast offsets are relative to the aligned pointer, while
  // subviews are relative to the source's logical origin. Never mix these.
  bool relativeToAlignedPointer = false;
};

static FailureOr<AccessRegion> getRegion(Value value, OffsetAnalysis &analysis,
                                         unsigned depth = 0) {
  if (depth > 32)
    return failure();
  value = stripCasts(value);
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type)
    return failure();
  if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto offset = analysis.getValue(view.getMixedOffsets().front());
    if (!offset)
      return failure();
    AccessRegion region;
    region.base = view.getSource();
    region.offset = offset;
    region.relativeToAlignedPointer = true;
    for (OpFoldResult size : view.getMixedSizes())
      region.sizes.push_back(getSizeBound(size));
    for (OpFoldResult stride : view.getMixedStrides()) {
      auto constant = getConstantIntValue(stride);
      if (!constant || *constant <= 0)
        return failure();
      region.strides.push_back(*constant);
    }
    return region;
  }
  auto view = value.getDefiningOp<memref::SubViewOp>();
  if (!view) {
    // Treat other memrefs as opaque anchors at their logical origin. This is
    // useful even for unknown producers, provided both queries share the SSA
    // value. Clients must account for the anchor when changing executions.
    AccessRegion region;
    region.base = value;
    region.offset = analysis.constant(0);
    int64_t ignoredOffset;
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
    if (failed(getStridesAndOffset(type, region.strides, ignoredOffset)))
#else
    if (failed(type.getStridesAndOffset(region.strides, ignoredOffset)))
#endif
      return failure();
    for (int64_t size : type.getShape())
      region.sizes.push_back(ShapedType::isDynamic(size)
                                 ? std::nullopt
                                 : std::optional<int64_t>(size));
    return region;
  }
  auto source = getRegion(view.getSource(), analysis, depth + 1);
  if (failed(source))
    return failure();
  AccessRegion region;
  region.base = source->base;
  region.offset = source->offset;
  region.relativeToAlignedPointer = source->relativeToAlignedPointer;
  auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes();
  auto strides = view.getMixedStrides();
  auto dropped = view.getDroppedDims();
  for (unsigned dim = 0; dim < offsets.size(); ++dim) {
    auto offset = analysis.getValue(offsets[dim]);
    auto stride = getConstantIntValue(strides[dim]);
    int64_t sourceStride = source->strides[dim], resultStride;
    if (!offset || !stride || *stride <= 0 || sourceStride <= 0 ||
        ShapedType::isDynamic(sourceStride) ||
        llvm::MulOverflow(sourceStride, *stride, resultStride))
      return failure();
    region.offset = analysis.add(
        region.offset,
        analysis.multiply(offset, analysis.constant(sourceStride)));
    if (!region.offset)
      return failure();
    if (dropped[dim])
      continue;
    auto size = getSizeBound(sizes[dim]);
    // An in-bounds subview with positive integer stride cannot contain more
    // elements along a dimension than its source, including clamped tails.
    if (source->sizes[dim])
      size = size ? std::min(*size, *source->sizes[dim]) : source->sizes[dim];
    region.sizes.push_back(size);
    region.strides.push_back(resultStride);
  }
  return region;
}

static FailureOr<Interval> getInterval(const AccessRegion &region) {
  int64_t extent = 1;
  for (auto [size, stride] : llvm::zip_equal(region.sizes, region.strides)) {
    if (!size || stride <= 0 || ShapedType::isDynamic(stride))
      return failure();
    if (!*size)
      return Interval{region.offset, 0};
    int64_t contribution;
    if (llvm::MulOverflow(*size - 1, stride, contribution) ||
        llvm::AddOverflow(extent, contribution, extent))
      return failure();
  }
  return Interval{region.offset, extent};
}

/// A rectangle with contiguous columns and a shared row stride.
/// Sizes bound the access extent; they imply no constraint on the stride.
struct StridedRegion {
  Value base;
  Value offset;
  Value rowStride;
  int64_t rows;
  int64_t columns;
};

static FailureOr<StridedRegion>
getStridedRegion(Value value, OffsetAnalysis &analysis, unsigned depth = 0) {
  if (depth > 32)
    return failure();
  value = stripCasts(value);
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 2)
    return failure();
  if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto strides = view.getMixedStrides();
    if (getConstantIntValue(strides[1]) != 1)
      return failure();
    auto stride = analysis.getValue(strides[0]);
    auto offset = analysis.getValue(view.getMixedOffsets().front());
    auto sizes = view.getMixedSizes();
    auto rows = getSizeBound(sizes[0]), columns = getSizeBound(sizes[1]);
    if (!stride || !offset || !rows || !columns)
      return failure();
    return StridedRegion{view.getSource(), offset, stride, *rows, *columns};
  }
  auto view = value.getDefiningOp<memref::SubViewOp>();
  if (!view || view.getSourceType().getRank() != 2)
    return failure();
  for (OpFoldResult stride : view.getMixedStrides())
    if (getConstantIntValue(stride) != 1)
      return failure();
  auto region = getStridedRegion(view.getSource(), analysis, depth + 1);
  if (failed(region))
    return failure();
  auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes();
  auto rowOffset = analysis.getValue(offsets[0]);
  auto columnOffset = analysis.getValue(offsets[1]);
  region->offset = analysis.add(
      analysis.add(region->offset,
                   analysis.multiply(rowOffset, region->rowStride)),
      columnOffset);
  if (!region->offset)
    return failure();
  if (auto rows = getSizeBound(sizes[0]))
    region->rows = std::min(region->rows, *rows);
  if (auto columns = getSizeBound(sizes[1]))
    region->columns = std::min(region->columns, *columns);
  return region;
}

static bool areDisjoint(const StridedRegion &a, const StridedRegion &b,
                        OffsetAnalysis &analysis) {
  if (a.rowStride != b.rowStride)
    return false;
  if (!a.rows || !a.columns || !b.rows || !b.columns)
    return true;
  int64_t minStride = analysis.getPositiveLowerBound(a.rowStride);
  if (!minStride)
    return false;
  Value distance = analysis.subtract(b.offset, a.offset);
  if (!distance)
    return false;
  int64_t coefficient = 0, constant = 0;
  // Match the simplified MLIR expression q*S+c.
  if (auto add = distance.getDefiningOp<arith::AddIOp>()) {
    auto c = getConstantIntValue(add.getRhs());
    if (!c)
      return false;
    constant = *c;
    distance = add.getLhs();
  }
  if (auto c = getConstantIntValue(distance)) {
    if (llvm::AddOverflow(constant, *c, constant))
      return false;
  } else if (distance == a.rowStride) {
    coefficient = 1;
  } else if (auto mul = distance.getDefiningOp<arith::MulIOp>();
             mul && mul.getLhs() == a.rowStride) {
    auto q = getConstantIntValue(mul.getRhs());
    if (!q)
      return false;
    coefficient = *q;
  } else {
    return false;
  }

  // For S >= minStride > 0, compare q*S+c with each tile's bounding span.
  // A nonnegative coefficient permits substituting the known lower bound.
  auto wide = [](int64_t n) { return llvm::APInt(128, n, /*isSigned=*/true); };
  auto nonnegative = [&](llvm::APInt factor, llvm::APInt constant) {
    if (factor.isNegative())
      return false;
    bool overflow;
    auto value = factor.smul_ov(wide(minStride), overflow);
    if (overflow)
      return false;
    value = value.sadd_ov(constant, overflow);
    return !overflow && !value.isNegative();
  };
  auto q = wide(coefficient), c = wide(constant);
  auto ra = wide(a.rows - 1), rb = wide(b.rows - 1);
  auto ca = wide(a.columns), cb = wide(b.columns);
  return nonnegative(q - ra, c - ca) || nonnegative(-q - rb, -c - cb);
}

struct MemoryRegion {
  enum class Coordinates { Unknown, LogicalOrigin, AlignedPointer };
  Value base;
  Type elementType;
  Coordinates coordinates = Coordinates::Unknown;
  std::optional<Interval> interval;
  std::optional<StridedRegion> stridedRegion;
};

static FailureOr<MemoryRegion> getMemoryRegion(Value value,
                                               OffsetAnalysis &analysis) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type)
    return failure();
  MemoryRegion result;
  result.base = getStorage(value);
  result.elementType = type.getElementType();
  auto region = getRegion(value, analysis);
  if (succeeded(region)) {
    auto interval = getInterval(*region);
    if (succeeded(interval)) {
      result.base = region->base;
      result.coordinates = region->relativeToAlignedPointer
                               ? MemoryRegion::Coordinates::AlignedPointer
                               : MemoryRegion::Coordinates::LogicalOrigin;
      result.interval = *interval;
      return result;
    }
  }
  auto strided = getStridedRegion(value, analysis);
  if (succeeded(strided)) {
    result.base = strided->base;
    result.coordinates = MemoryRegion::Coordinates::AlignedPointer;
    result.stridedRegion = *strided;
    return result;
  }
  return result;
}

} // namespace

AliasResult HIVMAliasAnalysis::alias(Value lhs, Value rhs, Operation *context) {
  // Reinterpret casts may change the range, so query MLIR AA on the storage.
  if (aliasAnalysis.alias(getStorage(lhs), getStorage(rhs)).isNo())
    return AliasResult::NoAlias;
  OffsetAnalysis analysis(scope, context);
  auto a = getMemoryRegion(lhs, analysis), b = getMemoryRegion(rhs, analysis);
  if (failed(a) || failed(b) || a->base != b->base ||
      a->elementType != b->elementType || a->coordinates != b->coordinates)
    return AliasResult::MayAlias;
  if (a->stridedRegion && b->stridedRegion &&
      areDisjoint(*a->stridedRegion, *b->stridedRegion, analysis))
    return AliasResult::NoAlias;
  if (a->interval && b->interval &&
      areDisjoint(*a->interval, *b->interval, analysis))
    return AliasResult::NoAlias;
  return AliasResult::MayAlias;
}

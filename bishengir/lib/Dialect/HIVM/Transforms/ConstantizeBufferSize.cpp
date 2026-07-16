//===- ConstantizeBufferSize.cpp --------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONSTANTIZEBUFFERSIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
Value getOperandForDimOrSymbol(AffineExpr expr, AffineMap map,
                               ValueRange operands) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
    return operands[dimExpr.getPosition()];
  if (auto symbolExpr = dyn_cast<AffineSymbolExpr>(expr))
    return operands[map.getNumDims() + symbolExpr.getPosition()];
  return {};
}

/// This pattern is useful for sinking affine.max ops.
/// Rewrite max(C, min(A, B)) into min(max(C, A), max(C, B)).
/// If C & B are constant, fold max(C, min(A, B)) into max(C, K).
/// Helping constantize compute the upper bound of dynamic dimension values.
struct DistributeAffineMaxOverMin
    : public OpRewritePattern<affine::AffineMaxOp> {
  using OpRewritePattern<affine::AffineMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineMaxOp maxOp,
                                PatternRewriter &rewriter) const override {
    AffineMap maxMap = maxOp.getAffineMap();
    ValueRange maxOperands = maxOp.getMapOperands();

    // Judge if the maxOp is max(C, min(A, B)).
    std::optional<int64_t> constantLowerBound;
    affine::AffineMinOp minOp;
    for (AffineExpr expr : maxMap.getResults()) {
      // If the expr include several constant expr, then we compute
      // the max value of all constant expr or get the only constant expr.
      // max(C, D, min(A, B)) -> max(max(C, D), min(A, B))
      if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
        constantLowerBound =
            constantLowerBound.has_value()
                ? std::max(*constantLowerBound, constExpr.getValue())
                : constExpr.getValue();
        continue;
      }

      // If the expr is not constant expr, then it must be min(A, B).
      // Besides, only one min(A, B) is allowed.
      Value operand = getOperandForDimOrSymbol(expr, maxMap, maxOperands);
      if (!operand)
        return failure();
      auto operandMinOp = operand.getDefiningOp<affine::AffineMinOp>();
      if (!operandMinOp)
        return failure();
      if (minOp && minOp != operandMinOp)
        return failure();
      minOp = operandMinOp;
    }
    
    // Only if the maxOp is max(C, min(A, B)), we can rewrite it.
    if (!constantLowerBound.has_value() || !minOp)
      return failure();

    Location loc = maxOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    AffineMap minMap = minOp.getAffineMap();
    ValueRange minOperands = minOp.getMapOperands();

    // Rewrite max(C, min(A, B)) into min(max(C, A), max(C, B)).
    SmallVector<Value> newMinOperands;
    SmallVector<AffineExpr> newMinExprs;
    for (AffineExpr minExpr : minMap.getResults()) {
      // If the minExpr is constant expr, for example A, then we compute
      // max(C, A) as constant expr and add it to newMinExprs.
      if (auto constExpr = dyn_cast<AffineConstantExpr>(minExpr)) {
        newMinExprs.push_back(rewriter.getAffineConstantExpr(
            std::max(*constantLowerBound, constExpr.getValue())));
        continue;
      }

      // If the minExpr is not constant expr, for example B, then we compute
      // max(C, B) as a new map and add it to newMinOperands and add affine
      // expr max(C, B) to newMinExprs.
      auto innerMaxMap = AffineMap::get(
          minMap.getNumDims(), minMap.getNumSymbols(),
          {rewriter.getAffineConstantExpr(*constantLowerBound), minExpr}, ctx);
      Value innerMax =
          rewriter.create<affine::AffineMaxOp>(loc, innerMaxMap, minOperands);
      newMinOperands.push_back(innerMax);
      newMinExprs.push_back(
          rewriter.getAffineSymbolExpr(newMinOperands.size() - 1));
    }

    if (newMinExprs.empty())
      return failure();

    auto newMinMap =
        AffineMap::get(/*dimCount=*/0, newMinOperands.size(), newMinExprs, ctx);
    rewriter.replaceOpWithNewOp<affine::AffineMinOp>(maxOp, newMinMap,
                                                     newMinOperands);
    return success();
  }
};

/// Result of resolving a dynamic dimension — either an exact constant or
/// a conservative upper bound (worst-case size).
struct ResolvedDim {
  int64_t value;
  bool isExact; ///< true if the value is the exact runtime constant
};

/// Walk the SSA chain of a dynamic dimension value and try to resolve it to a
/// constant or at least an upper bound.  Handles arith.minsi (upper bound),
/// arith.maxsi, affine.apply, arith.{constant,addi,subi,muli},
/// arith.index_cast.  Returns the resolved bound or std::nullopt on failure.
std::optional<ResolvedDim> resolveDynamicDimToBoundImpl(Value dim) {
  if (!dim)
    return std::nullopt;

  auto *defOp = dim.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  // --- constants (exact) ---
  if (auto constOp = dyn_cast<arith::ConstantIndexOp>(defOp))
    return ResolvedDim{constOp.value(), /*isExact=*/true};
  if (auto constOp = dyn_cast<arith::ConstantIntOp>(defOp))
    return ResolvedDim{constOp.value(), /*isExact=*/true};

  // --- minsi(a, b): max = min(max(a), max(b)) ---
  if (auto minOp = dyn_cast<arith::MinSIOp>(defOp)) {
    auto a = resolveDynamicDimToBoundImpl(minOp.getLhs());
    auto b = resolveDynamicDimToBoundImpl(minOp.getRhs());
    if (a.has_value() && b.has_value()) {
      int64_t bound = std::min(a->value, b->value);
      // result is exact if both operands are exact, OR if one exact operand
      // is <= the other's bound (since then the min is forced to that exact
      // value)
      bool exact = (a->isExact && b->isExact) ||
                   (a->isExact && a->value <= b->value) ||
                   (b->isExact && b->value <= a->value);
      return ResolvedDim{bound, exact};
    }
    if (a.has_value())
      return ResolvedDim{a->value, /*isExact=*/false};
    if (b.has_value())
      return ResolvedDim{b->value, /*isExact=*/false};
    return std::nullopt;
  }

  // --- maxsi(a, b): max = max(max(a), max(b)) ---
  // Always returns an upper bound (conservative) even if only one operand has a
  // bound. Exactness: true only if both operands are exact, OR the exact
  // operand is larger.
  if (auto maxOp = dyn_cast<arith::MaxSIOp>(defOp)) {
    auto a = resolveDynamicDimToBoundImpl(maxOp.getLhs());
    auto b = resolveDynamicDimToBoundImpl(maxOp.getRhs());
    if (!a.has_value() && !b.has_value())
      return std::nullopt;
    int64_t bound = std::max(a.has_value() ? a->value : INT64_MIN,
                             b.has_value() ? b->value : INT64_MIN);
    bool exact = false;
    if (a.has_value() && b.has_value()) {
      if (a->isExact && b->isExact)
        exact = true;
      else if (a->isExact && a->value >= b->value)
        exact = true;
      else if (b->isExact && b->value >= a->value)
        exact = true;
    } else if (a.has_value() && a->isExact) {
      // Only one operand known and it's exact → the max is that exact value
      exact = true;
    } else if (b.has_value() && b->isExact) {
      exact = true;
    }
    return ResolvedDim{bound, exact};
  }

  // --- affine.apply: evaluate with max values of operands to get upper bound
  // --- isExact is set to true only if all operands are exact (then constant
  // fold yields exact value).
  if (auto affineOp = dyn_cast<affine::AffineApplyOp>(defOp)) {
    SmallVector<ResolvedDim> resolvedOps;
    for (auto operand : affineOp.getOperands()) {
      auto resolved = resolveDynamicDimToBoundImpl(operand);
      if (!resolved.has_value())
        return std::nullopt;
      resolvedOps.push_back(*resolved);
    }
    auto map = affineOp.getAffineMap();

    // Upper bound: substitute max values for each operand
    SmallVector<Attribute> maxAttrs;
    for (auto &r : resolvedOps) {
      maxAttrs.push_back(
          IntegerAttr::get(IndexType::get(defOp->getContext()), r.value));
    }
    SmallVector<Attribute> maxResults;
    if (failed(map.constantFold(maxAttrs, maxResults)) || maxResults.empty())
      return std::nullopt;
    auto maxAttr = dyn_cast<IntegerAttr>(maxResults[0]);
    if (!maxAttr)
      return std::nullopt;
    int64_t maxVal = maxAttr.getValue().getSExtValue();

    // Exact only if every operand is exact (then min == max and constant fold
    // yields same)
    bool isExact = llvm::all_of(resolvedOps,
                                [](const ResolvedDim &r) { return r.isExact; });
    return ResolvedDim{maxVal, isExact};
  }

  // --- addi(a, b): max = max(a) + max(b) (non-negative assumed) ---
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    auto a = resolveDynamicDimToBoundImpl(addOp.getLhs());
    auto b = resolveDynamicDimToBoundImpl(addOp.getRhs());
    if (a.has_value() && b.has_value())
      return ResolvedDim{a->value + b->value, a->isExact && b->isExact};
    return std::nullopt;
  }

  // --- subi(a, b): conservative upper bound = max(a) - min(b).
  // Since min(b) is unknown, we only handle cases where b is known to be
  // non-negative and exact (so min(b)=max(b)=value). Otherwise return nothing
  // to avoid underestimating the bound.
  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    auto a = resolveDynamicDimToBoundImpl(subOp.getLhs());
    auto b = resolveDynamicDimToBoundImpl(subOp.getRhs());
    if (!a.has_value())
      return std::nullopt;
    // If b is exact and non‑negative, then max(a-b) = max(a) - b
    if (b.has_value() && b->isExact && b->value >= 0)
      return ResolvedDim{a->value - b->value, a->isExact && b->isExact};
    // If b is not known precisely or could be negative, we cannot give a safe
    // UB.
    return std::nullopt;
  }

  // --- muli(a, b): max = max(a) * max(b) (assumes non-negative operands) ---
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    auto a = resolveDynamicDimToBoundImpl(mulOp.getLhs());
    auto b = resolveDynamicDimToBoundImpl(mulOp.getRhs());
    if (a.has_value() && b.has_value()) {
      // Check for potential overflow in multiplication (just a safety clamp)
      int64_t product;
      if (__builtin_mul_overflow(a->value, b->value, &product))
        return std::nullopt;
      return ResolvedDim{product, a->isExact && b->isExact};
    }
    return std::nullopt;
  }

  // --- arith.index_cast: recurse to source ---
  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return resolveDynamicDimToBoundImpl(castOp.getIn());

  return std::nullopt;
}

FailureOr<int64_t> resolveDynamicDimToBound(Value dim) {
  auto res = resolveDynamicDimToBoundImpl(dim);
  if (res.has_value()) {
    return res->value;
  } else {
    return failure();
  }
}
} // namespace

namespace {
struct ConstantizeBufferPass
    : public impl::ConstantizeBufferSizeBase<ConstantizeBufferPass> {
  void runOnOperation() override;
};

template <typename AllocLikeOp>
struct ConstantizeAllocLikeOp : public OpRewritePattern<AllocLikeOp> {
public:
  using OpRewritePattern<AllocLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocLikeOp op,
                                PatternRewriter &rewriter) const override {
    auto dynSizes = op.getDynamicSizes();
    // No change if alloc is already fully static.
    if (dynSizes.empty())
      return failure();

    SmallVector<Value> newDynSizes;
    SmallVector<int64_t> newStaticSizes;
    MemRefType currentMemRefType = op.getType();
    auto currentSizes = currentMemRefType.getShape();
    int dynSizeIdx = 0;
    for (auto size : currentSizes) {
      if (!ShapedType::isDynamic(size)) {
        newStaticSizes.push_back(size);
        continue;
      }
      auto dynSize = dynSizes[dynSizeIdx];
      dynSizeIdx++;
      FailureOr<int64_t> upperBound =
          ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::UB, dynSize,
              /*stopCondition=*/nullptr, /*closedUB=*/true);
      if (failed(upperBound)) {
        upperBound = resolveDynamicDimToBound(dynSize);
      }
      // If failed to compute constant upper bound, do nothing.
      if (failed(upperBound)) {
        newStaticSizes.push_back(size);
        newDynSizes.push_back(dynSize);
        continue;
      }
      newStaticSizes.push_back(*upperBound);
    }
    if (newStaticSizes == currentSizes) {
      return rewriter.notifyMatchFailure(op, " already constantized");
    }

    // Construct new alloc with all dimensions constantized
    auto totalBits = utils::getStaticTotalSizeInBits(
        newStaticSizes, currentMemRefType.getElementType());
    if (!totalBits.has_value()) {
      return rewriter.notifyMatchFailure(
          op, " all dynamic dimensions should be constantized");
    }

    auto bufferSize = getAnnotateBufferSizeInBits(op.getResult());
    if (bufferSize.has_value()) {
      if (totalBits > bufferSize)
        return rewriter.notifyMatchFailure(
            op, " constantized buffer size should not exceed set buffer size");

      // remove all buffer_size_in_byte annotations if alloc can be constantized
      eraseAnnotateBufferSizeUsers(op.getResult(), rewriter);
    }

    int64_t totalBytes = static_cast<int64_t>(
        llvm::divideCeil(totalBits.value(), utils::kBitsToByte));
    auto newType =
        MemRefType::get({totalBytes}, rewriter.getI8Type(), mlir::AffineMap{},
                        currentMemRefType.getMemorySpace());
    Location loc = op->getLoc();
    auto newAlloc = rewriter.create<AllocLikeOp>(loc, newType);
    auto startOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.replaceOpWithNewOp<memref::ViewOp>(
        op, /*resultType=*/currentMemRefType, /*source=*/newAlloc.getResult(),
        /*byte_shift=*/startOffset,
        /*sizes=*/op->getOperands());
    return success();
  }

private:
  std::optional<int64_t> getAnnotateBufferSizeInBits(Value value) const {
    std::optional<Operation *> markMaybe =
        utils::getAnnotateOpWithAttr(value, kBufferSizeInByteAttr);
    if (!markMaybe.has_value()) {
      return std::nullopt;
    }
    auto markOp = cast<annotation::MarkOp>(markMaybe.value());
    return markOp->getAttrOfType<IntegerAttr>(kBufferSizeInByteAttr).getInt();
  }

  void eraseAnnotateBufferSizeUsers(Value value,
                                    PatternRewriter &rewriter) const {
    SmallVector<Operation *> annotateUsers;
    llvm::for_each(value.getUsers(), [&](Operation *user) {
      if (utils::isAnnotationWithAttr(user, kBufferSizeInByteAttr))
        annotateUsers.push_back(user);
    });

    for (Operation *user : annotateUsers)
      rewriter.eraseOp(user);
  }
};

} // namespace

void ConstantizeBufferPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  if (hacc::utils::isMemBasedArch(funcOp->getParentOfType<ModuleOp>()))
    patterns.add<DistributeAffineMaxOverMin>(patterns.getContext());
  patterns.add<ConstantizeAllocLikeOp<memref::AllocOp>,
               ConstantizeAllocLikeOp<memref::AllocaOp>>(patterns.getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createConstantizeBufferSizePass() {
  return std::make_unique<ConstantizeBufferPass>();
}

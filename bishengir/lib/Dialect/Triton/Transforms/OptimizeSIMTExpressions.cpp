//===-- OptimizeSIMTExpressions.cpp - SIMT expression rewrites -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Optimize Triton expressions after optional SIMT auto blockification:
//   1. Keep shared elementwise work narrow across a factor-2/4 broadcast.
//      Split operands that differ between partitions, then restore the result.
//      A broadcast of an ordered two-term sum can use the same partitioning.
//   2. Fold x ^ select(c, x ^ y, 0) into select(c, y, x).
// Broadcast candidates are built in a detached block and committed only when
// they share work and stay within the rewrite's size limits.
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "optimize-simt-expressions"
#define DBGS() (llvm::dbgs() << "[OptimizeSIMTExpressions] ")

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_OPTIMIZESIMTEXPRESSIONS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

// Only single-result, region-free and side-effect-free operations may be
// cloned across partitions. Triton bitcast is included explicitly.
static bool isPointwise(Operation *op) {
  return op && op->getNumResults() == 1 && op->getNumRegions() == 0 &&
         isMemoryEffectFree(op) &&
         (op->hasTrait<OpTrait::Elementwise>() ||
          isa<mlir::triton::BitcastOp>(op));
}

// factor is the expanded axis size; stride is the product of dimensions to
// its right. For [2, 1, 4] -> [2, 4, 4], factor=4 and stride=4.
struct BroadcastInfo {
  mlir::triton::BroadcastOp op;
  int64_t factor;
  int64_t stride;
};

// Find one rank-preserving broadcast through optional reshapes. Only one
// static, unencoded dimension may expand from 1 to 2 or 4.
static std::optional<BroadcastInfo> getBroadcastInfo(Value value) {
  while (auto reshape = value.getDefiningOp<mlir::triton::ReshapeOp>())
    value = reshape.getSrc();
  auto broadcast = value.getDefiningOp<mlir::triton::BroadcastOp>();
  if (!broadcast)
    return std::nullopt;
  auto src = broadcast.getSrc().getType();
  auto dst = broadcast.getType();
  if (!src.hasStaticShape() || !dst.hasStaticShape() || src.getEncoding() ||
      dst.getEncoding() || src.getRank() != dst.getRank() ||
      dst.getNumElements() == 0)
    return std::nullopt;
  std::optional<unsigned> axis;
  for (unsigned i = 0; i < src.getRank(); ++i) {
    if (src.getShape()[i] == dst.getShape()[i])
      continue;
    if (axis || src.getShape()[i] != 1 || dst.getShape()[i] <= 1)
      return std::nullopt;
    axis = i;
  }
  // Keep both uniform and mixed rewrites within small binary partitions.
  if (!axis || (dst.getShape()[*axis] != 2 && dst.getShape()[*axis] != 4))
    return std::nullopt;
  int64_t stride = 1;
  for (unsigned i = *axis + 1; i < dst.getRank(); ++i)
    stride *= dst.getShape()[i];
  return BroadcastInfo{broadcast, dst.getShape()[*axis], stride};
}

// Recognize reduce_add -> expand_dims -> broadcast when the reduced axis has
// exactly two elements. Replacing [a, b] with the ordered add a+b needs no
// reassociation; larger reductions and other combiners remain opaque.
static mlir::triton::ReduceOp getTwoTermSum(BroadcastInfo info) {
  if (info.factor != 2)
    return {};
  auto expand = info.op.getSrc().getDefiningOp<mlir::triton::ExpandDimsOp>();
  if (!expand)
    return {};
  auto reduce = expand.getSrc().getDefiningOp<mlir::triton::ReduceOp>();
  if (!reduce || reduce.getNumOperands() != 1 ||
      reduce.getAxis() != expand.getAxis() ||
      cast<RankedTensorType>(reduce.getOperand(0).getType()).getShape() !=
          info.op.getType().getShape())
    return {};
  Block &body = reduce.getCombineOp().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2 ||
      !isa<arith::AddFOp, arith::AddIOp>(body.front()) ||
      body.front().getOperand(0) != body.getArgument(0) ||
      body.front().getOperand(1) != body.getArgument(1) ||
      body.getTerminator()->getOperand(0) != body.front().getResult(0))
    return {};
  return reduce;
}

// Evaluate only the small constant/mask expressions needed to prove that
// each partition is a splat. With factor=2 and stride=1, the mask
// [1, 0, 1, 0] becomes [1, 1] and [0, 0]. Unknown values return nullopt.
static std::optional<int64_t> evaluateElement(Value value, int64_t index,
                                              unsigned depth = 0) {
  if (depth > 12)
    return std::nullopt;
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    Attribute attr = constant.getValue();
    if (auto dense = dyn_cast<DenseElementsAttr>(attr))
      attr = dense.getValues<Attribute>()[index];
    if (auto integer = dyn_cast<IntegerAttr>(attr)) {
      if (integer.getValue().getBitWidth() <= 64)
        return integer.getValue().getSExtValue();
      return std::nullopt;
    }
    if (auto fp = dyn_cast<FloatAttr>(attr)) {
      if (fp.getValue().isZero() && !fp.getValue().isNegative())
        return 0;
      if (fp.getValue().isExactlyValue(1.0))
        return 1;
    }
    return std::nullopt;
  }
  if (auto range = dyn_cast<mlir::triton::MakeRangeOp>(op))
    return range.getStart() + index;
  if (auto broadcast = dyn_cast<mlir::triton::BroadcastOp>(op)) {
    auto srcShape = broadcast.getSrc().getType().getShape();
    auto dstShape = broadcast.getType().getShape();
    int64_t srcIndex = 0, stride = 1;
    for (int i = dstShape.size() - 1; i >= 0; --i) {
      int64_t coordinate = index % dstShape[i];
      index /= dstShape[i];
      if (srcShape[i] != 1)
        srcIndex += coordinate * stride;
      stride *= srcShape[i];
    }
    return evaluateElement(broadcast.getSrc(), srcIndex, depth + 1);
  }
  if (isa<mlir::triton::ExpandDimsOp, mlir::triton::ReshapeOp,
          mlir::triton::SplatOp, arith::ExtSIOp>(op))
    return evaluateElement(op->getOperand(0), index, depth + 1);
  if (auto castOp = dyn_cast<arith::SIToFPOp>(op)) {
    auto input = evaluateElement(castOp.getIn(), index, depth + 1);
    // Only fold exactly representable mask values.  Forwarding arbitrary
    // integers would ignore the rounding performed by this conversion.
    if (input && *input >= -1 && *input <= 1)
      return input;
    return std::nullopt;
  }
  if (auto ext = dyn_cast<arith::ExtUIOp>(op)) {
    auto input = evaluateElement(ext.getIn(), index, depth + 1);
    unsigned bits =
        getElementTypeOrSelf(ext.getIn().getType()).getIntOrFloatBitWidth();
    if (input && bits < 64)
      return static_cast<int64_t>(APInt(bits, *input).getZExtValue());
    return std::nullopt;
  }
  if (auto sub = dyn_cast<arith::SubIOp>(op)) {
    auto lhs = evaluateElement(sub.getLhs(), index, depth + 1);
    auto rhs = evaluateElement(sub.getRhs(), index, depth + 1);
    auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
    if (!integer)
      return std::nullopt;
    unsigned bits = integer.getWidth();
    if (lhs && rhs && bits <= 64)
      return (APInt(bits, *lhs) - APInt(bits, *rhs)).getSExtValue();
  }
  return std::nullopt;
}

// Build partitioned expressions in a detached block. The value cache avoids
// rebuilding shared SSA subexpressions, and rejection leaves the original IR
// untouched. Actual candidate size bounds replace a separate cost model.
struct BroadcastRewrite {
  BroadcastRewrite(Operation *root, BroadcastInfo seed)
      : builder(root->getContext()), root(root), seed(seed),
        count(cast<RankedTensorType>(root->getResult(0).getType())
                  .getNumElements()) {
    builder.setInsertionPointToEnd(&staging);
  }

  // Sharing at least one operation is required; these limits bound generated
  // traversal and IR size, not device latency or a measured speedup.
  bool isSupported() {
    return hasSharedWork && parts.size() <= 32 &&
           staging.getOperations().size() <= 128;
  }

  Block staging;
  OpBuilder builder;

  // Return one narrow value per broadcast partition, recursively following
  // the producer chain. For broadcast(a)*broadcast(b)+x with factor=2, the
  // product becomes [p, p], x becomes [x0, x1], and the result becomes
  // [p+x0, p+x1]. Previously visited SSA values reuse their cached parts.
  SmallVector<Value> emitParts(Value value, unsigned depth = 0) {
    if (auto it = parts.find(value); it != parts.end())
      return it->second;
    if (parts.size() >= 32 || staging.getOperations().size() > 128)
      return {};
    auto remember = [&](SmallVector<Value> result) {
      parts.try_emplace(value, result);
      return result;
    };
    auto type = partType(value);
    // Scan at most 8192 result elements for partition-wise splat constants.
    // This caps compile-time evaluation; larger tensors still use the other
    // rewrite paths. Arbitrary dense constants are not materialized because
    // not every downstream target supports them.
    if (count <= 8192) {
      SmallVector<Attribute> constants;
      for (int64_t part = 0; part < seed.factor; ++part) {
        std::optional<int64_t> first;
        bool splat = true;
        for (int64_t i = 0; i < count / seed.factor; ++i) {
          int64_t index = (i / seed.stride) * seed.factor * seed.stride +
                          part * seed.stride + i % seed.stride;
          auto element = evaluateElement(value, index);
          if (!element || (first && first != element)) {
            splat = false;
            break;
          }
          first = element;
        }
        if (!splat)
          break;
        Attribute attr;
        if (auto fp = dyn_cast<FloatType>(type.getElementType()))
          attr = builder.getFloatAttr(fp, *first);
        else
          attr = builder.getIntegerAttr(type.getElementType(), *first);
        constants.push_back(attr);
      }
      if (constants.size() == static_cast<size_t>(seed.factor)) {
        SmallVector<Value> values;
        for (Attribute attr : constants)
          values.push_back(builder.create<arith::ConstantOp>(
              root->getLoc(), type, DenseElementsAttr::get(type, attr)));
        return remember(std::move(values));
      }
    }
    auto repeat = [&](Value result) {
      return remember(SmallVector<Value>(seed.factor, result));
    };
    if (auto info = getBroadcastInfo(value)) {
      if (info->factor == seed.factor && info->stride == seed.stride) {
        if (depth < 12) {
          // Recreate the original two-term add on narrow inputs, then reuse
          // its result in every partition. Input multiplies by zero remain.
          if (auto sum = getTwoTermSum(*info)) {
            auto inputs = emitParts(sum.getOperand(0), depth + 1);
            if (inputs.empty())
              return {};
            hasSharedWork = true;
            return repeat(clone(&sum.getCombineOp().front().front(), type,
                                {inputs[0], inputs[1]}));
          }
        }
        return repeat(reshape(info->op.getSrc(), type));
      }
    }
    Operation *def = value.getDefiningOp();
    if (auto constant = dyn_cast_or_null<arith::ConstantOp>(def)) {
      if (auto attr = dyn_cast<DenseElementsAttr>(constant.getValue()))
        if (attr.isSplat())
          return repeat(builder.create<arith::ConstantOp>(
              root->getLoc(), type,
              DenseElementsAttr::get(type, attr.getSplatValue<Attribute>())));
    }
    if (auto splat = dyn_cast_or_null<mlir::triton::SplatOp>(def))
      return repeat(builder.create<mlir::triton::SplatOp>(root->getLoc(), type,
                                                          splat.getSrc()));
    // An existing reshape user marks a producer boundary. Ignore temporary
    // users in staging so speculative reshapes cannot block their own walk.
    bool boundary =
        def != root && llvm::any_of(value.getUsers(), [&](Operation *user) {
          return user->getBlock() != &staging &&
                 isa<mlir::triton::ReshapeOp>(user);
        });
    if (def && depth < 12 && !boundary) {
      if (auto reshape = dyn_cast<mlir::triton::ReshapeOp>(def)) {
        auto result = emitParts(reshape.getSrc(), depth + 1);
        return result.empty() ? result : remember(std::move(result));
      }
      // Rebuild pure operations per partition only when all operands have
      // the same element count. Identical SSA operand tuples share one clone.
      if (isPointwise(def) && def->getNumOperands() > 0 &&
          llvm::all_of(def->getOperandTypes(), [&](Type type) {
            auto tensor = dyn_cast<RankedTensorType>(type);
            return tensor && tensor.hasStaticShape() &&
                   tensor.getNumElements() == count;
          })) {
        SmallVector<SmallVector<Value>> inputs;
        for (Value input : def->getOperands()) {
          auto result = emitParts(input, depth + 1);
          if (result.empty())
            return {};
          inputs.push_back(std::move(result));
        }
        SmallVector<Value> result, previous;
        bool uniform = true;
        for (int64_t i = 0; i < seed.factor; ++i) {
          SmallVector<Value> operands;
          for (auto &input : inputs)
            operands.push_back(input[i]);
          if (i && operands == previous)
            result.push_back(result.back());
          else {
            result.push_back(clone(def, type, operands));
            uniform &= i == 0;
          }
          previous = std::move(operands);
        }
        hasSharedWork |= uniform;
        return remember(std::move(result));
      }
    }
    // If a value cannot be decomposed further, preserve its values by
    // reshaping, transposing and splitting along the selected axis.
    auto element = type.getElementType();
    auto matrixType = RankedTensorType::get(
        {count / (seed.factor * seed.stride), seed.factor, seed.stride},
        element);
    auto matrix = reshape(value, matrixType);
    auto trans = builder.create<mlir::triton::TransOp>(
        root->getLoc(), matrix, SmallVector<int32_t>{0, 2, 1});
    SmallVector<int64_t> binaryShape{count / seed.factor};
    for (int64_t k = seed.factor; k > 1; k /= 2)
      binaryShape.push_back(2);
    SmallVector<Value> result{
        reshape(trans, RankedTensorType::get(binaryShape, element))};
    while (result.size() < static_cast<size_t>(seed.factor)) {
      SmallVector<Value> left, right;
      for (Value input : result) {
        auto split =
            builder.create<mlir::triton::SplitOp>(root->getLoc(), input);
        left.push_back(split.getOutLHS());
        right.push_back(split.getOutRHS());
      }
      // Split the least significant axis first, but return natural part order.
      llvm::append_range(left, right);
      result = std::move(left);
    }
    return remember(std::move(result));
  }

private:
  RankedTensorType partType(Value value) const {
    return RankedTensorType::get(
        {count / seed.factor},
        cast<RankedTensorType>(value.getType()).getElementType());
  }
  Value reshape(Value value, RankedTensorType type) {
    if (value.getType() == type)
      return value;
    return builder.create<mlir::triton::ReshapeOp>(root->getLoc(), type, value,
                                                   false);
  }
  // Preserve the original operation's attributes and properties when cloning
  // it at the narrower partition type.
  Value clone(Operation *op, RankedTensorType type, ValueRange operands) {
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(type);
    state.addAttributes(llvm::to_vector(op->getDiscardableAttrs()));
    state.propertiesAttr = op->getPropertiesAsAttribute();
    return builder.create(state)->getResult(0);
  }
  Operation *root;
  BroadcastInfo seed;
  int64_t count;
  bool hasSharedWork = false;
  DenseMap<Value, SmallVector<Value>> parts;
};

// Start at a pure expression with a non-pointwise consumer, then search its
// operand graph for a usable broadcast. Mixed consumers can split their
// unrelated full-size inputs while sharing work from the broadcast branch.
struct HoistBroadcastConsumers : RewritePattern {
  explicit HoistBroadcastConsumers(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isPointwise(op) || op->use_empty())
      return failure();
    auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!type || !type.hasStaticShape() || type.getEncoding() ||
        type.getNumElements() == 0 ||
        !llvm::any_of(op->getUsers(),
                      [](Operation *user) { return !isPointwise(user); }))
      return failure();
    int64_t count = type.getNumElements();
    SmallVector<Value> pending(op->getOperands());
    DenseSet<Value> visited;
    SmallVector<BroadcastInfo> seeds;
    // Search only a bounded pointwise producer graph. Try each distinct
    // (factor, stride) direction; a rejected seed need not block a later one.
    for (unsigned i = 0; i < pending.size() && i < 64; ++i) {
      Value value = pending[i];
      if (!visited.insert(value).second)
        continue;
      auto operandType = dyn_cast<RankedTensorType>(value.getType());
      if (!operandType || !operandType.hasStaticShape() ||
          operandType.getNumElements() != count)
        continue;
      if (auto info = getBroadcastInfo(value)) {
        if (llvm::none_of(seeds, [&](BroadcastInfo existing) {
              return existing.factor == info->factor &&
                     existing.stride == info->stride;
            }))
          seeds.push_back(*info);
        continue;
      }
      Operation *def = value.getDefiningOp();
      if (isPointwise(def) &&
          !llvm::any_of(value.getUsers(), [](Operation *user) {
            return isa<mlir::triton::ReshapeOp>(user);
          }))
        llvm::append_range(pending, def->getOperands());
    }
    for (BroadcastInfo seed : seeds)
      if (succeeded(rewriteForSeed(op, type, seed, rewriter)))
        return success();
    return failure();
  }

private:
  // Restore the original shape and order: broadcast a uniform result, or
  // join and transpose distinct partitions. Commit only after the final size
  // check includes these restoration operations.
  LogicalResult rewriteForSeed(Operation *op, RankedTensorType type,
                               BroadcastInfo seed,
                               PatternRewriter &rewriter) const {
    int64_t count = type.getNumElements();
    BroadcastRewrite candidate(op, seed);
    auto parts = candidate.emitParts(op->getResult(0));
    if (parts.empty() || !candidate.isSupported())
      return failure();
    auto &builder = candidate.builder;
    Value result;
    if (llvm::all_of(parts,
                     [&](Value value) { return value == parts.front(); })) {
      auto srcType = RankedTensorType::get(
          seed.op.getSrc().getType().getShape(), type.getElementType());
      result = builder.create<mlir::triton::ReshapeOp>(op->getLoc(), srcType,
                                                       parts.front(), false);
      auto dstType = RankedTensorType::get(seed.op.getType().getShape(),
                                           type.getElementType());
      result = builder.create<mlir::triton::BroadcastOp>(op->getLoc(), dstType,
                                                         result);
    } else {
      while (parts.size() > 1) {
        SmallVector<Value> next;
        for (size_t i = 0, half = parts.size() / 2; i < half; ++i)
          next.push_back(builder.create<mlir::triton::JoinOp>(
              op->getLoc(), parts[i], parts[i + half]));
        parts = std::move(next);
      }
      auto matrixType = RankedTensorType::get(
          {count / (seed.factor * seed.stride), seed.stride, seed.factor},
          type.getElementType());
      auto matrix = builder.create<mlir::triton::ReshapeOp>(
          op->getLoc(), matrixType, parts.front(), false);
      result = builder.create<mlir::triton::TransOp>(
          op->getLoc(), matrix, SmallVector<int32_t>{0, 2, 1});
    }
    result = builder.create<mlir::triton::ReshapeOp>(op->getLoc(), type, result,
                                                     false);
    if (!candidate.isSupported())
      return failure();
    while (!candidate.staging.empty())
      rewriter.moveOpBefore(&candidate.staging.front(), op);
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Cancel a repeated XOR operand through a conditional update:
// x ^ select(c, x ^ y, 0) -> select(c, y, x).
// The inner and outer XOR must use the same base SSA value; the inactive arm
// must be zero. A single-use select has no other consumers to preserve.
struct FoldConditionalXor : OpRewritePattern<arith::XOrIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::XOrIOp op,
                                PatternRewriter &rewriter) const override {
    for (unsigned i = 0; i < 2; ++i) {
      Value base = op->getOperand(1 - i);
      auto select = op->getOperand(i).getDefiningOp<arith::SelectOp>();
      if (!select || !select->hasOneUse())
        continue;
      for (unsigned branch = 0; branch < 2; ++branch) {
        Value active = select->getOperand(1 + branch);
        Value inactive = select->getOperand(2 - branch);
        auto update = active.getDefiningOp<arith::XOrIOp>();
        if (!update || !matchPattern(inactive, m_Zero()))
          continue;
        Value other;
        if (update.getLhs() == base)
          other = update.getRhs();
        else if (update.getRhs() == base)
          other = update.getLhs();
        if (!other)
          continue;
        rewriter.replaceOpWithNewOp<arith::SelectOp>(
            op, select.getCondition(), branch == 0 ? other : base,
            branch == 0 ? base : other);
        return success();
      }
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct OptimizeSIMTExpressionsPass
    : public impl::OptimizeSIMTExpressionsBase<OptimizeSIMTExpressionsPass> {
  void runOnOperation() override {
    auto fn = getOperation();
    LLVM_DEBUG(DBGS() << "running on " << fn.getName() << '\n');

    // Do not let the driver CSE constants across enclosing regions: that can
    // move constants out of nested regions even when no pattern matches, which
    // is unrelated to the rewrites this pass performs.
    GreedyRewriteConfig config;
    config.cseConstants = false;

    // Greedy rewriting lets one pattern expose opportunities for the other.
    RewritePatternSet patterns(&getContext());
    patterns.add<HoistBroadcastConsumers>(&getContext());
    patterns.add<FoldConditionalXor>(&getContext());
    (void)applyPatternsGreedily(fn, std::move(patterns), config);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createOptimizeSIMTExpressionsPass() {
  return std::make_unique<OptimizeSIMTExpressionsPass>();
}

} // namespace triton
} // namespace bishengir

//===- PromoteL1Gather.cpp - Promote hinted gathers to L1 -----------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadCapabilityQuery.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
#define GEN_PASS_DEF_PROMOTEL1GATHER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

// Keep only the operations needed to rewrite a gather. Shape and row-view
// information are obtained from these operations rather than cached separately.
struct Gather {
  bufferization::ToTensorOp view;
  scf::ForOp loop;
  LoadOp load;
  VBrcOp fill;

  MemRefType getType() { return cast<MemRefType>(view.getMemref().getType()); }
};

// Removing the payload select leaves this mask conversion with only a later
// score broadcast user. Keep it beside its predicate so VF fusion does not
// spill i1 between vector functions.
void hoistMaskMaterializations(scf::ForOp loop, int64_t rows,
                               PatternRewriter &rewriter) {
  Block &block = *loop->getBlock();
  DominanceInfo dominance;
  for (auto select : llvm::make_early_inc_range(block.getOps<VSelOp>())) {
    Value predicate = select.getSrc()[0];
    auto type = dyn_cast<RankedTensorType>(predicate.getType());
    Operation *producer = predicate.getDefiningOp();
    if (!type || type.getRank() != 1 || type.getDimSize(0) != rows ||
        !producer || producer->getBlock() != &block ||
        !producer->isBeforeInBlock(loop) || !loop->isBeforeInBlock(select) ||
        !select.hasPureTensorSemantics() ||
        isa<ShapedType>(select.getSrc()[1].getType()) ||
        isa<ShapedType>(select.getSrc()[2].getType()) ||
        !dominance.dominates(select.getSrc()[1], producer) ||
        !dominance.dominates(select.getSrc()[2], producer))
      continue;
    auto empty = select.getDst()[0].getDefiningOp<tensor::EmptyOp>();
    if (!empty || empty->getNumOperands())
      continue;
    if (empty->getBlock() == &block && producer->isBeforeInBlock(empty))
      rewriter.moveOpBefore(empty, producer);
    rewriter.moveOpAfter(select, producer);
  }
}

FailureOr<Gather> matchGather(Value operand, DataLayoutAttr layout) {
  bool enabled = false;
  bool hasMatmul = false;
  for (Operation *user : operand.getUsers()) {
    if (auto hint = dyn_cast<annotation::MarkOp>(user)) {
      auto attr = hint->getAttrOfType<BoolAttr>(EnableL1GatherAttr::name);
      enabled |= attr && attr.getValue();
    } else if (isa<MmadL1Op>(user)) {
      hasMatmul = true;
    } else {
      return failure();
    }
  }
  auto view = operand.getDefiningOp<bufferization::ToTensorOp>();
  auto alloc = view ? view.getMemref().getDefiningOp<memref::AllocOp>()
                    : memref::AllocOp{};
  if (!enabled || !hasMatmul || !alloc || alloc.getType().getRank() != 2 ||
      !alloc.getType().hasStaticShape() ||
      !alloc.getType().getLayout().isIdentity())
    return failure();
  memref::SubViewOp row;
  VBrcOp fill;
  for (Operation *user : alloc->getUsers()) {
    if (user == view)
      continue;
    if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
      if (row)
        return failure();
      row = subview;
    } else if (auto broadcast = dyn_cast<VBrcOp>(user)) {
      if (fill || broadcast.getDst() != alloc ||
          (!matchPattern(broadcast.getSrc(), m_Zero()) &&
           !matchPattern(broadcast.getSrc(), m_PosZeroFloat())))
        return failure();
      fill = broadcast;
    } else {
      return failure();
    }
  }
  auto loop = row ? dyn_cast<scf::ForOp>(row->getParentOp()) : scf::ForOp{};
  int64_t rows = alloc.getType().getDimSize(0);
  int64_t columns = alloc.getType().getDimSize(1);
  if (!loop || !loop->hasAttr(ParallelLoopAttr::name) || loop.getNumResults() ||
      getConstantIntValue(loop.getLowerBound()) != 0 ||
      getConstantIntValue(loop.getStep()) != 1 ||
      (!fill && getConstantIntValue(loop.getUpperBound()) != rows) ||
      row.getMixedOffsets()[0].dyn_cast<Value>() != loop.getInductionVar() ||
      getConstantIntValue(row.getMixedOffsets()[1]) != 0 ||
      row.getStaticSizes() != ArrayRef<int64_t>{1, columns} ||
      row.getStaticStrides() != ArrayRef<int64_t>{1, 1} || !row->hasOneUse())
    return failure();
  auto load = dyn_cast<LoadOp>(*row->getUsers().begin());
  if (!load || load.getDst() != row || load.getInitOutBuffer() ||
      (load.getLeftPaddingNum() &&
       getConstantIntValue(load.getLeftPaddingNum()) != 0) ||
      (load.getRightPaddingNum() &&
       getConstantIntValue(load.getRightPaddingNum()) != 0) ||
      load.getMayImplicitTransposeWithLastAxis())
    return failure();
  // Normalization may retain a pad value even for a complete row. It has no
  // effect when the source covers the destination and both paddings are zero.
  auto sourceType = dyn_cast<MemRefType>(load.getSrc().getType());
  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset;
  if (!sourceType || sourceType.getShape() != ArrayRef<int64_t>{1, columns} ||
      failed(getStridesAndOffset(sourceType, sourceStrides, sourceOffset)) ||
      sourceStrides.back() != 1)
    return failure();
  if (alloc->getBlock() != loop->getBlock() ||
      view->getBlock() != loop->getBlock() || !alloc->isBeforeInBlock(loop) ||
      !loop->isBeforeInBlock(view) ||
      (fill &&
       (fill->getBlock() != loop->getBlock() || !fill->isBeforeInBlock(loop))))
    return failure();
  // Preserve the loop's scalar address computations and control flow. The
  // only write may be the payload load into its private row view.
  WalkResult result = loop.walk([&](Operation *op) -> WalkResult {
    if (op == loop || op == load || op == row ||
        isa<scf::IfOp, scf::YieldOp>(op))
      return WalkResult::advance();
    auto effects = getEffectsRecursively(op);
    return failure(
        !effects ||
        llvm::any_of(*effects,
                     [](const auto &effect) {
                       return !isa<MemoryEffects::Read>(effect.getEffect());
                     }) ||
        llvm::any_of(
            llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()),
            [](Type type) { return isa<TensorType>(type); }));
  });
  if (result.wasInterrupted() || (load->getBlock() != loop.getBody() && !fill))
    return failure();

  auto blocks = layout.getFractalBlockSizes();
  // ND2NZ encodes row and column counts in 16-bit descriptor fields.
  if (failed(blocks) || rows <= 0 || columns <= 0 || rows % blocks->first ||
      columns % blocks->second || !llvm::isUInt<16>(rows) ||
      !llvm::isUInt<16>(columns))
    return failure();
  for (OpOperand &use : operand.getUses()) {
    auto user = dyn_cast<MmadL1Op>(use.getOwner());
    if (!user)
      continue;
    auto target = user.getOperandsTargetFractalLayout();
    if ((use.getOperandNumber() == 0 ? target.a : target.b) != layout)
      return failure();
  }
  return Gather{view, loop, load, fill};
}

// Compare scalar expressions without interpreting their arithmetic. Region
// equivalence also handles guarded metadata reads in the two row loops.
class RowExpressions {
public:
  RowExpressions(scf::ForOp lhs, scf::ForOp rhs) : lhs(lhs), rhs(rhs) {
    equivalent.map(lhs.getInductionVar(), rhs.getInductionVar());
  }

  bool same(Value a, Value b) {
    if (a == b || equivalent.lookupOrNull(a) == b)
      return true;
    Operation *left = a.getDefiningOp(), *right = b.getDefiningOp();
    if (!left || !right || !lhs->isProperAncestor(left) ||
        !rhs->isProperAncestor(right) ||
        cast<OpResult>(a).getResultNumber() !=
            cast<OpResult>(b).getResultNumber())
      return false;
    return OperationEquivalence::isEquivalentTo(
        left, right, [&](Value x, Value y) { return success(same(x, y)); },
        [&](Value x, Value y) { equivalent.map(x, y); },
        OperationEquivalence::IgnoreLocations);
  }

  FailureOr<int64_t> columnDistance(Value a, Value b) {
    Builder builder(a.getContext());
    auto delta = AffineMap::get(
        2, 0, builder.getAffineDimExpr(0) - builder.getAffineDimExpr(1));
    return ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::EQ,
        ValueBoundsConstraintSet::Variable(delta, ArrayRef<Value>{b, a}),
        [&](Value value, std::optional<int64_t> dim,
            ValueBoundsConstraintSet &bounds) {
          if (dim)
            return false;
          if (value == a)
            return true;
          if (!same(a, value))
            return false;
          bounds.bound(value) == a;
          return true;
        });
  }

  bool sameGuards(Gather a, Gather b) {
    Operation *left = a.load, *right = b.load;
    while (left->getParentOp() != lhs) {
      auto leftIf = dyn_cast<scf::IfOp>(left->getParentOp());
      auto rightIf = dyn_cast<scf::IfOp>(right->getParentOp());
      if (!leftIf || !rightIf ||
          (left->getParentRegion() == &leftIf.getThenRegion()) !=
              (right->getParentRegion() == &rightIf.getThenRegion()) ||
          !same(leftIf.getCondition(), rightIf.getCondition()))
        return false;
      left = leftIf;
      right = rightIf;
    }
    return right->getParentOp() == rhs;
  }

private:
  scf::ForOp lhs, rhs;
  IRMapping equivalent;
};

// Moving a gather also moves its metadata reads. Check every read, not just
// the BF16 payload, against intervening writes and frees.
bool canMoveBefore(Gather gather, scf::ForOp before, AliasAnalysis &aliases) {
  SmallVector<Value> reads;
  auto readEffects = getEffectsRecursively(gather.loop);
  if (!readEffects)
    return false;
  for (auto &effect : *readEffects)
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      if (!effect.getValue())
        return false;
      reads.push_back(effect.getValue());
    }
  for (Operation *op = before; op != gather.loop; op = op->getNextNode()) {
    if (isa<annotation::MarkOp>(op) && op->hasAttr(EnableL1GatherAttr::name))
      continue;
    auto effects = getEffectsRecursively(op);
    if (!effects || !llvm::all_of(*effects, [&](const auto &effect) {
          return isa<MemoryEffects::Read, MemoryEffects::Allocate>(
                     effect.getEffect()) ||
                 (effect.getValue() && llvm::all_of(reads, [&](Value read) {
                    return aliases.alias(effect.getValue(), read).isNo();
                  }));
        }))
      return false;
  }
  return true;
}

// The MLA kernels load adjacent column tiles with the same row addresses and
// guards. Read their contiguous union once and give each matmul an L1 view.
SmallVector<Gather> adjacentGathers(Gather first, DataLayoutAttr layout) {
  SmallVector<Gather> group{first};
  auto source = first.load.getSrc().getDefiningOp<memref::ReinterpretCastOp>();
  if (!source)
    return group;
  Value offset = dyn_cast<Value>(source.getMixedOffsets().front());
  if (!offset)
    return group;
  int64_t columns = first.getType().getDimSize(1);
  AliasAnalysis aliases(first.loop->getParentOfType<func::FuncOp>());
  for (Operation *op = first.view->getNextNode(); op; op = op->getNextNode()) {
    auto view = dyn_cast<bufferization::ToTensorOp>(op);
    if (!view)
      continue;
    auto next = matchGather(view, layout);
    if (failed(next) || next->loop->getBlock() != first.loop->getBlock() ||
        !first.loop->isBeforeInBlock(next->loop) ||
        next->loop.getUpperBound() != first.loop.getUpperBound() ||
        next->getType().getDimSize(0) != first.getType().getDimSize(0) ||
        next->getType().getElementType() != first.getType().getElementType() ||
        static_cast<bool>(next->fill) != static_cast<bool>(first.fill) ||
        next->load.getEvictionPolicyAttr() !=
            first.load.getEvictionPolicyAttr())
      continue;
    auto nextSource =
        next->load.getSrc().getDefiningOp<memref::ReinterpretCastOp>();
    if (!nextSource || nextSource.getSource() != source.getSource())
      continue;
    Value nextOffset = dyn_cast<Value>(nextSource.getMixedOffsets().front());
    if (!nextOffset)
      continue;
    RowExpressions expressions(first.loop, next->loop);
    auto distance = expressions.columnDistance(offset, nextOffset);
    if (failed(distance) || *distance != columns ||
        !llvm::isUInt<16>(columns + next->getType().getDimSize(1)) ||
        !expressions.sameGuards(first, *next) ||
        !canMoveBefore(*next, first.loop, aliases))
      continue;
    group.push_back(*next);
    columns += next->getType().getDimSize(1);
  }
  return group;
}

struct PromoteL1Gather : OpRewritePattern<MmadL1Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MmadL1Op matmul,
                                PatternRewriter &rewriter) const override {
    for (Value operand : {matmul.getA(), matmul.getB()}) {
      if (!llvm::any_of(operand.getUsers(), [](Operation *user) {
            auto enabled =
                user->getAttrOfType<BoolAttr>(EnableL1GatherAttr::name);
            return isa<annotation::MarkOp>(user) && enabled &&
                   enabled.getValue();
          }))
        continue;
      auto layouts = matmul.getOperandsTargetFractalLayout();
      auto layout = cast<DataLayoutAttr>(operand == matmul.getA() ? layouts.a
                                                                  : layouts.b);
      auto match = matchGather(operand, layout);
      if (failed(match))
        continue;
      Type elementType = match->getType().getElementType();
      auto request =
          bishengir::npu::cubeload::Nd2NzCapabilityRequestBuilder::create(
              elementType, AddressSpace::GM, elementType, AddressSpace::L1,
              false);
      if (!isNativeCubeLoadRequest(matmul, request.build()))
        continue;
      auto group = adjacentGathers(*match, layout);
      int64_t rows = match->getType().getDimSize(0), columns = 0;
      for (auto gather : group)
        columns += gather.getType().getDimSize(1);
      Location loc = match->loop.getLoc();
      auto nd =
          DataLayoutAttr::get(rewriter.getContext(), hivm::DataLayout::ND);
      SmallVector<OpFoldResult> shape{rewriter.getIndexAttr(rows),
                                      rewriter.getIndexAttr(columns)};
      auto fractal =
          computeMixedTargetLayoutShape(shape, nd, layout, rewriter, loc);
      if (failed(fractal))
        continue;
      auto blocks = *layout.getFractalBlockSizes();
      rewriter.setInsertionPoint(match->loop);
      auto bufferType = MemRefType::get(
          decomposeMixedValues(*fractal).first, elementType,
          MemRefLayoutAttrInterface{},
          AddressSpaceAttr::get(rewriter.getContext(), AddressSpace::L1));
      Value buffer = rewriter.create<memref::AllocOp>(loc, bufferType);
      if (match->fill) {
        OpBuilder::InsertionGuard insertion(rewriter);
        // An unconditional row load overwrites the entire prefix. Only a
        // partial tile needs initialization; masked rows still always do.
        if (match->load->getBlock() == match->loop.getBody()) {
          Value count = rewriter.create<arith::ConstantIndexOp>(loc, rows);
          Value partial = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ne, match->loop.getUpperBound(),
              count);
          auto guard = rewriter.create<scf::IfOp>(loc, partial, false);
          rewriter.setInsertionPointToStart(guard.thenBlock());
        }
        rewriter.create<VBrcOp>(loc, TypeRange{}, match->fill.getSrc(), buffer,
                                rewriter.getDenseI64ArrayAttr({}));
      }
      auto loop = match->loop;
      auto load = match->load;
      auto row = load.getDst().getDefiningOp<memref::SubViewOp>();
      // This is now a native cube DMA loop, not an extracted UB gather.
      rewriter.modifyOpInPlace(loop,
                               [&] { loop->removeAttr(ExtractLoadStoreAttr); });
      rewriter.setInsertionPoint(row);
      SmallVector<OpFoldResult> offsets{loop.getInductionVar(),
                                        rewriter.getIndexAttr(0)};
      auto fractalOffsets =
          computeTargetLayoutOffset(offsets, nd, layout, rewriter, loc);
      assert(succeeded(fractalOffsets));
      SmallVector<OpFoldResult> sizes{
          rewriter.getIndexAttr(columns / blocks.second),
          rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
          rewriter.getIndexAttr(blocks.second)};
      SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
      Value destination = rewriter.create<memref::SubViewOp>(
          loc, buffer, *fractalOffsets, sizes, strides);
      rewriter.setInsertionPoint(load);
      Value source = load.getSrc();
      if (group.size() > 1) {
        auto view = source.getDefiningOp<memref::ReinterpretCastOp>();
        auto type = MemRefType::get(
            {1, columns}, elementType,
            StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                                   {columns, 1}),
            view.getType().getMemorySpace());
        source = rewriter.create<memref::ReinterpretCastOp>(
            loc, type, view.getSource(), view.getMixedOffsets().front(),
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                                   rewriter.getIndexAttr(columns)},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(columns),
                                   rewriter.getIndexAttr(1)});
      }
      rewriter.create<ND2NZOp>(loc, TypeRange{}, source, destination,
                               rewriter.getUnitAttr());
      Operation *oldSource = load.getSrc().getDefiningOp();
      rewriter.eraseOp(load);
      rewriter.eraseOp(row);
      // Its dead address use would otherwise prevent the sinking below.
      if (oldSource && isOpTriviallyDead(oldSource))
        rewriter.eraseOp(oldSource);
      // Address calculations used only by a guarded DMA belong in its branch.
      // Use the standard sinking utility rather than matching arithmetic ops.
      DominanceInfo dominance;
      loop.walk([&](scf::IfOp guard) {
        SmallVector<Region *> regions;
        getSinglyExecutedRegionsToSink(guard, regions);
        controlFlowSink(
            regions, dominance,
            [&](Operation *op, Region *) {
              return op->getBlock() == loop.getBody() && isMemoryEffectFree(op);
            },
            [&](Operation *op, Region *region) {
              rewriter.moveOpBefore(op, &region->front(),
                                    region->front().begin());
            });
      });
      int64_t column = 0;
      Operation *lastTile = loop;
      for (auto gather : group) {
        rewriter.setInsertionPointAfter(lastTile);
        Value tile = buffer;
        if (group.size() > 1) {
          SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0),
                                            rewriter.getIndexAttr(column)};
          auto tileOffsets =
              computeTargetLayoutOffset(offsets, nd, layout, rewriter, loc);
          SmallVector<OpFoldResult> tileShape{
              rewriter.getIndexAttr(rows),
              rewriter.getIndexAttr(gather.getType().getDimSize(1))};
          auto tileSizes = computeMixedTargetLayoutShape(tileShape, nd, layout,
                                                         rewriter, loc);
          tile = rewriter.create<memref::SubViewOp>(loc, buffer, *tileOffsets,
                                                    *tileSizes, strides);
        }
        Value result = rewriter.create<bufferization::ToTensorOp>(
            loc, tile, /*restrict=*/true, /*writable=*/true);
        lastTile = result.getDefiningOp();
        for (auto hint :
             llvm::make_early_inc_range(gather.view.getResult().getUsers()))
          if (isa<annotation::MarkOp>(hint) &&
              hint->hasAttr(EnableL1GatherAttr::name))
            rewriter.eraseOp(hint);
        column += gather.getType().getDimSize(1);
        auto alloc = gather.view.getMemref().getDefiningOp<memref::AllocOp>();
        rewriter.replaceOp(gather.view, result);
        if (gather.loop != loop)
          rewriter.eraseOp(gather.loop);
        if (gather.fill)
          rewriter.eraseOp(gather.fill);
        rewriter.eraseOp(alloc);
      }
      hoistMaskMaterializations(loop, rows, rewriter);
      return success();
    }
    return failure();
  }
};

struct PromoteL1GatherPass : impl::PromoteL1GatherBase<PromoteL1GatherPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    PromoteL1GatherBase::getDependentDialects(registry);
    arith::registerValueBoundsOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PromoteL1Gather>(&getContext());
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createPromoteL1GatherPass() {
  return std::make_unique<PromoteL1GatherPass>();
}

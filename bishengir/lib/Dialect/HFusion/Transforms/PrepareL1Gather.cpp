//===- PrepareL1Gather.cpp - Normalize hinted row gathers -----------------===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
#define GEN_PASS_DEF_PREPAREL1GATHER
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

bool isZero(Value value) {
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    value = fill.getInputs().front();
  return matchPattern(value, m_Zero()) || matchPattern(value, m_PosZeroFloat());
}

// Scalarize before HFusion legalization, while arithmetic has its original
// scalarizable semantics and linalg operations still have scalar regions.
// In particular, do not maintain an opcode list for index or mask arithmetic.
class ScalarMetadata {
public:
  explicit ScalarMetadata(scf::ForOp loop)
      : loop(loop), aliases(loop->getParentOfType<func::FuncOp>()) {}

  bool canRead(Value value) {
    if (!isa<ShapedType>(value.getType()))
      return dominance.dominates(value, loop);
    if (supported.contains(value))
      return true;
    auto type = dyn_cast<RankedTensorType>(value.getType());
    Operation *op = value.getDefiningOp();
    if (!type || !op || !dominance.dominates(value, loop))
      return false;
    if (auto view = dyn_cast<bufferization::ToTensorOp>(op)) {
      if (!findLoad(view))
        return false;
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto elements = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!elements || !elements.isSplat())
        return false;
    } else if (isa<tensor::ExpandShapeOp, tensor::ReshapeOp>(op)) {
      auto sourceType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!type.hasStaticShape() || !sourceType ||
          !sourceType.hasStaticShape() || type.getNumElements() == 0 ||
          !canRead(op->getOperand(0)))
        return false;
    } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (linalgOp.getNumDpsInits() != 1 ||
          linalgOp.getNumParallelLoops() != linalgOp.getNumLoops() ||
          !linalgOp.getIndexingMapsArray().back().isIdentity())
        return false;
      Block *body = linalgOp.getBlock();
      for (auto [operand, argument] :
           llvm::zip(op->getOperands(), body->getArguments()))
        if (!argument.use_empty() && !canRead(operand))
          return false;
      for (Operation &nested : body->without_terminator())
        if (!isMemoryEffectFree(&nested) || nested.getNumRegions())
          return false;
    } else if (op->hasTrait<OpTrait::Scalarizable>() &&
               op->getNumResults() == 1 && !op->getNumRegions() &&
               isMemoryEffectFree(op)) {
      if (!llvm::all_of(op->getOperands(),
                        [&](Value operand) { return canRead(operand); }))
        return false;
    } else {
      return false;
    }
    supported.insert(value);
    return true;
  }

  Value read(IRRewriter &rewriter, Value value, ValueRange indices) {
    if (!isa<ShapedType>(value.getType()))
      return value;
    for (auto &entry : cache[value])
      if (llvm::equal(entry.first, indices) &&
          dominance.dominates(entry.second, &*rewriter.getInsertionPoint()))
        return entry.second;
    Operation *op = value.getDefiningOp();
    Location loc = op->getLoc();
    Value scalar;
    if (auto [load, padding] = loads.lookup(value); load) {
      Value count = rewriter.create<memref::DimOp>(loc, load.getSource(), 0);
      Value valid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, indices.front(), count);
      auto guard = rewriter.create<scf::IfOp>(
          loc, TypeRange{getElementTypeOrSelf(value)}, valid, true);
      OpBuilder::InsertionGuard insertion(rewriter);
      rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
      Value loaded =
          rewriter.create<memref::LoadOp>(loc, load.getSource(), indices);
      rewriter.create<scf::YieldOp>(loc, loaded);
      rewriter.setInsertionPointToStart(&guard.getElseRegion().front());
      rewriter.create<scf::YieldOp>(loc, padding);
      scalar = guard.getResult(0);
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      scalar = rewriter.create<arith::ConstantOp>(
          loc, cast<TypedAttr>(cast<DenseElementsAttr>(constant.getValue())
                                   .getSplatValue<Attribute>()));
    } else if (isa<tensor::ExpandShapeOp, tensor::ReshapeOp>(op)) {
      Value source = op->getOperand(0);
      auto sourceType = cast<RankedTensorType>(source.getType());
      auto type = cast<RankedTensorType>(value.getType());
      auto dims = AffineMap::getMultiDimIdentityMap(type.getRank(),
                                                    rewriter.getContext());
      auto linear = linearize(rewriter.getContext(), dims.getResults(),
                              computeStrides(type.getShape()));
      auto map = AffineMap::get(
          type.getRank(), 0,
          delinearize(linear, computeStrides(sourceType.getShape())),
          rewriter.getContext());
      auto sourceIndices = getValueOrCreateConstantIndexOp(
          rewriter, loc,
          affine::makeComposedFoldedMultiResultAffineApply(
              rewriter, loc, map, getAsOpFoldResult(indices)));
      scalar = read(rewriter, source, sourceIndices);
    } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      IRMapping mapping;
      Block *body = linalgOp.getBlock();
      auto maps = linalgOp.getIndexingMapsArray();
      for (auto [operand, argument, map] :
           llvm::zip(op->getOperands(), body->getArguments(), maps)) {
        if (argument.use_empty())
          continue;
        auto inputIndices = getValueOrCreateConstantIndexOp(
            rewriter, loc,
            affine::makeComposedFoldedMultiResultAffineApply(
                rewriter, loc, map, getAsOpFoldResult(indices)));
        mapping.map(argument, read(rewriter, operand, inputIndices));
      }
      for (Operation &nested : body->without_terminator()) {
        if (auto index = dyn_cast<linalg::IndexOp>(nested))
          mapping.map(index.getResult(), indices[index.getDim()]);
        else
          rewriter.clone(nested, mapping);
      }
      scalar = mapping.lookupOrDefault(body->getTerminator()->getOperand(0));
    } else {
      SmallVector<Value> operands;
      for (Value operand : op->getOperands())
        operands.push_back(read(rewriter, operand, indices));
      OperationState state(loc, op->getName());
      state.addOperands(operands);
      state.addTypes(getElementTypeOrSelf(value));
      state.addAttributes(op->getAttrs());
      scalar = rewriter.create(state)->getResult(0);
    }
    cache[value].emplace_back(SmallVector<Value>(indices), scalar);
    return scalar;
  }

private:
  bool findLoad(bufferization::ToTensorOp view) {
    auto alloc = view.getMemref().getDefiningOp<memref::AllocOp>();
    if (!alloc || alloc.getType().getRank() != 1)
      return false;
    memref::SubViewOp destination;
    linalg::FillOp fill;
    for (Operation *user : alloc->getUsers()) {
      if (user == view || isa<annotation::MarkOp>(user))
        continue;
      if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
        if (destination)
          return false;
        destination = subview;
      } else if (auto candidate = dyn_cast<linalg::FillOp>(user)) {
        if (fill)
          return false;
        fill = candidate;
      } else {
        return false;
      }
    }
    if (!destination || !destination->hasOneUse() || !fill ||
        !dominance.dominates(fill.getInputs().front(), loop) ||
        !destination.hasZeroOffset() || !destination.hasUnitStride())
      return false;
    auto copy = dyn_cast<memref::CopyOp>(*destination->getUsers().begin());
    if (!copy || copy.getTarget() != destination ||
        copy->getBlock() != loop->getBlock() || !copy->isBeforeInBlock(loop))
      return false;
    Operation *fillScope = copy->getBlock()->findAncestorOpInBlock(*fill);
    if (!fillScope || !fillScope->isBeforeInBlock(copy))
      return false;
    for (Operation *between = copy->getNextNode();
         between != loop->getNextNode(); between = between->getNextNode()) {
      // compile_hint records an optimization request, not a memory write.
      if (isa<annotation::MarkOp>(between) &&
          between->hasAttr(hivm::EnableL1GatherAttr::name))
        continue;
      auto effects = getEffectsRecursively(between);
      if (!effects || !llvm::all_of(*effects, [&](const auto &effect) {
            return isa<MemoryEffects::Read, MemoryEffects::Allocate>(
                       effect.getEffect()) ||
                   (effect.getValue() &&
                    aliases.alias(effect.getValue(), copy.getSource()).isNo());
          }))
        return false;
    }
    loads[view] = {copy, fill.getInputs().front()};
    return true;
  }

  scf::ForOp loop;
  AliasAnalysis aliases;
  DominanceInfo dominance;
  DenseSet<Value> supported;
  DenseMap<Value, std::pair<memref::CopyOp, Value>> loads;
  DenseMap<Value, SmallVector<std::pair<SmallVector<Value>, Value>>> cache;
};

bool isRowMask(Value value) {
  if (auto broadcast = value.getDefiningOp<linalg::BroadcastOp>())
    return broadcast.getDimensions() == ArrayRef<int64_t>{1};
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    return !isa<ShapedType>(fill.getInputs().front().getType());
  Operation *op = value.getDefiningOp();
  return op && op->hasTrait<OpTrait::Scalarizable>() &&
         llvm::all_of(op->getOperands(), isRowMask);
}

// Match the frontend's unit-column address suffix. The bounded index cast and
// sign extension preserve every column, and the final cast preserves addition
// modulo the target index width. Leave the row arithmetic uninterpreted.
Value getScalarGatherRowBase(memref::ReinterpretCastOp source,
                             scf::ForOp columns, int64_t count) {
  auto offset = dyn_cast<Value>(source.getMixedOffsets().front());
  auto indexCast = offset ? offset.getDefiningOp<arith::IndexCastOp>()
                          : arith::IndexCastOp{};
  auto add = indexCast ? indexCast.getIn().getDefiningOp<arith::AddIOp>()
                       : arith::AddIOp{};
  auto type =
      add ? dyn_cast<IntegerType>(add.getResult().getType()) : IntegerType{};
  if (!type || type.getWidth() <
                   DataLayout::closest(source)
                       .getTypeSizeInBits(columns.getInductionVar().getType())
                       .getFixedValue())
    return {};
  for (auto [base, column] : {std::pair(add.getLhs(), add.getRhs()),
                              std::pair(add.getRhs(), add.getLhs())}) {
    auto extension = column.getDefiningOp<arith::ExtSIOp>();
    auto index = extension
                     ? extension.getIn().getDefiningOp<arith::IndexCastOp>()
                     : arith::IndexCastOp{};
    if (index && index.getIn() == columns.getInductionVar() &&
        llvm::isIntN(cast<IntegerType>(index.getResult().getType()).getWidth(),
                     count - 1))
      return base;
  }
  return {};
}

// Some frontend versions materialize K[column, row] with scalar tensor inserts
// instead of a row gather followed by linalg.transpose. Recover that same form
// only for the complete, explicitly hinted nest seen in sparse GQA.
bool prepareScalarTranspose(annotation::MarkOp hint, BoolAttr enabled,
                            IRRewriter &rewriter) {
  auto select = hint.getSrc().getDefiningOp<arith::SelectOp>();
  auto type = dyn_cast<RankedTensorType>(hint.getSrc().getType());
  if (!select || !type || type.getRank() != 2 || !type.hasStaticShape() ||
      type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0 ||
      !isZero(select.getFalseValue()))
    return false;
  auto mask = select.getCondition().getDefiningOp<linalg::BroadcastOp>();
  auto outer = select.getTrueValue().getDefiningOp<scf::ForOp>();
  if (!mask || mask.getDimensions() != ArrayRef<int64_t>{0} || !outer ||
      outer.getNumResults() != 1 || !outer->hasOneUse() ||
      !llvm::hasSingleElement(outer.getBody()->without_terminator()))
    return false;
  auto inner = dyn_cast<scf::ForOp>(outer.getBody()->front());
  if (!inner || inner.getNumResults() != 1 || !inner->hasOneUse() ||
      !outer.getInductionVar().getType().isIndex() ||
      !inner.getInductionVar().getType().isIndex() ||
      getConstantIntValue(outer.getLowerBound()) != 0 ||
      getConstantIntValue(inner.getLowerBound()) != 0 ||
      getConstantIntValue(outer.getStep()) != 1 ||
      getConstantIntValue(inner.getStep()) != 1 ||
      getConstantIntValue(outer.getUpperBound()) != type.getDimSize(0) ||
      getConstantIntValue(inner.getUpperBound()) != type.getDimSize(1) ||
      inner.getInitArgs().front() != outer.getRegionIterArgs().front() ||
      !outer.getRegionIterArgs().front().hasOneUse() ||
      !inner.getRegionIterArgs().front().hasOneUse() ||
      outer.getBody()->getTerminator()->getOperand(0) != inner.getResult(0))
    return false;
  auto insert = inner.getBody()
                    ->getTerminator()
                    ->getOperand(0)
                    .getDefiningOp<tensor::InsertOp>();
  if (!insert || insert->getBlock() != inner.getBody() ||
      !insert->hasOneUse() ||
      insert.getDest() != inner.getRegionIterArgs().front() ||
      !llvm::equal(insert.getIndices(), ValueRange{outer.getInductionVar(),
                                                   inner.getInductionVar()}))
    return false;
  auto load = insert.getScalar().getDefiningOp<memref::LoadOp>();
  auto source =
      load ? load.getMemRef().getDefiningOp<memref::ReinterpretCastOp>()
           : memref::ReinterpretCastOp{};
  DominanceInfo dominance;
  if (!load || load->getBlock() != inner.getBody() || !load->hasOneUse() ||
      load.getResult().getType() != type.getElementType() ||
      load.getIndices().size() != 1 ||
      getConstantIntValue(load.getIndices().front()) != 0 || !source ||
      source.getStaticSizes() != ArrayRef<int64_t>{1} ||
      source.getStaticStrides() != ArrayRef<int64_t>{1} ||
      !dominance.dominates(source.getSource(), outer))
    return false;
  Value base = getScalarGatherRowBase(source, outer, type.getDimSize(0));
  if (!base)
    return false;
  ScalarMetadata metadata(outer);
  if (!metadata.canRead(select.getCondition()))
    return false;
  for (Operation &op : inner.getBody()->without_terminator()) {
    if (&op == load || &op == insert)
      continue;
    if (!isMemoryEffectFree(&op) || op.getNumRegions())
      return false;
    if (auto extract = dyn_cast<tensor::ExtractOp>(op);
        extract && !metadata.canRead(extract.getTensor()))
      return false;
  }
  SetVector<Operation *> address;
  BackwardSliceOptions options;
  options.inclusive = true;
  options.omitBlockArguments = true;
  options.filter = [&](Operation *op) {
    return op->getBlock() == inner.getBody();
  };
  getBackwardSlice(base, &address, options);
  for (Operation *op : address)
    for (Value operand : op->getOperands())
      if (operand == outer.getInductionVar() ||
          operand == inner.getRegionIterArgs().front())
        return false;

  Location loc = outer.getLoc();
  rewriter.setInsertionPoint(outer);
  auto bufferType = MemRefType::get({type.getDimSize(1), type.getDimSize(0)},
                                    type.getElementType());
  Value buffer = rewriter.create<memref::AllocOp>(loc, bufferType);
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(type.getElementType()));
  rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{buffer});
  auto loop = rewriter.create<scf::ForOp>(
      loc, inner.getLowerBound(), inner.getUpperBound(), inner.getStep());
  loop->setAttrs(inner->getAttrs());
  loop->setAttr(hivm::ParallelLoopAttr::name, rewriter.getUnitAttr());
  rewriter.setInsertionPointToStart(loop.getBody());
  Value valid = metadata.read(rewriter, select.getCondition(),
                              {outer.getLowerBound(), loop.getInductionVar()});
  SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1),
                                  rewriter.getIndexAttr(type.getDimSize(0))};
  SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(type.getDimSize(0)),
                                    rewriter.getIndexAttr(1)};
  Value row = rewriter.create<memref::SubViewOp>(
      loc, buffer,
      ArrayRef<OpFoldResult>{loop.getInductionVar(), rewriter.getIndexAttr(0)},
      sizes,
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(1)});
  auto guard = rewriter.create<scf::IfOp>(loc, valid, false);
  rewriter.setInsertionPointToStart(guard.thenBlock());
  IRMapping mapping;
  mapping.map(inner.getInductionVar(), loop.getInductionVar());
  for (Operation *op : address) {
    if (auto extract = dyn_cast<tensor::ExtractOp>(op)) {
      SmallVector<Value> indices;
      for (Value index : extract.getIndices())
        indices.push_back(mapping.lookupOrDefault(index));
      mapping.map(extract.getResult(),
                  metadata.read(rewriter, extract.getTensor(), indices));
    } else {
      rewriter.clone(*op, mapping);
    }
  }
  Value offset = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), mapping.lookupOrDefault(base));
  auto rowType = MemRefType::get(
      {1, type.getDimSize(0)}, type.getElementType(),
      StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                             {type.getDimSize(0), 1}),
      source.getType().getMemorySpace());
  Value rowSource = rewriter.create<memref::ReinterpretCastOp>(
      loc, rowType, source.getSource(), offset, sizes, strides);
  rewriter.create<memref::CopyOp>(loc, rowSource, row);
  rewriter.setInsertionPointAfter(loop);
  Value rows = rewriter.create<bufferization::ToTensorOp>(
      loc, buffer, /*restrict=*/true, /*writable=*/true);
  rewriter.create<annotation::MarkOp>(hint.getLoc(), rows)
      ->setAttr(hivm::EnableL1GatherAttr::name, enabled);
  rewriter.setInsertionPoint(select);
  Value empty = rewriter.create<tensor::EmptyOp>(loc, type.getShape(),
                                                 type.getElementType());
  auto transpose = rewriter.create<linalg::TransposeOp>(
      loc, rows, empty, ArrayRef<int64_t>{1, 0});
  rewriter.replaceOp(select, transpose.getResults());
  rewriter.modifyOpInPlace(
      hint, [&] { hint->removeAttr(hivm::EnableL1GatherAttr::name); });
  if (hint.isAttrEmpty())
    rewriter.eraseOp(hint);
  rewriter.eraseOp(outer);
  return true;
}

void prepareGather(annotation::MarkOp hint, IRRewriter &rewriter) {
  auto enabled = hint->getAttrOfType<BoolAttr>(hivm::EnableL1GatherAttr::name);
  if (!enabled || !enabled.getValue())
    return;
  if (prepareScalarTranspose(hint, enabled, rewriter))
    return;
  Value result = hint.getSrc();
  auto transpose = result.getDefiningOp<linalg::TransposeOp>();
  if (transpose) {
    if (transpose.getPermutation() != ArrayRef<int64_t>{1, 0})
      return;
    result = transpose.getInput();
  }
  if (result.getDefiningOp<bufferization::ToTensorOp>())
    for (Operation *user : result.getUsers())
      if (auto select = dyn_cast<arith::SelectOp>(user);
          select && select.getTrueValue() == result) {
        result = select.getResult();
        break;
      }
  auto type = dyn_cast<RankedTensorType>(result.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape())
    return;
  Value source = result;
  Value mask;
  Value prefixRows;
  tensor::ExtractSliceOp prefix;
  if (auto select = source.getDefiningOp<arith::SelectOp>()) {
    if (!isZero(select.getFalseValue()) || !isRowMask(select.getCondition()))
      return;
    mask = select.getCondition();
    source = select.getTrueValue();
  } else if (auto insert = source.getDefiningOp<tensor::InsertSliceOp>()) {
    prefix = insert.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!prefix || !prefix->hasOneUse() || !isZero(insert.getDest()) ||
        prefix.getType().getRank() != 2 ||
        !prefix.isSameAs(
            insert, [](OpFoldResult a, OpFoldResult b) { return a == b; }) ||
        !prefix.hasZeroOffset() || !prefix.hasUnitStride() ||
        getConstantIntValue(prefix.getMixedSizes()[1]) != type.getDimSize(1))
      return;
    prefixRows = dyn_cast<Value>(prefix.getMixedSizes()[0]);
    if (!prefixRows)
      return;
    source = prefix.getSource();
  }
  auto view = source.getDefiningOp<bufferization::ToTensorOp>();
  auto alloc = view ? view.getMemref().getDefiningOp<memref::AllocOp>()
                    : memref::AllocOp{};
  if (!alloc || alloc.getType().getShape() != type.getShape() ||
      !alloc.getType().getLayout().isIdentity())
    return;
  memref::SubViewOp row;
  for (Operation *user : alloc->getUsers()) {
    if (user == view || isa<annotation::MarkOp>(user))
      continue;
    auto subview = dyn_cast<memref::SubViewOp>(user);
    if (!subview || row)
      return;
    row = subview;
  }
  auto loop = row ? dyn_cast<scf::ForOp>(row->getParentOp()) : scf::ForOp{};
  if (!loop || !loop->hasAttr(hivm::ParallelLoopAttr::name) ||
      loop.getNumResults() || getConstantIntValue(loop.getLowerBound()) != 0 ||
      (!prefixRows &&
       getConstantIntValue(loop.getUpperBound()) != type.getDimSize(0)) ||
      getConstantIntValue(loop.getStep()) != 1 ||
      row.getMixedOffsets()[0].dyn_cast<Value>() != loop.getInductionVar() ||
      getConstantIntValue(row.getMixedOffsets()[1]) != 0 ||
      row.getStaticSizes() != ArrayRef<int64_t>{1, type.getDimSize(1)} ||
      !row.hasUnitStride() || !row->hasOneUse())
    return;
  auto copy = dyn_cast<memref::CopyOp>(*row->getUsers().begin());
  if (!copy || copy.getTarget() != row || copy->getBlock() != loop.getBody() ||
      alloc->getBlock() != loop->getBlock() ||
      view->getBlock() != loop->getBlock() || !alloc->isBeforeInBlock(loop) ||
      !loop->isBeforeInBlock(view))
    return;
  // A prefix only observes the first count rows. Shortening its loop is safe
  // when the discarded iterations have no effects besides the private copy.
  if (prefixRows &&
      llvm::any_of(loop.getBody()->without_terminator(), [&](Operation &op) {
        if (&op == copy)
          return false;
        auto effects = getEffectsRecursively(&op);
        return !effects || llvm::any_of(*effects, [](const auto &effect) {
          return !isa<MemoryEffects::Read>(effect.getEffect());
        });
      }))
    return;
  if (source != result) {
    Operation *consumer = prefix ? prefix : result.getDefiningOp();
    if (llvm::any_of(source.getUsers(), [&](Operation *user) {
          return user != consumer && user != hint;
        }))
      return;
  }
  ScalarMetadata metadata(loop);
  if (mask && !metadata.canRead(mask))
    return;
  SmallVector<tensor::ExtractOp> extracts;
  loop.walk([&](tensor::ExtractOp extract) { extracts.push_back(extract); });
  if (!llvm::all_of(extracts, [&](tensor::ExtractOp extract) {
        return metadata.canRead(extract.getTensor());
      }))
    return;

  // The frontend may attach the load hint after its implicit transpose.
  // Keep the request on the gathered rows when matmul folds that transpose.
  if (transpose) {
    rewriter.setInsertionPoint(hint);
    rewriter.create<annotation::MarkOp>(hint.getLoc(), source)
        ->setAttr(hivm::EnableL1GatherAttr::name, enabled);
    rewriter.modifyOpInPlace(
        hint, [&] { hint->removeAttr(hivm::EnableL1GatherAttr::name); });
    if (hint.isAttrEmpty())
      rewriter.eraseOp(hint);
  }

  rewriter.setInsertionPointToStart(loop.getBody());
  Value valid;
  if (mask) {
    Value zero = rewriter.create<arith::ConstantIndexOp>(loop.getLoc(), 0);
    valid = metadata.read(rewriter, mask, {loop.getInductionVar(), zero});
  } else if (prefixRows) {
    rewriter.setInsertionPoint(loop);
    Value bound = rewriter.create<arith::MinSIOp>(
        loop.getLoc(), loop.getUpperBound(), prefixRows);
    rewriter.modifyOpInPlace(loop, [&] { loop.setUpperBound(bound); });
  }
  for (tensor::ExtractOp extract : extracts) {
    rewriter.setInsertionPoint(extract);
    rewriter.replaceOp(extract, metadata.read(rewriter, extract.getTensor(),
                                              extract.getIndices()));
  }
  if (mask || prefixRows) {
    rewriter.setInsertionPoint(loop);
    Value zero = rewriter.create<arith::ConstantOp>(
        loop.getLoc(), rewriter.getZeroAttr(type.getElementType()));
    rewriter.create<linalg::FillOp>(loop.getLoc(), ValueRange{zero},
                                    ValueRange{alloc.getResult()});
    if (valid) {
      rewriter.setInsertionPoint(copy);
      auto guard = rewriter.create<scf::IfOp>(copy.getLoc(), valid, false);
      copy->moveBefore(guard.thenBlock()->getTerminator());
    }
    rewriter.replaceAllUsesWith(result, source);
    rewriter.eraseOp(result.getDefiningOp());
    if (prefix)
      rewriter.eraseOp(prefix);
  }
}

struct PrepareL1GatherPass : impl::PrepareL1GatherBase<PrepareL1GatherPass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation().walk(
        [&](annotation::MarkOp mark) { prepareGather(mark, rewriter); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hfusion::createPrepareL1GatherPass() {
  return std::make_unique<PrepareL1GatherPass>();
}

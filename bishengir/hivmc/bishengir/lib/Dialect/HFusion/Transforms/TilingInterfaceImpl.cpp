//===- TilingInterfaceImpl.cpp - Implementation of TilingInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/TilingInterfaceImpl.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/OpInterfaceUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/TilingInterface.h"

#include <optional>

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::hfusion;

//===----------------------------------------------------------------------===//
// Utility methods for implementation of Tiling Interface for Linalg ops
//===----------------------------------------------------------------------===//

/// Return the SSA values that represent the data point accessed using a given
/// `indexingMap` for a given point in the iteration space represented by `ivs`.
static SmallVector<Value> getIndicesForAccess(OpBuilder &b, Location loc,
                                              AffineMap indexingMap,
                                              ValueRange ivs) {
  SmallVector<Value> indices;
  indices.reserve(indexingMap.getNumResults());
  for (auto result : indexingMap.getResults()) {
    AffineMap m = AffineMap::get(indexingMap.getNumDims(),
                                 indexingMap.getNumSymbols(), result);
    Value v = b.create<affine::AffineApplyOp>(loc, m, ivs);
    indices.push_back(v);
  }
  return indices;
}

/// Method to inline the payload of a `linalgOp` given the iteration space
/// point and values for the arguments of the payload.
static LogicalResult inlinePayload(OpBuilder &b, LinalgOp linalgOp,
                                   ValueRange ivs, ValueRange argValues) {
  Block *body = linalgOp.getBlock();
  IRMapping map;
  map.map(body->getArguments(), argValues);
  for (auto &op : body->without_terminator()) {
    if (auto indexOp = dyn_cast<IndexOp>(&op)) {
      map.map(indexOp.getResult(), ivs[indexOp.getDim()]);
      continue;
    }
    b.clone(op, map);
  }

  Operation *terminator = body->getTerminator();
  Location loc = terminator->getLoc();
  for (const auto &operand : llvm::enumerate(terminator->getOperands())) {
    Value toStore = map.lookupOrDefault(operand.value());
    OpOperand *storeInto = linalgOp.getDpsInitOperand(operand.index());
    auto indices = getIndicesForAccess(
        b, loc, linalgOp.getMatchingIndexingMap(storeInto), ivs);
    b.create<memref::StoreOp>(
        loc, toStore, linalgOp.getDpsInitOperand(operand.index())->get(),
        indices);
  }
  return success();
}

// This impl. is copied from
// mlir/lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp

//===----------------------------------------------------------------------===//
// External Model for implementing `TilingInterface` for `LinalgOp`s.
//===----------------------------------------------------------------------===//

namespace {
/// External model implementation of TilingInterface for LinalgOps. An external
/// model implementation is used for now till the use of `TilingInterface` is
/// on-par with the current Linalg tiling + fusion patterns. Once it is
/// maybe possible to move this into the op-definition (though there are
/// advantages to leaving it as an external model)
template <typename LinalgOpTy>
struct LinalgOpTilingInterface
    : public TilingInterface::ExternalModel<LinalgOpTilingInterface<LinalgOpTy>,
                                            LinalgOpTy> {
  /// Return the loop iterator type.
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    LinalgOpTy concreteOp = cast<LinalgOpTy>(op);
    return concreteOp.getIteratorTypesArray();
  }

  /// Return the iteration domain range.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<OpFoldResult> allShapesSizes =
        linalgOp.createFlatListOfOperandDims(b, loc);
    AffineMap map = linalgOp.getShapesToLoopsMap();

    return llvm::to_vector(
        llvm::map_range(map.getResults(), [&](AffineExpr loopExpr) {
          OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
              b, loc, loopExpr, allShapesSizes);
          return Range{b.getIndexAttr(0), ofr, b.getIndexAttr(1)};
        }));
  }

  /// Instantiate the tiled implementation of the operation.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
    // specified could lead to out of bounds accesses.
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<Value> valuesToTile = linalgOp->getOperands();
    SmallVector<Value, 4> tiledOperands = makeTiledShapes(
        b, loc, linalgOp, valuesToTile, offsets, sizes, {}, true);

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(linalgOp, tiledOperands);

    Operation *tiledOp = clone(b, linalgOp, resultTensorTypes, tiledOperands);
    if constexpr (std::is_same<LinalgOpTy, ArangeOp>()) {
      // Since arange op has a built-in offset argument, we want to set and use
      // that instead of offset'ing the indices within the block. This helps the
      // lowering process be easier
      offsetArangeOp(b, tiledOp, offsets);
    } else
      offsetIndices(b, cast<LinalgOp>(tiledOp), offsets);

    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
  }

  /// Utility to fetch the offsets and sizes when applied as per the indexing
  /// map of the linalg op. This helps in fusing the linalg op as a consumer of
  /// a given slice op.
  void
  getMappedOffsetAndSize(LinalgOp linalgOp, OpBuilder &b, AffineMap indexingMap,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes,
                         SmallVectorImpl<OpFoldResult> &mappedOffsets,
                         SmallVectorImpl<OpFoldResult> &mappedSizes) const {
    unsigned numLoops = linalgOp.getNumLoops();
    auto tilingInterfaceOp = cast<TilingInterface>(linalgOp.getOperation());
    mappedOffsets.resize(numLoops);
    mappedSizes.resize(numLoops);
    if (!indexingMap.isPermutation()) {
      SmallVector<Range> iterationDomain =
          tilingInterfaceOp.getIterationDomain(b);
      for (const auto &&[index, value] : llvm::enumerate(iterationDomain)) {
        mappedOffsets[index] = value.offset;
        mappedSizes[index] = value.size;
      }
    }
    for (const auto &&[index, value] :
         llvm::enumerate(indexingMap.getResults())) {
      unsigned dimPosition = cast<AffineDimExpr>(value).getPosition();
      assert(dimPosition < numLoops);
      mappedOffsets[dimPosition] = offsets[index];
      mappedSizes[dimPosition] = sizes[index];
    }
  }

  /// Method to return the position of the result tile computed by the tiled
  /// operation.
  LogicalResult getIterationDomainTileFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    auto linalgOp = cast<LinalgOp>(op);

    // Check that the indexing map used for the operand is a projected
    // permutation. This could be relaxed with a more general approach that can
    // map the offsets and sizes from the operand to iteration space tiles
    // (filling in full extent for dimensions not used to access the result).
    AffineMap indexingMap =
        linalgOp.getMatchingIndexingMap(&op->getOpOperand(operandNumber));
    if (!indexingMap.isProjectedPermutation()) {
      return op->emitError()
             << "unhandled get iter domain position when operand is not "
                "accessed using a permuted projection";
    }

    getMappedOffsetAndSize(linalgOp, b, indexingMap, offsets, sizes,
                           iterDomainOffsets, iterDomainSizes);
    return success();
  }

  /// Return the details of the output tile generated by the tiled
  /// implementation.
  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    Location loc = op->getLoc();
    LinalgOp linalgOp = cast<LinalgOp>(op);

    AffineExpr d0;
    bindDims(b.getContext(), d0);
    SmallVector<OpFoldResult> subShapeSizes =
        llvm::to_vector(llvm::map_range(sizes, [&](OpFoldResult ofr) {
          return affine::makeComposedFoldedAffineApply(b, loc, d0 - 1, ofr);
        }));

    OpOperand *outOperand = linalgOp.getDpsInitOperand(resultNumber);
    SliceParameters sliceParams = computeSliceParameters(
        b, loc, outOperand->get(), sizes,
        linalgOp.getMatchingIndexingMap(outOperand), offsets,
        /*ubs*/ {}, subShapeSizes, true);
    resultOffsets = sliceParams.offsets;
    resultSizes = sliceParams.sizes;
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    auto linalgOp = cast<LinalgOp>(op);

    // Check that the indexing map used for the output is a projected
    // permutation. This could be relaxed with a more general approach that can
    // map the offsets and sizes from the result to iteration space tiles
    // (filling in full extent for dimensions not used to access the result).
    AffineMap indexingMap =
        linalgOp.getIndexingMapMatchingResult(op->getResult(resultNumber));
    if (!indexingMap.isProjectedPermutation()) {
      return op->emitOpError(
          "unhandled tiled implementation generation when result is not "
          "accessed using a permuted projection");
    }

    getMappedOffsetAndSize(linalgOp, b, indexingMap, offsets, sizes,
                           iterDomainOffsets, iterDomainSizes);
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    SmallVector<OpFoldResult> mappedOffsets;
    SmallVector<OpFoldResult> mappedSizes;
    if (failed(getIterationDomainTileFromResultTile(
            op, b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
      return failure();
    }
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    FailureOr<TilingResult> tilingResult =
        tilingInterfaceOp.getTiledImplementation(b, mappedOffsets, mappedSizes);

    if (failed(tilingResult))
      return failure();

    if (tilingResult->tiledOps.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");

    return TilingResult{
        tilingResult->tiledOps,
        SmallVector<Value>{tilingResult->tiledValues[resultNumber]}};
  }

  /// Method to generate the tiled implementation of an operation from the tile
  /// of the operand.
  FailureOr<TilingResult> getTiledImplementationFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {
    SmallVector<OpFoldResult> mappedOffsets;
    SmallVector<OpFoldResult> mappedSizes;
    if (failed(getIterationDomainTileFromOperandTile(
            op, b, operandNumber, offsets, sizes, mappedOffsets,
            mappedSizes))) {
      return failure();
    }
    return getTiledImplementation(op, b, mappedOffsets, mappedSizes);
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &builder,
                                             Location loc,
                                             ValueRange ivs) const {
    auto linalgOp = cast<LinalgOp>(op);
    if (!linalgOp.hasPureBufferSemantics())
      return op->emitOpError("expected operation to have buffer semantics");

    SmallVector<Value> indexedValues;
    indexedValues.reserve(linalgOp->getNumOperands());
    Location linalgOpLoc = op->getLoc();
    /// Load the data corresponding to the block arguments that
    /// represent input operands.
    for (OpOperand &operand : linalgOp->getOpOperands()) {
      if (!linalgOp.payloadUsesValueFromOperand(&operand)) {
        indexedValues.push_back(nullptr);
        continue;
      }
      if (linalgOp.isScalar(&operand)) {
        indexedValues.push_back(operand.get());
        continue;
      }
      SmallVector<Value> indices = getIndicesForAccess(
          builder, linalgOpLoc, linalgOp.getMatchingIndexingMap(&operand), ivs);
      Value load =
          builder.create<memref::LoadOp>(linalgOpLoc, operand.get(), indices);
      indexedValues.push_back(load);
    }

    /// Inline the op payload and store the result.
    return inlinePayload(builder, linalgOp, ivs, indexedValues);
  }
};

/// External model implementation of PartialReductionInterface for
/// `linalg.reduce` using `linalg.reduce` as merging op.
namespace ReduceOpPartialReductionInterfaceUsingNamedOp {
FailureOr<SmallVector<Value>> generateInitialTensorForPartialReduction(
    Operation *op, OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
    ArrayRef<int> reductionDims) {
  auto linalgOp = cast<LinalgOp>(op);
  OpBuilder::InsertionGuard guard(b);

  if (!linalgOp.hasPureTensorSemantics())
    return op->emitOpError("expected operation to have tensor semantics");

  SmallVector<Value> inits;
  for (int initIdx = 0, e = linalgOp.getNumDpsInits(); initIdx < e; ++initIdx) {
    // Insert the new parallel dimension based on the index of the reduction
    // loops. This could be controlled by user for more flexibility.
    SmallVector<Operation *, 4> combinerOps;
    if (!matchReduction(linalgOp.getRegionOutputArgs(), initIdx, combinerOps) ||
        combinerOps.size() != 1)
      return op->emitOpError("Failed to analysis the reduction operation.");

    Operation *reductionOp = combinerOps[0];
    std::optional<TypedAttr> identity = arith::getNeutralElement(reductionOp);
    if (!identity.has_value())
      return op->emitOpError(
          "Failed to get an identity value for the reduction operation.");

    ArrayRef<int64_t> oldShape =
        linalgOp.getShape(linalgOp.getDpsInitOperand(initIdx));
    ArrayRef<int64_t> inputShape =
        linalgOp.getShape(linalgOp.getDpsInputOperand(initIdx));

    // Calculate the new shape, we insert the new dimensions based on the
    // index of the reduction dimensions.
    SmallVector<int64_t> newOutputShape;
    SmallVector<Value> dynamicDims;
    int64_t currReductionDims = 0;
    DenseSet<int> reductionDimsSet(reductionDims.begin(), reductionDims.end());
    for (int64_t idx :
         llvm::seq<int64_t>(0, oldShape.size() + reductionDims.size())) {
      bool isReduceDim = reductionDimsSet.contains(idx);
      auto tileSize = sizes[idx];
      if (isReduceDim) {
        if (!isConstantIntValue(tileSize, 0)) {
          dispatchIndexOpFoldResults(tileSize, dynamicDims, newOutputShape);
        } else {
          auto reduceDim = inputShape[idx];
          newOutputShape.push_back(reduceDim);
          if (ShapedType::isDynamic(reduceDim))
            dynamicDims.push_back(b.create<tensor::DimOp>(
                loc, linalgOp.getDpsInputOperand(initIdx)->get(), idx));
        }
        currReductionDims++;
        continue;
      }
      int64_t oldIdx = idx - currReductionDims;
      int64_t dim = oldShape[oldIdx];
      newOutputShape.push_back(dim);
      if (ShapedType::isDynamic(dim))
        dynamicDims.push_back(b.create<tensor::DimOp>(
            loc, linalgOp.getDpsInitOperand(initIdx)->get(), oldIdx));
    }
    Value emptyTensor = b.create<tensor::EmptyOp>(
        loc, newOutputShape, linalgOp.getRegionOutputArgs()[initIdx].getType(),
        dynamicDims);
    Value constantOp = b.create<arith::ConstantOp>(loc, *identity);
    auto identityTensor =
        b.create<linalg::FillOp>(loc, constantOp, emptyTensor);
    inits.push_back(identityTensor.getResult(0));
  }

  return inits;
}

/// Main difference with community's implementation is as follows:
///
/// Community:
///   ```mlir
///     linalg.generic ins(%slice) outs(%accumulator) {
//        ^bb0(%in, %out)
///          %ret = some_computation(%in, %out)
///          linalg.yield %ret
///     }
///   ```
///
/// BiShengIR:
///   ```mlir
///     linalg.generic ins(%slice, %accumulator) outs(%accumulator) {
//        ^bb0(%in, %in1, %out)
///          %ret = some_computation(%in, %in1)
///          linalg.yield %ret
///     }
///   ```
///
/// The BiShengIR version is easier to specialize to named linalg ops.
FailureOr<TilingResult> tileToPartialReduction(Operation *op, OpBuilder &b,
                                               Location loc, ValueRange init,
                                               ArrayRef<OpFoldResult> offsets,
                                               ArrayRef<OpFoldResult> sizes,
                                               ArrayRef<int> reductionDims) {
  OpBuilder::InsertionGuard guard(b);
  auto linalgOp = cast<LinalgOp>(op);

  // Step 1. Extend init maps to have reduction dimension dims, since we
  // are converting them to parallel dimensions.
  SmallVector<AffineMap> newInitMaps;
  newInitMaps.reserve(linalgOp.getNumDpsInits());
  for (int idx : llvm::seq<int>(0, linalgOp.getNumDpsInits())) {
    // TODO: linalg::Generic doesn't have getDpsInitOperands. Can replace
    // this with a for range loop when we have it.
    AffineMap newMap =
        linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(idx));
    for (int redPos : reductionDims) {
      newMap = newMap.insertResult(b.getAffineDimExpr(redPos), redPos);
    }
    newInitMaps.push_back(newMap);
  }

  // Step 2a: Extract a slice of the input operands.
  SmallVector<Value, 4> tiledInputs = makeTiledShapes(
      b, loc, linalgOp, linalgOp.getDpsInputs(), offsets, sizes, {}, true);

  // Step 2b: Extract a slice of the init operands.
  SmallVector<Value, 1> tiledInits;
  for (auto [valueMap, valueToTile] : llvm::zip_equal(newInitMaps, init)) {
    int64_t initRank = valueMap.getNumResults();
    SmallVector<OpFoldResult> initOffset(initRank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> initStride(initRank, b.getIndexAttr(1));
    SmallVector<OpFoldResult> initSizes;
    for (AffineExpr dimExpr : valueMap.getResults()) {
      auto dim = cast<AffineDimExpr>(dimExpr);
      initSizes.push_back(sizes[dim.getPosition()]);
    }
    // TODO: Use SubsetExtractOpInterface here once available.
    auto extractSlice = b.create<tensor::ExtractSliceOp>(
        loc, valueToTile, initOffset, initSizes, initStride);
    tiledInits.push_back(extractSlice);
    // BEGIN BISHENGIR_EXTENTION
    tiledInputs.push_back(extractSlice);
    // END BISHENGIR_EXTENTION
  }

  // Update the indexing maps.

  // BEGIN BISHENGIR_EXTENTION
  SmallVector<AffineMap> newMaps = linalgOp.getIndexingMapsArray();
  SmallVector<AffineMap> additionalMaps;
  // Change the init maps.
  for (int idx : llvm::seq<int>(0, linalgOp.getNumDpsInits())) {
    // TODO: linalg::Generic doesn't have getDpsInitOperands. Can replace
    // this with a for range loop when we have it.
    OpOperand *initOperand = linalgOp.getDpsInitOperand(idx);
    int64_t mapIdx = linalgOp.getIndexingMapIndex(initOperand);
    newMaps[mapIdx] = newInitMaps[idx];
    additionalMaps.push_back(newInitMaps[idx]);
  }
  newMaps.append(additionalMaps);
  // END BISHENGIR_EXTENTION

  // Step 3. Change the reduction dim iterator types.
  SmallVector<utils::IteratorType> newIteratorTypes =
      linalgOp.getIteratorTypesArray();
  for (int dim : reductionDims)
    newIteratorTypes[dim] = utils::IteratorType::parallel;

  // Step 4. Create the new generic op.
  auto genericOp =
      b.create<GenericOp>(loc, ValueRange(tiledInits).getTypes(), tiledInputs,
                          tiledInits, newMaps, newIteratorTypes);
  IRMapping mapping;
  op->getRegion(0).cloneInto(&genericOp.getRegion(),
                             genericOp.getRegion().begin(), mapping);

  // BEGIN BISHENGIR_EXTENTION
  Region &r = genericOp.getRegion();
  for (auto tiledInit : tiledInits) {
    Type elemType = getElementTypeOrSelf(tiledInit.getType());
    r.getBlocks().front().addArgument(elemType, linalgOp->getLoc());
  }
  // END BISHENGIR_EXTENTION

  return TilingResult{
      {genericOp.getOperation()},
      llvm::map_to_vector(genericOp->getResults(),
                          [](OpResult r) -> Value { return r; })};
}

FailureOr<MergeResult> mergeReductions(Operation *op, OpBuilder &b,
                                       Location loc, ValueRange partialReduce,
                                       ArrayRef<int> reductionDims) {
  auto linalgOp = cast<LinalgOp>(op);
  SmallVector<int64_t> reductionDimsInt64(reductionDims.begin(),
                                          reductionDims.end());
  auto reduction = b.create<linalg::ReduceOp>(
      loc, partialReduce, linalgOp.getDpsInits(), reductionDimsInt64,
      [&linalgOp](OpBuilder &b, Location loc, ValueRange inputs) {
        int64_t numInits = linalgOp.getNumDpsInits();
        SmallVector<Value> yieldedValues;
        for (int idx : llvm::seq<int>(0, numInits)) {
          // Get the combiner op.
          SmallVector<Operation *, 4> combinerOps;
          matchReduction(linalgOp.getRegionOutputArgs(), idx, combinerOps);
          Operation *clonedReductionOp = b.clone(*combinerOps[0]);
          // Combine the input at idx and output at numInits + idx.
          clonedReductionOp->setOperand(0, inputs[idx]);
          clonedReductionOp->setOperand(1, inputs[numInits + idx]);
          // Yield.
          yieldedValues.push_back(clonedReductionOp->getResult(0));
        }
        b.create<linalg::YieldOp>(loc, yieldedValues);
      });
  return MergeResult{
      {reduction.getOperation()},
      llvm::map_to_vector(reduction->getResults(),
                          [](OpResult r) -> Value { return r; })};
}
} // namespace ReduceOpPartialReductionInterfaceUsingNamedOp

/// External model implementation of TilingInterface for `hfusion.deinterleave`.
struct DeinterleaveOpTiling
    : public TilingInterface::ExternalModel<DeinterleaveOpTiling,
                                            DeinterleaveOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto deinterleaveOp = cast<DeinterleaveOp>(op);
    int64_t rank =
        cast<ShapedType>(deinterleaveOp.getResult(0).getType()).getRank();
    // only the last dim has concat iterator type, other dims have parallel
    // iterator type
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    iteratorTypes[rank - 1] = utils::IteratorType::concat;
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    ReifiedRankedShapedTypeDims reifiedShapes;
    (void)reifyResultShapes(b, op, reifiedShapes);
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    // Initialize all the ranges to {zero, one, one}. All the `ub`s are
    // overwritten.
    SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
    for (const auto &ub : enumerate(reifiedShapes[0]))
      loopRanges[ub.index()].size = ub.value();
    return loopRanges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto deinterleaveOp = cast<DeinterleaveOp>(op);
    Location loc = deinterleaveOp->getLoc();
    auto oldResultType =
        cast<ShapedType>(deinterleaveOp->getResult(0).getType());
    SmallVector<Value> dynResultShapes;
    SmallVector<int64_t> staticResultShapes;
    dispatchIndexOpFoldResults(sizes, dynResultShapes, staticResultShapes);
    RankedTensorType resultType = RankedTensorType::get(
        staticResultShapes, oldResultType.getElementType());
    // newSizes[i] = sizes[i]             if i != deinterleave_axis
    // newSizes[i] = sizes[i] * chan_num  if i == deinterleave_axis
    // TODO: current implementation assume deinterleave axis will not be tiled
    int64_t chanNum = deinterleaveOp.getDeInterLeaveChannelNum();
    int64_t rank = static_cast<int64_t>(sizes.size());
    SmallVector<OpFoldResult> newSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (i == rank - 1) {
        AffineExpr mulExpr =
            b.getAffineSymbolExpr(0) * b.getAffineSymbolExpr(1);
        auto inputSize = affine::makeComposedFoldedAffineApply(
            b, op->getLoc(), mulExpr, {sizes[i], b.getIndexAttr(chanNum)});
        newSizes.push_back(inputSize);
        continue;
      }
      newSizes.push_back(sizes[i]);
    }
    // Build normalized offsets (pad/truncate if needed). If offsets is empty,
    // treat offsets as zeros.
    OpFoldResult zero = b.getIndexAttr(0);
    SmallVector<OpFoldResult> normalizedOffsets;
    if (offsets.empty()) {
      normalizedOffsets.assign(rank, zero);
    } else {
      normalizedOffsets.assign(offsets.begin(), offsets.end());
      if ((int64_t)normalizedOffsets.size() < rank)
        normalizedOffsets.resize(rank, zero);
      else if ((int64_t)normalizedOffsets.size() > rank)
        normalizedOffsets.resize(rank);
    }
    // Compute newOffsets: for last dim, offsets[last] * chanNum
    SmallVector<OpFoldResult> newOffsets;
    AffineExpr mulExprSym =
        b.getAffineSymbolExpr(0) * b.getAffineSymbolExpr(1);
    for (int64_t i = 0; i < rank; ++i) {
      if (i == rank - 1) {
        // multiply offset by chanNum
        auto offsetMul = affine::makeComposedFoldedAffineApply(
            b, loc, mulExprSym,
            {normalizedOffsets[i], b.getIndexAttr(chanNum)});
        newOffsets.push_back(offsetMul);
      } else {
        newOffsets.push_back(normalizedOffsets[i]);
      }
    }
    // Use one-strides for the extract slice.
    SmallVector<OpFoldResult> oneStrides(rank, b.getIndexAttr(1));
    Value newSlice = b.create<tensor::ExtractSliceOp>(
        loc, deinterleaveOp.getInput(), newOffsets, newSizes, oneStrides);
    // Preserve the channel index attribute when creating new DeinterleaveOp.
    auto indexAttr = deinterleaveOp.getChannelIndexAttr();
    auto newDeinterleaveOp =
        b.create<DeinterleaveOp>(loc, resultType, newSlice, indexAttr);
    newDeinterleaveOp->setAttrs(getPrunedAttributeList(
        deinterleaveOp, DeinterleaveOp::getAttributeNames()));
    return TilingResult{{newDeinterleaveOp}, {newDeinterleaveOp->getResult(0)}};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return getTiledImplementation(op, b, offsets, sizes);
  }
};

/// External model implementation of TilingInterface for `hfusion.interleave`.
struct InterleaveOpTiling
    : public TilingInterface::ExternalModel<InterleaveOpTiling, InterleaveOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto interleaveOp = cast<InterleaveOp>(op);
    int64_t rank =
        cast<ShapedType>(interleaveOp.getOutput().getType()).getRank();
    // only the last dim has concat iterator type, other dims have parallel
    // iterator type
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    iteratorTypes[rank - 1] = utils::IteratorType::concat;
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    ReifiedRankedShapedTypeDims reifiedShapes;
    (void)reifyResultShapes(b, op, reifiedShapes);
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    // Initialize all the ranges to {zero, one, one}. All the `ub`s are
    // overwritten.
    SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
    for (const auto &ub : enumerate(reifiedShapes[0]))
      loopRanges[ub.index()].size = ub.value();
    return loopRanges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto interleaveOp = cast<InterleaveOp>(op);
    Location loc = interleaveOp->getLoc();
    auto oldResultType = cast<ShapedType>(interleaveOp.getOutput().getType());
    SmallVector<Value> dynResultShapes;
    SmallVector<int64_t> staticResultShapes;
    dispatchIndexOpFoldResults(sizes, dynResultShapes, staticResultShapes);
    RankedTensorType resultType = RankedTensorType::get(
        staticResultShapes, oldResultType.getElementType());
    // newSizes[i] = sizes[i]             if i != interleave_axis
    // newSizes[i] = sizes[i] / chan_num  if i == interleave_axis
    // TODO: current implementation assume interleave axis will not be tiled
    int64_t chanNum = interleaveOp.getInterLeaveChannelNums();
    int64_t rank = static_cast<int64_t>(sizes.size());
    SmallVector<OpFoldResult> newSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (i == rank - 1) {
        AffineExpr divExpr =
            b.getAffineSymbolExpr(0).floorDiv(b.getAffineSymbolExpr(1));
        auto inputSize = affine::makeComposedFoldedAffineApply(
            b, op->getLoc(), divExpr, {sizes[i], b.getIndexAttr(chanNum)});
        newSizes.push_back(inputSize);
        continue;
      }
      newSizes.push_back(sizes[i]);
    }
    // Build newOffsets so that the last-dimension offset is offset / chanNum.
    // If offsets is empty, treat offsets as zeros.
    OpFoldResult zero = b.getIndexAttr(0);
    SmallVector<OpFoldResult> normalizedOffsets;
    if (offsets.empty()) {
      normalizedOffsets.assign(rank, zero);
    } else {
      normalizedOffsets.assign(offsets.begin(), offsets.end());
      // If offsets length mismatches rank, pad/truncate defensively:
      if ((int64_t)normalizedOffsets.size() < rank)
        normalizedOffsets.resize(rank, zero);
      else if ((int64_t)normalizedOffsets.size() > rank)
        normalizedOffsets.resize(rank);
    }
    SmallVector<OpFoldResult> newOffsets;
    AffineExpr divExpr =
        b.getAffineSymbolExpr(0).floorDiv(b.getAffineSymbolExpr(1));
    for (int64_t i = 0; i < rank; ++i) {
      if (i == rank - 1) {
        // last dim -> divide offset by chanNum
        auto offsetDiv = affine::makeComposedFoldedAffineApply(
            b, loc, divExpr, {normalizedOffsets[i], b.getIndexAttr(chanNum)});
        newOffsets.push_back(offsetDiv);
      } else {
        newOffsets.push_back(normalizedOffsets[i]);
      }
    }
    SmallVector<OpFoldResult> oneStrides(offsets.size() ? offsets.size() : rank,
                                         b.getIndexAttr(1));
    SmallVector<Value> inputs = interleaveOp.getInput();
    SmallVector<Value> sliceInputs;
    for (Value input : inputs) {
      Value newSlice = b.create<tensor::ExtractSliceOp>(loc, input, newOffsets,
                                                        newSizes, oneStrides);
      sliceInputs.push_back(newSlice);
    }
    auto newInterleaveOp = b.create<InterleaveOp>(loc, resultType, sliceInputs);
    newInterleaveOp->setAttrs(getPrunedAttributeList(
        interleaveOp, InterleaveOp::getAttributeNames()));
    return TilingResult{{newInterleaveOp}, {newInterleaveOp.getOutput()}};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return getTiledImplementation(op, b, offsets, sizes);
  }
};

} // namespace

template <typename OpType>
static void registerOne(MLIRContext *ctx) {
  OpType::template attachInterface<LinalgOpTilingInterface<OpType>>(*ctx);
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(MLIRContext *ctx) {
  (registerOne<OpTypes>(ctx), ...);
}

#define GET_OP_LIST

void hfusion::registerTilingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, hfusion::HFusionDialect *dialect) {
        registerAll<
#include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.cpp.inc"
            >(ctx);
        hfusion::DeinterleaveOp::attachInterface<DeinterleaveOpTiling>(*ctx);
        hfusion::InterleaveOp::attachInterface<InterleaveOpTiling>(*ctx);
      });
}

RegisterOpInterfaceOverride(
    /*Op=*/linalg::ReduceOp, /*Interface=*/PartialReductionOpInterface,
    /*InterfaceMethod=*/generateInitialTensorForPartialReduction,
    /*Impl=*/
    &ReduceOpPartialReductionInterfaceUsingNamedOp::
        generateInitialTensorForPartialReduction);

RegisterOpInterfaceOverride(
    /*Op=*/linalg::ReduceOp, /*Interface=*/PartialReductionOpInterface,
    /*InterfaceMethod=*/tileToPartialReduction,
    /*Impl=*/
    &ReduceOpPartialReductionInterfaceUsingNamedOp::tileToPartialReduction);

RegisterOpInterfaceOverride(
    /*Op=*/linalg::ReduceOp, /*Interface=*/PartialReductionOpInterface,
    /*InterfaceMethod=*/mergeReductions,
    /*Impl=*/
    &ReduceOpPartialReductionInterfaceUsingNamedOp::mergeReductions);

//===- HIVMTilingInterfaceImpl.cpp - Implementation of TilingInterface ----===//
//
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
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/HIVMTilingInterfaceImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-tiling-interface"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::hivm;

namespace mlir {
namespace hivm {

namespace {

//===----------------------------------------------------------------------===//
// Common default tiling implementation for HIVM Ops.
//
// Note: We defined all the `XXXImpl` functions because for some HIVM ops, we
// want to "override" a certain tiling interface function.
// If we use template specialization, we sill have to rewrite all the functions.
// And we cannot use multiple inheritance to define a base class that holds
// the common implementation because it seems like the vtable was incorrect.
//===----------------------------------------------------------------------===//

/// Return the loop iterator type, HIVM has its own iterator types
///
/// Take a look at this example
///   %61 = hivm.hir.vsub ins(%59, %60 : tensor<16x16xf32>, tensor<16x16xf32>)
///                       outs(%10 : tensor<16x16xf32>) -> tensor<16x16xf32>
/// It corresponds to the
/// direction of the looping of a certain operators In an elementwise
/// operation, for example, each layer of dimensions represents one loop
/// In this case there are two iterator types, both of them is a parallel
inline SmallVector<utils::IteratorType>
getLoopIteratorTypesImpl(Operation *op) {
  HIVMStructuredOp concreteOp = cast<HIVMStructuredOp>(op);
  return tiling_helper::convertToLinalgIteratorTypes(
      concreteOp.getIteratorTypesArray());
}

/// Return the iteration domain range.
/// This function calculates the bounds for each loop in a structured
/// operation.
inline SmallVector<Range> getIterationDomainImpl(Operation *op, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op->getLoc();
  auto hivmOp = cast<HIVMStructuredOp>(op);
  // Get all dimension sizes from all operands as a flat list
  // ins(%a : tensor<bxcxf32>) outs(%d : tensor<exfxf32>)
  // Would return b, c, e, f
  // Pay attention that it's a OpFoldResult, so dynamic will also be supported
  SmallVector<OpFoldResult> allShapesSizes =
      hivmOp.createFlatListOfOperandDims(b, loc);
  LDBG("Printing allShapesSizes");
  LLVM_DEBUG(llvm::for_each(allShapesSizes,
                            [&](OpFoldResult &opFoldRes) { LDBG(opFoldRes); }));
  // Get the mapping from results to inputs.
  AffineMap map = hivmOp.getShapesToLoopsMap();
  // For each loop dimension, create a Range object defining its bounds
  return llvm::to_vector(
      llvm::map_range(map.getResults(), [&](AffineExpr loopExpr) {
        // Apply the affine expression to convert from loop space to shape
        // space This computes the actual size for this loop dimension
        OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
            b, loc, loopExpr, allShapesSizes);
        // Create a Range with:
        // - start value of 0
        // - size based on the computed dimension
        // - step size of 1
        return Range{b.getIndexAttr(0), ofr, b.getIndexAttr(1)};
      }));
}

inline FailureOr<TilingResult>
getTiledImplementationImpl(Operation *op, OpBuilder &b,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes) {
  // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
  // specified could lead to out of bounds accesses.
  Location loc = op->getLoc();
  auto hivmOp = cast<HIVMStructuredOp>(op);
  SmallVector<Value> valuesToTile = hivmOp->getOperands();
  SmallVector<Value, 4> tiledOperands = tiling_helper::makeTiledShapes(
      b, loc, hivmOp, valuesToTile, offsets, sizes, {}, true);

  SmallVector<Type> resultTensorTypes =
      tiling_helper::getTensorOutputTypes(hivmOp, tiledOperands);

  Operation *tiledOp = clone(b, hivmOp, resultTensorTypes, tiledOperands);
  // TODO: if it has body, should handle index semantics
  assert(!hivmOp.hasIndexSemantics());
  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

// Return the details of the output tile generated by the tiled
// implementation.
inline LogicalResult
getResultTilePositionImpl(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes,
                          SmallVector<OpFoldResult> &resultOffsets,
                          SmallVector<OpFoldResult> &resultSizes) {
  Location loc = op->getLoc();
  HIVMStructuredOp hivmOp = cast<HIVMStructuredOp>(op);

  AffineExpr d0;
  bindDims(b.getContext(), d0);
  SmallVector<OpFoldResult> subShapeSizes =
      llvm::to_vector(llvm::map_range(sizes, [&b, &loc, &d0](OpFoldResult ofr) {
        return affine::makeComposedFoldedAffineApply(b, loc, d0 - 1, ofr);
      }));

  OpOperand *outOperand = hivmOp.getDpsInitOperand(resultNumber);
  SliceParameters sliceParams =
      computeSliceParameters(b, loc, outOperand->get(), sizes,
                             hivmOp.getMatchingIndexingMap(outOperand), offsets,
                             /*ubs*/ {}, subShapeSizes, true);
  resultOffsets = sliceParams.offsets;
  resultSizes = sliceParams.sizes;
  return success();
}

/// Utility to fetch the offsets and sizes when applied as per the indexing
/// map of the hivm op. This helps in fusing the hivm op as a consumer of
/// a given slice op.
inline void getMappedOffsetAndSizeImpl(
    HIVMStructuredOp hivmOp, OpBuilder &b, AffineMap indexingMap,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &mappedOffsets,
    SmallVectorImpl<OpFoldResult> &mappedSizes) {
  unsigned numLoops = hivmOp.getNumLoops();
  auto tilingInterfaceOp = cast<TilingInterface>(hivmOp.getOperation());
  // Initialize offsets and sizes.
  // For some HIVM ops, we preserve the rank before and after reduction. For
  // example, for hivm.hir.vreduce, the indexing maps are:
  //   input:   (d0, d1) -> (d0, d1)
  //   result:  (d0, d1) -> (d0, 0)
  // In that case, the mapped offset and size for d1 in terms of the slice
  // is always going to be zero and one, respectively.
  //
  // However, because the result mapping is not a (projected) permutation,
  // we cannot fill the offsets and sizes directly.
  // E.g.,
  //   input:   (d0, d1) -> (d0, d1)
  //   result:  (d0, d1) -> (0, d0)
  //                         |
  //                        we can only assume this is d1
  mappedOffsets = SmallVector<OpFoldResult>(numLoops, b.getI64IntegerAttr(0));
  mappedSizes = SmallVector<OpFoldResult>(numLoops, b.getI64IntegerAttr(1));

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
    auto maybeDimExpr = dyn_cast<AffineDimExpr>(value);
    if (maybeDimExpr) {
      mappedOffsets[maybeDimExpr.getPosition()] = offsets[index];
      mappedSizes[maybeDimExpr.getPosition()] = sizes[index];
      continue;
    }
    // If it's not a dim, we only support const zero
    assert(isa<AffineConstantExpr>(value));
  }
}

inline LogicalResult getIterationDomainTileFromResultTileImpl(
    Operation *op, OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  auto hivmOp = cast<HIVMStructuredOp>(op);

  // Check that the indexing map used for the output is a projected
  // permutation. This could be relaxed with a more general approach that can
  // map the offsets and sizes from the result to iteration space tiles
  // (filling in full extent for dimensions not used to access the result).
  AffineMap indexingMap =
      hivmOp.getIndexingMapMatchingResult(op->getResult(resultNumber));
  if (!indexingMap.isProjectedPermutation(/*allowZeroInResults=*/true)) {
    return op->emitOpError(
        "unhandled tiled implementation generation when result is not "
        "accessed using a permuted projection");
  }

  getMappedOffsetAndSizeImpl(hivmOp, b, indexingMap, offsets, sizes,
                             iterDomainOffsets, iterDomainSizes);
  return success();
}

inline FailureOr<TilingResult>
generateResultTileValueImpl(Operation *op, OpBuilder &b, unsigned resultNumber,
                            ArrayRef<OpFoldResult> offsets,
                            ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromResultTileImpl(
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

inline LogicalResult generateScalarImplementationImpl(
    [[maybe_unused]] Operation *op, [[maybe_unused]] OpBuilder &builder,
    [[maybe_unused]] Location loc, [[maybe_unused]] ValueRange ivs) {
  llvm_unreachable("HIVM Op doesn't have body");
}

//===----------------------------------------------------------------------===//
// Helper macros for the default implementation.
//===----------------------------------------------------------------------===//

#define DECLARE_DEFAULT_GET_LOOP_ITERATOR_TYPES                                \
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const { \
    return getLoopIteratorTypesImpl(op);                                       \
  }

#define DECLARE_DEFAULT_GET_LOOP_ITERATION_DOMAIN                              \
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {   \
    return getIterationDomainImpl(op, b);                                      \
  }

#define DECLARE_DEFAULT_GET_TILED_IMPLEMENTATION                               \
  FailureOr<TilingResult> getTiledImplementation(                              \
      Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,             \
      ArrayRef<OpFoldResult> sizes) const {                                    \
    return getTiledImplementationImpl(op, b, offsets, sizes);                  \
  }

#define DECLARE_DEFAULT_GET_RESULT_TILE_POSITION                               \
  LogicalResult getResultTilePosition(                                         \
      Operation *op, OpBuilder &b, unsigned resultNumber,                      \
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,            \
      SmallVector<OpFoldResult> &resultOffsets,                                \
      SmallVector<OpFoldResult> &resultSizes) const {                          \
    return getResultTilePositionImpl(op, b, resultNumber, offsets, sizes,      \
                                     resultOffsets, resultSizes);              \
  }

#define DECLARE_DEFAULT_GET_MAPPED_OFFSETS_AND_SIZE                            \
  void getMappedOffsetAndSize(                                                 \
      HIVMStructuredOp hivmOp, OpBuilder &b, AffineMap indexingMap,            \
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,            \
      SmallVectorImpl<OpFoldResult> &mappedOffsets,                            \
      SmallVectorImpl<OpFoldResult> &mappedSizes) const {                      \
    return getMappedOffsetAndSizeImpl(hivmOp, b, indexingMap, offsets, sizes,  \
                                      mappedOffsets, mappedSizes);             \
  }

#define DECLARE_DEFAULT_GET_ITERATION_DOMAIN_TILE_FROM_RESULT_TILE             \
  LogicalResult getIterationDomainTileFromResultTile(                          \
      Operation *op, OpBuilder &b, unsigned resultNumber,                      \
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,            \
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,                        \
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {                  \
    return getIterationDomainTileFromResultTileImpl(                           \
        op, b, resultNumber, offsets, sizes, iterDomainOffsets,                \
        iterDomainSizes);                                                      \
  }

#define DECLARE_DEFAULT_GET_GENERATE_RESULT_TILE_VALUE                         \
  FailureOr<TilingResult> generateResultTileValue(                             \
      Operation *op, OpBuilder &b, unsigned resultNumber,                      \
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {    \
    return generateResultTileValueImpl(op, b, resultNumber, offsets, sizes);   \
  }

#define DECLARE_DEFAULT_GET_SCALAR_IMPLEMENTATION                              \
  LogicalResult generateScalarImplementation(                                  \
      Operation *op, OpBuilder &builder, Location loc, ValueRange ivs) const { \
    return generateScalarImplementationImpl(op, builder, loc, ivs);            \
  }

//===----------------------------------------------------------------------===//
// Default tiling interface impl. for HIVM Structured Op
//===----------------------------------------------------------------------===//
template <typename HIVMOpTy>
struct HIVMOpTilingInterface
    : public TilingInterface::ExternalModel<HIVMOpTilingInterface<HIVMOpTy>,
                                     HIVMOpTy> {
  DECLARE_DEFAULT_GET_LOOP_ITERATOR_TYPES
  DECLARE_DEFAULT_GET_LOOP_ITERATION_DOMAIN
  DECLARE_DEFAULT_GET_RESULT_TILE_POSITION
  DECLARE_DEFAULT_GET_MAPPED_OFFSETS_AND_SIZE
  DECLARE_DEFAULT_GET_ITERATION_DOMAIN_TILE_FROM_RESULT_TILE
  DECLARE_DEFAULT_GET_GENERATE_RESULT_TILE_VALUE
  DECLARE_DEFAULT_GET_SCALAR_IMPLEMENTATION
  DECLARE_DEFAULT_GET_TILED_IMPLEMENTATION
};

//===----------------------------------------------------------------------===//
// Tiling interface impl. for MmadL1Op
//===----------------------------------------------------------------------===//
struct MmadL1OpTilingInterface
    : public TilingInterface::ExternalModel<MmadL1OpTilingInterface,
                                            hivm::MmadL1Op> {
  DECLARE_DEFAULT_GET_LOOP_ITERATOR_TYPES
  DECLARE_DEFAULT_GET_LOOP_ITERATION_DOMAIN
  DECLARE_DEFAULT_GET_RESULT_TILE_POSITION
  DECLARE_DEFAULT_GET_MAPPED_OFFSETS_AND_SIZE
  DECLARE_DEFAULT_GET_ITERATION_DOMAIN_TILE_FROM_RESULT_TILE
  DECLARE_DEFAULT_GET_GENERATE_RESULT_TILE_VALUE
  DECLARE_DEFAULT_GET_SCALAR_IMPLEMENTATION

  // Instantiate the tiled implementation of the operation.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
    // specified could lead to out of bounds accesses.
    Location loc = op->getLoc();
    auto hivmOp = cast<HIVMStructuredOp>(op);
    SmallVector<Value> valuesToTile = hivmOp->getOperands();
    // Compute the tiled operands.
    SmallVector<Value, 4> tiledOperands = tiling_helper::makeTiledShapes(
        b, loc, hivmOp, valuesToTile, offsets, sizes, {}, true);
    SmallVector<Type> resultTensorTypes =
        tiling_helper::getTensorOutputTypes(hivmOp, tiledOperands);
    // Make a clone of the original op but with tiled operands.
    Operation *tiledOp = clone(b, hivmOp, resultTensorTypes, tiledOperands);
    // Adjust the Real M/K/N values according to tile size.
    tryAdjustRealMKN(b, cast<hivm::MmadL1Op>(tiledOp), offsets, sizes);
    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
  }

private:
  Value getBoundedRealMKN(OpBuilder &b, Location loc, OpFoldResult iv,
                          OpFoldResult fullSize, OpFoldResult tileSize) const {
    OpBuilder::InsertionGuard g(b);
    AffineExpr dim0, dim1, dim2;
    MLIRContext *context = b.getContext();
    bindDims(context, dim0, dim1, dim2);
    // Take the min between the real size left in this iteration and the tiled
    // size to avoid out-of-bounds accesses.
    AffineMap minMap = AffineMap::inferFromExprList(
                           {ArrayRef<AffineExpr>{dim1 - dim2, dim0}}, context)
                           .front();
    Value fullSizeVal = getValueOrCreateConstantIndexOp(b, loc, fullSize);
    OpFoldResult boundedOfr = affine::makeComposedFoldedAffineMin(
        b, loc, minMap,
        SmallVector<OpFoldResult>{/*dim0=*/tileSize, /*dim1=*/fullSizeVal,
                                  /*dim2=*/iv});
    return getValueOrCreateConstantIndexOp(b, loc, boundedOfr);
  }

  /// Try to adjust the Real M/K/N size.
  ///
  /// For example, Matrix A has shape tensor<128x?xf16> and the real M is 127.
  /// If the tile size is 64, the real M should be a variable depending on the
  /// iv.
  void tryAdjustRealMKN(OpBuilder &b, hivm::MmadL1Op tiledOp,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> tileSizes) const {
    if (tileSizes.size() != tiledOp.getNumLoops())
      llvm_unreachable("Invalid tiling");

    OpBuilder::InsertionGuard g(b);
    Location loc = tiledOp->getLoc();
    b.setInsertionPoint(tiledOp);
    tiledOp.getRealMMutable().assign(getBoundedRealMKN(
        b, loc, offsets[0], tiledOp.getRealM(), tileSizes[0]));
    tiledOp.getRealNMutable().assign(getBoundedRealMKN(
        b, loc, offsets[1], tiledOp.getRealN(), tileSizes[1]));
    tiledOp.getRealKMutable().assign(getBoundedRealMKN(
        b, loc, offsets[2], tiledOp.getRealK(), tileSizes[2]));
  }
};

#undef DECLARE_DEFAULT_GET_LOOP_ITERATOR_TYPES
#undef DECLARE_DEFAULT_GET_LOOP_ITERATION_DOMAIN
#undef DECLARE_DEFAULT_GET_RESULT_TILE_POSITION
#undef DECLARE_DEFAULT_GET_MAPPED_OFFSETS_AND_SIZE
#undef DECLARE_DEFAULT_GET_ITERATION_DOMAIN_TILE_FROM_RESULT_TILE
#undef DECLARE_DEFAULT_GET_GENERATE_RESULT_TILE_VALUE
#undef DECLARE_DEFAULT_GET_SCALAR_IMPLEMENTATION
#undef DECLARE_DEFAULT_GET_TILED_IMPLEMENTATION

} // namespace

} // namespace hivm
} // namespace mlir

template <typename OpType> static void registerOne(MLIRContext *ctx) {
  OpType::template attachInterface<HIVMOpTilingInterface<OpType>>(*ctx);
}

/// Variadic helper function.
template <typename... OpTypes> static void registerAll(MLIRContext *ctx) {
  (registerOne<OpTypes>(ctx), ...);
}

#define GET_OP_LIST

void hivm::registerTilingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::hivm::HIVMDialect *dialect) {
        registerAll<
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
            >(ctx);
        registerOne<hivm::LoadOp>(ctx);
        registerOne<hivm::CopyOp>(ctx);
        registerOne<hivm::FixpipeOp>(ctx);
        registerOne<hivm::StoreOp>(ctx);
        hivm::MmadL1Op::attachInterface<MmadL1OpTilingInterface>(*ctx);
      });
}

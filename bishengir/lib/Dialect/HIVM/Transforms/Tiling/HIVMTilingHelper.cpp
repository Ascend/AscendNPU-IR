//===- HIVMTilingHelper.cpp - Implementation of HIVM Tiling Helper --------===//
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
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "hivm-tiling-helper"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hivm {
namespace tiling_helper {
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Utility methods for implementation of Tiling Interface for HIVM ops
//===----------------------------------------------------------------------===//

SmallVector<Type> getTensorOutputTypes(HIVMStructuredOp op,
                                       ValueRange operands) {
  if (op.hasPureBufferSemantics())
    return {};
  if (llvm::all_of(op.getDpsInits(),
                   [&](Value v) { return isa<MemRefType>(v.getType()); }))
    return {};
  return llvm::to_vector(
      llvm::map_range(op.getDpsInitsMutable(), [&](OpOperand &opOperand) {
        return operands[opOperand.getOperandNumber()].getType();
      }));
}

SmallVector<Value> insertSlicesBack(OpBuilder &builder, Location loc,
                                    HIVMStructuredOp op, ValueRange operands,
                                    ValueRange results) {
  if (op.hasPureBufferSemantics())
    return {};
  SmallVector<Value> tensorResults;
  tensorResults.reserve(results.size());
  // Insert a insert_slice for each output tensor.
  unsigned resultIdx = 0;
  for (OpOperand &opOperand : op.getDpsInitsMutable()) {
    // TODO: use an interface/adaptor to avoid leaking position in
    // `tiledOperands`.
    Value outputTensor = operands[opOperand.getOperandNumber()];
    if (auto sliceOp = outputTensor.getDefiningOp<tensor::ExtractSliceOp>()) {
      Value inserted = builder.create<tensor::InsertSliceOp>(
          loc, sliceOp.getSource().getType(), results[resultIdx],
          sliceOp.getSource(), sliceOp.getOffsets(), sliceOp.getSizes(),
          sliceOp.getStrides(), sliceOp.getStaticOffsets(),
          sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
      tensorResults.push_back(inserted);
    } else {
      tensorResults.push_back(results[resultIdx]);
    }
    ++resultIdx;
  }
  return tensorResults;
}

bool isTiled(AffineExpr expr, ArrayRef<OpFoldResult> tileSizes) {
  if (!expr)
    return false;
  TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the `map  varies with respect to a non-zero `tileSize`.
bool isTiled(AffineMap map, ArrayRef<OpFoldResult> tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

Value materializeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
                            const SliceParameters &sliceParams) {
  return utils::getSlice(builder, loc, valueToTile, sliceParams.offsets,
                         sliceParams.sizes, sliceParams.strides);
}

SmallVector<std::optional<SliceParameters>> computeAllSliceParameters(
    OpBuilder &builder, Location loc, HIVMStructuredOp hivmOp,
    ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds,
    bool omitPartialTileCheck) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](OpFoldResult v) { return !isZeroIndex(v); })) &&
         "expected as many ivs as non-zero sizes");

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<OpFoldResult> lbs =
      computeTileOffsets(builder, loc, ivs, tileSizes);
  SmallVector<OpFoldResult> subShapeSizes =
      computeTileSizes(builder, loc, tileSizes, sizeBounds);

  assert(static_cast<int64_t>(valuesToTile.size()) <=
             hivmOp->getNumOperands() &&
         "more value to tile than operands.");
  SmallVector<std::optional<SliceParameters>> allSliceParams;
  allSliceParams.reserve(valuesToTile.size());
  for (auto [opOperand, val] :
       llvm::zip(hivmOp->getOpOperands(), valuesToTile)) {
    Value shapedOp = val;
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for operand " << shapedOp);
    AffineMap map = hivmOp.getMatchingIndexingMap(&opOperand);
    // Use `opOperand` as is if it is not tiled and not an output tensor. Having
    // an extract/insert slice pair for all output tensors simplifies follow up
    // transformations such as padding and bufferization since the
    // extract/insert slice pairs make the accessed iteration argument
    // subdomains explicit.

    Type operandType = opOperand.get().getType();
    if (!isTiled(map, tileSizes) &&
        !(isa<RankedTensorType>(operandType) && hivmOp.isDpsInit(&opOperand))) {
      allSliceParams.push_back(std::nullopt);
      LLVM_DEBUG(llvm::dbgs()
                 << ": not tiled: use shape: " << operandType << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subshape...\n");

    allSliceParams.push_back(computeSliceParameters(
        builder, loc, shapedOp, tileSizes, map, lbs, sizeBounds, subShapeSizes,
        omitPartialTileCheck));
  }

  return allSliceParams;
}

SmallVector<Value>
makeTiledShapes(OpBuilder &builder, Location loc, HIVMStructuredOp hivmOp,
                ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
                ArrayRef<OpFoldResult> tileSizes,
                ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck) {
  SmallVector<std::optional<SliceParameters>> allSliceParameter =
      computeAllSliceParameters(builder, loc, hivmOp, valuesToTile, ivs,
                                tileSizes, sizeBounds, omitPartialTileCheck);
  SmallVector<Value> tiledShapes;
  for (auto item : llvm::zip(valuesToTile, allSliceParameter)) {
    Value valueToTile = std::get<0>(item);
    std::optional<SliceParameters> sliceParams = std::get<1>(item);
    tiledShapes.push_back(
        sliceParams.has_value()
            ? materializeTiledShape(builder, loc, valueToTile, *sliceParams)
            : valueToTile);
  }
  return tiledShapes;
}

/// Return the SSA values that represent the data point accessed using a given
/// `indexingMap` for a given point in the iteration space represented by `ivs`.
SmallVector<Value> getIndicesForAccess(OpBuilder &b, Location loc,
                                       AffineMap indexingMap, ValueRange ivs) {
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

/// Method to inline the payload of a `hivmOp` given the iteration space
/// point and values for the arguments of the payload.
LogicalResult inlinePayload(OpBuilder &b, HIVMStructuredOp hivmOp,
                            ValueRange ivs, ValueRange argValues) {
  Block *body = hivmOp->getBlock();
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
    OpOperand *storeInto = hivmOp.getDpsInitOperand(operand.index());
    auto indices = getIndicesForAccess(
        b, loc, hivmOp.getMatchingIndexingMap(storeInto), ivs);
    b.create<memref::StoreOp>(loc, toStore,
                              hivmOp.getDpsInitOperand(operand.index())->get(),
                              indices);
  }
  return success();
}

utils::IteratorType convertToLinalgIteratorType(hivm::IteratorType itType) {
  switch (itType) {
  case (hivm::IteratorType::kParallel):
  case (hivm::IteratorType::kBroadcast):
    return utils::IteratorType::parallel;
  case (hivm::IteratorType::kReduction):
    return utils::IteratorType::reduction;
  case (hivm::IteratorType::kConcat):
    return utils::IteratorType::concat;
  default:
    llvm_unreachable("Unhandled iterator types");
  };
}

SmallVector<utils::IteratorType> convertToLinalgIteratorTypes(
    const SmallVector<hivm::IteratorType> &iteratorTypes) {
  SmallVector<utils::IteratorType> linalgIteratorTypes;
  for (auto currentType : iteratorTypes) {
    linalgIteratorTypes.push_back(convertToLinalgIteratorType(currentType));
  }
  return linalgIteratorTypes;
}

} // namespace tiling_helper
} // namespace hivm
} // namespace mlir
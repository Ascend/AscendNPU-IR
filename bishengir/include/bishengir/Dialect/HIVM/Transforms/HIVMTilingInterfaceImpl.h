//===- HIVMTilingInterfaceImpl.h - Implementation of TilingInterface ------===//
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
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/Utils/Utils.cpp
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMTILINGINTERFACEIMPL_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMTILINGINTERFACEIMPL_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>
namespace mlir {
class DialectRegistry;

namespace hivm {
namespace tiling_helper {

using namespace mlir::linalg;
using namespace mlir::hivm;

// Helper visitor to determine whether an AffineExpr is tiled.
// This is achieved by traversing every AffineDimExpr with position `pos` and
// checking whether the corresponding `tileSizes[pos]` is non-zero.
// This also enforces only positive coefficients occur in multiplications.
//
// Example:
//   `d0 + 2 * d1 + d3` is tiled by [0, 0, 0, 2] but not by [0, 0, 2, 0]
//
struct TileCheck : public AffineExprVisitor<TileCheck> {
  TileCheck(ArrayRef<OpFoldResult> tileSizes) : tileSizes(tileSizes) {}

  void visitDimExpr(AffineDimExpr expr) {
    isTiled |= !isZeroIndex(tileSizes[expr.getPosition()]);
  }
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
    if (expr.getKind() == mlir::AffineExprKind::Mul)
      assert(cast<AffineConstantExpr>(expr.getRHS()).getValue() > 0 &&
             "nonpositive multiplying coefficient");
  }
  bool isTiled = false;
  ArrayRef<OpFoldResult> tileSizes;
};

SmallVector<Type> getTensorOutputTypes(HIVMStructuredOp op,
                                       ValueRange operands);

SmallVector<Value> insertSlicesBack(OpBuilder &builder, Location loc,
                                    HIVMStructuredOp op, ValueRange operands,
                                    ValueRange results);

bool isTiled(AffineExpr expr, ArrayRef<OpFoldResult> tileSizes);

// Checks whether the `map  varies with respect to a non-zero `tileSize`.
bool isTiled(AffineMap map, ArrayRef<OpFoldResult> tileSizes);

Value materializeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
                            const SliceParameters &sliceParams);
SmallVector<std::optional<SliceParameters>> computeAllSliceParameters(
    OpBuilder &builder, Location loc, HIVMStructuredOp hivmOp,
    ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds,
    bool omitPartialTileCheck);

SmallVector<Value>
makeTiledShapes(OpBuilder &builder, Location loc, HIVMStructuredOp hivmOp,
                ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
                ArrayRef<OpFoldResult> tileSizes,
                ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck);

/// Return the SSA values that represent the data point accessed using a given
/// `indexingMap` for a given point in the iteration space represented by `ivs`.
SmallVector<Value> getIndicesForAccess(OpBuilder &b, Location loc,
                                       AffineMap indexingMap, ValueRange ivs);

/// Method to inline the payload of a `hivmOp` given the iteration space
/// point and values for the arguments of the payload.
LogicalResult inlinePayload(OpBuilder &b, HIVMStructuredOp hivmOp,
                            ValueRange ivs, ValueRange argValues);

utils::IteratorType convertToLinalgIteratorType(hivm::IteratorType itType);

SmallVector<utils::IteratorType> convertToLinalgIteratorTypes(
    const SmallVector<hivm::IteratorType> &iteratorTypes);

} // namespace tiling_helper
void registerTilingInterfaceExternalModels(DialectRegistry &registry);
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMTILINGINTERFACEIMPL_H

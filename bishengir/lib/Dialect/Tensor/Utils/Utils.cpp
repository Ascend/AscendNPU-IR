//===-----------------------------Utils.cpp--------------------------------===//
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

#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "bishengir-tensor-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace tensor {
namespace reshape_utils {

using Hyperrectangle = SmallVector<HyperrectangularSlice>;
std::optional<Hyperrectangle>
getHyperrectangleFromArray(int64_t superviewShape, int64_t offset, int64_t size,
                           int64_t stride,
                           llvm::ArrayRef<int64_t> staticNewShape) {
  const int64_t n = static_cast<int64_t>(staticNewShape.size());
  if (n == 0) {
    LDBG("n == 0, hyperrectangle failed");
    return std::nullopt; // Handle empty shape
  }
  // Validate new shape matches total elements
  int64_t totalElements = 1;
  for (const int64_t dim : staticNewShape) {
    LDBG("Static new shape: " << dim);
    totalElements *= dim;
  }
  if (totalElements != superviewShape) {
    LDBG("Total elements are not the same as the superviewShape "
         << totalElements << " " << superviewShape);
    llvm::report_fatal_error("Total elements are not the same as the superviewShape");
  }
  // Compute row-major strides (step sizes between dimensions)
  llvm::SmallVector<int64_t> computedStrides(n, 1);
  for (int64_t i = n - 2; i >= 0; --i)
    computedStrides[i] = computedStrides[i + 1] * staticNewShape[i + 1];

  // Coordinate conversion helper (between old and new dimensions)
  auto unravel = [&](int64_t flatIndex) -> llvm::SmallVector<int64_t> {
    llvm::SmallVector<int64_t> coords(n, 0);
    for (int64_t i = 0; i < n; ++i) {
      coords[i] = flatIndex / computedStrides[i];
      flatIndex %= computedStrides[i];
    }
    return coords;
  };

  // Get start/end coordinates
  const int64_t flatStart = offset;
  const int64_t flatEnd = offset + (size - 1) * stride;
  if (flatEnd >= superviewShape)
    return std::nullopt; // Out of buffer check

  const auto lCoords = unravel(flatStart);
  const auto rCoords = unravel(flatEnd);

  LDBG("Ok in " << superviewShape << " " << offset << " " << size << " "
                << stride);
#ifndef NDEBUG
  for (auto l : lCoords) {
    LDBG("Dimension start: " << l);
  }
  for (auto r : rCoords) {
    LDBG("Dimension end: " << r);
  }
#endif
  if (size == 1) {
    Hyperrectangle result;
    for (int64_t i = 0; i < n; ++i)
      result.emplace_back(i, lCoords[i], 1, 0);
    return result;
  }

  // Find first differing dimension
  int64_t firstDiff = -1;
  for (int64_t i = 0; i < n; ++i) {
    if (lCoords[i] != rCoords[i]) {
      firstDiff = i;
      break;
    }
  }
  if (firstDiff == -1)
    return std::nullopt;

  LDBG("First diff is " << firstDiff);
  for (int64_t i = firstDiff + 1; i < n; ++i) {
    const bool validStart = (lCoords[i] == 0);
    const bool validEnd = (rCoords[i] == staticNewShape[i] - 1);
    if (!validStart || !validEnd) {
      LDBG("Not fully covered trailing");
      return std::nullopt;
    }
  }

  // Verify size matches coordinate span
  const int64_t span = rCoords[firstDiff] - lCoords[firstDiff] + 1;
  const int64_t trailingElements = computedStrides[firstDiff];

  LDBG("span : " << span << ", trailing: " << trailingElements << ", size "
                 << size);
  if (span * trailingElements != size) {
    LDBG("Coordinate span doesn't match");
    return std::nullopt;
  }

  // Build hyperrectangle description
  Hyperrectangle result;
  for (int64_t i = 0; i < n; ++i) {
    if (i < firstDiff) {
      // Fixed dimensions (size 1)
      result.emplace_back(i, lCoords[i], 1, 1);
    } else if (i == firstDiff) {
      // Main varying dimension
      result.emplace_back(i, lCoords[i], span, 1);
    } else {
      // Fully covered trailing dimensions
      result.emplace_back(i, 0, staticNewShape[i], 1);
    }
  }

  return result;
}

/// Creates an inverse ExpandShapeOp for a given CollapseShapeOp.
/// This allows undoing the effect of a collapse operation.
///
/// @param builder The builder to use for creating operations
/// @param collapseOp The CollapseShapeOp to invert
/// @return A new ExpandShapeOp that inverts the collapse
tensor::ExpandShapeOp
createCollapseInverse(OpBuilder &builder, tensor::CollapseShapeOp collapseOp) {
  return builder.create<tensor::ExpandShapeOp>(
      collapseOp.getLoc(), collapseOp.getSrcType(), collapseOp.getResult(),
      collapseOp.getReassociationIndices());
}

/// Creates an inverse CollapseShapeOp for a given ExpandShapeOp.
/// This allows undoing the effect of an expand operation.
///
/// @param builder The builder to use for creating operations
/// @param expandOp The ExpandShapeOp to invert
/// @return A new CollapseShapeOp that inverts the expand
tensor::CollapseShapeOp createExpandInverse(OpBuilder &builder,
                                            tensor::ExpandShapeOp expandOp) {
  return builder.create<tensor::CollapseShapeOp>(
      expandOp.getLoc(), expandOp.getSrcType(), expandOp.getResult(),
      expandOp.getReassociationIndices());
}

Value getReverseReshapedValue(OpBuilder &builder, Value initialValue,
                              const SmallVector<Operation *> &trace) {
  Value result = initialValue;
  for (Operation *op : trace) {
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
      result = builder.create<tensor::CollapseShapeOp>(
          result.getLoc(), /*resultType=*/expandOp.getSrcType(),
          /*src=*/result,
          /*reassociation=*/expandOp.getReassociation());
    } else if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
      result = builder.create<tensor::ExpandShapeOp>(
          result.getLoc(), /*resultType=*/collapseOp.getSrcType(),
          /*src=*/result,
          /*reassociation=*/collapseOp.getReassociationIndices());
    } else {
      llvm::report_fatal_error(
          "only support reshape Op including tensor::ExpandShapeOp "
          "and tensor::CollapseShapeOp");
    }
  }
  return result;
}

} // namespace reshape_utils

static Value reifyTensorDim(tensor::DimOp dimOp, OpBuilder &builder,
                            DenseMap<Value, Value> &settled) {
  // func.func %arg0 : <AxBxf32>, %mysterylow : index, %mysteryhigh : index
  // {
  //      %unaried = linalg.unary %arg0

  //      %myPad = tensor.pad %unaried low[3, %mysterylow]
  //      high[%mysteryhigh, 5] ->
  //              (?x?xf32)
  // }

  // Replace the usage of this using reify
  // %dim_a = tensor.dim %unaried, %c0 // A
  // %reify_res_a_0 = arith.add(%dim_a, 3)
  // %reify_res_a_1 = arith.add(%reify_res_a_0, %mysterylow)

  auto constIndex = dimOp.getConstantIndex();
  if (!constIndex.has_value()) {
    LDBG("WARN: Dynamic tensor.dim cannot be handled");
    return dimOp.getResult();
  }
  auto dimSrc = dimOp.getSource();
  if (isa<BlockArgument>(dimSrc)) {
    // Graceful return if its a dim on the argument
    // %dim_a = tensor.dim %arg0, %c0 // A
    return dimOp.getResult();
  }
  // Else try to reify
  if (auto reifyableOp =
          dimSrc.getDefiningOp<ReifyRankedShapedTypeOpInterface>()) {
    auto opResult = cast<OpResult>(dimSrc);
    ReifiedRankedShapedTypeDims shapes;
    builder.setInsertionPoint(reifyableOp);
    auto res = reifyableOp.reifyResultShapes(builder, shapes);
    if (failed(res))
      return dimOp.getResult();
    // Result of reify, get on the result number size and tensor.dim index
    auto currentVal = shapes[opResult.getResultNumber()][constIndex.value()];

    Value materializedReify;
    if (auto constInt = getConstantIntValue(currentVal)) {
      builder.setInsertionPointToStart(reifyableOp->getBlock());
      materializedReify = builder.create<arith::ConstantIndexOp>(
          reifyableOp.getLoc(), constInt.value());
    } else {
      materializedReify = currentVal.get<Value>();
    }
    settled[materializedReify] = materializedReify;
    // Set insertion point before usage of this tensor.dim
    return materializedReify;
  }
  return dimOp.getResult();
}

Value reifyShapeToArg(Value initialVal, std::optional<OpOperand *> opOpr,
                      OpBuilder &builder, DenseMap<Value, Value> &settled) {
  LDBG("Chain called " << initialVal);
  if (isa<BlockArgument>(initialVal))
    return initialVal;

  // If this has NOT been settled before (There exist a mapping which leads to
  // arg)
  if (!settled.contains(initialVal)) {
    OpBuilder::InsertionGuard guard(builder);
    if (!initialVal.getType().isIntOrIndex()) {
      LDBG("WARN: opOpr should not be shapes (unless its a blockArgument)"
           << initialVal);
    }
    if (auto dimOp = initialVal.getDefiningOp<tensor::DimOp>()) {
      settled[initialVal] = reifyTensorDim(dimOp, builder, settled);
    }
  }

  if (!settled.contains(initialVal)) {
    settled[initialVal] = initialVal;
  }

  Value nextVal = settled[initialVal];
  if (opOpr.has_value()) {
    builder.setInsertionPoint(opOpr.value()->getOwner());
    opOpr.value()->set(nextVal);
  }
  // Check if this is from tensor.dim

  if (auto *nextOp = nextVal.getDefiningOp()) {
    for (auto &nextOpOpr : nextOp->getOpOperands()) {
      reifyShapeToArg(nextOpOpr.get(), &nextOpOpr, builder, settled);
    }
  }
  return settled[initialVal];
}
} // namespace tensor
} // namespace mlir
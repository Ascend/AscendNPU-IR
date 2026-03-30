//===-----------------------------Utils.cpp--------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
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

std::string stringtifyElementKind(ElementKind kind) {
    switch (kind) {
        case ElementKind::HasMutation: return "HasMutation";
        case ElementKind::NoMutation:  return "NoMutation";
        case ElementKind::Unit:        return "Unit";
    }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ElementKind kind) {
    os << stringtifyElementKind(kind);
    return os;
}

using Hyperrectangle = SmallVector<HyperrectangularSlice>;
std::optional<Hyperrectangle>
getExtendHyperrectangleFromArray(int64_t superviewShape, int64_t offset,
                                 int64_t size, int64_t stride,
                                 llvm::ArrayRef<int64_t> staticNewShape) {
  const int64_t n = static_cast<int64_t>(staticNewShape.size());
  if (n == 0) {
    LDBG("n == 0, ExtendHyperrectangle failed");
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
    llvm_unreachable("Total elements are not the same as the superviewShape");
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

  // Collect all coordinates
  llvm::SmallVector<llvm::SmallVector<int64_t>> allCoords;
  allCoords.reserve(size);

  for (int64_t k = 0; k < size; ++k) {
    int64_t idx = offset + k * stride;
    if (idx >= superviewShape) {
      return std::nullopt;
    }
    allCoords.push_back(unravel(idx));
  }

  if (size == 1) {
    Hyperrectangle result;
    for (int64_t d = 0; d < n; ++d) {
      result.emplace_back(d, allCoords[0][d], 1, 1);
    }
    return result;
  }

  llvm::SmallVector<llvm::SmallVector<int64_t>> dimValues(n);
  for (const auto &coord : allCoords) {
    for (int64_t d = 0; d < n; ++d) {
      dimValues[d].push_back(coord[d]);
    }
  }

  Hyperrectangle result;
  int64_t totalProduct = 1;

  for (int64_t d = 0; d < n; ++d) {
    auto &vals = dimValues[d];
    // Deduplicate and sort
    std::set<int64_t> uniqueSet(vals.begin(), vals.end());
    llvm::SmallVector<int64_t> sorted(uniqueSet.begin(), uniqueSet.end());
    std::sort(sorted.begin(), sorted.end());

    if (sorted.size() == 1) {
      result.emplace_back(d, sorted[0], 1, 1);
      continue;
    }

    // Check if the intervals are in arithmetic progression
    int64_t dimStride = sorted[1] - sorted[0];
    if (dimStride <= 0)
      return std::nullopt;

    for (size_t i = 2; i < sorted.size(); ++i) {
      if (sorted[i] - sorted[i - 1] != dimStride) {
        return std::nullopt;
      }
    }

    int64_t dimOffset = sorted[0];
    int64_t dimSize = (int64_t)sorted.size();

    if (dimOffset + (dimSize - 1) * dimStride >= staticNewShape[d]) {
      return std::nullopt;
    }

    result.emplace_back(d, dimOffset, dimSize, dimStride);
    totalProduct *= dimSize;
  }

  // Verify size matches coordinate span
  if (totalProduct != size) {
    return std::nullopt;
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
      llvm_unreachable(
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
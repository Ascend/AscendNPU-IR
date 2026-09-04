//===------------------ Utils.cpp - HIVM Vectorize Utils-   ------------------===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMVectorize.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

#define DEBUG_TYPE "hivm-vectorize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir::hivm {
Value getIdentityElement(OpBuilder &builder, Location loc, Type elemType,
                         VectorArithKind kind) {
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    switch (kind) {
    case VectorArithKind::ADD:
    case VectorArithKind::SUB:
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(floatType, 0.0));
    case VectorArithKind::MUL:
    case VectorArithKind::DIV:
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(floatType, 1.0));
    case VectorArithKind::MAX:
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(
                   floatType, APFloat::getInf(floatType.getFloatSemantics(),
                                              /*Negative=*/true)));
    case VectorArithKind::MIN:
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(
                   floatType, APFloat::getInf(floatType.getFloatSemantics(),
                                              /*Negative=*/false)));
    }
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    switch (kind) {
    case VectorArithKind::ADD:
    case VectorArithKind::SUB:
      return builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(intType, 0));
    case VectorArithKind::MUL:
    case VectorArithKind::DIV:
      return builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(intType, 1));
    case VectorArithKind::MAX:
      return builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(
                   intType, APInt::getSignedMinValue(intType.getWidth())));
    case VectorArithKind::MIN:
      return builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(
                   intType, APInt::getSignedMaxValue(intType.getWidth())));
    }
  }
  llvm::report_fatal_error("unsupported element type for neutral element");
}

Value createVectorArithOp(OpBuilder &builder, Location loc,
                          VectorArithKind kind, Value lhs, Value rhs) {
  Type elemType = getElementTypeOrSelf(lhs.getType());

  if (isa<FloatType>(elemType)) {
    switch (kind) {
    case VectorArithKind::ADD:
      return builder.create<arith::AddFOp>(loc, lhs, rhs);
    case VectorArithKind::SUB:
      return builder.create<arith::SubFOp>(loc, lhs, rhs);
    case VectorArithKind::MUL:
      return builder.create<arith::MulFOp>(loc, lhs, rhs);
    case VectorArithKind::DIV:
      return builder.create<arith::DivFOp>(loc, lhs, rhs);
    case VectorArithKind::MAX:
      return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
    case VectorArithKind::MIN:
      return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
    }
  } else if (isa<IntegerType>(elemType)) {
    switch (kind) {
    case VectorArithKind::ADD:
      return builder.create<arith::AddIOp>(loc, lhs, rhs);
    case VectorArithKind::SUB:
      return builder.create<arith::SubIOp>(loc, lhs, rhs);
    case VectorArithKind::MUL:
      return builder.create<arith::MulIOp>(loc, lhs, rhs);
    case VectorArithKind::DIV:
      return builder.create<arith::DivSIOp>(loc, lhs, rhs); // or DivUIOp
    case VectorArithKind::MAX:
      return builder.create<arith::MaxSIOp>(loc, lhs, rhs);
    case VectorArithKind::MIN:
      return builder.create<arith::MinSIOp>(loc, lhs, rhs);
    }
  }

  llvm::report_fatal_error("unsupported element type for vector arithmetic");
}

LogicalResult checkVectorizePreconditions(Operation *op,
                                          ArrayRef<int64_t> vectorSizes) {
  auto structured = dyn_cast<HIVMStructuredOp>(op);
  if (!structured)
    return failure();
  if (!structured.getBroadcastArray().empty() ||
      !structured.getPermutationArray().empty())
    return failure();
  if (structured.getNumDpsInputs() == 0)
    return failure();
  auto shapedTy =
      dyn_cast<ShapedType>(structured.getDpsInputs().front().getType());
  if (!shapedTy ||
      shapedTy.getRank() != static_cast<int64_t>(vectorSizes.size()))
    return failure();
  if (llvm::any_of(vectorSizes, [](int64_t size) { return size <= 0; }))
    return failure();
  return success();
}

namespace {
unsigned getMaxElemBitWidth(HIVMStructuredOp op) {
  unsigned maxWidth = 0;
  for (Type type : op->getOperandTypes()) {
    Type elemTy = getElementTypeOrSelf(type);
    if (elemTy.isIndex())
      continue;
    unsigned width = elemTy.getIntOrFloatBitWidth();
    maxWidth = std::max(maxWidth, width);
  }

  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (hacc::utils::isAscend310B(module)) {
    // 300/310 does not support 64-bit types, using 32-bit instead
    return (maxWidth == 64) ? 32 : maxWidth;
  }
  return maxWidth;
}

std::optional<int64_t> getFirstNonUnitDim(ArrayRef<int64_t> shape) {
  if (shape.empty())
    return std::nullopt;
  int64_t rank = static_cast<int64_t>(shape.size());
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] > 1)
      return i;
  }
  return std::nullopt;
}

/// Loop extents used by the VL packing policy.
/// `computeStaticLoopSizes` asserts on dynamic shapes, so fall back to the
/// first DPS input (iteration-space rank) when any dim is dynamic.
SmallVector<int64_t> getLoopShape(HIVMStructuredOp op) {
  if (!op.hasDynamicShape())
    return SmallVector<int64_t>(op.computeStaticLoopSizes());
  if (op.getNumDpsInputs() == 0)
    return SmallVector<int64_t>(op.getNumLoops(), ShapedType::kDynamic);
  return SmallVector<int64_t>(op.getShape(op.getDpsInputOperand(0)));
}

// When the shape is dynamic, we only allow one dynamic dim, and the other dims
// should be unit dims. We will assign `capacity` as vector size for this
// dynamic dim, and other dims have vector size of one.
FailureOr<SmallVector<int64_t>>
computeDynamicVectorSizes(HIVMStructuredOp op, ArrayRef<int64_t> shape,
                          int64_t capacity) {
  int64_t rank = static_cast<int64_t>(shape.size());
  SmallVector<int64_t> vectorSizes(rank, 1);
  int64_t nonUnitDims = 0;
  for (int64_t i = rank - 1; i >= 0; --i) {
    if (shape[i] == 1)
      continue;
    nonUnitDims++;
    if (nonUnitDims >= 2)
      return op.emitError("Failed to compute dynamic vector sizes");
    vectorSizes[i] = capacity;
  }
  return vectorSizes;
}
} // namespace

FailureOr<SmallVector<int64_t>> computeVectorSizes(HIVMStructuredOp op) {
  SmallVector<int64_t> shape = getLoopShape(op);

  unsigned elemWidth = getMaxElemBitWidth(op);
  if (elemWidth <= 0)
    return op.emitError("Failed to compute max element bit width");
  int64_t elemWidthInBytes =
      llvm::divideCeil(elemWidth, mlir::utils::INTR_BITS_PER_BYTE);
  int64_t capacity = hivm::util::VL / elemWidthInBytes;
  LDBG("op shape: " << utils::debugger::to_string(shape));
  LDBG("vector capacity: " << capacity);

  if (op.hasDynamicShape())
    return computeDynamicVectorSizes(op, shape, capacity);

  int64_t rank = static_cast<int64_t>(shape.size());
  if (rank == 0)
    return op.emitError("Empty shape: rank is zero");
  auto first = getFirstNonUnitDim(shape);
  int64_t start = first.has_value() ? first.value() : rank - 1;
  int64_t end = rank - 1;
  int64_t remain = capacity;
  SmallVector<int64_t> vectorSizes(rank, 1);
  for (int64_t dim = end; dim >= start; dim--) {
    if (dim < 0 || static_cast<size_t>(dim) >= shape.size())
      return op.emitError("Invalid dimension index");
    if (shape[dim] <= 0)
      return op.emitError("Invalid shape dimension: must be positive");
    if (shape[dim] > remain)
      return op.emitError("Exceeds vector capacity");
    if (dim == start) {
      vectorSizes[dim] = remain;
      continue;
    }
    vectorSizes[dim] = shape[dim];
    remain /= shape[dim];
  }
  return vectorSizes;
}
} // namespace mlir::hivm

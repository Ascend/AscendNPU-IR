//===- Utils.cpp - HIVM vectorization utilities --------------------------===//
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

static SmallVector<Value> createZeroIndices(OpBuilder &builder, Location loc,
                                            int64_t rank) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  return SmallVector<Value>(rank, zero);
}

vector::TransferReadOp
createMaskedTransferRead(OpBuilder &builder, Location loc,
                         VectorType vectorType, Value source, Value padding,
                         Value mask, AffineMap permutationMap) {
  int64_t rank = vectorType.getRank();
  if (!permutationMap)
    permutationMap = builder.getMultiDimIdentityMap(rank);
  return builder.create<vector::TransferReadOp>(
      loc, vectorType, source, createZeroIndices(builder, loc, rank),
      permutationMap, padding, mask,
      builder.getBoolArrayAttr(SmallVector<bool>(rank, true)));
}

vector::TransferWriteOp createMaskedTransferWrite(OpBuilder &builder,
                                                  Location loc, Value vector,
                                                  Value destination, Value mask,
                                                  AffineMap permutationMap) {
  int64_t rank = cast<ShapedType>(destination.getType()).getRank();
  if (!permutationMap)
    permutationMap = builder.getMultiDimIdentityMap(rank);
  return builder.create<vector::TransferWriteOp>(
      loc, TypeRange(destination.getType()), vector, destination,
      createZeroIndices(builder, loc, rank), permutationMap, mask,
      builder.getBoolArrayAttr(SmallVector<bool>(rank, true)));
}

Value createShapeMask(OpBuilder &builder, Location loc, Value shaped,
                      ArrayRef<int64_t> vectorSizes) {
  auto shapedType = cast<ShapedType>(shaped.getType());
  SmallVector<Value> dimSizes;
  for (int64_t i = 0, e = shapedType.getRank(); i < e; ++i) {
    if (isa<TensorType>(shapedType))
      dimSizes.push_back(builder.create<tensor::DimOp>(loc, shaped, i));
    else
      dimSizes.push_back(builder.create<memref::DimOp>(loc, shaped, i));
  }
  return builder.create<vector::CreateMaskOp>(
      loc, VectorType::get(vectorSizes, builder.getI1Type()), dimSizes);
}

Value readOperand(OpBuilder &builder, Location loc, Value input,
                  ArrayRef<int64_t> vectorSizes, Value padding,
                  Value fullMask) {
  if (!isa<ShapedType>(input.getType()))
    return builder.create<vector::BroadcastOp>(
        loc, VectorType::get(vectorSizes, input.getType()), input);

  auto shapedType = cast<ShapedType>(input.getType());
  int64_t rank = static_cast<int64_t>(vectorSizes.size());
  assert(shapedType.getRank() == rank && "expected a checked operand rank");

  // A dim the operand carries at size one is read at size one and stretched
  // by vector.broadcast; every other dim -- including one narrower than the
  // vector shape, as the VL packing policy produces for leading dims -- is
  // read at the vector size and covered by `fullMask`.
  SmallVector<int64_t> readSizes(rank);
  for (int64_t i = 0; i < rank; ++i)
    readSizes[i] = shapedType.getDimSize(i) == 1 ? 1 : vectorSizes[i];
  bool needsBroadcast = readSizes != SmallVector<int64_t>(vectorSizes);

  // The narrow read only spans the operand's own extent, so it needs a mask
  // derived from the operand rather than from the op's iteration domain.
  Value mask = needsBroadcast ? createShapeMask(builder, loc, input, readSizes)
                              : fullMask;

  Type elementType = shapedType.getElementType();
  Value read = createMaskedTransferRead(builder, loc,
                                        VectorType::get(readSizes, elementType),
                                        input, padding, mask);
  if (!needsBroadcast)
    return read;
  return builder.create<vector::BroadcastOp>(
      loc, VectorType::get(vectorSizes, elementType), read);
}

LogicalResult checkVectorizePreconditions(Operation *op,
                                          ArrayRef<int64_t> vectorSizes) {
  auto structured = dyn_cast<HIVMStructuredOp>(op);
  if (!structured)
    return failure();
  if (structured.getNumDpsInputs() == 0 || structured.getNumDpsInits() != 1)
    return failure();
  auto output = structured.getDpsInitOperand(0)->get();
  auto outputType = dyn_cast<ShapedType>(output.getType());
  if (!outputType ||
      outputType.getRank() != static_cast<int64_t>(vectorSizes.size()))
    return failure();
  return success(
      llvm::none_of(vectorSizes, [](int64_t size) { return size <= 0; }));
}

LogicalResult checkShapedInputs(HIVMStructuredOp op,
                                ArrayRef<int64_t> vectorSizes) {
  // Checking all inputs here -- not just the first -- keeps these
  // vectorizers from bailing out after they have already emitted IR.
  return success(llvm::all_of(op.getDpsInputs(), [&](Value input) {
    auto shapedTy = dyn_cast<ShapedType>(input.getType());
    return shapedTy &&
           shapedTy.getRank() == static_cast<int64_t>(vectorSizes.size());
  }));
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

SmallVector<int64_t> getLoopShape(HIVMStructuredOp op) {
  if (!op.hasDynamicShape())
    return SmallVector<int64_t>(op.computeStaticLoopSizes());
  if (op.getNumDpsInputs() == 0 || op.getNumDpsInits() == 0)
    return SmallVector<int64_t>(op.getNumLoops(), ShapedType::kDynamic);

  // Reductions iterate over their input. Other supported ops iterate over the
  // destination, including broadcasts and transposes.
  OpOperand *domain = isa<VReduceOp>(op.getOperation())
                          ? op.getDpsInputOperand(0)
                          : op.getDpsInitOperand(0);
  return SmallVector<int64_t>(op.getShape(domain));
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

//===- HIVMVectorize.h - HIVM dialect trait definitions ---------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMVECTORIZE_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMVECTORIZE_H

#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::hivm {

vector::TransferReadOp
createMaskedTransferRead(OpBuilder &builder, Location loc,
                         VectorType vectorType, Value source, Value padding,
                         Value mask, AffineMap permutationMap = {});

vector::TransferWriteOp
createMaskedTransferWrite(OpBuilder &builder, Location loc, Value vector,
                          Value destination, Value mask,
                          AffineMap permutationMap = {});

/// Create mask over a `vectorSizes`-shaped vector that covers exactly the
/// extent of `shaped`, whose rank must be `vectorSizes.size()`.
Value createShapeMask(OpBuilder &builder, Location loc, Value shaped,
                      ArrayRef<int64_t> vectorSizes);

/// Read an operand into a vector, handling scalar splats and shaped
Value readOperand(OpBuilder &builder, Location loc, Value input,
                  ArrayRef<int64_t> vectorSizes, Value padding, Value fullMask);

/// Rejects what no vectorize() model serves: a non-structured op, missing DPS
/// inputs, multiple or missing outputs, a rank mismatch, or a non-positive
/// vector size. Broadcast operands are read at their own shape and stretched
/// as needed.
LogicalResult checkVectorizePreconditions(Operation *op,
                                          ArrayRef<int64_t> vectorSizes);

/// Require every DPS input to be shaped with the rank of `vectorSizes`.
/// Lowerings that support scalar operands through `readOperand` should not
/// call this check.
LogicalResult checkShapedInputs(HIVMStructuredOp op,
                                ArrayRef<int64_t> vectorSizes);

/// VL packing policy used by `hivm-vectorize-ops` and
/// `transform.hivm.vectorize`. Capacity is `hivm::util::VL` bytes / element
/// width (64 lanes for f32).
FailureOr<SmallVector<int64_t>> computeVectorSizes(HIVMStructuredOp op);
} // namespace mlir::hivm

#endif

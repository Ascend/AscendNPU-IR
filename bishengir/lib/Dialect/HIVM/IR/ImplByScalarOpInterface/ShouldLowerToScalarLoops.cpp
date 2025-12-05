//===- ShouldLowerToScalarLoops.cpp - HIVM should lower to scalar check ---===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/IR/TypeUtilities.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::hivm;

namespace mlir::hivm {

template <typename HIVMOP>
bool shouldCumOpLowerToScalarLoops(HIVMOP op) {
  if (!op.hasPureBufferSemantics()) {
    return false;
  }
  auto cumDims = op.getCumDims();
  if (cumDims.size() > 1) {
    // only support to lower to scalar ops for cum op with unique cum dim
    return false;
  }

  auto elemType = getElementTypeOrSelf(op.getDst());
  if (elemType.isInteger(64)) {
    return true;
  }

  // if it is last cum op with i64 elem type, lower to scalar ops
  auto hivmFlattenInterfaceOp = cast<hivm::FlattenInterface>(op.getOperation());
  FlattenOptions flattenOptions;
  flattenOptions.checkMarkStride = true;
  auto flattenResult = hivmFlattenInterfaceOp.getFlattened(flattenOptions);
  assert(succeeded(flattenResult));
  auto flattenedCumDims = flattenResult->barrierDims;
  assert(flattenedCumDims.size() == 1);
  auto flattenedRank = flattenResult->getRankAfterFlatten();
  return flattenedCumDims[0] == flattenedRank - 1;
}

} // namespace mlir::hivm

namespace mlir::hivm {

template <typename HIVMOP>
bool shouldModOpLowerToScalarLoops(HIVMOP op) {
  if (!op.hasPureBufferSemantics()) {
    return false;
  }
  
  if (op.hasHWUnsupportedScalarOperand()) {
    return true;
  }

  auto elemType = getElementTypeOrSelf(op.getOperandTypes()[0]);
  return elemType.isInteger(64) || elemType.isInteger(32);
}

} // namespace mlir::hivm

//===----------------------------------------------------------------------===//
// Macros to help generate `shouldLowerToScalarLoops`
//===----------------------------------------------------------------------===//

#define ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(OP_NAME)           \
  bool OP_NAME::shouldLowerToScalarLoops() {                                   \
    if (!hasPureBufferSemantics()) {                                           \
      return false;                                                            \
    }                                                                          \
                                                                               \
    if (hasHWUnsupportedScalarOperand())                                       \
      return true;                                                             \
                                                                               \
    auto elemType = getElementTypeOrSelf(getOperandTypes()[0]);                \
    return elemType.isInteger(64);                                             \
  }

#define ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(OP_NAME)               \
  bool OP_NAME::shouldLowerToScalarLoops() {                                   \
    return shouldCumOpLowerToScalarLoops(*this);                               \
  }
#define ENABLE_MOD_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(OP_NAME)               \
  bool OP_NAME::shouldLowerToScalarLoops() {                                   \
    return shouldModOpLowerToScalarLoops(*this);                               \
  }                                                                            \


ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VInterleaveOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VDeinterleaveOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VMulOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VAddOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VSubOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VMinOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VMaxOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VAbsOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VShLOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VShROp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VDivOp)
#undef ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL

ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VCumprodOp)
ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VCumsumOp)
#undef ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL

ENABLE_MOD_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VModOp)
#undef ENABLE_MOD_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL

//===----------------------------------------------------------------------===//
// VCmpOp
//===----------------------------------------------------------------------===//

namespace mlir::hivm {

bool shouldVCmpOpLowerToScalarLoopsImpl(VCmpOp op) {
  if (!op.hasPureBufferSemantics()) {
    return false;
  }
  Type srcType = op.getOperand(0).getType();
  if (!isa<MemRefType>(srcType) && !isa<TensorType>(srcType)) {
    return false;
  }
  if (!getElementTypeOrSelf(srcType).isInteger()) {
    return false;
  }

  CompareMode cmpMode = op.getCompareMode();
  return !getElementTypeOrSelf(srcType).isInteger(32) ||
         (cmpMode != CompareMode::NE && cmpMode != CompareMode::EQ);
}

} // namespace mlir::hivm

bool VCmpOp::shouldLowerToScalarLoops() {
  return shouldVCmpOpLowerToScalarLoopsImpl(*this);
}

//===----------------------------------------------------------------------===//
// VMulExtOp
//===----------------------------------------------------------------------===//

bool VMulExtOp::shouldLowerToScalarLoops() {
  if (!hasPureBufferSemantics()) {
    return false;
  }
  auto elemType = getElementTypeOrSelf(getOperandTypes()[0]);
  return elemType.isInteger(32) || elemType.isInteger(64);
}

//===----------------------------------------------------------------------===//
// VReduceOp
//===----------------------------------------------------------------------===//

namespace mlir::hivm {

static bool processIndexLeft(VReduceOp op, Type elemType) {
  // lower reduce_with_index op with integer-type src
  if (elemType.isInteger(64) || elemType.isInteger(32) ||
      elemType.isInteger(16)) {
    return true;
  }
  
  // lower reduce_with_index op with 3 or more dims
  if (elemType.isF16() || elemType.isF32() || elemType.isBF16()) {
    auto hivmFlattenInterfaceOp =
        cast<hivm::FlattenInterface>(op.getOperation());
    FlattenOptions flattenOptions;
    flattenOptions.checkMarkStride = true;
    auto flatttenResult = hivmFlattenInterfaceOp.getFlattened(flattenOptions);
    assert(succeeded(flatttenResult));
    auto flattenRank = flatttenResult->getRankAfterFlatten();
    return flattenRank > 2;
  }

  return false;
}

static bool processIndexRight(VReduceOp op, Type elemType) {
  MemRefType srcVecType = cast<MemRefType>(op.getSrc().getType());
  int rank = srcVecType.getRank();
  llvm::ArrayRef<int64_t> reduceDims = op.getReduceDims();
  assert(reduceDims.size() == 1 &&
         "reduce dimensions array is not decomposed yet");
  bool lastAxis = reduceDims[0] == rank - 1;
  bool isROrAR = rank == 1 || lastAxis;

  // lower reduce_with_index op with integer-type src
  // lower rightmost type reduce_with_index op in R and AR condition
  if (elemType.isInteger(64) || elemType.isInteger(32) ||
      elemType.isInteger(16) ||
      (isROrAR &&
       (elemType.isF16() || elemType.isF32() || elemType.isBF16()))) {

    return true;
  }

  // lower reduce_with_index op with 3 or more dims
  if (elemType.isF16() || elemType.isF32() || elemType.isBF16()) {
    auto hivmFlattenInterfaceOp =
        cast<hivm::FlattenInterface>(op.getOperation());
    FlattenOptions flattenOptions;
    flattenOptions.checkMarkStride = true;
    auto flatttenResult = hivmFlattenInterfaceOp.getFlattened(flattenOptions);
    assert(succeeded(flatttenResult));
    auto flattenRank = flatttenResult->getRankAfterFlatten();
    return flattenRank > 2;
  }

  return false;
}

bool shouldVReduceOpDecomposeToScalarImpl(VReduceOp op) {
  auto reduceOpArith = op.getArithAttr();
  auto reduceOpAttr = reduceOpArith.getReduceOp();
  auto elemType = getElementTypeOrSelf(op.getOperandTypes()[0]);
  bool shouldDecomposeToScalar = false;
  switch (reduceOpAttr) {
  case hivm::ReduceOperation::min:
  case hivm::ReduceOperation::max:
  case hivm::ReduceOperation::sum:
  case hivm::ReduceOperation::prod:
  case hivm::ReduceOperation::xori:
    shouldDecomposeToScalar = elemType.isInteger(64);
    break;
  case hivm::ReduceOperation::max_with_index_left:
  case hivm::ReduceOperation::min_with_index_left: {
    shouldDecomposeToScalar = processIndexLeft(op, elemType);
    break;
  }
  case hivm::ReduceOperation::max_with_index_right:
  case hivm::ReduceOperation::min_with_index_right: {
    shouldDecomposeToScalar = processIndexRight(op, elemType);
    break;
  }
  default:
    break;
  }

  return shouldDecomposeToScalar;
}

} // namespace mlir::hivm

bool VReduceOp::shouldLowerToScalarLoops() {
  if (!this->hasPureBufferSemantics()) {
    return false;
  }
  return shouldVReduceOpDecomposeToScalarImpl(*this);
}

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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/IR/TypeUtilities.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::hivm;

namespace mlir::hivm {

template <typename HIVMOP>
bool shouldCumOpLowerToScalarLoops_membase(HIVMOP op) {
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
bool shouldModOpLowerToScalarLoops_membase(HIVMOP op) {
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

// Membase version of default macro
#define ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(OP_NAME)      \
  bool OP_NAME::shouldLowerToScalarLoops_membase() {                              \
    if (!hasPureBufferSemantics()) {                                               \
      return false;                                                                \
    }                                                                              \
                                                                                   \
    if (hasHWUnsupportedScalarOperand())                                           \
      return true;                                                                 \
                                                                                   \
    auto elemType = getElementTypeOrSelf(getOperandTypes()[0]);                    \
    return elemType.isInteger(64);                                                 \
  }

// Regbase version of default macro
#define ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(OP_NAME)      \
  bool OP_NAME::shouldLowerToScalarLoops_regbase() {                               \
    if (!hasPureBufferSemantics()) {                                               \
      return false;                                                                \
    }                                                                              \
    auto elemType = getElementTypeOrSelf(getOperandTypes()[0]);                    \
    return util::isSIMTVF(getOperation()) ||                                       \
           elemType.isInteger(64);                                                 \
  }

ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VInterleaveOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VDeinterleaveOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VMulOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VAddOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VSubOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VMinOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VMaxOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VAbsOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VShLOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VShROp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VDivOp)

ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VInterleaveOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VDeinterleaveOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VMulOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VAddOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VSubOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VMinOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VMaxOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VAbsOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VShLOp)
ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE(VShROp)

#undef ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE
#undef ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_REGBASE

// Default wrapper: call membase version
bool VInterleaveOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VDeinterleaveOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VMulOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VAddOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VSubOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VMinOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VMaxOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VAbsOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VShLOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VShROp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VDivOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }

//===----------------------------------------------------------------------===//
// VCumsumOp (membase version)
//===----------------------------------------------------------------------===//

#define ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(OP_NAME)          \
  bool OP_NAME::shouldLowerToScalarLoops_membase() {                              \
    return shouldCumOpLowerToScalarLoops_membase(*this);                           \
  }

ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VCumprodOp)
ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VCumsumOp)

#undef ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE

// Mod op (membase version)
#define ENABLE_MOD_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(OP_NAME)          \
  bool OP_NAME::shouldLowerToScalarLoops_membase() {                              \
    return shouldModOpLowerToScalarLoops_membase(*this);                           \
  }

ENABLE_MOD_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE(VModOp)
#undef ENABLE_MOD_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL_MEMBASE

//===----------------------------------------------------------------------===//
// VCumsumOp (regbase version - shared by cum ops lowered to rank/dim-specialized
// library calls)
//===----------------------------------------------------------------------===//

template <typename HIVMOP>
static bool shouldCumOpWithTempLowerToScalarLoops_regbase(HIVMOP op) {
  if (!op.hasPureBufferSemantics()) {
    return false;
  }
  auto cumDims = op.getCumDims();
  if (cumDims.size() > 1) {
    return false;
  }

  auto elemType = getElementTypeOrSelf(op.getDst());
  return elemType.isInteger(64);
}

bool VCumsumOp::shouldLowerToScalarLoops_regbase() {
  return shouldCumOpWithTempLowerToScalarLoops_regbase(*this);
}

bool VCummaxOp::shouldLowerToScalarLoops_regbase() {
  return shouldCumOpWithTempLowerToScalarLoops_regbase(*this);
}

bool VCumminOp::shouldLowerToScalarLoops_regbase() {
  return shouldCumOpWithTempLowerToScalarLoops_regbase(*this);
}

//===----------------------------------------------------------------------===//
// VCumprodOp (regbase version)
//===----------------------------------------------------------------------===//

bool VCumprodOp::shouldLowerToScalarLoops_regbase() {
  if (!hasPureBufferSemantics()) {
    return false;
  }
  auto cumDims = getCumDims();
  if (cumDims.size() > 1) {
    return false;
  }
  auto elemType = getElementTypeOrSelf(getDst());
  if (elemType.isInteger(64)) {
    return true;
  }
  return false;
}

// Cum ops wrappers
bool VCumsumOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }
bool VCumprodOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_membase(); }

bool VCummaxOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_regbase(); }
bool VCumminOp::shouldLowerToScalarLoops() { return shouldLowerToScalarLoops_regbase(); }

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
// VMulExtUiOp (membase only)
//===----------------------------------------------------------------------===//

bool VMulExtUiOp::shouldLowerToScalarLoops_membase() {
  if (!hasPureBufferSemantics()) {
    return false;
  }
  auto elemType = getElementTypeOrSelf(getOperandTypes()[0]);
  return elemType.isInteger(32) || elemType.isInteger(64);
}

bool VMulExtUiOp::shouldLowerToScalarLoops() {
  return shouldLowerToScalarLoops_membase();
}

//===----------------------------------------------------------------------===//
// VReduceOp (membase version - helpers)
//===----------------------------------------------------------------------===//

namespace mlir::hivm {

static bool processIndexLeft_membase(VReduceOp op, Type elemType) {
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

static bool processIndexRight_membase(VReduceOp op, Type elemType) {
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

bool shouldVReduceOpDecomposeToScalarImpl_membase(VReduceOp op) {
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
    shouldDecomposeToScalar = processIndexLeft_membase(op, elemType);
    break;
  }
  case hivm::ReduceOperation::max_with_index_right:
  case hivm::ReduceOperation::min_with_index_right: {
    shouldDecomposeToScalar = processIndexRight_membase(op, elemType);
    break;
  }
  default:
    break;
  }

  return shouldDecomposeToScalar;
}

//===----------------------------------------------------------------------===//
// VReduceOp (regbase version - helpers)
//===----------------------------------------------------------------------===//

// If strides are unknown geometry of tensor is marked illegal and
// max_with_index/min_with_index are lowered to loops
//
// Motivation: some passes do not add strides due to bug or intentionally. In
// that case it's always safe to lower to loops by default
static bool isLegalAccessAlignment_regbase(VReduceOp op, MemRefType in) {
  if (!in || in.getShape().size() == 1) {
    return true;
  }

  // rely on fact that flattening already happened
  auto isLeadingDimReduce = [in](int rd) {
    return rd == static_cast<int>(in.getShape().size()) - 1;
  };

  auto reduceDimValid = [in, &isLeadingDimReduce](int rd) {
    assert(rd >= 0);

    auto shape = in.getShape();
    auto sla = dyn_cast<StridedLayoutAttr>(in.getLayout());
    if (!sla) {
      // If there is no strided layout attr we can't confirm validness of
      // alignment, default value is "no, not aligned, so lower to loops"
      return false;
    }
    auto strides = sla.getStrides();
    assert(strides.size() == shape.size());
    auto N = shape.size();
    SmallVector<int, 4> dctrs(N, 0);

    // unaligned access cases could be detected by checking
    // dimension start and first element accesses without checking all elements positions
    // for leading dim reduce only dimension start need to be checked
    auto dimCheckLim = isLeadingDimReduce(rd) ? 1 : 2;
    auto increment = [&dctrs, dimCheckLim, N]() {
      for (int i = N - 1; i >= 0; i--) {
        if (++dctrs[i] < dimCheckLim) {
          break;
        }
        dctrs[i] = 0;
      }
    };

    auto numAccesses = 1 << shape.size();
    auto elemSize = in.getElementType().getIntOrFloatBitWidth() / 8;

    for (int i = 0; i < numAccesses; i++) {
      auto accessAddr = 0;
      for (int j = 0; j < static_cast<int>(N); j++) {
        accessAddr += dctrs[j] * strides[j] * static_cast<int>(elemSize);
      }
      if (accessAddr % util::BL != 0) {
        return false;
      }
      increment();
    }

    return true;
  };

  auto ord = op.getReduceDims();
  return std::all_of(ord.begin(), ord.end(),
                     [&reduceDimValid](int rd) { return reduceDimValid(rd); });
}

// This is workaround to avoid unaligned accesses in template lib-side code
// Will be removed when argmin/argmax code will be generated by compiler
// without relying on any template lib-side code
// https://codehub-y.huawei.com/CompilerKernel/BiShengCompiler/AscendNPU-IR/issues/440
static bool isTLReduceGeometryLegal_regbase(VReduceOp op) {
  auto legalizer = [&op](Value v) {
    return !isLegalAccessAlignment_regbase(op, dyn_cast<MemRefType>(v.getType()));
  };

  if (auto ops = op.getOperands();
      std::any_of(ops.begin(), ops.end(), legalizer)) {
    return false;
  }

  if (auto res = op.getResult();
      std::any_of(res.begin(), res.end(), legalizer)) {
    return false;
  }

  return true;
}

bool shouldVReduceOpDecomposeToScalarImpl_regbase(VReduceOp op) {
  auto mod = op->getParentOfType<ModuleOp>();
  auto reduceOpArith = op.getArithAttr();
  auto reduceOpAttr = reduceOpArith.getReduceOp();
  if (hacc::utils::isRegBasedArch(mod)) {
    switch (reduceOpAttr) {
      case hivm::ReduceOperation::max_with_index:
      case hivm::ReduceOperation::min_with_index:
        return !isTLReduceGeometryLegal_regbase(op);
      default:
        return false;
    }
  }
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
  case hivm::ReduceOperation::max_with_index:
  case hivm::ReduceOperation::min_with_index: {
    if (elemType.isInteger(64) || elemType.isInteger(32) ||
        elemType.isInteger(16)) {
      shouldDecomposeToScalar = true;
      break;
    }

    if (elemType.isF16() || elemType.isF32() || elemType.isBF16()) {
      auto hivmFlattenInterfaceOp =
          cast<hivm::FlattenInterface>(op.getOperation());
      FlattenOptions flattenOptions;
      flattenOptions.checkMarkStride = true;
      auto flatttenResult = hivmFlattenInterfaceOp.getFlattened(flattenOptions);
      assert(succeeded(flatttenResult));
      auto flattenRank = flatttenResult->getRankAfterFlatten();
      shouldDecomposeToScalar = flattenRank > 2;
      break;
    }

    shouldDecomposeToScalar = false;
    break;
  default:
    break;
  }
  }

  return shouldDecomposeToScalar;
}

} // namespace mlir::hivm

bool VReduceOp::shouldLowerToScalarLoops() {
  if (!this->hasPureBufferSemantics()) {
    return false;
  }
  return shouldVReduceOpDecomposeToScalarImpl_membase(*this);
}

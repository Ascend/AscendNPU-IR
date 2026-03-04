//===- ShouldLowerToScalarLoops.cpp - HIVM should lower to scalar check ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

//===----------------------------------------------------------------------===//
// Macros to help generate `shouldLowerToScalarLoops`
//===----------------------------------------------------------------------===//

#define ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(OP_NAME)           \
  bool OP_NAME::shouldLowerToScalarLoops() {                                   \
    if (!hasPureBufferSemantics()) {                                           \
      return false;                                                            \
    }                                                                          \
    auto elemType = getElementTypeOrSelf(getOperandTypes()[0]);                \
    return getOperation()->hasAttr(utils::simtLabel) ||                        \
           elemType.isInteger(64);                                             \
  }

#define ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(OP_NAME)               \
  bool OP_NAME::shouldLowerToScalarLoops() {                                   \
    return shouldCumOpLowerToScalarLoops(*this);                               \
  }

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
#undef ENABLE_DEFAULT_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL

ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VCumprodOp)
ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL(VCumsumOp)
#undef ENABLE_CUM_OP_SHOULD_LOWER_TO_SCALAR_LOOPS_IMPL

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

// If strides are unknown geometry of tensor is marked illegal and
// max_with_index/min_with_index are lowered to loops
//
// Motivation: some passes do not add strides due to bug or intentionally. In
// that case it's always safe to lower to loops by default
static bool isLegalAccessAlignment(VReduceOp op, MemRefType in) {
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
static bool isTLReduceGeometryLegal(VReduceOp op) {
  auto legalizer = [&op](Value v) { 
    return !isLegalAccessAlignment(op, dyn_cast<MemRefType>(v.getType())); 
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

bool shouldVReduceOpDecomposeToScalarImpl(VReduceOp op) {
  auto mod = op->getParentOfType<ModuleOp>();
  auto reduceOpArith = op.getArithAttr();
  auto reduceOpAttr = reduceOpArith.getReduceOp();
  if (hacc::utils::isRegBasedArch(mod)) {
    switch (reduceOpAttr) {
      case hivm::ReduceOperation::max_with_index:
      case hivm::ReduceOperation::min_with_index:
        return !isTLReduceGeometryLegal(op); 
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
  return shouldVReduceOpDecomposeToScalarImpl(*this);
}

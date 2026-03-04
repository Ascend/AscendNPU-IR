//===- OpAdjustment.cpp - Operation target dimensions adjustment ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#define DEBUG_TYPE "flatten-common"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::utils;
using namespace mlir::utils::debugger;

void VBrcOp::adjustTargetDimensions([[maybe_unused]] OpBuilder &builder,
                                    const FlattenResult &result) {
  auto &adjustedDims = result.adjustedTargetDims;
  setBroadcastDims(adjustedDims);
}

void VReduceOp::adjustTargetDimensions([[maybe_unused]] OpBuilder &builder,
                                       const FlattenResult &result) {
  auto &adjustedDims = result.adjustedTargetDims;
  setReduceDims(adjustedDims);
}

void VTransposeOp::adjustTargetDimensions([[maybe_unused]] OpBuilder &builder,
                                          const FlattenResult &result) {
  auto &adjustedDims = result.adjustedTargetDims;
  LDBG(to_string(adjustedDims));
  assert(adjustedDims.size() == 2);
  auto rank = getShapeRank(getSrc());
  SmallVector<int64_t> permutation(rank.value_or(0));
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[adjustedDims[0]], permutation[adjustedDims[1]]);
  setPermutation(permutation);
}

void VCumsumOp::adjustTargetDimensions([[maybe_unused]] OpBuilder &builder,
                                       const FlattenResult &result) {
  auto &adjustedDims = result.adjustedTargetDims;

  setCumDims(adjustedDims);
}

void VCumprodOp::adjustTargetDimensions([[maybe_unused]] OpBuilder &builder,
                                        const FlattenResult &result) {
  auto &adjustedDims = result.adjustedTargetDims;

  setCumDims(adjustedDims);
}

void VPadOp::adjustTargetDimensions([[maybe_unused]] OpBuilder &builder,
                                    const FlattenResult &result) {
  auto &adjustedDims = result.adjustedTargetDims;
  auto rank = result.getRankAfterFlatten();
  // get new indices
  auto reassociationMap =
      getReassociationMapping(result.getInputReassociation());
  SmallVector<int64_t> newStaticLow(rank, 0);
  SmallVector<int64_t> newStaticHigh(rank, 0);
  auto origStaticLow = getStaticLow();
  auto origStaticHigh = getStaticHigh();
  for (size_t i = 0; i < adjustedDims.size(); ++i) {
    auto &originalDim = result.originalTargetDims[i];
    newStaticLow[adjustedDims[i]] = origStaticLow[originalDim];
    newStaticHigh[adjustedDims[i]] = origStaticHigh[originalDim];
  }
  setStaticLow(newStaticLow);
  setStaticHigh(newStaticHigh);
  return;
}

void VConcatOp::adjustTargetDimensions([[maybe_unused]] OpBuilder &builder,
                                       const FlattenResult &result) {
  auto &adjustedDims = result.adjustedTargetDims;
  setDim(adjustedDims[0]);
}

void VFlipOp::adjustTargetDimensions([[maybe_unused]] OpBuilder &builder,
                                     const FlattenResult &result) {
  for (size_t i = 0; i < result.adjustedTargetDims.size(); ++i) {
    auto &originalDim = result.originalTargetDims[i];
    if (originalDim == ssize_t(getFlipAxis())) {
      setFlipAxis(result.adjustedTargetDims[i]);
      break;
    }
  }
}

namespace mlir::hivm::detail {
void adjustElementwiseTargetDimensions(OpBuilder &builder, HIVMStructuredOp op,
                                       const FlattenResult &result) {
  auto &adjustedDims = result.adjustedTargetDims;
  LDBG("Should be here");
  if (op.existInlineBroadcastLoopDims()) {
    LDBG("Should be here");
    auto arrayDims = builder.getDenseI64ArrayAttr(adjustedDims);
    LogicalResult setResult = op.setIteratorTypesArray(
        mlir::hivm::IteratorType::kBroadcast, arrayDims);
    if (failed(setResult)) {
      llvm::report_fatal_error("Failed to set iterator types array");
    }
  } else if (op.existInlineTransposeLoopDims()) {
    LDBG(to_string(adjustedDims));
    auto arrayDims = builder.getDenseI64ArrayAttr(adjustedDims);
    LogicalResult setResult = op.setIteratorTypesArray(
        mlir::hivm::IteratorType::kTranspose, arrayDims);
    if (failed(setResult)) {
      llvm::report_fatal_error("Failed to set iterator types array");
    }
  }
}
} // namespace mlir::hivm::detail

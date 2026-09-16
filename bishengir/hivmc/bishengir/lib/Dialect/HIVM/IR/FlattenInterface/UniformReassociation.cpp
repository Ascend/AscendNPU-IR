//===- UniformReassociation.cpp - Flatten computing for uniform reassociation //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#include <algorithm>
#define DEBUG_TYPE "flatten-rank-preserving"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::utils;
using namespace mlir::utils::debugger;

namespace mlir::hivm::detail {

FlattenResult collapseUniformReassociation(FlattenResult &payload,
                                           const FlattenOptions &options) {
  // Process operations after its unit-flattened, run it like a pipeline
  auto rank = payload.getRankAfterFlatten();
  LDBG("Rank is " << rank);
  BitVector targetMask =
      mlir::utils::arrayToMask(payload.adjustedTargetDims, rank);
  auto contiguousMask =
      getContiguousAxesImpl(payload.getOperandTypes(DpsKind::kDpsAll));
  BitVector inputConsistencyMask;
  if (options.checkInputConsistency) {
    inputConsistencyMask =
        getInputConsistencyMask(payload.getOperandTypes(DpsKind::kDpsInput));
  }
  LDBG("Collapsing uniform reassociation op");
  // Create reassociation indices
  SmallVector<ReassociationIndices> reassociation;
  bool isTargetCollapsible =
      payload.op->hasTrait<OpTrait::CollapsibleConsecutiveTargetDimsTrait>();
  for (int i = 0; i < rank; i++) {
    bool createNewReassociationGroup = false;
    // If it's the first one or it has different target group
    if (i == 0 || (targetMask.test(i - 1) != targetMask.test(i)))
      createNewReassociationGroup = true;
    // If it's target preserving (may not collapse consecutive target , like
    // permutation, then create a new group)
    if (!isTargetCollapsible && targetMask.test(i))
      createNewReassociationGroup = true;
    // If the layout is not contiguous
    if (!contiguousMask.test(i))
      createNewReassociationGroup = true;
    if (options.checkInputConsistency) {
      bool mergeWithBefore = inputConsistencyMask.test(i);
      if (i >= 1 && !inputConsistencyMask.test(i - 1)) {
        mergeWithBefore = false;
      }
      if (!mergeWithBefore)
        createNewReassociationGroup = true;
    }
    if (createNewReassociationGroup)
      reassociation.emplace_back();
    reassociation.back().push_back(i);
  }
  LLDBG(to_string(reassociation));
  FlattenResult result = collapseOperandsUniformly(payload, reassociation);
  auto mapping = getReassociationMapping(reassociation);
  result.adjustBarrierAndTargetDims(mapping);
  return result;
}

static FlattenResult
initializePayload(HIVMStructuredOp op, const FlattenOptions &options,
                  ArrayRef<int64_t> barrierDims,
                  std::optional<ArrayRef<int64_t>> adjustedDims) {
  assert(!op.existInlineTransposeLoopDims());
  // Flatten Unit dimensions first
  FlattenResult payload(
      op.getOperation()); // Create pipeline from this pipeline payload
  // Initialize payload
  payload.fillWithIdentity();
  payload.barrierDims = llvm::to_vector(barrierDims);
  if (adjustedDims.has_value())
    payload.originalTargetDims = payload.adjustedTargetDims =
        llvm::to_vector(adjustedDims.value());
  else
    payload.originalTargetDims = payload.adjustedTargetDims =
        llvm::to_vector(barrierDims);
  if (options.checkMarkStride) {
    payload = computeAnnotationMarkedOp(payload);
  }
  return payload;
}

/// Rank-preserving operation example:
/// Input shape:  [a, b, c, d, e]  // original tensor dimensions
/// Output shape: [a, B, C, d, E]  // reshaped tensor dimensions
///
/// Where:
/// - Lowercase letters (a, d) = dimensions that keep the same size
/// - Uppercase letters (B, C, E) = dimensions that change size
/// - The positions that change are indices [1, 2, 4], corresponding to [b -> B,
/// c -> C, e -> E]
/// - barrierDims = [1, 2, 4] tracks which dimension indices are changing
///
/// Barrier operation (x | y):
/// - Means dimensions x and y cannot be in the same collapse group
/// - Used to determine which adjacent dimensions can be merged/collapsed
///
/// Constraints:
/// - a | b: dimensions 0 and 1 cannot be collapsed together
/// - c | d: dimensions 2 and 3 cannot be collapsed together
/// - b and c CAN potentially be collapsed (no barrier between them)
///
/// Additional rules:
/// - Non-contiguous subviews cannot be collapsed
/// - Last dimension cannot be an on-the-fly transposed dimension
FlattenResult collapseUniformReassociationPipeline(
    HIVMStructuredOp op, FlattenOptions &options, ArrayRef<int64_t> barrierDims,
    std::optional<ArrayRef<int64_t>> adjustedDims) {
  assert(!op.existInlineTransposeLoopDims());
  // Flatten Unit dimensions first
  FlattenResult payload =
      initializePayload(op, options, barrierDims, adjustedDims);
  LDBG("Done payload");
  FlattenResult unitResult = getFlattenedUnit(payload, options);
  LDBG("Unit adjusted dims " << to_string(unitResult.adjustedTargetDims));
  FlattenResult collapseUniformResult =
      collapseUniformReassociation(unitResult, options);
  LDBG("Composing two results");
  LDBG("New adjusted dims "
       << to_string(collapseUniformResult.adjustedTargetDims));
  auto composedResult =
      composeFlattenResults(unitResult, collapseUniformResult, op.getContext());
  LDBG("Here done composing");

  return composedResult;
}

FlattenResult getFlattenedBroadcastableOTF(HIVMStructuredOp op,
                                           FlattenOptions &options) {
  LDBG("Flatten broadcastable OTP");
  options.checkInputConsistency = true;
  // This operation is asserted to be elementwise
  SmallVector<int64_t> broadcastDims;
  op.getBroadcastLoopDims(broadcastDims);
  return mlir::hivm::detail::collapseUniformReassociationPipeline(
      op, options, broadcastDims);
}

FlattenResult getFlattenedUniformReassociation(HIVMStructuredOp op,
                                               FlattenOptions &options) {
  return mlir::hivm::detail::collapseUniformReassociationPipeline(
      cast<HIVMStructuredOp>(op.getOperation()), options, op.getLimitedAxes());
}
} // namespace mlir::hivm::detail

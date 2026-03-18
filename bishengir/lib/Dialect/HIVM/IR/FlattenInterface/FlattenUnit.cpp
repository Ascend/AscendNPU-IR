//===- FlattenUnit.cpp - Unit flattening logic ----------------------------===//
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

namespace mlir::hivm {
namespace detail {

/// Return a reassociation map from a mask
/// unitMask[i] is true if the current axis is a unit
static ReassociationMap
getReassociationFromUnitMask(const BitVector &unitMask) {
  ReassociationMap reassociationMap;
  reassociationMap.emplace_back();
  // e.g: [Unit, Unit, Val, Unit, Val, Val, Unit]
  int rank = static_cast<int>(unitMask.size());
  int idx = 0;
  for (; idx < rank && unitMask[idx]; ++idx) {
    // Loop all the unit in the front
    reassociationMap.back().push_back(idx);
  }
  // [[0, 1]]
  bool firstVal = true;
  for (; idx < rank;) {
    assert(!unitMask[idx]);
    // Push the Val [[0, 1, 2]] -> [[0, 1, 2, 3]] -> [[0, 1, 2, 3], [4]] ->
    // [[0, 1, 2, 3], [4], [5, 6]]
    if (!firstVal)
      reassociationMap.emplace_back();
    firstVal = false;
    // New elements must create a new one
    reassociationMap.back().push_back(idx++);
    for (; idx < rank && unitMask[idx]; idx++) {
      // Push the consecutive units
      reassociationMap.back().push_back(idx);
    }
  }
  return reassociationMap;
}

enum class MaskValue {
  Unit,           // Unit dimension (size 1), can be absorbed into adjacent groups
  Collapsible,    // Non-unit dimension that can be collapsed with other Collapsibles
  NonCollapsible  // Barrier dimension that cannot be collapsed, blocks unit absorption
};

/// Return a reassociation map from a ternary mask
/// - Unit: absorbed into adjacent groups
/// - Collapsible: forms groups, absorbs adjacent units
/// - NonCollapsible: isolated, cannot absorb units from either side
///
/// Example: [Unit, Coll, Unit, NonColl, Unit, Coll, Unit]
///   Result: [[0, 1, 2], [3], [4, 5, 6]]
///   - Units 0,2 absorbed by Collapsible@1
///   - NonCollapsible@3 is isolated
///   - Units 4,6 absorbed by Collapsible@5
static ReassociationMap
getReassociationFromMask(const SmallVector<MaskValue> &mask) {
  ReassociationMap reassociationMap;
  int rank = static_cast<int>(mask.size());
  if (rank == 0)
    return reassociationMap;

  // First pass: identify segments separated by NonCollapsible barriers
  // Each segment will be processed independently
  SmallVector<std::pair<int, int>> segments; // [start, end] pairs
  for (int i = 0; i < rank; ++i) {
    bool isCreateNewSegment = (i == 0) ||
                         (mask[i] == MaskValue::NonCollapsible) ||
                         (i > 0 && mask[i - 1] == MaskValue::NonCollapsible);
    if (isCreateNewSegment) {
      segments.emplace_back(i, i);
    }
    segments.back().second = i;
  }

  for (const auto &[start, end] : segments) {
    if (mask[start] == MaskValue::NonCollapsible) {
      reassociationMap.emplace_back();
      reassociationMap.back().push_back(start);
      continue;
    }
    reassociationMap.emplace_back();
    int idx = start;
    while (idx <= end && mask[idx] == MaskValue::Unit) {
      reassociationMap.back().push_back(idx++);
    }
    bool firstVal = true;
    while (idx <= end) {
      assert(mask[idx] != MaskValue::NonCollapsible);
      if (!firstVal) {
        reassociationMap.emplace_back();
      }
      firstVal = false;
      reassociationMap.back().push_back(idx++);
      while (idx <= end && mask[idx] == MaskValue::Unit) {
        reassociationMap.back().push_back(idx++);
      }
    }
  }

  return reassociationMap;
}

/// Build a ternary collapse mask from unit mask and barrier information
/// @param unitMask Boolean mask where true indicates unit dimension
/// @param barrierDims Indices of barrier dimensions
/// @param rank Total number of dimensions
/// @param options If the strict option is true, barriers become NonCollapsible; otherwise Collapsible
static SmallVector<MaskValue>
buildCollapseMask(const BitVector &unitMask,
                  ArrayRef<int64_t> barrierDims,
                  int rank,
                  FlattenOptions &options) {
  SmallVector<MaskValue> MaskValueInfo(rank);

  // Build barrier set for O(1) lookup
  llvm::DenseSet<int64_t> barrierSet(barrierDims.begin(), barrierDims.end());

  for (int i = 0; i < rank; ++i) {
    bool isBarrier = barrierSet.contains(i);
    bool isUnit = unitMask[i];

    if (options.strictBarrierWithUnit && isBarrier) {
      // Strict mode: barriers are always NonCollapsible
      MaskValueInfo[i] = MaskValue::NonCollapsible;
    } else if (isUnit && !isBarrier) {
      // Unit dimension (not a barrier)
      MaskValueInfo[i] = MaskValue::Unit;
    } else {
      // Non-unit or barrier in non-strict mode
      MaskValueInfo[i] = MaskValue::Collapsible;
    }
  }

  return MaskValueInfo;
}

static std::string toMaskString(const SmallVector<MaskValue> &mask) {
  std::string collapseMaskInfo = "[";
  for (size_t i = 0; i < mask.size(); ++i) {
    if (i > 0)
      collapseMaskInfo += ", ";
    switch (mask[i]) {
    case MaskValue::Unit:
      collapseMaskInfo += "U";
      break;
    case MaskValue::Collapsible:
      collapseMaskInfo += "C";
      break;
    case MaskValue::NonCollapsible:
      collapseMaskInfo += "N";
      break;
    }
  }
  return collapseMaskInfo + "]";
}

FlattenResult getFlattenedUnit(FlattenResult &payload,
                               FlattenOptions &options) {
  LDBG("payload dims " << to_string(payload.originalTargetDims));
  LDBG("payload adjusted dims " << to_string(payload.adjustedTargetDims));

  auto unitMask =
      getUnitAxesMaskImpl(payload.getOperandTypes(DpsKind::kDpsAll));
  int rank = static_cast<int>(unitMask.size());

  // Build ternary mask considering barriers and strict mode
  auto collapseMask = buildCollapseMask(
      unitMask, payload.barrierDims, rank,options);

  std::string maskInfo = toMaskString(collapseMask);
  LDBG("collapse mask: " << maskInfo);

  auto reassociations = getReassociationFromMask(collapseMask);
  LDBG(to_string(reassociations));

  FlattenResult result = collapseOperandsUniformly(payload, reassociations);
  auto mapping = getReassociationMapping(reassociations);
  result.adjustBarrierAndTargetDims(mapping);
  return result;
}

/// This assumes that input and inits have a different shape
/// but its also assumed that after unit flattened the rank shall be the same
/// still
FlattenResult
getFlattenedUnitTransposeLike(HIVMStructuredOp op,
                              [[maybe_unused]] const FlattenOptions &options,
                              ArrayRef<int64_t> permutationArray) {
  // Drop input first
  FlattenResult res(op.getOperation());
  res.originalTargetDims = llvm::to_vector(permutationArray);
  BitVector inputMask;
  for (OpOperand &opr : op->getOpOperands()) {
    auto val = opr.get();
    if (auto memrefType = dyn_cast<MemRefType>(val.getType())) {
      auto unitMask = getUnitAxesMaskImpl(memrefType);
      // Disable flattening the back because transposable OTF may not be able to
      // support back collapse, still need a pivot, collapse of the last element
      // will be done in the get flattened transposable phase
      if (!isa<VTransposeOp>(op))
        unitMask[static_cast<int>(unitMask.size()) - 1] = false;
      ReassociationMap newReassociation =
          getReassociationFromUnitMask(unitMask);
      res.reassociation.push_back(newReassociation);
      res.operandTypes.emplace_back(
          op.isDpsInput(&opr),
          collapseTypeIfMemRef(val.getType(), newReassociation));
      if (op.isDpsInput(&opr))
        inputMask = unitMask;
    } else {
      res.operandTypes.emplace_back(op.isDpsInput(&opr), val.getType());
    }
  }
  // For flatten unit, going to skip the unit dimension
  //                0, 1, 2, 3, 4, 5
  // e.g: input =  [A, B, C, D, E, F]
  // Permutation = [2, 3, 0, 4, 1, 5]
  // e.g: output = [C, D, A, E, B, F]
  // If some inputs are unit, then it means we can safely ignore it
  // If D and B is unit
  // newPermutation = [2, 0, 4, 5]
  // After compression, it will be [1, 0, 2, 3]
  size_t rank = op.getNumLoops();
  SmallVector<int64_t> adjustedDims;
  assert(rank == permutationArray.size());
  assert(rank == inputMask.size());
  for (size_t i = 0; i < rank; i++) {
    if (!inputMask[permutationArray[i]]) {
      adjustedDims.push_back(permutationArray[i]);
    }
  }
  LDBG("After unit collapse, permutation be: " << to_string(adjustedDims));
  res.adjustedTargetDims = utils::compressElements(adjustedDims);
  LDBG("Compress it: " << to_string(res.adjustedTargetDims));
  return res;
}
} // namespace detail
} // namespace mlir::hivm
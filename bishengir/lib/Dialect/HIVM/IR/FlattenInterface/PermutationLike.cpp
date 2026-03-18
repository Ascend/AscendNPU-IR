//===- PermutationLike.cpp - Permutation and transpose like flattening logic =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#include <algorithm>
#define DEBUG_TYPE "flatten-permutation-like"
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
using PermutationBlocks = SmallVector<SmallVector<int64_t>>;

static PermutationBlocks getPermutationBlocks(FlattenResult &unitResult) {
  auto permutationDims = unitResult.adjustedTargetDims;

  auto rank = unitResult.getRankAfterFlatten();

  // Lets say input has group and output has group, contiguous dimension will
  // take into account the dimensions which permutations are in a block
  // Last group is guaranteed to not be permuted
  // input:               [[A, B, C], [D, E], [F], [G], [H, I]]
  // output:              [[D, E], [G], [A, B, C], [F], [H, I]]
  // Permutation:         [[3, 4], [6], [0, 1, 2], [5], [7, 8]]
  // Inverse Permutation: [[3, 4, 5], [0, 1], [6], [2], [7, 8]]
  // Permutating dimensions are block wise, contiguous also did block wise
  auto inversePermutation = utils::inversePermutation(permutationDims);
  PermutationBlocks permutationBlocks;
  LDBG("Initialized unit results");
  LDBG(to_string(inversePermutation));
  LDBG(to_string(permutationDims));
  LDBG(rank);
  assert(rank == static_cast<int>(inversePermutation.size()));
  // For the permutation blocks we need to gather based on the permutation array
  for (int i = 0; i < rank; i++) {
    if (i == 0 || inversePermutation[i - 1] + 1 != inversePermutation[i]) {
      permutationBlocks.emplace_back();
    }
    permutationBlocks.back().push_back(inversePermutation[i]);
  }
  LDBG(to_string(permutationBlocks));
  return permutationBlocks;
}

/// @param fragmentedPermutation is the permutation array, but its fragmented
/// according to the group
/// @param inputReassociation Constructing it first because the
/// initReassociation is inferrable from the input
static void getFragmentedPermutationAndInputReassociation(
    const FlattenResult &unitResult, const PermutationBlocks &permutationBlocks,
    SmallVector<ReassociationMap> &fragmentedPermutation,
    ReassociationMap &inputReassociation) {
  // The blocks will be [[3, 4, 5], [0, 1], [6], [2], [7, 8]]
  // This will be used for us to iterate
  auto blockNum = permutationBlocks.size();
  int indexInput = 0;
  // Unit is flattened, check based on permutation and contiguous
  auto contiguousInputMask =
      getContiguousAxesImpl(unitResult.getOperandTypes(DpsKind::kDpsInput));
  auto contiguousInitMask =
      getContiguousAxesImpl(unitResult.getOperandTypes(DpsKind::kDpsInit));
  // Example: [[3, 4, 5], [0, 1], [6], [2], [7, 8]] is the permutationBlocks
  for (size_t i = 0; i < blockNum; ++i) {
    auto blockSize = permutationBlocks[i].size();
    ReassociationMap currentPartition;
    for (size_t j = 0; j < blockSize; ++j) {
      LDBG("Processing " << i << to_string(permutationBlocks));
      // Example: i == 0, block = [3, 4, 5]
      // if 4 is non contiguous, we will create [3], [4, 5]
      // in the third block (actually can be compressed to get the temporary
      // permutation assuming layout is identity) [3, 0, 6, 2, 7] -> [2, 0, 3,
      // 1, 4] but we will use the rank and use the non compressed rank instead
      int indexInit = permutationBlocks[i][j];
      bool createNewGroup = (j == 0);
      if (!contiguousInputMask[indexInput] || !contiguousInitMask[indexInit]) {
        // if one of them is not contiguous, we need to create a new group
        createNewGroup = true;
      }
      if (createNewGroup) {
        currentPartition.emplace_back();
      }
      currentPartition.back().push_back(indexInput);
      indexInput++;
    }
    inputReassociation.append(currentPartition);
    // Pivot will be [3, 0, 6, 2, 7] consecutively
    auto pivot = permutationBlocks[i].front();
    fragmentedPermutation[pivot] = std::move(currentPartition);
    // If we flatten this fragmented permutation, we will find which is what we
    // want, compressing the initial will return the permutation :D
    // Renumbering them will return the initReassociation!
    // [[3, 4], [6], [0], [1, 2], [5], [7, 8]]
  }
}

FlattenResult getFlattenedTransposeLike(HIVMStructuredOp op,
                                        FlattenOptions &options) {
  //  First of all, remove all the unit operations
  FlattenResult unitResult =
      getFlattenedUnitTransposeLike(op, options, op.getPermutationArray());
  LDBG(to_string(unitResult.getInputReassociation()));

  PermutationBlocks permutationBlocks = getPermutationBlocks(unitResult);
  ReassociationMap inputReassociation;
  auto rank = unitResult.getRankAfterFlatten();
  SmallVector<ReassociationMap> fragmentedPermutation(rank);
  getFragmentedPermutationAndInputReassociation(
      unitResult, permutationBlocks, fragmentedPermutation, inputReassociation);
  LDBG(to_string(fragmentedPermutation));
  ReassociationMap initReassociation;
  SmallVector<int64_t> newPermutationDims;
  for (auto &el : fragmentedPermutation) {
    initReassociation.append(el);
  }
  for (auto &el : initReassociation) {
    newPermutationDims.push_back(el.front());
  }

  // Construct the result
  FlattenResult res(op.getOperation());
  res.originalTargetDims = unitResult.originalTargetDims;
  res.adjustedTargetDims = utils::compressElements(newPermutationDims);
  res.barrierDims = res.adjustedTargetDims;
  utils::renumberReassociation(initReassociation);
  res.reassociation.push_back(inputReassociation);
  res.reassociation.push_back(initReassociation);
  for (auto type : unitResult.getOperandTypes(DpsKind::kDpsInput)) {
    res.operandTypes.emplace_back(
        true, collapseTypeIfMemRef(type, inputReassociation));
  }
  for (auto type : unitResult.getOperandTypes(DpsKind::kDpsInit)) {
    res.operandTypes.emplace_back(
        false, collapseTypeIfMemRef(type, initReassociation));
  }
  // Compose!
  auto composedResult =
      composeFlattenResults(unitResult, res, op->getContext());
  // So now take into account contiguous dimensions only
  return composedResult;
}
} // namespace mlir::hivm::detail

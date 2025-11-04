//===- FlattenUnit.cpp - Unit flattening logic ----------------------------===//
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

FlattenResult getFlattenedUnit(FlattenResult &payload) {
  LDBG("payload dims " << to_string(payload.originalTargetDims));
  LDBG("payload adjusted dims " << to_string(payload.adjustedTargetDims));
  auto unitMask =
      getUnitAxesMaskImpl(payload.getOperandTypes(DpsKind::kDpsAll));
  int rank = static_cast<int>(unitMask.size());
  auto limitationMask = utils::arrayToMask(payload.barrierDims, rank);
  if (!limitationMask.empty()) {
    // if this operation has limitation, for example
    unitMask &= limitationMask.flip();
  }
  auto reassociations = getReassociationFromUnitMask(unitMask);
  LDBG(to_string(reassociations));
  FlattenResult result =
      mlir::hivm::detail::collapseOperandsUniformly(payload, reassociations);
  auto mapping = getReassociationMapping(reassociations);
  result.adjustBarrierAndTargetDims(mapping);
  return result;
}

/// This assumes that input and inits have a different shape
/// but its also assumed that after unit flattened the rank shall be the same
/// still
FlattenResult
getFlattenedUnitTransposableOTF(HIVMStructuredOp op,
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
      // Disable flattening the back because vtranspose may not be able to
      // support back collapse, still need a pivot, collapse of the last element
      // will be done in the get flattened transposable phase
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

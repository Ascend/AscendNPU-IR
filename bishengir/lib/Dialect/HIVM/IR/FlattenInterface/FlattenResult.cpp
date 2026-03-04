//===- FlattenResult.cpp - Common implementation of flatten result --------===//
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

bool FlattenResult::isIdentityCollapse() const {
  return util::isIdentityCollapse(reassociation.front());
}

int FlattenResult::getRankAfterFlatten() const {
  if (reassociation.empty() || reassociation.front().empty() ||
      reassociation.front().back().empty())
    return 0;
  return reassociation.front().size();
}

SmallVector<Type> FlattenResult::getOperandTypes(DpsKind kind) const {
  SmallVector<Type> result;
  for (const auto &operandType : operandTypes) {
    bool isInput = operandType.first;
    switch (kind) {
    case DpsKind::kDpsInput:
      if (isInput) {
        result.push_back(operandType.second);
      }
      break;
    case DpsKind::kDpsInit:
      if (!isInput) {
        result.push_back(operandType.second);
      }
      break;
    case DpsKind::kDpsAll:
      result.push_back(operandType.second);
      break;
    }
  }
  return result;
}

void FlattenResult::fillWithIdentity() {
  int rank = 0;
  auto hivmOp = cast<HIVMStructuredOp>(op);
  for (auto opr : hivmOp.getHIVMOperands(false)) {
    if (auto shapedType = dyn_cast<ShapedType>(opr->get().getType())) {
      rank = shapedType.getRank();
    }
    operandTypes.emplace_back(hivmOp.isDpsInput(opr), opr->get().getType());
    operandOriginalVal.push_back(opr->get());
  }
  ReassociationMap identityCollapse;
  for (int i = 0; i < rank; i++)
    identityCollapse.push_back({i});
  reassociation = {identityCollapse};
}

std::optional<Type> FlattenResult::getOperandTypeAfterFlattened(Value val) {
  auto pos = llvm::find(operandOriginalVal, val);
  if (pos == operandOriginalVal.end())
    return std::nullopt;
  return operandTypes[pos - operandOriginalVal.begin()].second;
}

void FlattenResult::adjustBarrierAndTargetDims(ArrayRef<int64_t> mapping) {
  adjustedTargetDims = getNewIndexing(adjustedTargetDims, mapping);
  barrierDims = getNewIndexing(barrierDims, mapping);
}

} // namespace mlir::hivm
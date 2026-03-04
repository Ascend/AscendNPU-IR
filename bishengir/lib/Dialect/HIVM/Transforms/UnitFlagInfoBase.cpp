//===- UnitFlagInfoBase.h -- unit-flag-info base ---------------------------==//
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

#include "bishengir/Dialect/HIVM/Transforms/UnitFlagInfoBase.h"

using namespace mlir;
using namespace mlir::hivm;

std::optional<std::pair<SmallVector<UNIT_FLAG>, SmallVector<mlir::Value>>>
UnitFlagInfoBase::getUnitFlagLoopAwareArgs(Operation *op,
                                           IRRewriter &rewriter) {
  if (disabled()) {
    return {};
  }

  auto getIsFirstIterationValue = [&op, &rewriter](scf::ForOp forOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value lowerBound = forOp.getLowerBound();
    Value currentInd = forOp.getInductionVar();
    Value isFirstIter = rewriter.create<arith::CmpIOp>(
        op->getLoc(), arith::CmpIPredicate::eq, lowerBound, currentInd);
    return isFirstIter;
  };

  auto getIsLastIterationValue = [&op, &rewriter](scf::ForOp forOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    auto loc = op->getLoc();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();
    Value currentInd = forOp.getInductionVar();
    Value nextInd = rewriter.create<arith::AddIOp>(loc, currentInd, step);
    Value isLastIter = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, nextInd, upperBound);
    return isLastIter;
  };

  auto unitFlagModes = getUnitFlagModesAsSetAsWait(/*compress=*/true);
  SmallVector<Value> unitFlagConds;

  if (unitFlagModes.size() >= 2) {
    auto parentLoop = parentLoopAsSet ? parentLoopAsSet : parentLoopAsWait;
    assert(!parentLoopAsWait || (parentLoopAsWait == parentLoop));
    assert(parentLoop && (parentLoop == op->getParentOp()));
    unitFlagConds.push_back(getIsFirstIterationValue(parentLoop));
    unitFlagConds.push_back(getIsLastIterationValue(parentLoop));
  }

  if (unitFlagModes.size() == 3) {
    auto i1Ty = rewriter.getI1Type();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value constantTrueValue = rewriter.create<arith::ConstantOp>(
        op->getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 1));
    unitFlagConds.push_back(constantTrueValue);
  }

  return std::make_pair(unitFlagModes, unitFlagConds);
}

std::optional<std::pair<SmallVector<UNIT_FLAG>, SmallVector<mlir::Value>>>
UnitFlagInfoBase::getUnitFlagLinkedLoopArgs(Operation *op,
                                            IRRewriter &rewriter) {
  if (disabled()) {
    return {};
  }

  // unit-flag modes
  auto asSet = getUnitFlagModesAsSet(/*compress=*/true);
  auto asWait = getUnitFlagModesAsWait(/*compress=*/true);
  auto asSetAsWait = getUnitFlagModesAsSetAsWait(/*compress=*/true);
  if (asSet.empty()) {
    asSet.push_back(UNIT_FLAG::DISABLED);
  }
  if (asWait.empty()) {
    asWait.push_back(UNIT_FLAG::DISABLED);
  }
  assert(asSet.size() == 1);
  assert(asWait.size() == 1);
  assert(asSetAsWait.size() == 1);
  SmallVector<UNIT_FLAG> unitFlagModes;
  unitFlagModes.push_back(asSetAsWait.front());
  if (linkedLoopAsSet) {
    unitFlagModes.push_back(asWait.front());
  }
  if (linkedLoopAsWait) {
    unitFlagModes.push_back(asSet.front());
  }

  // unit-flag conditions
  auto getIsAliveLoopValue = [&op, &rewriter](scf::ForOp forOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value upperBound = forOp.getUpperBound();
    Value lowerBound = forOp.getLowerBound();
    return rewriter.create<arith::CmpIOp>(
        op->getLoc(), arith::CmpIPredicate::slt, lowerBound, upperBound);
  };
  SmallVector<Value> unitFlagConds;
  if (linkedLoopAsSet) {
    Value cond = getIsAliveLoopValue(linkedLoopAsSet);
    unitFlagConds.push_back(cond);
  }
  if (linkedLoopAsWait) {
    Value cond = getIsAliveLoopValue(linkedLoopAsWait);
    unitFlagConds.push_back(cond);
  }
  if (linkedLoopAsSet && linkedLoopAsWait) {
    Value cond = rewriter.create<arith::OrIOp>(op->getLoc(), unitFlagConds[0],
                                               unitFlagConds[1]);
    unitFlagConds.insert(unitFlagConds.begin(), cond);
  } else {
    auto i1Ty = rewriter.getI1Type();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value constantTrueValue = rewriter.create<arith::ConstantOp>(
        op->getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 1));
    unitFlagConds.push_back(constantTrueValue);
  }

  assert(unitFlagModes.size() == unitFlagConds.size());
  while (!unitFlagModes.empty() &&
         unitFlagModes.back() == UNIT_FLAG::DISABLED) {
    unitFlagModes.pop_back();
    unitFlagConds.pop_back();
  }
  return std::make_pair(unitFlagModes, unitFlagConds);
}

std::optional<std::pair<SmallVector<UNIT_FLAG>, SmallVector<mlir::Value>>>
UnitFlagInfoBase::getUnitFlagArgs(Operation *op, IRRewriter &rewriter) {
  if (disabled()) {
    return {};
  }
  bool hasParentLoop = parentLoopAsSet || parentLoopAsWait;
  bool hasLinkedLoop = linkedLoopAsSet || linkedLoopAsWait;
  if (!hasParentLoop && !hasLinkedLoop) {
    auto unitFlagModes = getUnitFlagModesAsSetAsWait(/*compress=*/true);
    assert(unitFlagModes.size() <= 1);
    return std::make_pair(unitFlagModes, SmallVector<mlir::Value>());
  }
  if (hasParentLoop && !hasLinkedLoop) {
    return getUnitFlagLoopAwareArgs(op, rewriter);
  }
  if (!hasParentLoop && hasLinkedLoop) {
    return getUnitFlagLinkedLoopArgs(op, rewriter);
  }
  llvm_unreachable("unexpected unit-flag-info state");
}

namespace mlir {
namespace hivm {

std::optional<UnitFlagInfoBase> checkUnitFlagSameBlockPattern(
    Operation *op1, Operation *op2, UnitFlagInfoBase unitFlagInfo1,
    UnitFlagInfoBase unitFlagInfo2, scf::ForOp backwardSyncLoop) {
  assert(op1 != nullptr && op2 != nullptr);
  hivm::MmadL1Op mmadL1Op;
  hivm::FixpipeOp fixpipeOp;
  bool mmadL1OpIsFrontOp = false;
  bool fixpipeOpIsFrontOp = false;
  if ((mmadL1Op = dyn_cast<MmadL1Op>(op1)) &&
      (fixpipeOp = dyn_cast<FixpipeOp>(op2))) {
    mmadL1OpIsFrontOp = true;
  } else if ((mmadL1Op = dyn_cast<MmadL1Op>(op2)) &&
             (fixpipeOp = dyn_cast<FixpipeOp>(op1))) {
    fixpipeOpIsFrontOp = true;
  } else {
    return {};
  }
  if (fixpipeOpIsFrontOp && unitFlagInfo1.disabledAsWait()) {
    return {};
  }
  if (fixpipeOpIsFrontOp && unitFlagInfo1.linkedLoopAsWait != nullptr) {
    return {};
  }
  if (fixpipeOp.getSrc() != mmadL1Op.getC()) {
    return {};
  }
  if (fixpipeOp->getBlock() != mmadL1Op->getBlock()) {
    return {};
  }
  auto unitFlagAsSet = UNIT_FLAG::ENABLED_WITH_UPDATE;
  auto unitFlagAsWait = mmadL1OpIsFrontOp ? UNIT_FLAG::ENABLED_WITH_UPDATE
                                          : UNIT_FLAG::ENABLED_WITHOUT_UPDATE;
  UnitFlagInfoBase unitFlagInfo(unitFlagAsSet, unitFlagAsWait);
  if (backwardSyncLoop) {
    unitFlagInfo.asSetLastIter = UNIT_FLAG::DISABLED;
    unitFlagInfo.asWaitFirstIter = UNIT_FLAG::DISABLED;
    unitFlagInfo.parentLoopAsSet = backwardSyncLoop;
    unitFlagInfo.parentLoopAsWait = backwardSyncLoop;
  }
  return unitFlagInfo;
}

std::optional<UnitFlagInfoBase> checkUnitFlagOpLoopOpPattern(
    Operation *op1, Operation *op2, UnitFlagInfoBase unitFlagInfo1,
    UnitFlagInfoBase unitFlagInfo2, scf::ForOp backwardSyncLoop) {
  hivm::MmadL1Op mmadL1Op;
  hivm::FixpipeOp fixpipeOp;
  bool mmadL1OpIsFrontOp = false;
  bool fixpipeOpIsFrontOp = false;
  assert(op1 != nullptr && op2 != nullptr);
  if ((mmadL1Op = dyn_cast<MmadL1Op>(op1)) &&
      (fixpipeOp = dyn_cast<FixpipeOp>(op2))) {
    mmadL1OpIsFrontOp = true;
  } else if ((mmadL1Op = dyn_cast<MmadL1Op>(op2)) &&
             (fixpipeOp = dyn_cast<FixpipeOp>(op1))) {
    fixpipeOpIsFrontOp = true;
  } else {
    return {};
  }
  if (fixpipeOpIsFrontOp && unitFlagInfo1.disabledAsWait()) {
    return {};
  }
  if (fixpipeOpIsFrontOp && unitFlagInfo1.linkedLoopAsWait != nullptr) {
    return {};
  }
  if (fixpipeOp.getSrc() != mmadL1Op.getC()) {
    return {};
  }
  auto forOp1 = dyn_cast<scf::ForOp>(op1->getParentOp());
  auto forOp2 = dyn_cast<scf::ForOp>(op2->getParentOp());
  auto unitFlagDisabled = UNIT_FLAG::DISABLED;
  auto unitFlagAsSet = UNIT_FLAG::ENABLED_WITH_UPDATE;
  auto unitFlagAsWait = mmadL1OpIsFrontOp ? UNIT_FLAG::ENABLED_WITH_UPDATE
                                          : UNIT_FLAG::ENABLED_WITHOUT_UPDATE;
  if (forOp1 && forOp2) {
    if (forOp1->getParentRegion() == forOp2->getParentRegion()) {
      UnitFlagInfoBase unitFlagInfo(unitFlagDisabled, unitFlagDisabled,
                                    unitFlagAsSet, unitFlagAsWait,
                                    unitFlagDisabled, unitFlagDisabled);
      unitFlagInfo.linkedLoopAsWait = forOp1;
      unitFlagInfo.linkedLoopAsSet = forOp2;
      unitFlagInfo.parentLoopAsSet = forOp1;
      unitFlagInfo.parentLoopAsWait = forOp2;
      return unitFlagInfo;
    }
  } else if (forOp1) {
    if (forOp1->getParentRegion() == op2->getParentRegion()) {
      UnitFlagInfoBase unitFlagInfo(unitFlagDisabled, unitFlagDisabled,
                                    unitFlagAsSet, unitFlagAsWait,
                                    unitFlagAsWait, unitFlagAsWait);
      unitFlagInfo.linkedLoopAsWait = forOp1;
      unitFlagInfo.parentLoopAsSet = forOp1;
      return unitFlagInfo;
    }
  } else if (forOp2) {
    if (forOp2->getParentRegion() == op1->getParentRegion()) {
      UnitFlagInfoBase unitFlagInfo(unitFlagAsSet, unitFlagAsSet, unitFlagAsSet,
                                    unitFlagAsWait, unitFlagDisabled,
                                    unitFlagDisabled);
      unitFlagInfo.linkedLoopAsSet = forOp2;
      unitFlagInfo.parentLoopAsWait = forOp2;
      return unitFlagInfo;
    }
  }
  return {};
}

} // namespace hivm
} // namespace mlir

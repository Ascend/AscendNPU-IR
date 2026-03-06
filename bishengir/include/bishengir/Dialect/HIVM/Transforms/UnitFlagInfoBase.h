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
#ifndef BISHENG_DIALECT_HIVM_UNITFLAGINFOBASE_H
#define BISHENG_DIALECT_HIVM_UNITFLAGINFOBASE_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include <cstddef>
#include <utility>

namespace mlir {
namespace hivm {

class UnitFlagInfoBase {
public:
  UNIT_FLAG asSetFirstIter{UNIT_FLAG::DISABLED};
  UNIT_FLAG asSetMidIters{UNIT_FLAG::DISABLED};
  UNIT_FLAG asSetLastIter{UNIT_FLAG::DISABLED};
  UNIT_FLAG asWaitFirstIter{UNIT_FLAG::DISABLED};
  UNIT_FLAG asWaitMidIters{UNIT_FLAG::DISABLED};
  UNIT_FLAG asWaitLastIter{UNIT_FLAG::DISABLED};
  scf::ForOp linkedLoopAsSet{nullptr};
  scf::ForOp linkedLoopAsWait{nullptr};
  scf::ForOp parentLoopAsSet{nullptr};
  scf::ForOp parentLoopAsWait{nullptr};

public:
  UnitFlagInfoBase() = default;
  virtual ~UnitFlagInfoBase() = default;

  UnitFlagInfoBase(UNIT_FLAG unitFlagModeAsSet, UNIT_FLAG unitFlagModeAsWait)
      : asSetFirstIter(unitFlagModeAsSet), asSetMidIters(unitFlagModeAsSet),
        asSetLastIter(unitFlagModeAsSet), asWaitFirstIter(unitFlagModeAsWait),
        asWaitMidIters(unitFlagModeAsWait), asWaitLastIter(unitFlagModeAsWait) {
  }

  UnitFlagInfoBase(UNIT_FLAG asSetFirstIter, UNIT_FLAG asSetMidIters,
                   UNIT_FLAG asSetLastIter, UNIT_FLAG asWaitFirstIter,
                   UNIT_FLAG asWaitMidIters, UNIT_FLAG asWaitLastIter)
      : asSetFirstIter(asSetFirstIter), asSetMidIters(asSetMidIters),
        asSetLastIter(asSetLastIter), asWaitFirstIter(asWaitFirstIter),
        asWaitMidIters(asWaitMidIters), asWaitLastIter(asWaitLastIter) {}

  void reset() {
    asSetFirstIter = UNIT_FLAG::DISABLED;
    asSetMidIters = UNIT_FLAG::DISABLED;
    asSetLastIter = UNIT_FLAG::DISABLED;
    asWaitFirstIter = UNIT_FLAG::DISABLED;
    asWaitMidIters = UNIT_FLAG::DISABLED;
    asWaitLastIter = UNIT_FLAG::DISABLED;
    linkedLoopAsSet = nullptr;
    linkedLoopAsWait = nullptr;
    parentLoopAsSet = nullptr;
    parentLoopAsWait = nullptr;
  }

  bool disabled() const { return disabledAsSet() && disabledAsWait(); }

  bool disabledAsSet() const {
    return llvm::all_equal(
        {UNIT_FLAG::DISABLED, asSetFirstIter, asSetMidIters, asSetLastIter});
  }

  bool disabledAsWait() const {
    return llvm::all_equal(
        {UNIT_FLAG::DISABLED, asWaitFirstIter, asWaitMidIters, asWaitLastIter});
  }

  static void updateMode(UNIT_FLAG &unitFlagMode,
                         const UNIT_FLAG &otherUnitFlagMode) {
    if (unitFlagMode < otherUnitFlagMode) {
      unitFlagMode = otherUnitFlagMode;
    }
  }

  void merge(const UnitFlagInfoBase &other, bool asSet = true,
             bool asWait = true) {
    if (asSet) {
      updateMode(this->asSetFirstIter, other.asSetFirstIter);
      updateMode(this->asSetMidIters, other.asSetMidIters);
      updateMode(this->asSetLastIter, other.asSetLastIter);
      this->linkedLoopAsSet = other.linkedLoopAsSet;
      if (other.parentLoopAsSet) {
        assert(!this->parentLoopAsSet ||
               this->parentLoopAsSet == other.parentLoopAsSet);
        this->parentLoopAsSet = other.parentLoopAsSet;
      }
    }
    if (asWait) {
      updateMode(this->asWaitFirstIter, other.asWaitFirstIter);
      updateMode(this->asWaitMidIters, other.asWaitMidIters);
      updateMode(this->asWaitLastIter, other.asWaitLastIter);
      this->linkedLoopAsWait = other.linkedLoopAsWait;
      if (other.parentLoopAsWait) {
        assert(!this->parentLoopAsWait ||
               this->parentLoopAsWait == other.parentLoopAsWait);
        this->parentLoopAsWait = other.parentLoopAsWait;
      }
    }
  }

  SmallVector<UNIT_FLAG>
  compressUnitFlagModes(SmallVector<UNIT_FLAG> unitFlagModes,
                        bool enable = true) const {
    if (enable) {
      assert(unitFlagModes.size() == 3);
      bool allEqual = llvm::all_equal(unitFlagModes);
      if (allEqual && unitFlagModes.front() == UNIT_FLAG::DISABLED) {
        return {};
      }
      if (allEqual) {
        return {unitFlagModes.front()};
      }
      if (unitFlagModes.back() == UNIT_FLAG::DISABLED) {
        unitFlagModes.pop_back();
      }
    }
    return unitFlagModes;
  }

  SmallVector<UNIT_FLAG> getUnitFlagModesAsSet(bool compress = false) const {
    return compressUnitFlagModes({asSetFirstIter, asSetLastIter, asSetMidIters},
                                 compress);
  }

  SmallVector<UNIT_FLAG> getUnitFlagModesAsWait(bool compress = false) const {
    return compressUnitFlagModes(
        {asWaitFirstIter, asWaitLastIter, asWaitMidIters}, compress);
  }

  SmallVector<UNIT_FLAG>
  getUnitFlagModesAsSetAsWait(bool compress = false) const {
    UNIT_FLAG unitFlagModeFirstIter = asSetFirstIter;
    UNIT_FLAG unitFlagModeMidIters = asSetMidIters;
    UNIT_FLAG unitFlagModeLastIter = asSetLastIter;
    updateMode(unitFlagModeFirstIter, asWaitFirstIter);
    updateMode(unitFlagModeMidIters, asWaitMidIters);
    updateMode(unitFlagModeLastIter, asWaitLastIter);
    return compressUnitFlagModes(
        {unitFlagModeFirstIter, unitFlagModeLastIter, unitFlagModeMidIters},
        compress);
  }

  std::optional<std::pair<SmallVector<UNIT_FLAG>, SmallVector<mlir::Value>>>
  getUnitFlagArgs(Operation *op, IRRewriter &rewriter);

  std::optional<std::pair<SmallVector<UNIT_FLAG>, SmallVector<mlir::Value>>>
  getUnitFlagLoopAwareArgs(Operation *op, IRRewriter &rewriter);

  std::optional<std::pair<SmallVector<UNIT_FLAG>, SmallVector<mlir::Value>>>
  getUnitFlagLinkedLoopArgs(Operation *op, IRRewriter &rewriter);
};

std::optional<UnitFlagInfoBase> checkUnitFlagSameBlockPattern(
    Operation *op1, Operation *op2, UnitFlagInfoBase unitFlagInfo1,
    UnitFlagInfoBase unitFlagInfo2, scf::ForOp backwardSyncLoop);

std::optional<UnitFlagInfoBase> checkUnitFlagOpLoopOpPattern(
    Operation *op1, Operation *op2, UnitFlagInfoBase unitFlagInfo1,
    UnitFlagInfoBase unitFlagInfo2, scf::ForOp backwardSyncLoop);

} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_UNITFLAGINFOBASE_H

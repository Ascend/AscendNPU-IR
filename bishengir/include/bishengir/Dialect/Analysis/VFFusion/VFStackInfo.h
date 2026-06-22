//===- VFStackInfo.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_VFSTACKINFO_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_VFSTACKINFO_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::analysis {

// Conservative VF stack-slot summary for one outlined vector function.
struct VFStackInfo {
  int64_t inputSlots = 0;
  int64_t outputSlots = 0;
  int64_t internalOperandSlots = 0;

  int64_t totalSlots() const {
    return inputSlots + outputSlots + internalOperandSlots;
  }

  bool exceeds(int64_t limit) const {
    return limit >= 0 && totalSlots() > limit;
  }
};

class VFStackInfoBuilder {
public:
  // Bisheng uses 16-bit stack slots; current conservative guard keeps 58 slots.
  static constexpr int64_t kDefaultSlotLimit = 58;

  explicit VFStackInfoBuilder(int64_t slotLimit = kDefaultSlotLimit)
      : slotLimit(slotLimit) {}
  explicit VFStackInfoBuilder(bool enableStackLimit)
      : slotLimit(enableStackLimit ? kDefaultSlotLimit : -1) {}

  bool isEnabled() const { return slotLimit >= 0; }
  int64_t getSlotLimit() const { return slotLimit; }

  bool fitsStack(ArrayRef<Operation *> ops) const;
  bool fitsInputs(ArrayRef<Operation *> ops) const;

  VFStackInfo compute(ArrayRef<Operation *> ops) const;

  static int64_t getVectorSlot(Value value);

private:
  int64_t slotLimit;
};

} // namespace mlir::analysis

#endif // BISHENGIR_DIALECT_ANALYSIS_VFFUSION_VFSTACKINFO_H

//===- VFStackInfo.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/VFFusion/VFStackInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::analysis;

namespace {

static int64_t sumSlots(const SetVector<Value> &values) {
  int64_t slots = 0;
  for (Value value : values)
    slots += VFStackInfoBuilder::getVectorSlot(value);
  return slots;
}

static int64_t getVectorSlotForType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    type = shapedType.getElementType();

  unsigned bitWidth = 16;
  if (auto intType = dyn_cast<IntegerType>(type)) {
    bitWidth = intType.getWidth();
  } else if (auto floatType = dyn_cast<FloatType>(type)) {
    bitWidth = floatType.getWidth();
  } else if (isa<IndexType>(type)) {
    bitWidth = 64;
  }

  return (static_cast<int64_t>(bitWidth) + 15) / 16;
}

static SetVector<Operation *> collectOps(ArrayRef<Operation *> roots) {
  SetVector<Operation *> ops;
  for (Operation *root : roots) {
    if (!root)
      continue;
    root->walk<WalkOrder::PreOrder>([&](Operation *op) { ops.insert(op); });
  }
  return ops;
}

} // namespace

static bool isNestedBlockArgument(Value value,
                                  const SetVector<Operation *> &ops) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return false;

  Operation *parent = blockArg.getOwner()->getParentOp();
  while (parent) {
    if (ops.contains(parent))
      return true;
    parent = parent->getParentOp();
  }
  return false;
}

int64_t VFStackInfoBuilder::getVectorSlot(Value value) {
  return getVectorSlotForType(value.getType());
}

VFStackInfo VFStackInfoBuilder::compute(ArrayRef<Operation *> roots) const {
  SetVector<Operation *> ops = collectOps(roots);

  SetVector<Value> inputs;
  SetVector<Value> outputs;
  SetVector<Value> internalOperands;

  // Conservatively treat every group operand as live within the VF.
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      if (isNestedBlockArgument(operand, ops))
        continue;
      Operation *defOp = operand.getDefiningOp();
      if (defOp && ops.contains(defOp))
        internalOperands.insert(operand);
      else
        inputs.insert(operand);
    }
  }

  // Results that leave the group also reserve slots in the outlined VF.
  for (Operation *op : ops) {
    for (Value result : op->getResults()) {
      bool usedOutside = result.use_empty();
      for (Operation *user : result.getUsers()) {
        if (!ops.contains(user)) {
          usedOutside = true;
          break;
        }
      }
      if (usedOutside)
        outputs.insert(result);
    }
  }

  VFStackInfo info;
  info.inputSlots = sumSlots(inputs);
  info.outputSlots = sumSlots(outputs);
  info.internalOperandSlots = sumSlots(internalOperands);
  return info;
}

bool VFStackInfoBuilder::fitsStack(ArrayRef<Operation *> ops) const {
  return !isEnabled() || compute(ops).totalSlots() <= slotLimit;
}

bool VFStackInfoBuilder::fitsInputs(ArrayRef<Operation *> ops) const {
  return !isEnabled() || compute(ops).inputSlots <= slotLimit;
}

//===- AgglomerationBlock.cpp - Agglomeration block boundaries ------------===//
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

#include "bishengir/Analysis/AgglomerationBlock.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hivm;

void AgglomerationBlock::add(Operation *op) {
  if (!op)
    return;
  if (ops.contains(op))
    return;
  ops.insert(op);
  boundariesDirty = true;
}

void AgglomerationBlock::remove(Operation *op) {
  if (ops.remove(op))
    boundariesDirty = true;
}

const llvm::SetVector<Value> &AgglomerationBlock::getInputs() const {
  ensureBoundaries();
  return inputs;
}

const llvm::SetVector<Value> &AgglomerationBlock::getOutputs() const {
  ensureBoundaries();
  return outputs;
}

void AgglomerationBlock::ensureBoundaries() const {
  if (boundariesDirty)
    recomputeBoundaries();
}

void AgglomerationBlock::recomputeBoundaries() const {
  inputs.clear();
  outputs.clear();

  llvm::DenseSet<Value> definedInside;
  for (Operation *outer : ops) {
    outer->walk<WalkOrder::PreOrder>([&](Operation *op) {
      for (Value result : op->getResults())
        definedInside.insert(result);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            definedInside.insert(argument);
    });
  }
  for (Operation *outer : ops) {
    outer->walk<WalkOrder::PreOrder>([&](Operation *op) {
      for (Value operand : op->getOperands())
        if (!definedInside.contains(operand))
          inputs.insert(operand);
    });
  }

  llvm::DenseSet<Operation *> opsInside;
  for (Operation *outer : ops)
    outer->walk([&](Operation *op) { opsInside.insert(op); });
  for (Operation *outer : ops) {
    for (Value result : outer->getResults()) {
      bool escapes = llvm::any_of(result.getUsers(), [&](Operation *user) {
        return !opsInside.contains(user);
      });
      if (escapes)
        outputs.insert(result);
    }
  }

  boundariesDirty = false;
}

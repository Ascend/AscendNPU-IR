//===- BufferPoolUtils.h - Load scheduling utilities ------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_BUFFERPOOLUTILS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_BUFFERPOOLUTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hivm::buffer_pool {

struct Access {
  Operation *op;
  PIPE pipe;
  bool write;
};

/// Follow local aliases; fail on function calls and unmodelled accesses.
LogicalResult collectAccesses(Value root, SmallVectorImpl<Access> &accesses);

struct LoadInfo {
  scope::ScopeOp scope;
  Operation *loop = nullptr;
  Block *body = nullptr;
  Value root;
  TCoreType core;
  SmallVector<Access> accesses;
  llvm::SetVector<Operation *> readers;
  int64_t originalSlots = 1;
};

struct TaskInfo {
  BlockArgument counter;
  Value start;
  Value bound;
  Value tripStart;
  Value tripBound;
  int64_t step = 1;
  int64_t tripStep = 1;
  scf::IfOp guard;
};

/// Identify the actual address counter and its original range/update.
FailureOr<TaskInfo> analyzeTask(const LoadInfo &load);
/// Analyze a candidate before Separate commits its load scope.
FailureOr<TaskInfo> analyzeTask(Block *body,
                                ArrayRef<Operation *> loadOperations,
                                ArrayRef<Operation *> readers);

/// Find the first LoadOp or ND2NZOp, or return null if none remains.
Operation *findLoadTransfer(scope::ScopeOp scope);

FailureOr<LoadInfo> analyzeLoad(scope::ScopeOp scope, Operation *transfer,
                                bool planned = false);
/// Common scalar-expression queries for task and stage analysis.
std::optional<int64_t> getConstantInt(Value value);
void collectCounters(Value value, Block *body, DenseSet<Value> &visited,
                     llvm::SetVector<Value> &counters);
void collectConjuncts(Value value, SmallVectorImpl<Value> &conjuncts);

/// Follow views/casts without guessing through control-flow merges.
Value getBufferRoot(Value value);

/// Locate the for body or while before/after block containing anchor.
Block *getLoopBodyBlock(LoopLikeOpInterface loop, Operation *anchor);

/// Whether this is the innermost marked scheduling loop.
bool isSchedulingLoop(Operation *op);

} // namespace mlir::hivm::buffer_pool

#endif

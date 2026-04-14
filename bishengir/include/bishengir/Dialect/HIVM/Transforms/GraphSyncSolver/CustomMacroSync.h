//===-------- CustomMacroSync.h ---- Graph Sync Solver -------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_CUSTOMMACROSYNC_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_CUSTOMMACROSYNC_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/EventIdSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace mlir::hivm::syncsolver {

inline bool slotHasExplicitPipes(hivm::SyncEventSlotAttr slot) {
  return static_cast<bool>(slot.getSetPipe());
}

inline std::pair<hivm::PIPE, hivm::PIPE>
getSlotPipePair(hivm::SyncEventSlotAttr slot) {
  return {slot.getSetPipe().getPipe(), slot.getWaitPipe().getPipe()};
}

inline std::pair<hivm::PIPE, hivm::PIPE>
resolveSlotPipePair(hivm::SyncEventSlotAttr slot, hivm::CustomMacroOp macro) {
  if (slotHasExplicitPipes(slot))
    return getSlotPipePair(slot);
  return {macro.getInPipe(), macro.getOutPipe()};
}

inline bool slotMacroWaits(hivm::SyncEventSlotAttr slot) {
  return slot.getMacroSync() == hivm::SyncEventSlotMacroSync::wait;
}

inline bool slotMacroSets(hivm::SyncEventSlotAttr slot) {
  return slot.getMacroSync() == hivm::SyncEventSlotMacroSync::set;
}

inline bool slotMacroInternal(hivm::SyncEventSlotAttr slot) {
  return slot.getMacroSync() == hivm::SyncEventSlotMacroSync::internal;
}

inline hivm::PIPE getSlotSetPipe(hivm::SyncEventSlotAttr slot) {
  return slot.getSetPipe().getPipe();
}

inline hivm::PIPE getSlotWaitPipe(hivm::SyncEventSlotAttr slot) {
  return slot.getWaitPipe().getPipe();
}

inline bool slotMatchesPipePair(hivm::SyncEventSlotAttr slot,
                                hivm::PIPE setPipe, hivm::PIPE waitPipe) {
  return slotHasExplicitPipes(slot) && getSlotSetPipe(slot) == setPipe &&
         getSlotWaitPipe(slot) == waitPipe;
}

inline std::optional<int64_t>
getSlotPinnedEventId(hivm::SyncEventSlotAttr slot) {
  if (!slot.getEvent())
    return std::nullopt;
  return static_cast<int64_t>(slot.getEvent().getEvent());
}

void applyCustomMacroPinnedEventId(ConflictPair &conflictPair,
                                   RWOperation *rwOp1, RWOperation *rwOp2,
                                   hivm::PIPE setPipe, hivm::PIPE waitPipe);

class CustomMacroSyncState {
public:
  bool hasConflict() const { return conflict; }
  StringRef conflictMessage() const { return conflictMessage_; }

  llvm::DenseMap<hivm::CustomMacroOp, llvm::SmallVector<int64_t>> &
  resolvedSlotEventIds() {
    return slotEventIds;
  }

  const llvm::DenseMap<hivm::CustomMacroOp, llvm::SmallVector<int64_t>> &
  resolvedSlotEventIds() const {
    return slotEventIds;
  }

  void collectReservedEventIds(func::FuncOp funcOp,
                               const SyncSolverOptions &options);

  void applyReservedEventIds(
      llvm::function_ref<std::unique_ptr<EventIdSolver> &(hivm::PIPE,
                                                          hivm::PIPE)>
          getEventIdSolver);

  void validatePinnedAssignments(
      llvm::ArrayRef<std::unique_ptr<ConflictPair>> conflictPairs);

  void injectBoundarySync(SyncMap &syncMapBefore, SyncMap &syncMapAfter,
                          func::FuncOp funcOp, OperationBase *funcIr,
                          const SyncSolverOptions &options,
                          llvm::function_ref<std::unique_ptr<EventIdSolver> &(
                              hivm::PIPE, hivm::PIPE)>
                              getEventIdSolver);

private:
  void markConflict(StringRef message);

  bool conflict{false};
  std::string conflictMessage_;
  llvm::DenseMap<std::tuple<hivm::PIPE, hivm::PIPE>, llvm::SmallVector<int64_t>>
      reservedEventIds;
  llvm::DenseMap<hivm::CustomMacroOp, llvm::SmallVector<int64_t>> slotEventIds;
};

class CustomMacroSyncCodegenState {
public:
  void setResolvedSlotEventIds(
      llvm::DenseMap<hivm::CustomMacroOp, llvm::SmallVector<int64_t>> ids) {
    resolvedIds = std::move(ids);
  }

  void recordBoundaryArgIfNeeded(
      OperationBase *opBase, SyncOp *syncOp, IRRewriter &rewriter,
      llvm::function_ref<Value(SetWaitOp *, Location)> getEventIdValue);

  void populateSyncRelatedArgs(func::FuncOp funcOp, IRRewriter &rewriter);

private:
  llvm::DenseMap<hivm::CustomMacroOp, llvm::SmallVector<Value>> collectedArgs;
  llvm::DenseMap<hivm::CustomMacroOp, llvm::SmallVector<int64_t>> resolvedIds;
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_CUSTOMMACROSYNC_H

//===- BufferPoolSync.h - Buffer pool synchronization ---------*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_BUFFERPOOLSYNC_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_BUFFERPOOLSYNC_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/EventIdSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::hivm::syncsolver {

/// Adapt explicit slot dependencies to the ordinary GSS conflict-pair pipeline.
/// A binding is one PoolSlotOp and its ready (producer) or release (consumer)
/// boundary. The translated GSS accesses inside that span provide its concrete
/// MTE2 writes and MMAD's virtual MTE1 readers. Derived views are
/// traced by IRTranslator; bindingIndex indexes exact PoolSlotOp results.
/// Enable places the binding inside its task guard. Its scope bounds placement;
/// nested layout expansion uses ordinary GSS rules without widening that guard.
///
/// Members share an EventIdNode per scheduling loop, pool and direction:
/// MTE2 -> MTE1 ready, MTE1 -> MTE2 free. Ordinary GSS allocates N IDs;
/// the two endpoints select IDs with their own access slots. Ready dependencies
/// run producer -> consumer; free dependencies run consumer -> next producer.
/// L1 padding keeps ordinary protection and the pool free WAIT stays at DMA.
///
/// covers() exempts only collected hazards ordered by these declared edges,
/// including producer reuse through ready -> consumer -> free. Forward hazards
/// within one binding, padding and accesses outside the pool stay in GSS.
class BufferPoolSync {
public:
  using GetSolver =
      llvm::function_ref<std::unique_ptr<EventIdSolver> &(PIPE, PIPE)>;

  LogicalResult collect(func::FuncOp function, OperationBase *ir,
                        const SyncSolverOptions &options);
  void reset(GetSolver getSolver);
  bool managesEventIds(PIPE src, PIPE dst) const;
  EventIdNode *getEventIdNode(const BufferPoolInfo &pool) const;
  void setEventIdNode(const BufferPoolInfo &pool, EventIdNode *node);
  void clearEventIdNodes() { sharedEventIdNodes.clear(); }
  LogicalResult forEachDependency(
      const DenseMap<OperationBase *, std::vector<Occurrence *>> &occurrences,
      llvm::function_ref<LogicalResult(Occurrence *, Occurrence *, PIPE, PIPE,
                                       EventIdInfo)>
          visit);
  bool hasAccess(RWOperation *op) const;
  bool covers(RWOperation *a, RWOperation *b, const MemInfo &infoA,
              const MemInfo &infoB) const;
  void eraseBoundaries();

private:
  using PipePair = std::pair<PIPE, PIPE>;
  struct Binding {
    PoolSlotOp slot;
    annotation::MarkOp end;
    size_t group;
    bool producer;
    // The translated region containing the binding. Concrete access
    // endpoints retain MMAD virtual readers for the existing placement logic.
    Scope *scope = nullptr;
    SmallVector<RWOperation *> accesses;
  };
  struct Pool {
    LoopLikeOpInterface loop;
    Block *body;
    Value root;
    std::optional<int64_t> id;
    int64_t size;
    bool hasProducer = false;
    bool hasConsumer = false;
    Loop *loopIR = nullptr;
  };

  SmallVector<Binding> bindings;
  SmallVector<Pool> pools;
  DenseMap<Value, size_t> bindingIndex;
  DenseMap<PipePair, SmallVector<int64_t>> fixedIds;
  DenseMap<std::tuple<Loop *, Value, unsigned>, EventIdNode *>
      sharedEventIdNodes;
};
} // namespace mlir::hivm::syncsolver
#endif

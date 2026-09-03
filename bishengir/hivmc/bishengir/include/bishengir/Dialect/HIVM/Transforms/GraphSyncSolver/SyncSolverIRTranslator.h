//===--------- IRTranslator.h ---- Graph Sync Solver ------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <utility>

namespace mlir::hivm::syncsolver {

class IRTranslator {
public:
  int64_t globalIndex{0};
  bool decomposeMmadl1Op{true};

  // Synchronization mode (normal or cross-core).
  SyncMode syncMode{SyncMode::NORMAL_SYNC};

  // Original MLIR function being processed (may be null for test-only Solver).
  func::FuncOp funcOp;

  // Architecture is memory based (A2/A3).
  bool isMemBasedArch{false};

  // Architecture is register based (A5).
  bool isRegBasedArch{true};

  // In-memory hierarchical IR (Function -> Scopes -> Ops) used by the solver.
  std::unique_ptr<OperationBase> funcIr;

  // Linearized occurrence sequence (sync IR) built from funcIr, each Occurrence
  // represents one appearance of an operation in the sync-analysis order.
  std::vector<std::unique_ptr<Occurrence>> syncIr;

  // Set of RW operations that expose unit-flag feature and need special
  // handling.
  llvm::DenseSet<RWOperation *> unitFlagFeaturedOps;

  // Map op -> list of occurrences in syncIr (quick lookup for an op's
  // occurrences).
  llvm::DenseMap<OperationBase *, std::vector<Occurrence *>> opAllOccurrences;

  // For a parent occurrence, list of its child occurrences.
  llvm::DenseMap<Occurrence *, llvm::SmallVector<Occurrence *>> occChildrenMem;

  // Processing order list created from syncIr that drives pairwise conflict
  // checks.
  std::vector<ProcessingOrder> processingOrders;

public:
  IRTranslator(SyncMode syncMode, func::FuncOp func)
      : syncMode(syncMode), funcOp(func) {
    auto funcOp = std::make_unique<syncsolver::Function>(func.getOperation());
    auto scopeOp = funcIrBuilder(func.getRegion(), funcOp.get());
    funcOp->body.push_back(std::move(scopeOp));
    funcIr = std::move(funcOp);
    syncIrBuilder(funcIr.get());
  }

  IRTranslator(SyncMode syncMode, std::unique_ptr<OperationBase> funcIr)
      : syncMode(syncMode), funcIr(std::move(funcIr)) {
    syncIrBuilder(this->funcIr.get());
  }

private:
  // Convert MLIR Region into the in-memory funcIr Scope representation.
  std::unique_ptr<Scope> funcIrBuilder(Region &region, OperationBase *parentOp);

  // Create a decomposed representation for certain MMAD L1 ops if enabled.
  std::unique_ptr<OperationBase> getDecomposedMmadl1(hivm::MmadL1Op mmadl1Op,
                                                     OperationBase *parentOp);

  // Generate processing orders (various flavors) used by the main algorithm.
  void generateProcessingOrders(Occurrence *scopeOcc, int l, int r,
                                bool isUseless);

  void generateProcessingOrders(int l, int r, bool isUseless);

  void generateProcessingOrders(int l1, int r1, int l2, int r2, bool isUseless);

  // Build sync IR occurrences from the operation tree.
  void syncIrBuilder(OperationBase *op, Occurrence *parentOcc = nullptr,
                     int depth = 0, bool isUseless = false);

  // Collect pointer-like operands reachable from a Value.
  llvm::SmallVector<Value> collectPointerLikeOps(Value val,
                                                 func::FuncOp funcOp);

  // Extract memory-related Values from a list of pointer values.
  llvm::SmallVector<Value>
  getMemoryOps(const SmallVector<Value> &vals,
               std::optional<func::FuncOp> funcOp = {});

  // Return read and write memory operand lists for an MLIR operation.
  std::pair<llvm::SmallVector<Value>, llvm::SmallVector<Value>>
  getReadWriteMemoryOps(Operation *op);

  // Return a wrapped Load/Store RWOperation when encountering affine/memref
  // load/store ops.
  template <typename OP>
  std::unique_ptr<OperationBase> getLoadStoreOp(OP op, OperationBase *parentOp);

  std::unique_ptr<OperationBase> getPipeInterfaceOp(hivm::OpPipeInterface op,
                                                    OperationBase *parentOp);

  std::unique_ptr<OperationBase> getTensorExtractOp(tensor::ExtractOp extractOp,
                                                    OperationBase *parentOp);

  std::unique_ptr<OperationBase> getCallOp(func::CallOp callOp,
                                           OperationBase *parentOp);

  std::optional<hivm::PIPE>
  getInferredPipe(Operation *op, TCoreType coreType,
                  const llvm::SmallVector<Value> &writeMemInfo);

  bool isUnlikelyCondition(Condition *condOp);
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H

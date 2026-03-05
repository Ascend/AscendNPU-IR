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
  // Configuration options.
  const SyncSolverOptions options;

  // Original MLIR function being processed (may be null for test-only Solver).
  func::FuncOp funcOp;

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

  // Processing order list created from syncIr that drives pairwise conflict
  // checks.
  std::vector<ProcessingOrder> processingOrders;

public:
  IRTranslator(func::FuncOp func, SyncSolverOptions options)
      : options(options), funcOp(func) {
    auto funcOp = std::make_unique<syncsolver::Function>(func.getOperation());
    auto scopeOp = funcIrBuilder(func.getRegion(), funcOp.get());
    funcOp->body.push_back(std::move(scopeOp));
    funcIr = std::move(funcOp);
    syncIrBuilder(funcIr.get());
  }

  IRTranslator(std::unique_ptr<OperationBase> funcIr, SyncSolverOptions options)
      : options(options), funcIr(std::move(funcIr)) {
    syncIrBuilder(this->funcIr.get());
  }

private:
  int64_t globalIndex{0};

  // Convert MLIR Region into the in-memory funcIr Scope representation.
  std::unique_ptr<Scope> funcIrBuilder(Region &region, OperationBase *parentOp,
                                       bool skipEmptyScopes = false);

  // Create a decomposed representation for certain MMAD L1 ops if enabled.
  std::unique_ptr<OperationBase> getDecomposedMmadl1(hivm::MmadL1Op mmadl1Op,
                                                     OperationBase *parentOp);

  // Generate processing orders (various flavors) used by the main algorithm.
  void generateProcessingOrders(Occurrence *occ1, Occurrence *occ2,
                                bool isUseless);
  void generateProcessingOrders(Loop *loopOp, Occurrence *occ, bool isUseless);
  void generateProcessingOrders(Scope *scopeOp, Occurrence *occ,
                                bool isUseless);
  void generateProcessingOrders(const llvm::SmallVector<Occurrence *> &occs,
                                bool isUseless);
  void generateProcessingOrders(const llvm::SmallVector<Occurrence *> &occs1,
                                const llvm::SmallVector<Occurrence *> &occs2,
                                bool isUseless);
  void generateProcessingOrders(RWOperation *rwOp1, RWOperation *rwOp2,
                                Occurrence *occ1, Occurrence *occ2,
                                bool isUseless);

  bool skipLaterIterations(Occurrence *occ1, Occurrence *occ2);

  // Build sync IR occurrences from the operation tree.
  void syncIrBuilder(OperationBase *op, Occurrence *parentOcc = nullptr,
                     int depth = 0, bool isUseless = false);

  // Collect pointer-like operands reachable from a Value.
  llvm::SmallVector<Value> tracebackMemVals(Value val, func::FuncOp funcOp);
  llvm::SmallVector<Value> tracebackMemValsStep(Value val);

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

  bool isParallelLoop(Loop *loopOp);
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERIRTRANSLATOR_H

//===-------- ReorderInstructionImpl.h - Impl. of instruction reorder -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TRITON_REORDERINSTRUCTION_H
#define BISHENGIR_DIALECT_TRITON_REORDERINSTRUCTION_H

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <utility>

namespace mlir {
namespace triton {
class ReorderInstructionsImpl {
public:
  ReorderInstructionsImpl() = default;

  void commonInstructionReorder(ModuleOp m, MLIRContext *ctx);

private:
  Value traceToPtr(Value val, FuncOp funcOp);

  Value traceToBlockOperand(BlockArgument arg, Operation *parentOp);

  LogicalResult visit(LoadOp loadOp);

  LogicalResult visit(StoreOp storeOp);

  LogicalResult getOpMemoryEffects(Operation *op);

  void constructTopoGraph(Block &block,
                          llvm::DenseMap<Operation *, int64_t> &outDegree,
                          llvm::DenseSet<Operation *> &tailNodes);

  int64_t
  calculateRegisterUsage(Operation *op,
                         llvm::DenseMap<Operation *, int64_t> outDegree);

  bool isReadConflict(Operation *op, llvm::DenseSet<Operation *> &reorderedOps);

  bool isWriteConflict(Operation *op,
                       llvm::DenseSet<Operation *> &reorderedOps);

  Operation *greedySelectSubgraph(
      llvm::SmallVector<std::pair<int64_t, Operation *>> &registerUsage,
      llvm::DenseSet<Operation *> &reorderedOps);

  void recursiveInsertOpTree(OpBuilder &builder, IRMapping &mapping,
                             Operation *curNode,
                             llvm::DenseSet<Operation *> &reorderedOps);

  void reorderInstructions(Block &block,
                           llvm::DenseMap<Operation *, int64_t> &outDegree,
                           llvm::DenseSet<Operation *> &tailNodes,
                           OpBuilder &builder);

  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> memReadOps;
  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> memWriteOps;
  llvm::DenseMap<Value, llvm::DenseSet<Operation *>> ptrReadByOps;
  llvm::DenseMap<Value, llvm::DenseSet<Operation *>> ptrWriteByOps;
};
} // namespace triton
} // namespace mlir

#endif // BISHENGIR_DIALECT_TRITON_REORDERINSTRUCTION_H

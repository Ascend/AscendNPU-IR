//===--------- PlanContext.h - Plan data structures for AV2
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_PLANCONTEXT_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_PLANCONTEXT_H

#include "bishengir/Dialect/Analysis/VFFusion/VFStackInfo.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/FusedNode.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/FusionUtils.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/RetriedOptions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/MapVector.h"
#include <memory>
#include <string>

namespace mlir {
namespace hfusion {

// Every fusable op(including LinalgOp and interleave/deinterleave) corresponds
// to a FusableOpInfo.
struct FusableOpInfo {
  template <typename PivotTy>
  using ConflictPivotMap = DenseMap<Operation *, DenseSet<PivotTy>>;

  std::string label;
  int64_t numLoops = 0;
  unsigned numReductionLoops = 0;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> tileSize;
  SmallVector<int64_t> tileInterchange;
  unsigned maxElemBitWidth = 1;
  // TODO: Conflict storage (copy/sync/op/group) currently lives in PlanContext
  // for convenience. Move to a dedicated ConflictTracker class to separate plan
  // state from conflict tracking, making incremental update semantics clearer.
  DenseSet<Operation *> copyConflicts;
  DenseSet<Operation *> syncConflicts;
  ConflictPivotMap<Operation *> opConflicts;
  ConflictPivotMap<const FusedNode *> groupConflicts;
  std::shared_ptr<FusedNode> fusedNode = nullptr;
};

class PlanContext {
  unsigned loopCount = 0;
  std::string nextLoopLabel() {
    return "outlined-loop-target-" + std::to_string(++loopCount);
  }

  unsigned maxFusedOps;
  bool enableMultipleConsumerFusion;
  bool enableCrossIfFusion;
  unsigned vectorLength;

  analysis::VFStackInfoBuilder &vfStack;

  llvm::MapVector<Operation *, FusableOpInfo> opInfoMap;

  SmallVector<std::shared_ptr<FusedNode>> fusedNodesInBlock;

public:
  PlanContext(const RetriedOptions &options,
              analysis::VFStackInfoBuilder &vfStack)
      : maxFusedOps(options.maxFusedOps),
        enableMultipleConsumerFusion(options.enableMultipleConsumerFusion),
        enableCrossIfFusion(options.enableCrossIfFusion),
        vectorLength(options.vectorLength), vfStack(vfStack) {}

  bool canFitStack(ArrayRef<Operation *> ops) const {
    return vfStack.fitsStack(ops);
  }

  unsigned getMaxFusedOps() const { return maxFusedOps; }
  bool getEnableMultipleConsumerFusion() const {
    return enableMultipleConsumerFusion;
  }
  bool getEnableCrossIfFusion() const { return enableCrossIfFusion; }
  unsigned getVectorLength() const { return vectorLength; }

  auto nodes() const {
    return llvm::make_range(fusedNodesInBlock.begin(), fusedNodesInBlock.end());
  }
  size_t size() const { return fusedNodesInBlock.size(); }
  std::shared_ptr<FusedNode> add(Operation *op) {
    auto fusedNode = std::make_shared<FusedNode>(*this, nextLoopLabel());
    fusedNodesInBlock.push_back(fusedNode);
    fusedNode->addLeaf(op);
    return fusedNode;
  }
  void resetForBlock() { fusedNodesInBlock.clear(); }

  bool hasOpInfo(Operation *op) const { return opInfoMap.contains(op); }
  FusableOpInfo &getInfo(Operation *op) {
    auto it = opInfoMap.find(op);
    if (it != opInfoMap.end())
      return it->second;
    llvm::report_fatal_error("get op info on not vectorize op.");
  }
  const FusableOpInfo &getInfo(Operation *op) const {
    auto it = opInfoMap.find(op);
    if (it != opInfoMap.end())
      return it->second;
    llvm::report_fatal_error("get op info on not vectorize op.");
  }
  auto opInfos() const {
    return llvm::make_range(opInfoMap.begin(), opInfoMap.end());
  }

  // Populate op info from `func`: label fusable ops, compute shapes, build
  // conflict lists. This is the main entry point for initializing PlanContext
  // from a device function before fusion planning.
  void initFusableOpInfoFrom(func::FuncOp func);

  // Compute tile sizes for all fused ops after fusion planning.
  void computeTileSize();

  void addSyncConflict(Operation *a, Operation *b);
  void addCopyConflict(Operation *a, Operation *b);
  void addOpConflict(Operation *a, Operation *b, Operation *pivot);
  void addGroupConflict(Operation *a, Operation *b, const FusedNode *pivot);

  bool hasConflict(Operation *a, Operation *b) const;
  bool hasConflict(const FusableOpInfo &a, Operation *b) const;

  void dissolvePivot(Operation *newOp, const FusedNode *node, Block *block);

private:
  // Register a fusable op with its label; computes numLoops/shape/bitWidth.
  void registerAndAnalyzeOp(Operation *op, const std::string &label);

  // Build conflict lists for all fusable ops in `func`.
  void computeConflictLists(func::FuncOp func);
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_PLANCONTEXT_H

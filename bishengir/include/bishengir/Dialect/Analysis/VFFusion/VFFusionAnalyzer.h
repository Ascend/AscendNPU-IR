//===- VFFusionAnalyzer.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_ANALYZER_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_ANALYZER_H

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionBlock.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFUnionFind.h"
#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir::analysis {

//===----------------------------------------------------------------------===//
// VFFusionAnalyzerBase
//===----------------------------------------------------------------------===//

// TODO: find a way to separate the implementation and declaration
template <class AnalyzerClass> class VFFusionAnalyzerBase {
public:
  /// Traversal order is done in PreOrder walk manner.
  /// Map from operations to their indices in the block traversal order.
  DenseMap<Operation *, size_t> opToIndex;

  /// Map from operations to their indices in the block traversal order.
  SmallVector<Operation *> opsInBlock;

  /// Union-find data structure representing fused operation groups.
  VFUnionFind dsu;

  /// Constructs a fusion analyzer with the specified fusion options.
  ///
  /// @param optionArg Configuration options controlling fusion behavior.
  explicit VFFusionAnalyzerBase(const VFFusionKindOption &optionArg)
      : option(optionArg) {};

  /// Implementation of the fusion algorithm
  /// WARN: must be overridden by derived classes.
  ///
  /// This method should analyze operations in the block and fuse compatible
  /// operations by calling `fuseIndexWith()` or similar methods.
  ///
  /// @param block The block containing operations to analyze for fusion.
  /// @return Success if fusion analysis completes, failure otherwise.
  LogicalResult fuseImpl(Block &block) {
    llvm_unreachable(
        "missing implementation fuseImpl for the specified FusionKind");
  }

  /// Retrieves the fused operation blocks after fusion analysis.
  ///
  /// This method first performs fusion analysis on the block, then groups
  /// operations according to their fusion sets. Operations with no operands
  /// are excluded from the results.
  ///
  /// @param block The block to analyze and retrieve fused groups from.
  /// @return A vector of `VFFusionBlock` objects, each containing operations
  ///         that should be fused together, or failure if fusion analysis
  ///         fails.
  FailureOr<SmallVector<VFFusionBlock>> retrieveFusedBlocks(Block &block) {
    if (failed(fuse(block)))
      return failure();

    SmallVector<VFFusionBlock> fusedBlocks(dsu.minIndex.size());

    for (Operation &op : block.getOperations()) {
      int parentIndex = dsu.find(opToIndex.at(&op));
      if (isSafeToExcludeOps(&op))
        continue;
      fusedBlocks[parentIndex].fuseOp(&op);
    }
    return fusedBlocks;
  }

  bool isOutlineableOp(Operation *op) const;
  bool fuseIndexWith(int x, int y);
  bool fuseOp(const VFFusionAnalyzerBase &block);

  LogicalResult fuse(Block &block) {
    return static_cast<AnalyzerClass *>(this)->fuseImpl(block);
  }

  virtual ~VFFusionAnalyzerBase() = default;

protected:
  /// Checks if fusing two operations would violate topological ordering.
  ///
  /// This ensures that all operations between two fused operations are also
  /// fused, preventing cases where unfused intermediate operations would
  /// break the dependency chain.
  ///
  /// Example:
  /// ```
  /// a -> b -> d
  ///  \_ c _/
  /// ```
  /// If `a`, `b`, and `d` are fused, then `c` must also be fused to maintain
  /// valid topological order.
  ///
  /// @param x Index of the first operation.
  /// @param y Index of the second operation.
  /// @return True if fusing would create invalid dependencies.
  bool hasInvalidDependencyIfFused(int x, int y);

  /// Validates that reshape operations remain at group boundaries after fusion.
  ///
  /// Reshape operations should only appear at the beginning or end of a
  /// fused group to maintain valid data flow transformations.
  bool areReshapesValidIfFused(const size_t xIndex, const size_t yIndex);

  /// Extended fusibility check (must be overridden by derived classes).
  ///
  /// This allows derived classes to implement fusion-kind-specific checks
  /// beyond the base validation logic.
  bool isFusibleImpl(const int xIndex, const int yIndex) {};

  // Check if two operations are fusible. (only fusible if it's on the same
  // block)
  bool isFusible(const int xIndex, const int yIndex) {
    Operation *const x = opsInBlock[xIndex];
    Operation *const y = opsInBlock[yIndex];
    assert(opToIndex.contains(x) && "missing operation in opToIndex");
    assert(opToIndex.contains(y) && "missing operation in opToIndex");

    // For CV affinity cases: split-mix-kernel is unavailable before vffusion pass.
    // Temporarily avoid vectorizing into vf functions via isInCubeScope
    // for vector ops in cube scope.
    if (scope::utils::isInCubeScope(x) || scope::utils::isInCubeScope(y))
      return false;

    if (!this->isOutlineableOp(x) || !this->isOutlineableOp(y))
      return false;

    if (hasInvalidDependencyIfFused(xIndex, yIndex))
      return false;

    if (!areReshapesValidIfFused(xIndex, yIndex))
      return false;

    return static_cast<AnalyzerClass *>(this)->isFusibleImpl(xIndex, yIndex);
  }

  // extended implementation for initialization.
  void initializeImpl(Block &block) {};

  void initialize(Block &block) {
    opToIndex.clear();
    opsInBlock.clear();

    size_t numberOps = 0;
    block.walk<WalkOrder::PreOrder>([this, &numberOps](Operation *const op) {
      opToIndex[op] = opsInBlock.size();
      opsInBlock.push_back(op);
      ++numberOps;
    });
    dsu = VFUnionFind(opsInBlock);

    // run extension initialization of specified fusionkind
    return static_cast<AnalyzerClass *>(this)->initializeImpl(block);
  }

  const VFFusionKindOption option;
};

template <class AnalyzerClass>
bool VFFusionAnalyzerBase<AnalyzerClass>::isOutlineableOp(
    Operation *const op) const {
  // skip control-flow operation
  if (!this->option.enableOutlineCF &&
      op->hasTrait<RegionBranchOpInterface::Trait>())
    return false;

  if (!this->option.enableOutlineArith &&
      isa<arith::ArithDialect>(op->getDialect())) {
    return false;
  }
  auto checkResult = op->walk([this](Operation *const opInside) -> WalkResult {
    // skip operation with memref operands
    if (!this->option.enableOutlineMemref &&
        any_of(opInside->getOperandTypes(),
               [](auto type) { return isa<MemRefType>(type); })) {
      return WalkResult::interrupt();
    }

    // skip arith operations
    return WalkResult::advance();
  });

  if (checkResult.wasInterrupted())
    return false;

  return !isa<func::CallOp>(op) && !reshape_utils::isReturnOp(op) &&
         !op->hasTrait<OpTrait::ReturnLike>();
}

template <class AnalyzerClass>
bool VFFusionAnalyzerBase<AnalyzerClass>::hasInvalidDependencyIfFused(
    const int x, const int y) {
  const int pxMax = (int)dsu.getMaxIndexUnion(x);
  const int pyMax = (int)dsu.getMaxIndexUnion(y);
  const int maxTopoRank = std::max(pyMax, pxMax);
  // all users of every ops in either unions should be defined later than it.
  // TODO: optimize using the smaller to larger technique instead of O(n)
  for (Operation *const op : opsInBlock) {
    const int curOpIndex = (int)dsu.getMaxIndexUnion(opToIndex.at(op));
    if (curOpIndex != pxMax && curOpIndex != pyMax)
      continue;
    for (Operation *const user : op->getUsers()) {
      const int opUnionIndex = (int)dsu.getMaxIndexUnion(opToIndex.at(user));
      // not in either unions
      if (opUnionIndex == pxMax || opUnionIndex == pyMax)
        continue;
      // will have not dominate use error
      if (maxTopoRank > opUnionIndex)
        return true;
    }
  }
  return false;
}

template <class AnalyzerClass>
bool VFFusionAnalyzerBase<AnalyzerClass>::fuseIndexWith(const int x,
                                                        const int y) {
  return dsu.join(x, y);
}

// consider case:
//   _ op1 _
//  /       \
// r        op3
//  \_ op2 _ /
// NOTE: can be optimized to not revisit the same operations multiple times
template <class AnalyzerClass>
bool VFFusionAnalyzerBase<AnalyzerClass>::areReshapesValidIfFused(
    const size_t xIndex, const size_t yIndex) {
  auto xOp = opsInBlock[xIndex];
  auto yOp = opsInBlock[yIndex];
  if (!isReshapeOp(xOp) && !isReshapeOp(yOp)) {
    return true;
  }
  if (isReshapeOp(xOp) &&
      isExpandShapeOpCanFuseIntoVsstbPatternTranspose(xOp)) {
    return true;
  }
  if (isReshapeOp(yOp) &&
      isExpandShapeOpCanFuseIntoVsstbPatternTranspose(yOp)) {
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// AllOpKindAnalyzer
//===----------------------------------------------------------------------===//

class AllOpKindAnalyzer : public VFFusionAnalyzerBase<AllOpKindAnalyzer> {
public:
  AllOpKindAnalyzer() = delete;

  bool isFusibleImpl(int xIndex, int yIndex);
  LogicalResult fuseImpl(Block &block);

  explicit AllOpKindAnalyzer(const VFFusionKindOption &option)
      : VFFusionAnalyzerBase<AllOpKindAnalyzer>(option) {};
  ~AllOpKindAnalyzer() override = default;
};

//===----------------------------------------------------------------------===//
// NMostOpKindAnalyzer
//===----------------------------------------------------------------------===//

class NMostOpKindAnalyzer : public VFFusionAnalyzerBase<NMostOpKindAnalyzer> {
public:
  NMostOpKindAnalyzer() = delete;

  bool isFusibleImpl(int xIndex, int yIndex);
  LogicalResult fuseImpl(Block &block);

  NMostOpKindAnalyzer(const VFFusionKindOption &option,
                      const size_t maxNumberOp)
      : VFFusionAnalyzerBase<NMostOpKindAnalyzer>(option), N(maxNumberOp) {};
  ~NMostOpKindAnalyzer() override = default;

private:
  // max number operations fusible ops including the ops inside regions.
  const size_t N;
};

//===----------------------------------------------------------------------===//
// MaxParallelAnalyzer
//===----------------------------------------------------------------------===//

class MaxParallelAnalyzer : public VFFusionAnalyzerBase<MaxParallelAnalyzer> {
public:
  MaxParallelAnalyzer() = delete;

  bool isFusibleImpl(int xIndex, int yIndex);
  void initializeImpl(Block &block);
  LogicalResult fuseImpl(Block &block);

  explicit MaxParallelAnalyzer(const VFFusionKindOption &option)
      : VFFusionAnalyzerBase<MaxParallelAnalyzer>(option){};
  ~MaxParallelAnalyzer() override = default;

private:
  std::vector<OpOperand *> getSortedConsumerOperands(Operation *producerOp);
  bool hasReductionToConsumer(const int producerIndex, const int consumerIndex);
  bool areFusibleOps(const int producerIndex, const int consumerIndex);
  bool fuseProducerConsumerImpl(Block &block);
  bool fuseIOBoundGroupsWithNearestConsumer();
  bool isIOBoundGroup(int groupId);
  bool parallelismSubModel(DenseSet<Operation *> &producerGroup,
                                  DenseSet<Operation *> &consumerGroup) const;
  bool execUnitUtilizationSubModel(DenseSet<Operation *> &producerGroup,
                                          DenseSet<Operation *> &consumerGroup) const;
  bool canFuseGroups(int producerGroupId, int consumerGroupId, int producerIndex);
  bool mergeGroups(const int producerGroupId, const int consumerGroupId);
  bool tryFuseGroups(int producerIndex, int consumerIndex,
                      int producerGroupId, int consumerGroupId);
  void printValidGroupCount();
  DenseMap<Operation *, size_t> opToGroupIndex;
  DenseMap<int64_t, DenseSet<Operation *>> AllFusedGroupBlocks;
  int stage = 1;
};

//===----------------------------------------------------------------------===//
// UBAwareOpKindAnalyzer
//===----------------------------------------------------------------------===//

/// Like AllOpKindAnalyzer but refuses merges whose estimated caller-side UB
/// would exceed a budget. Uses a consumer lookahead to speculatively expand
/// groups and a split-cost guard to allow merges that are no worse than
/// keeping groups separate.
class UBAwareOpKindAnalyzer
    : public VFFusionAnalyzerBase<UBAwareOpKindAnalyzer> {
public:
  UBAwareOpKindAnalyzer() = delete;

  bool isFusibleImpl(int xIndex, int yIndex);
  LogicalResult fuseImpl(Block &block);

  /// \param maxLookaheadDepth  How many rounds of consumer expansion to try
  ///   before falling back to the split-cost guard (0 disables lookahead).
  UBAwareOpKindAnalyzer(const VFFusionKindOption &option, int64_t ubBudgetBytes,
                        int64_t ubAlignBytes, int maxLookaheadDepth = 2)
      : VFFusionAnalyzerBase<UBAwareOpKindAnalyzer>(option),
        ubBudgetBytes_(ubBudgetBytes), ubAlignBytes_(ubAlignBytes),
        maxLookaheadDepth_(maxLookaheadDepth) {}
  ~UBAwareOpKindAnalyzer() override = default;

private:
  const int64_t ubBudgetBytes_;
  const int64_t ubAlignBytes_;
  const int maxLookaheadDepth_;

  int64_t estimateMergedGroupBytes(int xIndex, int yIndex);
  int64_t estimateSplitCostBytes(int xIndex, int yIndex);
};

} // namespace mlir::analysis

#endif

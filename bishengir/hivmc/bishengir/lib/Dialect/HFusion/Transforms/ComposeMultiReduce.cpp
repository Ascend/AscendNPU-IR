//===- ComposeMultiReduce.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"
#include "bishengir/Dialect/Utils/ReachabilityAnalyzer.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Analysis/TopologicalSortUtils.h"

#include "llvm/Support/Debug.h"
namespace mlir {
#define GEN_PASS_DEF_COMPOSEMULTIREDUCE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-compose-multi-reduce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::utils;
using namespace mlir::utils::debugger;

namespace {
struct ComposeMultiReducePass
    : public impl::ComposeMultiReduceBase<ComposeMultiReducePass> {
  explicit ComposeMultiReducePass(const ComposeMultiReduceOptions &options)
      : ComposeMultiReduceBase(options) {}

  /// Main entry point for the pass
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    ReachabilityAnalyzer analyzer(funcOp);

    initOpt();

    SmallVector<Operation *> reduceOps = collectReduceOperations(analyzer);
    SmallVector<SmallVector<linalg::ReduceOp>> reduceGrouping =
        groupCompatibleReduceOps(reduceOps, analyzer);

    composeGroupedOperations(reduceGrouping);
  }

private:
  /// Initialize pass options with defaults if not set
  void initOpt() {
    if (maxCompose == -1)
      maxCompose = std::numeric_limits<int>::max();
    if (maxDistDiff == -1)
      maxDistDiff = std::numeric_limits<int>::max();
  }

  /// Collect all reduce operations from the function
  /// @param analyzer The reachability analyzer for the function
  /// @return Vector of reduce operations
  SmallVector<Operation *>
  collectReduceOperations(const ReachabilityAnalyzer &analyzer) const {
    SmallVector<Operation *> reduceOps;
    for (Operation *op : analyzer.operationList) {
      if (isa<linalg::ReduceOp>(op)) {
        reduceOps.push_back(op);
      }
    }
    return reduceOps;
  }

  /// Group compatible reduce operations that can be composed
  /// @param reduceOps List of reduce operations to group
  /// @param analyzer Reachability analyzer for dependency checking
  /// @return Groups of compatible reduce operations
  SmallVector<SmallVector<linalg::ReduceOp>>
  groupCompatibleReduceOps(const SmallVector<Operation *> &reduceOps,
                           const ReachabilityAnalyzer &analyzer) const {
    SmallVector<SmallVector<linalg::ReduceOp>> reduceGrouping;

    for (Operation *tmpOp : reduceOps) {
      auto reduceOp = cast<linalg::ReduceOp>(tmpOp);
      if (!reduceOp || !isValidForGrouping(reduceOp)) {
        continue;
      }

      bool grouped = tryAddToExistingGroup(reduceOp, reduceGrouping, analyzer);
      if (!grouped) {
        LLVM_DEBUG(llvm::dbgs() << "Making new group\n";);
        reduceGrouping.push_back({reduceOp});
      }
    }

    return reduceGrouping;
  }

  /// Check if a reduce operation is valid for grouping
  /// @param reduceOp The reduce operation to check
  /// @return True if valid for grouping
  bool isValidForGrouping(linalg::ReduceOp reduceOp) const {
    LLVM_DEBUG(llvm::dbgs() << "Processing " << *reduceOp << "\n";);
    if (ShapedType::isDynamicShape(reduceOp.getStaticShape())) {
      LLVM_DEBUG(llvm::dbgs() << "Has dynamic shape, skipping ... \n";);
      return false;
    }
    return true;
  }

  /// Try to add a reduce operation to an existing group
  /// @param reduceOp The reduce operation to add
  /// @param reduceGrouping Existing groups of reduce operations
  /// @param analyzer Reachability analyzer for dependency checking
  /// @return True if successfully added to a group
  bool tryAddToExistingGroup(
      linalg::ReduceOp reduceOp,
      SmallVector<SmallVector<linalg::ReduceOp>> &reduceGrouping,
      const ReachabilityAnalyzer &analyzer) const {
    for (size_t i = 0; i < reduceGrouping.size(); ++i) {
      if (reduceGrouping[i].size() >= static_cast<size_t>(maxCompose))
        continue;

      LDBG("Checking Grouping " << i);

      if (canGroupWithExisting(reduceOp, reduceGrouping[i], analyzer)) {
        LLVM_DEBUG(llvm::dbgs() << "OK Grouping\n";);
        reduceGrouping[i].push_back(reduceOp);
        return true;
      }
    }
    return false;
  }

  /// Check if a reduce operation can be grouped with existing operations
  /// @param reduceOp The reduce operation to check
  /// @param group Existing group of reduce operations
  /// @param analyzer Reachability analyzer for dependency checking
  /// @return True if can be grouped
  bool canGroupWithExisting(linalg::ReduceOp reduceOp,
                            SmallVector<linalg::ReduceOp> &group,
                            const ReachabilityAnalyzer &analyzer) const {
    ShapeCompatibilityResult shapeResult =
        checkShapeCompatibility(reduceOp, group[0]);
    if (!shapeResult.compatible) {
      return false;
    }

    if (!checkGroupDependencies(reduceOp, group, analyzer)) {
      return false;
    }

    // Swap if needed for aggressive mode
    if (shapeResult.needsSwap) {
      std::swap(group.front(), group.back());
    }

    return true;
  }

  /// Result of shape compatibility check
  struct ShapeCompatibilityResult {
    bool compatible;
    bool needsSwap;
  };

  /// Check if shapes and dimensions are compatible between reduce operations
  /// @param newOp New reduce operation to check
  /// @param baseOp Base reduce operation to compare against
  /// @return Compatibility result
  ShapeCompatibilityResult
  checkShapeCompatibility(linalg::ReduceOp newOp,
                          linalg::ReduceOp baseOp) const {
    auto newShape = utils::getShape(newOp.getDpsInputs().begin()->getType());
    auto baseShape = utils::getShape(baseOp.getDpsInputs().begin()->getType());
    auto newDimensions = newOp.getDimensions();
    auto baseDimensions = baseOp.getDimensions();

    if (aggressive) {
      return checkAggressiveShapeCompatibility(newShape, baseShape,
                                               newDimensions, baseDimensions);
    }
    return checkStrictShapeCompatibility(newShape, baseShape, newDimensions,
                                         baseDimensions);
  }

  /// Check strict shape compatibility (exact match)
  ShapeCompatibilityResult checkStrictShapeCompatibility(
      ArrayRef<int64_t> newShape, ArrayRef<int64_t> baseShape,
      ArrayRef<int64_t> newDimensions, ArrayRef<int64_t> baseDimensions) const {
    bool sameType = newShape == baseShape;
    bool sameDimension = newDimensions.equals(baseDimensions);
    if (!(sameType && sameDimension)) {
      LLVM_DEBUG(llvm::dbgs() << "Incompatible type\n";);
      return {false, false};
    }
    return {true, false};
  }

  /// Check aggressive shape compatibility (allows reshaping)
  ShapeCompatibilityResult checkAggressiveShapeCompatibility(
      ArrayRef<int64_t> newShape, ArrayRef<int64_t> baseShape,
      ArrayRef<int64_t> newDimensions, ArrayRef<int64_t> baseDimensions) const {
    bool swapped = false;
    if (newShape.size() > baseShape.size()) {
      std::swap(newShape, baseShape);
      std::swap(newDimensions, baseDimensions);
      swapped = true;
    }

    SmallVector<ReassociationIndices> supposedExpand;
    SmallVector<ReassociationIndices> supposedCollapse;
    SmallVector<int64_t> newExpandShape;

    bool can = areLooseReassociationsCompatible(
        supposedExpand, supposedCollapse, to_vector(newShape),
        to_vector(baseShape), newExpandShape);
    if (!can) {
      return {false, false};
    }

    tensor::reshape_utils::renumberReassociation(supposedCollapse);
    tensor::reshape_utils::renumberReassociation(supposedExpand);

    if (supposedCollapse.size() !=
        static_cast<size_t>(supposedCollapse.back().back() + 1)) {
      return {false, false};
    }

    if (!verifyDimensionCompatibility(newDimensions, baseDimensions,
                                      supposedExpand)) {
      return {false, false};
    }

    return {true, swapped};
  }

  /// Verify dimension compatibility after expansion
  bool verifyDimensionCompatibility(
      ArrayRef<int64_t> newDimensions, ArrayRef<int64_t> baseDimensions,
      const SmallVector<ReassociationIndices> &supposedExpand) const {
    SmallVector<int64_t> expandedDimensions;
    DenseSet<int64_t> dimensionSet(newDimensions.begin(), newDimensions.end());

    for (size_t dim = 0; dim < supposedExpand.size(); ++dim) {
      if (dimensionSet.contains(dim))
        expandedDimensions.append(supposedExpand[dim].begin(),
                                  supposedExpand[dim].end());
    }

    return baseDimensions.equals(expandedDimensions);
  }

  /// Check dependencies between a reduce operation and a group
  bool checkGroupDependencies(linalg::ReduceOp reduceOp,
                              const SmallVector<linalg::ReduceOp> &group,
                              const ReachabilityAnalyzer &analyzer) const {
    int64_t maxDist = 0;

    for (linalg::ReduceOp groupOp : group) {
      LLVM_DEBUG(llvm::dbgs() << "Checking with " << *groupOp << "\n";);

      if (analyzer.hasDataDependency(reduceOp, groupOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Has data dependency\n";);
        return false;
      }

      int64_t dist = analyzer.getShortestPathFromAncestor(reduceOp, groupOp);
      maxDist = std::max(maxDist, dist);
      if (maxDist > maxDistDiff) {
        LLVM_DEBUG(llvm::dbgs() << "Bad distance\n";);
        return false;
      }
    }

    return true;
  }

  /// Compose grouped reduce operations
  void composeGroupedOperations(
      const SmallVector<SmallVector<linalg::ReduceOp>> &reduceGrouping) const {
    for (auto &group : reduceGrouping) {
      if (group.size() > 1) {
        composeReduceOps(const_cast<SmallVector<linalg::ReduceOp> &>(group));
      }
    }
  }

  /// Adjust operands for aggressive mode by expanding/collapsing shapes
  /// @param builder IR builder for creating operations
  /// @param group Group of reduce operations to adjust
  /// @param newGroup Output group with adjusted operations
  void
  adjustOperandsForAggressive(OpBuilder &builder,
                              SmallVector<linalg::ReduceOp> &group,
                              SmallVector<linalg::ReduceOp> &newGroup) const {
    auto baseType = group[0].getDpsInputs()[0].getType();
    auto baseInitType = group[0].getDpsInits()[0].getType();
    auto basePivotShape = utils::getShape(baseType);
    auto basePivotInitShape = utils::getShape(baseInitType);

    for (linalg::ReduceOp &reduceOp : group) {
      linalg::ReduceOp adjustedOp = adjustSingleReduceOp(
          builder, reduceOp, basePivotShape, basePivotInitShape);
      if (adjustedOp) {
        newGroup.push_back(adjustedOp);
      }
    }
  }

  /// Adjust a single reduce operation for aggressive mode
  linalg::ReduceOp
  adjustSingleReduceOp(OpBuilder &builder, linalg::ReduceOp reduceOp,
                       ArrayRef<int64_t> basePivotShape,
                       ArrayRef<int64_t> basePivotInitShape) const {
    auto expandInfo =
        computeExpansionInfo(reduceOp, basePivotShape, basePivotInitShape);
    if (!expandInfo.needsExpansion) {
      return reduceOp;
    }

    builder.setInsertionPoint(reduceOp);
    SmallVector<Value> newOperands =
        expandOperands(builder, reduceOp, expandInfo);

    linalg::ReduceOp clonedOp =
        createExpandedReduceOp(builder, reduceOp, newOperands);

    replaceWithCollapsedResults(builder, reduceOp, clonedOp,
                                expandInfo.supposedExpandInit);

    reduceOp->erase();
    return clonedOp;
  }

  /// Information needed for expansion
  struct ExpansionInfo {
    bool needsExpansion;
    SmallVector<ReassociationIndices> supposedExpand;
    SmallVector<ReassociationIndices> supposedExpandInit;
    SmallVector<int64_t> newExpandShape;
    SmallVector<int64_t> newExpandShapeInit;
  };

  /// Compute expansion information for a reduce operation
  ExpansionInfo
  computeExpansionInfo(linalg::ReduceOp reduceOp,
                       ArrayRef<int64_t> basePivotShape,
                       ArrayRef<int64_t> basePivotInitShape) const {
    ExpansionInfo info;
    SmallVector<ReassociationIndices> dummyCollapse;

    bool can = areLooseReassociationsCompatible(
        info.supposedExpand, dummyCollapse,
        utils::getShape(reduceOp.getDpsInputs()[0].getType()),
        to_vector(basePivotShape), info.newExpandShape);

    bool canInit = areLooseReassociationsCompatible(
        info.supposedExpandInit, dummyCollapse,
        utils::getShape(reduceOp.getDpsInits()[0].getType()),
        to_vector(basePivotInitShape), info.newExpandShapeInit);
    if (!can || !canInit) {
      llvm_unreachable("Input and inits unexpectedly failed to collapse");
    }

    tensor::reshape_utils::renumberReassociation(info.supposedExpand);
    tensor::reshape_utils::renumberReassociation(info.supposedExpandInit);

    info.needsExpansion = static_cast<int64_t>(info.supposedExpand.size()) !=
                          info.supposedExpand.back().back() + 1;

    return info;
  }

  /// Expand operands according to expansion info
  SmallVector<Value> expandOperands(OpBuilder &builder,
                                    linalg::ReduceOp reduceOp,
                                    const ExpansionInfo &expandInfo) const {
    SmallVector<Value> newOperands;
    newOperands.reserve(reduceOp->getNumOperands());

    for (OpOperand &opr : reduceOp->getOpOperands()) {
      Value expandedValue =
          expandSingleOperand(builder, reduceOp, opr, expandInfo);
      newOperands.push_back(expandedValue);
    }

    return newOperands;
  }

  /// Expand a single operand
  Value expandSingleOperand(OpBuilder &builder, linalg::ReduceOp reduceOp,
                            OpOperand &opr,
                            const ExpansionInfo &expandInfo) const {
    Value operandVal = opr.get();

    if (reduceOp.isDpsInput(&opr)) {
      return expandOpr(builder, reduceOp.getLoc(), operandVal,
                       expandInfo.supposedExpand, expandInfo.newExpandShape);
    }
    if (reduceOp.isDpsInit(&opr)) {
      return expandOpr(builder, reduceOp.getLoc(), operandVal,
                       expandInfo.supposedExpandInit,
                       expandInfo.newExpandShapeInit);
    }

    return operandVal;
  }

  Value expandOpr(OpBuilder &builder, Location loc, Value operandVal,
                  ArrayRef<ReassociationIndices> reassociation,
                  ArrayRef<int64_t> expandShape) const {
    if (isa<RankedTensorType>(operandVal.getType())) {
      auto *expandedOperand =
          tensor::reshape_utils::createNewReshapingOp<tensor::ExpandShapeOp,
                                                      OpBuilder>(
              builder, loc, operandVal, reassociation, expandShape);
      return expandedOperand->getResult(0);
    }
    return operandVal;
  }

  /// Create an expanded reduce operation with new operands
  linalg::ReduceOp
  createExpandedReduceOp(OpBuilder &builder, linalg::ReduceOp originalOp,
                         const SmallVector<Value> &newOperands) const {
    Operation *clonedOp = builder.clone(*originalOp.getOperation());
    auto clonedReduceOp = cast<linalg::ReduceOp>(clonedOp);
    clonedOp->setOperands(newOperands);

    // Update result types to match init types
    auto dpsInitRes = clonedReduceOp.getDpsInits();
    for (unsigned resIdx = 0; resIdx < clonedReduceOp->getNumResults();
         ++resIdx) {
      auto oldResult = clonedReduceOp->getResult(resIdx);
      oldResult.setType(dpsInitRes[resIdx].getType());
    }

    return clonedReduceOp;
  }

  /// Replace uses with collapsed results
  void replaceWithCollapsedResults(
      OpBuilder &builder, linalg::ReduceOp originalOp,
      linalg::ReduceOp expandedOp,
      const SmallVector<ReassociationIndices> &reassociation) const {
    SmallVector<Value> collapsedResults;
    collapsedResults.reserve(expandedOp->getNumResults());

    for (unsigned resIdx = 0; resIdx < expandedOp->getNumResults(); ++resIdx) {
      Value newResult = expandedOp->getResult(resIdx);
      auto collapsedOp = builder.create<tensor::CollapseShapeOp>(
          expandedOp.getLoc(), originalOp->getResult(resIdx).getType(),
          newResult, reassociation);
      collapsedResults.push_back(collapsedOp.getResult());
    }

    for (unsigned resIdx = 0; resIdx < originalOp->getNumResults(); ++resIdx) {
      originalOp->getResult(resIdx).replaceAllUsesWith(
          collapsedResults[resIdx]);
    }
  }

  /// Compose multiple reduce operations into a single multi-reduce
  /// @param group Group of reduce operations to compose
  void composeReduceOps(SmallVector<linalg::ReduceOp> &group) const {
    OpBuilder builder(group[0]->getParentRegion());

    SmallVector<linalg::ReduceOp> adjustedGroup;
    if (aggressive) {
      adjustOperandsForAggressive(builder, group, adjustedGroup);
      std::swap(group, adjustedGroup);
    }

    ComposedReduceInfo info = collectComposedReduceInfo(group);
    linalg::ReduceOp newReduceOp =
        createComposedReduceOp(builder, group[0], info);
    mergeReduceRegions(builder, group, newReduceOp, info);
    replaceAndEraseOldOps(group, newReduceOp);
    ensureTopologicalOrder(newReduceOp);
  }

  /// Information for composing reduce operations
  struct ComposedReduceInfo {
    SmallVector<Value> allInputs;
    SmallVector<Value> allOutputs;
  };

  /// Collect inputs and outputs from all reduce operations
  ComposedReduceInfo
  collectComposedReduceInfo(const SmallVector<linalg::ReduceOp> &group) const {
    ComposedReduceInfo info;

    for (linalg::ReduceOp op : group) {
      info.allInputs.append(op.getDpsInputs());
      info.allOutputs.append(op.getDpsInits().begin(), op.getDpsInits().end());
    }

    return info;
  }

  /// Create a new composed reduce operation
  linalg::ReduceOp
  createComposedReduceOp(OpBuilder &builder, linalg::ReduceOp firstOp,
                         const ComposedReduceInfo &info) const {
    builder.setInsertionPointAfter(firstOp);
    auto newReduceOp = builder.create<linalg::ReduceOp>(
        firstOp->getLoc(), TypeRange(info.allOutputs), info.allInputs,
        info.allOutputs, firstOp.getDimensions());

    newReduceOp->setAttr(ReduceComposeAttr::name, builder.getUnitAttr());
    return newReduceOp;
  }

  /// Merge regions from multiple reduce operations into one
  void mergeReduceRegions(OpBuilder &builder,
                          const SmallVector<linalg::ReduceOp> &group,
                          linalg::ReduceOp newReduceOp,
                          const ComposedReduceInfo &info) const {
    Region &newRegion = newReduceOp.getRegion();
    Block *newBlock = createMergedBlock(builder, newRegion, newReduceOp, info);

    builder.setInsertionPointToStart(newBlock);
    SmallVector<Value> yieldOperands = mergeOperationsFromGroups(
        builder, group, newBlock, info.allInputs.size());

    builder.create<linalg::YieldOp>(newReduceOp.getLoc(), yieldOperands);
  }

  /// Create merged block with appropriate arguments
  Block *createMergedBlock(OpBuilder &builder, Region &newRegion,
                           linalg::ReduceOp newReduceOp,
                           const ComposedReduceInfo &info) const {
    Block *newBlock = builder.createBlock(&newRegion);

    // Add block arguments for inputs
    for (auto input : info.allInputs) {
      newBlock->addArgument(cast<ShapedType>(input.getType()).getElementType(),
                            newReduceOp.getLoc());
    }

    // Add block arguments for outputs
    for (auto output : info.allOutputs) {
      newBlock->addArgument(cast<ShapedType>(output.getType()).getElementType(),
                            newReduceOp.getLoc());
    }

    return newBlock;
  }

  /// Merge operations from all groups into the new block
  SmallVector<Value>
  mergeOperationsFromGroups(OpBuilder &builder,
                            const SmallVector<linalg::ReduceOp> &group,
                            Block *newBlock, int64_t allInputSize) const {
    SmallVector<Value> yieldOperands;

    for (size_t i = 0, globalIdx = 0; i < group.size(); ++i) {
      SmallVector<Value> groupYieldOperands = mergeSingleGroupOperations(
          builder, group[i], newBlock, globalIdx, allInputSize);
      yieldOperands.append(groupYieldOperands);
    }

    return yieldOperands;
  }

  /// Merge operations from a single reduce operation
  SmallVector<Value> mergeSingleGroupOperations(OpBuilder &builder,
                                                linalg::ReduceOp reduceOp,
                                                Block *newBlock,
                                                size_t &globalIdx,
                                                int64_t allInputSize) const {
    Block &oldBlock = reduceOp.getRegion().front();
    IRMapping mapping = createBlockArgumentMapping(
        oldBlock, newBlock, globalIdx, allInputSize, reduceOp);

    SmallVector<Value> yieldOperands;

    for (Operation &op : oldBlock.getOperations()) {
      Operation *clonedOp = builder.clone(op, mapping);
      if (isa<linalg::YieldOp>(op)) {
        yieldOperands.append(
            llvm::to_vector(ValueRange({clonedOp->getOperands()})));
        clonedOp->erase();
      }
    }

    return yieldOperands;
  }

  /// Create mapping between old and new block arguments
  IRMapping createBlockArgumentMapping(Block &oldBlock, Block *newBlock,
                                       size_t &globalIdx, int64_t allInputSize,
                                       linalg::ReduceOp reduceOp) const {
    IRMapping mapping;
    int64_t resultNum = reduceOp->getNumResults();

    assert(oldBlock.getNumArguments() == resultNum * 2);
    assert(reduceOp.getNumDpsInits() == reduceOp.getNumDpsInputs());
    assert(reduceOp.getNumDpsInits() == resultNum);

    for (int64_t j = 0; j < resultNum; ++j, ++globalIdx) {
      mapping.map(oldBlock.getArgument(j), newBlock->getArgument(globalIdx));
      mapping.map(oldBlock.getArgument(resultNum + j),
                  newBlock->getArgument(allInputSize + globalIdx));
    }

    return mapping;
  }

  /// Replace uses of old operations and erase them
  void replaceAndEraseOldOps(const SmallVector<linalg::ReduceOp> &group,
                             linalg::ReduceOp newReduceOp) const {
    int resultIdx = 0;
    for (auto reduceOp : group) {
      for (auto res : reduceOp->getResults()) {
        res.replaceAllUsesWith(newReduceOp.getResult(resultIdx));
        resultIdx++;
      }
      reduceOp.erase();
    }
  }

  /// Ensure operations in the block are in topological order
  void ensureTopologicalOrder(linalg::ReduceOp newReduceOp) const {
    auto *parentRegion = newReduceOp->getParentRegion();
    assert(parentRegion != nullptr && "Parent region doesn't exist");

    auto &block = parentRegion->getBlocks().front();
    if (!block.verifyOpOrder()) {
      sortTopologically(&block);
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hfusion::createComposeMultiReduce(
    const ComposeMultiReduceOptions &options) {
  return std::make_unique<ComposeMultiReducePass>(options);
}

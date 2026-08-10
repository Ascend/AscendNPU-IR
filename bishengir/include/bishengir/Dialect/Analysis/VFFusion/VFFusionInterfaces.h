//===- VFFusionInterfaces.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_H

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionAnalyzer.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionBlock.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionOutliner.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFStackInfo.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include <cstddef>
#include <cstdint>

namespace mlir::analysis {

/// Type alias for a collection of fusion blocks.
using VFFusionBlockList = SmallVector<VFFusionBlock>;

class FusionKindBase {
  struct FusionCandidate {
    VFFusionBlock block;
    Operation *insertionPoint = nullptr;
  };

public:
  virtual FailureOr<VFFusionBlockList> analyzeBlockImpl(Block &block) {
    llvm::report_fatal_error("analyze block is not implemented");
  }

  /// Fuses operations in a block by analyzing, outlining, and creating function
  /// calls.
  ///
  /// This method performs the following steps:
  /// - Analyzes the block to identify fusable operation groups (fusion blocks)
  /// - For each valid fusion block, outlines the operations into a separate
  /// func
  /// - Replaces the outlined operations with a call to the newly created func
  ///
  /// A fusion block is considered valid if:
  /// - It contains more than one operation (confirm this again)
  /// - It doesn't contain all operations in the block (trivial case)
  ///
  /// @param block The block containing operations to analyze and fuse
  /// @param builder The OpBuilder used to create new operations during
  /// outlining
  /// @return success() if all fusion blocks were successfully processed,
  ///         failure() if analysis failed or any outlining/invocation creation
  ///         failed
  LogicalResult fuse(Block &block, OpBuilder &builder) {
    FailureOr<VFFusionBlockList> maybeFusionBlocks = analyzeBlockImpl(block);
    if (failed(maybeFusionBlocks))
      return failure();
    VFFusionBlockList &fusionBlocks = maybeFusionBlocks.value();
    for (auto &fusionBlock : fusionBlocks) {
      if (fusionBlock.getOps().empty())
        continue;

      SmallVector<FusionCandidate> candidates;
      if (option.maxVFParams < 0 && !option.enableVFStackLimit) {
        if (shouldOutlineCandidate(fusionBlock))
          candidates.push_back(
              {fusionBlock, fusionBlock.getOps().back()->getNextNode()});
      } else {
        candidates = splitByMaxFuncParams(fusionBlock);
      }

      for (FusionCandidate &candidate : candidates) {
        func::FuncOp funcOp =
            block.getParent()->getParentOfType<func::FuncOp>();
        auto maybeFusedFunction =
            outliner.outline(funcOp, candidate.block, builder);
        if (failed(maybeFusedFunction))
          return failure();

        // outline() changes the builder position to the new function body.
        builder.setInsertionPoint(candidate.insertionPoint);
        if (failed(outliner.createInvoke(maybeFusedFunction.value(),
                                         candidate.block, builder)))
          return failure();
      }
    }
    return success();
  }

  explicit FusionKindBase(const VFFusionKindOption &option) : option(option) {}

  virtual ~FusionKindBase() = default;

protected:
  static int64_t getTotalParamRegisterCost(const SetVector<Value> &inputs) {
    // TODO: Can be refactored with VFStackInfoBuilder.
    // Rule in bisheng: max 58 16-bit registers are allowed for the func params
    // if the func param is a pointer to 32-bit dtype, the registore slot
    // accounts for 2 16-bit registers.
    int64_t totalCost = 0;
    for (Value input : inputs)
      totalCost += getParamRegisterCost(input);
    return totalCost;
  }

  SmallVector<FusionCandidate>
  splitByMaxFuncParams(VFFusionBlock &fusionBlock) {
    VFStackInfoBuilder stackInfoBuilder(option.enableVFStackLimit);
    SmallVector<FusionCandidate> candidates;
    // The candidate suffix starting here shares the next surviving boundary.
    size_t pendingBegin = 0;
    VFFusionBlock currentBlock;
    bool hasCurrentBlock = false;
    auto fits = [&](const VFFusionBlock &blk) {
      return getTotalParamRegisterCost(blk.getInputs()) <= option.maxVFParams &&
             stackInfoBuilder.fitsStack(blk.getOps());
    };

    for (Operation *op : fusionBlock.getOps()) {
      // Grow the current block in place -- no per-op block copy.
      currentBlock.fuseOp(op);
      if (fits(currentBlock)) {
        hasCurrentBlock = true;
        continue;
      }

      // Overflows: roll back it.
      currentBlock.unfuseOp(op);
      if (hasCurrentBlock)
        appendSplitBlock(std::move(currentBlock), candidates, pendingBegin);

      currentBlock = VFFusionBlock();
      currentBlock.fuseOp(op);
      if (fits(currentBlock)) {
        hasCurrentBlock = true;
      } else {
        currentBlock = VFFusionBlock();
        hasCurrentBlock = false;
        assignPendingInsertionPoint(candidates, pendingBegin, op);
      }
    }

    if (hasCurrentBlock)
      appendSplitBlock(std::move(currentBlock), candidates, pendingBegin);

    // Inserting before the next op is equivalent to inserting after the parent
    // tail and naturally preserves the order of candidates sharing this anchor.
    assignPendingInsertionPoint(candidates, pendingBegin,
                                fusionBlock.getOps().back()->getNextNode());
    return candidates;
  }

  static int64_t getParamRegisterCost(Value value) {
    // TODO: Can be refactored with VFStackInfoBuilder.
    Type type = value.getType();
    if (auto shapedType = dyn_cast<ShapedType>(type))
      type = shapedType.getElementType();

    unsigned bitWidth = 16;
    if (auto intType = dyn_cast<IntegerType>(type)) {
      bitWidth = intType.getWidth();
    } else if (auto floatType = dyn_cast<FloatType>(type)) {
      bitWidth = floatType.getWidth();
    } else if (isa<IndexType>(type)) {
      bitWidth = 64;
    }

    return (static_cast<int64_t>(bitWidth) + 15) / 16;
  }

  VFFusionOutliner outliner;
  VFFusionBlockList analyzedBlocks; // Renamed from fusedBlock
  const VFFusionKindOption option;

private:
  // Keep the original whole-block and single-compute-op outline filtering.
  bool shouldOutlineCandidate(const VFFusionBlock &candidateBlock) const {
    // Skip if the candidate wraps the entire function body.
    if (candidateBlock.getOps().size() ==
        candidateBlock.getOps().front()->getBlock()->getOperations().size())
      return false;
    // Filter out for ops which make sense for fusion
    SmallVector<Operation *> computeOps =
        getComputeOps(SmallVector<Operation *>(candidateBlock.getOps()));
    // When a fusion block contains at most one op, the normal fusion
    // path is bypassed (continue).  However, certain ops that are known
    // to be processed by a dedicated downstream pass (e.g. reduce-sum
    // ops handled by TreeReduceV2) must still be *outlined* into a
    // standalone vector function so that the downstream pass can
    // recognise and transform them.  shouldSkipFusion() gates this:
    //   - returns true  → outline this single-op block (skip fusion,
    //                     but keep the op isolated for later handling)
    //   - returns false → skip entirely (no outline, no fusion)
    if (computeOps.size() > 1)
      return true;
    return !computeOps.empty() &&
           shouldSkipFusion(computeOps.front(), option);
  }

  void appendSplitBlock(VFFusionBlock &&splitBlock,
                        SmallVectorImpl<FusionCandidate> &candidates,
                        size_t &pendingBegin) {
    if (!shouldOutlineCandidate(splitBlock)) {
      // The skipped block survives and anchors the preceding outline run.
      assignPendingInsertionPoint(candidates, pendingBegin,
                                  splitBlock.getOps().front());
      return;
    }

    candidates.push_back({std::move(splitBlock), nullptr});
  }

  static void assignPendingInsertionPoint(
      SmallVectorImpl<FusionCandidate> &candidates, size_t &pendingBegin,
      Operation *insertionPoint) {
    for (size_t index = pendingBegin; index < candidates.size(); ++index) {
      candidates[index].insertionPoint = insertionPoint;
    }
    pendingBegin = candidates.size();
  }

  static SmallVector<Operation *>
  getComputeOps(SmallVector<Operation *> allOps) {
    SmallVector<Operation *> res;
    llvm::copy_if(allOps, std::back_inserter(res),
                  [](auto *op) { return isComputeOp(op); });
    return res;
  }
};

class AllOpKind : public FusionKindBase {
public:
  FailureOr<VFFusionBlockList> analyzeBlockImpl(Block &block) override;

  explicit AllOpKind(const VFFusionKindOption &option)
      : FusionKindBase(option), analyzer(option) {};

private:
  AllOpKindAnalyzer analyzer;
};

class MaxParallelKind : public FusionKindBase {
public:
  FailureOr<VFFusionBlockList> analyzeBlockImpl(Block &block) override;

  explicit MaxParallelKind(const VFFusionKindOption &option)
      : FusionKindBase(option), analyzer(option) {};

private:
  MaxParallelAnalyzer analyzer;
};

class UBAwareOpKind : public FusionKindBase {
public:
  FailureOr<VFFusionBlockList> analyzeBlockImpl(Block &block) override;

  explicit UBAwareOpKind(const VFFusionKindOption &option)
      : FusionKindBase(option),
        analyzer(option, option.ubBudgetBytes, option.ubAlignBytes) {};

private:
  UBAwareOpKindAnalyzer analyzer;
};

} // namespace mlir::analysis
#endif

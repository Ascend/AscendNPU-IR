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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>

namespace mlir::analysis {

/// Type alias for a collection of fusion blocks.
using VFFusionBlockList = SmallVector<VFFusionBlock>;

class FusionKindBase {
public:
  virtual FailureOr<VFFusionBlockList> analyzeBlockImpl(Block &block) {
    llvm_unreachable("analyze block is not implemented");
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
  LogicalResult fuse(Block &block, OpBuilder &builder,
                     int64_t maxVFParams = -1) {
    FailureOr<VFFusionBlockList> maybeFusionBlocks = analyzeBlockImpl(block);
    if (failed(maybeFusionBlocks))
      return failure();
    VFFusionBlockList &fusionBlocks = maybeFusionBlocks.value();
    for (auto &fusionBlock : fusionBlocks) {
      SmallVector<VFFusionBlock> candidateFusionBlocks;
      if (maxVFParams < 0) {
        candidateFusionBlocks.push_back(fusionBlock);
      } else {
        candidateFusionBlocks = splitByMaxFuncParams(fusionBlock, maxVFParams);
      }

      for (VFFusionBlock &candidateBlock : candidateFusionBlocks) {
        if (candidateBlock.getOps().size() <= 1 ||
            candidateBlock.getOps().size() == block.getOperations().size())
          continue;

        func::FuncOp funcOp =
            block.getParent()->getParentOfType<func::FuncOp>();
        auto maybeFusedFunction =
            outliner.outline(funcOp, candidateBlock, builder);
        if (failed(maybeFusedFunction))
          return failure();
        auto maybeCallOp = outliner.createInvoke(maybeFusedFunction.value(),
                                                 candidateBlock, builder);
        if (failed(maybeCallOp))
          return failure();
      }
    }
    return success();
  }

  explicit FusionKindBase(const VFFusionKindOption &option) : option(option) {}

  virtual ~FusionKindBase() = default;

protected:
  static int64_t getTotalParamRegisterCost(const SetVector<Value> &inputs) {
    // Rule in bisheng: max 58 16-bit registers are allowed for the func params
    // if the func param is a pointer to 32-bit dtype, the registore slot
    // accounts for 2 16-bit registers.
    int64_t totalCost = 0;
    for (Value input : inputs)
      totalCost += getParamRegisterCost(input);
    return totalCost;
  }

  static SmallVector<VFFusionBlock>
  splitByMaxFuncParams(VFFusionBlock &fusionBlock, int64_t maxVFParams) {
    SmallVector<VFFusionBlock> splitBlocks;
    VFFusionBlock currentBlock;
    bool hasCurrentBlock = false;

    for (Operation *op : fusionBlock.getOps()) {
      VFFusionBlock tentativeBlock = currentBlock;
      tentativeBlock.fuseOp(op);

      if (getTotalParamRegisterCost(tentativeBlock.recomputeInputs()) <=
          maxVFParams) {
        currentBlock = std::move(tentativeBlock);
        hasCurrentBlock = true;
        continue;
      }

      if (hasCurrentBlock)
        splitBlocks.push_back(currentBlock);

      VFFusionBlock singletonBlock;
      singletonBlock.fuseOp(op);
      if (getTotalParamRegisterCost(singletonBlock.recomputeInputs()) <=
          maxVFParams) {
        currentBlock = std::move(singletonBlock);
        hasCurrentBlock = true;
      } else {
        currentBlock = VFFusionBlock();
        hasCurrentBlock = false;
      }
    }

    if (hasCurrentBlock)
      splitBlocks.push_back(currentBlock);

    return splitBlocks;
  }

  static int64_t getParamRegisterCost(Value value) {
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
};

class AllOpKind : public FusionKindBase {
public:
  FailureOr<VFFusionBlockList> analyzeBlockImpl(Block &block) override;

  explicit AllOpKind(const VFFusionKindOption &option)
      : FusionKindBase(option), analyzer(option) {};

private:
  AllOpKindAnalyzer analyzer;
};

class NMostOpKind : public FusionKindBase {
public:
  FailureOr<VFFusionBlockList> analyzeBlockImpl(Block &block) override;

  explicit NMostOpKind(const VFFusionKindOption &option)
      : FusionKindBase(option), analyzer(option, N) {};

private:
  const size_t N = 8;
  NMostOpKindAnalyzer analyzer;
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

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
#include "mlir/Support/LogicalResult.h"

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
  /// - For each valid fusion block, outlines the operations into a separate func
  /// - Replaces the outlined operations with a call to the newly created func
  ///
  /// A fusion block is considered valid if:
  /// - It contains more than one operation (confirm this again)
  /// - It doesn't contain all operations in the block (trivial case)
  ///
  /// @param block The block containing operations to analyze and fuse
  /// @param builder The OpBuilder used to create new operations during outlining
  /// @return success() if all fusion blocks were successfully processed,
  ///         failure() if analysis failed or any outlining/invocation creation failed
  LogicalResult fuse(Block &block, OpBuilder &builder) {
    FailureOr<VFFusionBlockList> maybeFusionBlocks = analyzeBlockImpl(block);
    if (failed(maybeFusionBlocks))
      return failure();
    VFFusionBlockList &fusionBlocks = maybeFusionBlocks.value();
    for (auto &fusionBlock : fusionBlocks) {
      if (fusionBlock.getOps().size() <= 1 ||
          fusionBlock.getOps().size() == block.getOperations().size())
        continue;
      func::FuncOp funcOp = block.getParent()->getParentOfType<func::FuncOp>();
      auto maybeFusedFunction = outliner.outline(funcOp, fusionBlock, builder);
      if (failed(maybeFusedFunction))
        return failure();
      auto maybeCallOp = outliner.createInvoke(maybeFusedFunction.value(),
                                               fusionBlock, builder);
      if (failed(maybeCallOp))
        return failure();
    }
    return success();
  }

  explicit FusionKindBase(const VFFusionKindOption &option) : option(option) {
  }

  virtual ~FusionKindBase() = default;

protected:
  VFFusionOutliner outliner;
  VFFusionBlockList analyzedBlocks;  // Renamed from fusedBlock
  const VFFusionKindOption option;
};

class AllOpKind : public FusionKindBase {
public:
  FailureOr<VFFusionBlockList> analyzeBlockImpl(Block &block) override;

  explicit AllOpKind(const VFFusionKindOption &option)
    : FusionKindBase(option), analyzer(option) {
  };

private:
  AllOpKindAnalyzer analyzer;
};

class NMostOpKind : public FusionKindBase {
public:
  FailureOr<VFFusionBlockList> analyzeBlockImpl(Block &block) override;

  explicit NMostOpKind(const VFFusionKindOption &option)
      : FusionKindBase(option), analyzer(option, N){};
private:
  const size_t N = 8;
  NMostOpKindAnalyzer analyzer;
};

} // namespace mlir::analysis
#endif
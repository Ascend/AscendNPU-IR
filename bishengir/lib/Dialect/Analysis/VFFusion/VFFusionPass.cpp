//===- VFFusionPass.cpp --------- VF Fusion Pass --------------------------===//
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

#include "bishengir/Dialect/Analysis/VFFusion/Passes.h"
#include "bishengir/Dialect/Analysis/VFFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include <string>

#define DEBUG_TYPE "vf-fusion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {
#define GEN_PASS_DEF_VFFUSION
#include "bishengir/Dialect/Analysis/VFFusion/Passes.h.inc"
} // namespace mlir::analysis

using namespace mlir;
using namespace mlir::impl;

namespace mlir {
namespace analysis {
class VFFusionPass : public impl::VFFusionBase<VFFusionPass> {
  template <typename FusionKind>
  LogicalResult tryToFuse(Operation *op, OpBuilder &builder) const;

  VFFusionKindOption getFusionOption() const;

public:
  explicit VFFusionPass(const mlir::VFFusionOptions &options)
      : impl::VFFusionBase<VFFusionPass>(options) {}
  void runOnOperation() override;
  LogicalResult preProcess();

private:
  int64_t ubBudgetBytes_ = 0;
  int64_t ubAlignBytes_ = 0;
};

LogicalResult VFFusionPass::preProcess() {
  ModuleOp moduleOp = getOperation();
  RewritePatternSet patterns(&getContext());
  populateEmptifyReduceInitPatterns(patterns);
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    return moduleOp.emitError("fail to preprocess");
  }
  return success();
}

VFFusionKindOption VFFusionPass::getFusionOption() const {
  return VFFusionKindOption(enableOutlineCF, enableOutlineMemref,
                            enableOutlineArith, enableOutlineCube,
                            ubBudgetBytes_, ubAlignBytes_);
}

template <typename FusionKind>
LogicalResult VFFusionPass::tryToFuse(Operation *op, OpBuilder &builder) const {
  for (auto &region : op->getRegions()) {
    // if disabled, need to traverse the all operations inside operation's
    // regions.
    if (!enableOutlineCF) {
      for (auto &block : region.getBlocks()) {
        for (Operation &opBlock : block.getOperations()) {
          if (!opBlock.hasTrait<RegionBranchOpInterface::Trait>())
            continue;
          if (failed(tryToFuse<FusionKind>(&opBlock, builder)))
            return failure();
        }
      }
    }

    // only consider the outter most operations.
    for (auto &block : region.getBlocks()) {
      std::unique_ptr<FusionKindBase> fuser =
          std::make_unique<FusionKind>(getFusionOption());
      if (failed(fuser->fuse(block, builder)))
        return failure();
    }
  }
  return success();
}

static bool isCVCases(ModuleOp moduleOp) {
  auto result = moduleOp.walk([](Operation *op) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (funcOp->hasAttr(hivm::TPartOfMixAttr::name))
        return WalkResult::interrupt();
    }
    if (isa<scope::ScopeOp>(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

void VFFusionPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  RewritePatternSet patterns(&getContext());
  OpBuilder builder(moduleOp.getContext());
  OpBuilder::InsertionGuard insGuard(builder);

  if (enableOutlineCF)
    llvm_unreachable("unsupported at the moment");

  // for CV cases, temporarily bypass vffusion
  if (isCVCases(moduleOp))
    return;

  if (failed(preProcess())) {
    signalPassFailure();
    return;
  }

  ubBudgetBytes_ = 0;
  ubAlignBytes_ = 0;
  if (fusionMode == FusionMode::UBAwareOp) {
    // UB-aware mode requires outlining memref operands so that
    // materialize_in_destination stores are included in VF groups.
    // Without this, intermediate buffers persist in caller UB across VF
    // boundaries, causing false UB overflow in PlanMemory.
    enableOutlineMemref = true;

    if (auto spec = hacc::utils::getNPUTargetSpec(moduleOp)) {
      auto ubEntry = spec->getSpecForIdentifierEnum(hacc::DeviceSpec::UB_SIZE);
      ubBudgetBytes_ = cast<IntegerAttr>(ubEntry.getValue()).getInt() / 8;
      LDBG("UB budget from target spec: " << ubBudgetBytes_ << " bytes");

      auto alignEntry =
          spec->getSpecForIdentifierEnum(hacc::DeviceSpec::UB_ALIGN_SIZE);
      ubAlignBytes_ = cast<IntegerAttr>(alignEntry.getValue()).getInt() / 8;
      LDBG("UB align from target spec: " << ubAlignBytes_ << " bytes");
    }
  }

  // clone multi-use extract_slice to ensure each can be independently fused
  // into its uses.
  SmallVector<tensor::ExtractSliceOp> sliceOps;
  moduleOp.walk([&](tensor::ExtractSliceOp sliceOp) {
    if (sliceOp->use_empty() || sliceOp->hasOneUse())
      return;
      
    sliceOps.push_back(sliceOp);
  });

  for (tensor::ExtractSliceOp sliceOp : sliceOps) {
    // collect all uses
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : sliceOp->getUses()) {
      uses.push_back(&use);
    }

    // keep first use, clone others
    builder.setInsertionPointAfter(sliceOp);
    for (size_t i = 1; i < uses.size(); ++i) {
      Operation *clonedOp = builder.clone(*sliceOp);
      uses[i]->set(clonedOp->getResult(0));
    }
  }

  auto walkResult = moduleOp.walk([&](func::FuncOp funcOp) -> WalkResult {
    // Cube/MixCV function requires special fusion strategy (refer to
    // SplitMixKernel).
    // Currectly, only support VFFusion for AIV kernel.
    if (!enableOutlineCube && isCubeFunc(funcOp)) {
      return WalkResult::advance();
    }

    switch (fusionMode) {
    case FusionMode::AllOp:
      return WalkResult(
          this->tryToFuse<AllOpKind>(funcOp.getOperation(), builder));
    case FusionMode::NMostOp:
      return WalkResult(
          this->tryToFuse<NMostOpKind>(funcOp.getOperation(), builder));
    case FusionMode::MaxParallel:
      return WalkResult(
          this->tryToFuse<MaxParallelKind>(funcOp.getOperation(), builder));
    case FusionMode::UBAwareOp:
      return WalkResult(
          this->tryToFuse<UBAwareOpKind>(funcOp.getOperation(), builder));
    }
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    signalPassFailure();
}

std::unique_ptr<Pass> createVFFusionPass(const VFFusionOptions &option) {
  return std::make_unique<VFFusionPass>(option);
}

} // namespace analysis
} // namespace mlir
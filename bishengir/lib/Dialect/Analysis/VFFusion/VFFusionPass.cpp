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
#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
                            ubBudgetBytes_, ubAlignBytes_, enableRA, enableAR,
                            maxVFParams, enableVFStackLimit, enableCastOpt,
                            enableNewTreeReducePolicy);
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

/// TODO: Detect CV-specific patterns that require skipping VFFusion.
static bool hasCVSpecialPatternsSkip(ModuleOp moduleOp) {
  // Pattern 1: linalg.fill → linalg.mul → linalg.fill → linalg.add → linalg.exp
  // (DPS chain, all ops writing to the same output tensor).
  // VFFusion outlines this chain into a fused function, but the
  // function boundary changes the sync structure and downstream
  // extract_slice/insert_slice handling, causing scheduling gaps.
  bool hasFillMulFillAddExpPattern = false;
  moduleOp.walk([&](linalg::ExpOp expOp) {
    if (hasFillMulFillAddExpPattern)
      return;
    // exp's input should come from linalg.add
    Operation *addOp = expOp.getOperand(0).getDefiningOp();
    if (!addOp || !isa<linalg::AddOp>(addOp))
      return;
    // add's inputs: one from linalg.mul, one from linalg.fill
    Operation *lhs = addOp->getOperand(0).getDefiningOp();
    Operation *rhs = addOp->getOperand(1).getDefiningOp();
    Operation *mulOp = nullptr;
    if (isa<linalg::MulOp>(lhs) && isa<linalg::FillOp>(rhs)) {
      mulOp = lhs;
    } else if (isa<linalg::FillOp>(lhs) && isa<linalg::MulOp>(rhs)) {
      mulOp = rhs;
    } else {
      return;
    }
    // mul's input should include a linalg.fill
    for (Value operand : mulOp->getOperands()) {
      if (Operation *op = operand.getDefiningOp()) {
        if (isa<linalg::FillOp>(op)) {
          hasFillMulFillAddExpPattern = true;
          return;
        }
      }
    }
  });
  if (hasFillMulFillAddExpPattern) {
    LDBG("Skipping VFFusion: detected fill→mul→fill→add→exp "
         "pattern in CV case");
    return true;
  }

  // Pattern 2: hfusion.cast feeds two linalg.broadcast ops (one directly,
  // one through a tensor.extract_slice of the cast result); both broadcasts
  // feed linalg.sub → linalg.exp.
  // VFFusion outlining this chain breaks the shared cast + slice
  // relationship (the slice has a dynamic index and carries a
  // to_be_bubbled_slice hint), causing scheduling gaps downstream.
  bool hasCastBroadcastSubExpPattern = false;
  moduleOp.walk([&](linalg::ExpOp expOp) {
    if (hasCastBroadcastSubExpPattern)
      return;
    // exp's input should come from linalg.sub
    Operation *subOp = expOp.getOperand(0).getDefiningOp();
    if (!subOp || !isa<linalg::SubOp>(subOp))
      return;
    // sub's inputs: both should be linalg.broadcast
    Operation *lhsBrc = subOp->getOperand(0).getDefiningOp();
    Operation *rhsBrc = subOp->getOperand(1).getDefiningOp();
    if (!lhsBrc || !isa<linalg::BroadcastOp>(lhsBrc) || !rhsBrc ||
        !isa<linalg::BroadcastOp>(rhsBrc))
      return;
    // Resolve the defining op behind each broadcast's input:
    //  - one should be a tensor.extract_slice whose source is a hfusion.cast
    //  - the other should be the hfusion.cast directly
    Operation *lhsSrc = lhsBrc->getOperand(0).getDefiningOp();
    Operation *rhsSrc = rhsBrc->getOperand(0).getDefiningOp();
    if (!lhsSrc || !rhsSrc)
      return;
    Operation *lhsCast = nullptr;
    Operation *rhsCast = nullptr;
    bool lhsViaSlice = false;
    bool rhsViaSlice = false;
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(lhsSrc)) {
      lhsCast = sliceOp.getSource().getDefiningOp();
      lhsViaSlice = true;
    } else {
      lhsCast = lhsSrc;
    }
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(rhsSrc)) {
      rhsCast = sliceOp.getSource().getDefiningOp();
      rhsViaSlice = true;
    } else {
      rhsCast = rhsSrc;
    }
    if (!lhsCast || !rhsCast)
      return;
    // Both sources must resolve to the same hfusion.cast
    if (!isa<hfusion::CastOp>(lhsCast) || lhsCast != rhsCast)
      return;
    // One broadcast reaches the cast via extract_slice, the other directly
    if (lhsViaSlice == rhsViaSlice)
      return;
    hasCastBroadcastSubExpPattern = true;
  });
  if (hasCastBroadcastSubExpPattern) {
    LDBG("Skipping VFFusion: detected cast→broadcast(×2)→sub→exp "
         "pattern in CV case");
    return true;
  }

  return false;
}

void VFFusionPass::runOnOperation() {
  // TODO: dirty hack to make behaviour of AllOp same as disabled vf-fusion
  if (fusionMode == FusionMode::AllOp) {
    return;
  }
  ModuleOp moduleOp = getOperation();
  RewritePatternSet patterns(&getContext());
  OpBuilder builder(moduleOp.getContext());
  OpBuilder::InsertionGuard insGuard(builder);

  if (enableOutlineCF)
    llvm::report_fatal_error("unsupported at the moment");

  auto freezeRegisterTreeSelection = [&]() {
    if (!enableNewTreeReducePolicy || !enableRA)
      return;
    moduleOp->removeAttr(hfusion::kTreeReductionSelectionFrozenAttr);
    moduleOp->removeAttr(hfusion::kRegularTreeReductionScopeAttr);
    moduleOp->removeAttr(hfusion::kLegacyTreeReductionScopeAttr);
    SmallVector<Operation *> registerCandidates;
    moduleOp.walk([&](Operation *op) {
      op->removeAttr(hfusion::kRegisterTreeReductionSelectedAttr);
      op->removeAttr(hfusion::kRegularTreeReductionSelectedAttr);
      if (hfusion::isRegisterTreeReductionCandidate(op))
        registerCandidates.push_back(op);
    });
    bool selectedAllRegisterTrees = true;
    bool selectedAnyLegacyTree = false;
    for (Operation *op : registerCandidates) {
      bool selected = hfusion::shouldUseRegisterTreeReduction(op);
      selectedAllRegisterTrees &= selected;
      selectedAnyLegacyTree |= hfusion::shouldUseLegacyTreeReductionScope(op);
      llvm::StringLiteral selectedAttr =
          selected ? hfusion::kRegisterTreeReductionSelectedAttr
                   : hfusion::kRegularTreeReductionSelectedAttr;
      op->setAttr(selectedAttr, UnitAttr::get(&getContext()));
    }
    if (selectedAnyLegacyTree)
      moduleOp->setAttr(hfusion::kLegacyTreeReductionScopeAttr,
                        UnitAttr::get(&getContext()));
    else if (!registerCandidates.empty() && !selectedAllRegisterTrees)
      moduleOp->setAttr(hfusion::kRegularTreeReductionScopeAttr,
                        UnitAttr::get(&getContext()));
    // Later nested function passes may run in parallel and may also create
    // canonical linalg ops.  From this point on, consumers must use only the
    // per-op decisions above instead of walking a module which is being
    // rewritten concurrently.
    moduleOp->setAttr(hfusion::kTreeReductionSelectionFrozenAttr,
                      UnitAttr::get(&getContext()));
  };

  // For CV cases, bypass vffusion entirely when any op should skip fusion
  // (e.g. RA/AR sum-reductions handled by dedicated downstream passes).
  if (isCVCases(moduleOp)) {
    VFFusionKindOption option = getFusionOption();
    if (moduleOp
            .walk([&](Operation *op) -> WalkResult {
              return shouldSkipFusion(op, option) ? WalkResult::interrupt()
                                                  : WalkResult::advance();
            })
            .wasInterrupted()) {
      freezeRegisterTreeSelection();
      return;
    }
    if (hasCVSpecialPatternsSkip(moduleOp)) {
      freezeRegisterTreeSelection();
      return;
    }
  }

  if (failed(preProcess())) {
    signalPassFailure();
    return;
  }

  // Freeze the register-tree cost decision after preprocessing but before
  // fusion starts mutating the graph.  Linalg attributes are preserved by
  // VFFusion's outlining clones, so AutoVectorizeV2 observes the same
  // decision even when several reductions are split into separate private
  // functions.
  freezeRegisterTreeSelection();

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

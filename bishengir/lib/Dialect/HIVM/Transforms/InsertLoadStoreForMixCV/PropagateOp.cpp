//===- PropagateOp.cpp - Propagate pattern of InsertLoadStoreForMixCV -----===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/PropagateOp.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "insert-load-store-propagate-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {
namespace {

static bool checkPropagate(PropagationStep step,
                           UnrealizedConversionCastOp propagateOp) {
  auto addressSpaces = PropagatorUtil::getAddressSpace(propagateOp);
  switch (step) {
  case PropagationStep::L0C:
    return llvm::find(addressSpaces, hivm::AddressSpace::L0C) !=
           addressSpaces.end();
  case PropagationStep::LOCAL:
    return PropagatorUtil::getCoreType(propagateOp) ==
           TCoreType::CUBE_AND_VECTOR;
  case PropagationStep::GM:
    return llvm::find(addressSpaces, hivm::AddressSpace::GM) !=
           addressSpaces.end();
  case PropagationStep::UB:
    return PropagatorUtil::getCoreType(propagateOp) == TCoreType::VECTOR;
  case PropagationStep::L1:
    return PropagatorUtil::getCoreType(propagateOp) == TCoreType::CUBE;
  default:
    return true;
  }
}

/// True when `result` of `branch` must stay in L0C, so propagate-up must not
/// cross the region boundary for that result.
///
/// `normalized_in_L0C` on a RegionBranch is an ArrayAttr of result indices
/// (set by NormalizeMatmul). Results whose indices are not listed may still
/// be propagated. A UnitAttr form, or `hivm.remain_in_l0c`, applies to every
/// result of the op.
static bool isRegionResultRequiredInL0C(RegionBranchOpInterface branch,
                                        OpResult result) {
  if (branch->hasAttr(RemainInL0CAttr::name))
    return true;

  Attribute attr = branch->getAttr(kNormalizedInL0CAttr);
  if (!attr)
    return false;

  auto arrayAttr = dyn_cast<ArrayAttr>(attr);
  if (!arrayAttr)
    return true;

  uint64_t idx = result.getResultNumber();
  return llvm::any_of(arrayAttr, [idx](Attribute element) {
    auto intAttr = dyn_cast<IntegerAttr>(element);
    return intAttr && intAttr.getValue().getZExtValue() == idx;
  });
}

/// Mirror propagate markers across the RegionBranch connected component of
/// `seed`. While before/after channels stay separate because they share no
/// edges. Returns failure if `seed` is not on any forwarded edge (e.g. for IV).
static LogicalResult
propagateAlongRegionEdges(RegionBranchOpInterface branch, Value seed,
                          UnrealizedConversionCastOp propagateOp,
                          PatternRewriter &rewriter) {
  auto maybeSites = PropagatorUtil::collectRelatedPropagatorSites(branch, seed);
  if (failed(maybeSites))
    return failure();

  for (auto *opr : maybeSites->getUpSites())
    PropagatorUtil::createPropagatorUp(opr, propagateOp, rewriter);
  for (auto arg : maybeSites->getDownSites())
    PropagatorUtil::createPropagatorDown(arg, propagateOp, rewriter);
  LDBG("Propagated along RegionBranch edges from seed in "
       << branch << " (up=" << maybeSites->getUpSites().size()
       << ", down=" << maybeSites->getDownSites().size() << ")");
  return success();
}

struct Candidate {
  UnrealizedConversionCastOp propagator;
  size_t count = 0;
};

} // namespace

LogicalResult
ControlFlowPropagatePattern::matchAndRewrite(RegionBranchOpInterface branch,
                                             PatternRewriter &rewriter) const {
  if (step == PropagationStep::LOCAL || step == PropagationStep::ALL)
    return failure();

  bool changed = false;
  for (const PropagatorUtil::PropagatorSiteSet &sites :
       PropagatorUtil::collectIndependentPropagatorSiteGroups(branch)) {

    SmallVector<Candidate> candidates;
    size_t shapedSiteCount = 0;

    auto countPropagator = [&](UnrealizedConversionCastOp propagator) {
      if (!propagator || !checkPropagate(step, propagator))
        return;
      auto *candidate = llvm::find_if(candidates, [&](const Candidate &other) {
        return PropagatorUtil::haveSamePropagation(propagator,
                                                   other.propagator);
      });
      if (candidate == candidates.end())
        candidates.push_back({propagator, 1});
      else
        ++candidate->count;
    };

    for (OpOperand *operand : sites.getUpSites()) {
      if (!isa<ShapedType>(operand->get().getType()))
        continue;
      ++shapedSiteCount;
      countPropagator(PropagatorUtil::getUpSiteRequirement(operand));
    }
    for (Value value : sites.getDownSites()) {
      if (!isa<ShapedType>(value.getType()) || value.use_empty())
        continue;
      ++shapedSiteCount;
      countPropagator(PropagatorUtil::getDownSiteRequirement(value));
    }
    if (shapedSiteCount == 0 || candidates.empty())
      continue;

    Candidate *majority = nullptr;
    bool tied = false;
    for (Candidate &candidate : candidates) {
      if (!majority || candidate.count > majority->count) {
        majority = &candidate;
        tied = false;
      } else if (candidate.count == majority->count) {
        tied = true;
      }
    }
    if (tied || majority->count * 2 < shapedSiteCount ||
        majority->count == shapedSiteCount)
      continue;

    bool groupChanged = false;
    for (OpOperand *operand : sites.getUpSites()) {
      if (!isa<ShapedType>(operand->get().getType()))
        continue;
      auto existing = PropagatorUtil::getUpPropagator(operand);
      if (existing &&
          PropagatorUtil::haveSamePropagation(existing, majority->propagator))
        continue;
      PropagatorUtil::createPropagatorUp(operand, majority->propagator,
                                         rewriter);
      groupChanged = true;
    }
    for (Value value : sites.getDownSites()) {
      if (!isa<ShapedType>(value.getType()) || value.use_empty())
        continue;
      auto existing = PropagatorUtil::getDownPropagator(value);
      if (existing &&
          PropagatorUtil::haveSamePropagation(existing, majority->propagator))
        continue;
      PropagatorUtil::createPropagatorDown(value, majority->propagator,
                                           rewriter);
      groupChanged = true;
    }
    if (!groupChanged)
      continue;
    changed = true;
    LDBG("Propagated majority requirement across RegionBranch group in "
         << branch << " (count=" << majority->count
         << ", sites=" << shapedSiteCount << ")");
  }
  return success(changed);
}

LogicalResult PropagateDownPattern::propagateDownDmaOp(
    hivm::HIVMStructuredOp op, OpOperand &operand,
    UnrealizedConversionCastOp propagateOp, PatternRewriter &rewriter) const {
  // Same boundary rule as propagateDownForCustomLikeOp; DMA updates all inits
  // when any init operand is reached.
  LDBG("Propagating dma down: " << op << "\n" << propagateOp);
  for (auto *input : op.getDpsInputOperands()) {
    if (input->get() == operand.get()) {
      LDBG("Operand: " << input->get());
      PropagatorUtil::createPropagatorUp(input, propagateOp, rewriter);
      return success();
    }
  }

  bool isInit = false;
  for (auto init : op.getDpsInits()) {
    if (init == operand.get()) {
      isInit = true;
      break;
    }
  }
  if (isInit) {
    for (auto &init : op.getDpsInitsMutable())
      PropagatorUtil::createPropagatorUp(&init, propagateOp, rewriter);
    PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
    return success();
  }
  return failure();
}

LogicalResult
PropagateUpPattern::propagateUpDmaOp(hivm::HIVMStructuredOp op, OpResult res,
                                     UnrealizedConversionCastOp propagateOp,
                                     PatternRewriter &rewriter) const {
  // Same boundary rule as propagateUpForCustomLikeOp.
  LDBG("Propagating dma up: " << op << "\n" << propagateOp);
  PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
  for (auto &init : op.getDpsInitsMutable()) {
    PropagatorUtil::createPropagatorUp(&init, propagateOp, rewriter);
  }
  return success();
}

LogicalResult
PropagateDownPattern::matchAndRewrite(UnrealizedConversionCastOp propagateOp,
                                      PatternRewriter &rewriter) const {
  if (!propagateOp->hasAttr(kPropagateDownAttr))
    return failure();
  if (propagateOp->getResult(0).getType().isIntOrIndexOrFloat())
    return failure();
  if (!checkPropagate(step, propagateOp))
    return failure();
  SmallVector<OpOperand *> uses;
  for (auto &use : propagateOp->getUses())
    uses.push_back(&use);
  LogicalResult result = failure();
  for (auto *use : uses) {
    auto *user = use->getOwner();
    if (user->hasAttr(kPropagateUpAttr))
      continue;
    auto newRes =
        TypeSwitch<Operation *, LogicalResult>(user)
            .Case([&](RegionBranchOpInterface branch) {
              if (step != PropagationStep::ALL)
                return failure();
              return propagateAlongRegionEdges(branch, use->get(), propagateOp,
                                               rewriter);
            })
            .Case([&](RegionBranchTerminatorOpInterface terminator) {
              Operation *parent = terminator->getParentOp();
              if (step != PropagationStep::ALL)
                return failure();
              // Condition predicate is not a forwarded successor operand.
              if (isa<scf::ConditionOp>(terminator.getOperation()) &&
                  use->getOperandNumber() == 0)
                return failure();
              auto branch = dyn_cast<RegionBranchOpInterface>(parent);
              if (!branch)
                return failure();
              return propagateAlongRegionEdges(branch, use->get(), propagateOp,
                                               rewriter);
            })
            .Case<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.cpp.inc"
                >([&](auto op) {
              if (step == PropagationStep::LOCAL)
                return failure();
              auto dmaOp = dyn_cast<hivm::HIVMStructuredOp>(op.getOperation());
              if (!dmaOp)
                return failure();
              return propagateDownDmaOp(dmaOp, *use, propagateOp, rewriter);
            })
            .Case<hivm::CustomMacroOp, hivm::CustomOp>([&](Operation *op) {
              return PropagatorUtil::propagateDownForCustomLikeOp(
                  op, use, propagateOp, rewriter);
            })
            .Default([&](Operation *op) {
              PropagatorUtil::createPropagatorsUp(op, propagateOp, rewriter);
              PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
              return success();
            });
    if (succeeded(newRes))
      result = newRes;
  }
  return result;
}

LogicalResult
PropagateUpPattern::matchAndRewrite(UnrealizedConversionCastOp propagateOp,
                                    PatternRewriter &rewriter) const {
  if (!propagateOp->hasAttr(kPropagateUpAttr))
    return failure();
  auto input = propagateOp.getInputs()[0];
  auto res = dyn_cast<OpResult>(input);
  if (input.getType().isIntOrIndexOrFloat())
    return failure();
  if (!checkPropagate(step, propagateOp))
    return failure();
  if (!res) {
    if (step == PropagationStep::LOCAL)
      return failure();
    auto blockArgument = cast<BlockArgument>(input);
    Operation *parentOp = blockArgument.getOwner()->getParentOp();
    if (auto branch = dyn_cast<RegionBranchOpInterface>(parentOp)) {
      if (step != PropagationStep::ALL)
        return failure();
      auto maybeSites =
          PropagatorUtil::collectRelatedPropagatorSites(branch, blockArgument);
      if (failed(maybeSites)) {
        // Non-forwarded block args (e.g. scf.for induction var).
        PropagatorUtil::createPropagatorDown(blockArgument, propagateOp,
                                             rewriter);
        return success();
      }
      for (auto *opr : maybeSites->getUpSites())
        PropagatorUtil::createPropagatorUp(opr, propagateOp, rewriter);
      for (auto arg : maybeSites->getDownSites())
        PropagatorUtil::createPropagatorDown(arg, propagateOp, rewriter);
      return success();
    }
    PropagatorUtil::createPropagatorDown(blockArgument, propagateOp, rewriter);
    return success();
  }
  auto *defOp = res.getDefiningOp();
  if (!defOp || defOp->hasAttr(kPropagateDownAttr))
    return failure();
  return TypeSwitch<Operation *, LogicalResult>(defOp)
      .Case([&](RegionBranchOpInterface branch) {
        if (step != PropagationStep::ALL)
          return failure();

        if (isRegionResultRequiredInL0C(branch, res))
          return failure();
        // Unstructured load case should be propagated from the inside.
        if (auto forOp = dyn_cast<scf::ForOp>(branch.getOperation())) {
          if (forOp->hasAttr(ExtractLoadStoreAttr) &&
              !forOp->getParentOp()->hasAttr(ExtractLoadStoreAttr))
            return failure();
        }
        return propagateAlongRegionEdges(branch, res, propagateOp, rewriter);
      })
      .Case<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.cpp.inc"
          >([&](Operation *op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        auto dmaOp = cast<hivm::HIVMStructuredOp>(op);
        return propagateUpDmaOp(dmaOp, res, propagateOp, rewriter);
      })
      .Case([&](tensor::InsertSliceOp op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        // TODO: refactor this propagation logic for A5
        if (isRegBaseTarget) {
          PropagatorUtil::createPropagatorsUp(op, propagateOp, rewriter);
          PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
          return success();
        }

        if (PropagatorUtil::getCoreType(propagateOp) == TCoreType::CUBE) {
          PropagatorUtil::createPropagatorsUp(op, TCoreType::VECTOR,
                                              hivm::AddressSpace::UB, rewriter);
          PropagatorUtil::createPropagatorsDown(
              op, TCoreType::VECTOR, hivm::AddressSpace::UB, rewriter);
        } else {
          PropagatorUtil::createPropagatorsUp(op, propagateOp, rewriter);
          PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
        }
        return success();
      })
      .Case<hivm::CustomMacroOp, hivm::CustomOp>([&](Operation *op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        return PropagatorUtil::propagateUpForCustomLikeOp(op, propagateOp,
                                                          rewriter);
      })
      .Case([&](tensor::EmptyOp emptyOp) { return failure(); })
      .Default([&](Operation *op) {
        if (step == PropagationStep::LOCAL &&
            !llvm::all_of(op->getResultTypes(),
                          [](auto type) { return isa<ShapedType>(type); }))
          return failure();
        PropagatorUtil::createPropagatorsUp(op, propagateOp, rewriter);
        PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
        return success();
      });
}

} // namespace mlir::hivm

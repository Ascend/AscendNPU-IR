//===- CreatePreload.cpp - Create preload for CV pipelining --------------===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/ShapeRegistry.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CREATEPRELOAD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "hivm-create-preload"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

struct PreloadInfo {
  // Preload info
  size_t preloadNum;
  size_t maxPreloadNum;
  bool elideInactiveTail = false;
  Value maxPreloadValue;
  // Loop info
  Value indVar;
  Value lb;
  Value ub;
  Value step;
  // Kernel info
  TFuncCoreType funcCoreType;
  // Planned pointer roots for preload-local buffers and their physical buffer
  // counts. The annotation may be attached to a view of the planned root.
  DenseMap<Value, size_t> preloadLocalBuffers;
  // IRMapping for rewriting
  SmallVector<IRMapping> mappings;
};

struct CreatePreloadPass : public impl::CreatePreloadBase<CreatePreloadPass> {
  using Base = impl::CreatePreloadBase<CreatePreloadPass>;
  using Base::Base;
  void runOnOperation() override;
};

static std::optional<hivm::PointerCastOp> getLocalBuffer(Value v) {
  if (auto pointerCastOp = v.getDefiningOp<hivm::PointerCastOp>()) {
    if (utils::getAnnotateOpWithAttr(v, hivm::PreloadLocalBufferAttr::name)
            .has_value())
      return pointerCastOp;
    return std::nullopt;
  }
  if (auto scopeOp = v.getDefiningOp<scope::ScopeOp>()) {
    auto &scopeBody = scopeOp.getRegion().front();
    auto returnOp = cast<scope::ReturnOp>(scopeBody.getTerminator());
    auto resIdx = cast<OpResult>(v).getResultNumber();
    return getLocalBuffer(returnOp.getResults()[resIdx]);
  }
  if (auto blockArgument = dyn_cast<BlockArgument>(v)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArgument.getOwner()->getParentOp());
    if (!forOp)
      return std::nullopt;
    return getLocalBuffer(forOp.getTiedLoopInit(blockArgument)->get());
  }
  return std::nullopt;
}

static void collectPreloadLocalBuffers(func::FuncOp funcOp, PreloadInfo &info) {
  auto isPointerCast = [](Value value) {
    return static_cast<bool>(value.getDefiningOp<hivm::PointerCastOp>());
  };

  funcOp.walk([&](annotation::MarkOp markOp) {
    if (!utils::isAnnotationWithAttr(markOp,
                                     hivm::PreloadLocalBufferAttr::name))
      return;

    auto roots =
        utils::tracebackMemRefVecByTargetFn(markOp.getSrc(), isPointerCast);
    if (roots.size() != 1)
      return;

    auto pointerCastOp = roots.front().getDefiningOp<hivm::PointerCastOp>();
    if (!pointerCastOp || pointerCastOp.getAddrs().empty())
      return;

    // PlanMemory materializes one address per physical buffer, so the address
    // list is authoritative even when max_preload_num is larger.
    info.preloadLocalBuffers[pointerCastOp.getResult()] =
        pointerCastOp.getAddrs().size();
  });
}

static std::optional<size_t> getPreloadLocalBufferNum(hivm::PointerCastOp op,
                                                      const PreloadInfo &info) {
  auto it = info.preloadLocalBuffers.find(op.getResult());
  if (it != info.preloadLocalBuffers.end())
    return it->second;

  // Preserve support for the legacy form that stores the marker directly on
  // the pointer_cast operation instead of using annotation.mark.
  if (op->hasAttr(hivm::PreloadLocalBufferAttr::name) && !op.getAddrs().empty())
    return op.getAddrs().size();

  return std::nullopt;
}

static Value cloneLocalBuffer(Value oldValue, hivm::PointerCastOp op,
                              size_t preloadNum, PreloadInfo &info,
                              OpBuilder &b) {
  auto maybeBufferNum = getPreloadLocalBufferNum(op, info);
  assert(maybeBufferNum && *maybeBufferNum > 0 &&
         "expected a preload-local buffer with planned addresses");
  size_t bufferNum = *maybeBufferNum;
  SmallVector<Value> addrs(bufferNum);

  size_t preloadIdx = preloadNum % bufferNum;
  for (auto [i, addr] : llvm::enumerate(op.getAddrs())) {
    auto shiftedIdx = (bufferNum - preloadIdx - 1 + i) % bufferNum;
    addrs[shiftedIdx] = addr;
  }
  auto newPointerCastOp =
      b.create<hivm::PointerCastOp>(op.getLoc(), op.getType(), addrs);
  info.mappings[preloadNum].map(oldValue, newPointerCastOp);
  return newPointerCastOp;
}

static bool hasPreloadWorkspaceMark(Value value) {
  if (!value)
    return false;

  if (utils::getAnnotateOpWithAttr(value, hivm::PreloadWorkspaceAttr::name)
          .has_value()) {
    return true;
  }

  if (Operation *defOp = value.getDefiningOp()) {
    return defOp->hasAttr(hivm::PreloadWorkspaceAttr::name);
  }

  return false;
}

static bool isPreloadWorkspaceSubview(memref::SubViewOp subviewOp) {
  return hasPreloadWorkspaceMark(subviewOp.getSource());
}

static Value cloneWorkspaceSubview(memref::SubViewOp subviewOp,
                                   size_t preloadNum, PreloadInfo &info,
                                   OpBuilder &b) {
  auto offsets = subviewOp.getMixedOffsets();
  auto loc = subviewOp.getLoc();

  auto indVar = info.mappings[preloadNum].lookup(info.indVar);

  Value offset = b.create<arith::DivSIOp>(indVar.getLoc(), indVar, info.step);
  offset =
      b.create<arith::RemSIOp>(offset.getLoc(), offset, info.maxPreloadValue);
  offset =
      b.create<arith::IndexCastOp>(offset.getLoc(), b.getIndexType(), offset);
  offsets[0] = offset;

  auto src = info.mappings[preloadNum].lookupOrDefault(subviewOp.getSource());
  Value newResult;

  if (subviewOp.getSourceType().getRank() == subviewOp.getType().getRank()) {
    newResult = b.create<memref::SubViewOp>(loc, src, offsets,
                                            subviewOp.getMixedSizes(),
                                            subviewOp.getMixedStrides());
  } else {
    auto newResType = memref::SubViewOp::inferRankReducedResultType(
        subviewOp.getType().getShape(), subviewOp.getSourceType(), offsets,
        subviewOp.getMixedSizes(), subviewOp.getMixedStrides());
    newResult = b.create<memref::SubViewOp>(
        loc, cast<MemRefType>(newResType), src, offsets,
        subviewOp.getMixedSizes(), subviewOp.getMixedStrides());
  }

  auto adaptorOp =
      b.create<UnrealizedConversionCastOp>(loc, subviewOp.getType(), newResult);

  adaptorOp->setAttr("preload_adaptor", b.getUnitAttr());
  LDBG("Mapping " << subviewOp << '\n' << adaptorOp);
  info.mappings[preloadNum].map(subviewOp.getResult(), adaptorOp->getResult(0));

  return adaptorOp->getResult(0);
}

static Value
preprocessLoopArgs(scf::ForOp forOp, SmallVector<Value> &newInitArgs,
                   DenseMap<Value, hivm::PointerCastOp> &valueToAdapt,
                   const PreloadInfo &info, OpBuilder &b) {
  for (auto [initArg, iterArg] :
       llvm::zip_equal(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
    if (auto pointerCastOp = initArg.getDefiningOp<hivm::PointerCastOp>();
        pointerCastOp && getPreloadLocalBufferNum(pointerCastOp, info)) {
      valueToAdapt[iterArg] = pointerCastOp;
    } else {
      newInitArgs.push_back(initArg);
    }
  }
  if (info.elideInactiveTail)
    return info.ub;
  Value extraIters = b.create<arith::MulIOp>(info.ub.getLoc(),
                                             info.maxPreloadValue, info.step);
  Value expandedUb =
      b.create<arith::AddIOp>(info.ub.getLoc(), info.ub, extraIters);
  // Keep empty loops empty: extra drain IVs would still run cloned SET/WAIT.
  Value nonempty = b.create<arith::CmpIOp>(
      info.ub.getLoc(), arith::CmpIPredicate::slt, info.lb, info.ub);
  return b.create<arith::SelectOp>(info.ub.getLoc(), nonempty, expandedUb,
                                   info.lb);
}

static Value getPreloadCondition(const PreloadInfo &info, Location loc,
                                 OpBuilder &b) {
  if (info.elideInactiveTail)
    return b.create<arith::ConstantIntOp>(loc, 1, 1);
  auto &mapping = info.mappings[info.preloadNum];

  Value indVar = mapping.lookup(info.indVar);
  Value lb = mapping.lookupOrDefault(info.lb);
  Value ub = mapping.lookupOrDefault(info.ub);

  Value lowerCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, indVar, lb);
  Value upperCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, indVar, ub);

  auto andOp = b.create<arith::AndIOp>(loc, lowerCond, upperCond);
  auto markOp = b.create<annotation::MarkOp>(loc, andOp.getResult());
  markOp->setAttr(hivm::PreloadNumAttr::name,
                  b.getI32IntegerAttr(info.preloadNum));
  return andOp;
}

static bool isRematerializableAliasOp(Operation *op) {
  if (!op)
    return false;
  // View-like ops, pointer casts, and pure arithmetic ops can be rematerialized
  // as part of a result chain in the preload skip branch.
  return isa<ViewLikeOpInterface, hivm::PointerCastOp>(op) ||
         isa<arith::ArithDialect>(op->getDialect());
}

static void rewriteScopeReturnOp(ValueRange returnResults,
                                 const PreloadInfo &info, Location loc,
                                 OpBuilder &b) {
  SmallVector<Value> newReturns;
  for (auto res : returnResults) {
    if (!getLocalBuffer(res)) {
      res = info.mappings[info.preloadNum].lookupOrDefault(res);
      newReturns.push_back(res);
    }
  }
  b.create<scope::ReturnOp>(loc, newReturns);
}

static void rewriteBody(Block *body, PreloadInfo &info, OpBuilder &b) {
  for (auto &op : body->without_terminator()) {
    if (auto pointerCastOp = dyn_cast<hivm::PointerCastOp>(&op);
        pointerCastOp && getPreloadLocalBufferNum(pointerCastOp, info)) {
      cloneLocalBuffer(pointerCastOp, pointerCastOp, info.preloadNum, info, b);
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(&op);
               subviewOp && isPreloadWorkspaceSubview(subviewOp)) {
      cloneWorkspaceSubview(subviewOp, info.preloadNum, info, b);
    } else if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      auto &mapping = info.mappings[info.preloadNum];
      auto lb = mapping.lookupOrDefault(forOp.getLowerBound());
      auto ub = mapping.lookupOrDefault(forOp.getUpperBound());
      auto step = mapping.lookupOrDefault(forOp.getStep());
      SmallVector<Value> inits;
      for (auto init : forOp.getInits()) {
        inits.push_back(mapping.lookupOrDefault(init));
      }
      auto newForOp = b.create<scf::ForOp>(
          op.getLoc(), lb, ub, step, inits,
          [&](OpBuilder &b, Location loc, Value indVar, ValueRange args) {
            mapping.map(forOp.getInductionVar(), indVar);
            mapping.map(forOp.getRegionIterArgs(), args);
            rewriteBody(forOp.getBody(), info, b);
            b.clone(*forOp.getBody()->getTerminator(),
                    info.mappings[info.preloadNum]);
          });
      // Preserve iteration independence so sync analysis does not introduce
      // loop-carried barriers between disjoint page transfers.
      if (auto parallelAttr =
              forOp->getAttrOfType<UnitAttr>(hivm::ParallelLoopAttr::name))
        newForOp->setAttr(hivm::ParallelLoopAttr::name, parallelAttr);
      info.mappings[info.preloadNum].map(forOp->getResults(),
                                         newForOp->getResults());
    } else {
      b.clone(op, info.mappings[info.preloadNum]);
    }
  }
}

static Value lookThroughScopeResult(Value value) {
  while (auto scopeOp = value.getDefiningOp<scope::ScopeOp>()) {
    auto returnOp =
        cast<scope::ReturnOp>(scopeOp.getRegion().front().getTerminator());
    value = returnOp.getResults()[cast<OpResult>(value).getResultNumber()];
  }
  return value;
}

static void emitRematerializationError(Operation *definingOp,
                                       scope::ScopeOp scopeOp) {
  if (!definingOp) {
    scopeOp.emitOpError(
        "cannot rematerialize a preload scope result through an internal "
        "block argument");
  } else {
    definingOp->emitOpError(
        "is not a supported view while rematerializing a preload scope "
        "result");
  }
}

static FailureOr<Value>
rematerializeViewLikeResult(Value value, scope::ScopeOp scopeOp,
                            const IRMapping &preloadMapping,
                            IRMapping &viewMapping, OpBuilder &builder) {
  if (Value mapped = viewMapping.lookupOrNull(value))
    return mapped;

  Value source = lookThroughScopeResult(value);
  if (source != value) {
    auto rematerialized = rematerializeViewLikeResult(
        source, scopeOp, preloadMapping, viewMapping, builder);
    if (failed(rematerialized))
      return failure();
    viewMapping.map(value, *rematerialized);
    return *rematerialized;
  }

  Region *definingRegion = value.getParentRegion();
  if (!definingRegion || !scopeOp.getRegion().isAncestor(definingRegion)) {
    Value mapped = preloadMapping.lookupOrDefault(value);
    viewMapping.map(value, mapped);
    return mapped;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!isRematerializableAliasOp(definingOp)) {
    emitRematerializationError(definingOp, scopeOp);
    return failure();
  }

  for (Value operand : definingOp->getOperands()) {
    if (failed(rematerializeViewLikeResult(operand, scopeOp, preloadMapping,
                                           viewMapping, builder)))
      return failure();
  }

  builder.clone(*definingOp, viewMapping);
  return viewMapping.lookup(value);
}

static scf::IfOp rewriteScopeOp(Value cond, scope::ScopeOp scopeOp,
                                PreloadInfo &info, ValueRange loopArgs,
                                Location loc, OpBuilder &builder) {
  auto &scopeBody = scopeOp.getRegion().front();
  auto returnResults =
      cast<scope::ReturnOp>(scopeBody.getTerminator()).getResults();
  return builder.create<scf::IfOp>(
      loc, cond,
      [&](OpBuilder &b, Location loc) {
        SmallVector<Type> newScopeResultTypes;
        for (auto res : returnResults) {
          if (!getLocalBuffer(res))
            newScopeResultTypes.push_back(res.getType());
        }
        auto newOp =
            b.create<scope::ScopeOp>(scopeOp.getLoc(), newScopeResultTypes);
        newOp->setAttrs(scopeOp->getAttrs());
        newOp.setNoInline(false);

        Region &region = newOp.getRegion();
        Block *bodyBlock = b.createBlock(&region);

        b.setInsertionPointToEnd(bodyBlock);
        rewriteBody(&scopeBody, info, b);
        rewriteScopeReturnOp(returnResults, info, loc, b);
        b.setInsertionPointAfter(newOp);
        b.create<scf::YieldOp>(loc, newOp->getResults());
      },
      [&](OpBuilder &b, Location loc) {
        SmallVector<Value> newYields;
        IRMapping viewMapping;
        for (auto [res, retRes] :
             llvm::zip_equal(scopeOp->getResults(), returnResults)) {
          if (!getLocalBuffer(retRes)) {
            auto yieldUse = llvm::find_if(res.getUses(), [](OpOperand &use) {
              return isa<scf::YieldOp>(use.getOwner());
            });
            if (yieldUse != res.use_end()) {
              auto oprNum = yieldUse->getOperandNumber();
              newYields.push_back(loopArgs[oprNum]);
            } else if (auto pointerCastOp =
                           retRes.getDefiningOp<hivm::PointerCastOp>()) {
              auto newRes = b.clone(*pointerCastOp)->getResult(0);
              newYields.push_back(newRes);
            } else {
              // Rematerialize the view chain independently in the skip branch.
              auto rematerialized = rematerializeViewLikeResult(
                  retRes, scopeOp, info.mappings[info.preloadNum], viewMapping,
                  b);
              if (failed(rematerialized))
                llvm::report_fatal_error("Unhandled scope result case");
              newYields.push_back(*rematerialized);
            }
          }
        }
        b.create<scf::YieldOp>(loc, newYields);
      });
}

static bool isSynchronizationOp(Operation *op) {
  return op->hasTrait<OpTrait::HIVMSyncOpTrait>();
}

// A single stage needs no fill/drain iteration. Retain the guarded expansion
// if the otherwise inactive tail would execute effects outside stage scopes,
// including synchronization. Pointer construction and marks are metadata.
static bool
canElideInactiveTail(scf::ForOp loop,
                     const DenseMap<Operation *, size_t> &scopeToIdx,
                     size_t maxPreloadNum) {
  // An unguarded scalar counter would advance even on the inactive tick.
  for (auto [arg, yielded] :
       llvm::zip(loop.getRegionIterArgs(), loop.getYieldedValues()))
    if (yielded != arg && !scopeToIdx.contains(yielded.getDefiningOp()))
      return false;
  return maxPreloadNum == 1 &&
         llvm::all_of(scopeToIdx,
                      [](const auto &entry) { return entry.second == 0; }) &&
         llvm::all_of(loop.getBody()->without_terminator(), [&](Operation &op) {
           return scopeToIdx.contains(&op) ||
                  isa<hivm::PointerCastOp, annotation::MarkOp>(op) ||
                  (op.getNumRegions() == 0 && isMemoryEffectFree(&op));
         });
}

static void rewritePreloadLoop(scf::ForOp forOp,
                               const DenseMap<Operation *, size_t> &scopeToIdx,
                               size_t maxPreloadNum, bool elideInactiveTail) {
  IRRewriter rewriter(forOp.getContext());
  PreloadInfo info;
  info.funcCoreType = queryFuncCoreType(forOp->getParentOfType<func::FuncOp>())
                          .value_or(TFuncCoreType::AIC_OR_AIV);
  info.lb = forOp.getLowerBound();
  info.ub = forOp.getUpperBound();
  info.step = forOp.getStep();
  info.maxPreloadNum = maxPreloadNum;
  info.elideInactiveTail = elideInactiveTail;
  rewriter.setInsertionPoint(forOp);
  info.maxPreloadValue = rewriter.create<arith::ConstantOp>(
      info.ub.getLoc(),
      rewriter.getIntegerAttr(info.ub.getType(), info.maxPreloadNum));
  info.mappings.resize(maxPreloadNum);
  collectPreloadLocalBuffers(forOp->getParentOfType<func::FuncOp>(), info);

  SmallVector<Value> newInitArgs;
  DenseMap<Value, hivm::PointerCastOp> valueToAdapt;
  Value newUpperbound =
      preprocessLoopArgs(forOp, newInitArgs, valueToAdapt, info, rewriter);

  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), info.lb, newUpperbound, info.step, newInitArgs,
      [&](OpBuilder &b, Location loc, Value indVar, ValueRange args) {
        Value curIndVar = indVar;
        info.indVar = forOp.getInductionVar();
        for (int64_t preloadNum = maxPreloadNum - 1; preloadNum >= 0;
             preloadNum--) {
          info.mappings[preloadNum].map(info.indVar, curIndVar);
          auto newArgIter = args.begin();
          for (auto oldArg : forOp.getRegionIterArgs()) {
            if (auto it = valueToAdapt.find(oldArg); it != valueToAdapt.end()) {
              auto pointerCastOp = it->second;
              cloneLocalBuffer(oldArg, pointerCastOp, preloadNum, info, b);
            } else {
              info.mappings[preloadNum].map(oldArg, *newArgIter);
              ++newArgIter;
            }
          }
          curIndVar = b.createOrFold<arith::SubIOp>(curIndVar.getLoc(),
                                                    curIndVar, info.step);
        }

        for (auto &bodyOp : forOp.getBody()->without_terminator()) {
          if (!scopeToIdx.contains(&bodyOp)) {
            if (isSynchronizationOp(&bodyOp)) {
              b.clone(bodyOp, info.mappings[maxPreloadNum - 1]);
              continue;
            }
            if (auto pointerCastOp = dyn_cast<hivm::PointerCastOp>(&bodyOp)) {
              if (getPreloadLocalBufferNum(pointerCastOp, info)) {
                for (int64_t preloadNum = maxPreloadNum - 1; preloadNum >= 0;
                     preloadNum--) {
                  cloneLocalBuffer(pointerCastOp, pointerCastOp, preloadNum,
                                   info, b);
                }
                continue;
              }
            } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(&bodyOp);
                       subviewOp && isPreloadWorkspaceSubview(subviewOp)) {
              for (int64_t preloadNum = maxPreloadNum - 1; preloadNum >= 0;
                   preloadNum--) {
                cloneWorkspaceSubview(subviewOp, preloadNum, info, b);
              }
              continue;
            }
            for (auto &mapping : info.mappings)
              b.clone(bodyOp, mapping);
            continue;
          }
          auto scopeOp = cast<scope::ScopeOp>(&bodyOp);
          info.preloadNum = scopeToIdx.at(&bodyOp);

          LDBG("Handling preload scope: " << info.preloadNum << "\n" << bodyOp);

          auto loc = scopeOp.getLoc();
          Value cond = getPreloadCondition(info, loc, b);

          auto newOp = rewriteScopeOp(cond, scopeOp, info, args, loc, b);
          auto returnResults =
              cast<scope::ReturnOp>(scopeOp.getRegion().front().getTerminator())
                  .getResults();
          auto scopeResIter = scopeOp.result_begin();
          SmallVector<IRMapping> viewMappings(info.mappings.size());
          for (auto newRes : newOp->getResults()) {
            auto maybeLocalBuffer = getLocalBuffer(*scopeResIter);
            while (maybeLocalBuffer.has_value()) {
              for (auto &mapping : info.mappings)
                mapping.map(*scopeResIter,
                            mapping.lookup(maybeLocalBuffer->getResult()));
              ++scopeResIter;
              maybeLocalBuffer = getLocalBuffer(*scopeResIter);
            }

            Value scopeRes = *scopeResIter;
            auto resultNumber = cast<OpResult>(scopeRes).getResultNumber();
            Value returnResult = returnResults[resultNumber];
            Value source = lookThroughScopeResult(returnResult);
            if (!isa_and_nonnull<ViewLikeOpInterface>(source.getDefiningOp())) {
              for (auto &mapping : info.mappings)
                mapping.map(scopeRes, newRes);
            } else {
              for (auto [mapping, viewMapping] :
                   llvm::zip_equal(info.mappings, viewMappings)) {
                auto rematerialized = rematerializeViewLikeResult(
                    returnResult, scopeOp, mapping, viewMapping, b);
                assert(succeeded(rematerialized) &&
                       "Failed to rematerialize checked scope result");
                mapping.map(scopeRes, *rematerialized);
              }
            }
            ++scopeResIter;
          }
          for (; scopeResIter != scopeOp.result_end(); ++scopeResIter) {
            auto maybeLocalBuffer = getLocalBuffer(*scopeResIter);
            for (auto &mapping : info.mappings)
              mapping.map(*scopeResIter,
                          mapping.lookup(maybeLocalBuffer->getResult()));
          }
        }

        SmallVector<Value> newYields;
        if (!newInitArgs.empty()) {
          newYields =
              llvm::map_to_vector(forOp.getYieldedValues(), [&](Value yield) {
                return info.mappings[maxPreloadNum - 1].lookupOrDefault(yield);
              });
        }
        b.create<scf::YieldOp>(loc, newYields);
      });

  LDBG("New for loop:\n" << newForOp);

  // `preprocessLoopArgs` drops every iteration argument that carries a
  // preload-local buffer, because the new body re-materializes the rotation
  // belonging to each preload stage instead of threading one buffer through the
  // loop. The original loop still has one result per iteration argument, so
  // walk the original arguments rather than the new results and supply a
  // replacement for the dropped ones too.
  //
  // When a kept yield is itself a preload-local buffer, clone that buffer after
  // the loop instead of rewiring users to the in-loop definition (which would
  // not dominate post-loop uses once the unused yield is DCE'd).
  SmallVector<Value> newRes;
  newRes.reserve(forOp.getNumResults());
  auto newResIt = newForOp->result_begin();
  auto newYieldIt = newForOp.getYieldedValues().begin();
  for (auto [initArg, iterArg] :
       llvm::zip_equal(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
    if (valueToAdapt.contains(iterArg)) {
      newRes.push_back(initArg);
      continue;
    }
    if (auto maybeLocalBuffer = getLocalBuffer(*newYieldIt);
        maybeLocalBuffer.has_value()) {
      IRRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(newForOp);
      hivm::PointerCastOp src = *maybeLocalBuffer;
      newRes.push_back(rewriter.create<hivm::PointerCastOp>(
          src.getLoc(), src.getType(), src.getAddrs()));
    } else {
      newRes.push_back(*newResIt);
    }
    ++newResIt;
    ++newYieldIt;
  }
  // Mark for hivm-peel-preload (runs after GraphSyncSolver).
  newForOp->setAttr(hivm::MaxPreloadNumAttr::name,
                    rewriter.getI32IntegerAttr(maxPreloadNum));
  rewriter.replaceOp(forOp, newRes);
}

struct PropagateAdaptor : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("preload_adaptor"))
      return failure();
    auto input = op.getInputs()[0];
    for (auto &use : op->getUses()) {
      auto *user = use.getOwner();
      auto loc = user->getLoc();
      rewriter.setInsertionPoint(user);
      if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(user)) {
        auto maybeNewType = memref::ExpandShapeOp::computeExpandedType(
            cast<MemRefType>(input.getType()),
            expandShapeOp.getResultType().getShape(),
            expandShapeOp.getReassociationIndices());
        if (failed(maybeNewType))
          return expandShapeOp->emitOpError("Cannot expand shape");
        auto outputShape =
            getMixedValues(expandShapeOp.getStaticOutputShape(),
                           expandShapeOp.getOutputShape(), rewriter);
        auto newOp = rewriter.create<memref::ExpandShapeOp>(
            loc, maybeNewType.value(), input,
            expandShapeOp.getReassociationIndices(), outputShape);
        auto newAdaptorOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, expandShapeOp.getResultType(), newOp.getResult());
        newAdaptorOp->setAttr("preload_adaptor", rewriter.getUnitAttr());
        rewriter.replaceOp(expandShapeOp, newAdaptorOp);
        return success();
      }
      if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(user)) {
        auto newOp = rewriter.create<memref::CollapseShapeOp>(
            loc, input, collapseShapeOp.getReassociationIndices());
        auto newAdaptorOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, collapseShapeOp.getResultType(), newOp.getResult());
        newAdaptorOp->setAttr("preload_adaptor", rewriter.getUnitAttr());
        rewriter.replaceOp(collapseShapeOp, newAdaptorOp);
        return success();
      }
      rewriter.modifyOpInPlace(user, [&]() { use.set(input); });
      if (failed(verify(user))) {
        rewriter.modifyOpInPlace(user, [&]() { use.set(op->getResult(0)); });
        return user->emitOpError("Unhandled propagation");
      }
      return success();
    }
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

static void cleanupPreloadWorkspaceMarks(Operation &op) {
  op.walk([&](annotation::MarkOp markOp) {
    if (utils::isAnnotationWithAttr(markOp, hivm::PreloadWorkspaceAttr::name))
      markOp->removeAttr(hivm::PreloadWorkspaceAttr::name);
  });
}

struct PointerCastTraceResult {
  SmallVector<hivm::PointerCastOp> pointerCasts;
  SmallVector<arith::SelectOp> selectOps;
  SmallVector<Value> ifResults;
  SmallVector<Value> scopeResults;
};

static LogicalResult collectRootPointerCasts(ArrayRef<Value> roots,
                                             PointerCastTraceResult &result) {
  SmallVector<Value> worklist(roots.begin(), roots.end());
  DenseSet<Value> visited;
  DenseSet<Operation *> visitedSelects;
  DenseSet<Value> visitedIfs;
  DenseSet<Value> visitedScopes;
  DenseSet<Operation *> visitedPointerCasts;

  while (!worklist.empty()) {
    Value curr = worklist.pop_back_val();
    if (!curr || !visited.insert(curr).second)
      continue;
    if (!isa<MemRefType>(curr.getType()))
      return failure();

    if (auto pcOp = curr.getDefiningOp<hivm::PointerCastOp>()) {
      if (visitedPointerCasts.insert(pcOp).second)
        result.pointerCasts.push_back(pcOp);
      continue;
    }

    if (auto scopeOp = curr.getDefiningOp<scope::ScopeOp>()) {
      if (visitedScopes.insert(curr).second)
        result.scopeResults.push_back(curr);
      auto &scopeBody = scopeOp.getRegion().front();
      auto returnOp = cast<scope::ReturnOp>(scopeBody.getTerminator());
      auto resIdx = cast<OpResult>(curr).getResultNumber();
      worklist.push_back(returnOp.getResults()[resIdx]);
      continue;
    }

    if (auto ifOp = curr.getDefiningOp<scf::IfOp>()) {
      if (visitedIfs.insert(curr).second)
        result.ifResults.push_back(curr);
      if (ifOp.getElseRegion().empty())
        return failure();
      auto resIdx = cast<OpResult>(curr).getResultNumber();
      worklist.push_back(ifOp.thenYield().getResults()[resIdx]);
      worklist.push_back(ifOp.elseYield().getResults()[resIdx]);
      continue;
    }

    if (auto selectOp = curr.getDefiningOp<arith::SelectOp>()) {
      if (visitedSelects.insert(selectOp).second)
        result.selectOps.push_back(selectOp);
      worklist.push_back(selectOp.getTrueValue());
      worklist.push_back(selectOp.getFalseValue());
      continue;
    }

    if (auto blockArg = dyn_cast<BlockArgument>(curr)) {
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (!forOp || blockArg == forOp.getInductionVar())
        return failure();
      unsigned idx = blockArg.getArgNumber() - 1;
      worklist.push_back(forOp.getInitArgs()[idx]);
      worklist.push_back(forOp.getYieldedValues()[idx]);
      continue;
    }

    if (auto forOp = curr.getDefiningOp<scf::ForOp>()) {
      auto resIdx = cast<OpResult>(curr).getResultNumber();
      worklist.push_back(forOp.getInitArgs()[resIdx]);
      worklist.push_back(forOp.getYieldedValues()[resIdx]);
      continue;
    }

    if (auto castOp = curr.getDefiningOp<memref::CastOp>()) {
      worklist.push_back(castOp.getSource());
      continue;
    }
    if (auto bitcastOp = curr.getDefiningOp<hivm::BitcastOp>()) {
      worklist.push_back(bitcastOp.getSrc());
      continue;
    }
    if (auto convertLayoutOp = curr.getDefiningOp<hivm::ConvertLayoutOp>()) {
      worklist.push_back(convertLayoutOp.getSource());
      continue;
    }
    if (auto unrealizedCast =
            curr.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (unrealizedCast.getInputs().size() != 1)
        return failure();
      worklist.push_back(unrealizedCast.getInputs().front());
      continue;
    }
    if (auto transposeOp = curr.getDefiningOp<memref::TransposeOp>()) {
      worklist.push_back(transposeOp.getIn());
      continue;
    }
    if (auto collapseOp = curr.getDefiningOp<memref::CollapseShapeOp>()) {
      worklist.push_back(collapseOp.getSrc());
      continue;
    }
    if (auto expandOp = curr.getDefiningOp<memref::ExpandShapeOp>()) {
      worklist.push_back(expandOp.getSrc());
      continue;
    }

    return failure();
  }

  if (result.pointerCasts.empty())
    return failure();

  hivm::PointerCastOp first = result.pointerCasts.front();
  if (first.getAddrs().empty())
    return failure();
  for (hivm::PointerCastOp pc : llvm::drop_begin(result.pointerCasts)) {
    if (pc.getAddrs() != first.getAddrs() ||
        pc.getDynamicSizes() != first.getDynamicSizes())
      return failure();
  }
  return success();
}

static Value clonePointerCastForType(hivm::PointerCastOp srcOp, Type targetType,
                                     OpBuilder &builder) {
  if (srcOp.getType() == targetType)
    return builder.create<hivm::PointerCastOp>(srcOp.getLoc(), srcOp.getType(),
                                               srcOp.getAddrs(),
                                               srcOp.getDynamicSizes());
  auto targetMemRefType = cast<MemRefType>(targetType);
  if (targetMemRefType.hasStaticShape())
    return builder.create<hivm::PointerCastOp>(srcOp.getLoc(), targetType,
                                               srcOp.getAddrs());
  Value cloned = builder.create<hivm::PointerCastOp>(
      srcOp.getLoc(), srcOp.getType(), srcOp.getAddrs(),
      srcOp.getDynamicSizes());
  if (memref::CastOp::areCastCompatible(cloned.getType(), targetType))
    return builder.create<memref::CastOp>(srcOp.getLoc(), targetType, cloned);
  return builder
      .create<UnrealizedConversionCastOp>(srcOp.getLoc(), targetType, cloned)
      .getResult(0);
}

static void pruneUnusedScopeResultsInLoop(scf::ForOp forOp) {
  SmallVector<scope::ScopeOp> scopeOps;
  forOp.walk([&](scope::ScopeOp scopeOp) { scopeOps.push_back(scopeOp); });

  for (scope::ScopeOp scopeOp : scopeOps) {
    if (scopeOp.getNumResults() == 0)
      continue;
    llvm::BitVector keepResult(scopeOp.getNumResults(), false);
    for (unsigned i = 0, e = scopeOp.getNumResults(); i < e; ++i) {
      if (!scopeOp.getResult(i).use_empty())
        keepResult.set(i);
    }
    if (keepResult.all())
      continue;

    auto &scopeBody = scopeOp.getRegion().front();
    auto returnOp = cast<scope::ReturnOp>(scopeBody.getTerminator());
    SmallVector<Type> newResultTypes;
    SmallVector<Value> newReturnVals;
    SmallVector<Value> deadReturnVals;
    for (unsigned i = 0, e = scopeOp.getNumResults(); i < e; ++i) {
      if (keepResult.test(i)) {
        newResultTypes.push_back(scopeOp.getResult(i).getType());
        newReturnVals.push_back(returnOp.getResults()[i]);
      } else {
        deadReturnVals.push_back(returnOp.getResults()[i]);
      }
    }

    OpBuilder builder(scopeOp);
    auto newScopeOp =
        builder.create<scope::ScopeOp>(scopeOp.getLoc(), newResultTypes);
    newScopeOp->setAttrs(scopeOp->getAttrs());
    Region &newRegion = newScopeOp.getRegion();
    newRegion.takeBody(scopeOp.getRegion());
    Block &newBody = newRegion.front();
    auto oldReturnOp = cast<scope::ReturnOp>(newBody.getTerminator());
    builder.setInsertionPoint(oldReturnOp);
    builder.create<scope::ReturnOp>(oldReturnOp.getLoc(), newReturnVals);
    oldReturnOp.erase();

    unsigned newIdx = 0;
    for (unsigned i = 0, e = scopeOp.getNumResults(); i < e; ++i) {
      if (keepResult.test(i))
        scopeOp.getResult(i).replaceAllUsesWith(newScopeOp.getResult(newIdx++));
    }
    scopeOp.erase();

    for (Value deadVal : deadReturnVals) {
      if (Operation *defOp = deadVal.getDefiningOp()) {
        if (defOp->use_empty() &&
            isa<hivm::PointerCastOp, arith::SelectOp>(defOp) &&
            !utils::getAnnotateOpWithAttr(deadVal,
                                          hivm::PreloadLocalBufferAttr::name)
                 .has_value()) {
          defOp->erase();
        }
      }
    }
  }
}

static void preprocessPreloadLoop(scf::ForOp forOp) {
  auto isPointerCast = [](Value value) {
    return static_cast<bool>(value.getDefiningOp<hivm::PointerCastOp>());
  };

  // Ensure any preload-local PointerCastOp marked inside `forOp` but defined
  // outside `forOp` is cloned into the top of `forOp.getBody()` so
  // `rewritePreloadLoop` rotates it per preload stage.
  DenseMap<Operation *, hivm::PointerCastOp> externalLocalBuffers;
  DenseSet<Operation *> localBufferOps;
  forOp.walk([&](annotation::MarkOp markOp) {
    if (!utils::isAnnotationWithAttr(markOp,
                                     hivm::PreloadLocalBufferAttr::name))
      return;
    auto roots =
        utils::tracebackMemRefVecByTargetFn(markOp.getSrc(), isPointerCast);
    if (roots.size() != 1)
      return;
    auto pcOp = roots.front().getDefiningOp<hivm::PointerCastOp>();
    if (!pcOp || pcOp.getAddrs().empty())
      return;
    localBufferOps.insert(pcOp);
    if (!forOp.getBodyRegion().isAncestor(pcOp->getParentRegion()))
      externalLocalBuffers.try_emplace(pcOp, pcOp);
  });

  if (!externalLocalBuffers.empty()) {
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(forOp.getBody());
    for (auto &[opPtr, pcOp] : externalLocalBuffers) {
      auto clonedPC = bodyBuilder.create<hivm::PointerCastOp>(
          pcOp.getLoc(), pcOp.getType(), pcOp.getAddrs(),
          pcOp.getDynamicSizes());
      pcOp.getResult().replaceUsesWithIf(
          clonedPC.getResult(), [&](OpOperand &use) {
            return forOp.getBodyRegion().isAncestor(
                use.getOwner()->getParentRegion());
          });
      localBufferOps.erase(pcOp);
      localBufferOps.insert(clonedPC);
    }
  }

  auto findMarkedLocalBuffer =
      [&](ArrayRef<hivm::PointerCastOp> pcs) -> hivm::PointerCastOp {
    for (hivm::PointerCastOp pc : pcs) {
      if (localBufferOps.contains(pc) ||
          pc->hasAttr(hivm::PreloadLocalBufferAttr::name) ||
          utils::getAnnotateOpWithAttr(pc.getResult(),
                                       hivm::PreloadLocalBufferAttr::name)
              .has_value())
        return pc;
    }
    return nullptr;
  };

  unsigned numIterArgs = forOp.getRegionIterArgs().size();
  llvm::BitVector keepIterArg(numIterArgs, true);
  OpBuilder builder(forOp);

  for (unsigned idx = 0; idx < numIterArgs; ++idx) {
    Value initArg = forOp.getInitArgs()[idx];
    Value iterArg = forOp.getRegionIterArgs()[idx];
    Value yieldVal = forOp.getYieldedValues()[idx];
    Value forResult = forOp.getResult(idx);

    if (!isa<MemRefType>(yieldVal.getType()))
      continue;

    // An invariant memref already has the correct buffer selection. In
    // particular, cloning an outer-loop multi-buffer pointer_cast into this
    // loop would rotate its reads with this loop while its writes still use
    // the outer-loop buffer.
    if (initArg == yieldVal) {
      iterArg.replaceAllUsesWith(initArg);
      forResult.replaceAllUsesWith(initArg);
      keepIterArg.reset(idx);
      continue;
    }

    PointerCastTraceResult traceResult;
    if (failed(collectRootPointerCasts({yieldVal, initArg}, traceResult)))
      continue;

    hivm::PointerCastOp firstPC = traceResult.pointerCasts.front();
    hivm::PointerCastOp markedLocalPC =
        findMarkedLocalBuffer(traceResult.pointerCasts);
    keepIterArg.reset(idx);

    if (!forResult.use_empty()) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(forOp);
      Value replacement =
          clonePointerCastForType(firstPC, forResult.getType(), builder);
      forResult.replaceAllUsesWith(replacement);
    }

    if (!iterArg.use_empty()) {
      Value replacement;
      if (markedLocalPC && markedLocalPC->getBlock() == forOp.getBody() &&
          markedLocalPC.getType() == iterArg.getType()) {
        replacement = markedLocalPC.getResult();
      } else {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(forOp.getBody());
        replacement =
            clonePointerCastForType(firstPC, iterArg.getType(), builder);
      }
      iterArg.replaceAllUsesWith(replacement);
    }

    for (arith::SelectOp selectOp : traceResult.selectOps) {
      if (!selectOp.use_empty()) {
        Value replacement;
        if (markedLocalPC && markedLocalPC->getBlock() == forOp.getBody() &&
            markedLocalPC.getType() == selectOp.getType()) {
          replacement = markedLocalPC.getResult();
        } else {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(selectOp);
          replacement =
              clonePointerCastForType(firstPC, selectOp.getType(), builder);
        }
        selectOp.replaceAllUsesWith(replacement);
      }
      selectOp.erase();
    }

    for (Value ifRes : traceResult.ifResults) {
      if (!ifRes.use_empty()) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(ifRes.getDefiningOp());
        Value replacement =
            clonePointerCastForType(firstPC, ifRes.getType(), builder);
        ifRes.replaceAllUsesWith(replacement);
      }
    }

    for (Value scopeRes : traceResult.scopeResults) {
      if (llvm::any_of(scopeRes.getUses(), [&](OpOperand &use) {
            return use.getOwner() != forOp.getBody()->getTerminator();
          })) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(scopeRes.getDefiningOp());
        Value replacement =
            clonePointerCastForType(firstPC, scopeRes.getType(), builder);
        scopeRes.replaceUsesWithIf(replacement, [&](OpOperand &use) {
          return use.getOwner() != forOp.getBody()->getTerminator();
        });
      }
    }
  }

  // Also preprocess any remaining arith.select of memrefs inside `forOp` whose
  // branches trace to root PointerCastOps with identical address lists.
  SmallVector<arith::SelectOp> remainingSelects;
  forOp.walk([&](arith::SelectOp selectOp) {
    if (isa<MemRefType>(selectOp.getType()))
      remainingSelects.push_back(selectOp);
  });
  for (arith::SelectOp selectOp : remainingSelects) {
    PointerCastTraceResult traceResult;
    if (failed(collectRootPointerCasts({selectOp.getResult()}, traceResult)))
      continue;
    hivm::PointerCastOp firstPC = traceResult.pointerCasts.front();
    hivm::PointerCastOp markedLocalPC =
        findMarkedLocalBuffer(traceResult.pointerCasts);
    if (!selectOp.use_empty()) {
      Value replacement;
      if (markedLocalPC && markedLocalPC->getBlock() == forOp.getBody() &&
          markedLocalPC.getType() == selectOp.getType()) {
        replacement = markedLocalPC.getResult();
      } else {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(selectOp);
        replacement =
            clonePointerCastForType(firstPC, selectOp.getType(), builder);
      }
      selectOp.replaceAllUsesWith(replacement);
    }
    selectOp.erase();
  }

  scf::ForOp activeForOp = forOp;
  if (!keepIterArg.all()) {
    SmallVector<Value> newInitArgs;
    SmallVector<Value> newYieldVals;
    for (unsigned idx = 0; idx < numIterArgs; ++idx) {
      if (keepIterArg.test(idx)) {
        newInitArgs.push_back(forOp.getInitArgs()[idx]);
        newYieldVals.push_back(forOp.getYieldedValues()[idx]);
      }
    }

    builder.setInsertionPoint(forOp);
    auto newForOp = builder.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->setAttrs(forOp->getAttrs());

    Block *oldBody = forOp.getBody();
    Block *newBody = newForOp.getBody();
    if (newBody->mightHaveTerminator())
      newBody->getTerminator()->erase();

    auto oldTermIt = std::prev(oldBody->end());
    newBody->getOperations().splice(newBody->end(), oldBody->getOperations(),
                                    oldBody->begin(), oldTermIt);

    forOp.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
    unsigned newIdx = 0;
    for (unsigned idx = 0; idx < numIterArgs; ++idx) {
      if (keepIterArg.test(idx)) {
        forOp.getRegionIterArgs()[idx].replaceAllUsesWith(
            newForOp.getRegionIterArgs()[newIdx]);
        forOp.getResult(idx).replaceAllUsesWith(newForOp.getResult(newIdx));
        ++newIdx;
      }
    }

    builder.setInsertionPointToEnd(newBody);
    builder.create<scf::YieldOp>(oldTermIt->getLoc(), newYieldVals);
    forOp.erase();
    activeForOp = newForOp;
  }

  pruneUnusedScopeResultsInLoop(activeForOp);
}

void CreatePreloadPass::runOnOperation() {
  auto moduleOp = getOperation();
  SmallVector<scf::ForOp> candidateLoops;
  DenseSet<Operation *> seenCandidateLoops;
  moduleOp->walk([&](scope::ScopeOp scopeOp) {
    if (!scopeOp->hasAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name) &&
        !scopeOp->hasAttrOfType<IntegerAttr>(hivm::MaxPreloadNumAttr::name))
      return;
    if (auto parentForOp = dyn_cast<scf::ForOp>(scopeOp->getParentOp())) {
      if (seenCandidateLoops.insert(parentForOp).second)
        candidateLoops.push_back(parentForOp);
    }
  });
  for (scf::ForOp forOp : candidateLoops)
    preprocessPreloadLoop(forOp);

  DenseMap<scf::ForOp, DenseMap<Operation *, size_t>> loopToScopes;
  DenseMap<scf::ForOp, size_t> loopToMaxPreloadNum;
  DenseMap<scf::ForOp, SmallVector<scope::ScopeOp, 4>> uniqueSlots;
  moduleOp->walk([&](scope::ScopeOp scopeOp) {
    auto parentForOp = dyn_cast<scf::ForOp>(scopeOp->getParentOp());
    if (!parentForOp)
      return;
    auto parentFunc = parentForOp->getParentOfType<func::FuncOp>();
    bool allow =
        parentFunc && allowLoopShapeHeuristics(
                          this->bypassShapeRegistry, parentFunc.getName(),
                          this->enablePreload, this->workspaceMultiBufferNum);
    if (auto maxPreloadNumAttr = scopeOp->getAttrOfType<IntegerAttr>(
            hivm::MaxPreloadNumAttr::name)) {
      if (allow) {
        loopToMaxPreloadNum[parentForOp] =
            std::max(loopToMaxPreloadNum[parentForOp],
                     static_cast<size_t>(maxPreloadNumAttr.getInt()));
      } else {
        uniqueSlots[parentForOp].resize(maxPreloadNumAttr.getInt(), nullptr);
        loopToMaxPreloadNum[parentForOp] = uniqueSlots[parentForOp].size();
      }
    }
    if (auto preloadNumAttr =
            scopeOp->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name)) {
      auto preloadNum = preloadNumAttr.getInt();
      assert(preloadNum >= 0 && "PreloadNum must be non-negative integer");
      if (allow) {
        loopToScopes[parentForOp][scopeOp] = static_cast<size_t>(preloadNum);
      } else {
        auto &preloadVec = uniqueSlots[parentForOp];
        assert(preloadNum < static_cast<int64_t>(preloadVec.size()) &&
               "MaxPreloadNumAttr must be set");
        preloadVec[preloadNum] = scopeOp;
      }
    }
  });
  for (auto &[forOp, scopes] : uniqueSlots) {
    for (size_t i = 0, e = scopes.size(); i < e; ++i) {
      if (scopes[i])
        loopToScopes[forOp][scopes[i]] = i;
    }
  }

  if (loopToScopes.empty())
    return;

  for (auto &[forOp, scopeToIdx] : loopToScopes) {
    LDBG("Processing preload:\n" << forOp);
    assert(loopToMaxPreloadNum.contains(forOp) &&
           "MaxPreloadNumAttr must be set");
    size_t maxPreloadNum = loopToMaxPreloadNum[forOp];
    assert(maxPreloadNum > 0 && "MaxPreloadNum must be positive");
    // Loops in uniqueSlots did not pass the shared heuristic gate.
    rewritePreloadLoop(
        forOp, scopeToIdx, maxPreloadNum,
        !uniqueSlots.contains(forOp) &&
            canElideInactiveTail(forOp, scopeToIdx, maxPreloadNum));
  }

  cleanupPreloadWorkspaceMarks(*moduleOp);

  RewritePatternSet patterns(&getContext());
  patterns.add<PropagateAdaptor>(&getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    signalPassFailure();
  }

  LDBG("After processing preload:\n" << *moduleOp);

  PassManager pm(&getContext());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(scope::createInlineScopePass());

  if (failed(pm.run(moduleOp))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass>
mlir::hivm::createCreatePreloadPass(const CreatePreloadOptions &options) {
  return std::make_unique<CreatePreloadPass>(options);
}

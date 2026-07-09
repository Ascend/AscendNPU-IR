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
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DECL_CREATEPRELOAD
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
  Value maxPreloadValue;
  // Loop info
  Value indVar;
  Value lb;
  Value ub;
  Value step;
  // Kernel info
  TFuncCoreType funcCoreType;
  // IRMapping for rewriting
  SmallVector<IRMapping> mappings;
};

struct CreatePreloadPass : public impl::CreatePreloadBase<CreatePreloadPass> {
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

static Value cloneLocalBuffer(Value oldValue, hivm::PointerCastOp op,
                              size_t preloadNum, PreloadInfo &info,
                              OpBuilder &b) {
  SmallVector<Value> addrs(info.maxPreloadNum);
  for (auto [i, addr] : llvm::enumerate(op.getAddrs())) {
    auto shiftedIdx =
        (info.maxPreloadNum - preloadNum - 1 + i) % info.maxPreloadNum;
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
    if (defOp->hasAttr(hivm::PreloadWorkspaceAttr::name))
      return true;
  }

  return false;
}

static Value findPreloadWorkspaceMarkedValue(Value value) {
  auto roots = utils::tracebackMemRefVecByTargetFn(
      value, [](Value v) { return hasPreloadWorkspaceMark(v); });

  if (roots.empty())
    return Value();

  if (roots.size() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "[hivm-create-preload]: ambiguous preload workspace roots, "
               << "root size = " << roots.size() << "\n");
    return Value();
  }

  Value root = roots.front();
  if (!hasPreloadWorkspaceMark(root))
    return Value();

  return root;
}

static bool isPreloadWorkspaceSubview(memref::SubViewOp subviewOp) {
  if (subviewOp->hasAttr(hivm::PreloadWorkspaceAttr::name))
    return true;

  Value markedValue = findPreloadWorkspaceMarkedValue(subviewOp.getSource());
  if (!markedValue)
    return false;

  auto markedType = dyn_cast<MemRefType>(markedValue.getType());
  if (!markedType)
    return false;

  // Only rewrite subviews whose source still has the workspace root rank.
  // If a collapse_shape has already removed the preload slot dimension,
  // offset[0] is no longer the slot offset.
  return subviewOp.getSourceType().getRank() == markedType.getRank();
}

static Value
cloneWorkspaceSubview(memref::SubViewOp subviewOp, size_t preloadNum,
                      PreloadInfo &info, OpBuilder &b) {
  auto offsets = subviewOp.getMixedOffsets();
  auto loc = subviewOp.getLoc();

  auto indVar = info.mappings[preloadNum].lookup(info.indVar);

  Value offset = b.create<arith::DivSIOp>(indVar.getLoc(), indVar, info.step);
  offset = b.create<arith::RemSIOp>(offset.getLoc(), offset,
                                    info.maxPreloadValue);
  offset = b.create<arith::IndexCastOp>(offset.getLoc(), b.getIndexType(),
                                        offset);
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
        pointerCastOp &&
        pointerCastOp->hasAttr(hivm::PreloadLocalBufferAttr::name)) {
      valueToAdapt[iterArg] = pointerCastOp;
    } else {
      newInitArgs.push_back(initArg);
    }
  }
  Value newUpperbound = b.create<arith::MulIOp>(
      info.ub.getLoc(), info.maxPreloadValue, info.step);
  newUpperbound =
      b.create<arith::AddIOp>(info.ub.getLoc(), info.ub, newUpperbound);
  return newUpperbound;
}

static Value getPreloadCondition(const PreloadInfo &info, Location loc,
                                 OpBuilder &b) {
  auto &mapping = info.mappings[info.preloadNum];

  Value indVar = mapping.lookup(info.indVar);
  Value lb = mapping.lookupOrDefault(info.lb);
  Value ub = mapping.lookupOrDefault(info.ub);

  Value lowerCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, indVar, lb);
  Value upperCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, indVar, ub);

  return b.create<arith::AndIOp>(loc, lowerCond, upperCond);
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
        pointerCastOp && getLocalBuffer(pointerCastOp)) {
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
      info.mappings[info.preloadNum].map(forOp->getResults(),
                                         newForOp->getResults());
    } else {
      b.clone(op, info.mappings[info.preloadNum]);
    }
  }
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
        for (auto [res, retRes] :
             llvm::zip_equal(scopeOp->getResults(), returnResults)) {
          if (!getLocalBuffer(retRes)) {
            if (res.hasOneUse() && isa<scf::YieldOp>(*res.user_begin())) {
              auto oprNum = res.use_begin()->getOperandNumber();
              newYields.push_back(loopArgs[oprNum]);
            } else if (auto pointerCastOp =
                           retRes.getDefiningOp<hivm::PointerCastOp>()) {
              auto newRes = b.clone(*pointerCastOp)->getResult(0);
              newYields.push_back(newRes);
            } else {
              llvm::report_fatal_error("Unhandled scope result case");
            }
          }
        }
        b.create<scf::YieldOp>(loc, newYields);
      });
}

static void rewritePreloadLoop(scf::ForOp forOp,
                               SmallVector<scope::ScopeOp, 4> scopes,
                               size_t maxPreloadNum) {
  IRRewriter rewriter(forOp.getContext());
  PreloadInfo info;
  info.funcCoreType = queryFuncCoreType(forOp->getParentOfType<func::FuncOp>())
                          .value_or(TFuncCoreType::AIC_OR_AIV);
  info.lb = forOp.getLowerBound();
  info.ub = forOp.getUpperBound();
  info.step = forOp.getStep();
  info.maxPreloadNum = maxPreloadNum;
  rewriter.setInsertionPoint(forOp);
  info.maxPreloadValue = rewriter.create<arith::ConstantOp>(
      info.ub.getLoc(),
      rewriter.getIntegerAttr(info.ub.getType(), info.maxPreloadNum));
  info.mappings.resize(maxPreloadNum);

  SmallVector<Value> newInitArgs;
  DenseMap<Value, hivm::PointerCastOp> valueToAdapt;
  Value newUpperbound =
      preprocessLoopArgs(forOp, newInitArgs, valueToAdapt, info, rewriter);

  DenseMap<Operation *, size_t> scopeToIdx;
  for (size_t preloadNum = 0; preloadNum < scopes.size(); preloadNum++) {
    auto scopeOp = scopes[preloadNum];
    if (!scopeOp)
      continue;
    scopeToIdx[scopeOp] = preloadNum;
  }

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
            if (auto pointerCastOp = dyn_cast<hivm::PointerCastOp>(&bodyOp)) {
              if (getLocalBuffer(pointerCastOp)) {
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
          auto scopeResIter = scopeOp.result_begin();
          for (auto newRes : newOp->getResults()) {
            auto maybeLocalBuffer = getLocalBuffer(*scopeResIter);
            while (maybeLocalBuffer.has_value()) {
              for (auto &mapping : info.mappings)
                mapping.map(*scopeResIter,
                            mapping.lookup(maybeLocalBuffer.value()));
              ++scopeResIter;
              maybeLocalBuffer = getLocalBuffer(*scopeResIter);
            }
            for (auto &mapping : info.mappings)
              mapping.map(*scopeResIter, newRes);
            ++scopeResIter;
          }
          for (; scopeResIter != scopeOp.result_end(); ++scopeResIter) {
            auto maybeLocalBuffer = getLocalBuffer(*scopeResIter);
            for (auto &mapping : info.mappings)
              mapping.map(*scopeResIter,
                          mapping.lookup(maybeLocalBuffer.value()));
            ++scopeResIter;
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

  SmallVector<Value> newRes;
  for (auto [res, yield] :
       llvm::zip_equal(newForOp->getResults(), newForOp.getYieldedValues())) {
    if (auto maybeLocalBuffer = getLocalBuffer(yield);
        maybeLocalBuffer.has_value()) {
      newRes.push_back(maybeLocalBuffer.value());
    } else {
      newRes.push_back(res);
    }
  }
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

void CreatePreloadPass::runOnOperation() {
  auto moduleOp = getOperation();
  DenseMap<scf::ForOp, SmallVector<scope::ScopeOp, 4>> preload;
  DenseSet<int64_t> preloadNumSet;
  moduleOp->walk([&](scope::ScopeOp scopeOp) {
    if (auto maxPreloadNumAttr = scopeOp->getAttrOfType<IntegerAttr>(
            hivm::MaxPreloadNumAttr::name)) {
      auto parentForOp = cast<scf::ForOp>(scopeOp->getParentOp());
      preload[parentForOp].resize(maxPreloadNumAttr.getInt(), nullptr);
    }
    if (auto preloadNumAttr =
            scopeOp->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name)) {
      auto parentForOp = cast<scf::ForOp>(scopeOp->getParentOp());
      auto preloadNum = preloadNumAttr.getInt();
      auto &preloadVec = preload[parentForOp];
      assert(preloadNum >= 0 && "PreloadNum must be non-negative integer");
      assert(preloadNum < static_cast<int64_t>(preloadVec.size()) &&
             "MaxPreloadNumAttr must be set");
      preloadVec[preloadNum] = scopeOp;
      preloadNumSet.insert(preloadNum);
    }
  });

  if (preload.empty())
    return;

  for (auto &[forOp, scopes] : preload) {
    LDBG("Processing preload:\n" << forOp);
    scopes.resize(preloadNumSet.size(), nullptr);
    rewritePreloadLoop(forOp, scopes, preloadNumSet.size());
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

  if (failed(pm.run(moduleOp))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createCreatePreloadPass() {
  return std::make_unique<CreatePreloadPass>();
}

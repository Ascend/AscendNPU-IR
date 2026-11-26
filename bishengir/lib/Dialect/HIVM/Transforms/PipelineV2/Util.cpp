//===- Util.cpp - Common utilities for CV Pipelining V2 -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/GraphWriter.h"

#define DEBUG_TYPE "pipeline-v2-util"

namespace mlir {
namespace hivm {

bool hasLegacyCVPipelinedLoop(func::FuncOp funcOp) {
  if (!funcOp)
    return false;
  return funcOp
      .walk([](LoopLikeOpInterface loop) {
        if (loop->hasAttr(LegacyCVPipelineAttr::name))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

bool isPipelineReject(Operation *op) {
  for (Operation *curr = op; curr; curr = curr->getParentOp())
    if (curr->hasAttr(LegacyCVPipelineAttr::name))
      return true;
  return false;
}

bool isCoreOp(Operation *op) {
  if (!op || !op->getDialect())
    return false;
  return op->getDialect()->getNamespace() ==
             HIVMDialect::getDialectNamespace() &&
         isa<DestinationStyleOpInterface>(op);
}

std::optional<TCoreType> queryConcreteCoreType(Operation *op) {
  auto ct = detail::queryCoreTypeHelper(op);
  if (ct && (*ct == TCoreType::CUBE || *ct == TCoreType::VECTOR))
    return *ct;
  return std::nullopt;
}

std::optional<TCoreType> getScopeCoreType(ScopeOp scope) {
  if (!scope)
    return std::nullopt;
  if (auto attr =
          scope->getAttrOfType<TCoreTypeAttr>(kPipelinedLoopCoreTypeAttrName))
    return attr.getTcoretype();
  return std::nullopt;
}

Value traceValueDef(Value v) {
  if (!v)
    return nullptr;
  if (auto result = dyn_cast<OpResult>(v)) {
    Operation *defining = result.getOwner();
    Value srcVal =
        TypeSwitch<Operation *, Value>(defining)
            .Case<tensor::ReshapeOp, tensor::ExtractSliceOp,
                  tensor::CollapseShapeOp, tensor::ExpandShapeOp,
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
                  bufferization::ToTensorOp, bufferization::ToMemrefOp,
#else
                  bufferization::ToTensorOp, bufferization::ToBufferOp,
#endif
                  memref::CastOp, memref::CollapseShapeOp,
                  memref::ExpandShapeOp, memref::MemorySpaceCastOp,
                  memref::ReinterpretCastOp, memref::ReshapeOp, memref::ViewOp,
                  memref::SubViewOp>([](auto op) { return op->getOperand(0); })
            .Case([](tensor::InsertSliceOp insert) { return insert.getDest(); })
            .Case([result](LoopLikeOpInterface loop) {
              return loop.getTiedLoopInit(result)->get();
            })
            .Default([](Operation *) { return nullptr; });
    if (srcVal)
      return traceValueDef(srcVal);
    return result;
  }

  auto blkArg = dyn_cast<BlockArgument>(v);
  if (!blkArg) {
    LLVM_DEBUG(llvm::dbgs() << "[traceValueDef] expected block argument, got: "
                            << v << '\n');
    return nullptr;
  }
  Operation *parent = blkArg.getOwner()->getParentOp();
  auto loop = dyn_cast<LoopLikeOpInterface>(parent);
  if (!loop)
    return blkArg;
  OpOperand *init = loop.getTiedLoopInit(blkArg);
  return init ? traceValueDef(init->get()) : blkArg;
}

memref::AllocOp traceAlloc(Value v) {
  Value maybeAlloc = traceValueDef(v);
  return dyn_cast_if_present<memref::AllocOp>(maybeAlloc.getDefiningOp());
}

bool isCrossCoreCopy(Operation *copy) {
  auto copyOp = dyn_cast<CopyOp>(copy);
  if (!copyOp)
    return false;
  Value dst = copyOp.getDst();
  memref::AllocOp alloc = traceAlloc(dst);
  if (!alloc)
    return false;
  auto memSpaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(alloc.getType().getMemorySpace());
  if (!memSpaceAttr)
    return false;

  return memSpaceAttr.getAddressSpace() == AddressSpace::L1;
}

bool areCompatibleCoreTypes(TCoreType ct1, TCoreType ct2) {
  return !((ct1 == TCoreType::CUBE && ct2 == TCoreType::VECTOR) ||
           (ct1 == TCoreType::VECTOR && ct2 == TCoreType::CUBE));
}

FailureOr<TCoreType> resolveMergedCoreType(TCoreType ct1, TCoreType ct2) {
  if (!areCompatibleCoreTypes(ct1, ct2))
    return failure();
  if (ct1 == TCoreType::CUBE || ct2 == TCoreType::CUBE)
    return TCoreType::CUBE;
  if (ct1 == TCoreType::VECTOR || ct2 == TCoreType::VECTOR)
    return TCoreType::VECTOR;
  return ct1;
}

std::pair<Value, Value> getIfYieldedValues(scf::IfOp ifOp, unsigned resIdx) {
  auto thenYield = ifOp.thenYield();
  Value thenVal = (resIdx < thenYield.getNumOperands())
                      ? thenYield.getOperand(resIdx)
                      : Value();
  Value elseVal;
  if (ifOp.elseBlock()) {
    auto elseYield = ifOp.elseYield();
    if (resIdx < elseYield.getNumOperands())
      elseVal = elseYield.getOperand(resIdx);
  }
  return {thenVal, elseVal};
}

Value getOrCreateElseYieldValue(OpBuilder &builder, Location loc, Value thenVal,
                                Value elseVal) {
  if (elseVal)
    return elseVal;
  Type valType = thenVal.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(valType))
    return builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                           tensorType.getElementType());
  if (valType.isIndex())
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  if (auto intType = dyn_cast<IntegerType>(valType))
    return builder.create<arith::ConstantIntOp>(loc, 0, intType.getWidth());
  if (auto fltType = dyn_cast<FloatType>(valType))
    return builder.create<arith::ConstantFloatOp>(
        loc, fltType, APFloat(fltType.getFloatSemantics(), 0));
  return thenVal;
}

ScopeOp recreateScope(ScopeOp srcScope, ValueRange returnOperands,
                      Operation *insertBefore) {
  Block &block = srcScope.getRegion().front();
  if (block.empty() || !isa<scope::ReturnOp>(block.back())) {
    OpBuilder::atBlockEnd(&block).create<scope::ReturnOp>(srcScope.getLoc(),
                                                          returnOperands);
  } else {
    block.getTerminator()->setOperands(returnOperands);
  }
  OpBuilder builder(insertBefore ? insertBefore : srcScope.getOperation());
  auto newScope =
      builder.create<ScopeOp>(srcScope.getLoc(), TypeRange(returnOperands),
                              ValueRange(), srcScope->getAttrs());
  newScope.getRegion().takeBody(srcScope.getRegion());
  return newScope;
}

// NOTE: Moving individual ops across scopes is expensive; prefer merging scopes
// whenever possible.
void moveOpToScope(Operation *op, ScopeOp target, bool insertAtEnd) {
  if (!op || !target)
    return;

  if (op->getParentOfType<scf::IfOp>())
    llvm_unreachable("TODO: support moving op within scf.if");

  auto origScope = op->getParentOfType<ScopeOp>();
  if (!origScope || origScope == target)
    return;

  SmallPtrSet<Operation *, 16> opsToMoveSet = {op};
  if (isCoreOp(op)) {
    SmallVector<Operation *> worklist = {op};
    while (!worklist.empty()) {
      for (Value opnd : worklist.pop_back_val()->getOperands()) {
        Operation *defOp = opnd.getDefiningOp();
        if (!defOp || !origScope->isProperAncestor(defOp))
          continue;
        if (defOp->getParentOfType<scf::IfOp>())
          llvm_unreachable("TODO: support moving op within scf.if");
        if (!isCoreOp(defOp) && opsToMoveSet.insert(defOp).second)
          worklist.push_back(defOp);
      }
    }
  }

  Block &origBlock = origScope.getRegion().front();
  Block &targetBlock = target.getRegion().front();
  auto origReturn = cast<scope::ReturnOp>(origBlock.getTerminator());
  auto targetReturn = cast<scope::ReturnOp>(targetBlock.getTerminator());

  SmallVector<Operation *> orderedOpsToMove;
  for (Operation &currOp : origBlock) {
    if (opsToMoveSet.contains(&currOp))
      orderedOpsToMove.push_back(&currOp);
  }

  Operation *insertPt =
      insertAtEnd ? targetReturn.getOperation() : &targetBlock.front();

  IRMapping sharedPureMapping;
  SmallPtrSet<Operation *, 16> actuallyMovedOps;
  OpBuilder builder(insertPt);
  for (Operation *mOp : orderedOpsToMove) {
    bool hasRemainingUseInOrig =
        mOp != op && isMemoryEffectFree(mOp) &&
        llvm::any_of(mOp->getUsers(), [&](Operation *user) {
          return user != origReturn && origScope->isProperAncestor(user) &&
                 !opsToMoveSet.contains(user);
        });
    if (hasRemainingUseInOrig) {
      builder.clone(*mOp, sharedPureMapping);
      continue;
    }
    mOp->moveBefore(insertPt);
    for (OpOperand &opnd : mOp->getOpOperands()) {
      if (Value mapped = sharedPureMapping.lookupOrNull(opnd.get()))
        opnd.set(mapped);
    }
    actuallyMovedOps.insert(mOp);
  }

  for (Operation *mOp : actuallyMovedOps) {
    for (OpOperand &opnd : mOp->getOpOperands()) {
      if (auto res = dyn_cast<OpResult>(opnd.get());
          res && res.getOwner() == target.getOperation())
        opnd.set(targetReturn.getOperand(res.getResultNumber()));
    }
  }

  SmallVector<Value> newReturnOperands;
  SmallVector<std::pair<unsigned, unsigned>> keptResultMapping;
  DenseMap<Value, unsigned> valToNewResultIdx;
  bool scopeResultsChanged = false;

  for (auto [idx, retVal] : llvm::enumerate(origReturn.getOperands())) {
    if (Operation *retDef = retVal.getDefiningOp();
        retDef && actuallyMovedOps.contains(retDef)) {
      origScope.getResult(idx).replaceAllUsesWith(retVal);
      scopeResultsChanged = true;
      continue;
    }
    unsigned newIdx = newReturnOperands.size();
    newReturnOperands.push_back(retVal);
    keptResultMapping.emplace_back(idx, newIdx);
    valToNewResultIdx.try_emplace(retVal, newIdx);
  }

  SmallVector<std::pair<Value, unsigned>> depToNewResultIdx;
  for (Operation *mOp : orderedOpsToMove) {
    if (!actuallyMovedOps.contains(mOp))
      continue;
    for (Value opnd : mOp->getOperands()) {
      Operation *defOp = opnd.getDefiningOp();
      if (!defOp || !origScope->isProperAncestor(defOp))
        continue;
      auto [it, inserted] =
          valToNewResultIdx.try_emplace(opnd, newReturnOperands.size());
      if (inserted) {
        newReturnOperands.push_back(opnd);
        scopeResultsChanged = true;
      }
      depToNewResultIdx.emplace_back(opnd, it->second);
    }
  }

  ScopeOp activeOrigScope = origScope;
  if (scopeResultsChanged) {
    auto newOrigScope = recreateScope(origScope, newReturnOperands);
    for (auto [oldIdx, newIdx] : keptResultMapping) {
      origScope.getResult(oldIdx).replaceAllUsesWith(
          newOrigScope.getResult(newIdx));
    }
    origScope.erase();
    activeOrigScope = newOrigScope;
  }

  for (auto [depVal, newIdx] : depToNewResultIdx) {
    Value scopeRes = activeOrigScope.getResult(newIdx);
    depVal.replaceUsesWithIf(scopeRes, [&](OpOperand &use) {
      return target->isProperAncestor(use.getOwner());
    });
  }
}

/// Check the complete motion before changing either fragment. A matching
/// pipeline.if.cond identifies fragments of the same original conditional, but
/// previous scope merges may have placed other computations between them.
static bool canFuseIfFragments(scf::IfOp firstIf, scf::IfOp secondIf) {
  Block *firstBlock = firstIf->getBlock();
  auto hasUnavailableOperand = [&](Operation *op, bool allowIfResults) {
    return op
        ->walk([&](Operation *nested) {
          for (Value operand : nested->getOperands()) {
            Operation *def = operand.getDefiningOp();
            if (!def)
              continue;
            Operation *ancestor = firstBlock->findAncestorOpInBlock(*def);
            if (!ancestor || ancestor->isBeforeInBlock(firstIf))
              continue;
            // Direct uses of the first if's results in the second fragment's
            // branches are replaced with the corresponding branch yields.
            if (allowIfResults && def == firstIf.getOperation() &&
                secondIf->isProperAncestor(nested))
              continue;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        })
        .wasInterrupted();
  };

  // The second fragment will execute at firstIf's position. Do not cross
  // intervening effects in the first scope, even without an SSA dependency.
  for (Operation *op = firstIf->getNextNode();
       op && !op->hasTrait<OpTrait::IsTerminator>(); op = op->getNextNode())
    if (!isMemoryEffectFree(op))
      return false;

  bool firstIfIsPure = isMemoryEffectFree(firstIf);
  for (Operation &op : *secondIf->getBlock()) {
    if (&op == secondIf.getOperation())
      break;
    // Hoist the whole prefix in order or leave it untouched. Checking only
    // direct operands and moving a partial prefix can strand transitive or
    // region-captured dependencies after the fused conditional.
    if (hasUnavailableOperand(&op, /*allowIfResults=*/false) ||
        (!firstIfIsPure && !isMemoryEffectFree(&op)))
      return false;
  }
  return !hasUnavailableOperand(secondIf, /*allowIfResults=*/true);
}

ScopeOp mergeScopes(ScopeOp scope1, ScopeOp scope2, Operation *at) {
  if (!scope1 || !scope2)
    return ScopeOp();
  if (scope1 == scope2)
    return scope1;
  if (!scope1->getBlock() || scope1->getBlock() != scope2->getBlock())
    return ScopeOp();
  if (at && at->getBlock() != scope1->getBlock())
    return ScopeOp();

  auto ct1 = getScopeCoreType(scope1);
  auto ct2 = getScopeCoreType(scope2);
  if (!ct1 || !ct2)
    return ScopeOp();
  FailureOr<TCoreType> mergedCoreType = resolveMergedCoreType(*ct1, *ct2);
  if (failed(mergedCoreType))
    return ScopeOp();

  ScopeOp firstScope = scope1;
  ScopeOp secondScope = scope2;
  if (scope2->isBeforeInBlock(scope1))
    std::swap(firstScope, secondScope);

  Block &firstBlock = firstScope.getRegion().front();
  Block &secondBlock = secondScope.getRegion().front();
  auto firstReturn = cast<scope::ReturnOp>(firstBlock.getTerminator());
  auto secondReturn = cast<scope::ReturnOp>(secondBlock.getTerminator());

  // Redirect uses of firstScope's results inside secondScope to firstReturn's
  // operands.
  for (auto [retVal, res] :
       llvm::zip(firstReturn.getOperands(), firstScope.getResults())) {
    res.replaceUsesWithIf(retVal, [&](OpOperand &use) {
      return secondScope->isProperAncestor(use.getOwner());
    });
  }

  auto pruneUnusedResults = [](ScopeOp scope, scope::ReturnOp retOp) {
    SmallVector<Value> keptReturns, keptResults;
    for (auto [retVal, res] :
         llvm::zip(retOp.getOperands(), scope.getResults())) {
      if (!res.use_empty()) {
        keptReturns.push_back(retVal);
        keptResults.push_back(res);
      }
    }
    retOp->setOperands(keptReturns);
    return keptResults;
  };

  SmallVector<Value> keptFirstScopeResults =
      pruneUnusedResults(firstScope, firstReturn);
  SmallVector<Value> keptSecondScopeResults =
      pruneUnusedResults(secondScope, secondReturn);

  // Index scf.if operations in firstBlock by their pipeline.if.cond ID.
  DenseMap<int64_t, scf::IfOp> firstIfById;
  for (auto ifOp : firstBlock.getOps<scf::IfOp>()) {
    if (auto idAttr = ifOp->getAttrOfType<IntegerAttr>(kPipelineIfCondAttrName))
      firstIfById[idAttr.getInt()] = ifOp;
  }

  SmallVector<scf::IfOp> secondIfs(secondBlock.getOps<scf::IfOp>());
  for (scf::IfOp if2 : secondIfs) {
    auto idAttr = if2->getAttrOfType<IntegerAttr>(kPipelineIfCondAttrName);
    if (!idAttr)
      continue;
    scf::IfOp if1 = firstIfById.lookup(idAttr.getInt());
    if (!if1 || !canFuseIfFragments(if1, if2))
      continue;

    // The preflight proved that every prerequisite can move before if1.
    for (Operation &op : llvm::make_early_inc_range(secondBlock)) {
      if (&op == if2.getOperation())
        break;
      op.moveBefore(if1);
    }

    // Redirect uses of if1's results inside if2's branches to if1's branch
    // yield operands.
    for (auto [idx, res] : llvm::enumerate(if1.getResults())) {
      auto [thenVal, elseVal] = getIfYieldedValues(if1, idx);
      if (thenVal) {
        res.replaceUsesWithIf(thenVal, [&](OpOperand &use) {
          return if2.getThenRegion().isAncestor(
              use.getOwner()->getParentRegion());
        });
      }
      if (elseVal && if2.elseBlock()) {
        res.replaceUsesWithIf(elseVal, [&](OpOperand &use) {
          return if2.getElseRegion().isAncestor(
              use.getOwner()->getParentRegion());
        });
      }
    }

    SmallVector<Value> mergedThenYieldVals;
    SmallVector<Value> mergedElseYieldVals;
    SmallVector<Value> keptIfResults;

    for (scf::IfOp ifOp : {if1, if2}) {
      for (auto [idx, res] : llvm::enumerate(ifOp.getResults())) {
        if (res.use_empty())
          continue;
        auto [tVal, eVal] = getIfYieldedValues(ifOp, idx);
        keptIfResults.push_back(res);
        mergedThenYieldVals.push_back(tVal);
        mergedElseYieldVals.push_back(eVal);
      }
    }

    bool needElse = (if1.elseBlock() != nullptr) ||
                    (if2.elseBlock() != nullptr) || !keptIfResults.empty();

    OpBuilder ifBuilder(if1);
    auto mergedIf = ifBuilder.create<scf::IfOp>(
        if1.getLoc(), TypeRange(ValueRange(keptIfResults)), if1.getCondition(),
        needElse);
    mergedIf->setAttrs(if1->getAttrs());

    if (!mergedIf.thenBlock()->empty())
      mergedIf.thenBlock()->getTerminator()->erase();
    if (mergedIf.elseBlock() && !mergedIf.elseBlock()->empty())
      mergedIf.elseBlock()->getTerminator()->erase();

    auto moveBranchOps = [](Block *srcBlock, Block *dstBlock) {
      if (!srcBlock)
        return;
      for (Operation &op : llvm::make_early_inc_range(*srcBlock)) {
        if (!isa<scf::YieldOp>(&op))
          op.moveBefore(dstBlock, dstBlock->end());
      }
    };

    moveBranchOps(if1.thenBlock(), mergedIf.thenBlock());
    moveBranchOps(if2.thenBlock(), mergedIf.thenBlock());
    ifBuilder.setInsertionPointToEnd(mergedIf.thenBlock());
    ifBuilder.create<scf::YieldOp>(if1.thenYield().getLoc(),
                                   mergedThenYieldVals);

    if (needElse) {
      moveBranchOps(if1.elseBlock(), mergedIf.elseBlock());
      moveBranchOps(if2.elseBlock(), mergedIf.elseBlock());
      ifBuilder.setInsertionPointToEnd(mergedIf.elseBlock());
      Location elseLoc = if1.elseBlock()   ? if1.elseYield().getLoc()
                         : if2.elseBlock() ? if2.elseYield().getLoc()
                                           : if1.getLoc();
      SmallVector<Value> finalElseYieldVals;
      for (auto [thenVal, elseVal] :
           llvm::zip(mergedThenYieldVals, mergedElseYieldVals)) {
        finalElseYieldVals.push_back(
            getOrCreateElseYieldValue(ifBuilder, elseLoc, thenVal, elseVal));
      }
      ifBuilder.create<scf::YieldOp>(elseLoc, finalElseYieldVals);
    }

    for (auto [oldRes, newRes] :
         llvm::zip(keptIfResults, mergedIf.getResults()))
      oldRes.replaceAllUsesWith(newRes);

    firstIfById[idAttr.getInt()] = mergedIf;
    if1.erase();
    if2.erase();
  }

  // Move all remaining non-terminator ops from secondBlock into firstBlock
  // before firstReturn.
  for (Operation &op : llvm::make_early_inc_range(secondBlock)) {
    if (&op != secondReturn.getOperation())
      op.moveBefore(firstReturn);
  }

  SmallVector<Value> mergedReturnOperands(firstReturn.getOperands());
  llvm::append_range(mergedReturnOperands, secondReturn.getOperands());
  SmallVector<Value> oldScopeResultsToReplace(keptFirstScopeResults);
  llvm::append_range(oldScopeResultsToReplace, keptSecondScopeResults);

  auto mergedScope = recreateScope(firstScope, mergedReturnOperands, at);
  mergedScope->setAttr(
      kPipelinedLoopCoreTypeAttrName,
      TCoreTypeAttr::get(mergedScope.getContext(), *mergedCoreType));

  for (auto [oldRes, newRes] :
       llvm::zip(oldScopeResultsToReplace, mergedScope.getResults()))
    oldRes.replaceAllUsesWith(newRes);

  firstScope.erase();
  secondScope.erase();
  return mergedScope;
}

namespace {

static bool regionHasScope(Region &region) {
  return region
      .walk<WalkOrder::PreOrder>([](Operation *op) {
        if (isa<scf::ForOp>(op))
          return WalkResult::skip();
        if (isa<ScopeOp>(op))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static std::string getScopeKindLabel(ScopeOp scope) {
  std::string kind;
  if (auto ct = getScopeCoreType(scope))
    kind = stringifyTCoreType(*ct).str();
  else if (scope->hasAttr(kCVMergedScopeAttrName))
    kind = kCVMergedScopeAttrName.str();
  else
    kind = "untyped";
  if (auto preload = scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name))
    kind += llvm::formatv(", preload={0}", preload.getInt()).str();
  return kind;
}

struct ScopeDepNode;
struct ScopeCluster;

struct ScopeDepEdge {
  const ScopeDepNode *target = nullptr;
  SmallVector<std::string, 2> labels;
  bool isLoopCarried = false;

  std::string getJoinedLabel() const { return llvm::join(labels, ", "); }
};

struct ScopeDepNode {
  unsigned id = 0;
  ScopeOp scopeOp;
  std::optional<TCoreType> coreType;
  SmallVector<const ScopeDepNode *> children;
  SmallVector<ScopeDepEdge> edges;

  void addEdge(const ScopeDepNode *dst, StringRef edgeLabel, bool loopCarried) {
    for (ScopeDepEdge &existing : edges) {
      if (existing.target == dst && existing.isLoopCarried == loopCarried) {
        if (!edgeLabel.empty() &&
            !llvm::is_contained(existing.labels, edgeLabel))
          existing.labels.push_back(edgeLabel.str());
        return;
      }
    }
    children.push_back(dst);
    ScopeDepEdge newEdge;
    newEdge.target = dst;
    if (!edgeLabel.empty())
      newEdge.labels.push_back(edgeLabel.str());
    newEdge.isLoopCarried = loopCarried;
    edges.push_back(std::move(newEdge));
  }
};

struct ScopeClusterChild {
  const ScopeDepNode *node = nullptr;
  const ScopeCluster *cluster = nullptr;

  static ScopeClusterChild fromNode(const ScopeDepNode *n) {
    return {n, nullptr};
  }
  static ScopeClusterChild fromCluster(const ScopeCluster *c) {
    return {nullptr, c};
  }
};

struct ScopeCluster {
  unsigned id = 0;
  ScopeOp scopeOp;
  std::optional<TCoreType> coreType;
  SmallVector<ScopeClusterChild> children;
};

struct DOTScopeDepGraphInfo {
  scf::ForOp forOp;
  std::string funcName;
  std::unique_ptr<AsmState> asmState;
  SmallVector<std::unique_ptr<ScopeDepNode>> nodes;
  SmallVector<const ScopeDepNode *> nodePtrs;
  SmallVector<std::unique_ptr<ScopeCluster>> clusters;
  SmallVector<const ScopeCluster *> topLevelClusters;

  std::string formatValue(Value val) const {
    std::string str;
    llvm::raw_string_ostream os(str);
    val.printAsOperand(os, *asmState);
    return os.str();
  }

  explicit DOTScopeDepGraphInfo(scf::ForOp loop) : forOp(loop) {
    if (auto func = loop->getParentOfType<func::FuncOp>())
      funcName = func.getName().str();
    else
      funcName = "loop";

    Operation *asmRoot = loop.getOperation();
    while (Operation *parent = asmRoot->getParentOp())
      asmRoot = parent;
    asmState = std::make_unique<AsmState>(asmRoot);

    DenseMap<Operation *, ScopeDepNode *> leafScopeToNode;
    std::function<void(Region &, ScopeCluster *)> visitRegion =
        [&](Region &region, ScopeCluster *parentCluster) {
          for (Block &block : region) {
            for (Operation &op : block) {
              if (isa<scf::ForOp>(op))
                continue;
              if (auto scope = dyn_cast<ScopeOp>(op)) {
                if (regionHasScope(scope.getRegion())) {
                  auto cluster = std::make_unique<ScopeCluster>();
                  cluster->id = clusters.size();
                  cluster->scopeOp = scope;
                  cluster->coreType = getScopeCoreType(scope);
                  ScopeCluster *clusterPtr = cluster.get();
                  clusters.push_back(std::move(cluster));
                  if (parentCluster)
                    parentCluster->children.push_back(
                        ScopeClusterChild::fromCluster(clusterPtr));
                  else
                    topLevelClusters.push_back(clusterPtr);
                  visitRegion(scope.getRegion(), clusterPtr);
                } else {
                  auto node = std::make_unique<ScopeDepNode>();
                  node->id = nodes.size();
                  node->scopeOp = scope;
                  node->coreType = getScopeCoreType(scope);
                  ScopeDepNode *nodePtr = node.get();
                  leafScopeToNode[scope.getOperation()] = nodePtr;
                  nodePtrs.push_back(nodePtr);
                  nodes.push_back(std::move(node));
                  if (parentCluster)
                    parentCluster->children.push_back(
                        ScopeClusterChild::fromNode(nodePtr));
                }
              } else {
                for (Region &subRegion : op.getRegions())
                  visitRegion(subRegion, parentCluster);
              }
            }
          }
        };

    visitRegion(loop.getRegion(), /*parentCluster=*/nullptr);

    auto findEnclosingLeafScopeNode = [&](Operation *op) -> ScopeDepNode * {
      while (op && op != loop.getOperation()) {
        if (auto *n = leafScopeToNode.lookup(op))
          return n;
        op = op->getParentOp();
      }
      return nullptr;
    };

    auto yieldOp = cast<scf::YieldOp>(loop.getBody()->getTerminator());

    auto collectConsumerLeafNodes =
        [&](Value rootVal, SmallVectorImpl<ScopeDepNode *> &consumers) {
          SmallVector<Value, 8> worklist = {rootVal};
          SmallPtrSet<Value, 8> visited = {rootVal};
          SmallPtrSet<const ScopeDepNode *, 8> seenNodes;
          auto pushVal = [&](Value v) {
            if (v && visited.insert(v).second)
              worklist.push_back(v);
          };

          while (!worklist.empty()) {
            Value cur = worklist.pop_back_val();
            for (OpOperand &use : cur.getUses()) {
              Operation *user = use.getOwner();
              if (user == yieldOp.getOperation())
                continue;
              if (ScopeDepNode *dst = findEnclosingLeafScopeNode(user)) {
                if (seenNodes.insert(dst).second)
                  consumers.push_back(dst);
                continue;
              }
              if (!loop->isProperAncestor(user))
                continue;
              if (isa<scope::ReturnOp>(user)) {
                if (auto parentScope = dyn_cast<ScopeOp>(user->getParentOp());
                    parentScope && loop->isProperAncestor(parentScope)) {
                  unsigned idx = use.getOperandNumber();
                  if (idx < parentScope.getNumResults())
                    pushVal(parentScope.getResult(idx));
                }
              } else if (isa<scf::YieldOp>(user)) {
                if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp());
                    ifOp && loop->isProperAncestor(ifOp)) {
                  unsigned idx = use.getOperandNumber();
                  if (idx < ifOp.getNumResults())
                    pushVal(ifOp.getResult(idx));
                }
              } else if (auto ifOp = dyn_cast<scf::IfOp>(user)) {
                bool hasNestedScope = false;
                for (const auto &cand : nodes) {
                  if (ifOp->isProperAncestor(cand->scopeOp)) {
                    hasNestedScope = true;
                    if (seenNodes.insert(cand.get()).second)
                      consumers.push_back(cand.get());
                  }
                }
                if (!hasNestedScope) {
                  for (Value r : ifOp.getResults())
                    pushVal(r);
                }
              } else if (!isa<ScopeOp, scf::ForOp>(user)) {
                for (Value r : user->getResults())
                  pushVal(r);
              }
            }
          }
        };

    // Precompute consumer leaf scopes for each loop iter_arg.
    SmallVector<SmallVector<ScopeDepNode *>> iterArgConsumers(
        loop.getNumRegionIterArgs());
    for (unsigned argIdx = 0; argIdx < loop.getNumRegionIterArgs(); ++argIdx)
      collectConsumerLeafNodes(loop.getRegionIterArg(argIdx),
                               iterArgConsumers[argIdx]);

    // Trace SSA values produced by each leaf scope across nested wrapper
    // scopes, scf.if regions, and intervening non-scope operations:
    // 1. `scope result` -> ... -> `user leaf scope`
    // 2. `scope result` -> ... -> `scf.yield` -> `iter_arg` -> `user leaf
    // scope`
    for (const auto &srcNode : nodes) {
      SmallVector<Value, 8> worklist(srcNode->scopeOp.result_begin(),
                                     srcNode->scopeOp.result_end());
      SmallPtrSet<Value, 8> visited(worklist.begin(), worklist.end());
      auto pushVal = [&](Value v) {
        if (v && visited.insert(v).second)
          worklist.push_back(v);
      };

      while (!worklist.empty()) {
        Value curVal = worklist.pop_back_val();
        std::string curValName;
        auto getCurValName = [&]() -> const std::string & {
          if (curValName.empty())
            curValName = formatValue(curVal);
          return curValName;
        };

        for (OpOperand &use : curVal.getUses()) {
          Operation *user = use.getOwner();
          if (user == yieldOp.getOperation()) {
            unsigned argIdx = use.getOperandNumber();
            if (argIdx >= loop.getNumRegionIterArgs())
              continue;
            std::string label =
                llvm::formatv("LCD({0}->{1})", getCurValName(),
                              formatValue(loop.getRegionIterArg(argIdx)));
            for (ScopeDepNode *dstNode : iterArgConsumers[argIdx])
              srcNode->addEdge(dstNode, label, /*loopCarried=*/true);
            continue;
          }

          if (ScopeDepNode *dstNode = findEnclosingLeafScopeNode(user)) {
            if (dstNode != srcNode.get())
              srcNode->addEdge(dstNode, getCurValName(), /*loopCarried=*/false);
            continue;
          }

          if (!loop->isProperAncestor(user))
            continue;

          if (isa<scope::ReturnOp>(user)) {
            if (auto parentScope = dyn_cast<ScopeOp>(user->getParentOp());
                parentScope && loop->isProperAncestor(parentScope)) {
              unsigned idx = use.getOperandNumber();
              if (idx < parentScope.getNumResults())
                pushVal(parentScope.getResult(idx));
            }
          } else if (isa<scf::YieldOp>(user)) {
            if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp());
                ifOp && loop->isProperAncestor(ifOp)) {
              unsigned idx = use.getOperandNumber();
              if (idx < ifOp.getNumResults())
                pushVal(ifOp.getResult(idx));
            }
          } else if (auto ifOp = dyn_cast<scf::IfOp>(user)) {
            bool hasNestedScope = false;
            for (const auto &cand : nodes) {
              if (ifOp->isProperAncestor(cand->scopeOp)) {
                hasNestedScope = true;
                if (cand.get() != srcNode.get())
                  srcNode->addEdge(cand.get(), getCurValName(),
                                   /*loopCarried=*/false);
              }
            }
            if (!hasNestedScope) {
              for (Value r : ifOp.getResults())
                pushVal(r);
            }
          } else if (!isa<ScopeOp, scf::ForOp>(user)) {
            for (Value r : user->getResults())
              pushVal(r);
          }
        }
      }
    }
  }
};

} // namespace

} // namespace hivm
} // namespace mlir

namespace llvm {

using mlir::hivm::DOTScopeDepGraphInfo;
using mlir::hivm::getScopeKindLabel;
using mlir::hivm::ScopeCluster;
using mlir::hivm::ScopeClusterChild;
using mlir::hivm::ScopeDepEdge;
using mlir::hivm::ScopeDepNode;

template <> struct GraphTraits<DOTScopeDepGraphInfo *> {
  using NodeRef = const ScopeDepNode *;
  using ChildIteratorType =
      SmallVectorImpl<const ScopeDepNode *>::const_iterator;
  using nodes_iterator = SmallVectorImpl<const ScopeDepNode *>::const_iterator;

  static NodeRef getEntryNode(DOTScopeDepGraphInfo *G) {
    return G->nodePtrs.empty() ? nullptr : G->nodePtrs.front();
  }

  static ChildIteratorType child_begin(NodeRef N) {
    return N->children.begin();
  }
  static ChildIteratorType child_end(NodeRef N) { return N->children.end(); }

  static nodes_iterator nodes_begin(DOTScopeDepGraphInfo *G) {
    return G->nodePtrs.begin();
  }
  static nodes_iterator nodes_end(DOTScopeDepGraphInfo *G) {
    return G->nodePtrs.end();
  }
  static size_t size(DOTScopeDepGraphInfo *G) { return G->nodePtrs.size(); }
};

template <>
struct DOTGraphTraits<DOTScopeDepGraphInfo *> : public DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool isSimple = false)
      : DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(DOTScopeDepGraphInfo *G) {
    return "Scope Dependence Graph for '" + G->funcName + "'";
  }

  static std::string getSimpleNodeLabel(const ScopeDepNode *Node,
                                        DOTScopeDepGraphInfo *) {
    return formatv("Scope {0} ({1})", Node->id,
                   getScopeKindLabel(Node->scopeOp))
        .str();
  }

  static void printScopeBlock(raw_string_ostream &OS, const ScopeDepNode &Node,
                              const DOTScopeDepGraphInfo *G) {
    mlir::scope::ScopeOp scopeOp = Node.scopeOp;
    OS << formatv("Scope {0} ({1})", Node.id, getScopeKindLabel(scopeOp));
    if (scopeOp.getNumResults() > 0) {
      OS << " -> ";
      llvm::interleaveComma(scopeOp.getResults(), OS, [&](mlir::Value res) {
        OS << G->formatValue(res);
      });
    }
    OS << ":\n";
    for (mlir::Operation &op : scopeOp.getRegion().front()) {
      op.print(OS, *G->asmState);
      OS << "\n";
    }
  }

  static void noOpHandleComment(std::string &, unsigned &, unsigned) {}

  std::string getNodeLabel(const ScopeDepNode *Node, DOTScopeDepGraphInfo *G) {
    if (isSimple())
      return getSimpleNodeLabel(Node, G);
    return CompleteNodeLabelString(
        Node,
        function_ref<void(raw_string_ostream &, const ScopeDepNode &)>(
            [G](raw_string_ostream &OS, const ScopeDepNode &N) {
              printScopeBlock(OS, N, G);
            }),
        function_ref<void(std::string &, unsigned &, unsigned)>(
            noOpHandleComment));
  }

  static std::string getNodeAttributes(const ScopeDepNode *Node,
                                       DOTScopeDepGraphInfo *) {
    StringRef fillColor = "#e6e6e6";
    if (Node->coreType == mlir::hivm::TCoreType::CUBE)
      fillColor = "#ffcccc";
    else if (Node->coreType == mlir::hivm::TCoreType::VECTOR)
      fillColor = "#cce5ff";
    return ("style=filled, fillcolor=\"" + fillColor +
            "\", fontname=\"Courier\"")
        .str();
  }

  static std::string
  getEdgeAttributes(const ScopeDepNode *Node,
                    GraphTraits<DOTScopeDepGraphInfo *>::ChildIteratorType EI,
                    DOTScopeDepGraphInfo *) {
    size_t idx = std::distance(Node->children.begin(), EI);
    const ScopeDepEdge &edge = Node->edges[idx];
    std::string attrs =
        formatv("tooltip=\"Scope {0} -> Scope {1}\"", Node->id, edge.target->id)
            .str();
    std::string edgeLabel = edge.getJoinedLabel();
    if (!edgeLabel.empty())
      attrs += formatv(" label=\"{0}\"", DOT::EscapeString(edgeLabel)).str();
    if (edge.isLoopCarried) {
      attrs += " style=dashed color=red penwidth=2";
      if (Node->id >= edge.target->id)
        attrs += " constraint=false";
    }
    return attrs;
  }

  static void printCluster(const ScopeCluster &cluster,
                           const DOTScopeDepGraphInfo *G,
                           GraphWriter<DOTScopeDepGraphInfo *> &GW,
                           unsigned indent = 1) {
    raw_ostream &O = GW.getOStream();
    mlir::scope::ScopeOp scopeOp = cluster.scopeOp;
    O.indent(indent * 2) << "subgraph cluster_scope_" << cluster.id << " {\n";
    std::string label =
        formatv("Outer Scope {0} ({1})", cluster.id, getScopeKindLabel(scopeOp))
            .str();
    if (scopeOp.getNumResults() > 0 && !scopeOp.getRegion().empty()) {
      if (auto retOp = dyn_cast<mlir::scope::ReturnOp>(
              scopeOp.getRegion().front().getTerminator())) {
        std::string bindings;
        raw_string_ostream bOS(bindings);
        llvm::interleaveComma(
            llvm::zip(scopeOp.getResults(), retOp.getOperands()), bOS,
            [&](auto pair) {
              bOS << G->formatValue(std::get<0>(pair)) << " = "
                  << G->formatValue(std::get<1>(pair));
            });
        label += "\n" + bOS.str();
      }
    }
    O.indent((indent + 1) * 2)
        << "label = \"" << DOT::EscapeString(label) << "\";\n";
    O.indent((indent + 1) * 2) << "style = \"rounded,dashed,filled\";\n";
    O.indent((indent + 1) * 2) << "color = \"#666666\";\n";
    O.indent((indent + 1) * 2) << "fillcolor = \"#f8f8f8\";\n";
    O.indent((indent + 1) * 2) << "fontname = \"Courier\";\n";
    for (const ScopeClusterChild &child : cluster.children) {
      if (child.cluster) {
        printCluster(*child.cluster, G, GW, indent + 1);
      } else if (child.node) {
        O.indent((indent + 1) * 2)
            << "Node" << static_cast<const void *>(child.node) << ";\n";
      }
    }
    O.indent(indent * 2) << "}\n";
  }

  static void addCustomGraphFeatures(DOTScopeDepGraphInfo *const &G,
                                     GraphWriter<DOTScopeDepGraphInfo *> &GW) {
    for (const ScopeCluster *cluster : G->topLevelClusters)
      printCluster(*cluster, G, GW);
  }
};

} // namespace llvm

namespace mlir {
namespace hivm {

void writeScopeDependenceGraphToDotFile(scf::ForOp forOp, StringRef prefix,
                                        StringRef filename, bool shortNames) {
  if (!forOp)
    return;
  DOTScopeDepGraphInfo graphInfo(forOp);
  std::string outFilename;
  if (!filename.empty()) {
    outFilename =
        prefix.empty() ? filename.str() : (prefix + "." + filename).str();
  } else {
    Operation *rootOp = forOp->getParentOfType<func::FuncOp>();
    if (!rootOp)
      rootOp = forOp->getParentOp();
    SmallVector<scf::ForOp> scopedLoops;
    if (rootOp) {
      rootOp->walk([&](scf::ForOp loop) {
        if (regionHasScope(loop.getRegion()))
          scopedLoops.push_back(loop);
      });
    }
    std::string baseName = prefix.empty()
                               ? graphInfo.funcName
                               : (prefix + "." + graphInfo.funcName).str();
    if (scopedLoops.size() > 1) {
      auto it = llvm::find(scopedLoops, forOp);
      unsigned idx = (it != scopedLoops.end())
                         ? std::distance(scopedLoops.begin(), it)
                         : 0;
      outFilename = llvm::formatv("{0}.{1}.dot", baseName, idx).str();
    } else {
      outFilename = baseName + ".dot";
    }
  }
  llvm::errs() << "Writing '" << outFilename << "'...";

  std::error_code ec;
  llvm::raw_fd_ostream file(outFilename, ec, llvm::sys::fs::OF_Text);
  if (!ec)
    llvm::WriteGraph(file, &graphInfo, shortNames);
  else
    llvm::errs() << "  error opening file for writing!";
  llvm::errs() << "\n";
}

} // namespace hivm
} // namespace mlir

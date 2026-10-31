//===- OutlineCVScopes.cpp - Outline CV Scopes for CV Pipelining V2 -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Util.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "outline-cv-scopes"

namespace mlir {
namespace hivm {

using bishengir::memref_ext::AllocWorkspaceOp;

#define GEN_PASS_DEF_OUTLINECVSCOPES
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

namespace {

struct ConditionalOps {
  unsigned id = 0;
  scf::IfOp ifOp;
  Value condition;
  SmallPtrSet<Operation *, 16> thenOps;
  SmallPtrSet<Operation *, 16> elseOps;
  SmallPtrSet<Value, 8> outputs;
};

struct WorkItem {
  unsigned id = 0;
  TCoreType coreType = TCoreType::CUBE_OR_VECTOR;
  Value output = nullptr;
  SmallPtrSet<Value, 8> inputs;
  SmallPtrSet<Value, 8> outputs;
  SmallPtrSet<Operation *, 32> ops;
  SetVector<Operation *> orderedOps;
  DenseMap<Operation *, ConditionalOps> condOps;
  SmallPtrSet<Operation *, 16> hoistedOps;
};

class CVScopeOutliner {
public:
  explicit CVScopeOutliner(Operation *loopOp)
      : loopOp(loopOp), ctx(loopOp->getContext()) {}

  LogicalResult run();

private:
  // Core classification & predicates
  static bool isViewLikeMemRefOp(Operation *op);
  static Operation *getRootAlloc(Value memref);
  static bool isStoreFixpipeCopyWriter(Operation *usr, Value memrefVal);
  static bool walkAllocUsers(Operation *rootAlloc,
                             llvm::function_ref<bool(OpOperand &)> visitUse);
  static bool isAllocTargetOfStoreFixpipeCopy(Operation *rootAlloc);

  bool isStoppingToTensor(Operation *tracedOp, Value startOutput);
  TCoreType determineCoreType(Value output);
  bool canCloneOpInWorkItem(Operation *op, WorkItem *item) const;
  bool isIfResultOwnedByItem(scf::IfOp ifOp, unsigned resIdx,
                             WorkItem *item) const;

  // Analysis steps
  unsigned getOrAssignIfId(scf::IfOp ifOp);
  ConditionalOps &getOrCreateCondOps(WorkItem &item, scf::IfOp ifOp,
                                     Operation *branchOp = nullptr);
  bool hasMultipleNestedIf();
  bool hasIllegalRegionedOp();
  void addLocalOutput(Value val);
  void collectLocalOutputs();
  LogicalResult buildWorkItems();
  LogicalResult traceWorkItem(Value output, WorkItem &item, bool &failedLoop);
  LogicalResult sortWorkItems();
  void mapOpsToWorkItems();
  void collectUnmappedOps();
  void computeHoistedOps(WorkItem *item);
  void populateWorkItemIO(Operation *origLoopTerminator);

  // IR rewriting steps
  LogicalResult rewriteForOp(scf::ForOp origForOp);

  Operation *loopOp;
  MLIRContext *ctx;

  DenseMap<Operation *, unsigned> opOrder;
  DenseMap<Operation *, unsigned> ifOpToId;
  DenseMap<Value, Value> valToOutputAlias;
  SetVector<Value> localOutputs;
  SmallVector<std::unique_ptr<WorkItem>> workItems;
  SmallVector<WorkItem *> sortedWorkItems;
  DenseMap<Operation *, WorkItem *> opToWorkItemMap;
  unsigned nextWorkItemId = 0;
  unsigned nextIfId = 0;
};

struct OutlineCVScopesPass
    : public impl::OutlineCVScopesBase<OutlineCVScopesPass> {
protected:
  void runOnOperation() override;
};

} // namespace

//===----------------------------------------------------------------------===//
// CVScopeOutliner Implementation
//===----------------------------------------------------------------------===//

bool CVScopeOutliner::isViewLikeMemRefOp(Operation *op) {
  return op && isa<ViewLikeOpInterface, memref::SubViewOp, memref::ViewOp,
                   memref::CastOp, memref::ReinterpretCastOp,
                   memref::MemorySpaceCastOp, memref::CollapseShapeOp,
                   memref::ExpandShapeOp>(op);
}

Operation *CVScopeOutliner::getRootAlloc(Value memref) {
  for (Value curr = memref; curr;) {
    Operation *def = curr.getDefiningOp();
    if (!def)
      return nullptr;
    if (isa<memref::AllocOp, AllocWorkspaceOp>(def))
      return def;
    if (!isViewLikeMemRefOp(def) || def->getNumOperands() == 0)
      break;
    curr = def->getOperand(0);
  }
  return nullptr;
}

bool CVScopeOutliner::isStoreFixpipeCopyWriter(Operation *usr,
                                               Value memrefVal) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(usr))
    return copyOp.getTarget() == memrefVal;
  if (!isa<StoreOp, FixpipeOp, CopyOp>(usr))
    return false;
  auto dps = dyn_cast<DestinationStyleOpInterface>(usr);
  return dps && llvm::any_of(dps.getDpsInits(),
                             [&](Value init) { return init == memrefVal; });
}

bool CVScopeOutliner::walkAllocUsers(
    Operation *rootAlloc, llvm::function_ref<bool(OpOperand &)> visitUse) {
  if (!rootAlloc)
    return false;
  SmallVector<Value> queue(rootAlloc->getResults());
  SmallPtrSet<Value, 16> visited(queue.begin(), queue.end());
  while (!queue.empty()) {
    Value curr = queue.pop_back_val();
    for (OpOperand &use : curr.getUses()) {
      if (visitUse(use))
        return true;
      Operation *usr = use.getOwner();
      if (!isViewLikeMemRefOp(usr))
        continue;
      for (Value res : usr->getResults()) {
        if (visited.insert(res).second)
          queue.push_back(res);
      }
    }
  }
  return false;
}

bool CVScopeOutliner::isAllocTargetOfStoreFixpipeCopy(Operation *rootAlloc) {
  return walkAllocUsers(rootAlloc, [](OpOperand &use) {
    return isStoreFixpipeCopyWriter(use.getOwner(), use.get());
  });
}

bool CVScopeOutliner::isStoppingToTensor(Operation *tracedOp,
                                         Value startOutput) {
  auto toTensor = dyn_cast_or_null<bufferization::ToTensorOp>(tracedOp);
  if (!toTensor || toTensor.getResult() == startOutput)
    return false;
  return isAllocTargetOfStoreFixpipeCopy(getRootAlloc(toTensor.getOperand()));
}

TCoreType CVScopeOutliner::determineCoreType(Value output) {
  Operation *defOp = output.getDefiningOp();
  if (!defOp)
    return TCoreType::CUBE_AND_VECTOR;

  if (auto ct = queryConcreteCoreType(defOp))
    return *ct;

  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    auto [tVal, eVal] =
        getIfYieldedValues(ifOp, cast<OpResult>(output).getResultNumber());
    for (Value branchVal : {tVal, eVal}) {
      if (Operation *bDef = branchVal ? branchVal.getDefiningOp() : nullptr;
          bDef && ifOp->isProperAncestor(bDef))
        return determineCoreType(branchVal);
    }
  }

  SmallVector<Operation *> stack = {defOp};
  SmallPtrSet<Operation *, 16> visited = {defOp};

  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(defOp)) {
    std::optional<TCoreType> foundCore;
    walkAllocUsers(getRootAlloc(toTensor.getOperand()), [&](OpOperand &use) {
      Operation *usr = use.getOwner();
      if (!isCoreOp(usr))
        return false;
      if (auto ct = queryConcreteCoreType(usr)) {
        foundCore = *ct;
        return true;
      }
      if (visited.insert(usr).second)
        stack.push_back(usr);
      return false;
    });
    if (foundCore)
      return *foundCore;
  }

  while (!stack.empty()) {
    Operation *curr = stack.pop_back_val();
    if (isCoreOp(curr)) {
      if (auto ct = queryConcreteCoreType(curr))
        return *ct;
    }
    auto pushDefs = [&](ValueRange vals) {
      for (Value opnd : vals) {
        if (Operation *p = opnd.getDefiningOp(); p && visited.insert(p).second)
          stack.push_back(p);
      }
    };
    if (auto ifOp = dyn_cast<scf::IfOp>(curr)) {
      pushDefs(ifOp.thenYield().getOperands());
      if (ifOp.elseBlock())
        pushDefs(ifOp.elseYield().getOperands());
    }
    pushDefs(curr->getOperands());
  }
  return TCoreType::CUBE_AND_VECTOR;
}

bool CVScopeOutliner::canCloneOpInWorkItem(Operation *op,
                                           WorkItem *item) const {
  if (item && opToWorkItemMap.lookup(op) == item)
    return true;
  if (isCoreOp(op) || isa<memref::AllocOp, AllocWorkspaceOp,
                          bufferization::ToTensorOp, annotation::MarkOp>(op))
    return false;
  if (isViewLikeMemRefOp(op) && op->getNumOperands() > 0 &&
      getRootAlloc(op->getOperand(0)))
    return false;
  return isa<tensor::EmptyOp>(op) ||
         llvm::none_of(op->getResultTypes(), llvm::IsaPred<TensorType>);
}

bool CVScopeOutliner::isIfResultOwnedByItem(scf::IfOp ifOp, unsigned resIdx,
                                            WorkItem *item) const {
  auto [tVal, eVal] = getIfYieldedValues(ifOp, resIdx);
  Operation *tDef = tVal ? tVal.getDefiningOp() : nullptr;
  Operation *eDef = eVal ? eVal.getDefiningOp() : nullptr;
  bool tInIf = tDef && ifOp->isProperAncestor(tDef);
  bool eInIf = eDef && ifOp->isProperAncestor(eDef);
  if (tInIf || eInIf)
    return (tInIf && opToWorkItemMap.lookup(tDef) == item) ||
           (eInIf && opToWorkItemMap.lookup(eDef) == item);
  return (tDef && opToWorkItemMap.lookup(tDef) == item) ||
         (eDef && opToWorkItemMap.lookup(eDef) == item);
}

unsigned CVScopeOutliner::getOrAssignIfId(scf::IfOp ifOp) {
  auto [it, inserted] = ifOpToId.try_emplace(ifOp.getOperation(), nextIfId);
  if (inserted)
    ++nextIfId;
  return it->second;
}

ConditionalOps &CVScopeOutliner::getOrCreateCondOps(WorkItem &item,
                                                    scf::IfOp ifOp,
                                                    Operation *branchOp) {
  auto &cOps = item.condOps[ifOp];
  cOps.id = getOrAssignIfId(ifOp);
  cOps.ifOp = ifOp;
  cOps.condition = ifOp.getCondition();
  if (branchOp) {
    if (branchOp->getBlock() == ifOp.thenBlock())
      cOps.thenOps.insert(branchOp);
    else if (ifOp.elseBlock() && branchOp->getBlock() == ifOp.elseBlock())
      cOps.elseOps.insert(branchOp);
  }
  return cOps;
}

bool CVScopeOutliner::hasMultipleNestedIf() {
  return loopOp
      ->walk([&](scf::IfOp ifOp) {
        return ifOp->getParentOfType<scf::IfOp>() ? WalkResult::interrupt()
                                                  : WalkResult::advance();
      })
      .wasInterrupted();
}

bool CVScopeOutliner::hasIllegalRegionedOp() {
  for (Region &reg : loopOp->getRegions()) {
    for (Block &block : reg) {
      for (Operation &op : block) {
        if (op.getNumRegions() > 0 && !isa<scf::IfOp>(&op) &&
            !op.hasAttr(kPipelinedLoopCoreTypeAttrName) &&
            !op.hasAttr(TCoreTypeAttr::name))
          return true;
      }
    }
  }
  return false;
}

void CVScopeOutliner::addLocalOutput(Value val) {
  if (!val)
    return;
  Operation *defOp = val.getDefiningOp();
  if (!defOp || !loopOp->isProperAncestor(defOp))
    return;
  if (Value alias = valToOutputAlias.lookup(val))
    val = alias;
  if (!localOutputs.insert(val))
    return;
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    auto [tVal, eVal] =
        getIfYieldedValues(ifOp, cast<OpResult>(val).getResultNumber());
    for (Value branchVal : {tVal, eVal}) {
      Operation *bDef = branchVal ? branchVal.getDefiningOp() : nullptr;
      if (bDef && ifOp->isProperAncestor(bDef) &&
          !isStoppingToTensor(bDef, val) && !localOutputs.contains(branchVal))
        valToOutputAlias.try_emplace(branchVal, val);
    }
  }
}

void CVScopeOutliner::collectLocalOutputs() {
  unsigned nextOrder = 0;
  loopOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (op != loopOp)
      opOrder[op] = nextOrder++;
    if (auto ifOp = dyn_cast<scf::IfOp>(op))
      getOrAssignIfId(ifOp);
  });

  loopOp->walk([&](bufferization::ToTensorOp toTensor) {
    if (isAllocTargetOfStoreFixpipeCopy(getRootAlloc(toTensor.getOperand())))
      addLocalOutput(toTensor.getResult());
  });
  for (Region &reg : loopOp->getRegions()) {
    for (Block &block : reg) {
      for (auto yieldOp : block.getOps<scf::YieldOp>()) {
        if (yieldOp->getParentOp() == loopOp) {
          for (Value opnd : yieldOp.getOperands())
            addLocalOutput(opnd);
        }
      }
    }
  }
}

LogicalResult CVScopeOutliner::traceWorkItem(Value output, WorkItem &item,
                                             bool &failedLoop) {
  item.output = output;
  item.coreType = determineCoreType(output);

  SmallVector<Operation *> opStack;
  SmallPtrSet<Operation *, 32> visited;
  SmallPtrSet<Operation *, 32> inQueue;
  SmallPtrSet<Value, 32> visitedIfResults;
  SmallPtrSet<Value, 16> acceptedIfResults;
  DenseMap<Operation *, llvm::SmallSet<unsigned, 4>> pendingIfResults;
  SmallPtrSet<Operation *, 8> expandedAllocs;

  auto failNestedIf = [&]() {
    loopOp->emitWarning(
        "[OutlineCVScopes] multiple nested scf.if detected; skipping loop");
    failedLoop = true;
  };

  auto queueOp = [&](Operation *op) {
    if (!op || !loopOp->isProperAncestor(op))
      return;
    if ((isa<scf::IfOp>(op) || visited.insert(op).second) &&
        inQueue.insert(op).second)
      opStack.push_back(op);
  };

  auto popMaxOrderOp = [&]() -> Operation * {
    auto maxIt = llvm::max_element(opStack, [&](Operation *a, Operation *b) {
      return opOrder.lookup(a) < opOrder.lookup(b);
    });
    Operation *op = *maxIt;
    opStack.erase(maxIt);
    inQueue.erase(op);
    return op;
  };

  auto traceValue = [&](Value val) {
    if (!val || failedLoop)
      return;
    Operation *defOp = val.getDefiningOp();
    if (!defOp || !loopOp->isProperAncestor(defOp))
      return;
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      if (ifOp->getParentOfType<scf::IfOp>()) {
        failNestedIf();
        return;
      }
      pendingIfResults[ifOp].insert(cast<OpResult>(val).getResultNumber());
    }
    queueOp(defOp);
  };

  auto shouldStopAtValue = [&](Value res) -> bool {
    if (!res || !isa<TensorType>(res.getType()))
      return false;
    if (res == output || valToOutputAlias.lookup(res) == output)
      return false;
    if (localOutputs.contains(res) || valToOutputAlias.contains(res))
      return true;
    for (OpOperand &use : res.getUses()) {
      Operation *usr = use.getOwner();
      if (!loopOp->isProperAncestor(usr))
        continue;
      if (isa<scf::YieldOp>(usr)) {
        if (usr->getParentOp() == loopOp)
          return true;
        if (auto yieldIf = dyn_cast<scf::IfOp>(usr->getParentOp())) {
          unsigned resIdx = use.getOperandNumber();
          if (resIdx < yieldIf.getNumResults() &&
              !acceptedIfResults.contains(yieldIf.getResult(resIdx)))
            return true;
          continue;
        }
      }
      if (!item.ops.contains(usr))
        return true;
    }
    return false;
  };

  auto registerOpInIf = [&](Operation *op) -> bool {
    Operation *parentOp = op->getParentOp();
    if (parentOp == loopOp || !loopOp->isAncestor(parentOp))
      return true;
    scf::IfOp enclosingIf = op->getParentOfType<scf::IfOp>();
    if (!enclosingIf)
      return true;
    if (enclosingIf->getParentOfType<scf::IfOp>()) {
      failNestedIf();
      return false;
    }
    getOrCreateCondOps(item, enclosingIf, op);
    traceValue(enclosingIf.getCondition());
    return !failedLoop;
  };

  auto addOpAndTraceOperands = [&](Operation *op) -> bool {
    visited.insert(op);
    if (!registerOpInIf(op))
      return false;
    item.ops.insert(op);
    item.orderedOps.insert(op);
    for (Value opnd : op->getOperands()) {
      traceValue(opnd);
      if (failedLoop)
        return false;
    }
    return true;
  };

  auto expandAllocSubnet = [&](Operation *allocOp) -> bool {
    if (!allocOp || !loopOp->isProperAncestor(allocOp) ||
        !expandedAllocs.insert(allocOp).second)
      return true;
    if (!addOpAndTraceOperands(allocOp))
      return false;
    walkAllocUsers(allocOp, [&](OpOperand &use) {
      Operation *usr = use.getOwner();
      if (!loopOp->isProperAncestor(usr) || isa<bufferization::ToTensorOp>(usr))
        return false;
      return !addOpAndTraceOperands(usr);
    });
    return !failedLoop;
  };

  auto recordStoppedInput = [&](Value res) {
    Value canonOut = valToOutputAlias.lookup(res);
    if (!canonOut)
      canonOut = res;
    item.inputs.insert(canonOut);
    addLocalOutput(canonOut);
  };

  traceValue(output);
  if (failedLoop)
    return failure();

  while (!opStack.empty()) {
    Operation *currOp = popMaxOrderOp();

    if (isStoppingToTensor(currOp, output)) {
      item.inputs.insert(currOp->getResult(0));
      continue;
    }

    if (currOp->getNumRegions() > 0 && !isa<scf::IfOp>(currOp) &&
        !currOp->hasAttr(kPipelinedLoopCoreTypeAttrName) &&
        !currOp->hasAttr(TCoreTypeAttr::name)) {
      failedLoop = true;
      return failure();
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(currOp)) {
      if (ifOp->getParentOfType<scf::IfOp>()) {
        failNestedIf();
        return failure();
      }
      auto &pendingSet = pendingIfResults[ifOp];
      if (pendingSet.empty()) {
        getOrCreateCondOps(item, ifOp);
        traceValue(ifOp.getCondition());
        if (failedLoop)
          return failure();
        continue;
      }
      SmallVector<unsigned, 4> resIndices(pendingSet.begin(), pendingSet.end());
      pendingSet.clear();
      for (unsigned resIdx : resIndices) {
        Value ifRes = ifOp.getResult(resIdx);
        if (!visitedIfResults.insert(ifRes).second)
          continue;
        if (shouldStopAtValue(ifRes)) {
          recordStoppedInput(ifRes);
          continue;
        }
        acceptedIfResults.insert(ifRes);
        getOrCreateCondOps(item, ifOp);
        traceValue(ifOp.getCondition());
        auto [tVal, eVal] = getIfYieldedValues(ifOp, resIdx);
        traceValue(tVal);
        traceValue(eVal);
        if (failedLoop)
          return failure();
      }
      continue;
    }

    if (!canCloneOpInWorkItem(currOp, nullptr)) {
      bool stopped = false;
      for (Value res : currOp->getResults()) {
        if (shouldStopAtValue(res)) {
          recordStoppedInput(res);
          stopped = true;
        }
      }
      if (stopped)
        continue;
    }

    if (!addOpAndTraceOperands(currOp))
      return failure();

    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(currOp)) {
      if (!expandAllocSubnet(getRootAlloc(toTensor.getOperand())))
        return failure();
    } else if (isa<memref::AllocOp, AllocWorkspaceOp>(currOp)) {
      if (!expandAllocSubnet(currOp))
        return failure();
    }
  }

  if (!llvm::any_of(item.ops, isCoreOp))
    item.coreType = TCoreType::CUBE_AND_VECTOR;

  return success();
}

LogicalResult CVScopeOutliner::buildWorkItems() {
  bool failedLoop = false;
  for (size_t idx = 0; idx < localOutputs.size(); ++idx) {
    auto item = std::make_unique<WorkItem>();
    item->id = nextWorkItemId++;
    if (failed(traceWorkItem(localOutputs[idx], *item, failedLoop))) {
      if (failedLoop)
        return failure();
      continue;
    }
    if (!item->ops.empty() || !item->condOps.empty())
      workItems.push_back(std::move(item));
  }
  return failedLoop ? failure() : success();
}

LogicalResult CVScopeOutliner::sortWorkItems() {
  auto getOutputOrder = [&](Value out) -> unsigned {
    Operation *def = out ? out.getDefiningOp() : nullptr;
    if (!def)
      return 0;
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      auto [tVal, eVal] =
          getIfYieldedValues(ifOp, cast<OpResult>(out).getResultNumber());
      for (Value branchVal : {tVal, eVal}) {
        if (Operation *bDef = branchVal ? branchVal.getDefiningOp() : nullptr;
            bDef && ifOp->isProperAncestor(bDef))
          return opOrder.lookup(bDef);
      }
    }
    return opOrder.lookup(def);
  };

  llvm::stable_sort(workItems, [&](const std::unique_ptr<WorkItem> &a,
                                   const std::unique_ptr<WorkItem> &b) {
    return getOutputOrder(a->output) < getOutputOrder(b->output);
  });

  unsigned n = workItems.size();
  DenseMap<WorkItem *, unsigned> inDegree;
  DenseMap<WorkItem *, SmallVector<WorkItem *>> adj;

  for (const auto &item : workItems)
    inDegree[item.get()] = 0;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i != j && workItems[i]->output &&
          workItems[j]->inputs.contains(workItems[i]->output)) {
        adj[workItems[i].get()].push_back(workItems[j].get());
        inDegree[workItems[j].get()]++;
      }
    }
  }

  SmallVector<WorkItem *> q;
  for (const auto &item : workItems) {
    if (inDegree[item.get()] == 0)
      q.push_back(item.get());
  }

  for (size_t head = 0; head < q.size(); ++head) {
    WorkItem *curr = q[head];
    sortedWorkItems.push_back(curr);
    for (WorkItem *next : adj[curr]) {
      if (--inDegree[next] == 0)
        q.push_back(next);
    }
  }

  return (sortedWorkItems.size() == n) ? success() : failure();
}

void CVScopeOutliner::mapOpsToWorkItems() {
  for (WorkItem *item : sortedWorkItems) {
    for (Operation *op : item->orderedOps)
      opToWorkItemMap.try_emplace(op, item);
  }
}

void CVScopeOutliner::collectUnmappedOps() {
  std::unique_ptr<WorkItem> finalWorkItem = nullptr;
  auto checkAndAddUnmapped = [&](Operation *op) {
    if (opToWorkItemMap.contains(op) || isa<scf::IfOp>(op))
      return;
    if (isa<scf::YieldOp>(op) &&
        (op->getParentOp() == loopOp || isa<scf::IfOp>(op->getParentOp())))
      return;
    if (!finalWorkItem) {
      finalWorkItem = std::make_unique<WorkItem>();
      finalWorkItem->id = nextWorkItemId++;
      finalWorkItem->coreType = TCoreType::CUBE_AND_VECTOR;
    }
    finalWorkItem->ops.insert(op);
    finalWorkItem->orderedOps.insert(op);
    opToWorkItemMap[op] = finalWorkItem.get();

    if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp());
        ifOp && loopOp->isProperAncestor(ifOp))
      getOrCreateCondOps(*finalWorkItem, ifOp, op);
  };

  for (Region &reg : loopOp->getRegions()) {
    for (Block &block : reg) {
      for (Operation &op : block) {
        if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
          for (Operation &thenOp : *ifOp.thenBlock())
            checkAndAddUnmapped(&thenOp);
          if (ifOp.elseBlock()) {
            for (Operation &elseOp : *ifOp.elseBlock())
              checkAndAddUnmapped(&elseOp);
          }
        } else {
          checkAndAddUnmapped(&op);
        }
      }
    }
  }

  if (finalWorkItem) {
    sortedWorkItems.push_back(finalWorkItem.get());
    workItems.push_back(std::move(finalWorkItem));
  }
}

void CVScopeOutliner::computeHoistedOps(WorkItem *item) {
  for (auto &[_, cOps] : item->condOps) {
    scf::IfOp enclosingIf = cOps.ifOp;
    SmallVector<Operation *> candidateThenOps;
    for (Operation &tOp : *enclosingIf.thenBlock()) {
      if (!isa<scf::YieldOp>(&tOp) && cOps.thenOps.contains(&tOp))
        candidateThenOps.push_back(&tOp);
    }

    for (bool changed = true; changed;) {
      changed = false;
      for (Operation *tOp : candidateThenOps) {
        if (item->hoistedOps.contains(tOp))
          continue;
        bool isAlloc = isa<memref::AllocOp, AllocWorkspaceOp>(tOp) &&
                       isAllocTargetOfStoreFixpipeCopy(tOp);
        bool isCast = isa<memref::MemorySpaceCastOp, annotation::MarkOp,
                          bufferization::ToTensorOp, memref::CastOp>(tOp);
        if (!isAlloc && !isCast)
          continue;
        bool operandsAvailable =
            llvm::all_of(tOp->getOperands(), [&](Value opnd) {
              Operation *def = opnd.getDefiningOp();
              return !def || !enclosingIf->isAncestor(def) ||
                     (isCast && item->hoistedOps.contains(def));
            });
        if (operandsAvailable) {
          item->hoistedOps.insert(tOp);
          changed = true;
        }
      }
    }
  }
}

void CVScopeOutliner::populateWorkItemIO(Operation *origLoopTerminator) {
  for (WorkItem *item : sortedWorkItems) {
    item->inputs.clear();
    item->outputs.clear();
    item->ops.clear();
    for (Operation *op : item->orderedOps) {
      if (opToWorkItemMap.lookup(op) == item)
        item->ops.insert(op);
    }
    for (auto &[_, cOps] : item->condOps) {
      cOps.outputs.clear();
      cOps.thenOps.remove_if(
          [&](Operation *op) { return !item->ops.contains(op); });
      cOps.elseOps.remove_if(
          [&](Operation *op) { return !item->ops.contains(op); });
    }
    computeHoistedOps(item);
  }

  auto getIfOwner = [&](scf::IfOp ifOp, unsigned resIdx) -> WorkItem * {
    for (WorkItem *w : sortedWorkItems) {
      if (w->condOps.contains(ifOp) && isIfResultOwnedByItem(ifOp, resIdx, w))
        return w;
    }
    return nullptr;
  };

  struct PendingUse {
    Value val;
    WorkItem *consumer;
    scf::IfOp consumerIf;
  };
  SmallVector<PendingUse, 32> worklist;

  auto recordValueUse = [&](Value val, WorkItem *consumer,
                            scf::IfOp consumerIf) {
    if (!val)
      return;
    Operation *defOp = val.getDefiningOp();
    if (!defOp || !loopOp->isProperAncestor(defOp))
      return;

    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      unsigned resIdx = cast<OpResult>(val).getResultNumber();
      WorkItem *producer = getIfOwner(ifOp, resIdx);
      if (!producer)
        return;
      if (producer != consumer) {
        if (consumer)
          consumer->inputs.insert(val);
        producer->outputs.insert(val);
      }
      if (getOrCreateCondOps(*producer, ifOp).outputs.insert(val).second) {
        worklist.push_back({ifOp.getCondition(), producer, nullptr});
        auto [tVal, eVal] = getIfYieldedValues(ifOp, resIdx);
        if (tVal)
          worklist.push_back({tVal, producer, ifOp});
        if (eVal)
          worklist.push_back({eVal, producer, ifOp});
      }
      return;
    }

    WorkItem *producer = opToWorkItemMap.lookup(defOp);
    if (!producer)
      return;

    if (producer == consumer) {
      auto defIf = defOp->getParentOfType<scf::IfOp>();
      if (defIf && loopOp->isAncestor(defIf) &&
          !producer->hoistedOps.contains(defOp) && consumerIf != defIf &&
          getOrCreateCondOps(*producer, defIf).outputs.insert(val).second)
        worklist.push_back({defIf.getCondition(), producer, nullptr});
      return;
    }

    if (consumer && canCloneOpInWorkItem(defOp, consumer)) {
      if (consumer->ops.insert(defOp).second) {
        auto defIf = defOp->getParentOfType<scf::IfOp>();
        if (defIf && loopOp->isAncestor(defIf)) {
          getOrCreateCondOps(*consumer, defIf, defOp);
          if (!consumer->hoistedOps.contains(defOp))
            worklist.push_back({defIf.getCondition(), consumer, nullptr});
        } else {
          defIf = nullptr;
        }
        for (Value opnd : defOp->getOperands())
          worklist.push_back({opnd, consumer, defIf});
      }
      return;
    }

    if (consumer)
      consumer->inputs.insert(val);
    producer->outputs.insert(val);
    if (auto defIf = defOp->getParentOfType<scf::IfOp>();
        defIf && loopOp->isAncestor(defIf) &&
        !producer->hoistedOps.contains(defOp) &&
        getOrCreateCondOps(*producer, defIf).outputs.insert(val).second)
      worklist.push_back({defIf.getCondition(), producer, nullptr});
  };

  for (Value opnd : origLoopTerminator->getOperands())
    recordValueUse(opnd, /*consumer=*/nullptr, /*consumerIf=*/nullptr);

  for (WorkItem *item : sortedWorkItems) {
    for (Operation *op : item->orderedOps) {
      if (opToWorkItemMap.lookup(op) != item)
        continue;
      auto enclosingIf = op->getParentOfType<scf::IfOp>();
      if (enclosingIf &&
          (!loopOp->isAncestor(enclosingIf) || item->hoistedOps.contains(op)))
        enclosingIf = nullptr;
      if (enclosingIf)
        recordValueUse(enclosingIf.getCondition(), item, nullptr);
      for (Value opnd : op->getOperands())
        recordValueUse(opnd, item, enclosingIf);
    }
  }

  while (!worklist.empty()) {
    auto [val, consumer, consumerIf] = worklist.pop_back_val();
    recordValueUse(val, consumer, consumerIf);
  }
}

LogicalResult CVScopeOutliner::rewriteForOp(scf::ForOp origForOp) {
  OpBuilder builder(origForOp);
  Location loc = origForOp.getLoc();

  auto newForOp = builder.create<scf::ForOp>(
      loc, origForOp.getLowerBound(), origForOp.getUpperBound(),
      origForOp.getStep(), origForOp.getInitArgs());
  newForOp->setAttrs(origForOp->getAttrs());

  Block *origBody = origForOp.getBody();
  Block *newBody = newForOp.getBody();
  IRMapping globalIRMap;
  globalIRMap.map(origForOp.getInductionVar(), newForOp.getInductionVar());
  for (auto [origArg, newArg] :
       llvm::zip(origForOp.getRegionIterArgs(), newForOp.getRegionIterArgs()))
    globalIRMap.map(origArg, newArg);

  Operation *origLoopTerminator = origBody->getTerminator();
  populateWorkItemIO(origLoopTerminator);

  for (WorkItem *item : sortedWorkItems) {
    bool hasIfOutputs = llvm::any_of(item->condOps, [](const auto &pair) {
      return !pair.second.outputs.empty();
    });
    if (item->ops.empty() && !hasIfOutputs && item->outputs.empty())
      continue;

    SmallVector<Value> scopeReturnVals;
    for (Operation *op : item->orderedOps) {
      if (opToWorkItemMap.lookup(op) != item)
        continue;
      for (Value res : op->getResults()) {
        if (item->outputs.contains(res))
          scopeReturnVals.push_back(res);
      }
    }
    for (auto &[_, cOps] : item->condOps) {
      scf::IfOp enclosingIf = cOps.ifOp;
      for (auto [resIdx, ifRes] : llvm::enumerate(enclosingIf.getResults())) {
        if (item->outputs.contains(ifRes) &&
            isIfResultOwnedByItem(enclosingIf, resIdx, item))
          scopeReturnVals.push_back(ifRes);
      }
    }

    builder.setInsertionPointToEnd(newBody);
    auto placeholderScopeOp = builder.create<scope::ScopeOp>(loc, TypeRange{});
    placeholderScopeOp->setAttr(
        kPipelinedLoopCoreTypeAttrName,
        TCoreTypeAttr::get(builder.getContext(), item->coreType));

    Block *scopeBlock = builder.createBlock(&placeholderScopeOp.getRegion());
    builder.setInsertionPointToStart(scopeBlock);

    IRMapping scopeMap(globalIRMap);

    // Iterate through top-level operations of origBody
    for (Operation &origOp : *origBody) {
      if (&origOp == origLoopTerminator)
        continue;

      if (auto enclosingIf = dyn_cast<scf::IfOp>(&origOp)) {
        auto condIt = item->condOps.find(enclosingIf);
        if (condIt == item->condOps.end())
          continue;

        auto &cOps = condIt->second;
        SmallVector<Operation *> opsToHoist;
        SmallVector<Operation *> thenOpsToClone;
        for (Operation &tOp : *enclosingIf.thenBlock()) {
          if (isa<scf::YieldOp>(&tOp) || !cOps.thenOps.contains(&tOp))
            continue;
          if (item->hoistedOps.contains(&tOp))
            opsToHoist.push_back(&tOp);
          else
            thenOpsToClone.push_back(&tOp);
        }

        SmallVector<Operation *> elseOpsToClone;
        if (enclosingIf.elseBlock()) {
          for (Operation &eOp : *enclosingIf.elseBlock()) {
            if (!isa<scf::YieldOp>(&eOp) && cOps.elseOps.contains(&eOp))
              elseOpsToClone.push_back(&eOp);
          }
        }

        // Clone hoisted ops into scopeBlock before newIfOp
        builder.setInsertionPointToEnd(scopeBlock);
        for (Operation *hOp : opsToHoist)
          builder.clone(*hOp, scopeMap);

        SmallVector<Value> ifThenYieldVals;
        SmallVector<Value> ifElseYieldVals;
        SmallVector<Value> ifOrigResultsToMap;
        SmallPtrSet<Value, 8> ifYieldSet;

        for (auto [resIdx, ifRes] : llvm::enumerate(enclosingIf.getResults())) {
          auto [thenVal, elseVal] = getIfYieldedValues(enclosingIf, resIdx);
          if (!thenVal || !cOps.outputs.contains(ifRes) ||
              !ifYieldSet.insert(ifRes).second)
            continue;
          ifYieldSet.insert(thenVal);
          ifThenYieldVals.push_back(thenVal);
          ifElseYieldVals.push_back(elseVal);
          ifOrigResultsToMap.push_back(ifRes);
        }

        for (Operation *tOp : thenOpsToClone) {
          if (opToWorkItemMap.lookup(tOp) != item)
            continue;
          for (Value res : tOp->getResults()) {
            if (cOps.outputs.contains(res) && ifYieldSet.insert(res).second) {
              ifThenYieldVals.push_back(res);
              ifElseYieldVals.push_back(nullptr);
              ifOrigResultsToMap.push_back(res);
            }
          }
        }

        if (ifOrigResultsToMap.empty() && thenOpsToClone.empty() &&
            elseOpsToClone.empty())
          continue;

        Value cond = scopeMap.lookupOrDefault(enclosingIf.getCondition());
        bool needElse = !elseOpsToClone.empty() || !ifOrigResultsToMap.empty();

        builder.setInsertionPointToEnd(scopeBlock);
        auto newIfOp = builder.create<scf::IfOp>(
            enclosingIf.getLoc(), TypeRange(ValueRange(ifOrigResultsToMap)),
            cond, needElse);
        newIfOp->setAttr(kPipelineIfCondAttrName,
                         builder.getI64IntegerAttr(cOps.id));

        if (!newIfOp.thenBlock()->empty())
          newIfOp.thenBlock()->getTerminator()->erase();
        if (newIfOp.elseBlock() && !newIfOp.elseBlock()->empty())
          newIfOp.elseBlock()->getTerminator()->erase();

        // Populate thenBlock
        builder.setInsertionPointToStart(newIfOp.thenBlock());
        for (Operation *tOp : thenOpsToClone)
          builder.clone(*tOp, scopeMap);

        SmallVector<Value> thenYieldOperands;
        for (Value v : ifThenYieldVals)
          thenYieldOperands.push_back(scopeMap.lookupOrDefault(v));
        builder.setInsertionPointToEnd(newIfOp.thenBlock());
        builder.create<scf::YieldOp>(enclosingIf.thenYield().getLoc(),
                                     thenYieldOperands);

        // Populate elseBlock
        if (needElse) {
          builder.setInsertionPointToStart(newIfOp.elseBlock());
          for (Operation *eOp : elseOpsToClone)
            builder.clone(*eOp, scopeMap);

          builder.setInsertionPointToEnd(newIfOp.elseBlock());
          SmallVector<Value> elseYieldOperands;
          for (auto [thenVal, origElseVal] :
               llvm::zip(ifThenYieldVals, ifElseYieldVals)) {
            Value mappedElse =
                origElseVal ? scopeMap.lookupOrDefault(origElseVal) : Value();
            elseYieldOperands.push_back(getOrCreateElseYieldValue(
                builder, loc, scopeMap.lookupOrDefault(thenVal), mappedElse));
          }
          Location elseLoc =
              enclosingIf.elseBlock() ? enclosingIf.elseYield().getLoc() : loc;
          builder.create<scf::YieldOp>(elseLoc, elseYieldOperands);
        }

        // Map results of newIfOp
        auto mapIfValues = [&](ArrayRef<Value> vals, bool checkAncestor) {
          for (auto [yieldIdx, val] : llvm::enumerate(vals)) {
            if (!val)
              continue;
            Operation *d = val.getDefiningOp();
            if (!checkAncestor || (d && enclosingIf->isProperAncestor(d)))
              scopeMap.map(val, newIfOp.getResult(yieldIdx));
          }
        };
        mapIfValues(ifOrigResultsToMap, /*checkAncestor=*/false);
        mapIfValues(ifThenYieldVals, /*checkAncestor=*/true);
        mapIfValues(ifElseYieldVals, /*checkAncestor=*/true);

        builder.setInsertionPointToEnd(scopeBlock);
        continue;
      }

      // Top-level op in origBody
      if (item->ops.contains(&origOp)) {
        builder.setInsertionPointToEnd(scopeBlock);
        builder.clone(origOp, scopeMap);
      }
    }

    SmallVector<Value> uniqueMappedReturns;
    DenseMap<Value, unsigned> mappedValToIdx;
    SmallVector<std::pair<Value, unsigned>> origValToResIdx;

    for (Value ret : scopeReturnVals) {
      Value mapped = scopeMap.lookupOrDefault(ret);
      auto [it, inserted] =
          mappedValToIdx.try_emplace(mapped, uniqueMappedReturns.size());
      if (inserted)
        uniqueMappedReturns.push_back(mapped);
      origValToResIdx.emplace_back(ret, it->second);
    }

    auto scopeOp = recreateScope(placeholderScopeOp, uniqueMappedReturns);
    placeholderScopeOp.erase();

    for (auto [origVal, resIdx] : origValToResIdx)
      globalIRMap.map(origVal, scopeOp.getResult(resIdx));
  }

  // Final loop yield
  auto origYield = cast<scf::YieldOp>(origLoopTerminator);
  SmallVector<Value> yieldOperands;
  for (Value opnd : origYield.getOperands())
    yieldOperands.push_back(globalIRMap.lookupOrDefault(opnd));
  builder.setInsertionPointToEnd(newBody);
  builder.create<scf::YieldOp>(origYield.getLoc(), yieldOperands);

  newForOp->setAttr(CVPipelinedLoopAttr::name, builder.getUnitAttr());
  origForOp.replaceAllUsesWith(newForOp.getResults());
  origForOp.erase();
  return success();
}

LogicalResult CVScopeOutliner::run() {
  // Step 1: Query CoreOps and mark homogeneous loop core types
  SmallVector<Operation *> coreOps;
  loopOp->walk([&](Operation *op) {
    if (op != loopOp && isCoreOp(op))
      coreOps.push_back(op);
  });

  if (coreOps.empty()) {
    loopOp->setAttr(kPipelinedLoopCoreTypeAttrName,
                    TCoreTypeAttr::get(ctx, TCoreType::CUBE_AND_VECTOR));
    return success();
  }

  bool allCube = true;
  bool allVec = true;
  for (Operation *cop : coreOps) {
    auto core = queryConcreteCoreType(cop);
    allCube &= (core == TCoreType::CUBE);
    allVec &= (core == TCoreType::VECTOR);
  }

  if (allCube || allVec) {
    loopOp->setAttr(
        kPipelinedLoopCoreTypeAttrName,
        TCoreTypeAttr::get(ctx, allCube ? TCoreType::CUBE : TCoreType::VECTOR));
    return success();
  }

  // Mixed loop legality checks
  if (hasMultipleNestedIf()) {
    loopOp->emitWarning(
        "[OutlineCVScopes] multiple nested scf.if detected; skipping loop");
    return success();
  }

  if (hasIllegalRegionedOp())
    return success();

  // Step 2.1: Collect loop-local outputs
  collectLocalOutputs();

  // Step 2.2: DFS up def chains for each output
  if (failed(buildWorkItems()) || workItems.empty())
    return success();

  // Step 3: Topologically sort work items
  if (failed(sortWorkItems())) {
    loopOp->emitWarning(
        "[OutlineCVScopes] cycle detected in work item dependency graph; "
        "skipping loop");
    return success();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "=== Sorted WorkItems (" << sortedWorkItems.size()
                 << ") ===\n";
    for (WorkItem *item : sortedWorkItems) {
      llvm::dbgs() << "  Item " << item->id
                   << ": coreType=" << stringifyTCoreType(item->coreType)
                   << ", output=" << item->output << "\n";
      for (Value inp : item->inputs)
        llvm::dbgs() << "    input: " << inp << "\n";
    }
  });

  // Step 4: Map ops to work items (first-come, first-served)
  mapOpsToWorkItems();

  // Step 5: Final iteration for unmapped ops
  collectUnmappedOps();

  // Step 6: Create new loop with encapsulating scope.scope
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp))
    return rewriteForOp(forOp);
  return success();
}

//===----------------------------------------------------------------------===//
// OutlineCVScopesPass Implementation
//===----------------------------------------------------------------------===//

void OutlineCVScopesPass::runOnOperation() {
  SmallVector<Operation *> loopQueue;
  getOperation().walk<WalkOrder::PostOrder>(
      [&](scf::ForOp forOp) { loopQueue.push_back(forOp); });

  for (Operation *loopOp : loopQueue) {
    CVScopeOutliner outliner(loopOp);
    if (failed(outliner.run()))
      continue;
  }

  LLVM_DEBUG({
    getOperation().walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(CVPipelinedLoopAttr::name))
        writeScopeDependenceGraphToDotFile(forOp);
    });
  });
}

std::unique_ptr<Pass> createOutlineCVScopesPass() {
  return std::make_unique<OutlineCVScopesPass>();
}

} // namespace hivm
} // namespace mlir

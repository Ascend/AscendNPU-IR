//===- OutlineCVScopes.cpp - Outline CV Scopes for CV Pipelining V2 -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Util.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
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

// Classified region operations are cloned as a unit, including their implicit
// captures. Those captures must participate in tracing and scope exports too.
static SetVector<Value> getWorkItemOperands(Operation *op) {
  SetVector<Value> operands(op->operand_begin(), op->operand_end());
  getUsedValuesDefinedAbove(op->getRegions(), operands);
  return operands;
}

static Operation *ancestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

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
  Operation *outputOp = nullptr;
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
  static bool isViewLikeTensorOp(Operation *op);
  static Operation *getRootAlloc(Value memref);
  static bool isGMTarget(Value dst);
  static bool isStoreFixpipeCopyWriter(Operation *usr, Value memrefVal);
  static bool isLoadOrND2NZWriter(Operation *usr, Value memrefVal);
  static bool walkAllocUsers(Operation *rootAlloc,
                             llvm::function_ref<bool(OpOperand &)> visitUse);
  static bool isAllocTargetOfStoreFixpipeCopy(Operation *rootAlloc);
  static bool isAllocTargetOfMultiUserLoadOrND2NZ(Operation *rootAlloc);

  bool hasLoopLocalToTensor(Value memref);
  bool isDerivedFromLoopIterArg(Value val) const;
  bool isStoppingToTensor(Operation *tracedOp, Value startOutput);
  TCoreType determineCoreType(Value output);
  TCoreType determineCoreType(Operation *seedOp);
  bool canCloneOpInWorkItem(Operation *op, WorkItem *item) const;
  bool isIfResultOwnedByItem(scf::IfOp ifOp, unsigned resIdx,
                             WorkItem *item) const;

  // Analysis steps
  LogicalResult duplicateExtractScalarForCube();
  Value getIfConditionForItem(scf::IfOp ifOp, const WorkItem *item) const;
  unsigned getOrAssignIfId(scf::IfOp ifOp);
  ConditionalOps &getOrCreateCondOps(WorkItem &item, scf::IfOp ifOp,
                                     Operation *branchOp = nullptr);
  bool hasMultipleNestedIf();
  bool hasIllegalRegionedOp();
  void addLocalOutput(Value val);
  void addGMOutputOp(Operation *op);
  void collectLocalOutputs();
  LogicalResult buildWorkItems();
  LogicalResult traceWorkItem(Value output, Operation *seedOp, WorkItem &item,
                              bool &failedLoop);
  LogicalResult sortWorkItems();
  void mapOpsToWorkItems();
  LogicalResult collectScalarState();
  void collectUnmappedOps();
  void collectConditionalMerges();
  bool canExportValueFromWorkItem(Value val) const;
  bool trySplitVectorWorkItemOnCubeInput(WorkItem *item, Value cubeInp);
  bool splitVectorWorkItems(Operation *origLoopTerminator);
  LogicalResult sortPopulatedWorkItems();
  void computeHoistedOps(WorkItem *item);
  void populateWorkItemIO(Operation *origLoopTerminator);

  // IR rewriting steps
  LogicalResult rewriteForOp(scf::ForOp origForOp);
  LogicalResult verifyOutlinedScopes(scf::ForOp outlinedForOp) const;

  Operation *loopOp;
  MLIRContext *ctx;

  DenseMap<Operation *, unsigned> opOrder;
  DenseMap<Operation *, unsigned> ifOpToId;
  DenseMap<Value, Value> valToOutputAlias;
  SetVector<Value> localOutputs;
  SetVector<Operation *> gmOutputOps;
  SmallVector<std::unique_ptr<WorkItem>> workItems;
  SmallVector<WorkItem *> sortedWorkItems;
  DenseMap<Operation *, WorkItem *> opToWorkItemMap;
  DenseMap<Value, WorkItem *> conditionalMergeOwners;
  DenseMap<Operation *, Value> cubeIfConditions;
  SmallVector<Value> nestedCounters;
  Operation *checkpointClone = nullptr;
  unsigned nextWorkItemId = 0;
  unsigned nextIfId = 0;
  bool didSplitVectorItems = false;
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
  return op &&
         isa<ViewLikeOpInterface, memref::SubViewOp, memref::ViewOp,
             memref::CastOp, memref::ReinterpretCastOp,
             memref::MemorySpaceCastOp, memref::CollapseShapeOp,
             memref::ExpandShapeOp, memref::ReshapeOp>(op) &&
         llvm::none_of(op->getResultTypes(), llvm::IsaPred<TensorType>);
}

bool CVScopeOutliner::isViewLikeTensorOp(Operation *op) {
  return op &&
         isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
             tensor::ExtractSliceOp, tensor::ReshapeOp, tensor::CastOp>(op);
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

bool CVScopeOutliner::isGMTarget(Value dst) {
  for (Value curr = dst; curr;) {
    if (auto space = GetBufferSpaceAttr(curr))
      return space->getAddressSpace() == AddressSpace::GM;

    if (auto blkArg = dyn_cast<BlockArgument>(curr)) {
      Operation *parentOp = blkArg.getOwner()->getParentOp();
      if (auto loop = dyn_cast_or_null<LoopLikeOpInterface>(parentOp)) {
        if (OpOperand *init = loop.getTiedLoopInit(blkArg)) {
          curr = init->get();
          continue;
        }
      }
      return isa_and_nonnull<func::FuncOp>(parentOp);
    }

    Operation *def = curr.getDefiningOp();
    if (!def)
      return false;
    if (isa<AllocWorkspaceOp>(def))
      return true;
    if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(def)) {
      curr = insertSlice.getDest();
      continue;
    }
    if (isViewLikeMemRefOp(def) ||
        isa<bufferization::ToTensorOp, tensor::ExtractSliceOp,
            tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::ReshapeOp>(
            def)) {
      if (def->getNumOperands() == 0)
        return false;
      curr = def->getOperand(0);
      continue;
    }
    return false;
  }
  return false;
}

bool CVScopeOutliner::isStoreFixpipeCopyWriter(Operation *usr,
                                               Value memrefVal) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(usr))
    return copyOp.getTarget() == memrefVal;
  if (!isa<StoreOp, FixpipeOp, CopyOp>(usr))
    return false;
  if (isa<FixpipeOp>(usr)) {
    if (auto space = GetBufferSpaceAttr(memrefVal);
        space && space->getAddressSpace() == AddressSpace::L1)
      return false;
  }
  auto dps = dyn_cast<DestinationStyleOpInterface>(usr);
  return dps && llvm::any_of(dps.getDpsInits(),
                             [&](Value init) { return init == memrefVal; });
}

bool CVScopeOutliner::isLoadOrND2NZWriter(Operation *usr, Value memrefVal) {
  if (!isa<LoadOp, ND2NZOp>(usr))
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

bool CVScopeOutliner::isAllocTargetOfMultiUserLoadOrND2NZ(
    Operation *rootAlloc) {
  bool hasLoadOrND2NZ = walkAllocUsers(rootAlloc, [](OpOperand &use) {
    return isLoadOrND2NZWriter(use.getOwner(), use.get());
  });
  if (!hasLoadOrND2NZ)
    return false;
  unsigned numToTensorUses = 0;
  return walkAllocUsers(rootAlloc, [&](OpOperand &use) {
    auto toTensor = dyn_cast<bufferization::ToTensorOp>(use.getOwner());
    if (!toTensor)
      return false;
    SmallVector<Value, 8> queue = {toTensor.getResult()};
    SmallPtrSet<Value, 8> visited = {toTensor.getResult()};
    while (!queue.empty()) {
      Value curr = queue.pop_back_val();
      for (OpOperand &tensorUse : curr.getUses()) {
        Operation *usr = tensorUse.getOwner();
        if (isViewLikeTensorOp(usr)) {
          for (Value res : usr->getResults()) {
            if (visited.insert(res).second)
              queue.push_back(res);
          }
          continue;
        }
        if (++numToTensorUses > 1)
          return true;
      }
    }
    return false;
  });
}

bool CVScopeOutliner::hasLoopLocalToTensor(Value memref) {
  Operation *rootAlloc = getRootAlloc(memref);
  if (!rootAlloc)
    return false;
  return walkAllocUsers(rootAlloc, [&](OpOperand &use) {
    Operation *usr = use.getOwner();
    return isa<bufferization::ToTensorOp>(usr) && loopOp->isProperAncestor(usr);
  });
}

bool CVScopeOutliner::isDerivedFromLoopIterArg(Value val) const {
  auto loop = dyn_cast<LoopLikeOpInterface>(loopOp);
  if (!loop)
    return false;
  auto iterArgsRange = loop.getRegionIterArgs();
  if (iterArgsRange.empty())
    return false;
  SmallPtrSet<Value, 8> iterArgs(iterArgsRange.begin(), iterArgsRange.end());

  SmallVector<Value, 4> worklist = {val};
  SmallPtrSet<Value, 8> visited;
  while (!worklist.empty()) {
    Value curr = worklist.pop_back_val();
    if (!curr || !visited.insert(curr).second)
      continue;
    if (iterArgs.contains(curr))
      return true;
    Operation *def = curr.getDefiningOp();
    if (!def || !loopOp->isProperAncestor(def))
      continue;
    if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(def)) {
      worklist.push_back(insertSlice.getSource());
      worklist.push_back(insertSlice.getDest());
      continue;
    }
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(def)) {
      worklist.push_back(viewLike.getViewSource());
      continue;
    }
    if (isViewLikeTensorOp(def) || isViewLikeMemRefOp(def) ||
        isa<bufferization::ToTensorOp,
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
            bufferization::ToMemrefOp
#else
            bufferization::ToBufferOp
#endif
            >(def)) {
      if (def->getNumOperands() > 0)
        worklist.push_back(def->getOperand(0));
    }
  }
  return false;
}

bool CVScopeOutliner::isStoppingToTensor(Operation *tracedOp,
                                         Value startOutput) {
  auto toTensor = dyn_cast_or_null<bufferization::ToTensorOp>(tracedOp);
  if (!toTensor || toTensor.getResult() == startOutput)
    return false;
  Operation *rootAlloc = getRootAlloc(toTensor.getOperand());
  return isAllocTargetOfStoreFixpipeCopy(rootAlloc) ||
         isAllocTargetOfMultiUserLoadOrND2NZ(rootAlloc);
}

TCoreType CVScopeOutliner::determineCoreType(Value output) {
  Operation *defOp = output ? output.getDefiningOp() : nullptr;
  if (!defOp)
    return TCoreType::CUBE_AND_VECTOR;

  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    auto [tVal, eVal] =
        getIfYieldedValues(ifOp, cast<OpResult>(output).getResultNumber());
    for (Value branchVal : {tVal, eVal}) {
      if (Operation *bDef = branchVal ? branchVal.getDefiningOp() : nullptr;
          bDef && ifOp->isProperAncestor(bDef))
        return determineCoreType(branchVal);
    }
  }
  return determineCoreType(defOp);
}

TCoreType CVScopeOutliner::determineCoreType(Operation *defOp) {
  if (!defOp)
    return TCoreType::CUBE_AND_VECTOR;

  if (auto ct = queryConcreteCoreType(defOp))
    return *ct;

  SmallVector<Operation *> stack = {defOp};
  SmallPtrSet<Operation *, 16> visited = {defOp};

  while (!stack.empty()) {
    Operation *curr = stack.pop_back_val();
    if (isCoreOp(curr)) {
      if (auto ct = queryConcreteCoreType(curr))
        return *ct;
    }
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(curr)) {
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
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    Operation *memrefDef = loadOp.getMemRef().getDefiningOp();
    return !(memrefDef && memrefDef->hasAttr(kNormalizeMatmulCounterAttr)) &&
           !getRootAlloc(loadOp.getMemRef());
  }
  if (!isMemoryEffectFree(op))
    return false;
  if (isViewLikeMemRefOp(op) && op->getNumOperands() > 0 &&
      getRootAlloc(op->getOperand(0)))
    return false;
  return isa<tensor::EmptyOp>(op) || isViewLikeTensorOp(op) ||
         llvm::none_of(op->getResultTypes(), llvm::IsaPred<TensorType>);
}

bool CVScopeOutliner::isIfResultOwnedByItem(scf::IfOp ifOp, unsigned resIdx,
                                            WorkItem *item) const {
  if (auto *merge = conditionalMergeOwners.lookup(ifOp.getResult(resIdx)))
    return merge == item;
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

Value CVScopeOutliner::getIfConditionForItem(scf::IfOp ifOp,
                                             const WorkItem *item) const {
  if (item && item->coreType == TCoreType::CUBE) {
    if (Value cubeCond = cubeIfConditions.lookup(ifOp.getOperation()))
      return cubeCond;
  }
  return ifOp.getCondition();
}

LogicalResult CVScopeOutliner::duplicateExtractScalarForCube() {
  static constexpr llvm::StringLiteral kReplacementLabel =
      "DuplicateTensorExtractForCube::replacementLabel";
  static constexpr llvm::StringLiteral kNewExtractLabel =
      "DuplicateTensorExtractForCube::newExtractLabel";

  SmallVector<annotation::MarkOp> replacementMarks;
  loopOp->walk([&](annotation::MarkOp markOp) {
    if (markOp->hasAttr(kReplacementLabel) && !markOp.getValues().empty())
      replacementMarks.push_back(markOp);
  });
  if (replacementMarks.empty())
    return success();

  if (!checkpointClone)
    checkpointClone = OpBuilder(loopOp).clone(*loopOp);

  auto getNestedIfCores = [](scf::IfOp ifOp) -> std::pair<bool, bool> {
    bool hasCube = false;
    bool hasVec = false;
    ifOp->walk([&](Operation *nested) {
      if (nested == ifOp.getOperation())
        return;
      TCoreType ct = getCoreType(nested).value_or(TCoreType::CUBE_OR_VECTOR);
      if (ct == TCoreType::CUBE_OR_VECTOR) {
        if (auto concrete = queryConcreteCoreType(nested))
          ct = *concrete;
      }
      hasCube |= (ct == TCoreType::CUBE);
      hasVec |= (ct == TCoreType::VECTOR);
    });
    return {hasCube, hasVec};
  };

  OpBuilder builder(ctx);
  for (annotation::MarkOp markOp : replacementMarks) {
    Value original = markOp.getSrc();
    Value replacement = markOp.getValues()[0];

    Block *defBlock = nullptr;
    if (Operation *defOp = original.getDefiningOp())
      defBlock = defOp->getBlock();
    else if (auto blkArg = dyn_cast<BlockArgument>(original))
      defBlock = blkArg.getOwner();
    if (!defBlock)
      continue;

    DenseSet<Operation *> tainted;
    DenseSet<Operation *> cubeConsumers;
    SetVector<scf::IfOp> mixedCubeIfs;
    SetVector<Block *> taintedBranchBlocks;
    DenseSet<Value> visitedValues;
    SmallVector<Value> wl = {original};

    while (!wl.empty()) {
      Value v = wl.pop_back_val();
      if (!visitedValues.insert(v).second)
        continue;

      for (auto &[ifOpPtr, prevCond] : cubeIfConditions) {
        if (prevCond == v && ifOpPtr->getBlock() == defBlock)
          mixedCubeIfs.insert(cast<scf::IfOp>(ifOpPtr));
      }

      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        Operation *ancestor = ancestorInBlock(user, defBlock);
        if (!ancestor || ancestor == markOp.getOperation())
          continue;

        if (auto ifOp = dyn_cast<scf::IfOp>(ancestor)) {
          auto [hasCube, hasVec] = getNestedIfCores(ifOp);
          if (hasCube && hasVec) {
            if (user == ifOp.getOperation()) {
              if (!cubeIfConditions.contains(ifOp.getOperation()))
                mixedCubeIfs.insert(ifOp);
              continue;
            }
            Block *branchBlock = user->getBlock();
            Operation *branchAnc = ancestorInBlock(user, branchBlock);
            if (!branchAnc || isa<scf::YieldOp>(branchAnc))
              continue;
            TCoreType bCore =
                getCoreType(branchAnc).value_or(TCoreType::CUBE_OR_VECTOR);
            if (bCore == TCoreType::CUBE) {
              cubeConsumers.insert(branchAnc);
              continue;
            }
            if (bCore == TCoreType::VECTOR)
              continue;
            if (tainted.insert(branchAnc).second) {
              taintedBranchBlocks.insert(branchBlock);
              for (Value res : branchAnc->getResults())
                wl.push_back(res);
            }
            continue;
          }
          if (hasCube) {
            cubeConsumers.insert(ifOp.getOperation());
            continue;
          }
          if (hasVec)
            continue;
        }

        TCoreType ancCore =
            getCoreType(ancestor).value_or(TCoreType::CUBE_OR_VECTOR);
        if (ancCore == TCoreType::CUBE) {
          cubeConsumers.insert(ancestor);
          continue;
        }
        if (ancCore == TCoreType::VECTOR)
          continue;
        if (tainted.insert(ancestor).second) {
          for (Value res : ancestor->getResults())
            wl.push_back(res);
        }
      }
    }

    DenseSet<Operation *> needed;
    SmallVector<Operation *> seed;
    for (Operation *op : tainted) {
      bool reachesCube = false;
      for (Value res : op->getResults()) {
        for (Operation *u : res.getUsers()) {
          if (llvm::any_of(cubeConsumers, [&](Operation *consumer) {
                return consumer->isAncestor(u);
              })) {
            reachesCube = true;
            break;
          }
        }
        if (reachesCube)
          break;
      }
      if (reachesCube)
        seed.push_back(op);
    }
    for (scf::IfOp ifOp : mixedCubeIfs) {
      Value cond = cubeIfConditions.lookup(ifOp.getOperation());
      if (!cond)
        cond = ifOp.getCondition();
      if (Operation *condDef = cond.getDefiningOp();
          condDef && tainted.contains(condDef))
        seed.push_back(condDef);
    }

    while (!seed.empty()) {
      Operation *op = seed.pop_back_val();
      if (!needed.insert(op).second)
        continue;
      op->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          Operation *def = operand.getDefiningOp();
          if (!def)
            continue;
          if (tainted.contains(def) && !needed.contains(def)) {
            seed.push_back(def);
            continue;
          }
          Operation *defAnc = ancestorInBlock(def, defBlock);
          if (defAnc && tainted.contains(defAnc) && !needed.contains(defAnc))
            seed.push_back(defAnc);
        }
      });
    }

    bool originalUsedDirectly = false;
    for (Operation *consumer : cubeConsumers) {
      consumer->walk([&](Operation *opIn) {
        for (Value operand : opIn->getOperands()) {
          if (operand == original) {
            originalUsedDirectly = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (originalUsedDirectly)
        break;
    }
    if (!originalUsedDirectly) {
      for (scf::IfOp ifOp : mixedCubeIfs) {
        Value cond = cubeIfConditions.lookup(ifOp.getOperation());
        if (!cond)
          cond = ifOp.getCondition();
        if (cond == original) {
          originalUsedDirectly = true;
          break;
        }
      }
    }

    if (needed.empty() && !originalUsedDirectly) {
      bool escapesViaLoopYield = llvm::any_of(visitedValues, [&](Value val) {
        return llvm::any_of(val.getUsers(), [&](Operation *usr) {
          return isa<scf::YieldOp>(usr) && usr->getParentOp() == loopOp;
        });
      });
      if (!escapesViaLoopYield) {
        markOp->erase();
        if (Operation *replDef = replacement.getDefiningOp();
            replDef && replDef->hasAttr(kNewExtractLabel) &&
            replDef->use_empty())
          replDef->erase();
      }
      continue;
    }

    IRMapping irMap;
    irMap.map(original, replacement);

    builder.setInsertionPointAfter(markOp);
    for (Operation &op : *defBlock) {
      if (!needed.contains(&op))
        continue;
      Operation *cloned = builder.clone(op, irMap);
      builder.setInsertionPointAfter(cloned);
    }

    for (Block *branchBlock : taintedBranchBlocks) {
      for (Operation &op : llvm::make_early_inc_range(*branchBlock)) {
        if (!needed.contains(&op))
          continue;
        builder.setInsertionPoint(&op);
        builder.clone(op, irMap);
      }
    }

    for (Operation *consumer : cubeConsumers) {
      consumer->walk([&](Operation *opIn) {
        for (OpOperand &operand : opIn->getOpOperands()) {
          Value v = operand.get();
          if (v == original) {
            operand.set(replacement);
            continue;
          }
          if (Value mapped = irMap.lookupOrNull(v))
            operand.set(mapped);
        }
      });
    }

    for (scf::IfOp ifOp : mixedCubeIfs) {
      Value prevCond = cubeIfConditions.lookup(ifOp.getOperation());
      Value baseCond = prevCond ? prevCond : ifOp.getCondition();
      Value cubeCond = (baseCond == original) ? replacement
                                              : irMap.lookupOrDefault(baseCond);
      cubeIfConditions[ifOp.getOperation()] = cubeCond;
      if (prevCond && prevCond != cubeCond) {
        for (Operation *def = prevCond.getDefiningOp();
             def && def->getBlock() == defBlock && def->use_empty() &&
             isMemoryEffectFree(def);) {
          Operation *nextDef = def->getNumOperands() > 0
                                   ? def->getOperand(0).getDefiningOp()
                                   : nullptr;
          def->erase();
          def = nextDef;
        }
      }
    }

    markOp->erase();
  }

  return success();
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
  cOps.condition = getIfConditionForItem(ifOp, &item);
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
  // If branches are split between work items, so validate their operations
  // too. Other classified regions are kept intact as a single operation.
  WalkResult result = loopOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == loopOp || op->getNumRegions() == 0 || isa<scf::IfOp>(op))
      return WalkResult::advance();
    if (!op->hasAttr(kPipelinedLoopCoreTypeAttrName) &&
        !op->hasAttr(TCoreTypeAttr::name))
      return WalkResult::interrupt();
    return WalkResult::skip();
  });
  return result.wasInterrupted();
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

void CVScopeOutliner::addGMOutputOp(Operation *op) {
  if (!op || !loopOp->isProperAncestor(op))
    return;
  if (op->getNumResults() > 0) {
    for (Value res : op->getResults())
      addLocalOutput(res);
    return;
  }
  gmOutputOps.insert(op);
}

void CVScopeOutliner::collectLocalOutputs() {
  unsigned nextOrder = 0;
  loopOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (op != loopOp)
      opOrder[op] = nextOrder++;
    if (auto ifOp = dyn_cast<scf::IfOp>(op))
      getOrAssignIfId(ifOp);
  });

  loopOp->walk([&](Operation *op) {
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
      Operation *rootAlloc = getRootAlloc(toTensor.getOperand());
      if (isAllocTargetOfStoreFixpipeCopy(rootAlloc) ||
          isAllocTargetOfMultiUserLoadOrND2NZ(rootAlloc))
        addLocalOutput(toTensor.getResult());
      return;
    }
    if (auto storeOp = dyn_cast<StoreOp>(op)) {
      if (isGMTarget(storeOp.getDst()) &&
          !hasLoopLocalToTensor(storeOp.getDst()))
        addGMOutputOp(storeOp);
      return;
    }
    if (auto fixpipeOp = dyn_cast<FixpipeOp>(op)) {
      if (isGMTarget(fixpipeOp.getDst()) &&
          !hasLoopLocalToTensor(fixpipeOp.getDst()))
        addGMOutputOp(fixpipeOp);
    }
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
  loopOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == loopOp || isa<scf::IfOp>(op))
      return WalkResult::advance();
    if (op->getNumRegions() > 0)
      return WalkResult::skip();
    if (!isCoreOp(op) || op->getNumResults() == 0)
      return WalkResult::advance();
    if (!llvm::any_of(op->getOperands(), [&](Value opnd) {
          return isDerivedFromLoopIterArg(opnd);
        }))
      return WalkResult::advance();

    auto enclosingIf = op->getParentOfType<scf::IfOp>();
    bool inLoopIf = enclosingIf && loopOp->isProperAncestor(enclosingIf);
    for (Value res : op->getResults()) {
      if (valToOutputAlias.contains(res)) {
        addLocalOutput(res);
        continue;
      }
      if (inLoopIf && res.hasOneUse()) {
        OpOperand &use = *res.getUses().begin();
        if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner());
            yieldOp && yieldOp->getParentOp() == enclosingIf) {
          unsigned resIdx = use.getOperandNumber();
          if (resIdx < enclosingIf.getNumResults()) {
            auto [tVal, eVal] = getIfYieldedValues(enclosingIf, resIdx);
            Value otherVal = (tVal == res) ? eVal : tVal;
            Operation *otherDef = otherVal ? otherVal.getDefiningOp() : nullptr;
            if (otherVal &&
                (!otherDef || !enclosingIf->isProperAncestor(otherDef))) {
              addLocalOutput(enclosingIf.getResult(resIdx));
              continue;
            }
          }
        }
      }
      if (!inLoopIf || canExportValueFromWorkItem(res))
        addLocalOutput(res);
    }
    return WalkResult::advance();
  });
}

LogicalResult CVScopeOutliner::traceWorkItem(Value output, Operation *seedOp,
                                             WorkItem &item, bool &failedLoop) {
  item.output = output;
  item.outputOp = output ? output.getDefiningOp() : seedOp;
  item.coreType =
      output ? determineCoreType(output) : determineCoreType(seedOp);

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
    SmallVector<Value, 8> worklist = {res};
    SmallPtrSet<Value, 8> visitedVals = {res};
    while (!worklist.empty()) {
      Value curr = worklist.pop_back_val();
      for (OpOperand &use : curr.getUses()) {
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
        if (isViewLikeTensorOp(usr)) {
          for (Value usrRes : usr->getResults()) {
            if (visitedVals.insert(usrRes).second)
              worklist.push_back(usrRes);
          }
        }
      }
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
    traceValue(getIfConditionForItem(enclosingIf, &item));
    return !failedLoop;
  };

  auto addOpAndTraceOperands = [&](Operation *op) -> bool {
    visited.insert(op);
    if (!registerOpInIf(op))
      return false;
    item.ops.insert(op);
    item.orderedOps.insert(op);
    for (Value opnd : getWorkItemOperands(op)) {
      traceValue(opnd);
      if (failedLoop)
        return false;
    }
    return true;
  };

  auto expandAllocSubnet = [&](Operation *allocOp) -> bool {
    if (!allocOp || !expandedAllocs.insert(allocOp).second)
      return true;
    if (loopOp->isProperAncestor(allocOp) && !addOpAndTraceOperands(allocOp))
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

  if (output)
    traceValue(output);
  else
    queueOp(seedOp);
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
        traceValue(getIfConditionForItem(ifOp, &item));
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
        traceValue(getIfConditionForItem(ifOp, &item));
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
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(currOp)) {
      if (auto alloca = loadOp.getMemRef().getDefiningOp<memref::AllocaOp>();
          alloca && alloca->hasAttr(kNormalizeMatmulCounterAttr) &&
          !loopOp->isProperAncestor(alloca)) {
        for (Operation *user : alloca->getUsers()) {
          if (auto storeOp = dyn_cast<memref::StoreOp>(user);
              storeOp && storeOp.getMemRef() == alloca.getResult() &&
              loopOp->isProperAncestor(storeOp))
            queueOp(storeOp);
        }
      }
    }
  }

  if (!llvm::any_of(item.ops, isCoreOp))
    item.coreType = TCoreType::CUBE_AND_VECTOR;

  return success();
}

LogicalResult CVScopeOutliner::buildWorkItems() {
  bool failedLoop = false;
  auto addItem = [&](Value outVal, Operation *seedOp) -> LogicalResult {
    auto item = std::make_unique<WorkItem>();
    item->id = nextWorkItemId++;
    if (failed(traceWorkItem(outVal, seedOp, *item, failedLoop)))
      return failedLoop ? failure() : success();
    if (!item->ops.empty() || !item->condOps.empty())
      workItems.push_back(std::move(item));
    return success();
  };

  for (Operation *seedOp : gmOutputOps) {
    if (failed(addItem(/*outVal=*/nullptr, seedOp)))
      return failure();
  }
  for (size_t idx = 0; idx < localOutputs.size(); ++idx) {
    if (failed(addItem(localOutputs[idx], /*seedOp=*/nullptr)))
      return failure();
  }
  return failedLoop ? failure() : success();
}

LogicalResult CVScopeOutliner::sortWorkItems() {
  auto getOutputOrder = [&](const WorkItem &w) -> unsigned {
    if (Value out = w.output) {
      Operation *def = out.getDefiningOp();
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
    }
    return w.outputOp ? opOrder.lookup(w.outputOp) : 0;
  };

  llvm::stable_sort(workItems, [&](const std::unique_ptr<WorkItem> &a,
                                   const std::unique_ptr<WorkItem> &b) {
    return getOutputOrder(*a) < getOutputOrder(*b);
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

LogicalResult CVScopeOutliner::collectScalarState() {
  // Keep a counter's initialization, updates and reads ordered. Export the
  // scalar reads to consumers instead of cloning uninitialized allocations.
  SmallVector<Operation *> pending;
  for (auto alloca :
       cast<scf::ForOp>(loopOp).getBody()->getOps<memref::AllocaOp>())
    if (alloca->hasAttr(kNormalizeMatmulCounterAttr)) {
      llvm::append_range(pending, alloca->getUsers());
      nestedCounters.push_back(alloca.getResult());
    }
  SmallPtrSet<Operation *, 16> stateOps;
  while (!pending.empty()) {
    Operation *op = pending.pop_back_val();
    if (!loopOp->isProperAncestor(op) || !stateOps.insert(op).second)
      continue;
    if (op->getNumRegions() || isCoreOp(op) ||
        (!isMemoryEffectFree(op) &&
         !isa<memref::AllocaOp, memref::LoadOp, memref::StoreOp>(op)))
      return failure();
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        pending.push_back(def);
    if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp()))
      if (Operation *def = ifOp.getCondition().getDefiningOp())
        pending.push_back(def);
  }
  if (stateOps.empty())
    return success();

  // Keep top-level state operations separate so each state scope outputs at
  // most one result and scheduling preserves the memory access order.
  SmallVector<WorkItem *> states;
  for (Operation &op : *cast<scf::ForOp>(loopOp).getBody()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Operation *> branchStateOps;
      ifOp.walk([&](Operation *nested) {
        if (nested != ifOp.getOperation() && stateOps.contains(nested))
          branchStateOps.push_back(nested);
      });
      if (branchStateOps.empty())
        continue;
      auto state = std::make_unique<WorkItem>();
      state->id = nextWorkItemId++;
      state->coreType = TCoreType::CUBE_AND_VECTOR;
      for (Operation *nested : branchStateOps) {
        state->orderedOps.insert(nested);
        getOrCreateCondOps(*state, ifOp, nested);
      }
      states.push_back(state.get());
      workItems.push_back(std::move(state));
      continue;
    }
    bool updatesCounter = isa<scf::ForOp>(op) &&
                          llvm::any_of(getWorkItemOperands(&op), [&](Value v) {
                            return llvm::is_contained(nestedCounters, v);
                          });
    if (!stateOps.contains(&op) && !updatesCounter)
      continue;
    auto state = std::make_unique<WorkItem>();
    state->id = nextWorkItemId++;
    state->coreType = TCoreType::CUBE_AND_VECTOR;
    if (auto core =
            op.getAttrOfType<TCoreTypeAttr>(kPipelinedLoopCoreTypeAttrName))
      state->coreType = core.getTcoretype();
    state->orderedOps.insert(&op);
    states.push_back(state.get());
    workItems.push_back(std::move(state));
  }
  sortedWorkItems.insert(sortedWorkItems.begin(), states.begin(), states.end());
  return success();
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
    WorkItem *targetItem = nullptr;
    if (isa<annotation::MarkOp, DebugOp>(op)) {
      Value src = op->getOperand(0);
      if (Operation *defOp = src.getDefiningOp();
          defOp && loopOp->isProperAncestor(defOp)) {
        if (auto defIf = dyn_cast<scf::IfOp>(defOp)) {
          unsigned resIdx = cast<OpResult>(src).getResultNumber();
          for (WorkItem *w : sortedWorkItems) {
            if (!w->condOps.contains(defIf) ||
                !isIfResultOwnedByItem(defIf, resIdx, w))
              continue;
            if (targetItem && targetItem != w) {
              targetItem = nullptr;
              break;
            }
            targetItem = w;
          }
        } else {
          targetItem = opToWorkItemMap.lookup(defOp);
        }
      }
    }
    if (!targetItem) {
      if (!finalWorkItem) {
        finalWorkItem = std::make_unique<WorkItem>();
        finalWorkItem->id = nextWorkItemId++;
        finalWorkItem->coreType = TCoreType::CUBE_AND_VECTOR;
      }
      targetItem = finalWorkItem.get();
    }
    targetItem->ops.insert(op);
    targetItem->orderedOps.insert(op);
    opToWorkItemMap[op] = targetItem;

    if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp());
        ifOp && loopOp->isProperAncestor(ifOp))
      getOrCreateCondOps(*targetItem, ifOp, op);
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

// A conditional result cannot belong to either producer when its branches
// belong to different work items. Give the selection its own shared scope;
// populateWorkItemIO will export both branch values from their producers.
void CVScopeOutliner::collectConditionalMerges() {
  for (auto ifOp : cast<scf::ForOp>(loopOp).getBody()->getOps<scf::IfOp>()) {
    SmallPtrSet<WorkItem *, 4> branchItems;
    for (auto result : ifOp.getResults()) {
      if (conditionalMergeOwners.contains(result))
        continue;
      auto [thenValue, elseValue] =
          getIfYieldedValues(ifOp, result.getResultNumber());
      Operation *thenDef = thenValue ? thenValue.getDefiningOp() : nullptr;
      Operation *elseDef = elseValue ? elseValue.getDefiningOp() : nullptr;
      if (!thenDef || !elseDef || !ifOp->isProperAncestor(thenDef) ||
          !ifOp->isProperAncestor(elseDef))
        continue;
      WorkItem *thenOwner = opToWorkItemMap.lookup(thenDef);
      WorkItem *elseOwner = opToWorkItemMap.lookup(elseDef);
      if (!thenOwner || !elseOwner || thenOwner == elseOwner)
        continue;
      // The catch-all item may contain both a branch producer and consumers
      // of the selected result. Separate the branch operations so scheduling
      // the merge between producers and consumers does not create a cycle.
      for (auto [value, owner] : {std::make_pair(thenValue, thenOwner),
                                  std::make_pair(elseValue, elseOwner)}) {
        if (branchItems.contains(owner))
          continue;
        Operation *def = value.getDefiningOp();
        auto branch = std::make_unique<WorkItem>();
        branch->id = nextWorkItemId++;
        branch->coreType = determineCoreType(value);
        for (Operation *op : owner->orderedOps) {
          if (op->getBlock() != def->getBlock() ||
              opToWorkItemMap.lookup(op) != owner)
            continue;
          branch->orderedOps.insert(op);
          getOrCreateCondOps(*branch, ifOp, op);
          opToWorkItemMap[op] = branch.get();
        }
        branchItems.insert(branch.get());
        sortedWorkItems.push_back(branch.get());
        workItems.push_back(std::move(branch));
      }
      auto item = std::make_unique<WorkItem>();
      item->id = nextWorkItemId++;
      item->coreType = TCoreType::CUBE_AND_VECTOR;
      WorkItem *merge = item.get();
      getOrCreateCondOps(*merge, ifOp);
      sortedWorkItems.push_back(merge);
      workItems.push_back(std::move(item));
      conditionalMergeOwners[result] = merge;
    }
  }
}

bool CVScopeOutliner::canExportValueFromWorkItem(Value val) const {
  Operation *def = val ? val.getDefiningOp() : nullptr;
  if (!def || isa<scf::IfOp>(def))
    return true;
  auto enclosingIf = def->getParentOfType<scf::IfOp>();
  if (!enclosingIf || !loopOp->isProperAncestor(enclosingIf))
    return true;
  if (!isa<TensorType>(val.getType()))
    return true;
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(def)) {
    Value initVal = dpsOp.getTiedOpOperand(cast<OpResult>(val))->get();
    if (auto brc = initVal.getDefiningOp<VBrcOp>())
      initVal = brc.getDst();
    return isa_and_nonnull<tensor::EmptyOp>(initVal.getDefiningOp());
  }
  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(def)) {
    Operation *rootAlloc = getRootAlloc(toTensor.getOperand());
    if (!rootAlloc)
      return false;
    SmallPtrSet<Operation *, 8> chainInIf;
    for (Value curr = toTensor.getOperand(); curr;) {
      Operation *cDef = curr.getDefiningOp();
      if (!cDef || !enclosingIf->isProperAncestor(cDef))
        break;
      chainInIf.insert(cDef);
      if (cDef == rootAlloc)
        break;
      if (!isViewLikeMemRefOp(cDef) || cDef->getNumOperands() == 0)
        return false;
      curr = cDef->getOperand(0);
    }
    chainInIf.insert(toTensor.getOperation());
    return llvm::all_of(chainInIf, [&](Operation *cOp) {
      return llvm::all_of(cOp->getOperands(), [&](Value opnd) {
        Operation *oDef = opnd.getDefiningOp();
        return !oDef || !enclosingIf->isAncestor(oDef) ||
               chainInIf.contains(oDef);
      });
    });
  }
  return false;
}

bool CVScopeOutliner::trySplitVectorWorkItemOnCubeInput(WorkItem *item,
                                                        Value cubeInp) {
  SmallPtrSet<Value, 32> touchedValues;
  SmallPtrSet<Operation *, 32> touchedOps;
  SmallVector<Value, 32> valQueue;

  auto markValueTouched = [&](Value v) {
    if (v && touchedValues.insert(v).second)
      valQueue.push_back(v);
  };

  std::function<void(Operation *)> markOpTouched = [&](Operation *op) {
    if (!op || !item->ops.contains(op) || !touchedOps.insert(op).second)
      return;
    for (Value res : op->getResults())
      markValueTouched(res);

    if (isa<memref::AllocOp, AllocWorkspaceOp>(op) &&
        loopOp->isProperAncestor(op)) {
      walkAllocUsers(op, [&](OpOperand &use) {
        markOpTouched(use.getOwner());
        return false;
      });
    }

    if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
      for (Value init : dps.getDpsInits()) {
        if (isa<MemRefType>(init.getType())) {
          if (Operation *rootAlloc = getRootAlloc(init);
              rootAlloc && loopOp->isProperAncestor(rootAlloc)) {
            markOpTouched(rootAlloc);
            walkAllocUsers(rootAlloc, [&](OpOperand &use) {
              markOpTouched(use.getOwner());
              return false;
            });
          }
        } else if (isa<TensorType>(init.getType())) {
          SmallVector<Operation *, 4> viewChain;
          for (Value curr = init; curr;) {
            Operation *initDef = curr.getDefiningOp();
            if (!initDef || !item->ops.contains(initDef))
              break;
            if (isViewLikeTensorOp(initDef) && initDef->getNumOperands() > 0) {
              viewChain.push_back(initDef);
              curr = initDef->getOperand(0);
              continue;
            }
            if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(initDef)) {
              for (Operation *vOp : viewChain)
                markOpTouched(vOp);
              markOpTouched(initDef);
              if (Operation *rootAlloc = getRootAlloc(toTensor.getOperand());
                  rootAlloc && loopOp->isProperAncestor(rootAlloc)) {
                markOpTouched(rootAlloc);
                walkAllocUsers(rootAlloc, [&](OpOperand &use) {
                  markOpTouched(use.getOwner());
                  return false;
                });
              }
            }
            break;
          }
        }
      }
    }
  };

  auto drainValQueue = [&]() {
    while (!valQueue.empty()) {
      Value v = valQueue.pop_back_val();

      for (auto &[_, cOps] : item->condOps) {
        if (getIfConditionForItem(cOps.ifOp, item) == v) {
          for (Operation *tOp : cOps.thenOps)
            markOpTouched(tOp);
          for (Operation *eOp : cOps.elseOps)
            markOpTouched(eOp);
          for (auto [resIdx, ifRes] : llvm::enumerate(cOps.ifOp.getResults())) {
            if (isIfResultOwnedByItem(cOps.ifOp, resIdx, item))
              markValueTouched(ifRes);
          }
        }
      }

      for (OpOperand &use : v.getUses()) {
        Operation *usr = use.getOwner();
        if (!loopOp->isProperAncestor(usr))
          continue;
        if (isa<scf::YieldOp>(usr)) {
          if (auto ifOp = dyn_cast<scf::IfOp>(usr->getParentOp());
              ifOp && item->condOps.contains(ifOp)) {
            unsigned resIdx = use.getOperandNumber();
            if (resIdx < ifOp.getNumResults() &&
                isIfResultOwnedByItem(ifOp, resIdx, item))
              markValueTouched(ifOp.getResult(resIdx));
          }
          continue;
        }
        Operation *curr = usr;
        while (curr && curr != loopOp && !item->ops.contains(curr))
          curr = curr->getParentOp();
        if (curr && item->ops.contains(curr))
          markOpTouched(curr);
      }
    }
  };

  markValueTouched(cubeInp);
  drainValQueue();

  SetVector<Value> boundaryOutputs;
  SmallPtrSet<Operation *, 32> untouchedOwned;
  while (true) {
    for (Operation *op : item->orderedOps) {
      if (opToWorkItemMap.lookup(op) != item || touchedOps.contains(op) ||
          !isa<bufferization::AllocTensorOp>(op))
        continue;
      bool usedByTouched = false;
      bool usedByUntouched = false;
      for (Operation *usr : op->getUsers()) {
        if (opToWorkItemMap.lookup(usr) != item)
          continue;
        if (touchedOps.contains(usr))
          usedByTouched = true;
        else
          usedByUntouched = true;
      }
      if (usedByTouched && !usedByUntouched)
        touchedOps.insert(op);
    }

    untouchedOwned.clear();
    bool hasTouchedCoreOp = false;
    bool hasUntouchedCoreOp = false;
    for (Operation *op : item->orderedOps) {
      if (opToWorkItemMap.lookup(op) != item)
        continue;
      if (touchedOps.contains(op)) {
        if (isCoreOp(op))
          hasTouchedCoreOp = true;
      } else {
        untouchedOwned.insert(op);
        if (isCoreOp(op))
          hasUntouchedCoreOp = true;
      }
    }
    if (!hasTouchedCoreOp || !hasUntouchedCoreOp)
      return false;

    boundaryOutputs.clear();
    SmallPtrSet<Value, 32> visitedWalk;
    auto collectFrom = [&](Value rootVal) {
      SmallVector<Value, 16> stack = {rootVal};
      while (!stack.empty()) {
        Value v = stack.pop_back_val();
        if (!v || touchedValues.contains(v) || !visitedWalk.insert(v).second)
          continue;
        Operation *defOp = v.getDefiningOp();
        if (!defOp || !loopOp->isProperAncestor(defOp))
          continue;
        if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
          unsigned resIdx = cast<OpResult>(v).getResultNumber();
          if (isIfResultOwnedByItem(ifOp, resIdx, item))
            boundaryOutputs.insert(v);
          continue;
        }
        if (untouchedOwned.contains(defOp)) {
          if (canCloneOpInWorkItem(defOp, nullptr)) {
            for (Value opnd : getWorkItemOperands(defOp))
              stack.push_back(opnd);
          } else {
            boundaryOutputs.insert(v);
          }
          continue;
        }
        if (item->ops.contains(defOp) && canCloneOpInWorkItem(defOp, nullptr)) {
          for (Value opnd : getWorkItemOperands(defOp))
            stack.push_back(opnd);
        }
      }
    };

    for (Operation *tOp : item->orderedOps) {
      if (opToWorkItemMap.lookup(tOp) != item || !touchedOps.contains(tOp))
        continue;
      if (auto enIf = tOp->getParentOfType<scf::IfOp>();
          enIf && loopOp->isProperAncestor(enIf) &&
          !item->hoistedOps.contains(tOp))
        collectFrom(getIfConditionForItem(enIf, item));
      for (Value opnd : getWorkItemOperands(tOp))
        collectFrom(opnd);
    }
    for (auto &[_, cOps] : item->condOps) {
      for (auto [resIdx, ifRes] : llvm::enumerate(cOps.ifOp.getResults())) {
        if (!touchedValues.contains(ifRes) ||
            !isIfResultOwnedByItem(cOps.ifOp, resIdx, item))
          continue;
        collectFrom(getIfConditionForItem(cOps.ifOp, item));
        auto [tVal, eVal] = getIfYieldedValues(cOps.ifOp, resIdx);
        collectFrom(tVal);
        collectFrom(eVal);
      }
    }
    for (Value outVal : item->outputs) {
      if (!touchedValues.contains(outVal))
        collectFrom(outVal);
    }

    bool absorbedUnexportable = false;
    for (Value bVal : boundaryOutputs) {
      if (!canExportValueFromWorkItem(bVal)) {
        if (Operation *def = bVal.getDefiningOp()) {
          markOpTouched(def);
          drainValQueue();
          absorbedUnexportable = true;
        }
      }
    }
    if (!absorbedUnexportable)
      break;
  }

  if (boundaryOutputs.empty())
    return false;

  auto collectUntouchedSlice = [&](Value rootVal, bool stopAtBoundaries,
                                   SmallPtrSetImpl<Operation *> &sliceOps,
                                   SetVector<scf::IfOp> *sliceIfs) {
    SmallPtrSet<Value, 32> visitedVals;
    SmallPtrSet<Operation *, 8> expandedAllocs;
    SmallVector<Value, 32> stack = {rootVal};

    std::function<void(Operation *)> visitOp = [&](Operation *op) {
      if (!op || !untouchedOwned.contains(op) || !sliceOps.insert(op).second)
        return;
      if (auto enIf = op->getParentOfType<scf::IfOp>();
          enIf && loopOp->isProperAncestor(enIf))
        stack.push_back(getIfConditionForItem(enIf, item));
      for (Value opnd : getWorkItemOperands(op))
        stack.push_back(opnd);

      Operation *rootAlloc = nullptr;
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op))
        rootAlloc = getRootAlloc(toTensor.getOperand());
      else if (isa<memref::AllocOp, AllocWorkspaceOp>(op))
        rootAlloc = op;
      if (rootAlloc && expandedAllocs.insert(rootAlloc).second) {
        visitOp(rootAlloc);
        walkAllocUsers(rootAlloc, [&](OpOperand &use) {
          Operation *usr = use.getOwner();
          if (!isa<bufferization::ToTensorOp>(usr))
            visitOp(usr);
          return false;
        });
      }
    };

    while (!stack.empty()) {
      Value v = stack.pop_back_val();
      if (!v || !visitedVals.insert(v).second)
        continue;
      Operation *defOp = v.getDefiningOp();
      if (!defOp || !loopOp->isProperAncestor(defOp))
        continue;
      if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
        unsigned resIdx = cast<OpResult>(v).getResultNumber();
        if (v == rootVal ||
            ((!stopAtBoundaries || !boundaryOutputs.contains(v)) &&
             isIfResultOwnedByItem(ifOp, resIdx, item) &&
             !touchedValues.contains(v))) {
          if (sliceIfs)
            sliceIfs->insert(ifOp);
          stack.push_back(getIfConditionForItem(ifOp, item));
          auto [tVal, eVal] = getIfYieldedValues(ifOp, resIdx);
          stack.push_back(tVal);
          stack.push_back(eVal);
        }
        continue;
      }
      if (!untouchedOwned.contains(defOp)) {
        if (item->ops.contains(defOp) && canCloneOpInWorkItem(defOp, nullptr)) {
          for (Value opnd : getWorkItemOperands(defOp))
            stack.push_back(opnd);
        }
        continue;
      }
      if (stopAtBoundaries && v != rootVal && boundaryOutputs.contains(v))
        continue;
      visitOp(defOp);
    }
  };

  boundaryOutputs.remove_if([&](Value bVal) {
    SmallPtrSet<Operation *, 32> candidateOps;
    collectUntouchedSlice(bVal, /*stopAtBoundaries=*/false, candidateOps,
                          /*sliceIfs=*/nullptr);
    return llvm::none_of(candidateOps, isCoreOp);
  });
  if (boundaryOutputs.empty())
    return false;

  SmallVector<Value> sortedBoundaries(boundaryOutputs.begin(),
                                      boundaryOutputs.end());
  llvm::stable_sort(sortedBoundaries, [&](Value a, Value b) {
    return opOrder.lookup(a.getDefiningOp()) <
           opOrder.lookup(b.getDefiningOp());
  });

  SmallVector<WorkItem *, 4> newItemPtrs;
  for (Value bVal : sortedBoundaries) {
    SmallPtrSet<Operation *, 32> sliceOps;
    SetVector<scf::IfOp> sliceIfs;
    collectUntouchedSlice(bVal, /*stopAtBoundaries=*/true, sliceOps, &sliceIfs);
    if (llvm::none_of(sliceOps, isCoreOp))
      continue;

    SmallVector<Operation *, 32> orderedSlice(sliceOps.begin(), sliceOps.end());
    llvm::sort(orderedSlice, [&](Operation *a, Operation *b) {
      return opOrder.lookup(a) < opOrder.lookup(b);
    });

    auto newItem = std::make_unique<WorkItem>();
    newItem->id = nextWorkItemId++;
    newItem->output = bVal;
    newItem->outputOp = bVal.getDefiningOp();
    newItem->coreType = TCoreType::VECTOR;
    for (scf::IfOp ifOp : sliceIfs)
      getOrCreateCondOps(*newItem, ifOp);
    for (Operation *op : orderedSlice) {
      newItem->orderedOps.insert(op);
      opToWorkItemMap[op] = newItem.get();
      untouchedOwned.erase(op);
      if (auto enIf = op->getParentOfType<scf::IfOp>();
          enIf && loopOp->isProperAncestor(enIf))
        getOrCreateCondOps(*newItem, enIf, op);
    }
    newItemPtrs.push_back(newItem.get());
    workItems.push_back(std::move(newItem));
  }

  if (newItemPtrs.empty())
    return false;

  for (Operation *op : item->orderedOps) {
    if (!untouchedOwned.contains(op) || !isa<annotation::MarkOp, DebugOp>(op))
      continue;
    if (Operation *srcDef = op->getOperand(0).getDefiningOp()) {
      WorkItem *defOwner = opToWorkItemMap.lookup(srcDef);
      if (defOwner && defOwner != item) {
        defOwner->orderedOps.insert(op);
        opToWorkItemMap[op] = defOwner;
        untouchedOwned.erase(op);
        if (auto enIf = op->getParentOfType<scf::IfOp>();
            enIf && loopOp->isProperAncestor(enIf))
          getOrCreateCondOps(*defOwner, enIf, op);
      }
    }
  }

  auto it = llvm::find(sortedWorkItems, item);
  sortedWorkItems.insert(it, newItemPtrs.begin(), newItemPtrs.end());
  return true;
}

bool CVScopeOutliner::splitVectorWorkItems(Operation *origLoopTerminator) {
  bool anySplit = false;
  while (true) {
    populateWorkItemIO(origLoopTerminator);

    SmallPtrSet<Value, 16> cubeOutputs;
    DenseMap<Value, unsigned> cubeOutputPos;
    for (auto [idx, w] : llvm::enumerate(sortedWorkItems)) {
      if (w->coreType == TCoreType::CUBE) {
        for (Value out : w->outputs) {
          cubeOutputs.insert(out);
          cubeOutputPos[out] = idx;
        }
      }
    }
    if (cubeOutputs.empty())
      break;

    bool splitInRound = false;
    SmallVector<WorkItem *> snapshot(sortedWorkItems.begin(),
                                     sortedWorkItems.end());
    for (WorkItem *item : snapshot) {
      if (item->coreType != TCoreType::VECTOR)
        continue;
      SmallVector<Value, 4> itemCubeInputs;
      for (Value inp : item->inputs) {
        if (cubeOutputs.contains(inp))
          itemCubeInputs.push_back(inp);
      }
      if (itemCubeInputs.empty())
        continue;
      llvm::sort(itemCubeInputs, [&](Value a, Value b) {
        return cubeOutputPos.lookup(a) > cubeOutputPos.lookup(b);
      });
      for (Value cubeInp : itemCubeInputs) {
        if (trySplitVectorWorkItemOnCubeInput(item, cubeInp)) {
          splitInRound = true;
          anySplit = true;
          break;
        }
      }
    }
    if (!splitInRound)
      break;
  }
  return anySplit;
}

// Ownership and branch exports are only final after populateWorkItemIO.
// Schedule using all exported values, including those of the catch-all item
// and the shared conditional merges, rather than just WorkItem::output.
LogicalResult CVScopeOutliner::sortPopulatedWorkItems() {
  DenseMap<WorkItem *, SmallPtrSet<WorkItem *, 4>> predecessors;
  for (Value counter : nestedCounters) {
    WorkItem *previous = nullptr;
    for (Operation &op : *cast<scf::ForOp>(loopOp).getBody()) {
      if (op.getResults() == ValueRange{counter}) {
        previous = opToWorkItemMap.lookup(&op);
        continue;
      }
      SmallVector<WorkItem *, 2> owners;
      op.walk([&](Operation *nested) {
        if (!llvm::is_contained(nested->getOperands(), counter))
          return;
        WorkItem *owner = opToWorkItemMap.lookup(nested);
        if (!owner)
          owner = opToWorkItemMap.lookup(&op);
        if (owner && !llvm::is_contained(owners, owner))
          owners.push_back(owner);
      });
      if (owners.empty())
        continue;
      for (WorkItem *owner : owners) {
        if (previous && previous != owner)
          predecessors[owner].insert(previous);
        previous = owner;
      }
    }
  }
  SmallVector<WorkItem *> pending(sortedWorkItems);
  sortedWorkItems.clear();
  while (!pending.empty()) {
    auto ready = llvm::find_if(pending, [&](WorkItem *consumer) {
      return llvm::none_of(pending, [&](WorkItem *producer) {
        return producer != consumer &&
               (predecessors[consumer].contains(producer) ||
                llvm::any_of(consumer->inputs, [&](Value input) {
                  return producer->outputs.contains(input);
                }));
      });
    });
    if (ready == pending.end())
      return failure();
    sortedWorkItems.push_back(*ready);
    pending.erase(ready);
  }
  return success();
}

static Value getIntermediateInitValue(Value res) {
  if (!isa<TensorType>(res.getType()))
    return nullptr;
  auto dpsOp = res.getDefiningOp<DestinationStyleOpInterface>();
  assert(dpsOp && "expected intermediate tensor value to be defined by DPS op");
  Value initVal = dpsOp.getTiedOpOperand(cast<OpResult>(res))->get();
  // Preserve the broadcast padding on the active path, but reuse its backing
  // empty tensor as the value for the inactive branch.
  if (auto broadcast = initVal.getDefiningOp<VBrcOp>())
    initVal = broadcast.getDst();
  assert(initVal.getDefiningOp<tensor::EmptyOp>() &&
         "expected outs() of intermediate tensor value to be tensor.empty");
  return initVal;
}

void CVScopeOutliner::computeHoistedOps(WorkItem *item) {
  for (auto &[_, cOps] : item->condOps) {
    scf::IfOp enclosingIf = cOps.ifOp;
    SmallVector<Operation *> candidates;
    for (Region &region : enclosingIf->getRegions())
      for (Operation &op : region.getOps())
        if (cOps.thenOps.contains(&op) || cOps.elseOps.contains(&op))
          candidates.push_back(&op);

    for (bool changed = true; changed;) {
      changed = false;
      for (Operation *tOp : candidates) {
        if (item->hoistedOps.contains(tOp))
          continue;
        bool isAlloc =
            isa<memref::AllocOp, AllocWorkspaceOp>(tOp) &&
            (isAllocTargetOfStoreFixpipeCopy(tOp) ||
             isAllocTargetOfMultiUserLoadOrND2NZ(tOp) ||
             walkAllocUsers(tOp, [&](OpOperand &use) {
               auto toTensor =
                   dyn_cast<bufferization::ToTensorOp>(use.getOwner());
               return toTensor && item->output == toTensor.getResult();
             }));
        bool isCast =
            (isViewLikeMemRefOp(tOp) || isViewLikeTensorOp(tOp) ||
             isa<annotation::MarkOp, bufferization::ToTensorOp>(tOp)) &&
            llvm::any_of(tOp->getOperands(), [&](Value opnd) {
              return item->hoistedOps.contains(opnd.getDefiningOp());
            });
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
    item->hoistedOps.clear();
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
        worklist.push_back(
            {getIfConditionForItem(ifOp, producer), producer, nullptr});
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
        worklist.push_back(
            {getIfConditionForItem(defIf, producer), producer, nullptr});
      return;
    }

    if (consumer && canCloneOpInWorkItem(defOp, consumer)) {
      if (consumer->ops.insert(defOp).second) {
        auto defIf = defOp->getParentOfType<scf::IfOp>();
        if (defIf && loopOp->isAncestor(defIf)) {
          getOrCreateCondOps(*consumer, defIf, defOp);
          if (!consumer->hoistedOps.contains(defOp))
            worklist.push_back(
                {getIfConditionForItem(defIf, consumer), consumer, nullptr});
        } else {
          defIf = nullptr;
        }
        for (Value opnd : getWorkItemOperands(defOp))
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
      worklist.push_back(
          {getIfConditionForItem(defIf, producer), producer, nullptr});
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
        recordValueUse(getIfConditionForItem(enclosingIf, item), item, nullptr);
      for (Value opnd : getWorkItemOperands(op))
        recordValueUse(opnd, item, enclosingIf);
    }
  }

  while (!worklist.empty()) {
    auto [val, consumer, consumerIf] = worklist.pop_back_val();
    recordValueUse(val, consumer, consumerIf);
  }

  for (WorkItem *item : sortedWorkItems) {
    for (auto &[_, cOps] : item->condOps) {
      scf::IfOp enclosingIf = cOps.ifOp;
      for (Value outVal : cOps.outputs) {
        Operation *defOp = outVal.getDefiningOp();
        if (!defOp || defOp == enclosingIf ||
            opToWorkItemMap.lookup(defOp) != item ||
            item->hoistedOps.contains(defOp))
          continue;
        if (Value initVal = getIntermediateInitValue(outVal)) {
          if (Operation *initOp = initVal.getDefiningOp();
              initOp && enclosingIf->isProperAncestor(initOp))
            item->hoistedOps.insert(initOp);
        }
      }
    }
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
  // Resultless loops get an implicit yield from the builder. The rewritten
  // body creates its own terminator after all outlined scopes.
  if (!newBody->empty())
    newBody->getTerminator()->erase();
  IRMapping globalIRMap;
  globalIRMap.map(origForOp.getInductionVar(), newForOp.getInductionVar());
  for (auto [origArg, newArg] :
       llvm::zip(origForOp.getRegionIterArgs(), newForOp.getRegionIterArgs()))
    globalIRMap.map(origArg, newArg);

  Operation *origLoopTerminator = origBody->getTerminator();
  populateWorkItemIO(origLoopTerminator);
  if ((!conditionalMergeOwners.empty() || !nestedCounters.empty() ||
       didSplitVectorItems) &&
      failed(sortPopulatedWorkItems())) {
    origForOp.emitWarning(
        "[OutlineCVScopes] cyclic conditional merge; skipping loop");
    newForOp.erase();
    return failure();
  }

  for (WorkItem *item : sortedWorkItems) {
    bool hasIfOutputs = llvm::any_of(item->condOps, [](const auto &pair) {
      return !pair.second.outputs.empty();
    });
    if (item->ops.empty() && !hasIfOutputs && item->outputs.empty())
      continue;

    struct DeferredMerge {
      OpResult result;
      Value carried;
      bool updatedInThen;
    };
    SmallVector<DeferredMerge> deferredMerges;
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
            if (isa<scf::YieldOp>(&eOp) || !cOps.elseOps.contains(&eOp))
              continue;
            if (item->hoistedOps.contains(&eOp))
              opsToHoist.push_back(&eOp);
            else
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
          if (elseVal)
            ifYieldSet.insert(elseVal);
          Value updated = thenVal, carried = elseVal;
          auto isCarry = [&](Value value) {
            return llvm::is_contained(origForOp.getRegionIterArgs(), value);
          };
          if (!isCarry(carried))
            std::swap(updated, carried);
          // Export a hoisted buffer directly. The pass-through selection does
          // not belong to CUBE: capturing a VECTOR carry here keeps it on AIC.
          if (item->coreType == TCoreType::CUBE &&
              isa<TensorType>(ifRes.getType()) && isCarry(carried) &&
              item->hoistedOps.contains(updated.getDefiningOp()) &&
              item->outputs.contains(ifRes) &&
              llvm::none_of(ifRes.getUsers(), [&](Operation *user) {
                return item->ops.contains(user);
              })) {
            scopeMap.map(ifRes, scopeMap.lookup(updated));
            deferredMerges.push_back({ifRes, carried, updated == thenVal});
            continue;
          }
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
              ifElseYieldVals.push_back(getIntermediateInitValue(res));
              ifOrigResultsToMap.push_back(res);
            }
          }
        }

        for (Operation *eOp : elseOpsToClone) {
          if (opToWorkItemMap.lookup(eOp) != item)
            continue;
          for (Value res : eOp->getResults()) {
            if (!cOps.outputs.contains(res) || !ifYieldSet.insert(res).second)
              continue;
            ifThenYieldVals.push_back(getIntermediateInitValue(res));
            ifElseYieldVals.push_back(res);
            ifOrigResultsToMap.push_back(res);
          }
        }

        if (ifOrigResultsToMap.empty() && thenOpsToClone.empty() &&
            elseOpsToClone.empty())
          continue;

        Value cond =
            scopeMap.lookupOrDefault(getIfConditionForItem(enclosingIf, item));
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

        // Build both branches identically, exporting branch-local values
        // through if results before they are returned from the scope.
        auto populateBranch = [&](Block *block, ArrayRef<Operation *> ops,
                                  ArrayRef<Value> values, Location yieldLoc) {
          builder.setInsertionPointToStart(block);
          for (Operation *op : ops)
            builder.clone(*op, scopeMap);
          builder.setInsertionPointToEnd(block);
          SmallVector<Value> operands;
          for (auto [idx, value] : llvm::enumerate(values)) {
            Value mapped = value ? scopeMap.lookupOrDefault(value) : Value();
            operands.push_back(getOrCreateElseYieldValue(
                builder, loc, scopeMap.lookupOrDefault(ifOrigResultsToMap[idx]),
                mapped));
          }
          builder.create<scf::YieldOp>(yieldLoc, operands);
        };
        populateBranch(newIfOp.thenBlock(), thenOpsToClone, ifThenYieldVals,
                       enclosingIf.thenYield().getLoc());
        if (needElse)
          populateBranch(
              newIfOp.elseBlock(), elseOpsToClone, ifElseYieldVals,
              enclosingIf.elseBlock() ? enclosingIf.elseYield().getLoc() : loc);

        // Map results of newIfOp
        auto mapIfValues = [&](ArrayRef<Value> vals, bool checkAncestor) {
          for (auto [yieldIdx, val] : llvm::enumerate(vals)) {
            if (!val)
              continue;
            Operation *d = val.getDefiningOp();
            if (!checkAncestor || (d && enclosingIf->isProperAncestor(d) &&
                                   !item->hoistedOps.contains(d)))
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

    for (auto [result, carried, updatedInThen] : deferredMerges) {
      auto origIf = cast<scf::IfOp>(result.getOwner());
      builder.setInsertionPointToEnd(newBody);
      auto merge = builder.create<scope::ScopeOp>(origIf.getLoc(), TypeRange{});
      merge->setAttr(kPipelinedLoopCoreTypeAttrName,
                     TCoreTypeAttr::get(ctx, TCoreType::CUBE_AND_VECTOR));
      Block *mergeBlock = builder.createBlock(&merge.getRegion());
      builder.setInsertionPointToStart(mergeBlock);
      IRMapping mergeMap(globalIRMap);
      Value origCond = getIfConditionForItem(origIf, item);
      if (!mergeMap.contains(origCond)) {
        SmallVector<Operation *> condOps;
        SmallVector<Value> stack = {origCond};
        SmallPtrSet<Operation *, 8> visited;
        while (!stack.empty()) {
          Value v = stack.pop_back_val();
          if (mergeMap.contains(v))
            continue;
          Operation *def = v.getDefiningOp();
          if (!def || !origForOp->isProperAncestor(def) ||
              !visited.insert(def).second)
            continue;
          condOps.push_back(def);
          for (Value opnd : def->getOperands())
            stack.push_back(opnd);
        }
        llvm::sort(condOps, [&](Operation *a, Operation *b) {
          return opOrder.lookup(a) < opOrder.lookup(b);
        });
        for (Operation *op : condOps)
          builder.clone(*op, mergeMap);
      }
      auto select = builder.create<scf::IfOp>(
          origIf.getLoc(), TypeRange{result.getType()},
          mergeMap.lookupOrDefault(origCond), true);
      select->setAttr(kPipelineIfCondAttrName,
                      builder.getI64IntegerAttr(getOrAssignIfId(origIf)));
      Value updated = globalIRMap.lookup(result);
      carried = globalIRMap.lookupOrDefault(carried);
      builder.setInsertionPointToEnd(select.thenBlock());
      builder.create<scf::YieldOp>(origIf.getLoc(),
                                   updatedInThen ? updated : carried);
      builder.setInsertionPointToEnd(select.elseBlock());
      builder.create<scf::YieldOp>(origIf.getLoc(),
                                   updatedInThen ? carried : updated);
      auto newMerge = recreateScope(merge, select.getResults());
      merge.erase();
      globalIRMap.map(result, newMerge.getResult(0));
    }
  }

  // Final loop yield
  auto origYield = cast<scf::YieldOp>(origLoopTerminator);
  SmallVector<Value> yieldOperands;
  for (Value opnd : origYield.getOperands())
    yieldOperands.push_back(globalIRMap.lookupOrDefault(opnd));
  builder.setInsertionPointToEnd(newBody);
  builder.create<scf::YieldOp>(origYield.getLoc(), yieldOperands);

  // lookupOrDefault is valid for values outside the original loop, but must
  // never leave a reference into the original loop that we are about to delete.
  WalkResult unmappedUse = newForOp->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *owner = operand.getDefiningOp();
      if (auto arg = dyn_cast<BlockArgument>(operand))
        owner = arg.getOwner()->getParentOp();
      if (owner && origForOp->isAncestor(owner)) {
        LLVM_DEBUG(llvm::dbgs() << "Unmapped operand: " << operand
                                << "\nUser: " << *op << "\n");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (unmappedUse.wasInterrupted()) {
    origForOp.emitWarning(
        "[OutlineCVScopes] unmapped loop-local value; skipping loop");
    newForOp.erase();
    return failure();
  }

  if (failed(verifyOutlinedScopes(newForOp))) {
    newForOp.erase();
    return failure();
  }

  newForOp->setAttr(CVPipelinedLoopAttr::name, builder.getUnitAttr());
  origForOp.replaceAllUsesWith(newForOp.getResults());
  if (checkpointClone) {
    checkpointClone->erase();
    checkpointClone = nullptr;
  }
  origForOp.erase();
  return success();
}

LogicalResult
CVScopeOutliner::verifyOutlinedScopes(scf::ForOp outlinedForOp) const {
  for (auto scopeOp : outlinedForOp.getBody()->getOps<scope::ScopeOp>()) {
    if (scopeOp.getNumResults() > 1) {
      outlinedForOp.emitWarning(
          "[OutlineCVScopes] outlined scope must have at most 1 result");
      return failure();
    }
  }
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

  auto revertCheckpoint = [&]() {
    if (checkpointClone) {
      loopOp->replaceAllUsesWith(checkpointClone->getResults());
      loopOp->erase();
      checkpointClone = nullptr;
    }
  };

  if (failed(duplicateExtractScalarForCube())) {
    revertCheckpoint();
    return success();
  }

  // Step 2.1: Collect loop-local outputs
  collectLocalOutputs();

  // Step 2.2: DFS up def chains for each output
  if (failed(buildWorkItems()) || workItems.empty()) {
    revertCheckpoint();
    return success();
  }

  // Step 3: Topologically sort work items
  if (failed(sortWorkItems())) {
    loopOp->emitWarning(
        "[OutlineCVScopes] cycle detected in work item dependency graph; "
        "skipping loop");
    revertCheckpoint();
    return success();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "=== Sorted WorkItems (" << sortedWorkItems.size()
                 << ") ===\n";
    for (WorkItem *item : sortedWorkItems) {
      llvm::dbgs() << "  Item " << item->id
                   << ": coreType=" << stringifyTCoreType(item->coreType)
                   << ", output=";
      if (item->output)
        llvm::dbgs() << item->output;
      else if (item->outputOp)
        llvm::dbgs() << *item->outputOp;
      else
        llvm::dbgs() << "<unmapped>";
      llvm::dbgs() << "\n";
      for (Value inp : item->inputs)
        llvm::dbgs() << "    input: " << inp << "\n";
    }
  });

  // Step 4: Map ops to work items (first-come, first-served)
  if (failed(collectScalarState())) {
    revertCheckpoint();
    return success();
  }
  mapOpsToWorkItems();

  // Step 5: Final iteration for unmapped ops
  collectUnmappedOps();
  collectConditionalMerges();
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    didSplitVectorItems =
        splitVectorWorkItems(forOp.getBody()->getTerminator());
    if (didSplitVectorItems)
      collectConditionalMerges();
  }

  // Step 6: Create new loop with encapsulating scope.scope
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    LogicalResult res = rewriteForOp(forOp);
    if (failed(res))
      revertCheckpoint();
    return res;
  }
  revertCheckpoint();
  return success();
}

//===----------------------------------------------------------------------===//
// OutlineCVScopesPass Implementation
//===----------------------------------------------------------------------===//

void OutlineCVScopesPass::runOnOperation() {
  SmallVector<Operation *> loopQueue;
  getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
    // TODO: Temporary transition measure: skip loops already pipelined by
    // legacy CVPipelining.
    for (Operation *curr = forOp; curr != nullptr; curr = curr->getParentOp()) {
      if (curr->hasAttr(LegacyCVPipelineAttr::name))
        return;
    }
    loopQueue.push_back(forOp);
  });

  for (Operation *loopOp : loopQueue) {
    CVScopeOutliner outliner(loopOp);
    if (failed(outliner.run()))
      continue;
  }

  LLVM_DEBUG({
    getOperation().walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(CVPipelinedLoopAttr::name) &&
          !forOp->hasAttr(LegacyCVPipelineAttr::name))
        writeScopeDependenceGraphToDotFile(forOp, "outlined");
    });
  });
}

std::unique_ptr<Pass> createOutlineCVScopesPass() {
  return std::make_unique<OutlineCVScopesPass>();
}

} // namespace hivm
} // namespace mlir

//===--------- SyncSolverIRTranslator.cpp ------- Graph Sync Solver -------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <climits>
#include <iterator>
#include <memory>
#include <utility>

#define DEBUG_TYPE "hivm-graph-sync-solver-ir-translator"

using namespace mlir;
using namespace hivm::syncsolver;

// Resolve a Value into the underlying pointer-like Values used for memory
// conflict analysis (handles block args, selects, scf::If, scf::For/While
// results etc.).
llvm::SmallVector<Value> Solver::collectPointerOps(Value val) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    if (auto forOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (auto *iterArgOperand = forOp.getTiedLoopInit(blockArg)) {
        return collectPointerOps(iterArgOperand->get());
      }
    }

    if (auto whileOp =
            dyn_cast<scf::WhileOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg.getOwner()->getParent() == &whileOp.getAfter()) {
        auto argNum = blockArg.getArgNumber();
        return collectPointerOps(whileOp.getConditionOp().getArgs()[argNum]);
      } else {
        assert(blockArg.getOwner()->getParent() == &whileOp.getBefore());
        return collectPointerOps(whileOp.getTiedLoopInit(blockArg)->get());
      }
    }

    if (hacc::utils::isKernelArg(func, blockArg.getArgNumber(),
                                 hacc::KernelArgType::kWorkspace)) {
      bool isSplittedMixKernel =
          func->getAttrOfType<UnitAttr>(hivm::TPartOfMixAttr::name) != nullptr;
      if (isSplittedMixKernel) {
        return {};
      }
    }
    return {val};
  }

  auto *op = val.getDefiningOp();
  assert(op != nullptr);

  if (isa<hivm::PointerCastOp, tensor::EmptyOp>(op)) {
    return {val};
  }

  for (auto aliasInfo : getOperationAliasInfo(op)) {
    return collectPointerOps(aliasInfo.second);
  }

  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    llvm::SmallVector<Value> collectedOps;
    auto firstPath = collectPointerOps(selectOp.getTrueValue());
    auto secondPath = collectPointerOps(selectOp.getFalseValue());
    collectedOps.append(firstPath);
    collectedOps.append(secondPath);
    return collectedOps;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    llvm::SmallVector<Value> collectedOps;
    Operation::result_range resultVals = ifOp.getResults();
    auto it = std::find(resultVals.begin(), resultVals.end(), val);
    assert(it != resultVals.end());
    OpResult resultVal = *it;
    auto operandNum = resultVal.getResultNumber();
    // then
    auto thenYield = ifOp.thenYield();
    auto firstPath = collectPointerOps(thenYield->getOperand(operandNum));
    collectedOps.append(firstPath);
    // else
    auto elseYield = ifOp.elseYield();
    if (elseYield) {
      auto secondPath = collectPointerOps(elseYield->getOperand(operandNum));
      collectedOps.append(secondPath);
    }
    return collectedOps;
  }

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    auto resultNum = dyn_cast<OpResult>(val).getResultNumber();
    auto yieldedVal = forOp.getYieldedValues()[resultNum];
    return collectPointerOps(yieldedVal);
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    auto resultNum = dyn_cast<OpResult>(val).getResultNumber();
    auto yieldedVal = whileOp.getYieldedValues()[resultNum];
    return collectPointerOps(yieldedVal);
  }

  return {};
}

// Collect pointer operands for a vector of Values (flattening aliases).
llvm::SmallVector<Value> Solver::getMemOps(const SmallVector<Value> &vals) {
  SmallVector<Value> collectedOps;
  for (auto val : vals) {
    for (auto pointerOp : collectPointerOps(val)) {
      collectedOps.push_back(pointerOp);
    }
  }
  return collectedOps;
}

// Return read/write memory operands for a generic operation by consulting
// DestinationStyleOpInterface and ExtraBufferOpInterface.
std::pair<llvm::SmallVector<Value>, llvm::SmallVector<Value>>
Solver::getReadWriteMemOps(Operation *op) {
  assert(op != nullptr);
  llvm::SmallVector<Value> readMemVals;
  llvm::SmallVector<Value> writeMemVals;
  if (auto dsiOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    readMemVals = getMemOps(dsiOp.getDpsInputs());
    writeMemVals = getMemOps(dsiOp.getDpsInits());
  }
  if (auto extraBufferOp = dyn_cast<ExtraBufferOpInterface>(op)) {
    auto extraWriteMemVals = getMemOps(extraBufferOp.getExtraBuffers());
    llvm::append_range(writeMemVals, extraWriteMemVals);
  }
  return std::make_pair(readMemVals, writeMemVals);
}

// Wrap memref/affine load/store into RWOperation nodes when appropriate.
template <typename OP>
typename std::enable_if<std::is_same_v<OP, memref::LoadOp> ||
                            std::is_same_v<OP, affine::AffineLoadOp> ||
                            std::is_same_v<OP, affine::AffineStoreOp> ||
                            std::is_same_v<OP, memref::StoreOp>,
                        std::unique_ptr<OperationBase>>::type
Solver::getLoadStoreOp(OP *loadStoreOp, OperationBase *parentOp) {
  auto op = loadStoreOp->getOperation();
  auto pipe = hivm::PIPE::PIPE_S;
  auto coreTypeVal = hivm::TCoreType::CUBE_OR_VECTOR;
  auto coreType = hivm::getCoreType(op);
  if (succeeded(coreType)) {
    coreTypeVal = coreType.value();
  }
  llvm::SmallVector<Value> readMemVals;
  llvm::SmallVector<Value> writeMemVals;
  auto memorySpaceAttr = GetBufferSpaceAttr(loadStoreOp->getMemRef());
  if (!memorySpaceAttr.has_value()) {
    return nullptr;
  }
  if (std::is_same_v<OP, memref::LoadOp> ||
      std::is_same_v<OP, affine::AffineLoadOp>) {
    readMemVals = getMemOps({loadStoreOp->getMemRef()});
  } else {
    writeMemVals = getMemOps({loadStoreOp->getMemRef()});
  }
  auto rwOp = std::make_unique<RWOperation>(
      op, parentOp, pipe, pipe, coreTypeVal, readMemVals, writeMemVals);
  return rwOp;
}

// Decompose specific MmadL1 ops into a small inline sequence in the IR for
// easier sync handling.
std::unique_ptr<OperationBase>
Solver::getDecomposedMmadl1(hivm::MmadL1Op mmadl1Op, OperationBase *parentOp) {
  auto mmadl1LoopOp = std::make_unique<MmadL1LoopOp>(mmadl1Op, parentOp);
  auto scopeOp = std::make_unique<Scope>();
  scopeOp->parentOp = mmadl1LoopOp.get();

  auto loadL0aOp = std::make_unique<LoadL0AOp>(
      nullptr, scopeOp.get(), hivm::PIPE::PIPE_MTE1, hivm::PIPE::PIPE_MTE1,
      TCoreType::CUBE, getMemOps({mmadl1Op.getA()}), SmallVector<Value>());
  scopeOp->body.push_back(std::move(loadL0aOp));

  auto loadL0bOp = std::make_unique<LoadL0BOp>(
      nullptr, scopeOp.get(), hivm::PIPE::PIPE_MTE1, hivm::PIPE::PIPE_MTE1,
      TCoreType::CUBE, getMemOps({mmadl1Op.getB()}), SmallVector<Value>());
  scopeOp->body.push_back(std::move(loadL0bOp));

  if (auto bias = mmadl1Op.getPerChannelBias()) {
    auto loadBiasOp = std::make_unique<LoadBiasOp>(
        nullptr, scopeOp.get(), hivm::PIPE::PIPE_MTE1, hivm::PIPE::PIPE_MTE1,
        TCoreType::CUBE, getMemOps({mmadl1Op.getPerChannelBias()}),
        SmallVector<Value>());
    scopeOp->body.push_back(std::move(loadBiasOp));
  }

  auto mmadl0Op = std::make_unique<MmadL0Operation>(
      mmadl1Op, scopeOp.get(), hivm::PIPE::PIPE_M, hivm::PIPE::PIPE_M,
      TCoreType::CUBE, SmallVector<Value>(), getMemOps({mmadl1Op.getC()}));
  mmadl0Op->hasUnitFlagFeat = true;
  unitFlagFeaturedOps.insert(mmadl0Op.get());
  mmadl1LoopOp->mmadL0Op = mmadl0Op.get();
  scopeOp->body.push_back(std::move(mmadl0Op));

  mmadl1LoopOp->body.push_back(std::move(scopeOp));
  return mmadl1LoopOp;
}

// Build a Scope tree (funcIr) from MLIR Region recursively.
std::unique_ptr<Scope> Solver::funcIrBuilder(Region &region,
                                             OperationBase *parentOp) {
  auto scopeOp = std::make_unique<Scope>();
  scopeOp->parentOp = parentOp;

  for (auto &block : region.getBlocks()) {
    for (auto &op : block.getOperations()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        auto trueScope = funcIrBuilder(ifOp.getThenRegion(), nullptr);
        std::unique_ptr<Scope> falseScope;
        if (ifOp.elseBlock()) {
          falseScope = funcIrBuilder(ifOp.getElseRegion(), nullptr);
        }
        auto conditionOp = std::make_unique<Condition>(
            &op, scopeOp.get(), std::move(trueScope), std::move(falseScope));
        scopeOp->body.push_back(std::move(conditionOp));
      } else if (isa<LoopLikeOpInterface>(op)) {
        auto loopOp = std::make_unique<Loop>(&op, scopeOp.get());
        for (auto &region : op.getRegions()) {
          auto regionOp = funcIrBuilder(region, loopOp.get());
          loopOp->body.push_back(std::move(regionOp));
        }
        scopeOp->body.push_back(std::move(loopOp));
      } else if (auto mmadl1Op = dyn_cast<hivm::MmadL1Op>(op);
                 mmadl1Op && decomposeMmadl1Op) {
        auto decomposedOp = getDecomposedMmadl1(mmadl1Op, scopeOp.get());
        scopeOp->body.push_back(std::move(decomposedOp));
      } else if (auto pipeOp = dyn_cast<hivm::OpPipeInterface>(op)) {
        hivm::PIPE pipeRead;
        hivm::PIPE pipeWrite;
        if (pipeOp.isSinglePipeOp()) {
          pipeRead = pipeOp.getPipe();
          pipeWrite = pipeOp.getPipe();
        } else {
          pipeRead = pipeOp.getInPipe();
          pipeWrite = pipeOp.getOutPipe();
        }
        auto coreType = hivm::getCoreType(&op);
        auto coreTypeVal = hivm::TCoreType::CUBE_OR_VECTOR;
        if (succeeded(coreType)) {
          coreTypeVal = coreType.value();
        }
        auto [readMemOps, writeMemOps] = getReadWriteMemOps(&op);
        auto rwOp = std::make_unique<RWOperation>(&op, scopeOp.get(), pipeRead,
                                                  pipeWrite, coreTypeVal,
                                                  readMemOps, writeMemOps);
        if (isa<hivm::MmadL1Op, hivm::FixpipeOp>(op)) {
          rwOp->hasUnitFlagFeat = true;
          unitFlagFeaturedOps.insert(rwOp.get());
        }
        scopeOp->body.push_back(std::move(rwOp));
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
        if (auto rwOp = getLoadStoreOp(&storeOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        if (auto rwOp = getLoadStoreOp(&loadOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (auto rwOp = getLoadStoreOp(&storeOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      } else if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        if (auto rwOp = getLoadStoreOp(&loadOp, scopeOp.get())) {
          scopeOp->body.push_back(std::move(rwOp));
        }
      }
    }

    auto ghostOp = std::make_unique<Ghost>(nullptr, scopeOp.get(), &block);
    scopeOp->body.push_back(std::move(ghostOp));
  }

  return scopeOp;
}

// Various processing-order and sync IR builder helpers
// (generateProcessingOrders, syncIrBuilder).
void Solver::generateProcessingOrders(Occurrence *scopeOcc, int l, int r,
                                      bool isUseless) {
  int start = scopeOcc->syncIrIndex;
  int end = scopeOcc->syncIrEndIndex;
  assert(start != -1 && end != -1);
  for (int i = start; i < end; i++) {
    if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
      ProcessingOrder order(syncIr[i].get(), start + 1, i - 1, true, isUseless);
      processingOrders.push_back(order);
    }
  }
  for (int i = r; i >= l; i--) {
    if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
      ProcessingOrder order(syncIr[i].get(), start + 1, end - 1, false,
                            isUseless);
      processingOrders.push_back(order);
    }
  }
}

void Solver::generateProcessingOrders(int l, int r, bool isUseless) {
  for (int i = l; i <= r; i++) {
    if (llvm::isa_and_nonnull<Scope>(syncIr[i]->op)) {
      generateProcessingOrders(syncIr[i].get(), l, i - 1, isUseless);
      assert(syncIr[i]->syncIrIndex == i);
      assert(syncIr[i]->syncIrEndIndex != -1);
      i = syncIr[i]->syncIrEndIndex - 1;
      continue;
    }
    if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
      assert(syncIr[i]->syncIrIndex == i);
      ProcessingOrder order(syncIr[i].get(), l, i - 1, true, isUseless);
      processingOrders.push_back(order);
    }
  }
}

void Solver::generateProcessingOrders(int l1, int r1, int l2, int r2,
                                      bool isUseless) {
  assert(r1 < l2);
  for (int i = l2; i <= r2; i++) {
    if (llvm::isa_and_nonnull<Scope>(syncIr[i]->op)) {
      generateProcessingOrders(syncIr[i].get(), l1, r1, isUseless);
      assert(syncIr[i]->syncIrIndex == i);
      assert(syncIr[i]->syncIrEndIndex != -1);
      i = syncIr[i]->syncIrEndIndex - 1;
      continue;
    }
    if (llvm::isa_and_present<RWOperation>(syncIr[i]->op)) {
      assert(syncIr[i]->syncIrIndex == i);
      ProcessingOrder order(syncIr[i].get(), l1, r1, true, isUseless);
      processingOrders.push_back(order);
    }
  }
}

// Build the linearized sync IR (syncIr) and record occurrence ranges for
// analysis.
void Solver::syncIrBuilder(OperationBase *op, Occurrence *parentOcc, int depth,
                           bool isUseless) {
  assert(op != nullptr);
  int startIndex = globalIndex++;
  auto occ = std::make_unique<Occurrence>(op, parentOcc, depth, startIndex, -1);
  occ->syncIrIndex = static_cast<int>(syncIr.size());
  syncIr.push_back(std::move(occ));
  Occurrence *occPtr = syncIr.back().get();
  opAllOccurrences[op].push_back(occPtr);

  if (parentOcc != nullptr) {
    occChildrenMem[parentOcc].push_back(occPtr);
  }

  if (auto *scopeOp = dyn_cast<Scope>(op)) {

    bool unrollLoop = isa<Loop>(op);
    if (!unrollLoop) {
      for (auto &op : scopeOp->body) {
        syncIrBuilder(op.get(), occPtr, depth + 1, isUseless);
      }
    } else {
      for (auto &op : scopeOp->body) {
        syncIrBuilder(op.get(), occPtr, depth + 1, isUseless);
      }
      occPtr->loopSplitIndex = static_cast<int>(syncIr.size());
      for (auto &op : scopeOp->body) {
        syncIrBuilder(op.get(), occPtr, depth + 1, true);
      }
    }

    if (unrollLoop) {
      generateProcessingOrders(occPtr->syncIrIndex + 1,
                               occPtr->loopSplitIndex - 1, isUseless);

      generateProcessingOrders(occPtr->loopSplitIndex,
                               static_cast<int>(syncIr.size()) - 1, true);

      generateProcessingOrders(occPtr->syncIrIndex + 1,
                               occPtr->loopSplitIndex - 1,
                               occPtr->loopSplitIndex,
                               static_cast<int>(syncIr.size()) - 1, isUseless);
      ProcessingOrder order(nullptr, occPtr->syncIrIndex,
                            occPtr->loopSplitIndex - 1, false, false,
                            /*skip=*/true);
      processingOrders.push_back(order);
    } else if (op->opType == OpType::SCOPE) {
      generateProcessingOrders(occPtr->syncIrIndex + 1,
                               static_cast<int>(syncIr.size()) - 1, isUseless);
    }
  }

  int endIndex = globalIndex++;
  occPtr->endIndex = endIndex;
  occPtr->syncIrEndIndex = static_cast<int>(syncIr.size());
}

// Helpers to find first/last iteration occurrences relative to parent
// occurrences.
Occurrence *Solver::getFirstIterOcc(Occurrence *occ, Occurrence *parOcc) {
  assert(occ != nullptr && parOcc != nullptr);
  if (parOcc->depth + 1 < occ->depth) {
    auto *newParOcc = getFirstIterOcc(
        occ->getNthParent(occ->depth - parOcc->depth - 1), parOcc);
    return getFirstIterOcc(occ, newParOcc);
  }
  auto *it =
      std::find_if(occChildrenMem[parOcc].begin(), occChildrenMem[parOcc].end(),
                   [occ](Occurrence *curOcc) { return occ->op == curOcc->op; });
  assert(it != occChildrenMem[parOcc].end());
  return *it;
}

Occurrence *Solver::getLastIterOcc(Occurrence *occ, Occurrence *parOcc) {
  assert(occ != nullptr && parOcc != nullptr);
  if (parOcc->depth + 1 < occ->depth) {
    auto *newParOcc = getLastIterOcc(
        occ->getNthParent(occ->depth - parOcc->depth - 1), parOcc);
    return getLastIterOcc(occ, newParOcc);
  }
  auto it = std::find_if(
      occChildrenMem[parOcc].rbegin(), occChildrenMem[parOcc].rend(),
      [occ](Occurrence *curOcc) { return occ->op == curOcc->op; });
  assert(it != occChildrenMem[parOcc].rend());
  return *it;
}

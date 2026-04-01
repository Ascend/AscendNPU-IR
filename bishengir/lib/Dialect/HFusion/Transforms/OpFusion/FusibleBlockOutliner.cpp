//===- FusibleBlockOutliner.cpp - Separate fusible blocks from its func ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlockOutliner.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <iterator>

#define DEBUG_TYPE "hfusion-fuse-outliner"

namespace mlir {
namespace hfusion {
namespace opfusion {

FusibleBlockOutliner::FusibleBlockOutliner(FusibleBlocks &fusibleBlocks,
                                           const OutlineFuncOptions &options,
                                           bool shouldRemoveDuplicateAliasOuts)
    : fusibleBlocks_(fusibleBlocks), options_(options) {
  if (options.outputMode == OutputMode::Multiple) {
    if (shouldRemoveDuplicateAliasOuts)
      removeDuplicatedAliasOutputs();
    return;
  }

  LLVM_DEBUG(
      llvm::dbgs()
          << "Separating main fusible block for single output fusion\n";);
  FusibleBlocks newFusibleBlocks;
  for (FusibleBlock &curBlock : fusibleBlocks) {
    SetVector<Operation *> opSet(curBlock.getOps().begin(),
                                 curBlock.getOps().end());

    DenseMap<Operation *, size_t> topoOrder;
    auto allOps = curBlock.getOps();
    LLVM_DEBUG(llvm::dbgs() << "\n Blocks:\n";);
    for (size_t i = 0; i < allOps.size(); i++) {
      LLVM_DEBUG(llvm::dbgs() << *allOps[i] << "\n");
      topoOrder[allOps[i]] = i;
    }
    SetVector<Operation *> currentOutputs;
    for (Value v : curBlock.getOutputs()) {
      // re-fusion non picked fused blocks for single mode is not implemented
      Operation *outOp = v.getDefiningOp();
      currentOutputs.insert(outOp);
    }
    using PairTopoOperation = std::pair<size_t, Operation *>;
    // Gather reachable ops from outOp (backward), outOp is not included.

    // The following invariants hold:
    // - If a fusible blocks A is schedulable, then its connected subgraph
    // inside is schedulable
    // - One node can't be in multiple blocks in SingleMode
    auto collectFusedOps = [&](Operation *outOp) -> SmallVector<Operation *> {
      // Priority queue is needed to maintain the topological order
      // we need it to compute something like
      // A --> B
      // |  /
      // v L
      // C --> outFuse
      //
      // If the order of traversal C, A, B
      // When it's time to relax A, it will not be included
      // because B is not fused to C yet.
      //
      // Need to relax B first, thus forcing topological order
      llvm::PriorityQueue<PairTopoOperation> dijkstraQueue;
      dijkstraQueue.push(PairTopoOperation(topoOrder[outOp], outOp));
      DenseSet<Operation *> newOpped;
      newOpped.insert(outOp);
      while (!dijkstraQueue.empty()) {
        Operation *curOp = dijkstraQueue.top().second;
        dijkstraQueue.pop();
        for (const Value &operand : curOp->getOperands()) {
          Operation *defOp = operand.getDefiningOp();
          if (!defOp)
            continue;
          if (!opSet.contains(defOp))
            continue;

          LLVM_DEBUG(llvm::dbgs() << "Relaxing " << *defOp << "\n");
          // If its within the fused block
          // A --> B --> outFuse
          // |  /
          // v L
          // C --> outFuse
          // will just take all
          bool safeToFuse = true;
          if (options.outputMode == OutputMode::SingleAggressive) {
            // Always safe to fuse
            if (currentOutputs.contains(defOp))
              safeToFuse = false;
          } else if (options.outputMode == OutputMode::Single) {
            // Check if the usage for this is all inside the newOpped
            for (const Value &res : defOp->getResults()) {
              if (!safeToFuse)
                break;
              for (Operation *opUser : res.getUsers()) {
                if (!newOpped.contains(opUser)) {
                  safeToFuse = false;
                  break;
                }
              }
            }
          } else {
            assert(false && "outputMode not handled");
          }
          if (safeToFuse && !newOpped.contains(defOp)) {
            dijkstraQueue.push(PairTopoOperation(topoOrder[defOp], defOp));
            newOpped.insert(defOp);
          }
        }
      }

      SmallVector<Operation *> sortedNewOp(newOpped.begin(), newOpped.end());
      std::sort(sortedNewOp.begin(), sortedNewOp.end(),
                [&](Operation *opA, Operation *opB) {
                  return topoOrder[opA] < topoOrder[opB];
                });

      return sortedNewOp;
    };
    auto tmpOut = curBlock.getOutputs();
    for (Value v : tmpOut) {
      Operation *outOp = v.getDefiningOp();
      // re-fusion non picked fused blocks for single mode is not implemented
      const SmallVector<Operation *> &fusedOps = collectFusedOps(outOp);

      if (fusedOps.size() == 1)
        continue;
      SmallVector<Operation *, 1> tmpOutOp = {outOp};
      // Generating new ops
      newFusibleBlocks.emplace_back(fusedOps, curBlock.fusibleHelper_,
                                    tmpOutOp);
    }
  }
  fusibleBlocks_.swap(newFusibleBlocks);
}

void FusibleBlockOutliner::removeDuplicatedAliasOutputs() {
  FusibleBlocks newFusibleBlocks;
  for (FusibleBlock &curBlock : fusibleBlocks_) {
    // Remove duplicated operations that is not allowed here
    // If there's 2 consecutive reshapes then its undefined behavior
    // e.g: elemwise -> reshape -> reshape
    // Two aliases...

    // Assumptions, the reshape operations is already at the end,
    // no need to worry about the middle reshapes

    auto currentOutput = curBlock.getOutputs();
    llvm::DenseSet<Value> setOutput(currentOutput.begin(), currentOutput.end());
    llvm::DenseSet<Operation *> throwOps;
    for (Value v : currentOutput) {
      Operation *outOp = v.getDefiningOp();
      if (!isReshapeOp(outOp))
        continue;
      auto sourceReshape = getReshapeSource(outOp);
      // Ignore aliases
      if (setOutput.count(sourceReshape)) {
        throwOps.insert(outOp);
      }
    }
    SmallVector<Operation *> newOps;
    for (Operation *op : curBlock.getOps()) {
      if (throwOps.count(op))
        continue;
      newOps.push_back(op);
    }

    // If single outline op, this will be called,
    // because new Fusible block is assigned manually
    newFusibleBlocks.emplace_back(curBlock.getOps(), curBlock.fusibleHelper_,
                                  newOps);
  }

  fusibleBlocks_.swap(newFusibleBlocks);
}

SmallVector<func::FuncOp> FusibleBlockOutliner::getOutlinedFuncs() const {
  return outlinedFuncs_;
}

bool FusibleBlockOutliner::outline(const std::string &prefixOutline) {
  for (FusibleBlock &curBlock : fusibleBlocks_) {
    func::FuncOp fusedFunc = outlineFunc(curBlock, prefixOutline);
    if (!fusedFunc)
      return false;
    outlinedFuncs_.push_back(fusedFunc);

    func::CallOp fusionInvoke = createInvoke(fusedFunc, curBlock);
    if (!fusionInvoke)
      return false;
  }
  return true;
}

void FusibleBlockOutliner::setOutlineFuncAttributes(
    func::FuncOp &func, const FusionKind &fusionKind, OpBuilder &builder,
    bool isCallerHost) {
  func->setAttr(FusionKindAttr::name,
                FusionKindAttr::get(func->getContext(), fusionKind));
  // Set outlined function to be a device function.
  hacc::utils::setDevice(func);
  // If the caller is a host function, the device function has to be an entry.
  if (isCallerHost)
    hacc::utils::setDeviceEntry(func);
}

std::string FusibleBlockOutliner::getNewFusionName(llvm::StringRef symbolName,
                                                   llvm::StringRef prefixName) {
  return symbolName.str() + prefixName.str() + "_" + std::to_string(funcCnt_++);
}

void FusibleBlockOutliner::eraseTriviallyDeadOps(ArrayRef<Operation *> ops) {
  for (auto I = ops.rbegin(), E = ops.rend(); I != E; ++I) {
    Operation *curOp = *I;
    if (isOpTriviallyDead(curOp))
      curOp->erase();
  }
}

size_t
FusibleBlockOutliner::getNumOutsideUsesOfOp(SetVector<Operation *> &opsWithAuxs,
                                            Value out) const {
  return count_if(out.getUses(), [&opsWithAuxs](auto &use) {
    return !opsWithAuxs.contains(use.getOwner());
  });
}

func::FuncOp
FusibleBlockOutliner::outlineFunc(FusibleBlock &curBlock,
                                  const std::string &prefixOutline) {
  func::FuncOp parF = curBlock.getParentOfType<func::FuncOp>();
  OpBuilder curBuilder(parF.getContext());
  OpBuilder::InsertionGuard insGuard(curBuilder);
  curBuilder.setInsertionPoint(parF);
  // Create function prototype
  SmallVector<Type> outTypes;
  SetVector<Operation *> opsWithAuxs(curBlock.getOpWithAuxs().begin(),
                                     curBlock.getOpWithAuxs().end());

  for (Value out : curBlock.getOutputs())
    outTypes.append(options_.shouldKeepFuncSignature
                        ? getNumOutsideUsesOfOp(opsWithAuxs, out)
                        : 1,
                    out.getType());

  FunctionType funcTy = FunctionType::get(
      parF.getContext(), TypeRange(ValueRange(curBlock.getInputs())), outTypes);
  func::FuncOp newFunc = curBuilder.create<func::FuncOp>(
      curBlock.getLoc(), getNewFusionName(parF.getSymName(), prefixOutline),
      funcTy);
  setOutlineFuncAttributes(newFunc, curBlock.fusibleHelper_->getFusionKind(),
                           curBuilder, hacc::utils::isHost(parF));
  if (options_.alwaysInline)
    hacc::utils::setAlwaysInline(newFunc);

  // Create function body
  Block *entryBB = newFunc.addEntryBlock();
  curBuilder.setInsertionPointToStart(entryBB);

  // Clone operations and replace usages
  IRMapping curMap;
  for (auto [oldIn, newIn] :
       llvm::zip(curBlock.getInputs(), entryBB->getArguments()))
    curMap.map(oldIn, newIn);

  SetVector<Operation *> newOps;
  for (Operation *op : curBlock.getOpWithAuxs()) {
    newOps.insert(curBuilder.clone(*op, curMap));
    LLVM_DEBUG(llvm::dbgs() << "Cloning " << *op << "\n";);
  }

  SmallVector<Value> outs;
  for (Value out : curBlock.getOutputs()) {
    assert(curMap.getValueMap().contains(out));
    outs.append(options_.shouldKeepFuncSignature
                    ? getNumOutsideUsesOfOp(opsWithAuxs, out)
                    : 1,
                curMap.getValueMap().at(out));
  }
#ifndef NDEBUG
  if (options_.shouldKeepFuncSignature)
    assert(outs.size() == outTypes.size() &&
           "outs size should be equal to outTypes size");
#endif

  curBuilder.create<func::ReturnOp>(curBlock.getLoc(), ValueRange(outs));
  eraseTriviallyDeadOps(newOps.getArrayRef());
  return newFunc;
}

func::CallOp FusibleBlockOutliner::createInvoke(func::FuncOp newFunc,
                                                FusibleBlock &fusionBlock) {
  OpBuilder curBuilder(newFunc.getContext());
  OpBuilder::InsertionGuard insGuard(curBuilder);

  curBuilder.setInsertionPointAfter(fusionBlock.getLastOp());

  func::CallOp newInvoke = curBuilder.create<func::CallOp>(
      fusionBlock.getLoc(), newFunc, fusionBlock.getInputs());

  if (options_.shouldKeepFuncSignature) {
    size_t returnIdx = 0;
    SetVector<Operation *> opsWithAuxs;
    for (auto &op : fusionBlock.getOpWithAuxs())
      opsWithAuxs.insert(op);

    for (auto fusionBlockOut : fusionBlock.getOutputs()) {
      SmallVector<OpOperand *> uses;
      for (auto &use : fusionBlockOut.getUses()) {
        if (opsWithAuxs.contains(use.getOwner()))
          continue;
        uses.push_back(&use);
      }
      std::reverse(uses.begin(), uses.end());
      for (auto &use : llvm::make_early_inc_range(uses)) {
        use->set(newInvoke->getResult(returnIdx++));
      }
    }
  } else {
    for (auto [oldOut, newOut] :
         llvm::zip(fusionBlock.getOutputs(), newInvoke->getResults()))
      ((Value)oldOut).replaceAllUsesWith(newOut);
  }

  Block *curBlock = curBuilder.getInsertionBlock();
  if (!curBlock->verifyOpOrder())
    sortTopologically(curBlock);

  eraseTriviallyDeadOps(fusionBlock.getOpWithAuxs());
  return newInvoke;
}
} // namespace opfusion
} // namespace hfusion
} // namespace mlir

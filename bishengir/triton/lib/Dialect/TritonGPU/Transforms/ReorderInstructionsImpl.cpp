#include "triton/Dialect/TritonGPU/Transforms/ReorderInstructionsImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <iterator>

namespace mlir {
namespace triton {

Value ReorderInstructionsImpl::traceToBlockOperand(BlockArgument arg,
                                                   Operation *parentOp) {
  auto argIndex = arg.getArgNumber();
  if (auto forOp = llvm::dyn_cast<scf::ForOp>(parentOp)) {
    return forOp.getInitArgs()[argIndex - 1];
  } else if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(parentOp)) {
    if (arg.getOwner()->getParent() == &whileOp.getBefore()) {
      return whileOp.getInits()[argIndex];
    }
    auto condOp = llvm::dyn_cast<scf::ConditionOp>(
        whileOp.getBefore().front().getTerminator());
    return condOp.getArgs()[argIndex];
  }
  return nullptr;
}

Value ReorderInstructionsImpl::traceToPtr(Value val, FuncOp funcOp) {
  if (!val) {
    return nullptr;
  }
  if (!val.getDefiningOp()) {
    if (llvm::isa<PointerType>(val.getType()) &&
        llvm::is_contained(funcOp.getArguments(), val)) {
      return val;
    }
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
      auto blockOperand = traceToBlockOperand(
          blockArg, blockArg.getParentRegion()->getParentOp());
      return traceToPtr(blockOperand, funcOp);
    }
  } else if (auto addPtr = llvm::dyn_cast<AddPtrOp>(val.getDefiningOp())) {
    return traceToPtr(addPtr.getPtr(), funcOp);
  } else if (auto splatOp = llvm::dyn_cast<SplatOp>(val.getDefiningOp())) {
    return traceToPtr(splatOp.getSrc(), funcOp);
  } else if (auto broadcastOp =
                 llvm::dyn_cast<BroadcastOp>(val.getDefiningOp())) {
    return traceToPtr(broadcastOp.getSrc(), funcOp);
  } else if (auto bitcastOp = llvm::dyn_cast<BitcastOp>(val.getDefiningOp())) {
    return traceToPtr(bitcastOp.getSrc(), funcOp);
  } else if (auto advanceOp = llvm::dyn_cast<AdvanceOp>(val.getDefiningOp())) {
    return traceToPtr(advanceOp.getPtr(), funcOp);
  } else if (auto expandDimsOp =
                 llvm::dyn_cast<ExpandDimsOp>(val.getDefiningOp())) {
    return traceToPtr(expandDimsOp.getSrc(), funcOp);
  } else if (auto reshapeOp = llvm::dyn_cast<ReshapeOp>(val.getDefiningOp())) {
    return traceToPtr(reshapeOp.getSrc(), funcOp);
  } else if (auto convertLayoutOp =
                 llvm::dyn_cast<triton::gpu::ConvertLayoutOp>(
                     val.getDefiningOp())) {
    return traceToPtr(convertLayoutOp.getSrc(), funcOp);
  }
  return nullptr;
}

LogicalResult ReorderInstructionsImpl::visit(LoadOp loadOp) {
  auto ptr = traceToPtr(loadOp.getPtr(), loadOp->getParentOfType<FuncOp>());
  if (!ptr) {
    return failure();
  }
  if (!llvm::is_contained(ptrReadByOps, ptr)) {
    ptrReadByOps[ptr] = llvm::DenseSet<Operation *>();
  }
  ptrReadByOps[ptr].insert(loadOp);
  memReadOps[loadOp].insert(ptr);
  return success();
}

LogicalResult ReorderInstructionsImpl::visit(StoreOp storeOp) {
  auto ptr = traceToPtr(storeOp.getPtr(), storeOp->getParentOfType<FuncOp>());
  if (!ptr) {
    return failure();
  }
  if (!llvm::is_contained(ptrWriteByOps, ptr)) {
    ptrWriteByOps[ptr] = llvm::DenseSet<Operation *>();
  }
  ptrWriteByOps[ptr].insert(storeOp);
  memWriteOps[storeOp].insert(ptr);
  return success();
}

LogicalResult ReorderInstructionsImpl::getOpMemoryEffects(Operation *op) {
  llvm::DenseSet<Value> memReadVals, memWriteVals;
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      for (auto &nestedOp : block) {
        memReadVals.insert_range(memReadOps[&nestedOp]);
        memWriteVals.insert_range(memWriteOps[&nestedOp]);
      }
    }
  }
  memReadOps[op] = memReadVals;
  memWriteOps[op] = memWriteVals;
  if (auto loadOp = llvm::dyn_cast<LoadOp>(op)) {
    if (failed(visit(loadOp))) {
      return failure();
    }
  } else if (auto storeOp = llvm::dyn_cast<StoreOp>(op)) {
    if (failed(visit(storeOp))) {
      return failure();
    }
  } else {
    auto memEffectsInterface = llvm::dyn_cast<MemoryEffectOpInterface>(op);
    if (memEffectsInterface) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffectsInterface.getEffects(effects);
      if (effects.size() > 0) {
        return failure();
      }
    }
  }
  return success();
}

void ReorderInstructionsImpl::constructTopoGraph(
    Block &block, llvm::DenseMap<Operation *, int64_t> &outDegree,
    llvm::DenseSet<Operation *> &tailNodes) {
  for (auto &nestedOp : block) {
    outDegree[&nestedOp] =
        std::distance(nestedOp.getUsers().begin(), nestedOp.getUsers().end());
    if (outDegree[&nestedOp] == 0) {
      tailNodes.insert(&nestedOp);
    }
  }
}

int64_t ReorderInstructionsImpl::calculateRegisterUsage(
    Operation *op, llvm::DenseMap<Operation *, int64_t> outDegree) {
  int64_t registerUsage = 0;
  llvm::DenseSet<Operation *> curOps{op};
  llvm::DenseSet<Operation *> operands;
  while (!curOps.empty()) {
    for (auto curOp : curOps) {
      for (auto operand : curOp->getOperands()) {
        auto defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() != curOp->getBlock()) {
          continue;
        }
        outDegree[defOp]--;
        operands.insert(defOp);
        if (outDegree[curOp] == 0) {
          continue;
        }
        for (auto res : curOp->getResults()) {
          if (auto tensorType =
                  llvm::dyn_cast<RankedTensorType>(res.getType())) {
            auto enc = tensorType.getEncoding();
            if (auto distributedEnc =
                    llvm::dyn_cast<mlir::triton::gpu::DistributedEncodingTrait>(
                        enc)) {
              registerUsage += product(
                  distributedEnc.getElemsPerThread(tensorType.getShape()));
            } else {
              // unsupport layout
              return -1;
            }
          } else {
            registerUsage++;
          }
        }
      }
    }
    curOps.clear();
    std::swap(curOps, operands);
  }
  return registerUsage;
}

bool ReorderInstructionsImpl::isReadConflict(
    Operation *op, llvm::DenseSet<Operation *> &reorderedOps) {
  // check RAW
  // if read after write, we should maintain the order after reorder
  // instructions.
  for (auto ptr : memReadOps[op]) {
    for (auto maybeConflictOp : ptrWriteByOps[ptr]) {
      if (maybeConflictOp->getBlock() != op->getBlock()) {
        continue;
      }
      if (maybeConflictOp->isBeforeInBlock(op) &&
          !llvm::is_contained(reorderedOps, maybeConflictOp)) {
        return true;
      }
    }
  }
  return false;
}

bool ReorderInstructionsImpl::isWriteConflict(
    Operation *op, llvm::DenseSet<Operation *> &reorderedOps) {
  for (auto ptr : memWriteOps[op]) {
    // check write-write
    // we should maintain write ops relative order, which write to the same
    // ptr.
    for (auto maybeConflictOp : ptrWriteByOps[ptr]) {
      if (maybeConflictOp->getBlock() != op->getBlock()) {
        continue;
      }
      if (maybeConflictOp->isBeforeInBlock(op) &&
          !llvm::is_contained(reorderedOps, maybeConflictOp)) {
        return true;
      }
    }
    // check RAW
    for (auto maybeConflictOp : ptrReadByOps[ptr]) {
      if (maybeConflictOp->getBlock() != op->getBlock()) {
        continue;
      }
      if (maybeConflictOp->isBeforeInBlock(op) &&
          !llvm::is_contained(reorderedOps, maybeConflictOp)) {
        return true;
      }
    }
  }
  return false;
}

Operation *ReorderInstructionsImpl::greedySelectSubgraph(
    llvm::SmallVector<std::pair<int64_t, Operation *>> &registerUsage,
    llvm::DenseSet<Operation *> &reorderedOps) {
  // Prefer to select the subgraph use lowest registerNum
  for (auto [registerNum, tailNode] : registerUsage) {
    llvm::DenseSet<Operation *> curOps{tailNode};
    llvm::DenseSet<Operation *> operands;
    bool isConflict = false;
    while (!curOps.empty()) {
      for (auto curOp : curOps) {
        if (llvm::is_contained(memReadOps, curOp) &&
            isReadConflict(curOp, reorderedOps)) {
          isConflict = true;
          break;
        }
        if (llvm::is_contained(memWriteOps, curOp) &&
            isWriteConflict(curOp, reorderedOps)) {
          isConflict = true;
          break;
        }
        for (auto operand : curOp->getOperands()) {
          auto defOp = operand.getDefiningOp();
          if (defOp && defOp->getBlock() == curOp->getBlock()) {
            operands.insert(defOp);
          }
        }
      }
      if (isConflict) {
        break;
      }
      curOps.clear();
      std::swap(curOps, operands);
    }
    if (!isConflict) {
      return tailNode;
    }
  }
  return nullptr;
}

void ReorderInstructionsImpl::recursiveInsertOpTree(
    OpBuilder &builder, IRMapping &mapping, Operation *curNode,
    llvm::DenseSet<Operation *> &reorderedOps) {
  if (reorderedOps.contains(curNode)) {
    return;
  }
  // First visit curNode's operands
  for (auto operand : curNode->getOperands()) {
    auto defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == curNode->getBlock()) {
      recursiveInsertOpTree(builder, mapping, defOp, reorderedOps);
    }
  }
  // if curNode has subRegion, walk it and if subRegion use outer variable,
  // visit it.
  curNode->walk(
      [curNode, &reorderedOps, &builder, &mapping, this](Operation *op) {
        if (op == curNode) {
          return;
        }
        for (auto operand : op->getOperands()) {
          auto defOp = operand.getDefiningOp();
          if (defOp && defOp->getBlock() == curNode->getBlock() &&
              !llvm::is_contained(reorderedOps, defOp)) {
            recursiveInsertOpTree(builder, mapping, defOp, reorderedOps);
          }
        }
      });
  // copy and insert it to new block.
  reorderedOps.insert(curNode);
  builder.clone(*curNode, mapping);
}

void ReorderInstructionsImpl::reorderInstructions(
    Block &block, llvm::DenseMap<Operation *, int64_t> &outDegree,
    llvm::DenseSet<Operation *> &tailNodes, OpBuilder &builder) {
  llvm::SmallVector<std::pair<int64_t, Operation *>> registerUsage;
  llvm::DenseSet<Operation *> reorderedOps;
  IRMapping mapping;
  Block cloned;
  Operation *terminator = nullptr;
  for (auto tailNode : tailNodes) {
    if (tailNode->hasTrait<OpTrait::IsTerminator>()) {
      // one block should have only one terminator
      if (terminator != nullptr) {
        return;
      }
      terminator = tailNode;
    }
  }
  // terminator should be inserted finally.
  tailNodes.erase(terminator);
  builder.setInsertionPointToStart(&cloned);
  while (!tailNodes.empty()) {
    for (auto tailNode : tailNodes) {
      auto registerNum = calculateRegisterUsage(tailNode, outDegree);
      if (registerNum == -1) {
        return;
      }
      registerUsage.emplace_back(std::make_pair(registerNum, tailNode));
    }
    llvm::sort(registerUsage);
    auto tailNode = greedySelectSubgraph(registerUsage, reorderedOps);
    if (tailNode == nullptr) {
      return;
    }
    recursiveInsertOpTree(builder, mapping, tailNode, reorderedOps);
    tailNodes.erase(tailNode);
    registerUsage.clear();
  }
  recursiveInsertOpTree(builder, mapping, terminator, reorderedOps);
  llvm::SmallVector<Operation *> opToDelete;
  for (auto op = block.getOperations().rbegin();
       op != block.getOperations().rend(); op++) {
    opToDelete.emplace_back(&(*op));
  }
  for (size_t i = 0; i < opToDelete.size(); i++) {
    opToDelete[i]->erase();
  }
  block.getOperations().splice(block.begin(), cloned.getOperations());
}

void ReorderInstructionsImpl::commonInstructionReorder(ModuleOp m,
                                                       MLIRContext *ctx) {
  /* reorder instructions and short register life cycle
  we consider this:
  %0 = tt.load ... use 1 register
  %1 = tt.load ... use 2 registers
  tt.store %0 ...use 1 registers
  tt.store %1 ...use 0 registers

  after reorder:
  %0 = tt.load ... use 1 register
  tt.store %0 ... use 0 register
  %1 = tt.load ... use 1 register
  tt.store %1 ... use 0 register
  */
  bool hasMultiBasicBlock = false;
  m->walk([&hasMultiBasicBlock](Operation *op) {
    for (auto &region : op->getRegions()) {
      if (!region.hasOneBlock() && !region.empty()) {
        hasMultiBasicBlock = true;
        break;
      }
    }
  });
  if (hasMultiBasicBlock) {
    // not support multiple basic block
    return;
  }
  OpBuilder builder(ctx);
  llvm::SmallVector<Operation *> workQueue;
  m->walk<WalkOrder::PostOrder>([this, m, &workQueue](Operation *op) {
    if (op == m) {
      return WalkResult::advance();
    }
    if (succeeded(getOpMemoryEffects(op))) {
      workQueue.emplace_back(op);
      return WalkResult::advance();
    }
    return WalkResult::interrupt();
  });
  for (auto op : workQueue) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        llvm::DenseMap<Operation *, int64_t> outDegree;
        llvm::DenseSet<Operation *> tailNodes;
        constructTopoGraph(block, outDegree, tailNodes);
        reorderInstructions(block, outDegree, tailNodes, builder);
      }
    }
  }
}

} // namespace triton
} // namespace mlir

//===-------- ReorderOpsByBFS.h - reorder the ops by bfs --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/ReorderOpsByBFS.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <queue>
#include <set>

#define DEBUG_TYPE "hfusion-reorder-ops"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace mlir;

namespace mlir {
namespace hfusion {

#define GEN_PASS_DEF_REORDEROPSBYBFS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

namespace {
class OpReorderer {
public:
  explicit OpReorderer(Block *block, std::map<Operation *, int> *operationIdx)
      : block(block), operationIdx(operationIdx){};
  using ValueSetType = SmallVector<Value>;
  bool valLessThan(const mlir::Value &lhs, const mlir::Value &rhs) {
    // First, compare by operation ID
    auto *opL = lhs.getDefiningOp();
    auto *opR = rhs.getDefiningOp();
    if (opL && opR) {
      if (opL == opR) {
        return cast<OpResult>(lhs).getResultNumber() <
               cast<OpResult>(rhs).getResultNumber();
      }
      LLVM_DEBUG(llvm::dbgs() << *opL << " " << *opR << "\n";);
      assert(operationIdx->count(opL));
      assert(operationIdx->count(opR));
      return operationIdx->at(opL) < operationIdx->at(opR);
    }
    if (opL)
      return false;
    if (opR)
      return true;
    return cast<BlockArgument>(lhs).getArgNumber() <
           cast<BlockArgument>(rhs).getArgNumber();
  }

  void reorderOps(llvm::iterator_range<Block::iterator> ops);

private:
  Block *block;
  std::queue<Operation *> visitedOps;
  std::map<Operation *, int> ingreeMap;
  Operation *anchorOp = nullptr;
  std::map<Operation *, int> *operationIdx;
  void ensureOrderAndUnique(ValueSetType &v) {
    llvm::sort(v.begin(), v.end(), [&](const Value &a, const Value &b) {
      return valLessThan(a, b);
    });
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }
  bool isComputingOp(Operation *op);
  void enqueueOp(Operation *op);
  void traverse(Value arg);
  size_t getIngreeOfOp(Operation *op);
  void getOpNestedInVarsFromOutside(Block *pBlock, ValueSetType &ins,
                                    Liveness &liveness);
  ValueSetType getOpNestedInVarsFromOutside(Operation *op, Liveness &liveness);
};

bool OpReorderer::isComputingOp(Operation *op) {
  if (isa<linalg::LinalgOp>(op))
    return true;

  bool hasComputingOp = false;
  op->walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op)) {
      hasComputingOp = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasComputingOp;
}

void OpReorderer::enqueueOp(Operation *op) {
  LDBG("enqueue op : \n" << *op);
  if (anchorOp == nullptr) {
    op->moveBefore(block, block->getOperations().begin());
  } else {
    op->moveAfter(anchorOp);
  }
  LDBG("changed block : \n");
  LLVM_DEBUG(block->print(llvm::dbgs()));

  ingreeMap.erase(op);
  op->walk([&](Operation *inOp) {
    ingreeMap.erase(inOp);
    return WalkResult::advance();
  });

  anchorOp = op;
  if (!isComputingOp(op)) {
    LDBG(*op << " is not compute op, continue to find compute op");
    for (auto res : op->getResults()) {
      traverse(res);
    }
  } else {
    visitedOps.push(op);
  }
}

void OpReorderer::traverse(Value arg) {
  LDBG("traverse users of arg : \n" << arg);
  SmallVector<Operation *> users;
  for (auto *user : arg.getUsers()) {
    users.push_back(user);
  }
  sort(users.begin(), users.end());
  users.erase(std::unique(users.begin(), users.end()), users.end());
  sort(users.begin(), users.end(), [&](Operation *a, Operation *b) {
    assert(operationIdx->count(a));
    assert(operationIdx->count(b));
    return operationIdx->at(a) < operationIdx->at(b);
  });
  for (auto *user : users) {
    if (user->hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }

    Operation *processOp = block->findAncestorOpInBlock(*user);
    if (!processOp)
      continue;

    if (user != processOp && isa<linalg::LinalgOp>(processOp)) {
      continue;
    }

    LDBG("traverse op : \n" << *user);
    assert(!user->getOperands().empty() &&
           "no operands ops should be processed at the beginning");

    if (ingreeMap.find(user) == ingreeMap.end()) {
      ingreeMap[processOp] = static_cast<int64_t>(getIngreeOfOp(processOp));
    }

    ingreeMap[processOp]--;

    LDBG("update ingreeMap of " << *processOp << " : " << ingreeMap[processOp]
                                << "\n");

    if (ingreeMap[processOp] == 0) {
      enqueueOp(user);
    }
  }
}

size_t OpReorderer::getIngreeOfOp(Operation *op) {
  LDBG("compute ingree of compound op : " << *op);
  if (op->getRegions().empty() || isa<linalg::LinalgOp>(op)) {
    auto ingree = llvm::DenseSet<Value>(op->getOperands().begin(),
                                        op->getOperands().end())
                      .size();
    LDBG("ingree of compound op is " << ingree);
    return ingree;
  }

  Liveness liveness(op);
  ValueSetType outSideVars = getOpNestedInVarsFromOutside(op, liveness);
  LLVM_DEBUG(llvm::dbgs() << "Current op " << *op << "\n";);
  for (auto oper : op->getOperands()) {
    LDBG("push_backed var : " << oper);
    outSideVars.push_back(oper);
  }
  ensureOrderAndUnique(outSideVars);
  LDBG("ingree of compound op is " << outSideVars.size());
  return outSideVars.size();
}

void OpReorderer::getOpNestedInVarsFromOutside(Block *pBlock, ValueSetType &ins,
                                               Liveness &liveness) {
  llvm::DenseSet<Value> blockArgs;
  pBlock->walk([&](Block *nestedBlock) {
    blockArgs.insert(nestedBlock->getArguments().begin(),
                     nestedBlock->getArguments().end());
    return WalkResult::advance();
  });

  ValueSetType liveIns = llvm::to_vector(liveness.getLiveIn(pBlock));
  ensureOrderAndUnique(liveIns);
  for (auto var : liveIns) {
    if (!blockArgs.contains(var)) {
      LDBG("push_backed var : " << var);
      ins.push_back(var);
    }
  }
  ensureOrderAndUnique(ins);
}

OpReorderer::ValueSetType
OpReorderer::getOpNestedInVarsFromOutside(Operation *op, Liveness &liveness) {
  ValueSetType ins;
  for (auto &region : op->getRegions()) {
    for (auto &curBlock : region.getBlocks()) {
      getOpNestedInVarsFromOutside(&curBlock, ins, liveness);
    }
  }
  ensureOrderAndUnique(ins);
  return ins;
}

void OpReorderer::reorderOps(llvm::iterator_range<Block::iterator> ops) {
  if (ops.empty())
    return;

  LDBG("reorder block : \n");
  LLVM_DEBUG(block->print(llvm::dbgs()));

  // Find the ingree is 0 ops and push it to visitedOps first
  for (auto &op : block->getOperations()) {
    LLVM_DEBUG(llvm::dbgs() << "getOperations ? " << op << "\n";);
    Operation *processOp = block->findAncestorOpInBlock(op);
    if (processOp == nullptr || processOp != &op ||
        op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    if (getIngreeOfOp(&op) == 0) {
      enqueueOp(&op);
    }
  }

  LDBG("traverse from outside vars");
  auto *blockParentOp = block->getParentOp();
  LDBG("parentOp : " << *blockParentOp);
  Liveness liveness(blockParentOp);
  ValueSetType varsFromOutside;
  getOpNestedInVarsFromOutside(block, varsFromOutside, liveness);
  ValueSetType args = llvm::map_to_vector(
      block->getArguments(), [&](BlockArgument a) { return (Value)a; });
  ensureOrderAndUnique(args);
  for (auto arg : args) {
    traverse(arg);
  }
  ensureOrderAndUnique(varsFromOutside);
  for (auto arg : varsFromOutside) {
    LLVM_DEBUG(llvm::dbgs() << " ok looping " << arg << "\n";);
    traverse(arg);
  }

  LDBG("traverse further by bfs");
  while (!visitedOps.empty()) {
    auto *op = visitedOps.front();
    visitedOps.pop();
    for (auto res : op->getResults()) {
      traverse(res);
    }
  }
}

struct ReorderOpsByBFS : public impl::ReorderOpsByBFSBase<ReorderOpsByBFS> {
  void runOnOperation() final;
};
} // namespace

void reorderOpsByBFS(func::FuncOp funcOp) {
  std::map<Operation *, int> operationIdx;
  funcOp.walk([&operationIdx](Operation *op) {
    // 安全转换：operationIdx.size()返回size_t，用于索引分配，使用static_cast显式转换
    int curSize = static_cast<int>(operationIdx.size());
    LLVM_DEBUG(llvm::dbgs() << *op << "--" << curSize << "\n";;);
    operationIdx[op] = curSize;
  });
  funcOp.walk([&operationIdx](Region *op) {
    auto *parentOp = op->getParentOp();
    if (isa<linalg::LinalgOp>(parentOp))
      return WalkResult::skip();

    for (Block &block : op->getBlocks()) {
      LLVM_DEBUG(llvm::dbgs() << "here\n";);
      OpReorderer reorderer(&block, &operationIdx);
      if (block.back().hasTrait<OpTrait::IsTerminator>())
        reorderer.reorderOps(block.without_terminator());
      else
        reorderer.reorderOps(block);
    }
    return WalkResult::advance();
  });
}

void ReorderOpsByBFS::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  hfusion::reorderOpsByBFS(funcOp);
}

std::unique_ptr<Pass> createReorderOpsByBFS() {
  return std::make_unique<ReorderOpsByBFS>();
}

} // namespace hfusion
} // namespace mlir

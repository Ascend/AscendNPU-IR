
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "bishengir/Dialect/Utils/ReachabilityAnalyzer.h"
#include "llvm/Support/ErrorHandling.h"

#include <queue>

#define DEBUG_TYPE "reachability-analyzer"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace utils {

bool hasMemrefSides(Operation *op) {
  if (auto memRefUser = dyn_cast<MemoryEffectOpInterface>(op)) {
    // If the current op is a MemoryEffectOpInterface, propagate its effects
    SmallVector<MemoryEffects::EffectInstance> effects;
    bool hasMemRefEffect = memRefUser.hasEffect<MemoryEffects::Read>();
    hasMemRefEffect |= memRefUser.hasEffect<MemoryEffects::Write>();
    hasMemRefEffect |= memRefUser.hasEffect<MemoryEffects::Allocate>();
    return hasMemRefEffect;
  }
  return false;
}

inline void ReachabilityAnalyzer::getAndSetMemrefEdge(Operation *op,
                                                      Value ref) {
  int curIdx = opToIndexMap[op];
  if (lastOperationOnMemref.count(ref)) {
    if (curIdx == lastOperationOnMemref[ref])
      return;
    if (curIdx > lastOperationOnMemref[ref])
      edge[lastOperationOnMemref[ref]].push_back(curIdx);
    else
      llvm_unreachable("should be DAG");
  }
  lastOperationOnMemref[ref] = curIdx;
}

void ReachabilityAnalyzer::getMemrefFromOp(Operation *op) {
  if (cachedMemref.count(op)) {
    for (const auto ref : cachedMemref[op])
      getAndSetMemrefEdge(op, ref);
  }
  auto &ret = cachedMemref[op];
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    getAndSetMemrefEdge(op, loadOp.getMemRef());
  }
  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    getAndSetMemrefEdge(op, storeOp.getMemRef());
  }
  if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
    getAndSetMemrefEdge(op, allocOp.getMemref());
  }
  for (auto opr : op->getOperands())
    if (isa<BaseMemRefType>(opr.getType())) {
      getAndSetMemrefEdge(op, opr);
      ret.insert(opr);
    }
}

/// get all operations related to the same memref of the op
SmallVector<int> ReachabilityAnalyzer::getUsersOrRoot(Operation *op) {
  SmallVector<int> userIdxs;
  if (op == rootOp) {
    for (Region &region : rootOp->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument &arg : block.getArguments()) {
          for (Operation *user : arg.getUsers()) {
            auto it = opToIndexMap.find(user);
            if (it != opToIndexMap.end()) {
              userIdxs.push_back(it->second);
            }
          }
        }
      }
    }
  } else if (hasMemrefSides(op)) {
    getMemrefFromOp(op); // Store cache :)
  } else {
    for (const auto *user : op->getUsers()) {
      auto it = opToIndexMap.find(user);
      if (it != opToIndexMap.end()) {
        userIdxs.push_back(it->second);
      }
    }
  }
  return userIdxs;
}

ReachabilityAnalyzer::ReachabilityAnalyzer(Operation *parent) {
  // Helper function to get users or root block arguments
  // Collect operations, starting with the parent as root
  rootOp = parent;
  if (rootOp->getRegions().size() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "Currently not supporting not one region";);
    return;
  }
  if (!rootOp->getRegions().front().hasOneBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "Currently not supporting not one block";);
    return;
  }
  operationList.push_back(parent); // Parent is the root with index 0
  for (Region &region : parent->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        operationList.push_back(&op);
      }
    }
  }
  // Initialize opToIndexMap and reachabilityMatrix
  size_t numOps = operationList.size();
  for (size_t i = 0; i < numOps; ++i) {
    opToIndexMap[operationList[i]] = static_cast<int>(i);
  }
  initializeAdjacencyList(numOps);
  computeReachabilityMatrix(numOps);
}

void ReachabilityAnalyzer::initializeAdjacencyList(size_t numOps) {
  edge.resize(numOps);
  // Build the adjacency list using getUsersOrRoot
  for (size_t i = 0; i < numOps; ++i) {
    Operation *op = operationList[i];
    LLVM_DEBUG(llvm::dbgs() << "Current op " << *op << "\n";);
    for (const int v : getUsersOrRoot(op)) {
      edge[i].push_back(v);
    }
  }
  for (auto &adjList : edge) {
    llvm::sort(adjList.begin(), adjList.end());
    adjList.erase(std::unique(adjList.begin(), adjList.end()), adjList.end());
  }
}

void ReachabilityAnalyzer::computeReachabilityMatrix(size_t numOps) {

  reachabilityMatrix.resize(numOps, SmallVector<int64_t>(numOps, kMaxDistance));

  // BFS to compute reachability
  for (size_t start = 0; start < numOps; ++start) {
    std::queue<int64_t> posQueue;
    posQueue.push(static_cast<int64_t>(start));
    reachabilityMatrix[start][start] = 0;

    while (!posQueue.empty()) {
      int current = posQueue.front();
      posQueue.pop();

      assert(current >= 0 && current < ssize_t(numOps) &&
             "Current index out of bounds");
      assert(reachabilityMatrix[start][current] != kMaxDistance &&
             "Unreachable node in queue");

      const int64_t nextDistance = reachabilityMatrix[start][current] + 1;
      assert(nextDistance > 0 && "Distance overflow");

      for (int next : edge[current]) {
        assert(next >= 0 && next < ssize_t(numOps) &&
               "Next index out of bounds");

        if (reachabilityMatrix[start][next] == kMaxDistance ||
            reachabilityMatrix[start][next] > nextDistance) {
          reachabilityMatrix[start][next] = nextDistance;
          posQueue.push(next);
        }
      }
    }
  }
}

bool ReachabilityAnalyzer::isReachable(Operation *start,
                                       Operation *dest) const {
  auto startIt = opToIndexMap.find(start);
  auto destIt = opToIndexMap.find(dest);
  if (startIt == opToIndexMap.end() || destIt == opToIndexMap.end())
    return false;
  int startIdx = startIt->second;
  int destIdx = destIt->second;
  LLVM_DEBUG(llvm::dbgs() << "Checking reachability " << *start << " " << *dest
                          << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "here ok " << startIdx << " " << destIdx << "\n";);
  return reachabilityMatrix[startIdx][destIdx] != kMaxDistance;
}

bool ReachabilityAnalyzer::isReverseReachable(Operation *start,
                                              Operation *dest) const {
  return isReachable(dest, start);
}

bool ReachabilityAnalyzer::hasDataDependency(Operation *op1,
                                             Operation *op2) const {
  return isReachable(op1, op2) || isReachable(op2, op1);
}

int64_t ReachabilityAnalyzer::getReachabilityDistance(Operation *start,
                                                      Operation *dest) const {
  auto startIt = opToIndexMap.find(start);
  auto destIt = opToIndexMap.find(dest);
  if (startIt == opToIndexMap.end() || destIt == opToIndexMap.end())
    return kMaxDistance;
  int startIdx = startIt->second;
  int destIdx = destIt->second;
  return reachabilityMatrix[startIdx][destIdx];
}

SmallVector<int> ReachabilityAnalyzer::getLCA(Operation *start,
                                              Operation *dest) const {
  auto startIt = opToIndexMap.find(start);
  auto destIt = opToIndexMap.find(dest);
  SmallVector<int> commonAncestors;
  if (startIt == opToIndexMap.end() || destIt == opToIndexMap.end())
    return commonAncestors;
  int startIdx = startIt->second;
  int destIdx = destIt->second;

  // First, find all common ancestors
  for (size_t i = 0; i < reachabilityMatrix.size(); ++i) {
    if (reachabilityMatrix[i][startIdx] != kMaxDistance &&
        reachabilityMatrix[i][destIdx] != kMaxDistance) {
      commonAncestors.push_back(i);
    }
  }

  // Now, remove ancestors that have descendants which are also common ancestors
  SmallVector<int> lowestCommonAncestors;
  for (int ancestor : commonAncestors) {
    bool isLowest = true;
    for (int descendant : edge[ancestor]) {
      if (std::binary_search(commonAncestors.begin(), commonAncestors.end(),
                             descendant)) {
        isLowest = false;
        break;
      }
    }
    if (isLowest) {
      lowestCommonAncestors.push_back(ancestor);
    }
  }

  return lowestCommonAncestors;
}

int64_t
ReachabilityAnalyzer::getShortestPathFromAncestor(Operation *start,
                                                  Operation *dest) const {
  auto startIt = opToIndexMap.find(start);
  auto destIt = opToIndexMap.find(dest);
  if (startIt == opToIndexMap.end() || destIt == opToIndexMap.end())
    return kMaxDistance;
  int startIdx = startIt->second;
  int destIdx = destIt->second;

  SmallVector<int> lca = getLCA(start, dest);
  if (lca.empty())
    return kMaxDistance;

  // Find the ancestor with the shortest total path
  int64_t shortestDistance = kMaxDistance;
  for (int ancestorIdx : lca) {
    int64_t totalDistance = reachabilityMatrix[ancestorIdx][startIdx] +
                            reachabilityMatrix[ancestorIdx][destIdx];
    if (totalDistance < shortestDistance) {
      shortestDistance = totalDistance;
    }
  }

  return shortestDistance;
}

int64_t ReachabilityAnalyzer::getIndex(Operation *op) {
  auto it = opToIndexMap.find(op);
  if (it == opToIndexMap.end())
    return -1;
  return it->second;
}

Operation *ReachabilityAnalyzer::getOperation(int64_t index) {
  if (index < 0 || index >= static_cast<int64_t>(operationList.size()))
    return nullptr;
  return operationList[index];
}

} // namespace utils
} // namespace mlir

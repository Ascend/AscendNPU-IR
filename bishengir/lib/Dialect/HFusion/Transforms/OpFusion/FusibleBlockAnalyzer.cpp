//===- FusibleBlockAnalyzer.cpp - Generate fusible blocks by rules --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlockAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/Support/Debug.h"

#include <queue>

#define DEBUG_TYPE "hfusion-fuse-analyzer"

namespace mlir {
namespace hfusion {

using namespace opfusion;

FusibleBlocks getFusibleBlocks(func::FuncOp func,
                               FusibleHelper &fusibleHelper) {
  FusibleBlocks fusibleBlocks;
  for (Block &block : func.getBody()) {
    FusibleBlockAnalyzer analyzer(block, fusibleHelper);
    // This uses append, because it can
    // return multiple fusible blocks from 1 block
    fusibleBlocks.append(analyzer.getFusibleBlocks());
  }
  return fusibleBlocks;
}

namespace opfusion {

FusibleBlockAnalyzer::FusibleBlockAnalyzer(Block &block,
                                           const FusibleHelper &fusibleHelper)
    : fusibleHelper_(&fusibleHelper) {
  for (Operation &op : block.getOperations())
    ops_.insert(&op);

  // Initialize OpIdx
  for (size_t idx = 0; idx < ops_.size(); ++idx) {
    opIdx_[ops_[idx]] = idx;
  }

  disjointSet_.resize(ops_.size(), -1);
  setType_.resize(ops_.size());
  opMaxRank_.resize(ops_.size());
  opReduceDim_.resize(ops_.size());
  importantSize_.resize(ops_.size());
  shapePivot_.resize(ops_.size());
  topoRank_.resize(ops_.size());

  edge_.resize(ops_.size());
  revEdge_.resize(ops_.size());

  // Construct edge for blocks
  for (size_t idx = 0; idx < ops_.size(); ++idx) {
    Operation *op = ops_[idx];
    opMaxRank_[idx] = fusibleHelper_->obtainLastReduceRank(op);
    opReduceDim_[idx] = fusibleHelper_->obtainReduceDim(op);
    setType_[idx] = fusibleHelper_->obtainType(op);
    shapePivot_[idx] =
        fusibleHelper_->isShapePivot(op) ? static_cast<int>(idx) : -1;

    bool isOpImportant = fusibleHelper_->isImportantPattern(op);
    importantSize_[idx] = isOpImportant;
    if (isOpImportant)
      importantOpsNum_++;
  }
}

bool FusibleBlockAnalyzer::verifyRulesAndJoin(int nodeU, int nodeV,
                                              bool horizontal) {
  // Try to join
  int parentU = find(nodeU);
  int parentV = find(nodeV);
  // Already in one set, skip fusing
  if (parentU == parentV)
    return false;

  LLVM_DEBUG(llvm::dbgs() << "Joining and checking by reduce dimensions\n";);
  // Reduce dimension checker
  if (fusibleHelper_->isRestrictedByReduceRank(opMaxRank_[parentU],
                                               opMaxRank_[parentV])) {
    LLVM_DEBUG(llvm::dbgs() << "Not allowed by reduce ranks\n";);
    return false;
  }

  if (fusibleHelper_->isRestrictedByReduceDim(opReduceDim_[parentU],
                                              opReduceDim_[parentV])) {
    LLVM_DEBUG(llvm::dbgs() << "Not allowed by reduce dimensions\n";);
    return false;
  }

  // Node type checker
  if (fusibleHelper_->isRestrictedByNodeType(setType_[parentU],
                                             setType_[parentV], horizontal)) {
    LLVM_DEBUG(llvm::dbgs() << "Not allowed by node Type\n";);
    return false;
  }

  if (horizontal) {
    if (isRestrictedByShapePivot(parentU, parentV)) {
      LLVM_DEBUG(llvm::dbgs() << "Not allowed by shape pivot\n";);
      return false;
    }
  }

  // Restricted by dependency
  if (isRestrictedByDependency(parentU, parentV, horizontal)) {
    LLVM_DEBUG(llvm::dbgs() << "Not allowed by fusion graph dependency\n";);
    return false;
  }

  if (horizontal) {
    // Check the opposite as well
    if (isRestrictedByDependency(parentV, parentU, horizontal)) {
      LLVM_DEBUG(llvm::dbgs()
                     << "Not allowed by opposite fusion graph dependency\n";);
      return false;
    }
    if (importantSize_[parentU] == 0 || importantSize_[parentV] == 0) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "Skipping horizontal function without vv operations\n";);
      return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Joining\n";);
  join(parentU, parentV, horizontal);
  return true;
}

bool FusibleBlockAnalyzer::checkGroupRequirements(
    const SetVector<Operation *> &group) {
  // Count vv ops here
  int importantCount = 0;
  int matmulCount = 0;
  for (Operation *op : group) {
    if (FusibleHelper::isImportantPattern(op))
      importantCount++;
    if (FusibleHelper::getOpPattern(op) == OpPattern::kMatmul) {
      matmulCount++;
    }
  }
  // If its shallow cv, check if there is a matmul
  if (fusibleHelper_->getFusionKind() == FusionKind::ShallowCV ||
      fusibleHelper_->getFusionKind() == FusionKind::MixCV) {
    if (matmulCount == 0)
      return false;
  }
  if (importantCount <= 1) {
    LLVM_DEBUG(llvm::dbgs()
                   << "There is only one operation in this single function\n";);
    return false;
  }
  return true;
}

using NodeNodePair = std::pair<int32_t, int32_t>;

SmallVector<SetVector<Operation *>> FusibleBlockAnalyzer::fuseBlock() {
  reInitEdges();
  reinitTopoRank();
  SmallVector<NodeNodePair> fusionCandidates;
  for (size_t nodeU = 0; nodeU < ops_.size(); ++nodeU) {
    // Fusing
    Operation *op = ops_[nodeU];
    LLVM_DEBUG(llvm::dbgs() << "\n\nComputing " << nodeU << ": " << *op
                            << "\n ------- ";);

    // Restricted the fusion if it's shallowCV and shape is dynamic
    if (fusibleHelper_->isRestrictedByDynamicShape(op))
      continue;

    for (Operation *user : op->getUsers()) {
      int32_t nodeV = static_cast<int32_t>(opIdx_[user]);
      fusionCandidates.emplace_back(nodeU, nodeV);
    }
  }

  // Sort all users based on its ascending topoRank
  llvm::sort(fusionCandidates.begin(), fusionCandidates.end(),
             [&](const NodeNodePair &a, const NodeNodePair &b) {
               if (topoRank_[a.second] != topoRank_[b.second])
                 return topoRank_[a.second] < topoRank_[b.second];
               return topoRank_[a.first] < topoRank_[b.first];
             });
  for (const NodeNodePair &candidate : fusionCandidates) {
    Operation *nodeU = ops_[candidate.first];
    Operation *nodeV = ops_[candidate.second];
    LLVM_DEBUG(llvm::dbgs() << "Ok getting " << candidate.first << " "
                            << candidate.second << "\n";);
    LLVM_DEBUG(llvm::dbgs()
                   << "parent toporank: " << topoRank_[find(candidate.first)]
                   << " " << topoRank_[find(candidate.second)] << "\n";);
    if (fusibleHelper_->isRestrictedByDynamicShape(nodeV))
      continue;

    if (fusibleHelper_->isFusible(nodeU, nodeV)) {
      // Verify graph and parent rules
      LLVM_DEBUG(llvm::dbgs()
                     << "Fusible nodes " << *nodeU << " " << *nodeV << "\n";);
      verifyRulesAndJoin(candidate.first, candidate.second);
    }
  }

  int32_t horizontalFusionCount = 0;
  LLVM_DEBUG(llvm::dbgs() << "Horizontal fusion merging\n";);
  // Try to merge horizontal fusion
  // Should we merge this in the return value instead ??
  SmallVector<Operation *> sortedOps;

  llvm::copy_if(ops_, std::back_inserter(sortedOps), [&](Operation *op) {
    const int32_t currentIdx = opIdx_[op];
    const int32_t parIdx = find(currentIdx);
    if (parIdx != currentIdx || importantSize_[parIdx] < 1)
      return false;
    if (getSize(parIdx) == 1) {
      // Check schedulable
      return fusibleHelper_->schedulable(op);
    }
    LLVM_DEBUG(llvm::dbgs() << "Ok copying " << *op << "\n";);
    return true;
  });
  llvm::sort(sortedOps.begin(), sortedOps.end(),
             [&](Operation *a, Operation *b) {
               return topoRank_[opIdx_[a]] < topoRank_[opIdx_[b]];
             });
  // Filter
  for (size_t nodeU = 0; nodeU < sortedOps.size(); ++nodeU) {
    if (horizontalFusionCount >= fusibleHelper_->maxHorizontalFusion())
      break;
    // Fusing
    Operation *opU = sortedOps[nodeU];
    // Restricted the fusion if it's shallowCV and shape is dynamic
    if (fusibleHelper_->isRestrictedByDynamicShape(opU, true))
      continue;
    for (size_t nodeV = nodeU + 1; nodeV < sortedOps.size(); ++nodeV) {
      if (horizontalFusionCount >= fusibleHelper_->maxHorizontalFusion())
        break;
      Operation *opV = sortedOps[nodeV];
      if (fusibleHelper_->isRestrictedByDynamicShape(opV, true))
        continue;
      // Verify graph and parent rules
      LLVM_DEBUG(llvm::dbgs()
                     << "Fusible nodes " << *opU << " " << *opV << "\n";);
      horizontalFusionCount +=
          verifyRulesAndJoin(opIdx_[opU], opIdx_[opV], true);
    }
  }

  if (horizontalFusionCount >= fusibleHelper_->maxHorizontalFusion()) {
    LLVM_DEBUG(llvm::dbgs() << "Maximum horizontal fusion has reached\n";);
  }

  SmallVector<SetVector<Operation *>> groups(ops_.size());
  for (Operation *op : ops_) {
    groups[find(opIdx_.at(op))].insert(op);
  }

  SmallVector<SetVector<Operation *>> fusedGroups;
  for (const auto &group : groups)
    if (group.size() > 1) {
      if (checkGroupRequirements(group))
        fusedGroups.push_back(group);
    }

  return fusedGroups;
}

FusibleBlocks FusibleBlockAnalyzer::getFusibleBlocks() {
  FusibleBlocks res;
  for (const auto &ops : fuseBlock())
    res.emplace_back(ops.getArrayRef(), fusibleHelper_);
  return res;
}

int FusibleBlockAnalyzer::find(size_t x) {
  return disjointSet_[x] < 0 ? x : disjointSet_[x] = find(disjointSet_[x]);
}

int FusibleBlockAnalyzer::getSize(size_t x) { return -disjointSet_[find(x)]; }

bool FusibleBlockAnalyzer::reInitEdges() {
  for (size_t idx = 0; idx < ops_.size(); ++idx) {
    edge_[idx].clear();
    revEdge_[idx].clear();
  }
  for (size_t i = 0; i < ops_.size(); ++i) {
    int idx = find(i);
    Operation *op = ops_[i];
    for (Operation *userOp : op->getUsers()) {
      if (!opIdx_.contains(userOp)) {
        // Check if there is inter block control flow
        continue;
      }
      const int nextIdx = find(opIdx_[userOp]);
      if (idx == nextIdx)
        continue;
      // If not exist, then default fusible
      bool existing = edge_[idx].count(nextIdx) == 0 || edge_[idx][nextIdx];
      edge_[idx].insert(
          {nextIdx, existing && fusibleHelper_->isFusible(op, userOp)});
      revEdge_[nextIdx].insert(idx);
    }
  }
  return true;
}

void FusibleBlockAnalyzer::mergeEdge(int nodeA, int nodeB) {
  // Remove self loop
  edge_[nodeA].erase(nodeB);
  edge_[nodeB].erase(nodeA);
  // Merge all outdegree of nodeB into nodeA
  for (const auto &[targetNode, isFusible] : edge_[nodeB]) {
    // Make sure reverse edge of the target is available
    assert(revEdge_[targetNode].contains(nodeB));
    // Remove the indegree of the target and update with nodeA
    revEdge_[targetNode].erase(nodeB);
    revEdge_[targetNode].insert(nodeA);
    // If its restricted, then replace
    if (!isFusible) {
      edge_[nodeA][targetNode] = false;
    } else {
      // Else try to put in the value, would return false if non are made
      edge_[nodeA].try_emplace(targetNode, /* isFusible */ true);
    }
  }
  // Clear for B
  edge_[nodeB].clear();

  // Adjust inDegree of B and A, all indegree of B now points to A
  revEdge_[nodeA].erase(nodeB);
  revEdge_[nodeB].erase(nodeA);
  for (auto prevValue : revEdge_[nodeB]) {
    auto findMergeB = edge_[prevValue].find(nodeB);
    // Make sure that edge prevValue -> nodeB exists
    assert(findMergeB != edge_[prevValue].end());
    const bool isFusible = findMergeB->second;
    if (!isFusible) {
      // If it was restricted, then make a restriction
      // from prevValue -> nodeA
      edge_[prevValue][nodeA] = false;
    } else {
      // Else, just try emplace if not exist
      edge_[prevValue].try_emplace(nodeA, /* isFusible */ true);
    }
    edge_[prevValue].erase(nodeB);
    // Make sure nodeA has back edge to prevvalue
    revEdge_[nodeA].insert(prevValue);
  }
  // Clear for nodeB
  revEdge_[nodeB].clear();
}

// disjointSet_[x] has 2 different states:
// If its < 0: then -disjointSet_[x] size of the set with head x.
// if its >= 0: then disjointSet_[x] is the parent of the set
//
// For example, the union find data structures:
// Set1: {5, 7 (head), 8, 3};
// Set2: {4 (head), 2, 0};
// Set3: {9, 10 (head)};
// Set4: {1 (head)};
// disjointSet_[9] = 10; (in set3, parent of 9 is 10)
// disjointSet_[1] = -1; (Set4 has size 1)
// disjointSet_[7] = -4; (Set1 has size 4)
// disjointSet_[10] = -2; (Set3 has size 2)
// disjointSet_[0] = 4; (Set1 has size 4)
void FusibleBlockAnalyzer::join(int nodeA, int nodeB, bool isHorizontal) {
  // Fetching parents of both nodes
  nodeA = find(nodeA);
  nodeB = find(nodeB);
  // If the same set, then skip merging
  if (nodeA == nodeB)
    return;

  // Take note before swapping
  int preA = nodeA;
  int preB = nodeB;

  // Merge small to large
  // This make Union Find Data Structure to have
  // amortized almost linear time complexity
  if (disjointSet_[nodeA] > disjointSet_[nodeB]) {
    std::swap(nodeA, nodeB);
  }

  mergeEdge(nodeA, nodeB);

  // Both value is the size, move all elements in B to A
  // Size of A gets added to size of B
  disjointSet_[nodeA] += disjointSet_[nodeB];
  importantSize_[nodeA] += importantSize_[nodeB];

  // Propagate shape pivot if its a shape pivot
  if (shapePivot_[nodeB] != -1)
    shapePivot_[nodeA] = shapePivot_[nodeB];

  // Assign parent of B to A
  disjointSet_[nodeB] = nodeA;

  if (opMaxRank_[nodeA] < 0) {
    opMaxRank_[nodeA] = opMaxRank_[nodeB];
  }

  if (opReduceDim_[nodeA] < 0) {
    opReduceDim_[nodeA] = opReduceDim_[nodeB];
  }

  topoRank_[nodeA] = std::max(topoRank_[nodeA], topoRank_[nodeB]);

  setType_[nodeA] =
      fusibleHelper_->adjustType(setType_[preA], setType_[preB], isHorizontal);
}

bool FusibleBlockAnalyzer::isRestrictedByDependency(int startNode, int endNode,
                                                    bool horizontal) {
  // Reversed topological rank is restricted fusion
  if (!horizontal) {
    // Case like 1 -> 2 <- 1 <- 0 is possible
    LLVM_DEBUG(llvm::dbgs() << topoRank_[startNode] << " " << topoRank_[endNode]
                            << "\n";);
    assert(topoRank_[startNode] <= topoRank_[endNode]);
    auto findEd = edge_[startNode].find(endNode);
    if (findEd == edge_[startNode].end() || !findEd->second) {
      // If no fusible direct edge is found, then it must be restricted
      return true;
    }
    if (edge_[startNode].size() == 1) {
      // Simple direct check, there is no cycle,
      // so it means indegree must equal to 1
      return false;
    }
  }

  SmallVector<int> inDegree(ops_.size());
  BitVector isVisited(ops_.size(), false);

  // Find induced subgraph starting from startNode
  std::queue<int> visited;
  isVisited[startNode] = true;
  visited.push(startNode);

  // Two nodes are mergeable if the target
  // doesn't have any other dependency from the parent
  while (!visited.empty()) {
    int pos = visited.front();
    visited.pop();
    for (const auto &[targetNode, _] : edge_[pos]) {
      inDegree[targetNode]++;
      if (isVisited.test(targetNode))
        continue;

      isVisited.set(targetNode);
      if (horizontal) {
        visited.push(targetNode);
      } else {
        if (topoRank_[targetNode] <= topoRank_[endNode]) {
          visited.push(targetNode);
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Got indegree " << inDegree[endNode] << "\n";);
  if (!horizontal)
    return inDegree[endNode] > 1;
  return inDegree[endNode] > 0;
}

bool checkMatmulSameShape(Operation *a, Operation *b) {
  auto opA = dyn_cast<linalg::LinalgOp>(a);
  auto opB = dyn_cast<linalg::LinalgOp>(b);
  auto shapeA = utils::getShape(opA.getDpsInits()[0].getType());
  auto shapeB = utils::getShape(opB.getDpsInits()[0].getType());
  if (shapeA != shapeB)
    return false;
  auto shapeFrontA = utils::getShape(opA.getDpsInputs()[0].getType());
  auto shapeFrontB = utils::getShape(opB.getDpsInputs()[0].getType());
  if (isa<linalg::MatmulTransposeAOp>(opA))
    std::swap(shapeFrontA[0], shapeFrontA[1]);
  if (isa<linalg::MatmulTransposeAOp>(opB))
    std::swap(shapeFrontB[0], shapeFrontB[1]);
  return shapeFrontA == shapeFrontB;
}

bool checkOutputSameShape(Operation *a, Operation *b) {
  auto opA = dyn_cast<linalg::LinalgOp>(a);
  auto opB = dyn_cast<linalg::LinalgOp>(b);
  if (opA->getNumOperands() != opB->getNumOperands())
    return false;
  for (int i = 0; i < opA.getNumDpsInits(); i++) {
    if (utils::getShape(opA.getDpsInitOperand(i)->get().getType()) !=
        utils::getShape(opB.getDpsInitOperand(i)->get().getType())) {
      return false;
    }
  }
  return true;
}

bool FusibleBlockAnalyzer::isRestrictedByShapePivot(int nodeA, int nodeB) {
  if (shapePivot_[nodeA] == -1 || shapePivot_[nodeB] == -1)
    return false;
  auto *opA = ops_[shapePivot_[nodeA]];
  auto *opB = ops_[shapePivot_[nodeB]];
  switch (fusibleHelper_->getFusionKind()) {
  case FusionKind::ShallowCV:
  case FusionKind::ShallowVV:
    return false;
  case FusionKind::MixCV:
    return !checkMatmulSameShape(opA, opB);
  case FusionKind::PureElemwise:
  case FusionKind::AnyPB:
  case FusionKind::LastAxisPBR:
  case FusionKind::AnyPBR:
    return !checkOutputSameShape(opA, opB);
  case FusionKind::Unknown:
  default:
    llvm_unreachable("Fusion kind is not supported");
  }
}

void FusibleBlockAnalyzer::reinitTopoRank() {
  std::fill(topoRank_.begin(), topoRank_.end(), 0);
  SmallVector<int> inDegree(ops_.size(), 0);
  for (size_t i = 0; i < ops_.size(); i++) {
    for (const auto &[targetNode, _] : edge_[i]) {
      inDegree[targetNode]++;
    }
  }
  // Queue for nodes with no incoming edges
  std::queue<int> q;
  for (size_t i = 0; i < ops_.size(); i++) {
    if (inDegree[i] == 0) {
      q.push(i);
    }
  }

  int currentRank = 0;
  while (!q.empty()) {
    size_t size = q.size();
    while (size--) {
      int node = q.front();
      q.pop();
      topoRank_[node] = currentRank;
      LLVM_DEBUG(llvm::dbgs() << "Listing rank: " << node << " " << *ops_[node]
                              << " " << currentRank << "\n";);
      for (const auto &[targetNode, _] : edge_[node]) {
        inDegree[targetNode]--;
        if (inDegree[targetNode] == 0) {
          q.push(targetNode);
        }
      }
    }
    currentRank++;
  }
}

bool FusibleBlockAnalyzer::isFusible() {
  auto fusibleBlocks = getFusibleBlocks();
  if (fusibleBlocks.size() != 1)
    return false;

  auto fusibleBlock = fusibleBlocks.front();
  if (fusibleHelper_->getFusionKind() == FusionKind::MixCV) {
    if (fusibleBlock.getOutputs().size() != 1)
      return false;
  }
  // Check to see if the fusible block contains all the important ops.
  auto opWithAuxs = fusibleBlock.getOpWithAuxs();
  int fusibleCount = 0;
  for (Operation *op : opWithAuxs) {
    if (FusibleHelper::isImportantPattern(op))
      fusibleCount++;
  }
  return fusibleCount == importantOpsNum_;
}

} // namespace opfusion
} // namespace hfusion
} // namespace mlir

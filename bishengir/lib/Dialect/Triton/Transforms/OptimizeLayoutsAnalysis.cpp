//===--------------------OptimizeLayoutsAnalysis.cpp ----------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt  for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OptimizeLayoutsAnalysis.h"

#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>

using namespace mlir;

#define DEBUG_TYPE_NAME "optimize-layouts"
#define DBG(X)                                                                 \
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE_NAME "-analysis] " << X << "\n")

namespace mlir {
namespace triton {

const char *kLayoutActionAttr = "layout.action";
const char *kLayoutPriorityAttr = "layout.priority";
const char *kLayoutCostAttr = "layout.cost_estimate";

struct OptimizeLayoutsAnalysis::ConvertInfo {
  Operation *op = nullptr;
  Value src;
  Value dst;
  RankedTensorType srcTy;
  RankedTensorType dstTy;
  unsigned useCount = 0;
  bool inLoop = false;
  float srcAlignmentScore = 0.0f;
  float dstAlignmentScore = 0.0f;
  float conversionCost = 0.0f;
  int priority = 0;
  // additional metrics for better decisions
  bool feedsLoadStore = false;
  bool feedsReduce = false;
  bool feedsReshape = false;
  bool isF32Type = false;
  int numUsers = 0; // how deep in the block structure (later = higher)

  // Central cost: positive means prefer to keep, negative means prefer to
  // remove via propagate-down
  float centralCost = 0.0f;

  LayoutAction action = LayoutAction::None;
  float score = 0.0f;
};

struct OptimizeLayoutsAnalysis::Impl {
  explicit Impl(Operation *root) : rootOp(root) {}

  void runAnalysis() {
    collectConvertOps();
    computeBaseMetrics();
    computeCentralCosts();
    initialDecisions();
    annotateOps();
  }

  void refineAfterPropDown() {
    collectConvertOps();
    computeBaseMetrics();
    // recompute central costs & score before making propagate-up decisions
    computeCentralCosts();
    decidePropagateUpForRemaining();
    annotateOps();
    DBG("Dump after analysis");
    LLVM_DEBUG(rootOp->print(llvm::dbgs()));
  }

  // conservative test whether encoding looks like a "slice" encoding.
  static bool encodingNotRegular(Attribute enc) {
    if (!enc)
      return true; // no encoding → treat as non-regular

    std::string s;
    llvm::raw_string_ostream os(s);
    enc.print(os);
    os.flush();

    // regular if it contains "blocked".
    // So NOT regular if it does NOT contain it.
    return s.find("ttg.blocked") == std::string::npos;
  }

  llvm::SmallVector<Operation *, 64> getPropagateDownOrder() const {
    return propagateDownOrder;
  }

  llvm::SmallVector<Operation *, 64> getPropagateUpOrder() const {
    return propagateUpOrder;
  }

private:
  Operation *rootOp;
  llvm::SmallVector<ConvertInfo, 64> converts;
  llvm::SmallVector<Operation *, 64> propagateDownOrder;
  llvm::SmallVector<Operation *, 64> propagateUpOrder;

  void collectConvertOps() {
    converts.clear();

    rootOp->walk([&](Operation *op) {
      StringRef opname = op->getName().getStringRef();
      if (opname.find("convert_layout") != StringRef::npos) {
        ConvertInfo ci;
        ci.op = op;
        if (op->getNumOperands() > 0)
          ci.src = op->getOperand(0);
        if (op->getNumResults() > 0)
          ci.dst = op->getResult(0);
        if (auto t = mlir::dyn_cast<RankedTensorType>(ci.src.getType()))
          ci.srcTy = t;
        if (auto t = mlir::dyn_cast<RankedTensorType>(ci.dst.getType()))
          ci.dstTy = t;
        // store position for block depth calculation
        converts.push_back(ci);
      }
    });

    auto computeUserChainSize = [&](Value start, RankedTensorType dstTy) {
      if (!start || !dstTy)
        return 0;

      Attribute dstEnc = dstTy.getEncoding();
      // visited set for operations we've counted
      llvm::SmallPtrSet<Operation *, 16> visitedOps;

      // queue of values to explore
      llvm::SmallVector<Value, 16> queue;
      queue.push_back(start);

      while (!queue.empty()) {
        Value v = queue.pop_back_val();

        // iterate uses of this value
        for (Operation *user : v.getUsers()) {
          // check whether this user actually consumes an operand that has
          // the same encoding as dstEnc. (If multiple operands match, still
          // count once.)
          bool userUsesDstLayout = false;
          for (Value opnd : user->getOperands()) {
            if (opnd != v)
              continue;

            if (auto opTy = mlir::dyn_cast<RankedTensorType>(opnd.getType())) {
              if (opTy.getEncoding() == dstEnc) {
                userUsesDstLayout = true;
                break;
              }
            }
          }

          if (!userUsesDstLayout)
            continue; // do not count or follow this user

          // Count this user once
          if (!visitedOps.insert(user).second)
            continue; // already counted

          // enqueue the user's results so we reach indirect users that consume
          // dst-encoded results
          for (Value res : user->getResults()) {
            if (res && mlir::isa<RankedTensorType>(res.getType())) {
              queue.push_back(res);
            }
          }
        }
      }

      return static_cast<int>(visitedOps.size());
    };

    // calculate user-chain size
    for (auto &ci : converts) {
      if (ci.op && ci.dstTy) {
        ci.numUsers = computeUserChainSize(ci.dst, ci.dstTy);
      } else {
        ci.numUsers = 0;
      }
    }

    DBG("Collected convert_layout ops: " << converts.size());
  }

  void computeBaseMetrics() {
    for (auto &ci : converts) {
      ci.useCount = countUses(ci.dst);
      ci.inLoop = isInLoop(ci.op);
      ci.conversionCost = estimateConversionCost(ci);

      std::tie(ci.srcAlignmentScore, ci.dstAlignmentScore) =
          computeAlignmentScores(ci);

      ci.feedsLoadStore = checkFeedsLoadStore(ci);
      ci.feedsReduce = checkFeedsReduce(ci);
      ci.feedsReshape = checkFeedsReshape(ci);
      ci.isF32Type = ci.srcTy && ci.srcTy.getElementType().isF32();

      ci.priority =
          static_cast<int>(computePriority(ci) * (1 + ci.dstAlignmentScore));

      DBG("Metrics for op: uses="
          << ci.useCount << " inLoop=" << ci.inLoop
          << " numUsers=" << ci.numUsers << " srcAlign=" << ci.srcAlignmentScore
          << " dstAlign=" << ci.dstAlignmentScore
          << " feedsReduce=" << ci.feedsReduce << " prio=" << ci.priority);
    }
  }

  unsigned countUses(Value v) {
    if (!v)
      return 0;
    return std::distance(v.use_begin(), v.use_end());
  }

  bool isInLoop(Operation *op) {
    if (!op)
      return false;

    // Fast path: if the operation is nested inside a known loop op, return
    // true. Covers scf.for, scf.while, scf.parallel and affine.for.
    if (op->getParentOfType<scf::ForOp>() ||
        op->getParentOfType<scf::WhileOp>()) {
      return true;
    }
    return false;
  }

  bool checkFeedsLoadStore(const ConvertInfo &ci) {
    if (!ci.dst)
      return false;
    for (Operation *user : ci.dst.getUsers()) {
      if (isa<triton::StoreOp, triton::LoadOp>(user))
        return true;
    }
    return false;
  }

  bool checkFeedsReduce(const ConvertInfo &ci) {
    if (!ci.dst)
      return false;

    llvm::SmallVector<Value> worklist;
    llvm::SmallPtrSet<Value, 16> visited;

    worklist.push_back(ci.dst);

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();

      if (!visited.insert(v).second)
        continue;

      for (Operation *user : v.getUsers()) {
        // If there is a convert layout between we stop
        if (isa<gpu::ConvertLayoutOp>(user))
          return false;

        // Success condition
        if (isa<triton::ReduceOp>(user))
          return true;

        // Continue traversal
        for (Value res : user->getResults())
          worklist.push_back(res);
      }
    }

    return false;
  }

  bool checkFeedsReshape(const ConvertInfo &ci) {
    if (!ci.dst)
      return false;

    llvm::SmallVector<Value> worklist;
    llvm::SmallPtrSet<Value, 16> visited;

    worklist.push_back(ci.dst);

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();

      if (!visited.insert(v).second)
        continue;

      for (Operation *user : v.getUsers()) {
        // If there is a convert layout between we stop
        if (isa<gpu::ConvertLayoutOp>(user))
          return false;

        // Success condition
        if (isa<triton::ReshapeOp>(user))
          return true;

        // Continue traversal
        for (Value res : user->getResults())
          worklist.push_back(res);
      }
    }

    return false;
  }

  float estimateConversionCost(const ConvertInfo &ci) {
    if (!ci.srcTy)
      return 1.0f;
    int64_t elems = ci.srcTy.getNumElements();
    float base = std::log2(float(std::max<int64_t>(1, elems))) * 0.2f;
    return base + 1.0f;
  }

  // Compute alignment scores for the source and destination tensor types
  // in a ConvertInfo struct. Returns a pair: {srcAlignmentScore,
  // dstAlignmentScore}
  std::pair<float, float> computeAlignmentScores(const ConvertInfo &ci) {
    constexpr float kDefault = 0.5f;

    if (!ci.srcTy || !ci.dstTy)
      return {kDefault, kDefault};

    auto scoreForType = [&](RankedTensorType ty) -> float {
      if (!ty)
        return kDefault;

      Attribute enc = ty.getEncoding();
      if (!enc)
        return kDefault;

      // extract blocked encoding (layout mapping for threads/warps)
      BlockedEncodingAttr blocked = nullptr;
      if (auto sliceEnc = mlir::dyn_cast<SliceEncodingAttr>(enc)) {
        if (auto parent =
                mlir::dyn_cast<BlockedEncodingAttr>(sliceEnc.getParent()))
          blocked = parent;
      } else if (auto b = mlir::dyn_cast<BlockedEncodingAttr>(enc)) {
        blocked = b;
      }
      if (!blocked)
        return kDefault;

      ArrayRef<int64_t> shape = ty.getShape();
      if (shape.empty())
        return kDefault;

      // find the largest axis (dimension with the most elements)
      auto *largestPtr = llvm::max_element(shape);
      unsigned largestAxis = largestPtr - shape.data();
      int64_t largestSize = std::max<int64_t>(1, *largestPtr);

      auto elePerThreadsArray = llvm::to_vector(blocked.getSizePerThread());
      auto threadPerWarpsArray = llvm::to_vector(blocked.getThreadsPerWarp());
      auto orderArray = llvm::to_vector(blocked.getOrder());
      SmallVector<int64_t> elePerThreads(elePerThreadsArray.begin(),
                                         elePerThreadsArray.end());
      for (auto &v : elePerThreads)
        v = static_cast<int64_t>(v);

      SmallVector<int64_t> threadPerWarps(threadPerWarpsArray.begin(),
                                          threadPerWarpsArray.end());
      for (auto &v : threadPerWarps)
        v = static_cast<int64_t>(v);

      SmallVector<int64_t> order(orderArray.begin(), orderArray.end());
      for (auto &v : order)
        v = static_cast<int64_t>(v);

      auto getVal = [&](const SmallVector<int64_t> &arr,
                        unsigned idx) -> int64_t {
        return idx < arr.size() ? arr[idx] : 1;
      };

      int64_t axisSize = largestSize > 0 ? largestSize : 1;
      int64_t threadPerWarpsVal = getVal(threadPerWarps, largestAxis);
      int64_t elePerThreadsVal = getVal(elePerThreads, largestAxis);

      // Compute ratios to measure alignment efficiency (1 means very algin)
      double thread_ratio =
          std::min(1.0, double(threadPerWarpsVal) / double(axisSize));
      double size_ratio =
          std::min(1.0, double(elePerThreadsVal) / double(axisSize));

      // more weight on thread alignment
      double score = 0.6 * thread_ratio + 0.4 * size_ratio;

      // bonus if the largest axis is first in memory order (good for coalesced
      // access) ??
      if (!order.empty() && order[0] == (int64_t)largestAxis)
        score += 0.2;

      score = std::min(1.0, std::max(0.0, score));
      return static_cast<float>(score);
    };

    // compute scores for source and destination layouts
    float s = scoreForType(ci.srcTy);
    float d = scoreForType(ci.dstTy);

    return {s, d};
  }

  int computePriority(const ConvertInfo &ci) {
    int p = 0;

    if (ci.feedsLoadStore && ci.inLoop && ci.feedsReduce)
      p += 120;
    else if ((ci.feedsLoadStore && ci.inLoop) || ci.feedsReduce)
      p += 100;
    else if (ci.inLoop)
      p += 80;
    else if (ci.feedsLoadStore)
      p += 60;

    if (ci.isF32Type)
      p += 10;

    return p;
  }

  void computeCentralCosts() {
    // positive cost factors = prefer to KEEP the dst layout
    // negative cost factors = prefer to REMOVE via propagate-down

    constexpr float alignWeight = 2.5f;

    // negative: later ops are better to remove via propagate-down
    constexpr float numUsersWeight = -0.15f;

    for (auto &ci : converts) {
      float cost = 0.0f;

      // positive: if dst layout is better, we want to keep the convert
      float alignDelta = ci.dstAlignmentScore - ci.srcAlignmentScore;
      cost += alignDelta * alignWeight;

      // negative: later ops are better candidates for removal (propagate down)
      float numUserFactor = 1 / float(ci.numUsers);
      cost += numUserFactor * numUsersWeight;

      ci.centralCost = cost;

      DBG("Central cost for "
          << (ci.op ? ci.op->getName().getStringRef() : "<null>") << " = "
          << cost << " (positive=keep, negative=remove via propagate-down)");
    }
  }

  // phase 1: only decide between DontRemove and PropagateDown
  void initialDecisions() {
    // identify converts we should NOT remove under any circumstances
    for (auto &ci : converts) {
      ci.action = LayoutAction::None;

      if (!ci.srcTy || !ci.dstTy) {
        ci.action = LayoutAction::DontRemove;
        continue;
      }

      // rule 1: never remove converts that feed reduce operations (the upper
      // pass creates them)
      if (ci.feedsReduce || ci.feedsReshape) {
        ci.action = LayoutAction::DontRemove;
        DBG("Marking as DontRemove (feeds reduce): " << ci.op);
        continue;
      }

      // If not regular we can only propagate down this since when we try to
      // propagate up, some ops expect linear layout so we can not give a block
      // layout but block layout ops can use linear layout

      // TODO: later we have to decide if to propagate down or keep, but
      // currently all test shows that propagating down increase speed
      bool srcNonRegular = false;
      auto srcEnc = ci.srcTy.getEncoding();
      if (srcEnc)
        srcNonRegular = encodingNotRegular(srcEnc);

      if (srcNonRegular) {
        ci.action = LayoutAction::PropagateDown;
        DBG("Marking as PropagateDown (non-regular encoding): "
            << ci.op << " srcNonRegular=" << srcNonRegular);
        continue;
      }

      // rule 2: high priority converts with good destination alignment should
      // be kept
      if (ci.priority >= 70 && ci.dstAlignmentScore >= 0.7f) {
        ci.action = LayoutAction::DontRemove;
        DBG("Marking as DontRemove (high priority + good alignment): "
            << ci.op);
        continue;
      }
    }

    // identify converts suitable for propagate-down (remove by
    // propagating source down) when src and dst layout are similar
    for (auto &ci : converts) {
      if (ci.action != LayoutAction::None)
        continue;

      // good candidate for propagate-down if:
      // - source layout is almost as good as destination
      // - negative cost indicates it's beneficial to remove

      float alignmentRatio =
          ci.srcAlignmentScore / std::max(0.01f, ci.dstAlignmentScore);
      // src is at least 80% as good as dst
      bool goodSrcAlignment = alignmentRatio >= 0.8f;

      // only propagate down if cost is negative (prefer remove)
      bool beneficialToRemove = ci.centralCost < 0.0f;

      if (beneficialToRemove && goodSrcAlignment) {
        ci.action = LayoutAction::PropagateDown;
        DBG("Marking as PropagateDown: " << ci.op << " (cost=" << ci.centralCost
                                         << ", alignRatio=" << alignmentRatio
                                         << ")");
      } else {
        ci.action = LayoutAction::None;
      }
    }

    DBG("After phase 1: "
        << llvm::count_if(converts,
                          [](const ConvertInfo &ci) {
                            return ci.action == LayoutAction::DontRemove;
                          })
        << " DontRemove, "
        << llvm::count_if(converts,
                          [](const ConvertInfo &ci) {
                            return ci.action == LayoutAction::PropagateDown;
                          })
        << " PropagateDown, "
        << llvm::count_if(converts,
                          [](const ConvertInfo &ci) {
                            return ci.action == LayoutAction::None;
                          })
        << " undecided");
  }

  llvm::DenseMap<Operation *, ConvertInfo *> buildOpToInfoMap() {
    llvm::DenseMap<Operation *, ConvertInfo *> map;
    for (auto &ci : converts)
      if (ci.op)
        map[ci.op] = &ci;
    return map;
  }

  bool
  isBlockedByMajor(Operation *op,
                   const llvm::DenseSet<Operation *> &majorLayouts,
                   const llvm::DenseMap<Operation *, ConvertInfo *> &opToInfo,
                   int currentPrio, Operation *currentConvert) {
    if (!majorLayouts.count(op) || op == currentConvert)
      return false;

    auto it = opToInfo.find(op);
    if (it == opToInfo.end() || !it->second) {
      DBG("canPropagateUpSafely: hit major with NO ConvertInfo "
          "(currentPrio="
          << currentPrio << ") -> block\n");
      return true;
    }

    int otherPrio = it->second->priority;
    if (otherPrio >= currentPrio) {
      DBG("canPropagateUpSafely: hit major with prio "
          << otherPrio << " >= currentPrio " << currentPrio << " -> block\n");
      return true;
    }

    return false;
  }

  bool scanForLoopBodyForMajors(
      scf::ForOp forOp, const llvm::DenseSet<Operation *> &majorLayouts,
      const llvm::DenseMap<Operation *, ConvertInfo *> &opToInfo,
      int currentPrio, Operation *currentConvert) {
    for (Region &r : forOp->getRegions()) {
      for (Block &b : r) {
        for (Operation &opInBlock : b) {
          if (&opInBlock == currentConvert || !majorLayouts.count(&opInBlock))
            continue;

          auto it = opToInfo.find(&opInBlock);
          if (it == opToInfo.end() || !it->second) {
            LLVM_DEBUG(llvm::dbgs()
                       << "canPropagateUpSafely: major in scf.for body "
                       << "with NO ConvertInfo (currentPrio=" << currentPrio
                       << ") -> block\n");
            return true;
          }

          int otherPrio = it->second->priority;
          if (otherPrio >= currentPrio) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "canPropagateUpSafely: major in scf.for body with prio "
                << otherPrio << " >= currentPrio " << currentPrio
                << " -> block\n");
            return true;
          }
        }
      }
    }
    return false;
  }

  Value getForLoopInitOperand(scf::ForOp forOp, Value res) {
    if (auto opRes = mlir::dyn_cast<OpResult>(res)) {
      unsigned resNo = opRes.getResultNumber();
      unsigned numResults = forOp->getNumResults();
      if (resNo >= numResults)
        return Value();
      unsigned numOperands = forOp->getNumOperands();
      if (numOperands < numResults)
        return Value();

      unsigned iterStart = numOperands - numResults;
      return forOp->getOperand(iterStart + resNo);
    }
    return Value();
  }

  bool canPropagateUpSafely(
      Value startVal, const llvm::DenseSet<Operation *> &majorLayouts,
      const llvm::DenseMap<Operation *, ConvertInfo *> &opToInfo,
      Operation *currentConvert) {
    if (!startVal)
      return true;

    int currentPrio = 0;
    if (currentConvert) {
      auto it = opToInfo.find(currentConvert);
      if (it != opToInfo.end() && it->second)
        currentPrio = it->second->priority;
    }

    llvm::SmallVector<Value, 32> work;
    llvm::SmallPtrSet<Value, 32> visitedVals;
    work.push_back(startVal);

    const unsigned kMaxSteps = 200000;
    unsigned steps = 0;

    while (!work.empty() && steps < kMaxSteps) {
      steps++;
      Value cur = work.pop_back_val();

      if (!visitedVals.insert(cur).second) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "canPropagateUpSafely: detected cycle -> block propagation\n");
        return false;
      }

      if (auto ba = mlir::dyn_cast<BlockArgument>(cur))
        return true;

      Operation *def = cur.getDefiningOp();
      if (!def)
        return true;

      // if we hit a reduce op while walking *upwards*, block
      // propagation. Reduce ops change tensor ranks/layout semantics and we
      // cannot safely propagate a non-trivial layout through them.
      if (isa<triton::ReduceOp>(def)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "canPropagateUpSafely: hit ReduceOp -> block\n");
        return false;
      }

      // check major op on direct chain
      if (isBlockedByMajor(def, majorLayouts, opToInfo, currentPrio,
                           currentConvert))
        return false;

      // handle scf.for
      if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        if (scanForLoopBodyForMajors(forOp, majorLayouts, opToInfo, currentPrio,
                                     currentConvert))
          return false;

        if (Value initVal = getForLoopInitOperand(forOp, cur)) {
          if (!visitedVals.count(initVal))
            work.push_back(initVal);
          continue;
        } else {
          return true;
        }
      }

      // follow all tensor-like operands
      bool pushedAny = false;
      for (Value opd : def->getOperands()) {
        if (!mlir::isa<RankedTensorType>(opd.getType()))
          continue;
        if (visitedVals.count(opd))
          continue;

        work.push_back(opd);
        pushedAny = true;
      }

      if (!pushedAny)
        return true;
    }

    return true;
  }

  // pahse 2: for remaining converts, propagate upward when safe
  // major layouts can also be propagated up if they don't interfere with other
  // major layouts When we propagate up, the destination layout is preserved
  // implicitly
  void decidePropagateUpForRemaining() {
    auto opToInfo = buildOpToInfoMap();

    // if any convert op already has layout.action="propagate_up"
    // (these were created earlier during propagate-down), mark their
    // ConvertInfo so we don't overwrite them below.
    for (auto &ci : converts) {
      if (!ci.op)
        continue;
      if (auto actionAttr = ci.op->getAttrOfType<StringAttr>("layout.action")) {
        if (actionAttr.getValue() == "propagate_up") {
          ci.action = LayoutAction::PropagateUp;
          DBG("Preserving existing propagate_up on: " << ci.op);
        }
      }
    }

    // major layouts - these are high priority converts we want dst layout
    llvm::DenseSet<Operation *> majorLayouts;
    for (auto &ci : converts) {
      if (!ci.op)
        continue;
      if (ci.action == LayoutAction::DontRemove || ci.priority >= 80) {
        majorLayouts.insert(ci.op);
        DBG("Major layout: " << ci.op << " (priority=" << ci.priority << ")");
      }
    }

    // build processing order: top-down
    llvm::SmallVector<ConvertInfo *, 64> processingOrder;
    for (auto &ci : converts) {
      if (ci.op)
        processingOrder.push_back(&ci);
    }
    std::sort(processingOrder.begin(), processingOrder.end(),
              [&](ConvertInfo *a, ConvertInfo *b) {
                return mlir::utils::isBefore(a->op, b->op);
              });

    // Process converts in top-down order
    for (auto *ci : processingOrder) {
      if (!ci->op)
        continue;

      if (ci->action == LayoutAction::PropagateUp) {
        DBG("Already PropagateUp (preserved): " << ci->op);
        continue;
      }

      // check if we can safely propagate up without interfering with other
      // major layouts
      bool canPropagate =
          canPropagateUpSafely(ci->src, majorLayouts, opToInfo, ci->op);

      if (canPropagate && !ci->feedsReduce) {
        ci->action = LayoutAction::PropagateUp;
        DBG("Marking as PropagateUp: " << ci->op
                                       << " (priority=" << ci->priority << ")");
      } else {
        if (ci->action == LayoutAction::None) {
          ci->action = LayoutAction::DontRemove;
        }
        DBG("Marking as DontRemove (would interfere with other major layouts): "
            << ci->op);
      }
    }

    DBG("After phase 2: "
        << llvm::count_if(converts,
                          [](const ConvertInfo &ci) {
                            return ci.action == LayoutAction::DontRemove;
                          })
        << " DontRemove, "
        << llvm::count_if(converts,
                          [](const ConvertInfo &ci) {
                            return ci.action == LayoutAction::PropagateDown;
                          })
        << " PropagateDown, "
        << llvm::count_if(converts,
                          [](const ConvertInfo &ci) {
                            return ci.action == LayoutAction::PropagateUp;
                          })
        << " PropagateUp");
  }

  void annotateOps() {
    MLIRContext *ctx = rootOp->getContext();
    for (auto &ci : converts) {
      if (!ci.op)
        continue;
      ci.op->setAttr(kLayoutActionAttr,
                     StringAttr::get(ctx, actionToString(ci.action)));
      ci.op->setAttr(
          kLayoutPriorityAttr,
          IntegerAttr::get(IntegerType::get(ctx, 32), (int64_t)ci.priority));
      ci.op->setAttr(kLayoutCostAttr,
                     FloatAttr::get(FloatType::getF32(ctx), ci.centralCost));
    }
    DBG("Annotated " << converts.size() << " convert_layout ops");
  }
};

OptimizeLayoutsAnalysis::OptimizeLayoutsAnalysis(Operation *root)
    : impl(std::make_unique<Impl>(root)) {}

OptimizeLayoutsAnalysis::~OptimizeLayoutsAnalysis() = default;

void OptimizeLayoutsAnalysis::runAnalysis() { impl->runAnalysis(); }

void OptimizeLayoutsAnalysis::refineAfterPropDown() {
  impl->refineAfterPropDown();
}

llvm::SmallVector<Operation *, 64>
OptimizeLayoutsAnalysis::getPropagateDownOrder() const {
  return impl->getPropagateDownOrder();
}

llvm::SmallVector<Operation *, 64>
OptimizeLayoutsAnalysis::getPropagateUpOrder() const {
  return impl->getPropagateUpOrder();
}

} // namespace triton
} // namespace mlir

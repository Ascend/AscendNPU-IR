//===---------------LoopRestructureArangeOptimization.cpp------------------===//
//----------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt    for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <queue>

#define DEBUG_TYPE "loop-restructure-arange-optimization"

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_LOOPRESTRUCTUREARANGEOPTIMIZATION
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::arith;

/// Data structure to track store-load patterns
struct StoreLoadPattern {
  Operation *storeOp = nullptr;
  Operation *loadOp = nullptr;
  llvm::DenseSet<Operation *> dependentOps;
  int64_t embeddingSize = -1;
  SmallVector<Operation *, 4> maskOps;
  SmallVector<Operation *, 4> constantOps;
};

/// Try to extract a single int64_t from an Attribute that might be a
/// DenseIntElementsAttr (splat or single element) or an IntegerAttr.
static std::optional<int64_t> extractIntFromAttr(Attribute a) {
  if (auto i = mlir::dyn_cast<IntegerAttr>(a)) {
    return i.getValue().getSExtValue();
  }
  if (auto dense = mlir::dyn_cast<DenseElementsAttr>(a)) {
    if (mlir::isa<IntegerType>(dense.getType().getElementType())) {
      if (dense.isSplat()) {
        Attribute v = dense.getSplatValue<Attribute>();
        if (auto ia = mlir::dyn_cast<IntegerAttr>(v))
          return ia.getValue().getSExtValue();
      } else if (dense.getNumElements() == 1) {
        auto it = dense.value_begin<llvm::APInt>();
        return (*it).getSExtValue();
      }
    }
  }
  return std::nullopt;
}

/// Collect dependent ops from storeOp by walking backwards (operands defs).
/// This is used to find the block of ops that can be moved.
static void
findDependentOpsFromStore(triton::StoreOp storeOp,
                          llvm::DenseSet<Operation *> &dependentOps) {

  if (!storeOp)
    return;

  std::queue<Value> worklist;

  // include the store itself
  dependentOps.insert(storeOp.getOperation());

  //  include all store operands
  for (Value v : storeOp->getOperands()) {
    if (v)
      worklist.push(v);
  }

  // walk backwards including all ops of define ops
  while (!worklist.empty()) {
    Value v = worklist.front();
    worklist.pop();

    Operation *def = v.getDefiningOp();
    if (!def || !dependentOps.insert(def).second)
      continue;

    for (Value operand : def->getOperands()) {
      if (operand)
        worklist.push(operand);
    }
  }
}

/// Find mask and constant ops by starting from the store's mask operand(s)
/// and walking backwards (following operand defs). Collect mask chain ops and
/// collect any arith::ConstantOp found among operands in pattern.constantOps.
/// This is used to find the actual size of each load/store
static void findMaskAndConstantOps(Operation *storeOp,
                                   StoreLoadPattern &pattern) {
  if (!storeOp)
    return;

  // Helper to determine if a Value looks like a mask (tensor<...xi1>)
  SmallVector<Operation *, 8> starts;

  // Get mask from storeOp
  if (auto ttStoreOp = dyn_cast<triton::StoreOp>(storeOp)) {
    Value mask = ttStoreOp.getMask();
    if (mask) {
      if (Operation *def = mask.getDefiningOp())
        starts.push_back(def);
    }
  }

  // From start nodes walk mask-chain upwards; collect mask ops and constants
  const size_t MAX_VISIT = 1024;
  DenseSet<Operation *> visited;
  std::queue<Operation *> work;

  for (Operation *s : starts)
    if (s)
      work.push(s);

  while (!work.empty() && visited.size() < MAX_VISIT) {
    Operation *cur = work.front();
    work.pop();
    if (!cur || visited.count(cur))
      continue;

    visited.insert(cur);
    // add the visited op as part of mask chain
    pattern.maskOps.push_back(cur);

    // Check operands safely
    for (Value in : cur->getOperands()) {
      if (!in)
        continue;

      Operation *ddef = in.getDefiningOp();
      if (!ddef)
        continue; // skip values without defining op

      // Record constant operations
      if (auto constOp = dyn_cast<arith::ConstantOp>(ddef)) {
        pattern.constantOps.push_back(constOp);
      } else if (!visited.count(ddef)) {
        work.push(ddef);
      }
    }
  }
}

/// Attempt to extract a load size from constant ops collected.
/// Heuristic: look for integer dense constant splats and
/// power of Two
// TODO: currently this works but maybe in other test cases this method cant
// find the size of load
static bool extractEmbeddingSize(StoreLoadPattern &pattern) {
  int64_t bestSize = -1;
  for (Operation *cOp : pattern.constantOps) {
    if (!cOp)
      continue;
    auto cst = dyn_cast<arith::ConstantOp>(cOp);
    if (!cst)
      continue;
    Attribute val = cst.getValue();
    // only arith.constant dense<#> and extract this #
    auto dense = dyn_cast<DenseElementsAttr>(val);
    if (!dense)
      continue;
    if (auto maybeInt = extractIntFromAttr(val)) {
      int64_t v = *maybeInt;
      // want power of two
      if (v <= 0 || (static_cast<unsigned int>(v) & (static_cast<unsigned int>(v) - 1)) != 0)
        continue;
      auto tensorType = dyn_cast<RankedTensorType>(cst.getType());
      // get tensor last dim size
      if (!tensorType || tensorType.getShape().empty())
        continue;
      int64_t lastDim = tensorType.getShape().back();
      // only consider size if <= last dim size
      if (v > lastDim)
        continue;
      // chose the largest valid value
      if (v > bestSize)
        bestSize = v;
    }
  }
  if (bestSize > 0) {
    pattern.embeddingSize = bestSize;
    return true;
  }
  return false;
}

/// Preserve original program order: given a block and a set of ops, return
/// the ops in block order
static SmallVector<Operation *, 32>
orderedOpsInBlock(Block &blk, const llvm::DenseSet<Operation *> &set) {
  SmallVector<Operation *, 32> res;
  for (Operation &op : blk) {
    if (set.contains(&op))
      res.push_back(&op);
  }
  return res;
}

/// Replace only the last dimension if it equals oldLast; else
/// return original type. This is used to change ops return shapes from the old
/// size to the new shrink size from our grouping

// TODO: assumes the size of load is linked to the last dim, add other dims
// later
static Type replaceLastDimIfMatches(Type type, int64_t oldLast,
                                    int64_t newLast) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    ArrayRef<int64_t> shape = tensorType.getShape();
    if (!shape.empty()) {
      int64_t last = shape.back();
      if (last == oldLast) {
        SmallVector<int64_t> newShape(shape.begin(), shape.end());
        newShape.back() = newLast;
        return RankedTensorType::get(newShape, tensorType.getElementType());
      }
    }
  }
  return type;
}

/// Look through the group's dependent operations to find a broadcast op result
/// shape and return the last-dimension size if available. If multiple different
/// last-dims are discovered, prefer the largest. If none can be found,
/// return std::nullopt. This is to find the old size that fits all the old
/// loads/stores

// TODO: this is a simply way to find the old size given the test cases I have
// right now, but this might not work for all cases, need to find better algo
// later
static std::optional<int64_t> findLastDimFromBroadcasts(
    const SmallVectorImpl<const StoreLoadPattern *> &group) {
  int64_t best = -1;
  for (auto *pattern : group) {
    if (!pattern)
      continue;
    for (Operation *dep : pattern->dependentOps) {
      if (!dep)
        continue;
      // only consider broadcast like ops
      if (!isa<triton::BroadcastOp, triton::ExpandDimsOp>(dep))
        continue;
      // check result tensor shapes.
      for (Value res : dep->getResults()) {
        auto rt = dyn_cast<RankedTensorType>(res.getType());
        if (!rt)
          continue;
        ArrayRef<int64_t> shape = rt.getShape();
        if (shape.empty())
          continue;
        int64_t last = shape.back();
        // keep the largest valid one
        if (last > 0 && last > best)
          best = last;
      }
    }
  }
  if (best > 0)
    return best;
  return std::nullopt;
}

/// Ensure a Value with a ranked tensor whose last dim == oldSize is turned
/// into a Value with last-dim == newSize. If the value is already mapped in
/// valueMapping, return that mapped value. Otherwise, insert a tt.broadcast
/// that produces the adjusted type

// TODO: assumes size affect only the last dim of ops
static Value ensureValueWithNewLastDim(
    Value v, int64_t oldSize, int64_t newSize, PatternRewriter &rewriter,
    Location loc, llvm::DenseMap<Value, Value> &valueMapping,
    llvm::DenseMap<Operation *, Operation *> &clonedOpsMap, int groupID) {
  if (!v)
    return v;

  Type t = v.getType();
  auto rt = mlir::dyn_cast<RankedTensorType>(t);
  if (!rt)
    return v;

  ArrayRef<int64_t> shape = rt.getShape();
  if (shape.empty())
    return v;

  int64_t last = shape.back();
  if (last != oldSize)
    return v;

  // If this value is already produced by a cloned producer, return the clone
  // since we also check the inputs.
  if (Operation *def = v.getDefiningOp()) {
    if (clonedOpsMap.count(def)) {
      Operation *clonedProducer = clonedOpsMap[def];
      unsigned idx = mlir::cast<OpResult>(v).getResultNumber();
      Value mapped = clonedProducer->getResult(idx);
      valueMapping[v] = mapped;
      return mapped;
    }
  }
  Type elem = rt.getElementType();

  // New tensor type with last dim replaced
  SmallVector<int64_t> newShape(shape.begin(), shape.end());
  newShape.back() = newSize;
  RankedTensorType newType = RankedTensorType::get(newShape, elem);

  // create a triton.broadcast that produces the requested shape.
  rewriter.setInsertionPointAfterValue(v);
  auto bcast = rewriter.create<triton::BroadcastOp>(loc, newType, v);
  bcast.getOperation()->setAttr(
      "group_id", IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                                   APInt(32, groupID, /*isSigned=*/true)));
  Value newV = bcast.getResult();
  valueMapping[v] = newV;
  return newV;
}

/// Clone operations for a group using OperationState approach
static Operation *
cloneConstantOp(PatternRewriter &rewriter, Location loc,
                arith::ConstantOp constOp, int64_t oldSize, int64_t newSize,
                int groupID,
                llvm::DenseMap<Operation *, Operation *> &clonedOpsMap) {
  OperationState state(loc, constOp->getName());
  // Copy all attributes except "value"
  for (auto attr : constOp->getAttrs()) {
    if (attr.getName() == "value")
      continue;
    state.addAttribute(attr.getName(), attr.getValue());
  }
  // update to new result types
  SmallVector<Type, 1> newResultTypes;
  for (Type t : constOp->getResultTypes()) {
    newResultTypes.push_back(replaceLastDimIfMatches(t, oldSize, newSize));
  }
  state.addTypes(newResultTypes);
  // Rebuild/adjust the "value" attribute so its tensor type matches
  // newResultTypes[0]
  Attribute valAttr = constOp.getValue();
  Attribute newValAttr = valAttr;
  if (!newResultTypes.empty()) {
    if (auto newResRT = dyn_cast<RankedTensorType>(newResultTypes[0])) {
      if (auto dense = dyn_cast_or_null<DenseElementsAttr>(valAttr)) {
        if (dense.isSplat()) {
          Attribute splatVal = dense.getSplatValue<Attribute>();
          newValAttr = SplatElementsAttr::get(newResRT, splatVal);
        } else if (dense.getNumElements() == 1) {
          Type elemTy = newResRT.getElementType();
          if (mlir::isa<IntegerType>(elemTy)) {
            auto it = dense.value_begin<llvm::APInt>();
            IntegerAttr scalar = IntegerAttr::get(elemTy, *it);
            newValAttr = SplatElementsAttr::get(newResRT, scalar);
          } else if (mlir::isa<FloatType>(elemTy)) {
            auto it = dense.value_begin<llvm::APFloat>();
            FloatAttr scalar = FloatAttr::get(elemTy, *it);
            newValAttr = SplatElementsAttr::get(newResRT, scalar);
          }
        } else {
          //  create a splat from the first element
          Type elemTy = newResRT.getElementType();
          if (mlir::isa<IntegerType>(elemTy)) {
            auto it = dense.value_begin<llvm::APInt>();
            IntegerAttr scalar = IntegerAttr::get(elemTy, *it);
            newValAttr = SplatElementsAttr::get(newResRT, scalar);
          } else if (mlir::isa<FloatType>(elemTy)) {
            auto it = dense.value_begin<llvm::APFloat>();
            FloatAttr scalar = FloatAttr::get(elemTy, *it);
            newValAttr = SplatElementsAttr::get(newResRT, scalar);
          }
        }
      } else if (mlir::isa<IntegerAttr>(valAttr) || mlir::isa<FloatAttr>(valAttr)) {
        newValAttr = SplatElementsAttr::get(newResRT, valAttr);
      }
    }
  }
  if (newValAttr)
    state.addAttribute("value", newValAttr);
  Operation *cloned = rewriter.create(state);
  cloned->setAttr("group_id", rewriter.getI32IntegerAttr(groupID));
  clonedOpsMap[constOp] = cloned;
  return cloned;
}

static void cloneGroupOperations(
    PatternRewriter &rewriter, Location loc, Block &sourceBlock,
    Operation *insertionPoint,
    const SmallVectorImpl<const StoreLoadPattern *> &group, int64_t oldSize,
    int64_t newSize, llvm::DenseMap<Operation *, Operation *> &clonedOpsMap,
    int groupID, llvm::DenseMap<Value, Value> *initialValueMapping = nullptr) {
  // Collect all ops to clone
  llvm::DenseSet<Operation *> opsToClone;
  for (auto *p : group) {
    if (!p)
      continue;
    opsToClone.insert(p->dependentOps.begin(), p->dependentOps.end());
    if (p->storeOp)
      opsToClone.insert(p->storeOp);
  }
  // Preserve program order
  SmallVector<Operation *, 64> ordered =
      orderedOpsInBlock(sourceBlock, opsToClone);
  // Value mapping old -> new
  llvm::DenseMap<Value, Value> valueMapping;
  // If initial value mapping provided, use it as starting point
  if (initialValueMapping) {
    valueMapping = *initialValueMapping;
  }

  // Insert cloned ops before insertionPoint
  rewriter.setInsertionPoint(insertionPoint);
  for (Operation *op : ordered) {
    if (!op)
      continue;
    // Special case: triton.make_range
    if (auto mk = dyn_cast<triton::MakeRangeOp>(op)) {
      int64_t start = mk.getStart();
      int64_t end = mk.getEnd();
      // Only adjust when the original end matches oldSize (we want newSize)
      if (end == oldSize) {
        OperationState state(loc, op->getName());
        // Copy all attributes except "start" and "end" (we'll set them below)
        for (auto attr : op->getAttrs()) {
          if (attr.getName() == "start" || attr.getName() == "end")
            continue;
          state.addAttribute(attr.getName(), attr.getValue());
        }
        // Recreate start and end with updated values (preserve start, update
        // end)
        state.addAttribute(
            "start",
            IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                             APInt(32, start, /*isSigned=*/true)));
        state.addAttribute(
            "end", IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                                    APInt(32, newSize, /*isSigned=*/true)));
        for (Type t : op->getResultTypes())
          state.addTypes(replaceLastDimIfMatches(t, oldSize, newSize));
        // Create the adjusted make_range op
        Operation *newMk = rewriter.create(state);
        // Attach group id
        newMk->setAttr(
            "group_id",
            IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                             APInt(32, groupID, /*isSigned=*/true)));
        // Record mapping and continue
        clonedOpsMap[op] = newMk;
        valueMapping[mk.getResult()] = newMk->getResult(0);
        continue;
      }
    }

    // Special-case: arith.constant "value" attribute needs to match new result
    // type
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      Operation *cloned = cloneConstantOp(rewriter, loc, constOp, oldSize,
                                          newSize, groupID, clonedOpsMap);

      // Update value mapping
      for (auto it : llvm::zip(constOp->getResults(), cloned->getResults())) {
        valueMapping[std::get<0>(it)] = std::get<1>(it);
      }
      continue;
    }

    // General: clone ops but make sure input and return are correct shapes
    OperationState state(loc, op->getName());
    // Copy attributes
    state.addAttributes(op->getAttrs());
    // Remap operands
    for (Value v : op->getOperands()) {
      Value mapped = v;
      if (valueMapping.count(v)) {
        mapped = valueMapping[v];
      } else {
        mapped = ensureValueWithNewLastDim(v, oldSize, newSize, rewriter, loc,
                                           valueMapping, clonedOpsMap, groupID);
      }
      state.addOperands(mapped);
    }
    // Update result types
    for (Type t : op->getResultTypes()) {
      state.addTypes(replaceLastDimIfMatches(t, oldSize, newSize));
    }
    // Create the cloned op from the prepared state
    Operation *cloned = rewriter.create(state);
    // Attach group_id on cloned op
    cloned->setAttr("group_id", IntegerAttr::get(
                                    IntegerType::get(rewriter.getContext(), 32),
                                    APInt(32, groupID, /*isSigned=*/true)));
    clonedOpsMap[op] = cloned;
    // Map results
    for (auto it : llvm::zip(op->getResults(), cloned->getResults())) {
      valueMapping[std::get<0>(it)] = std::get<1>(it);
    }
  }
}

/// - Start with one group per embedding size (sorted)

/// - digit = # of groups
///   1)  if groups >= intial grouping (one group per embedding
///       size) dont do anything,
///   2)  if groups < intial grouping: merge the smallest two adjacent groups
///       together

///  If digit = 9 it means Greedy Algo grouping:
/// - Compute target = size of the largest group (do not touch that group)
/// - While there exists groups with size < target, pick the smallest group,
///   merge it with the neighbor (left or right) that makes the merged size
///   come closest to target (tie-break by key difference). Repeat until no
///   merges possible or all groups >= target.

// TODO: this logic can be changed, play around with algo and target size
struct GroupStruct {
  int64_t key;
  SmallVector<const StoreLoadPattern *, 4> members;
  int count() const { return (int)members.size(); }
};

static void greedyBalanceGroups(SmallVector<GroupStruct, 16> &groups) {
  // Find largest group size (target)
  int target = 0;
  for (auto &g : groups)
    target = std::max(target, g.count());

  LLVM_DEBUG({
    llvm::dbgs() << "Largest number of elements in a group (target) = "
                 << target << "\n";
  });

  // Greedy merge loop:
  bool changed = true;
  while (changed) {
    changed = false;
    // find the smallest group with count < target
    int smallestIdx = -1;
    int smallestCount = INT_MAX;
    for (unsigned i = 0; i < groups.size(); ++i) {
      int c = groups[i].count();
      if (c < target && c < smallestCount) {
        smallestCount = c;
        smallestIdx = static_cast<int>(i);
      }
    }
    if (smallestIdx == -1)
      break;

    // try merge smallestIdx with left or right neighbor
    int bestNeighbor = -1;
    int bestMergedCount = -1;
    int bestKeyDiff = INT_MAX;
    int left = smallestIdx - 1;
    int right = smallestIdx + 1;

    // Precompute maxAllowed as integer
    int maxAllowed = (target * 3) / 2;
    // Consider both neighbors with early continue checks to reduce nesting.
    for (int nb : {left, right}) {
      // skip out-of-range neighbors immediately
      if (nb < 0 || nb >= static_cast<int>(groups.size()))
        continue;
      int merged = groups[smallestIdx].count() + groups[nb].count();
      // Prevent merging if the merged group is far larger than target.
      if (merged > maxAllowed)
        continue;
      int penalty = std::abs(merged - target);
      int keyDiff;
      if (groups[smallestIdx].key == -1 || groups[nb].key == -1) {
        keyDiff = INT_MAX / 2;
      } else {
        keyDiff = static_cast<int>(
            std::llabs(groups[smallestIdx].key - groups[nb].key));
      }
      if (bestNeighbor == -1) {
        // first valid candidate
        bestNeighbor = nb;
        bestMergedCount = merged;
        bestKeyDiff = keyDiff;
        continue;
      }
      int currentBestPenalty = std::abs(bestMergedCount - target);
      if (penalty < currentBestPenalty ||
          (penalty == currentBestPenalty && keyDiff < bestKeyDiff)) {
        bestNeighbor = nb;
        bestMergedCount = merged;
        bestKeyDiff = keyDiff;
      }
    }
    if (bestNeighbor == -1) {
      // no neighbor to merge with
      break;
    }
    // Merge smaller into neighbor (keep order: merge into the neighbor's slot
    // and erase the other to keep adjacency)
    int a = std::min(smallestIdx, bestNeighbor);
    int b = std::max(smallestIdx, bestNeighbor);
    // merge groups[a] and groups[b] into groups[a]
    groups[a].members.append(groups[b].members.begin(),
                             groups[b].members.end());
    // update key: prefer known key, else average
    int64_t k1 = groups[a].key;
    int64_t k2 = groups[b].key;
    if (k1 == -1 && k2 != -1)
      groups[a].key = k2;
    else if (k2 == -1 && k1 != -1)
      groups[a].key = k1;
    else
      groups[a].key = (k1 + k2) / 2;
    // erase b
    groups.erase(groups.begin() + b);
    changed = true;
  }
}

static void digitControlledMergeGroups(SmallVector<GroupStruct, 16> &groups,
                                       int desiredGroups) {
  // For desiredGroups > 1: if already small enough, do nothing.
  if ((int)groups.size() <= desiredGroups)
    return;

  // Repeatedly merge adjacent pair whose merged size is smallest
  while ((int)groups.size() > desiredGroups) {
    int bestIdx = -1;
    int bestMergedCount = INT_MAX;
    int bestKeyDiff = INT_MAX;

    for (int i = 0; i + 1 < (int)groups.size(); ++i) {
      int merged = groups[i].count() + groups[i + 1].count();
      int keyDiff;
      if (groups[i].key == -1 || groups[i + 1].key == -1)
        keyDiff = INT_MAX / 2;
      else
        keyDiff = (int)std::llabs(groups[i].key - groups[i + 1].key);

      if (merged < bestMergedCount ||
          (merged == bestMergedCount && keyDiff < bestKeyDiff)) {
        bestIdx = i;
        bestMergedCount = merged;
        bestKeyDiff = keyDiff;
      }
    }

    if (bestIdx == -1) {
      // Nothing to merge (shouldn't happen), break to avoid infinite loop.
      break;
    }

    // Merge bestIdx and bestIdx+1 into bestIdx
    groups[bestIdx].members.append(groups[bestIdx + 1].members.begin(),
                                   groups[bestIdx + 1].members.end());
    int64_t k1 = groups[bestIdx].key;
    int64_t k2 = groups[bestIdx + 1].key;
    if (k1 == -1 && k2 != -1)
      groups[bestIdx].key = k2;
    else if (k2 == -1 && k1 != -1)
      groups[bestIdx].key = k1;
    else
      groups[bestIdx].key = (k1 + k2) / 2;

    groups.erase(groups.begin() + bestIdx + 1);
  }
}

static void groupAndBalancePatterns(
    llvm::ArrayRef<StoreLoadPattern> patterns,
    SmallVectorImpl<SmallVector<const StoreLoadPattern *, 4>> &outGroups,
    int digit) {

  // Map from key to bucket
  llvm::DenseMap<int64_t, SmallVector<const StoreLoadPattern *, 4>> buckets;
  for (const auto &p : patterns) {
    int64_t key = p.embeddingSize;
    buckets[key].push_back(&p);
  }

  // Sorted keys
  SmallVector<int64_t, 16> keys;
  for (auto &kv : buckets)
    keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());

  // Build initial groups
  SmallVector<GroupStruct, 16> groups;
  for (int64_t k : keys) {
    GroupStruct g;
    g.key = k;
    g.members = buckets[k];
    groups.push_back(std::move(g));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Initial groups by embedding size: (" << groups.size()
                 << " groups)\n";
    for (size_t i = 0; i < groups.size(); ++i) {
      llvm::dbgs() << " group[" << i << "] key=" << groups[i].key
                   << " count=" << groups[i].count() << "\n";
      for (auto *p : groups[i].members) {
        llvm::dbgs() << "  - store: ";
        if (p && p->storeOp)
          p->storeOp->print(llvm::dbgs());
        else
          llvm::dbgs() << "<null>";
        llvm::dbgs() << "\n";
      }
    }
  });

  if (groups.empty()) {
    outGroups.clear();
    return;
  }
  if (digit != 9) {
    // Compute what GREEDY would produce (on a copy) and print it so user can
    // see.
    SmallVector<GroupStruct, 16> greedyGroups = groups;
    greedyBalanceGroups(greedyGroups);
    LLVM_DEBUG({
      llvm::dbgs() << "*** GREEDY OPTIMAL SUGGESTS " << greedyGroups.size()
                   << " groups *** \n \n";
    });
  }

  if (digit == 9) {
    LLVM_DEBUG({ llvm::dbgs() << "RUNNING GREEDY \n"; });
    // Call helper for original greedy balancing
    greedyBalanceGroups(groups);
  } else {
    // Call helper for digit-controlled merging (1..8)
    digitControlledMergeGroups(groups, digit);

    // After user-controlled merging, print resulting groups so user sees the
    // final grouping that followed their request.
    LLVM_DEBUG({
      llvm::dbgs() << "USER-SPECIFIED merging to " << digit
                   << " groups produced " << groups.size() << " groups:\n";
    });
  }

  // Convert to output
  outGroups.clear();
  for (auto &g : groups) {
    SmallVector<const StoreLoadPattern *, 4> v;
    v.append(g.members.begin(), g.members.end());
    outGroups.push_back(std::move(v));
  }

  // sort by smaller amount of store, from testing this order gives slightly
  // better performance
  // TODO: play around with it
  std::sort(outGroups.begin(), outGroups.end(),
            [](const SmallVector<const StoreLoadPattern *, 4> &a,
               const SmallVector<const StoreLoadPattern *, 4> &b) {
              // Primary: smaller group first
              if (a.size() != b.size())
                return a.size() < b.size();

              // Tie-breaker: bigger embedding size/key first
              int64_t keyA = a.empty() ? -1 : a.front()->embeddingSize;
              int64_t keyB = b.empty() ? -1 : b.front()->embeddingSize;

              return keyA > keyB;
            });

  LLVM_DEBUG({
    llvm::dbgs() << "Balanced groups: (" << outGroups.size() << " groups)\n";
    for (size_t gi = 0; gi < outGroups.size(); ++gi) {
      auto &g = outGroups[gi];
      int64_t approxKey = (g.empty() ? -1 : g.front()->embeddingSize);
      llvm::dbgs() << " group[" << gi << "] approxKey=" << approxKey
                   << " size=" << g.size() << "\n";
      for (auto *p : g) {
        llvm::dbgs() << "  - store: ";
        if (p && p->storeOp)
          p->storeOp->print(llvm::dbgs());
        else
          llvm::dbgs() << "<null>";
        llvm::dbgs() << "\n";
      }
    }
  });
}

// Find the old group size

// TODO: currently this method/algo works for the cases I tested so far but may
// need to be generalized later
static int64_t
computeGroupOldSize(const SmallVectorImpl<const StoreLoadPattern *> &group,
                    triton::FuncOp parentFunc) {
  // find largest last-dim from broadcasts in dependentOps
  if (auto maybe = findLastDimFromBroadcasts(group)) {
    int64_t sz = *maybe;
    // Try to find a make_range op that had that end value in the function.
    // We don't need to return the op any more, so just return the size.
    Operation *foundMk = nullptr;
    parentFunc.walk([&](triton::MakeRangeOp mk) {
      if (mk.getEnd() == sz) {
        foundMk = mk.getOperation();
      }
    });
    if (foundMk)
      return sz;
  }
  return -1;
}

class LoopRestructureArangeOptimizationPass
    : public impl::LoopRestructureArangeOptimizationBase<
          LoopRestructureArangeOptimizationPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::StringRef myPassName = this->getArgument();
    int digit = mlir::triton::util::getPassColumnDigit(module, myPassName);
    if (digit == 0)
      return;
    if (digit == 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "digit == 1: single group requested; "
                 << "original is already one group -> no changes made. \n");
      return;
    }
    for (Operation &op : module.getOps()) {
      auto func = dyn_cast<triton::FuncOp>(op);
      if (!func)
        continue;
      if (failed(processFunction(func, digit))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult processFunction(triton::FuncOp func, int digit) {
    LLVM_DEBUG(llvm::dbgs()
               << "Processing function: " << func.getName() << "\n");

    // Collect store-load patterns

    // TODO: currently only support store a load value (cat). Only this pattern
    // we define as an independent block of ops we can move/split loops, but we
    // can expand it, any kernel with multiple stores can be split up to
    // seperated indepedent blocks
    SmallVector<StoreLoadPattern, 16> patterns;
    func.walk<WalkOrder::PreOrder>([&](triton::StoreOp storeOp) {
      Value storedVal;

      if (storeOp->getNumOperands() >= 2)
        storedVal = storeOp->getOperand(1);
      else
        WalkResult::advance();

      Operation *def = storedVal.getDefiningOp();
      if (!def)
        return;

      // Case 1: store(load)
      if (isa<triton::LoadOp>(def)) {

        StoreLoadPattern p;
        p.storeOp = storeOp;
        p.loadOp = def;

        findDependentOpsFromStore(storeOp, p.dependentOps);
        findMaskAndConstantOps(storeOp, p);
        extractEmbeddingSize(p);

        patterns.push_back(std::move(p));
        return;
      }

      // Case 2: store(broadcast(load))
      if (isa<triton::BroadcastOp>(def)) {
        Operation *innerLoad = nullptr;

        for (Value v : def->getOperands()) {
          if (Operation *d = v.getDefiningOp()) {
            if (isa<triton::LoadOp>(d)) {
              innerLoad = d;
              break;
            }
          }
        }

        if (!innerLoad)
          return;

        StoreLoadPattern p;
        p.storeOp = storeOp;
        p.loadOp = innerLoad;

        findDependentOpsFromStore(storeOp, p.dependentOps);
        findMaskAndConstantOps(storeOp, p);
        extractEmbeddingSize(p);

        patterns.push_back(std::move(p));
        return;
      }
    });

    if (patterns.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No store-load patterns found\n");
      return success();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Found " << patterns.size() << " store-load patterns\n");

    // Dump all patterns for debugging
    LLVM_DEBUG({
      llvm::dbgs() << "Dumping all patterns:\n";
      for (auto &p : patterns) {
        llvm::dbgs() << "=== StoreLoadPattern ===\n";

        llvm::dbgs() << " storeOp: ";
        if (p.storeOp)
          p.storeOp->print(llvm::dbgs());
        else
          llvm::dbgs() << "<null>";

        llvm::dbgs() << "\n loadOp: ";
        if (p.loadOp)
          p.loadOp->print(llvm::dbgs());
        else
          llvm::dbgs() << "<null>";

        llvm::dbgs() << "\n embeddingSize: " << p.embeddingSize << "\n";

        llvm::dbgs() << " maskOps (" << p.maskOps.size() << "):\n";
        for (Operation *m : p.maskOps) {
          if (m)
            m->print(llvm::dbgs());
          else
            llvm::dbgs() << "<null>";
          llvm::dbgs() << "\n";
        }

        llvm::dbgs() << " constantOps (" << p.constantOps.size() << "):\n";
        for (Operation *c : p.constantOps) {
          if (c)
            c->print(llvm::dbgs());
          else
            llvm::dbgs() << "<null>";
          llvm::dbgs() << "\n";
        }

        llvm::dbgs() << " dependentOps (" << p.dependentOps.size() << "):\n";
        for (Operation *d : p.dependentOps) {
          if (d)
            d->print(llvm::dbgs());
          else
            llvm::dbgs() << "<null>";
          llvm::dbgs() << "\n";
        }

        llvm::dbgs() << "=======================\n";
      }
    });

    for (auto &p : patterns) {
      if (p.embeddingSize <= 0) {
        LLVM_DEBUG(llvm::dbgs() << "SKIP PASS: Embedding size <= 0 \n");
        return success();
      }
    }

    // Group patterns by embedding size and balance them with simple greedy
    // merge
    SmallVector<SmallVector<const StoreLoadPattern *, 4>, 8> groups;
    groupAndBalancePatterns(patterns, groups, digit);
    LLVM_DEBUG(llvm::dbgs() << "Grouped into " << groups.size() << " groups\n");

    // Find a representative scf.for (if any)
    scf::ForOp foundFor = nullptr;
    int loopCount = 0;

    func.walk([&](scf::ForOp f) {
      loopCount++;
      foundFor = f;
      return WalkResult::advance();
    });
    // for now only work with 1 for loop
    if (loopCount > 1) {
      // if more than 1 loop this pass do nothing and just let it pass
      LLVM_DEBUG(llvm::dbgs()
                 << "SKIPPING: More than 1 loop is not supported yet \n");
      return success();
    }
    if (loopCount == 1) {
      for (const auto &p : patterns) {
        Operation *storeParent = p.storeOp->getParentOp();
        bool insideLoop = false;
        // Walk u the parent chain to check if store is nested in foundFor
        while (storeParent && storeParent != func) {
          if (storeParent == foundFor.getOperation()) {
            insideLoop = true;
            break;
          }
          storeParent = storeParent->getParentOp();
        }
        if (!insideLoop) {
          LLVM_DEBUG(llvm::dbgs()
                     << "SKIPPING: Store pattern not inside the single loop: "
                     << p.storeOp << "\n");
          return success();
        }
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "ALL" << patterns.size()
                 << " store patterns verified insid single loop \n");
    }
    // TODO: create a flag to enable splitting loops. Currently from testing
    // splitting has no effects to performance
    bool splitFlag = false;
    if (loopCount == 1) {
      return processSplitIntoGroupsReuseLoop(func, foundFor, groups);
      if (splitFlag)
        return processSplitIntoGroupsLoops(func, foundFor, groups);
    } else {
      return processSplitIntoGroupsNoLoops(func, groups);
    }
  }

  static bool isInsideLoop(Operation *op, scf::ForOp loop) {
    if (!op)
      return false;
    return op->getParentOfType<scf::ForOp>() == loop;
  }
  // Determine max embedding size in this group and
  // collect stores for deletion later
  static int64_t computeMaxEmbeddingAndCollectStores(
      const SmallVector<const StoreLoadPattern *, 4> &group,
      llvm::SmallVector<Operation *, 16> &storesToDelete) {
    int64_t maxSize = 0;
    for (auto *pattern : group) {
      maxSize = std::max(maxSize, pattern->embeddingSize);
      // collect stores for deletion later
      if (pattern->storeOp)
        storesToDelete.push_back(pattern->storeOp);
    }
    return maxSize;
  }

  // Initialize a fake StoreLoadPattern used to reuse clone logic
  static void initFakePattern(StoreLoadPattern &fake,
                              const llvm::DenseSet<Operation *> &ops,
                              int64_t embeddingSize) {
    fake.dependentOps = ops;
    fake.storeOp = nullptr;
    fake.loadOp = nullptr;
    fake.embeddingSize = embeddingSize;
    fake.maskOps.clear();
    fake.constantOps.clear();
  }

  // Split dependent ops: outside-loop vs inside-loop
  static void
  splitDependentOpsByLoop(const SmallVector<const StoreLoadPattern *, 4> &group,
                          scf::ForOp loop,
                          llvm::DenseSet<Operation *> &outsideLoopOps,
                          llvm::DenseSet<Operation *> &insideLoopOps) {
    for (auto *p : group) {
      for (Operation *op : p->dependentOps) {
        if (!op)
          continue;
        if (op->getBlock() == loop.getBody())
          insideLoopOps.insert(op);
        else if (op->getBlock() == loop->getBlock() &&
                 op->isBeforeInBlock(loop))
          outsideLoopOps.insert(op);
      }
      // include the storeOp itself
      if (!p->storeOp)
        continue;
      if (p->storeOp->getBlock() == loop.getBody())
        insideLoopOps.insert(p->storeOp);
      else if (p->storeOp->getBlock() == loop->getBlock() &&
               p->storeOp->isBeforeInBlock(loop))
        outsideLoopOps.insert(p->storeOp);
    }
  }
  LogicalResult processSplitIntoGroupsNoLoops(
      triton::FuncOp func,
      SmallVectorImpl<SmallVector<const StoreLoadPattern *, 4>> &groups) {

    // We clone at the end of the function body.
    PatternRewriter rewriter(func.getContext());

    Block &body = func.getBody().front();

    // Collect stores to delete after all cloning, just deleting the stores will
    // delete all the old code by in other pass like cse
    llvm::SmallVector<Operation *, 16> storesToDelete;

    // Process each group
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      Operation *insertionPoint = body.getTerminator();
      const auto &group = groups[gi];
      if (group.empty())
        continue;

      // Determine max embedding size in this group
      int64_t maxSize =
          computeMaxEmbeddingAndCollectStores(group, storesToDelete);

      // Map to track cloned operations for this group
      llvm::DenseMap<Operation *, Operation *> clonedOpsMap;

      // Compute group old size
      auto oldSize = computeGroupOldSize(group, func);

      // default success if cant find oldSize
      if (oldSize == -1) {
        return success();
      }

      LLVM_DEBUG(llvm::dbgs() << "Processing group with old size: " << oldSize
                              << ", new size: " << maxSize << "\n");
      // Clone operations for this group with new size, clone by group since
      // diff group have diff size
      cloneGroupOperations(rewriter, func.getLoc(), body, insertionPoint, group,
                           oldSize, maxSize, clonedOpsMap,
                           static_cast<int>(gi));
    }

    // erase original stores collected earlier
    for (Operation *store : storesToDelete) {
      if (store && store->getBlock()) {
        rewriter.eraseOp(store);
      }
    }

    return success();
  }

  LogicalResult processSplitIntoGroupsLoops(
      triton::FuncOp func, scf::ForOp loop,
      SmallVectorImpl<SmallVector<const StoreLoadPattern *, 4>> &groups) {

    PatternRewriter rewriter(func.getContext());
    Block &body = func.getBody().front();
    Operation *insertPt = body.getTerminator();
    llvm::SmallVector<Operation *, 16> storesToDelete;

    for (size_t gi = 0; gi < groups.size(); ++gi) {
      const auto &group = groups[gi];
      if (group.empty())
        continue;

      // get new/old size
      int64_t newSize =
          computeMaxEmbeddingAndCollectStores(group, storesToDelete);
      int64_t oldSize = computeGroupOldSize(group, func);
      // default success if cant find oldSize
      if (oldSize == -1) {
        return success();
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "processSplitIntoGroupsLoops: group " << gi
                 << " oldSize=" << oldSize << " newSize=" << newSize << "\n");

      llvm::DenseMap<Operation *, Operation *> clonedOpsMap;

      // Split dependent ops: outside-loop vs inside-loop
      llvm::DenseSet<Operation *> outsideLoopOps;
      llvm::DenseSet<Operation *> insideLoopOps;
      splitDependentOpsByLoop(group, loop, outsideLoopOps, insideLoopOps);

      // Clone outside-loop ops
      if (!outsideLoopOps.empty()) {
        // Create a temporary fake pattern for outside ops only populate the
        // dependentOps so we can reuse no loop clone logic
        StoreLoadPattern fake;
        initFakePattern(fake, outsideLoopOps, newSize);

        SmallVector<const StoreLoadPattern *, 4> fakeGroup;
        fakeGroup.push_back(&fake);

        cloneGroupOperations(rewriter, func.getLoc(), body, insertPt, fakeGroup,
                             oldSize, newSize, clonedOpsMap,
                             static_cast<int>(gi), nullptr);
      }

      // Clone the loop itself (same bounds,step, iter args)
      rewriter.setInsertionPoint(insertPt);
      // Remap loop bounds and init args if they were cloned in the outside step
      Value lowerBound = loop.getLowerBound();
      Value upperBound = loop.getUpperBound();
      Value step = loop.getStep();
      SmallVector<Value> initArgs = loop.getInitArgs();

      if (lowerBound.getDefiningOp() &&
          clonedOpsMap.count(lowerBound.getDefiningOp()))
        lowerBound = clonedOpsMap[lowerBound.getDefiningOp()]->getResult(0);
      if (upperBound.getDefiningOp() &&
          clonedOpsMap.count(upperBound.getDefiningOp()))
        upperBound = clonedOpsMap[upperBound.getDefiningOp()]->getResult(0);
      if (step.getDefiningOp() && clonedOpsMap.count(step.getDefiningOp()))
        step = clonedOpsMap[step.getDefiningOp()]->getResult(0);

      SmallVector<Value> newInitArgs;
      for (Value arg : initArgs) {
        if (arg.getDefiningOp() && clonedOpsMap.count(arg.getDefiningOp())) {
          newInitArgs.push_back(
              clonedOpsMap[arg.getDefiningOp()]->getResult(0));
        } else {
          newInitArgs.push_back(arg);
        }
      }

      auto newFor = rewriter.create<scf::ForOp>(func.getLoc(), lowerBound,
                                                upperBound, step, newInitArgs);

      newFor->setAttr(
          "group_id",
          IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                           APInt(32, gi, /*isSigned=*/true)));

      // Update clonedOpsMap with the new loop mapping
      clonedOpsMap[loop.getOperation()] = newFor.getOperation();

      // Clone inside-loop ops into new loop body using cloneGroupOperations
      if (!insideLoopOps.empty()) {
        // Create a temporary fake pattern for inside ops
        StoreLoadPattern fakeInside;
        initFakePattern(fakeInside, insideLoopOps, newSize);

        SmallVector<const StoreLoadPattern *, 4> fakeInsideGroup;
        fakeInsideGroup.push_back(&fakeInside);

        // Prepare value mapping for the new loop body context
        llvm::DenseMap<Value, Value> valueMapping;
        valueMapping[loop.getInductionVar()] = newFor.getInductionVar();
        // Map iter args from original loop to new loop
        for (auto it :
             llvm::zip(loop.getRegionIterArgs(), newFor.getRegionIterArgs())) {
          valueMapping[std::get<0>(it)] = std::get<1>(it);
        }

        // Set insertion point to the start of the new loop body
        rewriter.setInsertionPointToStart(newFor.getBody());
        // Call cloneGroupOperations for the inside ops with initial value
        // mapping
        cloneGroupOperations(rewriter, func.getLoc(), *loop.getBody(),
                             newFor.getBody()->getTerminator(), fakeInsideGroup,
                             oldSize, newSize, clonedOpsMap,
                             static_cast<int>(gi), &valueMapping);
      }
    }

    // Erase original stores
    for (Operation *s : storesToDelete) {
      if (s && s->getBlock())
        rewriter.eraseOp(s);
    }

    return success();
  }

  LogicalResult processSplitIntoGroupsReuseLoop(
      triton::FuncOp func, scf::ForOp loop,
      SmallVectorImpl<SmallVector<const StoreLoadPattern *, 4>> &groups) {
    PatternRewriter rewriter(func.getContext());
    Block &body = func.getBody().front();
    Operation *insertPt = body.getTerminator();
    llvm::SmallVector<Operation *, 16> storesToDelete;

    // Per-group collected inside/outside ops and sizes
    SmallVector<llvm::DenseSet<Operation *>, 4> perGroupInsideOps(
        groups.size());
    SmallVector<llvm::DenseSet<Operation *>, 4> perGroupOutsideOps(
        groups.size());
    SmallVector<int64_t, 4> groupNewSizes(groups.size(), 0);

    // gather per-group sets and sizes (don't create loops yet).
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      const auto &group = groups[gi];
      if (group.empty())
        continue;
      int64_t newSize = 0;
      for (auto *p : group) {
        newSize = std::max(newSize, p->embeddingSize);
        if (p->storeOp)
          storesToDelete.push_back(p->storeOp);
        for (Operation *op : p->dependentOps) {
          if (!op)
            continue;
          if (op->getBlock() == loop.getBody())
            perGroupInsideOps[gi].insert(op);
          else if (op->getBlock() == loop->getBlock() &&
                   op->isBeforeInBlock(loop))
            perGroupOutsideOps[gi].insert(op);
        }
        // include the storeOp itself
        if (p->storeOp) {
          if (p->storeOp->getBlock() == loop.getBody())
            perGroupInsideOps[gi].insert(p->storeOp);
          else if (p->storeOp->getBlock() == loop->getBlock() &&
                   p->storeOp->isBeforeInBlock(loop))
            perGroupOutsideOps[gi].insert(p->storeOp);
        }
      }
      groupNewSizes[gi] = newSize;
    }

    SmallVector<llvm::DenseMap<Operation *, Operation *>, 4> perGroupClonedMaps(
        groups.size());
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      const auto &group = groups[gi];
      if (group.empty())
        continue;
      if (perGroupOutsideOps[gi].empty())
        continue;

      StoreLoadPattern fakeOutside;
      fakeOutside.dependentOps = perGroupOutsideOps[gi];
      fakeOutside.storeOp = nullptr;
      fakeOutside.loadOp = nullptr;
      fakeOutside.embeddingSize = groupNewSizes[gi];
      fakeOutside.maskOps.clear();
      fakeOutside.constantOps.clear();

      SmallVector<const StoreLoadPattern *, 1> fakeGroup;
      fakeGroup.push_back(&fakeOutside);

      int64_t oldSize = computeGroupOldSize(group, func);
      // default success if cant find oldSize
      if (oldSize == -1) {
        return success();
      }

      // insert clones at insertPt (end of function body).
      cloneGroupOperations(rewriter, func.getLoc(), body, insertPt, fakeGroup,
                           oldSize, groupNewSizes[gi], perGroupClonedMaps[gi],
                           static_cast<int>(gi), /*valueMapping=*/nullptr);
    }
    rewriter.setInsertionPoint(insertPt);
    Value lowerBound = loop.getLowerBound();
    Value upperBound = loop.getUpperBound();
    Value step = loop.getStep();
    SmallVector<Value> initArgs = loop.getInitArgs();
    auto newFor = rewriter.create<scf::ForOp>(func.getLoc(), lowerBound,
                                              upperBound, step, initArgs);
    // We will map the original loop to this new loop when cloning inside ops
    Operation *origLoopOp = loop.getOperation();
    Operation *newForOp = newFor.getOperation();

    // For each group, clone inside-loop ops into newFor, using a per-group
    // clonedOpsMap that contains that group's outside clones + mapping of
    // loop.
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      const auto &group = groups[gi];
      if (group.empty())
        continue;

      if (perGroupInsideOps[gi].empty())
        continue;

      // Build per-group clonedOpsMap: start with the per-group outside clones,
      // then map the original loop -> newFor so cloneGroupOperations remaps
      // uses of the loop to the single cloned loop.
      llvm::DenseMap<Operation *, Operation *> clonedOpsMap;
      for (auto &kv : perGroupClonedMaps[gi])
        clonedOpsMap[kv.first] = kv.second;
      clonedOpsMap[origLoopOp] = newForOp;

      int64_t newSize = groupNewSizes[gi];
      int64_t oldSize = computeGroupOldSize(group, func);
      // default success if cant find oldSize
      if (oldSize == -1) {
        return success();
      }

      StoreLoadPattern fakeInside;
      fakeInside.dependentOps = perGroupInsideOps[gi];
      fakeInside.storeOp = nullptr;
      fakeInside.loadOp = nullptr;
      fakeInside.embeddingSize = newSize;
      fakeInside.maskOps.clear();
      fakeInside.constantOps.clear();
      SmallVector<const StoreLoadPattern *, 1> fakeInsideGroup;
      fakeInsideGroup.push_back(&fakeInside);

      // Value mapping for induction var and region iter args
      llvm::DenseMap<Value, Value> valueMapping;
      valueMapping[loop.getInductionVar()] = newFor.getInductionVar();
      for (auto it :
           llvm::zip(loop.getRegionIterArgs(), newFor.getRegionIterArgs())) {
        valueMapping[std::get<0>(it)] = std::get<1>(it);
      }

      // Insert cloned inside ops at the start of the new loop body
      rewriter.setInsertionPointToStart(newFor.getBody());
      cloneGroupOperations(rewriter, func.getLoc(), *loop.getBody(),
                           newFor.getBody()->getTerminator(), fakeInsideGroup,
                           oldSize, newSize, clonedOpsMap, static_cast<int>(gi),
                           &valueMapping);
    }

    // Erase original stores collected earlier
    for (Operation *s : storesToDelete) {
      if (s && s->getBlock())
        rewriter.eraseOp(s);
    }

    return success();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createLoopRestructureArangeOptimizationPass() {
  return std::make_unique<LoopRestructureArangeOptimizationPass>();
}

} // namespace triton
} // namespace bishengir

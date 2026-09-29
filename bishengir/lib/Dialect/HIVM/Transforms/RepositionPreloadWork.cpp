//===- RepositionPreloadWork.cpp ------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
//
// Post-CVPipelining placement on the unsplit MIX function. Cube/Vector
// interaction is visible here; SplitMixKernel then clones the placement
// onto backup / AIC / AIV so DelayedCrossCoreGSS still sees matching cube
// anchors.
//
// hivm-reposition-preload-work runs immediately after CVPipelining
// and before the second MarkMultiBuffer. It pulls ops that do not use an
// earlier scope's result into a new scope with the same preload_num, placed
// in front of the current scope, then runs the exclusive dest / copy-chain
// sink. The split is limited to ops consumed by a scope that still has a
// later cube: the first subsequent CUBE scope, skipping non-cube scopes in
// between, the same cube test as the copy-chain sink. mb=1 alloc sinking
// stays out of this early pass: those marks are written by MarkMultiBuffer,
// which has not run yet. MarkMultiBuffer sinks them after it marks.
//
// Unified rule: if load / VF / copy in a VECTOR is not read by the next
// CUBE (the nearest later CUBE, other scopes may sit in between), move
// the exclusive chain to the first scope after that CUBE that consumes
// the produced data. Scopes are considered from last to first so a sunk
// consumer can pull an earlier producer on the next pass.
//
// A VECTOR consumer receives the cluster. Several VECTOR consumers: the
// first one, and the value is returned again for later users. A CUBE
// consumer does not receive VECTOR ops; the cluster moves into the
// VECTOR immediately before the first consuming CUBE. No later CUBE, a
// user that still sits before that destination, or a CUBE that consumes
// the return while the producer is already the VECTOR in front of it,
// keeps the cluster. Cross-core preload distance is still preserved.
//
// A load → VF → hir.copy chain that writes a loop-level buffer for a later
// mmad is the same rule: unused intervening mmad → move the whole chain
// (including MTE2) into the VECTOR immediately before the consuming mmad.
//
// Single-slot (mb=1) preload-local allocs are sunk into their unique
// consumer. After SplitMixKernel this pass is a no-op.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/ShapeRegistry.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <optional>
#include <utility>

#define DEBUG_TYPE "hivm-reposition-preload-work"

namespace mlir {
#define GEN_PASS_DEF_SINKEXCLUSIVEPRELOADWORK
#define GEN_PASS_DEF_REPOSITIONPRELOADWORK
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static bool isSinkablePreloadMemref(Value v) {
  auto space = hivm::GetBufferSpaceAttr(v);
  if (!space)
    return false;
  auto as = space->getAddressSpace();
  return as == hivm::AddressSpace::UB || as == hivm::AddressSpace::L1;
}

static bool isViewLikeMemrefOp(Operation *op) {
  return isa<memref::SubViewOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
             memref::ReinterpretCastOp, memref::CastOp,
             memref::MemorySpaceCastOp>(op);
}

static bool hasAnnot(Value v, StringRef name) {
  return utils::getAnnotateOpWithAttr(v, name).has_value();
}

static std::optional<int64_t> getMultiBufferCount(Value v) {
  auto maybe = utils::getAnnotateOpWithAttr(v, hivm::MultiBufferAttr::name);
  if (!maybe)
    return std::nullopt;
  auto intAttr = dyn_cast_or_null<IntegerAttr>(
      (*maybe)->getAttr(hivm::MultiBufferAttr::name));
  if (!intAttr)
    return std::nullopt;
  return intAttr.getInt();
}

/// Single-slot preload-local: `hivm.multi_buffer = 1` and
/// `hivm.preload_local_buffer = 1`. These do not rotate across iterations,
/// so the loop-level lifetime / mark is unnecessary once the alloc lives
/// inside its consumer scope.
static bool isSingleSlotPreloadLocal(Value v) {
  if (!hasAnnot(v, hivm::PreloadLocalBufferAttr::name))
    return false;
  auto mb = getMultiBufferCount(v);
  return mb && *mb == 1;
}

/// Drop the `multi_buffer=1` + `preload_local_buffer=1` marks. Other marks
/// on the same value (TCB, tiling, …) are left alone.
static void eraseSingleSlotPreloadMarks(Value v) {
  SmallVector<annotation::MarkOp> marks;
  DenseSet<Operation *> seen;
  auto collect = [&](StringRef name) {
    for (Operation *op : utils::getAllAnnotateOpsWithAttr(v, name)) {
      if (seen.insert(op).second)
        marks.push_back(cast<annotation::MarkOp>(op));
    }
  };
  collect(hivm::PreloadLocalBufferAttr::name);
  collect(hivm::MultiBufferAttr::name);
  for (annotation::MarkOp mark : marks) {
    mark->removeAttr(hivm::PreloadLocalBufferAttr::name);
    auto intAttr = dyn_cast_or_null<IntegerAttr>(
        mark->getAttr(hivm::MultiBufferAttr::name));
    if (intAttr && intAttr.getInt() == 1)
      mark->removeAttr(hivm::MultiBufferAttr::name);
    if (mark.isAttrEmpty())
      mark.erase();
  }
}

/// Drop `preload_local_buffer` after the load leaves the preload producer.
/// Keep `multi_buffer` so later MarkMultiBuffer can still rotate slots.
static void erasePreloadLocalMark(Value v) {
  SmallVector<annotation::MarkOp> marks;
  for (Operation *op :
       utils::getAllAnnotateOpsWithAttr(v, hivm::PreloadLocalBufferAttr::name))
    marks.push_back(cast<annotation::MarkOp>(op));
  for (annotation::MarkOp mark : marks) {
    mark->removeAttr(hivm::PreloadLocalBufferAttr::name);
    if (mark.isAttrEmpty())
      mark.erase();
  }
}

static scope::ScopeOp enclosingForBodyScope(Operation *op, scf::ForOp forOp) {
  auto scope = op->getParentOfType<scope::ScopeOp>();
  if (!scope || !forOp->isAncestor(scope.getOperation()))
    return nullptr;
  while (scope->getParentOp() != forOp.getOperation()) {
    auto parent = scope->getParentOfType<scope::ScopeOp>();
    if (!parent || !forOp->isAncestor(parent.getOperation()))
      break;
    scope = parent;
  }
  return scope;
}

static Operation *firstUseInScope(Value v, scope::ScopeOp scope) {
  Operation *first = nullptr;
  for (Operation *user : v.getUsers()) {
    if (!scope->isAncestor(user))
      continue;
    Operation *inBody = user;
    while (inBody && inBody->getParentOp() != scope.getOperation())
      inBody = inBody->getParentOp();
    if (!inBody)
      continue;
    if (!first || inBody->isBeforeInBlock(first))
      first = inBody;
  }
  return first;
}

/// Insert at the first real body op so a forwarded src is consumed before
/// the consumer allocates other UB. Putting the writer next to a late
/// dest use lets PlanMemory overlay the src after we drop preload_local.
static Operation *consumerInsertPoint(scope::ScopeOp consumer) {
  Block &body = consumer.getRegion().front();
  for (Operation &op : body) {
    if (isa<hivm::AnchorOp>(&op))
      continue;
    return &op;
  }
  return body.getTerminator();
}

static void collectMarks(Value v, SmallVectorImpl<Operation *> &ops) {
  for (Operation *user : v.getUsers()) {
    if (isa<annotation::MarkOp>(user))
      ops.push_back(user);
  }
}

static void moveOpsBefore(ArrayRef<Operation *> ops, Operation *insertBefore) {
  SmallVector<Operation *> ordered(ops.begin(), ops.end());
  llvm::sort(ordered, [](Operation *a, Operation *b) {
    if (a->getBlock() != b->getBlock())
      return false;
    return a->isBeforeInBlock(b);
  });
  for (Operation *op : ordered)
    op->moveBefore(insertBefore);
}

/// Rebuild `oldScope` so it returns `newReturns`. Cluster ops listed in
/// `leaveBehind` stay in the old body so the caller can move them elsewhere.
static scope::ScopeOp
rebuildScopeLeavingOps(scope::ScopeOp oldScope, ArrayRef<Value> newReturns,
                       const DenseSet<Operation *> &leaveBehind) {
  OpBuilder builder(oldScope);
  SmallVector<Type> types;
  types.reserve(newReturns.size());
  for (Value v : newReturns)
    types.push_back(v.getType());

  auto newScope = builder.create<scope::ScopeOp>(oldScope.getLoc(), types);
  newScope->setAttrs(oldScope->getAttrDictionary());

  Block &oldBody = oldScope.getRegion().front();
  Block *newBody = builder.createBlock(&newScope.getRegion());
  auto *oldTerm = oldBody.getTerminator();
  SmallVector<Operation *> toMove;
  for (Operation &op : oldBody) {
    if (&op == oldTerm || leaveBehind.contains(&op))
      continue;
    toMove.push_back(&op);
  }
  for (Operation *op : toMove)
    op->moveBefore(newBody, newBody->end());

  builder.setInsertionPointToEnd(newBody);
  builder.create<scope::ReturnOp>(oldTerm->getLoc(), newReturns);

  for (auto [oldVal, oldRes] :
       llvm::zip(oldTerm->getOperands(), oldScope.getResults())) {
    for (auto [idx, newVal] : llvm::enumerate(newReturns)) {
      if (newVal != oldVal)
        continue;
      oldRes.replaceAllUsesWith(newScope.getResult(idx));
      break;
    }
  }
  return newScope;
}

static bool isProducerLocalEmpty(Value v, scope::ScopeOp producer) {
  auto empty = v.getDefiningOp<tensor::EmptyOp>();
  return empty && producer->isProperAncestor(empty);
}

static Value remapThroughReturns(Value v, ArrayRef<Value> newReturns,
                                 scope::ScopeOp newProducer) {
  for (auto [idx, retVal] : llvm::enumerate(newReturns)) {
    if (retVal == v)
      return newProducer.getResult(idx);
  }
  return v;
}

/// True when every user of `v` is a mark, a scope.return, or already in
/// `cluster`. Users outside `producer` are ignored: a loop-level alloc can
/// stay put while its producer-local load / to_tensor move.
static bool usersCoveredByCluster(Value v, scope::ScopeOp producer,
                                  const DenseSet<Operation *> &cluster) {
  for (Operation *user : v.getUsers()) {
    if (isa<annotation::MarkOp, scope::ReturnOp>(user))
      continue;
    if (!producer->isProperAncestor(user))
      continue;
    if (!cluster.contains(user))
      return false;
  }
  return true;
}

static bool isAllowedReturnedSliceOp(Operation *op) {
  if (isa<bufferization::ToTensorOp, hivm::LoadOp, memref::AllocOp,
          memref::SubViewOp, memref::ReinterpretCastOp, memref::CastOp,
          tensor::EmptyOp, tensor::ExpandShapeOp, affine::AffineApplyOp>(op))
    return true;
  if (op->getDialect() && op->getDialect()->getNamespace() == "arith")
    return true;
  return isa<DestinationStyleOpInterface>(op) &&
         op->getName().getStringRef().starts_with("hivm.hir.v");
}

static bool opExclusiveInCluster(Operation *op, scope::ScopeOp producer,
                                 const DenseSet<Operation *> &cluster) {
  for (Value res : op->getResults()) {
    if (!usersCoveredByCluster(res, producer, cluster))
      return false;
  }
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (Value init : dps.getDpsInits()) {
      if (!isa<MemRefType>(init.getType()))
        continue;
      if (!usersCoveredByCluster(init, producer, cluster))
        return false;
    }
  }
  return true;
}

static bool tryAddToReturnedCluster(Operation *op, scope::ScopeOp producer,
                                    DenseSet<Operation *> &cluster,
                                    SmallVector<Operation *> &order) {
  if (!op || cluster.contains(op) || !producer->isProperAncestor(op))
    return false;
  if (!isAllowedReturnedSliceOp(op))
    return false;
  DenseSet<Operation *> tentative(cluster);
  tentative.insert(op);
  if (!opExclusiveInCluster(op, producer, tentative))
    return false;
  cluster.insert(op);
  order.push_back(op);
  return true;
}

/// Walk a memref and its view chain (subview / cast / reinterpret) so a
/// `hir.load` that writes a subview of the to_tensor alloc is still pulled
/// into the cluster. Direct users of the root alloc do not include that
/// load, which otherwise leaves MTE2 in the producer.
static bool addMemrefViewTree(Value memref, scope::ScopeOp producer,
                              DenseSet<Operation *> &cluster,
                              SmallVector<Operation *> &order,
                              llvm::function_ref<bool(Operation *)> tryAdd) {
  if (!isa<MemRefType>(memref.getType()))
    return false;
  bool grew = false;
  DenseSet<Value> seen;
  SmallVector<Value> work = {memref};
  while (!work.empty()) {
    Value m = work.pop_back_val();
    if (!seen.insert(m).second)
      continue;
    if (Operation *def = m.getDefiningOp()) {
      if (tryAdd(def))
        grew = true;
      if (isViewLikeMemrefOp(def)) {
        for (Value operand : def->getOperands()) {
          if (isa<MemRefType>(operand.getType()))
            work.push_back(operand);
        }
      }
    }
    for (Operation *user : m.getUsers()) {
      if (tryAdd(user))
        grew = true;
      if (isViewLikeMemrefOp(user)) {
        for (Value res : user->getResults()) {
          if (isa<MemRefType>(res.getType()))
            work.push_back(res);
        }
      }
    }
  }
  return grew;
}

static bool
addMemrefViewTreeToReturnedCluster(Value memref, scope::ScopeOp producer,
                                   DenseSet<Operation *> &cluster,
                                   SmallVector<Operation *> &order) {
  return addMemrefViewTree(
      memref, producer, cluster, order, [&](Operation *op) {
        return tryAddToReturnedCluster(op, producer, cluster, order);
      });
}

/// Grow the exclusive producer-local slice of a returned tensor or leftover
/// memref: the DPS writer, its to_tensor / hir.load (including load-via-
/// subview), address arith / expand_shape, and exclusive tensor.empty.
/// Shared empties stay behind and are listed in `destEmpties` so the
/// consumer can allocate a fresh one. Producer-local shaped extras are
/// forwarded; scalar extras are cloned into the consumer together with
/// any producer-local memref view chain they load (CreatePreload cannot
/// rematerialize index / i1 as scope results).
static bool collectExclusiveReturnedCluster(
    Value root, scope::ScopeOp producer, SmallVector<Operation *> &clusterOps,
    SmallVector<Value> &destEmpties, SmallVector<Value> &extras) {
  Operation *rootOp = root.getDefiningOp();
  if (!rootOp || !producer->isProperAncestor(rootOp) ||
      !isAllowedReturnedSliceOp(rootOp))
    return false;

  DenseSet<Operation *> cluster;
  SmallVector<Operation *> order;
  cluster.insert(rootOp);
  order.push_back(rootOp);

  bool grew = true;
  while (grew) {
    grew = false;
    SmallVector<Operation *> current(order.begin(), order.end());
    for (Operation *op : current) {
      for (Value operand : op->getOperands()) {
        if (tryAddToReturnedCluster(operand.getDefiningOp(), producer, cluster,
                                    order))
          grew = true;
      }
      auto addMemrefUsers = [&](Value memref) {
        if (addMemrefViewTreeToReturnedCluster(memref, producer, cluster,
                                               order))
          grew = true;
      };
      if (isa<MemRefType>(root.getType()) && op == rootOp)
        addMemrefUsers(root);
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op))
        addMemrefUsers(toTensor.getMemref());
      if (auto load = dyn_cast<hivm::LoadOp>(op))
        addMemrefUsers(load.getDst());
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
        for (Value init : dps.getDpsInits())
          addMemrefUsers(init);
      }
    }
  }

  if (!usersCoveredByCluster(root, producer, cluster))
    return false;

  for (Operation *op : order) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    if (!dps)
      continue;
    for (Value init : dps.getDpsInits()) {
      if (!isa<RankedTensorType>(init.getType()))
        continue;
      if (!isProducerLocalEmpty(init, producer))
        continue;
      if (cluster.contains(init.getDefiningOp()))
        continue;
      if (!llvm::is_contained(destEmpties, init))
        destEmpties.push_back(init);
    }
  }

  for (Operation *op : order) {
    for (Value operand : op->getOperands()) {
      if (llvm::is_contained(destEmpties, operand))
        continue;
      Operation *def = operand.getDefiningOp();
      if (!def || !producer->isProperAncestor(def) || cluster.contains(def))
        continue;
      if (!llvm::is_contained(extras, operand))
        extras.push_back(operand);
    }
  }

  clusterOps = order;
  for (Operation *op : order) {
    for (Value res : op->getResults())
      collectMarks(res, clusterOps);
  }
  return true;
}

static Value cloneProducerScalar(Value v, scope::ScopeOp producer,
                                 Operation *insertBefore,
                                 DenseMap<Value, Value> &cloned);
static void collectForwardedDepsForScalarExtras(scope::ScopeOp producer,
                                                SmallVector<Value> &extras);
static void remapClonedScalarOperands(const DenseMap<Value, Value> &cloned,
                                      ArrayRef<Value> newReturns,
                                      scope::ScopeOp newProducer);

static bool isMmadLikeOp(Operation *op) {
  return op->getName().getStringRef().contains("mmad");
}

static bool scopeHasMmad(scope::ScopeOp scope) {
  bool found = false;
  scope.walk([&](Operation *op) {
    if (isMmadLikeOp(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool scopeBlocksCopySink(scope::ScopeOp scope) {
  if (scopeHasMmad(scope))
    return true;
  if (auto core = scope->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::kPipelinedLoopCoreTypeAttrName))
    return core.getTcoretype() == hivm::TCoreType::CUBE;
  return false;
}

static std::optional<int64_t> getScopePreloadNum(scope::ScopeOp scope) {
  if (auto attr = scope->getAttrOfType<IntegerAttr>("hivm.preload_num"))
    return attr.getInt();
  return std::nullopt;
}

static std::optional<hivm::TCoreType> getScopeCoreType(scope::ScopeOp scope) {
  if (auto core = scope->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::kPipelinedLoopCoreTypeAttrName))
    return core.getTcoretype();
  return std::nullopt;
}

static void collectAccessorScopes(Value v, scf::ForOp forOp,
                                  SmallVectorImpl<scope::ScopeOp> &out) {
  if (auto scope = v.getDefiningOp<scope::ScopeOp>()) {
    out.push_back(scope);
    return;
  }
  Value mem = v;
  if (auto toTensor = v.getDefiningOp<bufferization::ToTensorOp>())
    mem = toTensor.getMemref();
  if (!isa<MemRefType>(mem.getType()))
    return;
  // Climb to the root buffer so aliases created through other views count.
  while (Operation *def = mem.getDefiningOp()) {
    if (!isViewLikeMemrefOp(def))
      break;
    mem = def->getOperand(0);
  }

  if (!isSinkablePreloadMemref(mem))
    return;

  DenseSet<Value> seen;
  SmallVector<Value> work = {mem};
  while (!work.empty()) {
    Value m = work.pop_back_val();
    if (!seen.insert(m).second)
      continue;
    for (Operation *user : m.getUsers()) {
      if (isa<annotation::MarkOp>(user))
        continue;
      if (isViewLikeMemrefOp(user)) {
        for (Value res : user->getResults())
          if (isa<MemRefType>(res.getType()))
            work.push_back(res);
        continue;
      }
      if (auto scope = enclosingForBodyScope(user, forOp))
        out.push_back(scope);
    }
  }
}

/// Sinking moves `cluster` from `producer`'s preload stage to `consumer`'s.
/// If the cluster shares memory with a scope on the other core, that stage
/// distance is what the buffer's slot count was sized for, and required
/// to stay odd, so refuse the sink.
static bool changesCrossCoreDistance(ArrayRef<Operation *> cluster,
                                     scope::ScopeOp producer,
                                     scope::ScopeOp consumer,
                                     scf::ForOp forOp) {
  if (getScopePreloadNum(producer) == getScopePreloadNum(consumer))
    return false;
  auto consumerCore = getScopeCoreType(consumer);
  DenseSet<Operation *> inCluster(cluster.begin(), cluster.end());
  for (Operation *op : cluster) {
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp();
          def && inCluster.contains(def))
        continue;
      SmallVector<scope::ScopeOp> accessors;
      collectAccessorScopes(operand, forOp, accessors);
      for (scope::ScopeOp scope : accessors) {
        if (scope == producer || scope == consumer)
          continue;
        auto core = getScopeCoreType(scope);
        if (!core || !consumerCore || *core != *consumerCore)
          return true;
      }
    }
  }
  return false;
}

/// True when `scope` reads or writes `v`, including through a view /
/// to_tensor / extract_slice that still carries the returned value.
static bool valueUsedInsideScope(Value v, scope::ScopeOp scope) {
  SmallVector<Value> work = {v};
  DenseSet<Value> seen;
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    for (Operation *user : cur.getUsers()) {
      if (isa<annotation::MarkOp>(user))
        continue;
      if (scope->isAncestor(user))
        return true;
      if (isViewLikeMemrefOp(user) ||
          isa<bufferization::ToTensorOp, tensor::ExtractSliceOp,
              tensor::ExpandShapeOp, tensor::CollapseShapeOp>(user)) {
        for (Value res : user->getResults())
          work.push_back(res);
      }
    }
  }
  return false;
}

static scope::ScopeOp firstLaterCubeScope(scope::ScopeOp producer) {
  for (Operation *op = producer->getNextNode(); op; op = op->getNextNode()) {
    auto scope = dyn_cast<scope::ScopeOp>(op);
    if (scope && scopeBlocksCopySink(scope))
      return scope;
  }
  return nullptr;
}

static scope::ScopeOp vectorScopeImmediatelyBefore(scope::ScopeOp cube) {
  for (Operation *op = cube->getPrevNode(); op; op = op->getPrevNode()) {
    auto scope = dyn_cast<scope::ScopeOp>(op);
    if (!scope)
      continue;
    if (scopeBlocksCopySink(scope))
      return nullptr;
    return scope;
  }
  return nullptr;
}

/// Earliest for-body scope after `after` that uses `scopeRes`.
static scope::ScopeOp firstConsumingScopeAfter(scope::ScopeOp after,
                                               Value scopeRes,
                                               scf::ForOp forOp) {
  scope::ScopeOp first;
  for (Operation *user : scopeRes.getUsers()) {
    auto scope = enclosingForBodyScope(user, forOp);
    if (!scope || !after->isBeforeInBlock(scope))
      continue;
    if (!first || scope->isBeforeInBlock(first))
      first = scope;
  }
  return first;
}

/// A use in `producer`, outside the for-body scopes, or strictly before
/// `dest` cannot see a value defined in `dest`.
static bool hasUserBefore(Value scopeRes, scope::ScopeOp dest,
                          scope::ScopeOp producer, scf::ForOp forOp) {
  for (Operation *user : scopeRes.getUsers()) {
    auto scope = enclosingForBodyScope(user, forOp);
    if (!scope || scope == producer || scope->isBeforeInBlock(dest))
      return true;
  }
  return false;
}

/// VECTOR producer only. Cross the nearest later CUBE when it does not
/// consume `scopeRes`, and land in the first consumer after that CUBE.
/// A VECTOR consumer takes the cluster; a CUBE consumer takes the VECTOR
/// immediately in front of it. The nearest CUBE itself consuming the
/// return uses that same "VECTOR in front" slot, and only when it is
/// strictly after `producer`.
static scope::ScopeOp returnedTensorSinkTarget(scope::ScopeOp producer,
                                               Value scopeRes,
                                               scf::ForOp forOp) {
  if (scopeBlocksCopySink(producer))
    return nullptr;
  scope::ScopeOp cube = firstLaterCubeScope(producer);
  if (!cube)
    return nullptr;

  scope::ScopeOp dest;
  if (valueUsedInsideScope(scopeRes, cube)) {
    dest = vectorScopeImmediatelyBefore(cube);
  } else {
    scope::ScopeOp consumer = firstConsumingScopeAfter(cube, scopeRes, forOp);
    if (!consumer)
      return nullptr;
    dest = scopeBlocksCopySink(consumer)
               ? vectorScopeImmediatelyBefore(consumer)
               : consumer;
  }
  if (!dest || dest == producer || !producer->isBeforeInBlock(dest))
    return nullptr;
  if (scopeBlocksCopySink(dest))
    return nullptr;
  if (hasUserBefore(scopeRes, dest, producer, forOp))
    return nullptr;
  return dest;
}

/// Later users of a value that now lives in `consumer` must read a scope
/// result. Append `inner` to the return when it is not already yielded.
static scope::ScopeOp exportSunkValueToLaterUsers(scope::ScopeOp consumer,
                                                  Value inner,
                                                  scope::ScopeOp producer) {
  SmallVector<Operation *> outside;
  for (Operation *user : inner.getUsers()) {
    if (consumer->isAncestor(user) || producer->isAncestor(user))
      continue;
    outside.push_back(user);
  }
  if (outside.empty())
    return consumer;

  auto ret =
      dyn_cast<scope::ReturnOp>(consumer.getRegion().front().getTerminator());
  if (!ret)
    return consumer;
  auto rewire = [&](Value exported) {
    for (Operation *user : outside) {
      for (OpOperand &use : user->getOpOperands()) {
        if (use.get() == inner)
          use.set(exported);
      }
    }
  };
  for (auto [idx, operand] : llvm::enumerate(ret.getOperands())) {
    if (operand != inner)
      continue;
    rewire(consumer.getResult(idx));
    return consumer;
  }

  SmallVector<Value> rets(ret.getOperands().begin(), ret.getOperands().end());
  rets.push_back(inner);
  DenseSet<Operation *> none;
  scope::ScopeOp rebuilt = rebuildScopeLeavingOps(consumer, rets, none);
  rewire(rebuilt.getResult(rets.size() - 1));
  consumer.erase();
  return rebuilt;
}

/// Move exclusive returned tensor / leftover-memref clusters to the first
/// consumer after the nearest later CUBE that does not consume the return,
/// including MTE2 load through a subview. A CUBE consumer receives the
/// cluster in the VECTOR immediately before it. Shared tensor.empty stays
/// in the producer; the consumer gets a fresh empty. Scalar extras are
/// cloned (not returned). Producers are visited last-to-first, and the
/// walk repeats so an earlier producer can follow a sunk consumer.
static LogicalResult sinkReturnedTensorsToConsumerImpl(scf::ForOp forOp) {
  for (int iter = 0; iter < 32; ++iter) {
    SmallVector<scope::ScopeOp> scopes;
    for (Operation &op : *forOp.getBody()) {
      if (auto scope = dyn_cast<scope::ScopeOp>(&op))
        scopes.push_back(scope);
    }

    bool changed = false;
    // Last scope first: sinking a later consumer moves the use, so an
    // earlier producer can follow into that same scope on the next pass.
    for (scope::ScopeOp producer : llvm::reverse(scopes)) {
      if (!producer->getBlock())
        continue;
      auto ret = dyn_cast<scope::ReturnOp>(
          producer.getRegion().front().getTerminator());
      if (!ret)
        continue;

      struct Move {
        unsigned resIdx;
        Value dest;
        scope::ScopeOp consumer;
        SmallVector<Value> extras;
        SmallVector<Value> destEmpties;
        SmallVector<Operation *> cluster;
      };
      SmallVector<Move> moves;
      DenseSet<unsigned> moveIdx;
      DenseSet<Operation *> claimed;

      for (auto [idx, retVal] : llvm::enumerate(ret.getOperands())) {
        if (!isa<RankedTensorType, MemRefType>(retVal.getType()))
          continue;
        Value scopeRes = producer.getResult(idx);
        auto consumer = returnedTensorSinkTarget(producer, scopeRes, forOp);
        if (!consumer)
          continue;

        SmallVector<Operation *> cluster;
        SmallVector<Value> destEmpties;
        SmallVector<Value> extras;
        if (!collectExclusiveReturnedCluster(retVal, producer, cluster,
                                             destEmpties, extras))
          continue;
        collectForwardedDepsForScalarExtras(producer, extras);
        // VECTOR ops stay out of CUBE scopes. A CUBE consumer is rewritten
        // to the VECTOR in front of it before we get here.
        if (scopeBlocksCopySink(consumer))
          continue;
        if (cluster.empty())
          continue;
        if (changesCrossCoreDistance(cluster, producer, consumer, forOp))
          continue;
        if (llvm::any_of(cluster,
                         [&](Operation *op) { return claimed.contains(op); }))
          continue;

        Move m;
        m.resIdx = idx;
        m.dest = retVal;
        m.consumer = consumer;
        m.extras = std::move(extras);
        m.destEmpties = std::move(destEmpties);
        m.cluster = std::move(cluster);
        claimed.insert(m.cluster.begin(), m.cluster.end());
        moveIdx.insert(idx);
        moves.push_back(std::move(m));
      }
      if (moves.empty())
        continue;

      SmallVector<Value> newReturns;
      for (auto [idx, retVal] : llvm::enumerate(ret.getOperands())) {
        if (!moveIdx.contains(idx))
          newReturns.push_back(retVal);
      }
      DenseSet<Value> alreadyReturned(newReturns.begin(), newReturns.end());
      SmallVector<Value> extrasToAdd;
      for (const Move &m : moves) {
        for (Value extra : m.extras) {
          if (!isa<ShapedType, MemRefType>(extra.getType()))
            continue;
          if (alreadyReturned.insert(extra).second)
            extrasToAdd.push_back(extra);
        }
      }
      newReturns.append(extrasToAdd.begin(), extrasToAdd.end());

      DenseSet<Operation *> leaveBehind;
      for (const Move &m : moves)
        leaveBehind.insert(m.cluster.begin(), m.cluster.end());

      for (const Move &m : moves) {
        Value scopeRes = producer.getResult(m.resIdx);
        scopeRes.replaceAllUsesWith(m.dest);
      }

      scope::ScopeOp newProducer =
          rebuildScopeLeavingOps(producer, newReturns, leaveBehind);

      for (const Move &m : moves) {
        Operation *insertBefore = firstUseInScope(m.dest, m.consumer);
        if (!insertBefore)
          insertBefore = consumerInsertPoint(m.consumer);

        DenseMap<Value, Value> clonedScalars;
        for (Value extra : m.extras) {
          if (isa<ShapedType, MemRefType>(extra.getType()))
            continue;
          cloneProducerScalar(extra, newProducer, insertBefore, clonedScalars);
        }
        remapClonedScalarOperands(clonedScalars, newReturns, newProducer);

        for (Operation *op : m.cluster) {
          for (OpOperand &operand : op->getOpOperands()) {
            if (llvm::is_contained(m.destEmpties, operand.get()))
              continue;
            if (auto it = clonedScalars.find(operand.get());
                it != clonedScalars.end())
              operand.set(it->second);
            else
              operand.set(
                  remapThroughReturns(operand.get(), newReturns, newProducer));
          }
        }

        OpBuilder builder(insertBefore);
        for (Operation *op : m.cluster) {
          for (OpOperand &operand : op->getOpOperands()) {
            if (!llvm::is_contained(m.destEmpties, operand.get()))
              continue;
            auto oldEmpty = operand.get().getDefiningOp<tensor::EmptyOp>();
            if (!oldEmpty)
              continue;
            SmallVector<Value> dyn;
            for (Value sz : oldEmpty.getDynamicSizes()) {
              if (auto it = clonedScalars.find(sz); it != clonedScalars.end())
                dyn.push_back(it->second);
              else
                dyn.push_back(remapThroughReturns(sz, newReturns, newProducer));
            }
            operand.set(builder.create<tensor::EmptyOp>(
                oldEmpty.getLoc(), oldEmpty.getType(), dyn));
          }
        }
        moveOpsBefore(m.cluster, insertBefore);
        for (Operation *op : m.cluster) {
          auto alloc = dyn_cast<memref::AllocOp>(op);
          if (!alloc)
            continue;
          erasePreloadLocalMark(alloc.getResult());
        }
      }

      // A CUBE consumer, or a later VECTOR, still needs the sunk value as
      // a scope result of the first consumer.
      DenseMap<Operation *, scope::ScopeOp> liveConsumer;
      for (Move &m : moves) {
        Operation *key = m.consumer.getOperation();
        scope::ScopeOp current = liveConsumer.lookup(key);
        if (!current)
          current = m.consumer;
        liveConsumer[key] =
            exportSunkValueToLaterUsers(current, m.dest, producer);
      }

      producer.erase();

      if (failed(mlir::verify(forOp))) {
        forOp.emitError("verification failed after sinking dest tensor");
        return failure();
      }
      changed = true;
      break;
    }
    if (!changed)
      return success();
  }
  forOp.emitError("dest tensor sink did not reach a fixpoint");
  return failure();
}

static bool hasInterveningMmadScope(scope::ScopeOp producer,
                                    scope::ScopeOp consumer) {
  for (Operation *op = producer->getNextNode();
       op && op != consumer.getOperation(); op = op->getNextNode()) {
    auto scope = dyn_cast<scope::ScopeOp>(op);
    if (scope && scopeBlocksCopySink(scope))
      return true;
  }
  return false;
}

static scope::ScopeOp createVectorScopeBefore(scope::ScopeOp cube) {
  OpBuilder builder(cube);
  auto newScope = builder.create<scope::ScopeOp>(cube.getLoc(), TypeRange{});
  newScope.setNoInline(true);
  newScope->setAttr(
      hivm::kPipelinedLoopCoreTypeAttrName,
      hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR));
  Block *body = builder.createBlock(&newScope.getRegion());
  builder.setInsertionPointToEnd(body);
  builder.create<scope::ReturnOp>(cube.getLoc());
  return newScope;
}

/// VECTOR immediately before `consumer`. Do not walk past an unused CUBE:
/// that would land in an earlier VECTOR and keep blocking that CUBE. If
/// the nearest sibling is the producer or a CUBE, insert a fresh VECTOR.
static scope::ScopeOp pickCopyChainDestScope(scope::ScopeOp producer,
                                             scope::ScopeOp consumer) {
  for (Operation *op = consumer->getPrevNode(); op; op = op->getPrevNode()) {
    auto scope = dyn_cast<scope::ScopeOp>(op);
    if (!scope)
      continue;
    if (!scopeBlocksCopySink(scope) && scope != producer)
      return scope;
    break;
  }
  return createVectorScopeBefore(consumer);
}

static bool isAllowedCopySliceOp(Operation *op) {
  if (isa<hivm::CopyOp, tensor::ExtractSliceOp, tensor::InsertSliceOp,
          func::CallOp>(op))
    return true;
  return isAllowedReturnedSliceOp(op);
}

static bool isLoopLevelCopyDest(hivm::CopyOp copy, scope::ScopeOp producer,
                                scf::ForOp forOp) {
  Value dest = copy.getDst();
  Operation *destDef = dest.getDefiningOp();
  if (destDef && producer->isAncestor(destDef))
    return false;
  if (auto toTensor = dest.getDefiningOp<bufferization::ToTensorOp>())
    return toTensor->getParentOp() == forOp.getOperation();
  if (!isa<MemRefType>(dest.getType()))
    return false;
  if (destDef)
    return destDef->getParentOp() == forOp.getOperation();
  auto ba = dyn_cast<BlockArgument>(dest);
  return ba && ba.getOwner() == forOp.getBody();
}

static bool tryAddToCopyCluster(Operation *op, scope::ScopeOp producer,
                                DenseSet<Operation *> &cluster,
                                SmallVector<Operation *> &order) {
  if (!op || cluster.contains(op) || !producer->isProperAncestor(op))
    return false;
  if (!isAllowedCopySliceOp(op))
    return false;
  DenseSet<Operation *> tentative(cluster);
  tentative.insert(op);
  if (!opExclusiveInCluster(op, producer, tentative))
    return false;
  cluster.insert(op);
  order.push_back(op);
  return true;
}

static bool addMemrefViewTreeToCopyCluster(Value memref,
                                           scope::ScopeOp producer,
                                           DenseSet<Operation *> &cluster,
                                           SmallVector<Operation *> &order) {
  return addMemrefViewTree(
      memref, producer, cluster, order, [&](Operation *op) {
        return tryAddToCopyCluster(op, producer, cluster, order);
      });
}

static bool collectExclusiveCopyCluster(hivm::CopyOp copy,
                                        scope::ScopeOp producer,
                                        SmallVector<Operation *> &clusterOps,
                                        SmallVector<Value> &destEmpties,
                                        SmallVector<Value> &extras) {
  DenseSet<Operation *> cluster;
  SmallVector<Operation *> order;
  cluster.insert(copy.getOperation());
  order.push_back(copy.getOperation());

  bool grew = true;
  while (grew) {
    grew = false;
    SmallVector<Operation *> current(order.begin(), order.end());
    for (Operation *op : current) {
      for (Value operand : op->getOperands()) {
        if (tryAddToCopyCluster(operand.getDefiningOp(), producer, cluster,
                                order))
          grew = true;
      }
      auto addMemrefUsers = [&](Value memref) {
        if (addMemrefViewTreeToCopyCluster(memref, producer, cluster, order))
          grew = true;
      };
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op))
        addMemrefUsers(toTensor.getMemref());
      if (auto load = dyn_cast<hivm::LoadOp>(op))
        addMemrefUsers(load.getDst());
      if (auto cpy = dyn_cast<hivm::CopyOp>(op))
        addMemrefUsers(cpy.getDst());
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
        for (Value init : dps.getDpsInits())
          addMemrefUsers(init);
      }
    }
  }

  for (Operation *op : order) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    if (!dps)
      continue;
    for (Value init : dps.getDpsInits()) {
      if (!isa<RankedTensorType>(init.getType()))
        continue;
      if (!isProducerLocalEmpty(init, producer))
        continue;
      if (cluster.contains(init.getDefiningOp()))
        continue;
      if (!llvm::is_contained(destEmpties, init))
        destEmpties.push_back(init);
    }
  }

  for (Operation *op : order) {
    for (Value operand : op->getOperands()) {
      if (llvm::is_contained(destEmpties, operand))
        continue;
      Operation *def = operand.getDefiningOp();
      if (!def || !producer->isProperAncestor(def) || cluster.contains(def))
        continue;
      if (!llvm::is_contained(extras, operand))
        extras.push_back(operand);
    }
  }

  clusterOps = order;
  for (Operation *op : order) {
    for (Value res : op->getResults())
      collectMarks(res, clusterOps);
    auto cpy = dyn_cast<hivm::CopyOp>(op);
    if (!cpy)
      continue;
    for (Operation *user : cpy.getDst().getUsers()) {
      if (isa<annotation::MarkOp>(user) && producer->isProperAncestor(user))
        clusterOps.push_back(user);
    }
  }
  return !clusterOps.empty();
}

static bool isCopyDestViewUser(Operation *user) {
  return isViewLikeMemrefOp(user) ||
         isa<bufferization::ToTensorOp, bufferization::ToMemrefOp,
             tensor::ExtractSliceOp, tensor::InsertSliceOp,
             tensor::ExpandShapeOp, tensor::CollapseShapeOp>(user);
}

/// Unique later for-body scope that has an mmad reading `dest` (through
/// views / to_tensor / extract_slice). Marks and the producer copy itself
/// are ignored. Any later non-mmad consumer rejects the move.
static scope::ScopeOp uniqueLaterMmadUserScope(Value dest,
                                               scope::ScopeOp producer,
                                               scf::ForOp forOp) {
  DenseSet<scope::ScopeOp> found;
  DenseSet<Value> seen;
  SmallVector<Value> work = {dest};
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    for (Operation *user : v.getUsers()) {
      if (isa<annotation::MarkOp>(user))
        continue;
      if (auto copy = dyn_cast<hivm::CopyOp>(user)) {
        if (copy.getDst() == dest)
          continue;
      }
      if (isCopyDestViewUser(user)) {
        for (Value res : user->getResults())
          work.push_back(res);
        continue;
      }
      auto scope = enclosingForBodyScope(user, forOp);
      if (!scope || scope == producer)
        continue;
      if (!producer->isBeforeInBlock(scope))
        return nullptr;
      if (!isMmadLikeOp(user))
        return nullptr;
      found.insert(scope);
    }
  }
  if (found.size() != 1)
    return nullptr;
  return *found.begin();
}

static void moveEarlyToTensorsAfter(Value dest, scope::ScopeOp destScope) {
  SmallVector<Operation *> early;
  if (auto toTensor = dest.getDefiningOp<bufferization::ToTensorOp>()) {
    if (toTensor->getBlock() == destScope->getBlock() &&
        toTensor->isBeforeInBlock(destScope))
      early.push_back(toTensor);
  }
  for (Operation *user : dest.getUsers()) {
    auto toTensor = dyn_cast<bufferization::ToTensorOp>(user);
    if (!toTensor)
      continue;
    if (destScope->isAncestor(toTensor))
      continue;
    if (!toTensor->getBlock() || toTensor->getBlock() != destScope->getBlock())
      continue;
    if (!toTensor->isBeforeInBlock(destScope))
      continue;
    early.push_back(toTensor);
  }
  llvm::sort(early,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *op : llvm::reverse(early))
    op->moveAfter(destScope);
}

/// Clone a producer-local scalar (index / i1 / arith) and its scalar deps
/// into `destScope`. CreatePreload cannot rematerialize those types as
/// scope results (`Unhandled scope result case`), so they must not be
/// forwarded through `scope.return`.
///
/// A scalar `memref.load` may read a producer-local view
/// (`reinterpret_cast` / `subview` / `cast`). Those views often stay in
/// the producer because another load still uses them, so they cannot
/// join the dest-sink cluster. Clone the view chain too: its memref root
/// is a function arg or loop alloc that already dominates the consumer,
/// and its index operands are cloned as scalars. Do not clone allocs.
static bool shouldCloneProducerValue(Value v, Operation *def,
                                     scope::ScopeOp producer) {
  if (!def || !producer->isProperAncestor(def))
    return false;
  // MemRefType is a ShapedType; check views before the tensor early-out.
  if (isa<MemRefType>(v.getType()))
    return isViewLikeMemrefOp(def);
  if (isa<ShapedType>(v.getType()))
    return false;
  return true;
}

static Value cloneProducerScalar(Value v, scope::ScopeOp producer,
                                 Operation *insertBefore,
                                 DenseMap<Value, Value> &cloned) {
  if (auto it = cloned.find(v); it != cloned.end())
    return it->second;
  Operation *def = v.getDefiningOp();
  if (!shouldCloneProducerValue(v, def, producer))
    return v;
  IRMapping mapping;
  for (Value operand : def->getOperands()) {
    Value mapped = cloneProducerScalar(operand, producer, insertBefore, cloned);
    if (mapped != operand)
      mapping.map(operand, mapped);
  }
  OpBuilder builder(insertBefore);
  Operation *copy = builder.clone(*def, mapping);
  if (copy->getNumResults() == 0)
    return v;
  cloned[v] = copy->getResult(0);
  return copy->getResult(0);
}

/// A cloned scalar (`memref.load` / `tensor.extract`) may still read a
/// producer-local tensor or leftover memref. Those cannot be rematerialized
/// as views, so they must ride `scope.return` and be remapped afterwards.
static void collectForwardedCloneDeps(Value v, scope::ScopeOp producer,
                                      SmallVector<Value> &extras) {
  DenseSet<Value> seen;
  SmallVector<Value> work = {v};
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    Operation *def = cur.getDefiningOp();
    if (!def || !producer->isProperAncestor(def))
      continue;
    if (isa<MemRefType>(cur.getType())) {
      if (isViewLikeMemrefOp(def)) {
        work.append(def->operand_begin(), def->operand_end());
        continue;
      }
      if (!llvm::is_contained(extras, cur))
        extras.push_back(cur);
      continue;
    }
    if (isa<ShapedType>(cur.getType())) {
      if (!llvm::is_contained(extras, cur))
        extras.push_back(cur);
      continue;
    }
    work.append(def->operand_begin(), def->operand_end());
  }
}

static void collectForwardedDepsForScalarExtras(scope::ScopeOp producer,
                                                SmallVector<Value> &extras) {
  SmallVector<Value> scalars;
  for (Value extra : extras) {
    if (!isa<ShapedType, MemRefType>(extra.getType()))
      scalars.push_back(extra);
  }
  for (Value scalar : scalars)
    collectForwardedCloneDeps(scalar, producer, extras);
}

static void remapClonedScalarOperands(const DenseMap<Value, Value> &cloned,
                                      ArrayRef<Value> newReturns,
                                      scope::ScopeOp newProducer) {
  for (auto &entry : cloned) {
    Operation *copy = entry.second.getDefiningOp();
    if (!copy)
      continue;
    for (OpOperand &operand : copy->getOpOperands()) {
      operand.set(remapThroughReturns(operand.get(), newReturns, newProducer));
    }
  }
}

/// Move a load→VF→hir.copy chain past unused mmad scopes so the copy sits
/// in the VECTOR immediately before the mmad that actually reads it.
static LogicalResult sinkCopyChainPastUnusedMmad(scf::ForOp forOp) {
  for (int iter = 0; iter < 32; ++iter) {
    SmallVector<scope::ScopeOp> scopes;
    for (Operation &op : *forOp.getBody()) {
      if (auto scope = dyn_cast<scope::ScopeOp>(&op))
        scopes.push_back(scope);
    }

    bool changed = false;
    for (scope::ScopeOp producer : scopes) {
      if (!producer->getBlock() || scopeBlocksCopySink(producer))
        continue;
      auto ret = dyn_cast<scope::ReturnOp>(
          producer.getRegion().front().getTerminator());
      if (!ret)
        continue;

      SmallVector<hivm::CopyOp> copies;
      producer.walk([&](hivm::CopyOp op) { copies.push_back(op); });

      hivm::CopyOp copy;
      scope::ScopeOp consumer;
      for (hivm::CopyOp candidate : copies) {
        if (!isLoopLevelCopyDest(candidate, producer, forOp))
          continue;
        auto later =
            uniqueLaterMmadUserScope(candidate.getDst(), producer, forOp);
        if (!later || !scopeHasMmad(later))
          continue;
        if (!hasInterveningMmadScope(producer, later))
          continue;
        copy = candidate;
        consumer = later;
        break;
      }
      if (!copy)
        continue;

      Value dest = copy.getDst();
      scope::ScopeOp destScope = pickCopyChainDestScope(producer, consumer);
      if (!destScope || destScope == producer)
        continue;

      SmallVector<Operation *> cluster;
      SmallVector<Value> destEmpties;
      SmallVector<Value> extras;
      if (!collectExclusiveCopyCluster(copy, producer, cluster, destEmpties,
                                       extras))
        continue;
      collectForwardedDepsForScalarExtras(producer, extras);

      SmallVector<Value> newReturns(ret.getOperands().begin(),
                                    ret.getOperands().end());
      DenseSet<Value> alreadyReturned(newReturns.begin(), newReturns.end());
      SmallVector<Value> scalarExtras;
      for (Value extra : extras) {
        if (isa<ShapedType, MemRefType>(extra.getType())) {
          if (alreadyReturned.insert(extra).second)
            newReturns.push_back(extra);
        } else {
          scalarExtras.push_back(extra);
        }
      }

      DenseSet<Operation *> leaveBehind(cluster.begin(), cluster.end());
      scope::ScopeOp newProducer =
          rebuildScopeLeavingOps(producer, newReturns, leaveBehind);

      Operation *insertBefore = destScope.getRegion().front().getTerminator();
      DenseMap<Value, Value> clonedScalars;
      for (Value extra : scalarExtras)
        cloneProducerScalar(extra, newProducer, insertBefore, clonedScalars);
      remapClonedScalarOperands(clonedScalars, newReturns, newProducer);

      for (Operation *op : cluster) {
        for (OpOperand &operand : op->getOpOperands()) {
          if (llvm::is_contained(destEmpties, operand.get()))
            continue;
          if (auto it = clonedScalars.find(operand.get());
              it != clonedScalars.end())
            operand.set(it->second);
          else
            operand.set(
                remapThroughReturns(operand.get(), newReturns, newProducer));
        }
      }

      OpBuilder builder(insertBefore);
      for (Operation *op : cluster) {
        for (OpOperand &operand : op->getOpOperands()) {
          if (!llvm::is_contained(destEmpties, operand.get()))
            continue;
          auto oldEmpty = operand.get().getDefiningOp<tensor::EmptyOp>();
          if (!oldEmpty)
            continue;
          SmallVector<Value> dyn;
          for (Value sz : oldEmpty.getDynamicSizes()) {
            if (auto it = clonedScalars.find(sz); it != clonedScalars.end())
              dyn.push_back(it->second);
            else
              dyn.push_back(remapThroughReturns(sz, newReturns, newProducer));
          }
          operand.set(builder.create<tensor::EmptyOp>(oldEmpty.getLoc(),
                                                      oldEmpty.getType(), dyn));
        }
      }
      moveOpsBefore(cluster, insertBefore);
      producer.erase();
      moveEarlyToTensorsAfter(dest, destScope);

      if (failed(mlir::verify(forOp))) {
        forOp.emitError("verification failed after sinking copy chain");
        return failure();
      }
      changed = true;
      break;
    }
    if (!changed)
      return success();
  }
  forOp.emitError("copy-chain sink did not reach a fixpoint");
  return failure();
}

/// Walk marks and memref views to the unique for-body scope that actually
/// consumes `alloc`. View ops (subview / reshape / cast) in the loop body
/// are treated as part of the alloc, not as an outside use.
static std::optional<std::pair<scope::ScopeOp, SmallVector<Operation *>>>
uniqueAllocUserScope(Value alloc, scf::ForOp forOp) {
  DenseSet<scope::ScopeOp> scopes;
  SmallVector<Operation *> views;
  DenseSet<Operation *> seen;
  SmallVector<Operation *> work(alloc.getUsers().begin(),
                                alloc.getUsers().end());
  while (!work.empty()) {
    Operation *user = work.pop_back_val();
    if (!seen.insert(user).second)
      continue;
    if (isa<annotation::MarkOp>(user))
      continue;
    if (isa<scope::ReturnOp, scf::YieldOp>(user))
      return std::nullopt;
    if (isViewLikeMemrefOp(user)) {
      views.push_back(user);
      for (Operation *next : user->getUsers())
        work.push_back(next);
      continue;
    }
    auto scope = enclosingForBodyScope(user, forOp);
    if (!scope)
      return std::nullopt;
    scopes.insert(scope);
  }
  if (scopes.size() != 1)
    return std::nullopt;
  return std::make_pair(*scopes.begin(), std::move(views));
}

/// `multi_buffer=1` + `preload_local_buffer=1` allocs whose only non-mark
/// (and non-view) users sit in one scope become ordinary scope-private
/// buffers: delete those two marks and, if the alloc still lives in the
/// parent loop body, move it (and its view chain) next to the first use.
/// TCB / tiling marks stay. `multi_buffer > 1` stays at loop scope so
/// PlanMemory can still rotate slots across iterations.
static void sinkSingleSlotPreloadAllocsImpl(scf::ForOp forOp) {
  SmallVector<memref::AllocOp> allocs;
  forOp.walk([&](memref::AllocOp alloc) {
    Value v = alloc.getResult();
    if (!isSinkablePreloadMemref(v) || !isSingleSlotPreloadLocal(v))
      return;
    if (llvm::any_of(v.getUsers(), [](Operation *user) {
          return isa<scope::ReturnOp, scf::YieldOp>(user);
        }))
      return;
    allocs.push_back(alloc);
  });

  for (memref::AllocOp alloc : allocs) {
    auto target = uniqueAllocUserScope(alloc, forOp);
    if (!target)
      continue;

    scope::ScopeOp scope = target->first;
    ArrayRef<Operation *> views = target->second;
    if (!scope->isAncestor(alloc.getOperation())) {
      Operation *insertBefore = firstUseInScope(alloc, scope);
      for (Operation *view : views) {
        if (insertBefore)
          break;
        insertBefore = firstUseInScope(view->getResult(0), scope);
      }
      if (!insertBefore)
        insertBefore = consumerInsertPoint(scope);
      SmallVector<Operation *> cluster;
      cluster.push_back(alloc.getOperation());
      collectMarks(alloc, cluster);
      for (Operation *view : views) {
        if (!scope->isAncestor(view))
          cluster.push_back(view);
      }
      moveOpsBefore(cluster, insertBefore);
    }
    eraseSingleSlotPreloadMarks(alloc);
  }
}

static LogicalResult sinkPreloadLocalAllocs(scf::ForOp forOp) {
  sinkSingleSlotPreloadAllocsImpl(forOp);
  return success();
}

static constexpr llvm::StringLiteral kIndependentPreloadPrefixAttr =
    "hivm.independent_preload_prefix";

static bool isVectorPreloadScope(scope::ScopeOp scope) {
  if (!scope->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name))
    return false;
  auto core = scope->getAttrOfType<hivm::TCoreTypeAttr>(
      hivm::kPipelinedLoopCoreTypeAttrName);
  return core && core.getTcoretype() == hivm::TCoreType::VECTOR;
}

/// First scope after `scope` that is CUBE (core type or an mmad). Non-cube
/// scopes in between are skipped. This is "the next cube scope", not "the
/// immediate next scope is cube".
static scope::ScopeOp firstSubsequentCubeScope(scope::ScopeOp scope) {
  for (Operation *op = scope->getNextNode(); op; op = op->getNextNode()) {
    auto next = dyn_cast<scope::ScopeOp>(op);
    if (next && scopeBlocksCopySink(next))
      return next;
  }
  return nullptr;
}

/// Sibling scope in `scope`'s block that contains `user`, if any.
static scope::ScopeOp siblingScopeOf(Operation *user, scope::ScopeOp scope) {
  for (Operation *parent = user->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getBlock() != scope->getBlock())
      continue;
    return dyn_cast<scope::ScopeOp>(parent);
  }
  return nullptr;
}

/// True when every user of `op` is consumed by `scope` itself, by `cube`, or
/// by a sibling that still sits before `cube`. A use after that first cube
/// belongs to a later consumer and must stay in `scope`.
static bool consumedBeforeOrByCube(Operation *op, scope::ScopeOp scope,
                                   scope::ScopeOp cube) {
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (scope->isAncestor(user))
        continue;
      scope::ScopeOp userScope = siblingScopeOf(user, scope);
      if (!userScope)
        return false;
      if (userScope == cube)
        continue;
      if (!userScope->isBeforeInBlock(cube))
        return false;
    }
  }
  return true;
}

static Value traceMemrefRoot(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto sub = dyn_cast<memref::SubViewOp>(def)) {
      v = sub.getSource();
      continue;
    }
    if (auto exp = dyn_cast<memref::ExpandShapeOp>(def)) {
      v = exp.getSrc();
      continue;
    }
    if (auto col = dyn_cast<memref::CollapseShapeOp>(def)) {
      v = col.getSrc();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      v = cast.getSource();
      continue;
    }
    if (auto rcast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      v = rcast.getSource();
      continue;
    }
    if (auto mcast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
      v = mcast.getSource();
      continue;
    }
    break;
  }
  return v;
}

/// True when an earlier sibling scope reads or writes this memref (through
/// views). Cube fixpipe into a UB alloc, then `to_tensor` between scopes, is
/// the usual way a vector scope observes the previous scope.
static bool
memrefTouchedByEarlierScope(Value memref,
                            const DenseSet<scope::ScopeOp> &earlier) {
  Value root = traceMemrefRoot(memref);
  SmallVector<Operation *> work(root.getUsers().begin(), root.getUsers().end());
  DenseSet<Operation *> seen;
  while (!work.empty()) {
    Operation *user = work.pop_back_val();
    if (!seen.insert(user).second)
      continue;
    if (isViewLikeMemrefOp(user) || isa<annotation::MarkOp>(user) ||
        isa<memref::MemorySpaceCastOp>(user)) {
      for (Value res : user->getResults())
        for (Operation *next : res.getUsers())
          work.push_back(next);
      continue;
    }
    for (scope::ScopeOp scope : earlier)
      if (scope->isProperAncestor(user))
        return true;
  }
  return false;
}

static bool definedByEarlierScope(Value v, scope::ScopeOp current,
                                  const DenseSet<scope::ScopeOp> &earlier,
                                  DenseSet<Value> &seen) {
  if (!seen.insert(v).second)
    return false;
  if (auto result = dyn_cast<OpResult>(v)) {
    if (auto scope = dyn_cast<scope::ScopeOp>(result.getOwner()))
      if (earlier.contains(scope))
        return true;
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  for (scope::ScopeOp scope : earlier)
    if (scope->isProperAncestor(def))
      return true;
  if (current->isProperAncestor(def))
    return false;
  if (isa<MemRefType>(v.getType()) && memrefTouchedByEarlierScope(v, earlier))
    return true;
  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(def)) {
    if (memrefTouchedByEarlierScope(toTensor.getMemref(), earlier))
      return true;
  }
  for (Value operand : def->getOperands())
    if (definedByEarlierScope(operand, current, earlier, seen))
      return true;
  return false;
}

static Operation *topLevelInScope(Operation *op, scope::ScopeOp scope) {
  while (op && op->getParentOp() != scope.getOperation())
    op = op->getParentOp();
  return op;
}

static bool opDependsOnEarlier(Operation *op, scope::ScopeOp scope,
                               const DenseSet<scope::ScopeOp> &earlier,
                               const DenseSet<Operation *> &dependentTops) {
  bool bad = false;
  op->walk([&](Operation *inner) {
    if (bad)
      return WalkResult::interrupt();
    for (Value operand : inner->getOperands()) {
      DenseSet<Value> seen;
      if (definedByEarlierScope(operand, scope, earlier, seen)) {
        bad = true;
        return WalkResult::interrupt();
      }
      Operation *def = operand.getDefiningOp();
      if (!def || !scope->isProperAncestor(def))
        continue;
      Operation *top = topLevelInScope(def, scope);
      if (top && dependentTops.contains(top)) {
        bad = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return bad;
}

static bool isHoistableWork(Operation *op) {
  if (isa<func::CallOp, bufferization::ToTensorOp>(op))
    return true;
  return op->getDialect() && op->getDialect()->getNamespace() == "hivm";
}

/// A bare destination buffer: no operands, so the dependence scan always calls
/// it independent even when the only reader is a DPS op that stays behind.
static bool isBareDestination(Operation *op) {
  return isa<tensor::EmptyOp, bufferization::AllocTensorOp, memref::AllocOp,
             memref::AllocaOp>(op);
}

/// Drop bare destinations that no hoisted op reads back out of `independent`.
///
/// Hoisting one buys nothing - an empty tensor costs no cycles - and it makes
/// the consumer reach its destination as a `scope.return` result instead of as
/// the defining alloc. MarkStrideAlign then puts `hivm.stride_align_dims` on
/// the scope result, EnableStrideAlign finds no alloc to reshape, and the UB
/// padding is silently dropped: `kernel_sdpa_bwd_q`'s 8x64x16 tile stayed at a
/// 2048-byte row stride instead of the 8x65x16x1 / 1040-element layout the
/// unhoisted pipeline produces.
static void
keepBareDestinationsWithConsumer(SmallVectorImpl<Operation *> &independent,
                                 scope::ScopeOp scope) {
  DenseSet<Operation *> indepSet(independent.begin(), independent.end());
  bool shrank = true;
  while (shrank) {
    shrank = false;
    for (Operation *op : independent) {
      if (!indepSet.contains(op) || !isBareDestination(op))
        continue;
      bool readByHoisted = false;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          Operation *top = topLevelInScope(user, scope);
          if (top && top != op && indepSet.contains(top)) {
            readByHoisted = true;
            break;
          }
        }
        if (readByHoisted)
          break;
      }
      if (readByHoisted)
        continue;
      indepSet.erase(op);
      shrank = true;
    }
  }
  llvm::erase_if(independent,
                 [&](Operation *op) { return !indepSet.contains(op); });
}

/// CreatePreload's skip branch can rebuild a scope result only when it is a
/// shaped value (bufferized before that pass), a pointer cast, or an `arith`
/// op whose operands are likewise rebuildable. `affine.apply` index results
/// are none of those and hit `Unhandled scope result case`.
static bool scalarChainRematerializable(Value v, scope::ScopeOp scope,
                                        DenseSet<Value> &seen) {
  if (isa<ShapedType>(v.getType()))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def || !scope->isProperAncestor(def))
    return true;
  if (!seen.insert(v).second)
    return true;
  bool remat =
      isa<hivm::PointerCastOp>(def) ||
      (def->getDialect() && isa<arith::ArithDialect>(def->getDialect()));
  if (!remat)
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return scalarChainRematerializable(operand, scope, seen);
  });
}

/// Keep an op that would export a non-rematerializable scalar in the consumer
/// scope. Its users there still need the value, and CreatePreload cannot
/// invent it in the preload skip branch.
static void keepUnsafeScalarExports(SmallVectorImpl<Operation *> &independent,
                                    scope::ScopeOp scope) {
  bool shrank = true;
  while (shrank) {
    shrank = false;
    DenseSet<Operation *> indepSet(independent.begin(), independent.end());
    SmallVector<Operation *> drop;
    for (Operation *op : independent) {
      for (Value result : op->getResults()) {
        bool escapes = llvm::any_of(result.getUsers(), [&](Operation *user) {
          if (!scope->isProperAncestor(user))
            return true;
          Operation *top = topLevelInScope(user, scope);
          return !top || !indepSet.contains(top);
        });
        if (!escapes)
          continue;
        DenseSet<Value> seen;
        if (!scalarChainRematerializable(result, scope, seen)) {
          drop.push_back(op);
          break;
        }
      }
    }
    if (drop.empty())
      break;
    DenseSet<Operation *> dropSet(drop.begin(), drop.end());
    llvm::erase_if(independent,
                   [&](Operation *op) { return dropSet.contains(op); });
    shrank = true;
  }
}

/// Move ops that do not use an earlier sibling scope into a new scope placed
/// immediately before `scope`, with the same preload_num. Ops that use the
/// earlier scope stay. Only ops consumed by this scope, or by a sibling at
/// or before the first subsequent CUBE scope, are split out. A VECTOR scope
/// with no later cube is left whole. MergeSamePreloadScopes must not fold
/// the new scope back; it is marked `hivm.independent_preload_prefix`.
static LogicalResult
hoistIndependentOpsInScope(scope::ScopeOp scope,
                           const DenseSet<scope::ScopeOp> &earlier) {
  if (!isVectorPreloadScope(scope) ||
      scope->hasAttr(kIndependentPreloadPrefixAttr))
    return success();
  scope::ScopeOp cube = firstSubsequentCubeScope(scope);
  if (!cube)
    return success();
  Block &body = scope.getRegion().front();
  Operation *terminator = body.getTerminator();

  DenseSet<Operation *> dependent;
  bool grew = true;
  while (grew) {
    grew = false;
    for (Operation &op : body) {
      if (&op == terminator || dependent.contains(&op))
        continue;
      if (!opDependsOnEarlier(&op, scope, earlier, dependent))
        continue;
      dependent.insert(&op);
      grew = true;
    }
  }

  SmallVector<Operation *> independent;
  for (Operation &op : body) {
    if (&op == terminator || dependent.contains(&op))
      continue;
    independent.push_back(&op);
  }
  llvm::erase_if(independent, [&](Operation *op) {
    return !consumedBeforeOrByCube(op, scope, cube);
  });
  keepBareDestinationsWithConsumer(independent, scope);
  keepUnsafeScalarExports(independent, scope);
  keepBareDestinationsWithConsumer(independent, scope);
  if (independent.empty() || dependent.empty())
    return success();
  if (!llvm::any_of(independent, isHoistableWork))
    return success();

  DenseSet<Operation *> indepSet(independent.begin(), independent.end());
  SmallVector<Value> exported;
  for (Operation *op : independent) {
    for (Value result : op->getResults()) {
      bool escapes = llvm::any_of(result.getUsers(), [&](Operation *user) {
        if (!scope->isProperAncestor(user))
          return true;
        Operation *top = topLevelInScope(user, scope);
        return !top || !indepSet.contains(top);
      });
      if (escapes)
        exported.push_back(result);
    }
  }

  OpBuilder builder(scope);
  SmallVector<Type> types;
  for (Value v : exported)
    types.push_back(v.getType());
  auto newScope = builder.create<scope::ScopeOp>(scope.getLoc(), types);
  newScope->setAttrs(scope->getAttrDictionary());
  newScope->setAttr(kIndependentPreloadPrefixAttr, builder.getUnitAttr());
  newScope.setNoInline(true);
  Block *newBody = builder.createBlock(&newScope.getRegion());
  for (Operation *op : independent)
    op->moveBefore(newBody, newBody->end());
  builder.setInsertionPointToEnd(newBody);
  builder.create<scope::ReturnOp>(scope.getLoc(), exported);
  for (auto [val, res] : llvm::zip(exported, newScope.getResults())) {
    val.replaceUsesWithIf(res, [&](OpOperand &use) {
      return !newScope->isProperAncestor(use.getOwner());
    });
  }
  return success();
}

static LogicalResult hoistIndependentPreloadWork(scf::ForOp forOp) {
  SmallVector<scope::ScopeOp> scopes;
  for (Operation &op : *forOp.getBody()) {
    if (auto scope = dyn_cast<scope::ScopeOp>(&op))
      scopes.push_back(scope);
  }
  for (scope::ScopeOp scope : scopes) {
    if (!scope->getBlock())
      continue;
    DenseSet<scope::ScopeOp> earlier;
    for (Operation *prev = scope->getPrevNode(); prev;
         prev = prev->getPrevNode()) {
      if (auto earlierScope = dyn_cast<scope::ScopeOp>(prev))
        earlier.insert(earlierScope);
    }
    if (failed(hoistIndependentOpsInScope(scope, earlier)))
      return failure();
  }
  return success();
}

struct SinkExclusivePreloadWorkPass
    : public impl::SinkExclusivePreloadWorkBase<SinkExclusivePreloadWorkPass> {
  using Base = impl::SinkExclusivePreloadWorkBase<SinkExclusivePreloadWorkPass>;
  using Base::Base;
  void runOnOperation() override;
};

/// True after SplitMixKernel cloned the MIX function into `_mix_aic` /
/// `_mix_aiv`. Dest-sink on those copies (or the mix backup) is not
/// isomorphic, so GSS cannot map mix-side sync onto cube anchors.
static bool isSplitMixModule(ModuleOp moduleOp) {
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (hivm::hasMixFuncSuffix(funcOp.getName()))
      return true;
  }
  return false;
}

static bool dominatesOp(Operation *def, Operation *user) {
  if (def->getBlock() == user->getBlock())
    return def->isBeforeInBlock(user);
  Operation *parent = user->getParentOp();
  if (!parent)
    return false;
  if (def->getBlock() == parent->getBlock())
    return def->isBeforeInBlock(parent);
  return dominatesOp(def, parent);
}

/// Follow scope results back to the value the scope actually returns, so a
/// matmul that reads a left matrix through an earlier scope matches the
/// matmul that produced that matrix inside the scope.
static Value traceThroughScopeResults(Value v) {
  DenseSet<Value> seen;
  while (Operation *def = v.getDefiningOp()) {
    if (!seen.insert(v).second)
      break;
    auto scope = dyn_cast<scope::ScopeOp>(def);
    if (!scope)
      break;
    auto ret =
        dyn_cast<scope::ReturnOp>(scope.getRegion().front().getTerminator());
    if (!ret)
      break;
    auto result = dyn_cast<OpResult>(v);
    if (!result || result.getResultNumber() >= ret->getNumOperands())
      break;
    v = ret->getOperand(result.getResultNumber());
  }
  return v;
}

/// Identity of a matmul's left matrix: the memref behind a `to_tensor`, or
/// the SSA value itself when the matrix is not buffer-backed.
static Value leftMatrixIdentity(Value v) {
  v = traceThroughScopeResults(v);
  auto toTensor = v.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor)
    return v;
  Value mem = toTensor.getMemref();
  while (Operation *def = mem.getDefiningOp()) {
    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def))
      mem = cast.getSource();
    else if (auto sub = dyn_cast<memref::SubViewOp>(def))
      mem = sub.getSource();
    else if (auto castOp = dyn_cast<memref::CastOp>(def))
      mem = castOp.getSource();
    else if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def))
      mem = reinterpret.getSource();
    else
      break;
  }
  return mem;
}

/// Outermost op that still sits directly in `loop`'s body and contains `op`.
static Operation *outermostInLoop(Operation *op, scf::ForOp loop) {
  if (!op || !loop->isAncestor(op))
    return nullptr;
  while (op->getParentOp() != loop)
    op = op->getParentOp();
  return op;
}

/// The loop-body op that should be treated as the end of this matmul: its
/// fixpipe when that fixpipe is a later sibling container, otherwise the
/// matmul's own container.
static Operation *matmulEndInLoop(hivm::MmadL1Op mmad, scf::ForOp loop) {
  Operation *end = outermostInLoop(mmad, loop);
  for (Operation *user : mmad->getUsers()) {
    if (!isa<hivm::FixpipeOp>(user))
      continue;
    Operation *userTop = outermostInLoop(user, loop);
    if (userTop && end && userTop->getBlock() == end->getBlock() &&
        end->isBeforeInBlock(userTop))
      end = userTop;
  }
  return end;
}

/// Move a later `mmadL1` that reads the same left matrix as an earlier one
/// up to that earlier matmul. Containers are the ops directly in the loop
/// body, so a matmul wrapped in a scope moves together with that scope.
/// The later matmul stays put when an operand it needs is produced by
/// something that has to remain between them.
static void clusterSharedLeftMatrixMads(scf::ForOp loop) {
  bool moved = true;
  while (moved) {
    moved = false;
    SmallVector<hivm::MmadL1Op> mmads;
    loop.walk([&](hivm::MmadL1Op mmad) { mmads.push_back(mmad); });

    for (size_t i = 0; i < mmads.size() && !moved; ++i) {
      hivm::MmadL1Op first = mmads[i];
      Operation *anchor = matmulEndInLoop(first, loop);
      if (!anchor)
        continue;
      for (size_t j = i + 1; j < mmads.size() && !moved; ++j) {
        hivm::MmadL1Op second = mmads[j];
        if (leftMatrixIdentity(second.getA()) !=
            leftMatrixIdentity(first.getA()))
          continue;
        Operation *secondMmadTop = outermostInLoop(second, loop);
        Operation *secondEnd = matmulEndInLoop(second, loop);
        if (!secondMmadTop || !secondEnd ||
            secondEnd->getBlock() != anchor->getBlock() ||
            !anchor->isBeforeInBlock(secondMmadTop))
          continue;

        DenseSet<Operation *> cluster;
        cluster.insert(secondMmadTop);
        cluster.insert(secondEnd);
        bool grew = true;
        while (grew) {
          grew = false;
          SmallVector<Operation *> snapshot(cluster.begin(), cluster.end());
          for (Operation *root : snapshot) {
            root->walk([&](Operation *op) {
              for (Value operand : op->getOperands()) {
                Operation *def = operand.getDefiningOp();
                if (!def)
                  continue;
                Operation *defTop = outermostInLoop(def, loop);
                if (!defTop || cluster.contains(defTop) ||
                    defTop->getBlock() != anchor->getBlock())
                  continue;
                if (anchor->isBeforeInBlock(defTop) &&
                    (defTop == secondEnd ||
                     defTop->isBeforeInBlock(secondEnd))) {
                  cluster.insert(defTop);
                  grew = true;
                }
              }
            });
          }
        }

        bool stuck = false;
        for (Operation *root : cluster) {
          root->walk([&](Operation *op) {
            if (stuck)
              return;
            for (Value operand : op->getOperands()) {
              Operation *def = operand.getDefiningOp();
              if (!def || root->isAncestor(def))
                continue;
              Operation *defTop = outermostInLoop(def, loop);
              if (defTop && cluster.contains(defTop))
                continue;
              if (def == anchor || anchor->isAncestor(def))
                continue;
              if (!dominatesOp(def, anchor))
                stuck = true;
            }
          });
          if (stuck)
            break;
        }
        if (stuck)
          continue;

        bool alreadyTogether = true;
        for (Operation *op = anchor->getNextNode(); op && op != secondMmadTop;
             op = op->getNextNode()) {
          if (!cluster.contains(op)) {
            alreadyTogether = false;
            break;
          }
        }
        if (alreadyTogether)
          continue;

        SmallVector<Operation *> ordered(cluster.begin(), cluster.end());
        llvm::sort(ordered, [](Operation *a, Operation *b) {
          return a->isBeforeInBlock(b);
        });
        Operation *ip = anchor;
        for (Operation *op : ordered) {
          op->moveAfter(ip);
          ip = op;
        }
        moved = true;
      }
    }
  }
}

static bool allScopesPinnedToZero(scf::ForOp forOp) {
  bool sawScope = false;
  bool allZero = true;
  forOp.walk([&](scope::ScopeOp scope) {
    sawScope = true;
    auto attr = scope->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name);
    if (!attr || attr.getInt() != 0)
      allZero = false;
  });
  return sawScope && allZero;
}

void SinkExclusivePreloadWorkPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  if (isSplitMixModule(moduleOp))
    return;
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (!hivm::allowLoopShapeHeuristics(this->bypassShapeRegistry,
                                        funcOp.getName(), this->enablePreload,
                                        this->workspaceMultiBufferNum))
      continue;
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (scf::ForOp forOp : forOps) {
      // Every scope at preload_num 0 means the next iteration is not
      // overlapped. Sinking a returned dest across those scopes can place
      // it before the value it reads.
      if (!allScopesPinnedToZero(forOp) &&
          failed(sinkReturnedTensorsToConsumerImpl(forOp)))
        return signalPassFailure();
      if (failed(sinkCopyChainPastUnusedMmad(forOp)))
        return signalPassFailure();
      if (failed(sinkPreloadLocalAllocs(forOp)))
        return signalPassFailure();
    }
  }
}

struct RepositionPreloadWorkPass
    : public impl::RepositionPreloadWorkBase<RepositionPreloadWorkPass> {
  using Base = impl::RepositionPreloadWorkBase<RepositionPreloadWorkPass>;
  using Base::Base;
  void runOnOperation() override;
};

void RepositionPreloadWorkPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  if (isSplitMixModule(moduleOp))
    return;
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (!hivm::allowLoopShapeHeuristics(this->bypassShapeRegistry,
                                        funcOp.getName(), this->enablePreload,
                                        this->workspaceMultiBufferNum))
      continue;
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (scf::ForOp forOp : forOps) {
      clusterSharedLeftMatrixMads(forOp);
      // Every scope at preload_num 0 means the next iteration is not
      // overlapped. Sinking a returned dest across those scopes can place
      // it before the value it reads.
      if (!allScopesPinnedToZero(forOp) &&
          failed(sinkReturnedTensorsToConsumerImpl(forOp)))
        return signalPassFailure();
      if (failed(sinkCopyChainPastUnusedMmad(forOp)))
        return signalPassFailure();
      if (failed(hoistIndependentPreloadWork(forOp)))
        return signalPassFailure();
    }
  }
}

} // namespace

LogicalResult mlir::hivm::sinkReturnedTensorsToConsumer(scf::ForOp forOp) {
  return sinkReturnedTensorsToConsumerImpl(forOp);
}

void mlir::hivm::sinkSingleSlotPreloadAllocs(scf::ForOp forOp) {
  sinkSingleSlotPreloadAllocsImpl(forOp);
}

std::unique_ptr<Pass> mlir::hivm::createSinkExclusivePreloadWorkPass(
    const SinkExclusivePreloadWorkOptions &options) {
  return std::make_unique<SinkExclusivePreloadWorkPass>(options);
}

std::unique_ptr<Pass> mlir::hivm::createRepositionPreloadWorkPass(
    const RepositionPreloadWorkOptions &options) {
  return std::make_unique<RepositionPreloadWorkPass>(options);
}

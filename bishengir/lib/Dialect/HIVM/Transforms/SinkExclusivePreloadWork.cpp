//===- SinkExclusivePreloadWork.cpp ----------------------------*- C++ -*-===//
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
// Post-CVPipelining sink on the unsplit MIX function. Cube/Vector
// interaction is visible here; InsertAnchors / SplitMixKernel then clone
// the placement onto backup / AIC / AIV so DelayedCrossCoreGSS still sees
// matching cube anchors.
//
// Unified rule: if load / VF / copy in a VECTOR is not read by the next
// CUBE, move the exclusive chain past that cube so it sits next to the
// mmad (or the VECTOR that feeds it). Sitting next to the consumer already
// means there is no intervening unused mmad, so the chain stays.
//
// Exclusive dest / src clusters (returned tensor or leftover memref,
// DPS writer, to_tensor, hir.load through a subview, address arith,
// expand) move into the unique later VECTOR consumer. An unused CUBE
// between producer and consumer does not keep the MTE2 load behind.
// A CUBE consumer is left to the copy-chain sink so VECTOR ops are not
// inserted into a cube scope.
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

#define DEBUG_TYPE "hivm-sink-exclusive-preload-work"

namespace mlir {
#define GEN_PASS_DEF_SINKEXCLUSIVEPRELOADWORK
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
  auto intAttr =
      dyn_cast_or_null<IntegerAttr>((*maybe)->getAttr(hivm::MultiBufferAttr::name));
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
    auto intAttr =
        dyn_cast_or_null<IntegerAttr>(mark->getAttr(hivm::MultiBufferAttr::name));
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

  auto newScope =
      builder.create<scope::ScopeOp>(oldScope.getLoc(), types);
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

/// Unique for-body scope that uses `scopeRes`, and it must sit after
/// `producer`. Multiple user scopes or a use outside any sibling scope
/// (e.g. `scf.yield`) reject the move.
static scope::ScopeOp uniqueLaterUserScope(Value scopeRes,
                                           scope::ScopeOp producer,
                                           scf::ForOp forOp) {
  scope::ScopeOp found;
  for (Operation *user : scopeRes.getUsers()) {
    auto scope = enclosingForBodyScope(user, forOp);
    if (!scope || scope == producer)
      return nullptr;
    if (!producer->isBeforeInBlock(scope))
      return nullptr;
    if (found && found != scope)
      return nullptr;
    found = scope;
  }
  return found;
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
static bool addMemrefViewTree(
    Value memref, scope::ScopeOp producer, DenseSet<Operation *> &cluster,
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

static bool addMemrefViewTreeToReturnedCluster(Value memref,
                                               scope::ScopeOp producer,
                                               DenseSet<Operation *> &cluster,
                                               SmallVector<Operation *> &order) {
  return addMemrefViewTree(memref, producer, cluster, order, [&](Operation *op) {
    return tryAddToReturnedCluster(op, producer, cluster, order);
  });
}

/// Grow the exclusive producer-local slice of a returned tensor or leftover
/// memref: the DPS writer, its to_tensor / hir.load (including load-via-
/// subview), address arith / expand_shape, and exclusive tensor.empty.
/// Shared empties stay behind and are listed in `destEmpties` so the
/// consumer can allocate a fresh one. Producer-local shaped extras are
/// forwarded; scalar extras are cloned into the consumer (CreatePreload
/// cannot rematerialize index / i1 as scope results).
static bool collectExclusiveReturnedCluster(Value root, scope::ScopeOp producer,
                                            SmallVector<Operation *> &clusterOps,
                                            SmallVector<Value> &destEmpties,
                                            SmallVector<Value> &extras) {
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

/// Move exclusive returned tensor / leftover-memref clusters into the
/// unique later VECTOR consumer, including MTE2 load through a subview.
/// Shared tensor.empty stays in the producer; the consumer gets a fresh
/// empty. Scalar extras are cloned (not returned). Runs to a fixpoint so
/// a forwarded leftover can be sunk on the next iteration.
static LogicalResult sinkReturnedTensorsToConsumerImpl(scf::ForOp forOp) {
  for (int iter = 0; iter < 32; ++iter) {
    SmallVector<scope::ScopeOp> scopes;
    for (Operation &op : *forOp.getBody()) {
      if (auto scope = dyn_cast<scope::ScopeOp>(&op))
        scopes.push_back(scope);
    }

    bool changed = false;
    for (scope::ScopeOp producer : scopes) {
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
        auto consumer = uniqueLaterUserScope(scopeRes, producer, forOp);
        if (!consumer)
          continue;

        SmallVector<Operation *> cluster;
        SmallVector<Value> destEmpties;
        SmallVector<Value> extras;
        if (!collectExclusiveReturnedCluster(retVal, producer, cluster,
                                             destEmpties, extras))
          continue;
        // VECTOR ops stay out of CUBE scopes. Copy-chain sink places
        // load→VF→copy next to a later mmad; a returned tensor whose
        // only user is a cube is left here.
        if (scopeBlocksCopySink(consumer))
          continue;
        if (cluster.empty())
          continue;
        if (llvm::any_of(cluster, [&](Operation *op) {
              return claimed.contains(op);
            }))
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
                dyn.push_back(
                    remapThroughReturns(sz, newReturns, newProducer));
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
  auto newScope =
      builder.create<scope::ScopeOp>(cube.getLoc(), TypeRange{});
  newScope.setNoInline(true);
  newScope->setAttr(hivm::kPipelinedLoopCoreTypeAttrName,
                    hivm::TCoreTypeAttr::get(builder.getContext(),
                                             hivm::TCoreType::VECTOR));
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

static bool addMemrefViewTreeToCopyCluster(Value memref, scope::ScopeOp producer,
                                           DenseSet<Operation *> &cluster,
                                           SmallVector<Operation *> &order) {
  return addMemrefViewTree(memref, producer, cluster, order, [&](Operation *op) {
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
    if (!toTensor->getBlock() ||
        toTensor->getBlock() != destScope->getBlock())
      continue;
    if (!toTensor->isBeforeInBlock(destScope))
      continue;
    early.push_back(toTensor);
  }
  llvm::sort(early, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  for (Operation *op : llvm::reverse(early))
    op->moveAfter(destScope);
}

/// Clone a producer-local scalar (index / i1 / arith) and its scalar deps
/// into `destScope`. CreatePreload cannot rematerialize those types as
/// scope results (`Unhandled scope result case`), so they must not be
/// forwarded through `scope.return`.
static Value cloneProducerScalar(Value v, scope::ScopeOp producer,
                                 Operation *insertBefore,
                                 DenseMap<Value, Value> &cloned) {
  if (auto it = cloned.find(v); it != cloned.end())
    return it->second;
  if (isa<ShapedType, MemRefType>(v.getType()))
    return v;
  Operation *def = v.getDefiningOp();
  if (!def || !producer->isProperAncestor(def))
    return v;
  IRMapping mapping;
  for (Value operand : def->getOperands()) {
    Value mapped =
        cloneProducerScalar(operand, producer, insertBefore, cloned);
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
              dyn.push_back(
                  remapThroughReturns(sz, newReturns, newProducer));
          }
          operand.set(builder.create<tensor::EmptyOp>(
              oldEmpty.getLoc(), oldEmpty.getType(), dyn));
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
  SmallVector<Operation *> work(alloc.getUsers().begin(), alloc.getUsers().end());
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
static void sinkSingleSlotPreloadAllocs(scf::ForOp forOp) {
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
  sinkSingleSlotPreloadAllocs(forOp);
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

void SinkExclusivePreloadWorkPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  if (isSplitMixModule(moduleOp))
    return;
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (scf::ForOp forOp : forOps) {
      if (failed(sinkReturnedTensorsToConsumerImpl(forOp)))
        return signalPassFailure();
      if (failed(sinkCopyChainPastUnusedMmad(forOp)))
        return signalPassFailure();
      if (failed(sinkPreloadLocalAllocs(forOp)))
        return signalPassFailure();
    }
  }
}

} // namespace

LogicalResult mlir::hivm::sinkReturnedTensorsToConsumer(scf::ForOp forOp) {
  return sinkReturnedTensorsToConsumerImpl(forOp);
}

std::unique_ptr<Pass> mlir::hivm::createSinkExclusivePreloadWorkPass() {
  return std::make_unique<SinkExclusivePreloadWorkPass>();
}

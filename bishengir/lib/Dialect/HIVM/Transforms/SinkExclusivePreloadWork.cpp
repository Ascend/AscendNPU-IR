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
// Post-CVPipelining sink on the unsplit MIX function.
// `multi_buffer=1` + `preload_local_buffer=1` allocs whose only non-mark
// (and non-view) users sit in one scope become ordinary scope-private
// buffers. After SplitMixKernel this pass is a no-op.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
    for (scf::ForOp forOp : forOps)
      sinkSingleSlotPreloadAllocs(forOp);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createSinkExclusivePreloadWorkPass() {
  return std::make_unique<SinkExclusivePreloadWorkPass>();
}

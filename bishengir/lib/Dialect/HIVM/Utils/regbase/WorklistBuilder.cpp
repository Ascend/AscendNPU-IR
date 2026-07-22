//===- WorklistBuilder.cpp - Build worklists for HIVM core types ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/regbase/WorklistBuilder.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

static constexpr char DEBUG_TYPE[] = "cv-pipelining";

using llvm::dbgs;

namespace mlir {
using namespace hivm;
using hivm::detail::queryCoreTypeHelper;

static constexpr llvm::StringLiteral CubeOnlyAttrName = "pipeline.cubeonly";
static constexpr llvm::StringLiteral VecOnlyAttrName = "pipeline.veconly";

// Per-tensor compile hint. Attached to a bufferization::ToTensorOp result via
// `annotation.mark %t {cv_pipeline_lazy_load = true}` (or `false`).
//   * `true`  : opts this tensor into the lazy-load path even when the
//               kernel-level enable-lazy-loading switch is off.
//   * `false` : opts this tensor out of the lazy-load path even when the
//               kernel-level switch is on; a warning is emitted to flag the
//               override.
static constexpr llvm::StringLiteral CVPipelineLazyLoadAttrName =
    "cv_pipeline_lazy_load";

//===----------------------------------------------------------------------===//
// Static helpers (mirroring CVPipelining.cpp's free helpers)
//===----------------------------------------------------------------------===//

static int getMultibufferCount(annotation::MarkOp marker) {
  auto attrDict = marker->getAttrDictionary();
  hivm::util::validateMultiBufferAttr(attrDict);
  auto multibufferAttr = llvm::dyn_cast_if_present<IntegerAttr>(
      marker->getAttr(MultiBufferAttr::name));
  if (!multibufferAttr)
    return -1;
  return multibufferAttr.getInt();
}

static Value traceValueDef(Value v) {
  if (!v)
    return nullptr;
  if (auto result = dyn_cast<OpResult>(v)) {
    Operation *defining = result.getOwner();
    Value srcVal =
        TypeSwitch<Operation *, Value>(defining)
            .Case<tensor::ReshapeOp, tensor::ExtractSliceOp,
                  tensor::CollapseShapeOp, tensor::ExpandShapeOp,
                  bufferization::ToTensorOp, bufferization::ToMemrefOp,
                  memref::CastOp, memref::CollapseShapeOp,
                  memref::ExpandShapeOp, memref::MemorySpaceCastOp,
                  memref::ReinterpretCastOp, memref::ReshapeOp, memref::ViewOp,
                  memref::SubViewOp>([](auto op) { return op->getOperand(0); })
            .Case([](tensor::InsertSliceOp insert) { return insert.getDest(); })
            .Case([result](LoopLikeOpInterface loop) {
              return loop.getTiedLoopInit(result)->get();
            })
            .Default([](Operation *op) { return nullptr; });
    if (srcVal)
      return traceValueDef(srcVal);
    return result;
  }

  // In case of Block Argument
  auto blkArg = dyn_cast<BlockArgument>(v);
  if (!blkArg) {
    LLVM_DEBUG(dbgs() << "[traceValueDef] expected block argument, got: " << v
                      << '\n');
    return nullptr;
  }
  Operation *parent = blkArg.getOwner()->getParentOp();
  auto loop = dyn_cast<LoopLikeOpInterface>(parent);
  if (!loop)
    return blkArg;
  return traceValueDef(loop.getTiedLoopInit(blkArg)->get());
}

static memref::AllocOp traceAlloc(Value v) {
  Value maybeAlloc = traceValueDef(v);
  return dyn_cast_if_present<memref::AllocOp>(maybeAlloc.getDefiningOp());
}

static bool isCrossCoreCopy(Operation *copy) {
  auto copyOp = dyn_cast<CopyOp>(copy);
  if (!copyOp)
    return false;
  Value dst = copyOp.getDst();
  memref::AllocOp alloc = traceAlloc(dst);
  if (!alloc)
    return false;
  auto memSpaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(alloc.getType().getMemorySpace());
  if (!memSpaceAttr)
    return false;

  return memSpaceAttr.getAddressSpace() == AddressSpace::L1;
}

/// True if `op` is a CV-pipelining "separator" — a store-like op that forms a
/// boundary between vector and cube workitems. Splitting the pipeline loop on
/// these ops yields the per-core workitems that CVPipelining schedules.
/// Covers `hivm.hir.fixpipe`, `hivm.hir.store`, and the cross-core variant of
/// `hivm.hir.copy`.
static bool isSeparator(Operation *op) {
  return isa<FixpipeOp, StoreOp>(op) || isCrossCoreCopy(op);
}

/// Check to see if op is what we consider a "core op" that is only available on
/// either a cube or vector core
static bool isCoreOp(Operation &op) {
  return op.hasAttr(CubeOnlyAttrName) || op.hasAttr(VecOnlyAttrName) ||
         (isa_and_nonnull<HIVMDialect>(op.getDialect()) &&
          isa<DestinationStyleOpInterface>(op));
}

/// True if `op` is a HIVM DMA op whose alloc-backed memref destination may have
/// a `bufferization.to_tensor` reader that crosses CV-pipelining workitem
/// boundaries — i.e. we need `traceMemrefSubnet` to register it in
/// `outputMemrefMap` so `migrateOps` can find the writer for any
/// cross-workitem `localOutput`. Tensor-form DMA ops are ordinary value
/// producers for CVPipelining and must not enter the memref subnet tracer.
///
/// Includes memref-dst `hivm.hir.load`, `hivm.hir.fixpipe`, the cross-core
/// variant of `hivm.hir.copy`, and memref-dst `hivm.hir.nd2nz` (the fused
/// GM->L1 load with NZ layout conversion that `CombineOptimizedConvertLayout`
/// emits in place of a LoadOp under `--enable-layout-optimization=true`).
static bool isMemrefSubnetWriter(Operation *op) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dps || dps.getNumDpsInits() != 1 ||
      !isa<MemRefType>(dps.getDpsInitOperand(0)->get().getType()))
    return isCrossCoreCopy(op);
  return isa<LoadOp, FixpipeOp, ND2NZOp>(op) || isCrossCoreCopy(op);
}

/// True if `op` is a HIVM "load-like" DMA op that pulls data from GM into
/// UB/L1. Such ops are not workitem seeds (extractAvailableOps skips them);
/// instead they are pulled into each consumer workitem during dependency
/// tracing. Under `enableLazyLoading`, they may be cloned into multiple
/// consumer workitems so each stage loads independently from GM.
///
/// Currently covers `hivm.hir.load` and `hivm.hir.nd2nz` — both read GM and
/// write to a UB/L1 alloc that the consumer mmadL1/vector op will read via
/// `bufferization.to_tensor`.
static bool isLoadLikeOp(Operation *op) { return isa<LoadOp, ND2NZOp>(op); }

/// Recognize an indirect-gather loop expressed entirely in non-HIVM ops:
///
///   scf.for ... iter_args(%t = %init) -> (tensor<...>) {
///     ...
///     %addr = memref.reinterpret_cast %gm_ptr ...
///     %v    = memref.load %addr[...]
///     %new  = tensor.insert %v into %t[...]
///     scf.yield %new
///   } {ExtractedLoadOrStore}
///
/// Why a fallback is needed: `illegalRegionedOp` classifies a region op by
/// walking its body for HIVM ops. The loop above has none — only memref
/// load/cast/tensor.insert/arith — so the walk leaves it unannotated.
/// `extractAvailableOps` then skips unannotated non-`isCoreOp` ops as
/// workitem seeds, and the loop only enters a workitem reactively through
/// consumer tracing — i.e. wherever the loop's tensor result is eventually
/// used. For a gather whose output feeds a post-matmul vector chain, that
/// puts the gather in a Vector workitem *after* the cube even though its
/// data dependencies allow it to run alongside the other pre-cube gathers,
/// breaking the intended schedule.
///
/// What this predicate looks for: the `ExtractedLoadOrStore` attribute that
/// the producer set on these gather loops, plus a body that has both
///   * a `memref.load` (the scalar gather read), and
///   * a writeback into the iter_arg
///     (`tensor.insert` / `tensor.insert_slice`, with `memref.store` and
///     `memref.copy` also accepted for the post-bufferization form used by
///     the unit test).
/// The caller has already established that the body contains no HIVM op.
///
/// Note on context: CVPipelining runs in the pre-bufferization HIVM
/// pipeline, so we cannot rely on the kernel-argument memref's
/// `#hivm.address_space<gm>` attribute (added later by
/// `createInferHIVMMemScopePass`) being present; we also see the
/// `tensor.insert` form of the writeback rather than the post-bufferization
/// `memref.store`. The 32x128 rectangular gathers go through `memref.copy`
/// which earlier CV-communication passes rewrite to `hivm.hir.load` — those
/// scf.fors get tagged via the regular HIVMDialect/queryCoreTypeHelper
/// path, so only the scalar (single-element) form needs this fallback.
static bool isExtractedScalarGather(scf::ForOp forOp) {
  if (!forOp->hasAttr("ExtractedLoadOrStore"))
    return false;
  bool hasMemRefLoad = false;
  bool hasTensorWrite = false;
  forOp.getBody()->walk([&](Operation *inner) {
    if (isa<memref::LoadOp>(inner))
      hasMemRefLoad = true;
    else if (isa<tensor::InsertOp, tensor::InsertSliceOp, memref::StoreOp,
                 memref::CopyOp>(inner))
      hasTensorWrite = true;
  });
  return hasMemRefLoad && hasTensorWrite;
}

/// Inspect a region op's body for core-type content. Sets pipeline.cubeonly /
/// pipeline.veconly if the region is uniformly one core type. For mixed-core
/// regions:
///   - Loop mode (CV pipelining): returns true to reject the loop.
///   - Block mode (split-if): returns false; leaves the op unannotated so the
///     split-if pattern can recurse on mixed inner ifs without bailing out.
static bool illegalRegionedOp(Operation &op, bool isLoopMode) {
  if (op.getRegions().empty())
    return false;
  bool hasCube = false;
  bool hasVector = false;
  // The walk callback signature is fixed by MLIR (Operation*); we keep it as
  // a pointer and immediately work via member-access on the pointer below.
  WalkResult result = op.walk([&hasCube, &hasVector](Operation *curOp) {
    if (!isa_and_nonnull<HIVMDialect>(curOp->getDialect()))
      return WalkResult::advance();
    auto core = queryCoreTypeHelper(curOp).value_or(TCoreType::CUBE_OR_VECTOR);
    if (core == TCoreType::CUBE_OR_VECTOR && isCrossCoreCopy(curOp))
      core = TCoreType::VECTOR;
    if (core == TCoreType::VECTOR) {
      if (hasCube)
        return WalkResult::interrupt();
      hasVector = true;
    } else if (core == TCoreType::CUBE) {
      if (hasVector)
        return WalkResult::interrupt();
      hasCube = true;
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    if (isLoopMode) {
      op.emitWarning("[cv-pipelining] unsupported regioned op");
      return true;
    }
    // Block mode: leave the op opaque; don't annotate, don't fail.
    return false;
  }

  // Body had no HIVM op so neither flag fired. Catch the non-HIVM
  // scalar-gather shape (see isExtractedScalarGather) and classify it as
  // Vector so `extractAvailableOps` treats it as a seed.
  if (!hasCube && !hasVector) {
    if (auto forOp = dyn_cast<scf::ForOp>(&op);
        forOp && isExtractedScalarGather(forOp))
      hasVector = true;
  }

  auto unit = UnitAttr::get(op.getContext());
  if (hasCube)
    op.setAttr(CubeOnlyAttrName, unit);
  else if (hasVector)
    op.setAttr(VecOnlyAttrName, unit);
  return false;
}

/// Get the highest level parent op that is not the containing op
static Operation *getContainedParent(Operation *containing, Operation &inner) {
  Operation *cur = &inner;
  Operation *parent = cur->getParentOp();
  while (parent && parent != containing && containing->isAncestor(cur)) {
    cur = parent;
    parent = cur->getParentOp();
  }
  return cur;
}

static Operation *getContainedParent(Operation *containing, Value inner) {
  Operation *defining = inner.getDefiningOp();
  if (defining)
    return getContainedParent(containing, *defining);
  return nullptr;
}

// Given memref value, populate users with all operations that uses any aliasing
// memrefs as `memrefVal`
static void memrefDFS(Value memrefVal, SmallVector<Operation *> &users) {
  SmallVector<Operation *> traceStack;
  DenseSet<Operation *> visited;
  Value rootVal = traceValueDef(memrefVal);
  if (!rootVal)
    return;
  traceStack.append(rootVal.user_begin(), rootVal.user_end());
  while (!traceStack.empty()) {
    Operation *op = traceStack.pop_back_val();
    if (visited.contains(op))
      continue;
    visited.insert(op);

    // Since the same gm memref could be loaded multiple times, avoid pulling
    // the load into current work item if its loaded value is not being used
    if (isLoadLikeOp(op)) {
      auto dpsOp = cast<DestinationStyleOpInterface>(op);
      Value dst = dpsOp.getDpsInitOperand(0)->get();
      if (traceValueDef(dst) != rootVal)
        continue;
    }

    users.push_back(op);

    // If not memref result, dont need to trace any more
    if (op->getNumResults() == 1 &&
        !isa<MemRefType>(op->getResult(0).getType()))
      continue;
    traceStack.append(op->user_begin(), op->user_end());
  }
}

//===----------------------------------------------------------------------===//
// WorklistBuilder implementation
//===----------------------------------------------------------------------===//

WorklistBuilder::WorklistBuilder(scf::ForOp loop, int numMultibuffer,
                                 bool enableLazyLoading)
    : targetBlock(loop.getBody()), scopeOp(loop.getOperation()),
      pipelineLoop(loop), isLoopMode(true), numMultibuffer(numMultibuffer),
      enableLazyLoading(enableLazyLoading),
      yieldedVals(loop.getYieldedValues().begin(),
                  loop.getYieldedValues().end()) {}

WorklistBuilder::WorklistBuilder(Block *block)
    : targetBlock(block), scopeOp(block->getParentOp()), isLoopMode(false) {}

void WorklistBuilder::mapOpToItem(Operation &op, WorkItem &item) {
  if (item.ops.contains(&op))
    return;
  if (opToWorkItemMap.contains(&op))
    opToWorkItemMap[&op].push_back(&item);
  else
    opToWorkItemMap[&op] = {&item};
  item.ops.insert(&op);
}

LazyLoadHint WorklistBuilder::getLazyLoadHint(Value v) {
  auto maybeMark = utils::getAnnotateOpWithAttr(v, CVPipelineLazyLoadAttrName);
  if (!maybeMark)
    return LazyLoadHint::None;
  auto mark = cast<annotation::MarkOp>(*maybeMark);
  auto attr = mark->getAttrOfType<BoolAttr>(CVPipelineLazyLoadAttrName);
  if (!attr)
    return LazyLoadHint::None;
  return attr.getValue() ? LazyLoadHint::Enable : LazyLoadHint::Disable;
}

void WorklistBuilder::warnHintOverride(Value v) {
  auto maybeMark = utils::getAnnotateOpWithAttr(v, CVPipelineLazyLoadAttrName);
  if (!maybeMark)
    return;
  auto mark = cast<annotation::MarkOp>(*maybeMark);
  if (!warnedOverrideMarks.insert(mark).second)
    return;

  // Prefer printing on the LoadOp for source-location clarity: it's the op
  // that would have been cloned across stages had the hint not vetoed it.
  Operation *warnTarget = mark.getOperation();
  if (auto tt = dyn_cast_or_null<bufferization::ToTensorOp>(v.getDefiningOp()))
    if (auto it = outputMemrefMap.find(tt); it != outputMemrefMap.end())
      if (auto load = dyn_cast<LoadOp>(it->second.getOperation()))
        warnTarget = load.getOperation();

  auto diag = warnTarget->emitWarning()
              << "[cv-pipelining] " << CVPipelineLazyLoadAttrName
              << "=false overrides kernel-level enable-lazy-loading=true; "
                 "lazy loading is disabled for this tensor";
  if (warnTarget != mark)
    diag.attachNote(mark->getLoc())
        << "see `" << CVPipelineLazyLoadAttrName << " = false` hint here";
}

void WorklistBuilder::warnHintIgnoredForCrossCore(Value v) {
  auto maybeMark = utils::getAnnotateOpWithAttr(v, CVPipelineLazyLoadAttrName);
  if (!maybeMark)
    return;
  auto mark = cast<annotation::MarkOp>(*maybeMark);
  if (!warnedOverrideMarks.insert(mark).second)
    return;

  Operation *warnTarget = mark;
  if (auto tt = dyn_cast_or_null<bufferization::ToTensorOp>(v.getDefiningOp()))
    if (auto it = outputMemrefMap.find(tt); it != outputMemrefMap.end())
      if (auto load = dyn_cast<LoadOp>(it->second.getOperation()))
        warnTarget = load;

  auto diag = warnTarget->emitWarning()
              << "[cv-pipelining] " << CVPipelineLazyLoadAttrName
              << "=false is ignored: load result is consumed by both CUBE and "
                 "VECTOR cores, lazy loading is required for correctness; "
                 "disable cv-pipelining entirely to opt out";
  if (warnTarget != mark)
    diag.attachNote(mark->getLoc())
        << "see `" << CVPipelineLazyLoadAttrName << " = false` hint here";
}

void WorklistBuilder::diagnoseLazyLoadHints() {
  // Walk targetBlock to collect, in IR order, the distinct values that carry
  // at least one `cv_pipeline_lazy_load` mark.  We then query all marks per
  // value via `utils::getAllAnnotateOpsWithAttr` to keep grouping bookkeeping
  // out of this function.
  llvm::SetVector<Value> markedSrcs;
  targetBlock->walk([&markedSrcs](annotation::MarkOp mark) {
    if (mark.isAnnotatedBy(CVPipelineLazyLoadAttrName))
      markedSrcs.insert(mark.getSrc());
  });

  for (Value src : markedSrcs) {
    SmallVector<Operation *> marks =
        utils::getAllAnnotateOpsWithAttr(src, CVPipelineLazyLoadAttrName);
    if (marks.empty())
      continue;
    auto probe = cast<annotation::MarkOp>(marks.front());

    // (1) Duplicate `cv_pipeline_lazy_load` marks on the same value.
    if (marks.size() > 1) {
      auto diag = probe->emitWarning()
                  << "[cv-pipelining] tensor carries " << marks.size() << " `"
                  << CVPipelineLazyLoadAttrName
                  << "` annotation.mark ops; only the first one will be "
                     "honored";
      for (size_t i = 1; i < marks.size(); ++i)
        diag.attachNote(marks[i]->getLoc())
            << "duplicate `" << CVPipelineLazyLoadAttrName << "` mark here";
    }

    // (2) Target must be a bufferization::ToTensorOp result.
    auto tt = dyn_cast_or_null<bufferization::ToTensorOp>(src.getDefiningOp());
    if (!tt) {
      probe->emitWarning()
          << "[cv-pipelining] `" << CVPipelineLazyLoadAttrName
          << "` hint is ignored: marked value is not produced by "
             "`bufferization.to_tensor`";
      continue;
    }

    // (3) The to_tensor must be backed by a `hivm.hir.load`.
    auto it = outputMemrefMap.find(tt);
    if (it == outputMemrefMap.end() ||
        !isa<LoadOp>(it->second.getOperation())) {
      probe->emitWarning()
          << "[cv-pipelining] `" << CVPipelineLazyLoadAttrName
          << "` hint is ignored: tensor is not backed by `hivm.hir.load`";
      continue;
    }
  }
}

FailureOr<bool> WorklistBuilder::isCrossCoreLoad(const Value loaded) const {
  if (!loaded)
    return false;
  bool hasCube = false;
  bool hasVec = false;
  SmallVector<Value> stack;
  DenseSet<Operation *> visited;
  stack.push_back(loaded);
  while (!stack.empty()) {
    Value v = stack.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      // Descend through view-like / passthrough ops that just rename the
      // loaded tensor; the real consumer (and thus the real core type)
      // is downstream.
      if (isa<tensor::ExtractSliceOp, tensor::CastOp, tensor::ExpandShapeOp,
              tensor::CollapseShapeOp, tensor::ReshapeOp>(user)) {
        for (Value r : user->getResults())
          stack.push_back(r);
        continue;
      }
      if (isa<annotation::MarkOp, DebugOp>(user))
        continue;
      // Per-op core hints set by `illegalRegionedOp` take precedence over
      // the trait-based query.
      std::optional<TCoreType> core;
      if (user->hasAttr(CubeOnlyAttrName))
        core = TCoreType::CUBE;
      else if (user->hasAttr(VecOnlyAttrName))
        core = TCoreType::VECTOR;
      else
        core = queryCoreTypeHelper(user);
      if (!core)
        continue;
      switch (*core) {
      case TCoreType::CUBE:
        hasCube = true;
        break;
      case TCoreType::VECTOR:
        hasVec = true;
        break;
      case TCoreType::CUBE_AND_VECTOR:
        // Op runs on both cores; that alone makes the load cross-core.
        hasCube = true;
        hasVec = true;
        break;
      case TCoreType::CUBE_OR_VECTOR:
        // Ambiguous — either core could end up running this op.  We
        // cannot safely classify the load; signal failure so the caller
        // bails out of pipelining for this loop (lazy load is a
        // legality requirement when cross-core, and we can't prove the
        // load isn't cross-core here).
        return failure();
      }
      if (hasCube && hasVec)
        return true;
    }
  }
  return hasCube && hasVec;
}

FailureOr<bool> WorklistBuilder::shouldLazyLoadFor(Operation *op) {
  // Find the candidate to_tensor whose hint we should consult, with
  // shape-specific fallbacks when no candidate exists.
  bufferization::ToTensorOp tt;
  if (auto asTT = dyn_cast<bufferization::ToTensorOp>(op)) {
    auto it = outputMemrefMap.find(asTT);
    if (it == outputMemrefMap.end())
      return false;
    if (!isa<LoadOp>(it->second.getOperation()))
      return false;
    tt = asTT;
  } else if (auto load = dyn_cast<LoadOp>(op)) {
    for (auto &kv : outputMemrefMap) {
      if (kv.second.getOperation() == load.getOperation()) {
        tt = kv.first;
        break;
      }
    }
    if (!tt) {
      // No backing to_tensor; the load is tensor-form or otherwise opaque
      // to outputMemrefMap.  Still auto-enable lazy load when the load's
      // own tensor result has consumers on both cores.
      if (load->getNumResults() > 0 &&
          isa<TensorType>(load->getResult(0).getType())) {
        FailureOr<bool> crossCore = isCrossCoreLoad(load->getResult(0));
        if (failed(crossCore))
          return failure();
        if (*crossCore)
          return true;
      }
      return enableLazyLoading;
    }
  } else {
    return false;
  }

  FailureOr<bool> isCrossCore = isCrossCoreLoad(tt.getResult());
  if (failed(isCrossCore))
    return failure();

  switch (getLazyLoadHint(tt.getResult())) {
  case LazyLoadHint::Enable:
    return true;
  case LazyLoadHint::Disable:
    if (*isCrossCore) {
      // Legality: cross-core consumption forces lazy load; the explicit
      // `false` hint cannot veto it without producing incorrect results.
      warnHintIgnoredForCrossCore(tt.getResult());
      return true;
    }
    if (enableLazyLoading)
      warnHintOverride(tt.getResult());
    return false;
  case LazyLoadHint::None:
    return enableLazyLoading || *isCrossCore;
  }
  return tt.emitWarning("invalid LazyLoadHint enumerator");
}

/// DFS to find all ops that are dependent on separator
LogicalResult WorklistBuilder::populateDependencies(Operation &separator) {
  SmallVector<Operation *> dfsStack = {&separator};
  DenseSet<Operation *> visited;

  while (!dfsStack.empty()) {
    Operation *op = dfsStack.pop_back_val();
    if (visited.contains(op) || !scopeOp->isAncestor(op))
      continue;
    visited.insert(op);
    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
      Value init = dpsOp.getDpsInitOperand(0)->get();
      if (isa<MemRefType>(init.getType())) {
        Value src = traceValueDef(init);
        auto blkArg = dyn_cast<BlockArgument>(src);
        if (!blkArg) {
          op = src.getDefiningOp();
          if (!isa<memref::AllocOp>(op))
            return failure();
        } else if (!isa<func::FuncOp>(blkArg.getOwner()->getParentOp())) {
          return failure();
        }
      }
    }
    for (Value result : op->getResults()) {
      if (!isa<ShapedType>(result.getType()))
        continue;

      for (OpOperand &use : result.getUses()) {
        Operation *usr = use.getOwner();

        Operation *scopedUsr = getContainedParent(scopeOp, *usr);
        if (isa<scf::YieldOp, scf::ConditionOp>(scopedUsr) ||
            scopedUsr == &separator)
          continue;
        dfsStack.push_back(scopedUsr);
        if (dependenceMap.contains(scopedUsr))
          dependenceMap[scopedUsr].insert(&separator);
        else
          dependenceMap[scopedUsr] = DenseSet<Operation *>({&separator});
      }
    }
  }
  return success();
}

/// Populate dependencies that are carried between loop iterations (iter args,
/// yield operands). Only meaningful in loop mode; no-op in block mode.
void WorklistBuilder::populateLoopCarriedDependencies() {
  auto maybeYield = pipelineLoop.getYieldedValuesMutable();
  if (!maybeYield.has_value())
    return;
  for (OpOperand &yieldOperand : *maybeYield) {
    Value yieldVal = yieldOperand.get();
    if (!isa<TensorType>(yieldVal.getType()))
      continue;
    Operation *defining = yieldVal.getDefiningOp();
    if (!defining || !scopeOp->isAncestor(defining))
      continue;
    BlockArgument iterArg =
        pipelineLoop.getRegionIterArgs()[yieldOperand.getOperandNumber()];
    SmallVector<Operation *> dfsStack(iterArg.getUsers());
    DenseSet<Operation *> visited;
    while (!dfsStack.empty()) {
      Operation *op = getContainedParent(scopeOp, *dfsStack.pop_back_val());
      if (visited.contains(op) || op == defining)
        continue;
      visited.insert(op);
      if (isa<DestinationStyleOpInterface>(op)) {
        if (loopCarriedDependenceMap.contains(op))
          loopCarriedDependenceMap[op].insert(defining);
        else
          loopCarriedDependenceMap[op] = {defining};
        continue;
      }
      for (Operation *usr : op->getUsers()) {
        if (isa<scf::YieldOp>(usr))
          continue;
        dfsStack.push_back(usr);
      }
    }
  }
  LLVM_DEBUG({
    for (auto &[val, set] : loopCarriedDependenceMap) {
      dbgs() << *val << " depends on:\n";
      for (auto *op : set) {
        dbgs() << "\t";
        op->dump();
      }
    }
  });
}

/// Helper to trace the alloc (if within scope), toTensor, and various casts.
LogicalResult
WorklistBuilder::traceMemrefSubnet(Operation &start,
                                   SmallVector<Operation *> &workingStack) {
  // When we get here, `start` should be one of three ops:
  // 1. Fixpipe
  // 2. Copy
  // 3. Load
  // All of which have the `outs` as second operand
  DestinationStyleOpInterface writer = nullptr;
  Value targetOperand = start.getOperand(1);
  if (isa<TensorType>(targetOperand.getType()))
    writer = cast<DestinationStyleOpInterface>(&start);

  // Remember the original separator so we don't re-queue it onto
  // workingStack — otherwise it would be popped again and re-enter
  // traceMemrefSubnet in an infinite loop (e.g. when a Fixpipe writes
  // directly to a func-arg memref and the upward trace yields no alloc).
  Operation *separatorOp = &start;
  Operation *traceStart = &start;
  Operation *defining = targetOperand.getDefiningOp();
  // First trace all the way up
  while (defining) {
    if (!scopeOp->isAncestor(defining))
      break;
    traceStart = defining;
    if (isa<memref::AllocOp, bishengir::memref_ext::AllocWorkspaceOp>(defining))
      break;
    if (isa<memref::CastOp, memref::ReinterpretCastOp,
            memref::MemorySpaceCastOp, memref::CollapseShapeOp,
            memref::ExpandShapeOp, memref::SubViewOp, memref::ViewOp,
            bufferization::ToTensorOp, tensor::ExtractSliceOp>(defining))
      defining = defining->getOperand(0).getDefiningOp();
    else
      return defining->emitWarning(
          "[cv-pipelining] unexpected memref op in chain");
  }
  SmallVector<Operation *> userTraceStack = {traceStart};
  bufferization::ToTensorOp toTensor = nullptr;

  while (!userTraceStack.empty()) {
    Operation *def = userTraceStack.pop_back_val();
    if (def != separatorOp)
      workingStack.push_back(def);
    for (OpOperand &use : def->getUses()) {
      Operation *usr = use.getOwner();
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(usr)) {
        // Only count dps as a writer if this use is its init operand.
        // Reads via `ins` (e.g. hivm.hir.store's src) do not constitute
        // a write to the traced memref.
        OpOperand *init = dps.getDpsInitOperand(0);
        if (!init || init != &use)
          continue;
        if (writer)
          return usr->emitWarning("[cv-pipelining] expecting only one op "
                                  "writing to a defined memref");
        writer = dps;
        continue;
      }
      if (auto tt = dyn_cast<bufferization::ToTensorOp>(usr)) {
        if (toTensor)
          return usr->emitWarning(
              "[cv-pipelining] expecting only one toTensor for a "
              "defined memref");
        toTensor = tt;
        workingStack.push_back(usr);
        continue;
      }
      if (isa<memref::CastOp, memref::ReinterpretCastOp,
              memref::MemorySpaceCastOp, memref::CollapseShapeOp,
              memref::ExpandShapeOp, memref::SubViewOp, memref::ViewOp>(usr)) {
        userTraceStack.push_back(usr);
      }
    }
  }
  if (toTensor && !writer) {
    LLVM_DEBUG(dbgs() << "toTensor: "; toTensor->dump());
    return toTensor->emitWarning(
        "[cv-pipelining] expecting toTensor to have dps op to write to it");
  }
  if (toTensor && writer)
    outputMemrefMap[toTensor] = writer;
  return success();
}

LogicalResult
WorklistBuilder::traceOperands(Value operand, WorkItem &item,
                               SmallVector<Operation *> &workingStack) const {
  if (!operand)
    return success();
  Operation *defining = getContainedParent(scopeOp, operand);
  if (item.ops.contains(defining))
    return success();
  if (!defining) {
    auto iterArg = dyn_cast<BlockArgument>(operand);
    if (!iterArg)
      return scopeOp->emitWarning(
          "[cv-pipelining] expected non-op-defined value to be block "
          "argument");
    for (Operation *usr : iterArg.getUsers()) {
      if (isa<DebugOp>(usr) && !item.ops.contains(usr) &&
          usr->getParentOp() == scopeOp)
        workingStack.push_back(usr);
    }
    if (iterArg.getOwner()->getParentOp() != scopeOp ||
        iterArg.getArgNumber() == 0)
      return success();
    // Need to pull defining op into this work item to guarantee safety,
    // should already be guaranteed by extractAvailableOps
    if (isa<TensorType>(operand.getType()))
      return success();
    // Iter-arg → yielded-value hop only meaningful when the scope is a
    // LoopLikeOpInterface. In block mode (scope is scf::IfOp), bail out.
    auto loopLike = dyn_cast<LoopLikeOpInterface>(scopeOp);
    if (!loopLike)
      return success();
    auto tiedYield = loopLike.getTiedLoopYieldedValue(iterArg);
    if (!tiedYield)
      return success();
    Value yieldVal = tiedYield->get();
    defining = yieldVal.getDefiningOp();
    if (!defining || defining->getParentOp() != scopeOp)
      return success();
  }
  if (defining->getParentOp() != scopeOp)
    return success();
  if (isa<MemRefType>(operand.getType()))
    memrefDFS(operand, workingStack);
  if (!item.ops.contains(defining))
    workingStack.push_back(defining);
  return success();
}

/// Trace producers of every operand of `op` *except* its DPS-init (writeback
/// destination) operands. Used by `traceDependentOps` for memref-subnet
/// writers (Load / Fixpipe / cross-core Copy / ND2NZ): the init is the `dst`
/// memref reached separately via `traceMemrefSubnet`, while every other
/// operand — ins, plus scalar params like LoadOp's init-condition and
/// padding values — is a real data dependency that must be queued onto
/// `workingStack`.
LogicalResult WorklistBuilder::traceNonInitOperands(
    Operation &op, WorkItem &item,
    SmallVector<Operation *> &workingStack) const {
  auto dps = dyn_cast<DestinationStyleOpInterface>(&op);
  for (Value operand : op.getOperands()) {
    if (dps && llvm::is_contained(dps.getDpsInits(), operand))
      continue;
    if (failed(traceOperands(operand, item, workingStack)))
      return failure();
  }
  return success();
}

/// Trace each op in the initial set of ops in each WorkItem, get non-HIVM ops
/// that are operands for each op
LogicalResult WorklistBuilder::traceDependentOps(WorkItem &item) {
  SmallVector<Operation *> workingStack(item.ops.begin(), item.ops.end());

  while (!workingStack.empty()) {
    Operation *op = workingStack.pop_back_val();
    // If op is nested inside a top-level op already part of this workitem
    // (e.g. a Fixpipe inside an scf.if separator), skip it — it will be
    // cloned along with its enclosing region by migrateOps. Its outputs are
    // tracked via outputMemrefMap populated elsewhere.
    {
      Operation *top = getContainedParent(scopeOp, *op);
      if (top != op && item.ops.contains(top))
        continue;
    }
    if (isCoreOp(*op)) {
      if (opToWorkItemMap.contains(op)) {
        // If Core Op is already inserted into a different work item, then we
        // don't include it here
        if (!item.ops.contains(op)) {
          // With lazy loading (kernel-level switch, per-tensor compile
          // hint, or auto cross-core legality), allow Load ops to be
          // cloned into multiple work items so each stage loads
          // independently from GM.
          if (!isLoadLikeOp(op))
            continue;
          FailureOr<bool> shouldLazy = shouldLazyLoadFor(op);
          if (failed(shouldLazy))
            return failure();
          if (!*shouldLazy)
            continue;
        }
      } else if (!isLoadLikeOp(op)) {
        // Separators (Store/Fixpipe/cross-core Copy) that reach here via a
        // shared memref alias chain have not been assigned to any workitem
        // yet — they will be picked up in a subsequent extractAvailableOps
        // round for the other core. Skip rather than fail.
        if (isSeparator(op))
          continue;
        // Load-like ops are pulled into their consuming work items; apart from
        // that, if we get here we depend on an op that has not satisfied its
        // dependency.
        return op->emitWarning(
            "[cv-pipelining] cannot pipeline op due to dependency");
      }
    }
    // Other than core ops, we can skip them if we already inserted them into
    // this work item
    else if (item.ops.contains(op))
      continue;
    // Determine whether a to_tensor should be skipped because it has already
    // been allocated to another work item.  The default guard fires for all
    // to_tensor ops, but with lazy loading we lift the restriction for
    // to_tensor ops whose backing writer is a LoadOp: those are cloned into
    // every consuming work item independently.  Lazy loading can be enabled
    // either by the kernel-level switch or by a per-tensor compile hint
    // (annotation.mark on the tensor result).
    bool skipToTensor =
        isa<bufferization::ToTensorOp>(op) && opToWorkItemMap.contains(op);
    if (skipToTensor) {
      FailureOr<bool> shouldLazy =
          shouldLazyLoadFor(cast<bufferization::ToTensorOp>(op));
      if (failed(shouldLazy))
        return failure();
      if (*shouldLazy)
        skipToTensor = false;
    }
    if (op->getParentOp() != scopeOp || isa<scf::YieldOp>(op) ||
        skipToTensor)
      continue;
    LLVM_DEBUG(dbgs() << "Inserting \t"; op->dump());
    mapOpToItem(*op, item);
    toBePipelined.erase(op);

    // A counter load in a VECTOR item must drag its advance-clone along so the
    // counter is incremented per stage instead of read-only at 0. Pulling the
    // clone here lets traceDependentOps wire its inits/operands and migrateOps
    // remap them, avoiding a dangling cube-path tensor when the loop is erased.
    if (item.core == TCoreType::VECTOR)
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        auto it = counterClones.find(load.getMemRef());
        if (it != counterClones.end() && !item.ops.contains(it->second))
          workingStack.push_back(it->second);
      }
    for (Operation *usr : op->getUsers())
      if (isa<annotation::MarkOp, DebugOp>(usr))
        mapOpToItem(*usr, item);

    // Handle load/fixpipe/copy dealing with memref memref
    if (isMemrefSubnetWriter(op)) {
      if (failed(traceMemrefSubnet(*op, workingStack)))
        return failure();
      if (failed(traceNonInitOperands(*op, item, workingStack)))
        return failure();
      continue;
    }

    // Handle nested ops as well
    WalkResult walkResult =
        op->walk([this, op, &item, &workingStack](Operation *nestedOp) {
          // For nested Load/Fixpipe/CrossCoreCopy (e.g. inside a lifted scf.if
          // separator), populate outputMemrefMap via traceMemrefSubnet and
          // trace their non-init operands so migrateOps/expandOutputInits can
          // resolve outputs back to their nested writer.
          if (nestedOp != op && isMemrefSubnetWriter(nestedOp)) {
            if (failed(traceMemrefSubnet(*nestedOp, workingStack)))
              return WalkResult::interrupt();
            if (failed(traceNonInitOperands(*nestedOp, item, workingStack)))
              return WalkResult::interrupt();
            return WalkResult::advance();
          }
          for (Value operand : nestedOp->getOperands())
            if (failed(traceOperands(operand, item, workingStack)))
              return WalkResult::interrupt();
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      return failure();
  }
  return success();
}

/// Fill each WorkItem with ops that will eventually go into their own jam loops
LogicalResult
WorklistBuilder::populateWorkItem(SmallVector<Operation *> &availableOps,
                                  TCoreType core) {
  auto item = std::make_shared<WorkItem>();
  item->core = core;

#ifndef NDEBUG
  static int id = 0;
  item->id = id++;
#endif

  for (Operation *op : availableOps)
    mapOpToItem(*op, *item);

  LLVM_DEBUG({
    dbgs() << "[populateWorkItem] Initial set{\n";
    for (Operation *op : item->ops) {
      dbgs() << '\t';
      op->dump();
    }
    dbgs() << "[populateWorkItem] } // Initial set\n";
  });

  if (traceDependentOps(*item).failed())
    return failure();
  worklist.push_back(item);
  return success();
}

/// Find ops that have no dependencies; can be executed if all other previously
/// extracted ops are done.
LogicalResult
WorklistBuilder::extractAvailableOps(SmallVector<Operation *> &extractedOps,
                                     TCoreType &core) {
  SetVector<Operation *> potentiallyAvailable;

  for (Operation &op : *targetBlock) {
    if (opToWorkItemMap.contains(&op))
      continue;
    TCoreType maybeCore = op.hasAttr(CubeOnlyAttrName) ? TCoreType::CUBE
                          : op.hasAttr(VecOnlyAttrName)
                              ? TCoreType::VECTOR
                              : TCoreType::CUBE_OR_VECTOR;
    if (maybeCore == hivm::TCoreType::CUBE_OR_VECTOR) {
      if (!isCoreOp(op) || isLoadLikeOp(&op))
        continue;
      maybeCore = queryCoreTypeHelper(&op).value_or(TCoreType::CUBE_OR_VECTOR);
      if (maybeCore != TCoreType::VECTOR && isCrossCoreCopy(&op))
        maybeCore = TCoreType::VECTOR;
      // Block-mode-only: CUBE_OR_VECTOR core ops (e.g. plain tensor-level
      // hivm.hir.copy) get pulled in as deps of consuming core ops via
      // traceDependentOps; silently skip here. Loop mode falls through to
      // the unexpected-core-type warning below — preserving cv-pipelining's
      // contract that any flexibly-typed core op left here is a bug.
      if (!isLoopMode && maybeCore == TCoreType::CUBE_OR_VECTOR)
        continue;
    }

    if (maybeCore != TCoreType::VECTOR && maybeCore != TCoreType::CUBE)
      return op.emitWarning("[cv-pipelining] unexpected core type for op");
    // Only gather ops of the same core type
    if (((maybeCore == TCoreType::VECTOR || isCrossCoreCopy(&op)) &&
         core == TCoreType::CUBE) ||
        ((maybeCore == TCoreType::CUBE && core == TCoreType::VECTOR)))
      continue;
    core = maybeCore;
    if (!dependenceMap.contains(&op) || dependenceMap[&op].empty())
      potentiallyAvailable.insert(&op);
  }

  DenseSet<Operation *> deferredOps;
  for (Operation *op : potentiallyAvailable) {
    if (!loopCarriedDependenceMap.contains(op))
      continue;
    if (llvm::all_of(loopCarriedDependenceMap[op], [&](Operation *dependantOp) {
          return potentiallyAvailable.contains(dependantOp);
        }))
      continue;
    deferredOps.insert(op);
  }

  // Propagate the loop-carried dependencies through the available ops
  SmallVector<Operation *> dfsStack;
  dfsStack.append(deferredOps.begin(), deferredOps.end());
  while (!dfsStack.empty()) {
    Operation *op = dfsStack.pop_back_val();
    if (deferredOps.contains(op))
      continue;
    if (potentiallyAvailable.contains(op))
      deferredOps.insert(op);
    for (Operation *usr : op->getUsers())
      dfsStack.push_back(usr);
  }

  // Coalesce same-core DPS-init chains: when an op's sole result feeds the
  // init operand of another same-core DPS core-op `usr` that is still
  // unavailable this round, defer the producer so it lands in the same
  // WorkItem as `usr`. Without this, the producer's result becomes a cross-
  // WorkItem localOutput, and expandOutputInits cannot expand the backing
  // buffer when the init is not a tensor.empty / to_tensor (e.g. an
  // accumulator chained from another mmad's result).
  DenseSet<Operation *> chainDeferred;
  auto findBlockedCoChainConsumer =
      [this, &core, &potentiallyAvailable, &deferredOps,
       &chainDeferred](Operation &op) -> Operation * {
    if (op.getNumResults() != 1)
      return nullptr;
    Value res = op.getResult(0);
    if (!res.hasOneUse())
      return nullptr;
    OpOperand &use = *res.getUses().begin();
    Operation *usr = use.getOwner();
    if (usr->getParentOp() != scopeOp)
      return nullptr;
    if (!isCoreOp(*usr))
      return nullptr;
    auto userDps = dyn_cast<DestinationStyleOpInterface>(usr);
    if (!userDps || userDps.getNumDpsInits() != 1)
      return nullptr;
    if (userDps.getDpsInitOperand(0) != &use)
      return nullptr;
    TCoreType usrCore =
        usr->hasAttr(CubeOnlyAttrName) ? TCoreType::CUBE
        : usr->hasAttr(VecOnlyAttrName)
            ? TCoreType::VECTOR
            : queryCoreTypeHelper(usr).value_or(TCoreType::CUBE_OR_VECTOR);
    if (usrCore != core)
      return nullptr;
    if (opToWorkItemMap.contains(usr))
      return nullptr;
    // `usr` must be unavailable this round: blocked by deps (not in
    // potentiallyAvailable) or already deferred earlier in this pass.
    if (potentiallyAvailable.contains(usr) && !chainDeferred.contains(usr) &&
        !deferredOps.contains(usr))
      return nullptr;
    return usr;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : potentiallyAvailable) {
      if (deferredOps.contains(op) || chainDeferred.contains(op))
        continue;
      if (Operation *usr = findBlockedCoChainConsumer(*op)) {
        LLVM_DEBUG({
          dbgs() << "[extractAvailableOps] deferring acc-chain producer:\n  ";
          op->dump();
          dbgs() << "  for blocked consumer:\n  ";
          usr->dump();
        });
        chainDeferred.insert(op);
        changed = true;
      }
    }
  }

  // Commit chain deferrals only if doing so leaves at least one op for this
  // round; otherwise we would starve the round and createWorkItems would
  // terminate prematurely with `cannot pipeline loop`.
  size_t remaining = 0;
  for (Operation *op : potentiallyAvailable)
    if (!deferredOps.contains(op) && !chainDeferred.contains(op))
      ++remaining;
  if (remaining > 0) {
    deferredOps.insert(chainDeferred.begin(), chainDeferred.end());
  } else if (!chainDeferred.empty()) {
    LLVM_DEBUG(dbgs() << "[extractAvailableOps] skipping acc-chain deferrals "
                         "to avoid empty round\n");
  }

  potentiallyAvailable.set_subtract(deferredOps);
  extractedOps.append(potentiallyAvailable.takeVector());
  return success();
}

/// Block-mode-only: compute localOutputs for each WorkItem (block-level value
/// uses crossing into another WorkItem or escaping via the block's terminator).
/// Loop mode handles this in its own markOutputs (which also tracks
/// yieldedOutputs).
void WorklistBuilder::computeLocalOutputs() {
  DenseMap<Operation *, WorkItem *> opOwner;
  for (auto &item : worklist)
    for (Operation *op : item->ops)
      opOwner[op] = item.get();

  for (auto &item : worklist) {
    for (Operation *op : item->ops) {
      for (Value result : op->getResults()) {
        bool externalUse = false;
        for (Operation *user : result.getUsers()) {
          // Walk up to the scope-level ancestor. Nested users (e.g. inside a
          // for-loop body) ride with their parent's clone; what matters is
          // whether that parent sits in this WorkItem or elsewhere.
          Operation *ancestor = user;
          while (ancestor && ancestor->getParentOp() != scopeOp)
            ancestor = ancestor->getParentOp();
          if (!ancestor) {
            externalUse = true;
            break;
          }
          if (isa<scf::YieldOp>(ancestor)) {
            // Only the scope-level yield escapes scopeOp.
            externalUse = true;
            break;
          }
          auto it = opOwner.find(ancestor);
          if (it == opOwner.end() || it->second != item.get()) {
            externalUse = true;
            break;
          }
        }
        if (externalUse)
          item->localOutputs.push_back({result, nullptr});
      }
    }
  }
}

/// Unified entry point used by both CV pipelining (loop mode) and split-if
/// (block mode). Performs the same core algorithm: scan ops → collect
/// separators → seed dependence map → alternate round-based extraction →
/// trace operand dependencies. Loop-only steps (loop-carried-dep analysis,
/// multibuffer reconciliation, ≥2-WorkItem requirement) are skipped in block
/// mode.
FailureOr<WorklistBuildResult> WorklistBuilder::build() {
  int multibuffer = numMultibuffer > 1 ? numMultibuffer : 2;

  for (Operation &op : targetBlock->getOperations()) {
    if (isCoreOp(op))
      toBePipelined.insert(&op);
    if (isSeparator(&op)) {
      separators.push_back(&op);
      continue;
    }
    if (isLoopMode) {
      if (auto mark = dyn_cast<annotation::MarkOp>(&op)) {
        if (numMultibuffer != -1)
          continue;
        int markMultibuffer = getMultibufferCount(mark);
        if (markMultibuffer == -1)
          continue;
        if (multibuffer < 2)
          multibuffer = markMultibuffer;
        else if (multibuffer != markMultibuffer)
          multibuffer = std::min(multibuffer, markMultibuffer);
        continue;
      }
    }
    if (illegalRegionedOp(op, isLoopMode))
      return failure();
  }
  LLVM_DEBUG({
    dbgs() << "[build] Separators:\n";
    for (Operation *op : separators) {
      dbgs() << "\t";
      op->dump();
    }
    if (isLoopMode)
      dbgs() << "\tmultibuffer = " << multibuffer << "\n\n";
  });

  if (isLoopMode) {
    if (multibuffer < 2)
      return failure();
    if (numMultibuffer < 1)
      numMultibuffer = multibuffer;
  }

  // Seed dependence map from separators (common to both modes).
  for (Operation *separator : separators)
    if (populateDependencies(*separator).failed())
      return failure();

  // Loop-carried deps only exist in loop mode.
  if (isLoopMode)
    populateLoopCarriedDependencies();

  // Round-based extraction, alternating CUBE/VECTOR. Bound iterations by the
  // number of block-level ops (plus slack) — each round consumes at least one
  // op into a WorkItem, so more iterations than that indicates a bug.
  SmallVector<Operation *> independentOps;
  TCoreType core = hivm::TCoreType::CUBE_OR_VECTOR;
  const size_t maxRounds = targetBlock->getOperations().size() + 2;
  bool extractionDone = false;
  for (size_t round = 0; round < maxRounds && !extractionDone; ++round) {
    if (extractAvailableOps(independentOps, core).failed())
      return failure();
    if (independentOps.empty()) {
      extractionDone = true;
      continue;
    }
    if (core == hivm::TCoreType::CUBE_OR_VECTOR)
      return failure();
    if (populateWorkItem(independentOps, core).failed())
      return failure();

    for (auto &[op, dependant] : dependenceMap)
      for (Operation *processed : independentOps)
        dependant.erase(processed);
    independentOps.clear();

    if (core == TCoreType::VECTOR)
      core = TCoreType::CUBE;
    else if (core == TCoreType::CUBE)
      core = TCoreType::VECTOR;
    else
      return failure();
  }
  if (!extractionDone)
    return failure();

  // Block-mode fallback: collect any remaining ops (not assigned to any
  // WorkItem by the round-based extraction) into a single CUBE_OR_VECTOR
  // remainder. Handles:
  //   - blocks with no specific-core ops (only CUBE_OR_VECTOR / non-core);
  //   - mixed-core region ops left unannotated by illegalRegionedOp so the
  //     split-if pattern can recurse on them;
  //   - blocks where all ops are non-core (e.g. empty-like then-branch with
  //     just tensor.empty + yield).
  // Also clears toBePipelined so the post-extraction safety check below
  // doesn't fire in block mode.
  if (!isLoopMode) {
    std::shared_ptr<WorkItem> remainder;
    for (Operation &op : *targetBlock) {
      if (isa<scf::YieldOp>(&op) || opToWorkItemMap.contains(&op))
        continue;
      if (!remainder) {
        remainder = std::make_shared<WorkItem>();
        remainder->core = TCoreType::CUBE_OR_VECTOR;
      }
      mapOpToItem(op, *remainder);
      toBePipelined.erase(&op);
    }
    if (remainder)
      worklist.push_back(remainder);
  }

  // Loop mode: any core op left unassigned is a hard failure.
  // Block mode: the fallback above absorbs leftovers, so toBePipelined is
  // guaranteed empty here.
  if (!toBePipelined.empty()) {
    LLVM_DEBUG({
      for (Operation *op : toBePipelined)
        op->dump();
    });
    if (isLoopMode)
      return scopeOp->emitWarning("[cv-pipelining] cannot pipeline loop due "
                                  "to loop carried dependencies");
    return failure();
  }

  // Loop mode needs ≥2 WorkItems to form an alternating pipeline.
  // Block mode is fine with any worklist size; caller decides.
  if (isLoopMode && worklist.size() < 2)
    return failure();

  // Sort each WorkItem's ops in original block order so consumers that clone
  // ops directly from WorkItem.ops see a valid topological order.
  for (auto &item : worklist) {
    auto vec = item->ops.takeVector();
    llvm::sort(
        vec, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    for (Operation *op : vec)
      item->ops.insert(op);
  }

  // outputMemrefMap is fully populated now that traceMemrefSubnet has run for
  // every separator -- emit non-fatal diagnostics about misplaced or
  // duplicated `cv_pipeline_lazy_load` hints. No-op when no such hints are
  // present (e.g. in block mode, where split-if's input typically has none).
  diagnoseLazyLoadHints();

  // Block mode populates localOutputs here. Loop mode's consumer
  // (WorklistBuilder::markOutputs) does the equivalent computation itself,
  // plus yieldedOutputs tracking.
  if (!isLoopMode)
    computeLocalOutputs();

  // Return COPIES rather than moves: post-build consumers (e.g. CVPipelining
  // holding the WorklistBuilder as a member) need wb's state to remain valid
  // for follow-up queries like wb.shouldLazyLoadFor(op).  worklist holds
  // shared_ptrs and the maps are small, so the copy is cheap.
  WorklistBuildResult result;
  result.worklist = worklist;
  result.opToWorkItemMap = opToWorkItemMap;
  result.outputMemrefMap = outputMemrefMap;
  result.resolvedMultibuffer = isLoopMode ? numMultibuffer : -1;
  return result;
}

} // namespace mlir

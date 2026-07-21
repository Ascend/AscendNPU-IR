//===- CVPipelining.cpp --- Pipelining pass for mix-cv ops ----------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/TileUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/regbase/WorklistBuilder.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"

#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "cv-pipelining"

using llvm::dbgs;
namespace mlir {
using namespace hivm;
#include "bishengir/Dialect/Utils/Util.h"
#define GEN_PASS_DEF_CVPIPELINING
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

using bishengir::memref_ext::AllocWorkspaceOp;
using hivm::detail::queryCoreTypeHelper;

static constexpr llvm::StringLiteral CubeOnlyAttrName = "pipeline.cubeonly";
static constexpr llvm::StringLiteral VecOnlyAttrName = "pipeline.veconly";

namespace {


struct CVPipelineImpl {
  CVPipelineImpl(LoopLikeOpInterface loop, int multibuffer, bool skewMode,
                 bool enableLazyLoading)
      : pipelineLoop(loop), newLoop(nullptr), builder(loop->getContext()),
        numMultibuffer(multibuffer), enableSkewMode(skewMode),
        enableLazyLoading(enableLazyLoading),
        yieldedVals(loop.getYieldedValues().begin(),
                    loop.getYieldedValues().end()) {}

  LogicalResult run();

private:
  LogicalResult createWorkItems();

  LogicalResult populateDependencies(Operation *separator);

  void populateLoopCarriedDependencies();

  LogicalResult extractAvailableOps(SmallVector<Operation *> &extractedOps,
                                    TCoreType &core);

  LogicalResult populateWorkItem(SmallVector<Operation *> &availableOps,
                                 TCoreType core);

  LogicalResult traceDependentOps(WorkItem *item);

  LogicalResult traceMemrefSubnet(Operation *start,
                                  SmallVector<Operation *> &workingStack);

  LogicalResult traceOperands(Value operand, WorkItem *item,
                              SmallVector<Operation *> &workingStack);

  void collectAtomicEffects();

  void markOutputs();

  /// Absorb non-core "merger" ops (e.g. `arith.select`, `arith.cmpi`) that
  /// sit between a work-item op's output and a `scf.yield` operand into the
  /// producing work item. Without this, those ops are never cloned into any
  /// work-item forOp nor into newLoop, so the trailing terminator clone in
  /// migrateOps copies the operand reference verbatim and ends up pointing
  /// at an op inside the soon-to-be-erased pipelineLoop.
  ///
  /// Returns failure when a merger chain cannot be cleanly attributed to a
  /// single work item (chain spans multiple work items, or has no work-item
  /// producer at all). In that case `run()` reverts and the pass becomes a
  /// no-op for this loop.
  LogicalResult absorbMergerOpsIntoWorkItems();

  LogicalResult expandOutputInits(WorkItem *item);
  LogicalResult expandOutputInitsForPreload(WorkItem *item);

  LogicalResult createNewLoops();

  void mapOpToItem(Operation *op, WorkItem *item);

  LogicalResult migrateOps();

  LogicalResult createNewLoopsForPreloadWithScopes();
  LogicalResult markScopesForPreload();

  void expandWorkspace(OpBuilder &builder);
  LogicalResult migrateOpsForPreload(OpBuilder &builder);

  // Undo partial IR changes by erasing newLoop if it was created.
  void revert();

  Value createSubview(OpBuilder &builder, Location loc, Value from, Type to,
                      Value iv);
  Value createToTensor(OpBuilder &builder, Location loc, Value src);
  // Returns failure() on malformed input. On success, the returned Value may
  // still be nullptr when there is no masking subview to update.
  FailureOr<Value> updateMaskingSubview(OpBuilder &builder, Location loc,
                                        Value expanded, OpOperand *initOperand,
                                        Value iv);

  // ===========================================================================
  // Data members
  // ===========================================================================

  // Loop being pipelined
  scf::ForOp pipelineLoop;

  // Unrolled pipelineLoop that will replace it once we're done
  scf::ForOp newLoop;

  OpBuilder builder;

  // Number of multibuffer/pipeline stages/unroll iterations
  int numMultibuffer;

  // Use skew-mode pipelining instead of the default unroll-mode
  bool enableSkewMode;

  // When true, load ops are cloned into each consuming work item rather than
  // passing their results via expanded multi-buffered tensors.
  bool enableLazyLoading;

  // Pipelines we focus on that will be pipelined, everything else will be
  // traced from these based on the operands
  DenseSet<Operation *> toBePipelined;

  // Mapping from the converted memref to the op that writes to it (i.e.
  // FixPipeOp)
  DenseMap<bufferization::ToTensorOp, DestinationStyleOpInterface>
      outputMemrefMap;

  // Separator ops that form the boundry of vector and cube ops (i.e. FixPipeOp
  // or CopyOp)
  SmallVector<Operation *> separators;

  // Map of each operation and what it depends on
  DenseMap<Operation *, DenseSet<Operation *>> dependenceMap;

  // Lookup for yielded values
  SetVector<Value> yieldedVals;

  // Map of each operation with yielded tensor and what depends on it (reverse
  // of dependenceMap)
  DenseMap<Operation *, DenseSet<Operation *>> loopCarriedDependenceMap;

  // Since work items need to be referenced in multiple locations, we use
  // shared_ptr to avoid references being destroyed by vector reallocations
  SmallVector<std::shared_ptr<WorkItem>> worklist;

  // Non-DPS ops could potentially be cloned to various different work items
  DenseMap<Operation *, SmallVector<WorkItem *>> opToWorkItemMap;

  // Corresponding expanded tensors for each output of work items
  DenseMap<Value, Value> expandedTensorMap;

  // Mapping from each op under atomic effect to its atomic kind and data type
  DenseMap<Operation *, AtomicEffect> atomicEffectMap;

  // If the atomic effect is still active at the end of the loop body, this
  // holds that trailing state so it can be restored after the pipelined loops.
  std::optional<AtomicEffect> trailingAtomicEffect;

  // Mapping from the original pipelineLoop to the newLoop to guide the cloning
  // process
  IRMapping globalIRMap;

  DenseMap<Value, Value> localOuputsToRedurnRes;

  DenseSet<Operation *> toErase;

  DenseMap<AllocWorkspaceOp, WorkspaceAllocParams> workspaceAllocs_;
  DenseMap<Value, Value> expandedWorkspaceMap_;
};

struct CVPipeliningPass
    : public ::mlir::impl::CVPipeliningBase<CVPipeliningPass> {
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

static LogicalResult
filterMarkOpsValuesInsideLoop(ArrayRef<annotation::MarkOp> markOps,
                              scf::ForOp loop) {
  bool hasIlligalMarkOp = false;
  for (auto markOp : markOps) {
    SmallVector<mlir::Value> newValues;
    SmallVector<unsigned> indicesToKeep;
    OperandRange oldValues = markOp.getValues();

    for (auto [i, val] : llvm::enumerate(oldValues)) {
      Operation *defineOp = val.getDefiningOp();
      if (defineOp && loop->isAncestor(defineOp)) {
        newValues.push_back(val);
        indicesToKeep.push_back(i);
      }
    }

    if (newValues.size() == oldValues.size())
      continue;

    hasIlligalMarkOp = true;
    mlir::ArrayAttr keysAttr = markOp.getKeysAttr();
    llvm::SmallVector<mlir::Attribute> newKeys;
    if (keysAttr && keysAttr.size() == oldValues.size()) {
      for (unsigned idx : indicesToKeep) {
        newKeys.push_back(keysAttr[idx]);
      }
    } else if (keysAttr) {
      newKeys.assign(keysAttr.begin(), keysAttr.end());
    }

    mlir::OpBuilder builder(markOp);
    builder.setInsertionPoint(markOp);
    auto newMarkOp = builder.create<annotation::MarkOp>(
        markOp.getLoc(), markOp.getSrc(), newValues,
        newKeys.empty() ? mlir::ArrayAttr() : builder.getArrayAttr(newKeys));

    for (auto namedAttr : markOp->getAttrs()) {
      if (namedAttr.getName() != "keys") {
        newMarkOp->setAttr(namedAttr.getName(), namedAttr.getValue());
      }
    }

    markOp.erase();
  }

  if (hasIlligalMarkOp)
    return failure();
  return success();
}

static int getMultibufferCount(annotation::MarkOp marker) {
  auto attrDict = marker->getAttrDictionary();
  hivm::util::validateMultiBufferAttr(attrDict);
  auto multibufferAttr = llvm::dyn_cast_if_present<IntegerAttr>(
      marker->getAttr(MultiBufferAttr::name));
  if (!multibufferAttr)
    return -1;
  return multibufferAttr.getInt();
}

static void removeWorkspaceMultiBufferMarks(Operation *root) {
  SmallVector<annotation::MarkOp> marksToErase;
  root->walk([&](annotation::MarkOp markOp) {
    if (!markOp->hasAttr(MultiBufferAttr::name))
      return WalkResult::advance();
    if (!isa_and_nonnull<AllocWorkspaceOp>(markOp.getSrc().getDefiningOp()))
      return WalkResult::advance();
    markOp->removeAttr(MultiBufferAttr::name);
    if (!markOp->hasAttr(PreloadLocalBufferAttr::name))
      marksToErase.push_back(markOp);
    return WalkResult::advance();
  });
  for (annotation::MarkOp markOp : marksToErase)
    markOp.erase();
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

/// Check to see if op is what we consider a "core op" that is only available on
/// either a cube or vector core
static bool isCoreOp(Operation *op) {
  return op->hasAttr(CubeOnlyAttrName) || op->hasAttr(VecOnlyAttrName) ||
         (isa_and_nonnull<HIVMDialect>(op->getDialect()) &&
          isa<DestinationStyleOpInterface>(op));
}

/// Validate if we can pipeline ops with respect to its regions.
/// Returns false if we can operate on it, otherwise true
static bool illegalRegionedOp(Operation *op) {
  if (op->getRegions().empty())
    return false;
  bool hasCube = false;
  bool hasVector = false;
  WalkResult result = op->walk([&hasCube, &hasVector](Operation *curOp) {
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
    op->emitWarning("[cv-pipelining] unsupported regioned op");
    return true;
  }

  auto unit = UnitAttr::get(op->getContext());
  if (hasCube)
    op->setAttr(CubeOnlyAttrName, unit);
  else if (hasVector)
    op->setAttr(VecOnlyAttrName, unit);
  return false;
}

/// Get the highest level parent op that is not the containing op
static Operation *getContainedParent(Operation *containing, Operation *inner) {
  Operation *parent = inner->getParentOp();
  while (parent && parent != containing && containing->isAncestor(inner)) {
    inner = parent;
    parent = inner->getParentOp();
  }
  return inner;
}

static Operation *getContainedParent(Operation *containing, Value inner) {
  Operation *defining = inner.getDefiningOp();
  if (defining)
    return getContainedParent(containing, defining);
  return nullptr;
}

/// Expand workspace by taking original workspace, and adding a multibuffer dim
/// enable cvpipelining: if multibuffer is 2, original workspace is <16x16xf16>,
/// then new expanded workspace is <2x16x16xf16> enable preload: if work item
/// num is 4, original workspace is <16x16xf16>, then new expanded workspace is
/// <4x16x16xf16>
void CVPipelineImpl::expandWorkspace(OpBuilder &builder) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(pipelineLoop);
  for (auto [alloc, info] : workspaceAllocs_) {
    Location loc = alloc.getLoc();
    MemRefType origType = alloc.getType();
    ArrayRef<int64_t> origShape = origType.getShape();

    // if enable preload, workspace size depends on work item num
    int64_t expandSize = info.multibuffer;
    SmallVector<int64_t> newShape = {expandSize};
    newShape.append(origShape.begin(), origShape.end());
    auto newType = MemRefType::get(newShape, origType.getElementType());
    auto newAlloc = builder.create<AllocWorkspaceOp>(
        loc, newType, alloc.getWorkspaceArg(), alloc.getDynamicSize(),
        alloc.getOffset());

    // Here we replace the tensor with a memref, this is to avoid further
    // complications with the extract->use->insert->yield pattern
    expandedWorkspaceMap_[alloc] = newAlloc;
    info.marker.getSrcMutable().set(newAlloc);
    info.marker->removeAttr(MultiBufferAttr::name);

    toErase.insert(alloc);
    toErase.insert(info.toTensor);
  }
}

static AllocWorkspaceOp getAllocWorkspace(Value v) {
  Operation *defining = v.getDefiningOp();
  if (!defining)
    return nullptr;

  if (auto alloc = dyn_cast<AllocWorkspaceOp>(defining))
    return alloc;

  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(defining))
    return getAllocWorkspace(toTensor.getMemref());

  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(defining))
    return getAllocWorkspace(dpsOp.getTiedOpOperand(cast<OpResult>(v))->get());
  return nullptr;
}

// Implementation of slice operations
static tensor::InsertSliceOp createInsertSlice(OpBuilder &builder, Location loc,
                                               Value src, Value into,
                                               Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  auto originalType = cast<TensorType>(src.getType());
  SmallVector<OpFoldResult> offsets, sizes, strides;
  if (!iv.getType().isIndex())
    iv = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), iv);
  offsets.push_back(iv);
  offsets.append(originalType.getRank(), const0);

  sizes.push_back(const1);
  for (int i = 0; i < originalType.getRank(); ++i) {
    if (originalType.isDynamicDim(i))
      sizes.push_back(builder.createOrFold<tensor::DimOp>(loc, src, i));
    else
      sizes.push_back(builder.getIndexAttr(originalType.getDimSize(i)));
  }

  strides.append(originalType.getRank() + 1, const1);

  return builder.create<tensor::InsertSliceOp>(loc, src, into, offsets, sizes,
                                               strides);
}

static tensor::ExtractSliceOp createExtractSlice(OpBuilder &builder,
                                                 Location loc, Value from,
                                                 Type to, Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  auto newType = cast<TensorType>(from.getType());

  if (!iv.getType().isIndex())
    iv = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), iv);
  offsets.push_back(iv);
  offsets.append(newType.getRank() - 1, const0);
  sizes.push_back(const1);
  for (int i = 1; i < newType.getRank(); ++i) {
    if (newType.isDynamicDim(i))
      sizes.push_back(builder.createOrFold<tensor::DimOp>(loc, from, i));
    else
      sizes.push_back(builder.getIndexAttr(newType.getDimSize(i)));
  }

  strides.append(newType.getRank(), const1);
  return builder.create<tensor::ExtractSliceOp>(loc, cast<RankedTensorType>(to),
                                                from, offsets, sizes, strides);
}

static void createAttrForPreloadWS(OpBuilder &builder, Value markedVal) {
  auto markedOp = markedVal.getDefiningOp();
  markedOp->setAttr(hivm::PreloadWorkspaceAttr::name, builder.getUnitAttr());
}

static Value createWorkspaceSubview(OpBuilder &builder, Location loc,
                                    Value from, Value iv,
                                    bool isPreload = false) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  auto newType = cast<MemRefType>(from.getType());

  // Set up offsets
  if (!iv.getType().isIndex())
    iv = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), iv);
  offsets.push_back(iv);
  offsets.append(newType.getRank() - 1, const0);
  // Set up sizes
  sizes.push_back(const1);
  for (int i = 1; i < newType.getRank(); ++i) {
    if (newType.isDynamicDim(i))
      sizes.push_back(builder.createOrFold<memref::DimOp>(loc, from, i));
    else
      sizes.push_back(builder.getIndexAttr(newType.getDimSize(i)));
  }

  // ... and strides
  strides.append(newType.getRank(), const1);

  // Somehow builder does not want to create rank reduced version of subview, so
  // we reduce it manually
  auto subview =
      builder.create<memref::SubViewOp>(loc, from, offsets, sizes, strides);
  if (isPreload)
    createAttrForPreloadWS(builder, subview);
  SmallVector<ReassociationIndices> reass{{0, 1}};
  for (unsigned i = 2; i < subview.getType().getRank(); ++i)
    reass.push_back({i});
  return builder.create<memref::CollapseShapeOp>(loc, subview, reass);
}

static void
processWorkspaceOutputs(OpBuilder &builder, WorkItem *item,
                        DenseMap<Value, Value> &expandedWorkspaceMap,
                        const IRMapping &loopMap) {
  scf::ForOp forOp = item->forOp;
  for (Operation *output : item->workspaceOutputs) {
    auto dpsOp = cast<DestinationStyleOpInterface>(output);
    Value original = getAllocWorkspace(dpsOp.getDpsInitOperand(0)->get());
    Value newAlloc = expandedWorkspaceMap.lookup(original);
    Operation *store = loopMap.lookupOrDefault(output);
    builder.setInsertionPoint(store);
    Location loc = store->getLoc();
    Value newDst =
        createWorkspaceSubview(builder, loc, newAlloc, forOp.getInductionVar());
    if (auto storeOp = dyn_cast<StoreOp>(store))
      builder.create<StoreOp>(loc, TypeRange{}, storeOp.getSrc(), newDst);
    else if (auto fixpipe = dyn_cast<FixpipeOp>(store))
      builder.create<FixpipeOp>(
          loc, TypeRange{}, fixpipe.getSrc(), newDst, fixpipe.getDmaModeAttr(),
          fixpipe.getDualDstModeAttr(), fixpipe.getSubBlockIdxAttr(),
          fixpipe.getPreQuantAttr(),
          fixpipe.getPreReluAttr(), fixpipe.getChannelSplitAttr());

    // Before forOp so to_tensor dominates in-loop and later consumers.
    builder.setInsertionPoint(forOp);
    auto workspaceOp = builder.create<bufferization::ToTensorOp>(
        loc, newAlloc, /*restrict*/ true);
    expandedWorkspaceMap[original] = workspaceOp;

    Value sliceIdx = forOp.getInductionVar();
    SmallVector<OpOperand *> operandsToUpdate;
    for (OpOperand &operand : store->getUses())
      operandsToUpdate.push_back(&operand);
    for (OpOperand *operand : operandsToUpdate) {
      Operation *userOp = operand->getOwner();
      if (isa<annotation::MarkOp>(userOp)) {
        userOp->erase();
        continue;
      }
      if (!isa<TensorType>(operand->get().getType()))
        continue;
      builder.setInsertionPoint(userOp);
      Value sliceOp = createExtractSlice(builder, loc, workspaceOp,
                                         operand->get().getType(), sliceIdx);
      operand->set(sliceOp);
    }
    store->erase();
  }
}

static void processWorkspaceOutputUsers(
    OpBuilder &builder, WorkItem *item,
    const DenseMap<Operation *, SmallVector<WorkItem *>> &opToWorkItemMap,
    const DenseMap<Value, Value> &expandedWorkspaceMap,
    SmallVector<std::pair<OpOperand *, Value>> &replaces, scf::ForOp newLoop) {
  for (Operation *output : item->workspaceOutputs) {
    for (OpOperand &operand : output->getUses()) {
      Operation *owner = operand.getOwner();
      if (opToWorkItemMap.contains(owner))
        continue;
      if (!newLoop->isAncestor(owner))
        // only usr in new loop need to handle
        continue;
      Value sliceIdx;
      builder.setInsertionPoint(owner);
      if (isa<scf::YieldOp>(owner)) {
        sliceIdx = builder.create<arith::ConstantIndexOp>(
            owner->getLoc(), item->multibuffer - 1);
      } else
        sliceIdx = owner->getParentOfType<scf::ForOp>().getInductionVar();
      auto alloc = getAllocWorkspace(operand.get());
      Value mappedTensor = expandedWorkspaceMap.lookup(alloc);
      Value slice = createExtractSlice(builder, owner->getLoc(), mappedTensor,
                                       operand.get().getType(), sliceIdx);
      replaces.emplace_back(&operand, slice);
    }
  }
}

static void processOutputUsers(
    OpBuilder &builder, ArrayRef<std::shared_ptr<WorkItem>> worklist,
    const DenseMap<Operation *, SmallVector<WorkItem *>> &opToWorkItemMap,
    const DenseMap<Value, Value> &expandedWorkspaceMap, scf::ForOp newLoop) {
  SmallVector<std::pair<OpOperand *, Value>> replaces;
  for (auto &item : worklist) {
    processWorkspaceOutputUsers(builder, item.get(), opToWorkItemMap,
                                expandedWorkspaceMap, replaces, newLoop);
  }
  for (auto [operand, replace] : replaces) {
    Operation *owner = operand->getOwner();
    if (newLoop->isAncestor(owner))
      operand->set(replace);
  }
}

Value CVPipelineImpl::createSubview(OpBuilder &builder, Location loc,
                                    Value from, Type to, Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  auto targetTy = cast<MemRefType>(to);
  if (!iv.getType().isIndex())
    iv = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), iv);
  offsets.push_back(iv);
  offsets.append(targetTy.getRank(), const0);
  sizes.push_back(const1);
  for (int64_t dim : targetTy.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      pipelineLoop->emitWarning(
          "[cv-pipelining] unexpected dynamic dim in target memref");
      return nullptr;
    }
    sizes.push_back(builder.getIndexAttr(dim));
  }
  strides.append(targetTy.getRank() + 1, const1);
  int64_t offset;
  SmallVector<int64_t> layoutStrides;
  if (getStridesAndOffset(targetTy, layoutStrides, offset).failed()) {
    pipelineLoop->emitWarning("[cv-pipelining] unexpected memref layout");
    return nullptr;
  }
  auto layout = StridedLayoutAttr::get(builder.getContext(),
                                       ShapedType::kDynamic, layoutStrides);
  Attribute srcMemSpace = cast<MemRefType>(from.getType()).getMemorySpace();
  auto finalTy = MemRefType::Builder(targetTy).setLayout(layout).setMemorySpace(
      srcMemSpace);
  Value subview = builder.create<memref::SubViewOp>(loc, finalTy, from, offsets,
                                                    sizes, strides);
  if (srcMemSpace != targetTy.getMemorySpace())
    subview = builder.create<memref::MemorySpaceCastOp>(
        loc, MemRefType(MemRefType::Builder(finalTy).setMemorySpace(nullptr)),
        subview);
  return subview;
}

/// Walk backward from every loop-yield operand through non-core ops that
/// are not yet claimed by any WorkItem. Every such "merger" op must end up
/// owned by a WorkItem so that:
///   - it is cloned into that work-item's forOp during `migrateOps`
///     (using the per-WorkItem `irMap`, which correctly remaps both the
///     reconstructed induction variable and work-item-produced values), and
///   - its result, when it equals a `yieldedVals` entry, can be picked up
///     by the existing `yieldedOutputs` mechanism in `markOutputs` /
///     `createNewLoops`.
///
/// A chain rooted at a yield operand is absorbed into work item `W` iff
/// every chain operand that is defined inside `pipelineLoop`'s body either
///   - belongs to `W` already (work-item op result), or
///   - belongs to another op in the same merger chain, or
///   - is a block argument of `pipelineLoop` (iter_arg or the IV).
///
/// If a chain references results from two or more distinct work items, or
/// has no work-item producer at all (purely iter_arg/IV-driven), we cannot
/// safely absorb it and return failure so `run()` reverts cleanly.
LogicalResult CVPipelineImpl::absorbMergerOpsIntoWorkItems() {
  Block *body = pipelineLoop.getBody();
  Operation *terminator = body->getTerminator();

  for (Value yieldOperand : terminator->getOperands()) {
    Operation *root = yieldOperand.getDefiningOp();
    if (!root || root->getBlock() != body)
      continue; // block arg or defined outside body — nothing to absorb
    if (opToWorkItemMap.contains(root))
      continue; // already a direct work-item output

    // Collect the chain of unclaimed non-core ops feeding `yieldOperand`
    // and the set of work items that ultimately produce its data.
    SetVector<Operation *> chain;
    SmallPtrSet<WorkItem *, 4> producers;
    SmallVector<Operation *> stack{root};
    while (!stack.empty()) {
      Operation *cur = stack.pop_back_val();
      if (!chain.insert(cur))
        continue;
      for (Value operand : cur->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue; // block arg (iter_arg / IV) — fine
        if (def->getBlock() != body)
          continue; // outside pipelineLoop body — fine
        auto it = opToWorkItemMap.find(def);
        if (it != opToWorkItemMap.end()) {
          for (WorkItem *wi : it->getSecond())
            producers.insert(wi);
          continue; // don't walk through work-item ops
        }
        stack.push_back(def);
      }
    }

    if (producers.size() != 1)
      return pipelineLoop->emitWarning("[cv-pipelining] cannot absorb merger "
                                       "chain into a single work "
                                       "item: the yielded value depends on ")
             << producers.size()
             << " work-item producer(s); refusing to pipeline this loop";

    WorkItem *target = *producers.begin();
    for (Operation *m : chain) {
      target->ops.insert(m);
      opToWorkItemMap[m].push_back(target);
      LLVM_DEBUG(dbgs() << "[absorbMergerOps] absorbed into work item: ";
                 m->print(dbgs()); dbgs() << '\n');
    }
  }

  // Counter-advance absorption (normalize-matmul counter advanced directly in
  // the loop body). The `memref.store %inc` back to a counter alloca and the
  // `arith.addi` producing %inc have no SSA result and never reach scf.yield,
  // so the yield-chain walk above misses them and migrateOps would drop them,
  // leaving the post-loop load reading the initial 0. (The regioned-op case is
  // handled separately by preprocessCounterAllocas's vector-safe clone.)
  for (const auto &item : worklist) {
    SmallPtrSet<Value, 4> loadedCounters;
    for (Operation *op : item->ops)
      if (auto load = dyn_cast<memref::LoadOp>(op))
        if (auto a = load.getMemRef().getDefiningOp<memref::AllocaOp>())
          if (a->hasAttr(kNormalizeMatmulCounterAttr))
            loadedCounters.insert(load.getMemRef());

    for (Operation &op : *body) {
      auto store = dyn_cast<memref::StoreOp>(&op);
      if (!store || !loadedCounters.contains(store.getMemRef()) ||
          opToWorkItemMap.contains(&op))
        continue;
      if (Operation *inc = store.getValue().getDefiningOp())
        if (isa<arith::AddIOp>(inc) && !opToWorkItemMap.contains(inc)) {
          item->ops.insert(inc);
          opToWorkItemMap[inc].push_back(item.get());
          LLVM_DEBUG(dbgs() << "[absorbMergerOps] absorbed counter addi: ";
                     inc->print(dbgs()); dbgs() << '\n');
        }
      item->ops.insert(&op);
      opToWorkItemMap[&op].push_back(item.get());
      LLVM_DEBUG(dbgs() << "[absorbMergerOps] absorbed counter store: ";
                 op.print(dbgs()); dbgs() << '\n');
    }
  }

  return success();
}

Value CVPipelineImpl::createToTensor(OpBuilder &builder, Location loc,
                                     Value src) {
  auto memref = dyn_cast<MemRefType>(src.getType());
  if (!memref) {
    pipelineLoop->emitWarning("[cv-pipelining] expected MemRefType source");
    return nullptr;
  }
  if (memref.getMemorySpace()) {
    auto newMemRef = MemRefType::get(memref.getShape(), memref.getElementType(),
                                     memref.getLayout());
    src = builder.create<memref::MemorySpaceCastOp>(loc, newMemRef, src);
  }

  return builder.create<bufferization::ToTensorOp>(loc, src, /*restrict*/ true,
                                                   /*writable*/ true);
}

void CVPipelineImpl::mapOpToItem(Operation *op, WorkItem *item) {
  if (item->ops.contains(op))
    return;
  if (opToWorkItemMap.contains(op))
    opToWorkItemMap[op].push_back(item);
  else
    opToWorkItemMap[op] = {item};
  item->ops.insert(op);
}

/// DFS to find all ops that are dependent on separator
LogicalResult CVPipelineImpl::populateDependencies(Operation *separator) {
  SmallVector<Operation *> dfsStack = {separator};
  DenseSet<Operation *> visited;

  while (!dfsStack.empty()) {
    Operation *op = dfsStack.pop_back_val();
    if (visited.contains(op) || !pipelineLoop->isAncestor(op))
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

        Operation *scopedUsr = getContainedParent(pipelineLoop, usr);
        if (isa<scf::YieldOp, scf::ConditionOp>(scopedUsr) ||
            scopedUsr == separator)
          continue;
        dfsStack.push_back(scopedUsr);
        if (dependenceMap.contains(scopedUsr))
          dependenceMap[scopedUsr].insert(separator);
        else
          dependenceMap[scopedUsr] = DenseSet<Operation *>({separator});
      }
    }
  }
  return success();
}

/// Populate dependencies that are carried between loop iterations (ItarArgs,
/// Yield Operands)
void CVPipelineImpl::populateLoopCarriedDependencies() {
  auto maybeYield = pipelineLoop.getYieldedValuesMutable();
  if (!maybeYield.has_value())
    return;
  for (OpOperand &yieldOperand : *maybeYield) {
    Value yieldVal = yieldOperand.get();
    // We only care about the tensor values
    if (!isa<TensorType>(yieldVal.getType()))
      continue;
    Operation *defining = yieldVal.getDefiningOp();
    if (!defining || !pipelineLoop->isAncestor(defining))
      continue;
    BlockArgument iterArg =
        pipelineLoop.getRegionIterArgs()[yieldOperand.getOperandNumber()];
    SmallVector<Operation *> dfsStack(iterArg.getUsers());
    DenseSet<Operation *> visited;
    while (!dfsStack.empty()) {
      Operation *op = getContainedParent(pipelineLoop, dfsStack.pop_back_val());
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

/// Helper to trace the alloc (if within pipelineLoop), toTensor, and
/// potentially various casts along the way
LogicalResult
CVPipelineImpl::traceMemrefSubnet(Operation *start,
                                  SmallVector<Operation *> &workingStack) {
  if (llvm::all_of(start->getOperands(),
                   [](Value val) { return isa<TensorType>(val.getType()); }))
    return success();

  DestinationStyleOpInterface writer = nullptr;
  Value targetOperand = start->getOperand(1);
  if (isa<TensorType>(targetOperand.getType()))
    writer = cast<DestinationStyleOpInterface>(start);

  Operation *defining = targetOperand.getDefiningOp();
  while (defining) {
    if (!pipelineLoop->isAncestor(defining))
      break;
    start = defining;
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
  SmallVector<Operation *> userTraceStack = {start};
  bufferization::ToTensorOp toTensor = nullptr;

  while (!userTraceStack.empty()) {
    Operation *def = userTraceStack.pop_back_val();
    workingStack.push_back(def);
    for (Operation *usr : def->getUsers()) {
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(usr)) {
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
    users.push_back(op);

    // If not memref result, dont need to trace any more
    if (op->getNumResults() == 1 &&
        !isa<MemRefType>(op->getResult(0).getType()))
      continue;
    traceStack.append(op->user_begin(), op->user_end());
  }
}

LogicalResult
CVPipelineImpl::traceOperands(Value operand, WorkItem *item,
                              SmallVector<Operation *> &workingStack) {
  if (!operand)
    return success();
  Operation *defining = getContainedParent(pipelineLoop, operand);
  if (item->ops.contains(defining))
    return success();
  if (!defining) {
    auto iterArg = dyn_cast<BlockArgument>(operand);
    if (!iterArg)
      return pipelineLoop->emitWarning(
          "[cv-pipelining] expected non-op-defined value to be block argument");
    for (Operation *usr : iterArg.getUsers()) {
      if (isa<DebugOp>(usr) && !item->ops.contains(usr) &&
          usr->getParentOp() == pipelineLoop)
        workingStack.push_back(usr);
    }
    if (iterArg.getOwner()->getParentOp() != pipelineLoop ||
        iterArg.getArgNumber() == 0)
      return success();
    // Need to pull defining op into this work item to guarentee safety,
    // should already be guarenteed by extractAvailableOps
    if (isa<TensorType>(operand.getType()))
      return success();
    Value yieldVal = pipelineLoop.getTiedLoopYieldedValue(iterArg)->get();
    defining = yieldVal.getDefiningOp();
    if (!defining || defining->getParentOp() != pipelineLoop)
      return success();
  }
  if (defining->getParentOp() != pipelineLoop)
    return success();
  // If defining is a memref, then trace everything that also uses that memref.
  if (isa<MemRefType>(operand.getType()))
    memrefDFS(operand, workingStack);
  // To tensor ops are handled as a part of the memref operand for
  // load/fixpipe/copy
  if (!item->ops.contains(defining))
    workingStack.push_back(defining);
  return success();
}

/// Trace each op in the initial set of ops in each WorkItem, get non-HIVM ops
/// that are operands for each op
LogicalResult CVPipelineImpl::traceDependentOps(WorkItem *item) {
  SmallVector<Operation *> workingStack(item->ops.begin(), item->ops.end());

  while (!workingStack.empty()) {
    Operation *op = workingStack.pop_back_val();
    if (isCoreOp(op)) {
      if (opToWorkItemMap.contains(op)) {
        // If Core Op is already inserted into a different work item, then we
        // don't include it here
        if (!item->ops.contains(op)) {
          // With lazy loading, allow Load ops to be cloned into multiple
          // work items so each stage loads independently from GM.
          if (!(enableLazyLoading && isa<LoadOp>(op)))
            continue;
        }
      } else if (!isa<LoadOp>(op))
        // Load ops are pulled into their consuming work items, apart from that,
        // if we get here that means we depend on an op that have not satisfied
        // their dependency
        return op->emitWarning(
            "[cv-pipelining] cannot pipeline op due to dependency");
    }
    // Other than core ops, we can skip them if we already inserted them into
    // this work item
    else if (item->ops.contains(op))
      continue;

    // Determine whether a to_tensor should be skipped because it has already
    // been allocated to another work item.  The default guard fires for all
    // to_tensor ops, but with lazy loading we lift the restriction for
    // to_tensor ops whose backing writer is a LoadOp: those are cloned into
    // every consuming work item independently.
    bool skipToTensor =
        isa<bufferization::ToTensorOp>(op) && opToWorkItemMap.contains(op);
    if (enableLazyLoading && skipToTensor) {
      auto tt = cast<bufferization::ToTensorOp>(op);
      auto it = outputMemrefMap.find(tt);
      if (it != outputMemrefMap.end() && isa<LoadOp>(it->second.getOperation()))
        skipToTensor = false;
    }
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
      // workspace related toTensor should skip in c220
      Operation *defining = toTensor.getMemref().getDefiningOp();
      if (isa_and_nonnull<AllocWorkspaceOp>(defining))
        skipToTensor = true;
    }
    if (op->getParentOp() != pipelineLoop || isa<scf::YieldOp>(op) ||
        skipToTensor)
      continue;
    LLVM_DEBUG(dbgs() << "Inserting \t"; op->dump());
    mapOpToItem(op, item);
    toBePipelined.erase(op);
    for (Operation *usr : op->getUsers())
      if (isa<annotation::MarkOp, DebugOp>(usr))
        mapOpToItem(usr, item);

    if (isa<LoadOp, FixpipeOp>(op) || isCrossCoreCopy(op)) {
      if (failed(traceMemrefSubnet(op, workingStack)))
        return failure();
      for (Value operand : op->getOperands()) {
        if (failed(traceOperands(operand, item, workingStack)))
          return failure();
      }
      continue;
    }

    // Handle nested ops as well
    WalkResult walkResult = op->walk([&](Operation *nestedOp) {
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
CVPipelineImpl::populateWorkItem(SmallVector<Operation *> &availableOps,
                                 TCoreType core) {
  auto item = std::make_shared<WorkItem>();
  item->core = core;

#ifndef NDEBUG
  static int id = 0;
  item->id = id++;
#endif

  // ExtractOps made sure that there are only one core type of ops in
  // availableOps, no need to check here
  for (Operation *op : availableOps) {
    mapOpToItem(op, item.get());
  }
  LLVM_DEBUG({
    dbgs() << "[populateWorkItem] Initial set{\n";
    for (Operation *op : item->ops) {
      dbgs() << '\t';
      op->dump();
    }
    dbgs() << "[populateWorkItem] } // Initial set\n";
  });

  if (traceDependentOps(item.get()).failed())
    return failure();
  worklist.push_back(item);
  return success();
}

/// Find ops that have no dependencies, i.e. ops that can be executed if all
/// other previously extracted ops are done executing
LogicalResult
CVPipelineImpl::extractAvailableOps(SmallVector<Operation *> &extractedOps,
                                    TCoreType &core) {
  SetVector<Operation *> potentiallyAvailable;

  for (Operation &op : *pipelineLoop.getBody()) {
    if (opToWorkItemMap.contains(&op))
      continue;
    TCoreType maybeCore = op.hasAttr(CubeOnlyAttrName) ? TCoreType::CUBE
                          : op.hasAttr(VecOnlyAttrName)
                              ? TCoreType::VECTOR
                              : TCoreType::CUBE_OR_VECTOR;
    if (maybeCore == hivm::TCoreType::CUBE_OR_VECTOR) {
      if (!isCoreOp(&op) || isa<LoadOp>(&op))
        continue;
      maybeCore = queryCoreTypeHelper(&op).value_or(TCoreType::CUBE_OR_VECTOR);
      if (maybeCore != TCoreType::VECTOR && isCrossCoreCopy(&op))
        maybeCore = TCoreType::VECTOR;
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

  // Propagate the loop carried dependencies throughout the potentially
  // available ops
  SmallVector<Operation *> dfsStack;
  dfsStack.append(deferredOps.begin(), deferredOps.end());
  while (!dfsStack.empty()) {
    Operation *op = dfsStack.pop_back_val();
    if (deferredOps.contains(op))
      continue;
    if (potentiallyAvailable.contains(op))
      deferredOps.insert(op);
    for (Operation *usr : op->getUsers()) {
      dfsStack.push_back(usr);
    }
  }
  potentiallyAvailable.set_subtract(deferredOps);
  extractedOps.append(potentiallyAvailable.takeVector());

  return success();
}

/// Walk the pipeline loop body and record which store-like ops (FixpipeOp,
/// StoreOp) are under an active atomic effect.
void CVPipelineImpl::collectAtomicEffects() {
  std::optional<AtomicEffect> current;
  for (Operation &op : *pipelineLoop.getBody()) {
    if (auto setAtomic = dyn_cast<SetAtomicOp>(&op)) {
      if (setAtomic.getKind() != AtomicKind::NONE)
        current = AtomicEffect{setAtomic.getKind(), setAtomic.getTypeAttr()};
      else
        current = std::nullopt;
      continue;
    }
    if (current && isa<FixpipeOp, StoreOp>(&op))
      atomicEffectMap[&op] = *current;
  }
  trailingAtomicEffect = current;
  LLVM_DEBUG({
    dbgs() << "[collectAtomicEffects] Ops under atomic effect:\n";
    for (auto &[op, effect] : atomicEffectMap) {
      dbgs() << "\t" << stringifyAtomicKind(effect.kind) << " ";
      if (effect.type)
        dbgs() << effect.type;
      dbgs() << ": ";
      op->dump();
    }
  });
}

static LogicalResult
markWorkspaceOps(Operation *op,
                 DenseMap<AllocWorkspaceOp, WorkspaceAllocParams> &allocs,
                 unsigned multibuffer) {
  if (auto mark = dyn_cast<annotation::MarkOp>(op)) {
    if (auto alloc = llvm::dyn_cast_if_present<AllocWorkspaceOp>(
            mark.getSrc().getDefiningOp())) {
      if (allocs.contains(alloc)) {
        allocs[alloc].multibuffer = multibuffer;
        allocs[alloc].marker = mark;
      } else
        allocs[alloc] = {multibuffer, mark, nullptr};
      return success();
    }
  }
  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
    auto alloc = llvm::dyn_cast_if_present<AllocWorkspaceOp>(
        toTensor.getMemref().getDefiningOp());
    if (!alloc)
      return success();

    if (!toTensor.getResult().hasOneUse() ||
        llvm::range_size(alloc.getResult().getUsers()) != 2)
      return alloc->emitWarning(
          "Expecting alloc_workspace and its tensor to only have one user "
          "(excluding annotation.mark)");

    if (allocs.contains(alloc))
      allocs[alloc].toTensor = toTensor;
    else
      allocs[alloc] = {0, nullptr, toTensor};
  }
  return success();
}

LogicalResult CVPipelineImpl::createWorkItems() {
  int multibuffer = numMultibuffer > 1 ? numMultibuffer : 2;
  Block *blk = pipelineLoop.getBody();
  for (Operation &op : blk->getOperations()) {
    if (isCoreOp(&op))
      toBePipelined.insert(&op);
    if (isa<FixpipeOp, StoreOp>(&op) || isCrossCoreCopy(&op))
      separators.push_back(&op);
    else if (auto mark = dyn_cast<annotation::MarkOp>(&op)) {
      if (numMultibuffer != -1)
        continue;
      int markMultibuffer = getMultibufferCount(mark);
      if (markMultibuffer == -1)
        continue;
      if (multibuffer <= 2)
        multibuffer = markMultibuffer;
      else if (multibuffer != markMultibuffer) {
        multibuffer = std::min(multibuffer, markMultibuffer);
      }
    } else if (illegalRegionedOp(&op)) {
      return failure();
    }

    if (markWorkspaceOps(&op, workspaceAllocs_, multibuffer).failed())
      // mark workspace only for c220
      return failure();
  }
  LLVM_DEBUG({
    dbgs() << "[createWorkItems] Separators:\n";
    for (Operation *op : separators) {
      dbgs() << "\t";
      op->dump();
    }
    dbgs() << "\tmultibuffer = " << multibuffer << "\n\n";
  });
  if (multibuffer < 2)
    return failure();

  if (numMultibuffer < 1)
    numMultibuffer = multibuffer;

  for (Operation *separator : separators)
    if (populateDependencies(separator).failed())
      return failure();
  populateLoopCarriedDependencies();

  SmallVector<Operation *> independentOps;
  bool done = false;
  TCoreType core = hivm::TCoreType::CUBE_OR_VECTOR;
  while (!done) {
    if (extractAvailableOps(independentOps, core).failed() ||
        core == hivm::TCoreType::CUBE_OR_VECTOR)
      return failure();

    if (independentOps.empty()) {
      done = true;
      break;
    }

    if (populateWorkItem(independentOps, core).failed())
      return failure();

    for (auto &[op, dependant] : dependenceMap) {
      for (Operation *processed : independentOps)
        dependant.erase(processed);
    }
    independentOps.clear();
    if (core == TCoreType::VECTOR)
      core = TCoreType::CUBE;
    else if (core == TCoreType::CUBE)
      core = TCoreType::VECTOR;
    else
      return failure();
  }
  if (!toBePipelined.empty()) {
    LLVM_DEBUG({
      for (Operation *op : toBePipelined)
        op->dump();
    });
    return pipelineLoop->emitWarning("[cv-pipelining] cannot pipeline loop due "
                                     "to loop carried dependencies");
  }
  if (worklist.size() < 2)
    return failure();
  return success();
}

void CVPipelineImpl::markOutputs() {
  for (const auto &item : worklist) {
    for (Operation *op : item->ops) {
      if (isa<StoreLikeOpInterface>(op) &&
          getAllocWorkspace(cast<DestinationStyleOpInterface>(op)
                                .getDpsInitOperand(0)
                                ->get())) {
        item->workspaceOutputs.push_back(op);
        continue;
      }
      if (isa<tensor::EmptyOp>(op))
        continue;
      if (enableLazyLoading) {
        if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
          auto it = outputMemrefMap.find(toTensor);
          if (it != outputMemrefMap.end() &&
              isa<LoadOp>(it->second.getOperation()))
            continue;
        }
      }
      for (Value result : op->getResults()) {
        if (yieldedVals.contains(result)) {
          unsigned opNumber = static_cast<unsigned>(std::distance(
              yieldedVals.begin(), llvm::find(yieldedVals, result)));
          item->yieldedOutputs.push_back(std::make_pair(result, opNumber));
          continue;
        }
        if (!isa<TensorType>(result.getType()))
          continue;

        for (Operation *usr : result.getUsers()) {
          // Nested consumers (e.g. load/store inside a per-row scf.for) are
          // not in `opToWorkItemMap`; resolve to the top-level owner first.
          // Otherwise cross-item tensors are missed and `migrateOps` leaves
          // dangling uses on `pipelineLoop`.
          Operation *owner = getContainedParent(pipelineLoop, usr);
          if (owner && opToWorkItemMap.contains(owner) &&
              !item->ops.contains(owner)) {
            item->localOutputs.push_back(std::make_pair(result, nullptr));
            break;
          }
        }
      }
    }
  }
}

LogicalResult CVPipelineImpl::expandOutputInits(WorkItem *item) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(newLoop.getBody());
  for (auto &[output, expanded] : item->localOutputs) {
    Operation *defining = output.getDefiningOp();
    if (!defining)
      return pipelineLoop->emitWarning(
          "[cv-pipelining] expected work item output to be result of op");
    Location loc = defining->getLoc();
    SmallVector<int64_t> newShape({numMultibuffer});
    bufferization::ToTensorOp toTensor = nullptr;
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(defining)) {
      if (dps.getNumDpsInits() != 1)
        return dps->emitWarning(
            "[cv-pipelining] expected dps op with exactly one init");
      Value init = dps.getDpsInitOperand(0)->get();
      defining = init.getDefiningOp();
      if (!defining)
        return dps->emitWarning(
            "[cv-pipelining] expected dps init to be result of op");
      if (isa<tensor::EmptyOp>(defining)) {
        auto origTy = dyn_cast<TensorType>(init.getType());
        if (!origTy)
          return defining->emitWarning(
              "[cv-pipelining] expected output to be tensor type");
        auto shapeArr = origTy.getShape();
        newShape.append(shapeArr.begin(), shapeArr.end());
        auto newType = RankedTensorType::get(newShape, origTy.getElementType());
        expanded = builder.create<tensor::EmptyOp>(loc, newType, ValueRange());
        continue;
      }
    }
    toTensor = dyn_cast<bufferization::ToTensorOp>(defining);
    if (!toTensor)
      return defining->emitWarning(
          "[cv-pipelining] expected to_tensor for non-tensor-empty output");
    auto alloc = traceAlloc(toTensor.getMemref());
    if (!alloc)
      return toTensor->emitWarning(
          "[cv-pipelining] expected alloc from toTensor");

    auto origTy = alloc.getMemref().getType();
    if (!origTy.hasStaticShape())
      return alloc->emitWarning(
          "[cv-pipelining] expected temporary buffer to be static");
    newShape.append(origTy.getShape().begin(), origTy.getShape().end());
    auto memspace = origTy.getMemorySpace();
    auto newType = MemRefType::get(newShape, origTy.getElementType(),
                                   MemRefLayoutAttrInterface(), memspace);
    auto expandedAlloc = builder.create<memref::AllocOp>(
        loc, newType, ValueRange(), alloc.getAlignmentAttr());
    // Mark the expanded alloc with an `annotation.mark` carrying
    // `hivm.cv_pipelined_multi_buffer` so downstream passes (notably
    // the ND2NZOp aggregated-decompose pad/vbrc path) know this
    // storage is sliced into per-stage slots — any pre-init must
    // target only the current slot, never the whole alloc.
    auto markOp =
        builder.create<annotation::MarkOp>(loc, expandedAlloc.getResult());
    markOp->setAttr(hivm::CVPipelinedMultiBufferAttr::name,
                    UnitAttr::get(builder.getContext()));
    expanded = expandedAlloc;
  }
  return success();
}

LogicalResult CVPipelineImpl::expandOutputInitsForPreload(WorkItem *item) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(pipelineLoop.getBody());
  for (auto &[output, expanded] : item->localOutputs) {
    Operation *defining = output.getDefiningOp();
    if (!defining)
      return pipelineLoop->emitWarning(
          "[cv-pipelining] expected work item output to be "
          "result of op");
    Location loc = defining->getLoc();
    bufferization::ToTensorOp toTensor = nullptr;
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(defining)) {
      if (dps.getNumDpsInits() != 1)
        return dps->emitWarning(
            "[cv-pipelining] expected dps op with exactly one "
            "init");
      Value init = dps.getDpsInitOperand(0)->get();
      defining = init.getDefiningOp();
      if (!defining)
        return dps->emitWarning(
            "[cv-pipelining] expected dps init to be result of "
            "op");
      if (isa<tensor::EmptyOp>(defining)) {
        continue;
      }
    }
    toTensor = dyn_cast<bufferization::ToTensorOp>(defining);
    if (!toTensor)
      return defining->emitWarning("[cv-pipelining] expected to_tensor for "
                                   "non-tensor-empty output");
    auto alloc = traceAlloc(toTensor.getMemref());
    if (!alloc)
      return toTensor->emitWarning(
          "[cv-pipelining] expected alloc from toTensor");
    auto origTy = alloc.getMemref().getType();
    if (!origTy.hasStaticShape())
      return alloc->emitWarning(
          "[cv-pipelining] expected temporary buffer to be "
          "static");
    auto memspace = origTy.getMemorySpace();
    auto newType = MemRefType::get(origTy.getShape(), origTy.getElementType(),
                                   MemRefLayoutAttrInterface(), memspace);
    expanded = builder.create<memref::AllocOp>(loc, newType, ValueRange(),
                                               alloc.getAlignmentAttr());
    alloc.replaceAllUsesWith(expanded);
    LLVM_DEBUG(dbgs() << "[Preload expand localOutputs] alloc: "; alloc.dump());
    LLVM_DEBUG(dbgs() << "[Preload expand localOutputs] expanded: ";
               expanded.dump());
    item->ops.remove(alloc);
    alloc->erase();
  }
  return success();
}

/// Create the unrolled newLoop to replace the original pipelineLoop, as well as
/// a jam loop for each work item
LogicalResult CVPipelineImpl::createNewLoops() {
  builder.setInsertionPoint(pipelineLoop);
  Value lb = pipelineLoop.getLowerBound();
  Value ub = pipelineLoop.getUpperBound();
  Value originStep = pipelineLoop.getStep();
  Location loc = pipelineLoop->getLoc();
  Type origTy = originStep.getType();
  Value unrollVal = builder.create<arith::ConstantOp>(
      loc, origTy, builder.getIntegerAttr(origTy, numMultibuffer));
  Value newStep = builder.create<arith::MulIOp>(loc, originStep, unrollVal);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value pipelineIters =
      builder.create<arith::ConstantIndexOp>(loc, numMultibuffer);
  newLoop =
      builder.create<scf::ForOp>(loc, lb, ub, newStep, pipelineLoop.getInits());
  if (newLoop->getNumResults() == 0)
    newLoop.getBody()->getTerminator()->erase();

  globalIRMap.map(pipelineLoop.getRegionIterArgs(),
                  newLoop.getRegionIterArgs());

  // Common values needed to create inner loops
  builder.setInsertionPointToStart(newLoop.getBody());
  IndexType indexTy = builder.getIndexType();
  Value iv = newLoop.getInductionVar();
  Value origIV = pipelineLoop.getInductionVar();
  if (!ub.getType().isIndex()) {
    ub = builder.create<arith::IndexCastOp>(loc, indexTy, ub);
    iv = builder.create<arith::IndexCastOp>(loc, indexTy, iv);
    originStep = builder.create<arith::IndexCastOp>(loc, indexTy, originStep);
  }
  AffineExpr d0, d1, s0, s1;
  MLIRContext *ctx = builder.getContext();
  bindDims(ctx, d0, d1);
  bindSymbols(ctx, s0, s1);
  // Affine map for reconstructing IV, innerIV * originalStep + outerIV
  auto ivMap = AffineMap::get(1, 2, d0 * s0 + s1, ctx);
  // d0: ub, d1: iv, d2: oldStep
  AffineExpr cappedUBExpr = (d0 - d1).ceilDiv(s0);
  Value cappedUB = builder.create<affine::AffineApplyOp>(
      loc, cappedUBExpr, ValueRange({ub, iv, originStep}));
  Value actualUB = builder.create<arith::MinUIOp>(loc, cappedUB, pipelineIters);

  for (auto &item : worklist) {
    // Reset insertion point after we're done with this item
    OpBuilder::InsertionGuard g(builder);
    if (failed(expandOutputInits(item.get())))
      return failure();

    // Create iter arg inits in order: yieldOutputs followed by localOutputs
    SmallVector<Value> inits;
    for (auto [output, opNumber] : item->yieldedOutputs) {
      BlockArgument iterArg = newLoop.getRegionIterArg(opNumber);
      inits.push_back(iterArg);
    }
    for (auto expandedOutputPair : item->localOutputs) {
      Value expandedInit = expandedOutputPair.second;
      if (expandedInit != nullptr && isa<TensorType>(expandedInit.getType()))
        inits.push_back(expandedInit);
    }

    // Actually create the work item loop
    item->forOp = builder.create<scf::ForOp>(loc, c0, actualUB, c1, inits);
    item->forOp->setAttrs(
        {NamedAttribute(builder.getStringAttr(kPipelinedLoopCoreTypeAttrName),
                        TCoreTypeAttr::get(ctx, item->core)),
         NamedAttribute(builder.getStringAttr(kMultibufferUnrollAttrName),
                        builder.getI32IntegerAttr(numMultibuffer))});
    builder.setInsertionPointToStart(item->forOp.getBody());
    Value workItemIV = item->forOp.getInductionVar();
    item->reconstructedIV = builder.create<affine::AffineApplyOp>(
        loc, ivMap, ValueRange{workItemIV, originStep, iv});

    // Remap yield results
    unsigned numResult = 0;
    for (auto [output, opNumber] : item->yieldedOutputs) {
      globalIRMap.map(pipelineLoop.getYieldedValues()[opNumber],
                      item->forOp->getResult(numResult++));
    }

    item->irMap = globalIRMap;
    // Remap the induction variables
    if (origIV.getType() != indexTy) {
      Value ivCast = builder.create<arith::IndexCastOp>(loc, origIV.getType(),
                                                        item->reconstructedIV);
      item->irMap.map(origIV, ivCast);
    } else
      item->irMap.map(origIV, item->reconstructedIV);

    // Remap the yield results within the work item
    unsigned yieldArg = 0;
    for (auto [output, opNumber] : item->yieldedOutputs) {
      item->irMap.map(pipelineLoop.getRegionIterArg(opNumber),
                      item->forOp.getRegionIterArg(yieldArg++));
    }

    // If inits are empty, the default builder creates a yield by default, we
    // don't want that right now so we remove it
    if (inits.empty())
      item->forOp.getBody()->getTerminator()->erase();
  }
  return success();
}

FailureOr<Value> CVPipelineImpl::updateMaskingSubview(OpBuilder &builder,
                                                      Location loc,
                                                      Value expanded,
                                                      OpOperand *initOperand,
                                                      Value iv) {
  auto subview =
      dyn_cast<memref::SubViewOp>(initOperand->get().getDefiningOp());
  if (!subview)
    return Value(nullptr);
  if (!isa<memref::AllocOp>(subview.getSource().getDefiningOp())) {
    subview->emitWarning("[cv-pipelining] expected subview to be from alloc");
    return failure();
  }
  SmallVector<OpFoldResult> offsets, sizes, strides;
  Attribute cst1Attr = builder.getI64IntegerAttr(1);
  if (!iv.getType().isIndex())
    iv = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), iv);
  offsets.push_back(iv);
  offsets.append(subview.getMixedOffsets());
  sizes.push_back(cst1Attr);
  sizes.append(subview.getMixedSizes());
  strides.push_back(cst1Attr);
  strides.append(subview.getMixedStrides());
  int64_t offset;
  auto targetTy = cast<MemRefType>(initOperand->get().getType());
  SmallVector<int64_t> layoutStrides;
  if (getStridesAndOffset(targetTy, layoutStrides, offset).failed()) {
    subview->emitWarning("[cv-pipelining] unexpected memref layout");
    return failure();
  }
  auto layout = StridedLayoutAttr::get(builder.getContext(),
                                       ShapedType::kDynamic, layoutStrides);
  auto finalTy = MemRefType::Builder(targetTy).setLayout(layout);
  auto newSubView = builder.create<memref::SubViewOp>(loc, finalTy, expanded,
                                                      offsets, sizes, strides);
  subview->replaceAllUsesWith(newSubView);
  return Value(newSubView);
}

LogicalResult CVPipelineImpl::migrateOps() {
  DenseSet<Operation *> toEraseMarkOp;
  for (Operation &op : pipelineLoop.getBody()->getOperations()) {
    auto it = opToWorkItemMap.find(&op);
    if (it == opToWorkItemMap.end()) {
      LLVM_DEBUG(dbgs() << "[cv-pipelining] Skipping: "; op.dump());
      continue;
    }
    for (WorkItem *target : it->getSecond()) {
      builder.setInsertionPointToEnd(target->forOp.getBody());
      auto atomicIt = atomicEffectMap.find(&op);
      if (atomicIt != atomicEffectMap.end()) {
        auto &effect = atomicIt->getSecond();
        builder.create<SetAtomicOp>(op.getLoc(), effect.kind, effect.type);
      }
      builder.clone(op, target->irMap);
      if (atomicIt != atomicEffectMap.end()) {
        builder.create<SetAtomicOp>(op.getLoc(), AtomicKind::NONE,
                                    atomicIt->getSecond().type);
      }
    }
  }
  LLVM_DEBUG(dbgs() << "\n\n[migrateOps] After cloning:\n";
             newLoop->getParentOfType<func::FuncOp>()->dump());

  SmallVector<Value> yieldVals;
  for (auto &item : worklist) {
    for (auto [orig, argNo] : item->yieldedOutputs) {
      Value newVal = item->irMap.lookup(orig);
      yieldVals.push_back(newVal);
    }

    // Replace workspace stores in c220
    processWorkspaceOutputs(builder, item.get(), expandedWorkspaceMap_,
                            item->irMap);

    auto argIt =
        item->forOp.getRegionIterArgs().begin() + item->yieldedOutputs.size();
    auto resIt = item->forOp.getResults().begin() + item->yieldedOutputs.size();
    Value iv = item->forOp.getInductionVar();
    for (auto [orig, expanded] : item->localOutputs) {
      Operation *defining = orig.getDefiningOp();
      LLVM_DEBUG(dbgs() << "orig: " << orig << "\n\texpanded: " << expanded
                        << '\n');
      if (auto toTensor =
              dyn_cast_if_present<bufferization::ToTensorOp>(defining)) {
        // Set `defining` to the op that writes to the tensor i.e. the actual
        // defining op for the tensor
        defining = this->outputMemrefMap[toTensor];
      }
      defining = item->irMap.lookup(defining);
      auto dps = dyn_cast_if_present<DestinationStyleOpInterface>(defining);
      if (!dps)
        return pipelineLoop->emitWarning(
            "[cv-pipelining] expected destination passing style op for output");
      builder.setInsertionPoint(dps);
      Location loc = dps->getLoc();

      if (dps.getNumDpsInits() != 1)
        return dps->emitWarning(
            "[cv-pipelining] expected dps op with exactly one init");
      OpOperand *initOperand = dps.getDpsInitOperand(0);
      Value newResult = *resIt;
      if (isa<TensorType>(initOperand->get().getType())) {
        Value extracted =
            createExtractSlice(builder, loc, *argIt, orig.getType(), iv);
        initOperand->set(extracted);
        Value newOutput = dps->getResult(0);
        builder.setInsertionPointAfterValue(newOutput);
        Value yieldVal = createInsertSlice(builder, loc, newOutput, *argIt, iv);
        orig.replaceUsesWithIf(newOutput, [&](OpOperand &use) {
          return item->forOp->isAncestor(use.getOwner());
        });
        resIt++;
        argIt++;
        yieldVals.push_back(yieldVal);
      } else if (auto targetTy =
                     dyn_cast<MemRefType>(initOperand->get().getType())) {
        Value internalDef = item->irMap.lookup(orig);
        // If there are masking subviews, update those first
        FailureOr<Value> updatedSubviewOr =
            updateMaskingSubview(builder, loc, expanded, initOperand, iv);
        if (failed(updatedSubviewOr))
          return failure();
        Value updatedSubview = *updatedSubviewOr;
        // Then replace the toTensor operand if it is not updated
        auto innerToTensor = dyn_cast_if_present<bufferization::ToTensorOp>(
            internalDef.getDefiningOp());
        if (!innerToTensor)
          return dps->emitWarning("[cv-pipelining] expected memref outputs to "
                                  "be passed as tensors");
        OpOperand *memrefOperand = &innerToTensor.getMemrefMutable();
        if (memrefOperand->get() != updatedSubview) {
          Value toTensorSubview = nullptr;
          builder.setInsertionPointToStart(item->forOp.getBody());
          if (!updatedSubview) {
            toTensorSubview = createSubview(builder, loc, expanded,
                                            initOperand->get().getType(), iv);
            if (!toTensorSubview)
              return failure();
            initOperand->set(toTensorSubview);
          } else {
            toTensorSubview = createSubview(builder, loc, expanded,
                                            memrefOperand->get().getType(), iv);
            if (!toTensorSubview)
              return failure();
          }
          memrefOperand->set(toTensorSubview);
        }
        builder.setInsertionPointAfter(item->forOp);
        newResult = createToTensor(builder, loc, expanded);
        if (!newResult)
          return failure();
      } else
        return dps->emitWarning("[cv-pipelining] unexpected output type that "
                                "is not tensor or memref");

      // Update outside users
      LLVM_DEBUG(dbgs() << "[cv-pipelining] Updating user of "; orig.dump());
      SmallVector<OpOperand *> toReplace;
      for (OpOperand &use : orig.getUses()) {
        Operation *owner = use.getOwner();
        LLVM_DEBUG(dbgs().indent(4) << *owner << '\n');
        if (pipelineLoop->isAncestor(owner) || item->forOp->isAncestor(owner)) {
          LLVM_DEBUG(dbgs().indent(8) << "Not in user loop, skipped\n");
          continue;
        }
        toReplace.push_back(&use);
      }
      for (OpOperand *use : toReplace) {
        Operation *owner = use->getOwner();
        // At this point the loop should only contain the pipeline loops we
        // created
        Operation *ownerLoop = getContainedParent(newLoop, owner);
        Value userIV = cast<scf::ForOp>(ownerLoop).getInductionVar();
        builder.setInsertionPoint(owner);
        Value newUse = createExtractSlice(builder, owner->getLoc(), newResult,
                                          use->get().getType(), userIV);
        use->set(newUse);
      }
    }

    builder.setInsertionPointToEnd(item->forOp.getBody());
    builder.create<scf::YieldOp>(item->forOp->getLoc(), yieldVals);
    yieldVals.clear();
  }

  // this needed to be done after all the ops have been migrated
  // into their own loops: update user of local outputs to extract slices of
  // the for output in c220
  processOutputUsers(builder, worklist, opToWorkItemMap, expandedWorkspaceMap_,
                     newLoop);

  builder.setInsertionPointToEnd(newLoop.getBody());
  if (trailingAtomicEffect) {
    builder.create<SetAtomicOp>(pipelineLoop->getLoc(),
                                trailingAtomicEffect->kind,
                                trailingAtomicEffect->type);
  }
  builder.clone(*pipelineLoop.getBody()->getTerminator(), globalIRMap);
  return success();
}

/// Migrate all ops from block into scope for each work item
LogicalResult CVPipelineImpl::migrateOpsForPreload(OpBuilder &builder) {
  for (auto &item : worklist) {
    LLVM_DEBUG(dbgs() << "\n\nMigrating ops for work item #" << item->id
                      << "\n");
    for (auto &[origOutput, expanded] : item->localOutputs) {
      if (localOuputsToRedurnRes.count(origOutput) == 0)
        continue;
      LLVM_DEBUG(dbgs() << "\n\nMigrating localOutputs #" << origOutput
                        << "\n");

      Value newOutput = globalIRMap.lookup(origOutput);
      Value returnOutput = localOuputsToRedurnRes[origOutput];
      for (OpOperand &operand : newOutput.getUses()) {
        operand.set(returnOutput);
      }
    }
    for (Operation *output : item->workspaceOutputs) {
      auto dpsOp = cast<DestinationStyleOpInterface>(output);
      Value wsAlloc = getAllocWorkspace(dpsOp.getDpsInitOperand(0)->get());
      if (!wsAlloc) {
        continue;
      }
      LLVM_DEBUG(dbgs() << "\n\nMigrating workspaceOutputs #" << wsAlloc
                        << "\n");

      Location loc = output->getLoc();
      auto newWsAlloc = expandedWorkspaceMap_[wsAlloc];
      Operation *storeLikeOp = globalIRMap.lookupOrDefault(output);
      builder.setInsertionPoint(storeLikeOp);
      auto sliceIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value newDst =
          createWorkspaceSubview(builder, loc, newWsAlloc, sliceIdx, true);

      if (auto storeOp = dyn_cast<StoreOp>(storeLikeOp)) {
        builder.create<StoreOp>(loc, TypeRange{}, storeOp.getSrc(), newDst);
      } else if (auto fixpipe = dyn_cast<FixpipeOp>(storeLikeOp)) {
        builder.create<FixpipeOp>(
            loc, TypeRange{}, fixpipe.getSrc(), newDst,
            fixpipe.getDmaModeAttr(), fixpipe.getDualDstModeAttr(),
            fixpipe.getSubBlockIdxAttr(), fixpipe.getPreQuantAttr(),
            fixpipe.getPreReluAttr(),
            fixpipe.getChannelSplitAttr());
      }

      builder.setInsertionPointAfter(item->scopeOp);
      auto workspaceOp = builder.create<bufferization::ToTensorOp>(
          loc, newWsAlloc, /*restrict*/ true);
      for (OpOperand &operand : storeLikeOp->getUses()) {
        Operation *userOp = operand.getOwner();
        if (auto loadOp = dyn_cast<LoadOp>(userOp)) {
          builder.setInsertionPoint(loadOp);
          auto sliceIdx1 = builder.create<arith::ConstantIndexOp>(loc, 0);
          Value sliceOp = createExtractSlice(
              builder, loc, workspaceOp, operand.get().getType(), sliceIdx1);
          createAttrForPreloadWS(builder, sliceOp);
          operand.set(sliceOp);
        }
      }
      LLVM_DEBUG(dbgs() << "\n\nAfter migrate:\n";
                 pipelineLoop->getParentOfType<func::FuncOp>()->dump());
      storeLikeOp->erase();
    }
  }
  return success();
}

LogicalResult CVPipelineImpl::createNewLoopsForPreloadWithScopes() {
  int32_t preloadNum = static_cast<int32_t>(worklist.size()) - 1;
  for (auto &item : worklist) {
    // Reset insertion point after we're done with this item
    OpBuilder::InsertionGuard g(builder);
    if (failed(expandOutputInitsForPreload(item.get())))
      return failure();

    LLVM_DEBUG(dbgs() << "Creating scope for work item #" << item->id << '\n');
    scf::ForOp parentFor = pipelineLoop;

    // collect return values
    SmallVector<Value> returnTensors{};
    for (auto &localOutput : item->localOutputs) {
      returnTensors.push_back(localOutput.first);
    }
    for (auto &yieldedOutput : item->yieldedOutputs) {
      returnTensors.push_back(yieldedOutput.first);
    }
    if (returnTensors.empty() && item->workspaceOutputs.empty())
      return success();

    builder.setInsertionPoint(parentFor.getBody()->getTerminator());
    Location loc = pipelineLoop->getLoc();

    auto newScopeOp =
        builder.create<scope::ScopeOp>(loc, TypeRange(returnTensors));
    newScopeOp.setNoInline(true);
    newScopeOp->setAttr(kPipelinedLoopCoreTypeAttrName,
                        TCoreTypeAttr::get(builder.getContext(), item->core));
    newScopeOp->setAttr(
        hivm::PreloadNumAttr::name,
        IntegerAttr::get(IntegerType::get(newScopeOp->getContext(), 32),
                         preloadNum));
    // TODO: add a new pass to analyze max preload num
    newScopeOp->setAttr(
        hivm::MaxPreloadNumAttr::name,
        IntegerAttr::get(IntegerType::get(newScopeOp->getContext(), 32),
                         worklist.size()));

    Region &region = newScopeOp.getRegion();
    Block *bodyBlock = builder.createBlock(&region);
    builder.setInsertionPointToEnd(bodyBlock);
    IRMapping scopeMap(globalIRMap);

    LLVM_DEBUG(dbgs() << "Created scope for work item #" << item->id << " with "
                      << returnTensors.size() << " results\n");
    for (Operation &op : parentFor.getBody()->getOperations()) {
      if (!item->ops.contains(&op))
        continue;

      builder.clone(op, scopeMap);
      toErase.insert(&op);
    }

    globalIRMap = scopeMap;
    builder.setInsertionPointToEnd(bodyBlock);
    SmallVector<Value> newReturnTensors;

    for (auto returnTensor : returnTensors) {
      if (scopeMap.contains(returnTensor)) {
        Value newReturnTensor = scopeMap.lookup(returnTensor);
        newReturnTensors.push_back(newReturnTensor);
      } else {
        newReturnTensors.push_back(returnTensor);
      }
    }
    builder.create<scope::ReturnOp>(loc, ValueRange(newReturnTensors));

    size_t resultIdx = 0;
    for (auto &localOutput : item->localOutputs) {
      Value returnTensor = localOutput.first;
      Value scopeResult = newScopeOp->getResult(resultIdx++);
      localOuputsToRedurnRes[returnTensor] = scopeResult;
      globalIRMap.map(returnTensor, scopeResult);
    }

    for (auto &yieldedOutput : item->yieldedOutputs) {
      Value returnTensor = yieldedOutput.first;
      Value scopeResult = newScopeOp->getResult(resultIdx++);

      pipelineLoop.getBody()->getTerminator()->replaceUsesOfWith(returnTensor,
                                                                 scopeResult);
      globalIRMap.map(returnTensor, scopeResult);
    }

    item->scopeOp = newScopeOp;
    preloadNum--;
  }

  return success();
}

LogicalResult CVPipelineImpl::markScopesForPreload() {
  toErase.clear();

  if (failed(createNewLoopsForPreloadWithScopes()))
    return failure();

  if (failed(migrateOpsForPreload(builder))) {
    revert();
    return failure();
  }

  LLVM_DEBUG({
    for (auto item : worklist) {
      dbgs() << "after createNewLoopsForPreloadWithScopes WorkItem #"
             << item->id << ":---------------\n";
      item->scopeOp->dump();
      if (!item->localOutputs.empty())
        dbgs() << "\tLocal outputs:\n";
      for (auto p : item->localOutputs) {
        Value output = p.first;
        dbgs().indent(4) << output << '\n';
      }
      if (!item->yieldedOutputs.empty())
        dbgs() << "\tYield outputs:\n";
      for (auto [output, number] : item->yieldedOutputs)
        dbgs().indent(4) << output << " at " << number << '\n';
    }
  });

  LLVM_DEBUG(dbgs() << "\n\nAfter clone before erase:\n";
             pipelineLoop->getParentOfType<func::FuncOp>()->dump());

  LLVM_DEBUG(dbgs() << "toErase.size() scope for work item all"
                    << toErase.size() << " results\n");
  Operation *eraseOp = *toErase.begin();
  while (!toErase.empty()) {
    if (eraseOp == nullptr)
      eraseOp = *toErase.begin();
    auto usrBegin = eraseOp->user_begin();
    if (usrBegin == eraseOp->user_end()) {
      LLVM_DEBUG({
        dbgs() << "eraseOp:";
        eraseOp->dump();
        dbgs() << ":---------------\n";
      });
      eraseOp->erase();
      toErase.erase(eraseOp);
      eraseOp = nullptr;
      continue;
    }
    Operation *usrOp = *usrBegin;
    Operation *usrParent = usrOp->getParentOp();
    while (!isa<func::FuncOp>(usrParent)) {
      if (toErase.contains(usrParent)) {
        usrOp = usrParent;
        break;
      }
      if (!usrParent)
        return eraseOp->emitWarning(
            "[cv-pipelining] reached null parent while tracing users");
      usrParent = usrParent->getParentOp();
    }

    if (!toErase.contains(usrOp)) {
      LLVM_DEBUG(dbgs() << "func" << "\n\nDef: " << *eraseOp
                        << "\nUser: " << *usrOp << '\n');
      return usrOp->emitWarning(
          "[cv-pipelining] cannot erase user of pipelined op, aborting "
          "pipelining pass");
    }
    eraseOp = usrOp;
  }
  LLVM_DEBUG(dbgs() << "\n\nAfter everything:\n";
             newLoop->getParentOfType<func::FuncOp>()->dump());
  return success();
}

void CVPipelineImpl::revert() {
  if (!newLoop)
    return;
  for (auto [newRes, oldRes] :
       llvm::zip(newLoop.getResults(), pipelineLoop.getResults()))
    newRes.replaceAllUsesWith(oldRes);
  newLoop->erase();
}

LogicalResult CVPipelineImpl::run() {
  collectAtomicEffects();
  if (createWorkItems().failed()) {
    revert();
    return failure();
  }
  if (failed(absorbMergerOpsIntoWorkItems())) {
    revert();
    return failure();
  }
  markOutputs();
  LLVM_DEBUG({
    for (auto item : worklist) {
      dbgs() << "WorkItem #" << item->id << ":\n";
      for (Operation *op : item->ops)
        op->dump();
      if (!item->localOutputs.empty())
        dbgs() << "\tLocal outputs:\n";
      for (auto p : item->localOutputs) {
        Value output = p.first;
        dbgs().indent(4) << output << '\n';
      }
      if (!item->yieldedOutputs.empty())
        dbgs() << "\tYield outputs:\n";
      for (auto [output, number] : item->yieldedOutputs)
        dbgs().indent(4) << output << " at " << number << '\n';
      if (!item->workspaceOutputs.empty())
        dbgs() << "\tWorkspace outputs:\n";
      for (auto output : item->workspaceOutputs) {
        dbgs().indent(4) << *output << '\n';
      }
    }
  });

  expandWorkspace(builder);

  if (enableSkewMode) {
    return markScopesForPreload();
  }

  if (failed(createNewLoops())) {
    revert();
    return failure();
  }

  LLVM_DEBUG(dbgs() << "\n\n[migrateOps] After createNewLoops:\n";
             newLoop->getParentOfType<func::FuncOp>()->dump());

  if (failed(migrateOps())) {
    revert();
    return failure();
  }

  pipelineLoop.replaceAllUsesWith(newLoop.getResults());

  LLVM_DEBUG(dbgs() << "\n\nAfter everything:\n";
             newLoop->getParentOfType<func::FuncOp>()->dump());
  if (failed(newLoop.verify())) {
    revert();
    return failure();
  }

  SmallVector<annotation::MarkOp> markOps;
  newLoop.walk([&](annotation::MarkOp markOp) {
    markOps.push_back(markOp);
    return WalkResult::advance();
  });
  if (failed(filterMarkOpsValuesInsideLoop(markOps, newLoop))) {
    revert();
    return failure();
  }

  pipelineLoop->erase();
  return success();
}

void CVPipeliningPass::runOnOperation() {
  auto mod = dyn_cast<ModuleOp>(getOperation());
  if ((mod && !hacc::utils::isRegBasedArch(mod))) {
    func::FuncOp func = getOperation();
    DenseSet<scf::ForOp> pipelinedLoops;

    if (this->setDepthInUnrollMode == 1 || this->setDepthInUnrollMode == 0)
      return;

    func->walk<WalkOrder::PreOrder>([&pipelinedLoops, this](scf::ForOp loop) {
      auto parentLoop = loop->getParentOfType<scf::ForOp>();

      while (parentLoop) {
        if (pipelinedLoops.contains(parentLoop))
          return;
        parentLoop = parentLoop->getParentOfType<scf::ForOp>();
      }

      CVPipelineImpl impl(loop, this->setDepthInUnrollMode, this->enableSkewMode,
                          this->enableLazyLoading);
      if (impl.run().succeeded())
        pipelinedLoops.insert(loop);
    });

    removeWorkspaceMultiBufferMarks(func);
  } else {
      // First find loop to operate on
    func::FuncOp func = getOperation();
    SmallVector<scf::ForOp> pipelineCandidates;

    // Disabled via options
    if (this->pipelineDepth == 1 || this->pipelineDepth == 0)
      return;

    // Disable CVP once batchmatmul is found
    SmallVector<func::FuncOp> funcOps{func};
    if (hasBatchMatmulLoopInAicFuncs(funcOps))
      return;

    // We want to work on the innermost loop first, so post order walk
    func->walk<WalkOrder::PostOrder>([&pipelineCandidates](scf::ForOp loop) {
      pipelineCandidates.push_back(loop);
    });

    DenseSet<scf::ForOp> pipelinedNest;
    for (auto loop : pipelineCandidates) {
      // Don't attempt pipeline if nested loop succeeded already
      if (pipelinedNest.contains(loop))
        continue;

      auto parentLoop = loop->getParentOfType<scf::ForOp>();
      CVPipelineRegbaseImpl impl(loop, this->pipelineDepth, this->pipelineMode,
                          this->enableLazyLoading);

      // Mark all parent loops to not attempt pipelining to save compile time
      if (impl.run().succeeded())
        while (parentLoop) {
          pipelinedNest.insert(parentLoop);
          parentLoop = parentLoop->getParentOfType<scf::ForOp>();
        }
    }
  }
}

std::unique_ptr<Pass>
hivm::createCVPipeliningPass(const CVPipeliningOptions &options) {
  return std::make_unique<CVPipeliningPass>(options);
}
} // namespace mlir

//===- CVPipelining.cpp --- Pipelining pass for mix-cv ops ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "cv-pipelining"

using llvm::dbgs;
namespace mlir {
using namespace hivm;

#define GEN_PASS_DEF_CVPIPELINING
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

using hivm::detail::queryCoreTypeHelper;

static constexpr llvm::StringLiteral CubeOnlyAttrName = "pipeline.cubeonly";
static constexpr llvm::StringLiteral VecOnlyAttrName = "pipeline.veconly";
namespace {

struct WorkspaceAllocParams {
  unsigned multibuffer;
  annotation::MarkOp marker;
  bufferization::ToTensorOp toTensor;
};

struct WorkItem {
  // Values that are referred by other work items later will be stored in this
  // list. Everything here requires the tensor types to be expanded by
  // Multibuffer times, e.g. <16xf16> into <2x16xf16>
  SmallVector<std::pair<Value, Value>> localOutputs;

  DenseSet<Operation *> ops;

  // Values that are yielded in the parent for loop
  SmallVector<std::pair<Value, unsigned>> yieldedOutputs;

  // Vector or Cube, other types shouldn't end up in here
  TCoreType core;

  // After unrolling the parent for loop, the upper bound for "reroll"ed loops
  // are computed and inserted here. Created in "unrollOuterLoop" Value
  // upperBound; The for op corresponding to the multibuffering, constructed in
  // "constructPipelineLoop"
  scf::ForOp forOp;

  IRMapping irMap;

  // Reconstructed original induction variable
  Value reconstructedIV;
#ifndef NDEBUG
  int id;
#endif
};

struct CVPipelineImpl {
  CVPipelineImpl(LoopLikeOpInterface loop, int multibuffer)
      : pipelineLoop(loop), builder(loop->getContext()),
        numMultibuffer(multibuffer),
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

  void traceMemrefSubnet(Operation *start,
                         SmallVector<Operation *> &workingStack);

  void markOutputs();

  void expandOutputInits(WorkItem *item);

  void createNewLoops();

  void mapOpToItem(Operation *op, WorkItem *item);

  void migrateOps();

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

  // Mapping from the original pipelineLoop to the newLoop to guide the cloning
  // process
  IRMapping globalIRMap;
};

struct CVPipeliningPass
    : public ::mlir::impl::CVPipeliningBase<CVPipeliningPass> {
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

static int getMultibufferCount(annotation::MarkOp marker) {
  auto multibufferAttr = llvm::cast_if_present<IntegerAttr>(
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
  assert(blkArg && "Expecting non-OpResult value to be block argument");
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
    op->emitWarning("CV-Pipelining: Unsupported regioned op");
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

static tensor::InsertSliceOp createInsertSlice(OpBuilder &builder, Location loc,
                                               Value src, Value into,
                                               Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  auto originalType = cast<TensorType>(src.getType());
  SmallVector<OpFoldResult> offsets, sizes, strides;
  offsets.push_back(iv);
  offsets.append(originalType.getRank(), const0);

  // Set up the sizes
  sizes.push_back(const1);
  for (int i = 0; i < originalType.getRank(); ++i) {
    if (originalType.isDynamicDim(i))
      sizes.push_back(builder.createOrFold<tensor::DimOp>(loc, src, i));
    else
      sizes.push_back(builder.getIndexAttr(originalType.getDimSize(i)));
  }

  // And strides should be all ones
  strides.append(originalType.getRank() + 1, const1);

  return builder.create<tensor::InsertSliceOp>(loc, src, into, offsets, sizes,
                                               strides);
}

static Value createExtractSlice(OpBuilder &builder, Location loc, Value from,
                                Type to, Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  auto newType = cast<TensorType>(from.getType());

  // Set up offsets
  offsets.push_back(iv);
  offsets.append(newType.getRank() - 1, const0);
  // Set up sizes
  sizes.push_back(const1);
  for (int i = 1; i < newType.getRank(); ++i) {
    if (newType.isDynamicDim(i))
      sizes.push_back(builder.createOrFold<tensor::DimOp>(loc, from, i));
    else
      sizes.push_back(builder.getIndexAttr(newType.getDimSize(i)));
  }

  // ... and strides
  strides.append(newType.getRank(), const1);
  auto finalTy = cast<RankedTensorType>(to);
  return builder.create<tensor::ExtractSliceOp>(loc, finalTy, from, offsets,
                                                sizes, strides);
}

static Value createSubview(OpBuilder &builder, Location loc, Value from,
                           Type to, Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  auto targetTy = cast<MemRefType>(to);
  offsets.push_back(iv);
  offsets.append(targetTy.getRank(), const0);
  sizes.push_back(const1);
  for (int64_t dim : targetTy.getShape()) {
    assert(!ShapedType::isDynamic(dim));
    sizes.push_back(builder.getIndexAttr(dim));
  }
  strides.append(targetTy.getRank() + 1, const1);
  int64_t offset;
  SmallVector<int64_t> layoutStrides;
  if (getStridesAndOffset(targetTy, layoutStrides, offset).failed())
    llvm_unreachable("Unexpected memref layout");
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
void CVPipelineImpl::traceMemrefSubnet(Operation *start,
                                       SmallVector<Operation *> &workingStack) {
  // When we get here, `start` should be one of three ops:
  // 1. Fixpipe
  // 2. Copy
  // 3. Load
  // All of which have the `outs` as second operand
  DestinationStyleOpInterface writer = nullptr;
  Value targetOperand = start->getOperand(1);
  if (isa<TensorType>(targetOperand.getType()))
    writer = cast<DestinationStyleOpInterface>(start);

  Operation *defining = targetOperand.getDefiningOp();
  // First trace all the way up
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
      llvm_unreachable("Unexpected memref op in chain");
  }
  SmallVector<Operation *> userTraceStack = {start};
  bufferization::ToTensorOp toTensor = nullptr;

  while (!userTraceStack.empty()) {
    Operation *def = userTraceStack.pop_back_val();
    workingStack.push_back(def);
    for (Operation *usr : def->getUsers()) {
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(usr)) {
        assert(!writer && "Expecting only one op writing to a defined memref");
        writer = dps;
        continue;
      }
      if (auto tt = dyn_cast<bufferization::ToTensorOp>(usr)) {
        assert(!toTensor && "Expecting only one toTensor for a defined memref");
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
    llvm_unreachable("Expecting toTensor to have dps op to write to it");
  }
  if (toTensor && writer)
    outputMemrefMap[toTensor] = writer;
}

// Given memref value, populate users with all operations that uses any aliasing
// memrefs as `memrefVal`
static void memrefDFS(Value memrefVal, SmallVector<Operation *> &users) {
  SmallVector<Operation *> traceStack;
  DenseSet<Operation *> visited;
  Value rootVal = traceValueDef(memrefVal);
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

static void traceOperands(Value operand, scf::ForOp pipelineLoop,
                          WorkItem *item,
                          SmallVector<Operation *> &workingStack) {
  if (!operand)
    return;
  Operation *defining = getContainedParent(pipelineLoop, operand);
  if (item->ops.contains(defining))
    return;
  if (!defining) {
    auto iterArg = dyn_cast<BlockArgument>(operand);
    assert(iterArg && "Expecting non-op-defined value to be block argument");
    for (Operation *usr : iterArg.getUsers()) {
      if (isa<DebugOp>(usr) && !item->ops.contains(usr) &&
          usr->getParentOp() == pipelineLoop)
        workingStack.push_back(usr);
    }
    if (iterArg.getOwner()->getParentOp() != pipelineLoop ||
        iterArg.getArgNumber() == 0)
      return;
    // Need to pull defining op into this work item to guarentee safety,
    // should already be guarenteed by extractAvailableOps
    if (isa<TensorType>(operand.getType()))
      return;
    Value yieldVal = pipelineLoop.getTiedLoopYieldedValue(iterArg)->get();
    defining = yieldVal.getDefiningOp();
    if (!defining || defining->getParentOp() != pipelineLoop)
      return;
  }
  if (defining->getParentOp() != pipelineLoop)
    return;
  // If defining is a memref, then trace everything that also uses that memref.
  if (isa<MemRefType>(operand.getType()))
    memrefDFS(operand, workingStack);
  // To tensor ops are handled as a part of the memref operand for
  // load/fixpipe/copy
  if (!item->ops.contains(defining))
    workingStack.push_back(defining);
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
        if (!item->ops.contains(op))
          continue;
      } else if (!isa<LoadOp>(op))
        // Load ops are pulled into their consuming work items, apart from that,
        // if we get here that means we depend on an op that have not satisfied
        // their dependency
        return op->emitWarning("Cannot pipeline op due to dependency");
    }
    // Other than core ops, we can skip them if we already inserted them into
    // this work item
    else if (item->ops.contains(op))
      continue;
    if (op->getParentOp() != pipelineLoop || isa<scf::YieldOp>(op) ||
        (isa<bufferization::ToTensorOp>(op) && opToWorkItemMap.contains(op)))
      continue;
    LLVM_DEBUG(dbgs() << "Inserting \t"; op->dump());
    mapOpToItem(op, item);
    toBePipelined.erase(op);
    for (Operation *usr : op->getUsers())
      if (isa<annotation::MarkOp, DebugOp>(usr))
        mapOpToItem(usr, item);

    // Handle load/fixpipe/copy dealing with memref memref
    if (isa<LoadOp, FixpipeOp>(op) || isCrossCoreCopy(op)) {
      traceMemrefSubnet(op, workingStack);
      // Deal with the ins operand
      traceOperands(op->getOperand(0), pipelineLoop, item, workingStack);
      if (auto load = dyn_cast<LoadOp>(op)) {
        traceOperands(load.getInitCondition(), pipelineLoop, item,
                      workingStack);
        traceOperands(load.getLeftPaddingNum(), pipelineLoop, item,
                      workingStack);
        traceOperands(load.getRightPaddingNum(), pipelineLoop, item,
                      workingStack);
        traceOperands(load.getPadValue(), pipelineLoop, item, workingStack);
      }
      continue;
    }

    // Handle nested ops as well
    op->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands())
        traceOperands(operand, pipelineLoop, item, workingStack);
    });
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

    assert(maybeCore == TCoreType::VECTOR || maybeCore == TCoreType::CUBE);
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

/// Split loop based on separator ops into individual work items
LogicalResult CVPipelineImpl::createWorkItems() {
  int multibuffer = numMultibuffer > 1 ? numMultibuffer : 2;
  Block *blk = pipelineLoop.getBody();
  for (Operation &op : blk->getOperations()) {
    if (isCoreOp(&op))
      toBePipelined.insert(&op);
    if (isa<FixpipeOp, StoreOp>(&op) || isCrossCoreCopy(&op))
      separators.push_back(&op);
    else if (auto mark = dyn_cast<annotation::MarkOp>(&op)) {
      // Compile option override
      if (numMultibuffer != -1)
        continue;
      int markMultibuffer = getMultibufferCount(mark);
      if (markMultibuffer == -1)
        continue;
      if (multibuffer < 2)
        multibuffer = markMultibuffer;
      else if (multibuffer != markMultibuffer) {
        // Conflict in multibuffer count, use smallest one
        multibuffer = std::min(multibuffer, markMultibuffer);
      }
    } else if (illegalRegionedOp(&op) || isa<SetAtomicOp>(&op)) {
      // Illegal op, do nothing and return
      return failure();
    }
  } // end for op
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

  // Set up dependencies
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
    // Alternate the core type being extracted.
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
    return pipelineLoop->emitWarning(
        "cannot pipeline loop due to loop carried dependencies");
  }
  if (worklist.size() < 2)
    return failure();
  return success();
}

/// Check ops in each work item to see if they will be used by other WorkItems
/// (localOutputs) or yielded into the next iteration (yieldedOutputs)
void CVPipelineImpl::markOutputs() {
  for (const auto &item : worklist) {
    for (Operation *op : item->ops) {
      if (isa<tensor::EmptyOp>(op))
        continue;
      for (Value result : op->getResults()) {
        if (yieldedVals.contains(result)) {
          unsigned opNumber = static_cast<unsigned>(std::distance(
              yieldedVals.begin(), llvm::find(yieldedVals, result)));
          item->yieldedOutputs.push_back(std::make_pair(result, opNumber));
          continue;
        }
        // For local outputs, we only care about tensor values, since
        // others will be duplicated
        if (!isa<TensorType>(result.getType()))
          continue;

        for (Operation *usr : result.getUsers()) {
          if (opToWorkItemMap.contains(usr) && !item->ops.contains(usr)) {
            item->localOutputs.push_back(std::make_pair(result, nullptr));
            break;
          } // End loop over result.users
        } // End loop over op->results
      } // End loop over item->ops
    } // End loop over worklist
  }
}

static Value createToTensor(OpBuilder &builder, Location loc, Value src) {
  auto memref = dyn_cast<MemRefType>(src.getType());
  assert(memref && "Expecting creating toTensor from MemRefType");
  if (memref.getMemorySpace()) {
    auto newMemRef = MemRefType::get(memref.getShape(), memref.getElementType(),
                                     memref.getLayout());
    src = builder.create<memref::MemorySpaceCastOp>(loc, newMemRef, src);
  }

  return builder.create<bufferization::ToTensorOp>(loc, src, /*restrict*/ true,
                                                   /*writable*/ true);
}

/// Expand the localOutputs of each work item by number of multibuffer/pipeline
/// stages.
void CVPipelineImpl::expandOutputInits(WorkItem *item) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(newLoop.getBody());
  for (auto &[output, expanded] : item->localOutputs) {
    Operation *defining = output.getDefiningOp();
    assert(defining && "Expecting work item outputs to be result of op");
    Location loc = defining->getLoc();
    SmallVector<int64_t> newShape({numMultibuffer});
    bufferization::ToTensorOp toTensor = nullptr;
    // We take the init and expand it
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(defining)) {
      assert(dps.getNumDpsInits() == 1);
      Value init = dps.getDpsInitOperand(0)->get();
      defining = init.getDefiningOp();
      assert(defining &&
             "Expecting init operand of dps op to be a result of op");
      if (isa<tensor::EmptyOp>(defining)) {
        auto origTy = dyn_cast<TensorType>(init.getType());
        assert(origTy && "Expecting output to be tensor type");
        auto shapeArr = origTy.getShape();
        newShape.append(shapeArr.begin(), shapeArr.end());
        // TODO: Add support for dynamic dims
        auto newType = RankedTensorType::get(newShape, origTy.getElementType());
        expanded = builder.create<tensor::EmptyOp>(loc, newType, ValueRange());
        continue;
      }
    }
    toTensor = dyn_cast<bufferization::ToTensorOp>(defining);
    assert(toTensor && "Expecting to_tensor for non-tensor-empty outputs");
    // Find the alloc
    auto alloc = traceAlloc(toTensor.getMemref());
    assert(alloc && "Expecting alloc from toTensor");
    auto origTy = alloc.getMemref().getType();
    assert(origTy.hasStaticShape() &&
           "Expecting all temporary buffers to be static");
    newShape.append(origTy.getShape().begin(), origTy.getShape().end());
    auto memspace = origTy.getMemorySpace();
    auto newType = MemRefType::get(newShape, origTy.getElementType(),
                                   MemRefLayoutAttrInterface(), memspace);
    expanded = builder.create<memref::AllocOp>(loc, newType, ValueRange(),
                                               alloc.getAlignmentAttr());
  }
}

/// Create the unrolled newLoop to replace the original pipelineLoop, as well as
/// a jam loop for each work item
void CVPipelineImpl::createNewLoops() {
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
    expandOutputInits(item.get());

    // Create iter arg inits in order: yieldOutputs followed by localOutputs
    SmallVector<Value> inits;
    for (auto [output, opNumber] : item->yieldedOutputs) {
      BlockArgument iterArg = newLoop.getRegionIterArg(opNumber);
      inits.push_back(iterArg);
    }
    for (auto expandedOutputPair : item->localOutputs) {
      Value expandedInit = expandedOutputPair.second;
      if (isa<TensorType>(expandedInit.getType()))
        inits.push_back(expandedInit);
    }

    // Actually create the work item loop
    item->forOp = builder.create<scf::ForOp>(loc, c0, actualUB, c1, inits);
    item->forOp->setAttrs(
        {NamedAttribute(kPipelinedLoopCoreTypeAttrName,
                        TCoreTypeAttr::get(ctx, item->core)),
         NamedAttribute(kMultibufferUnrollAttrName,
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
}

static Value updateMaskingSubview(OpBuilder &builder, Location loc,
                                  Value expanded, OpOperand *initOperand,
                                  Value iv) {
  auto subview =
      dyn_cast<memref::SubViewOp>(initOperand->get().getDefiningOp());
  if (!subview)
    return nullptr;
  assert(isa<memref::AllocOp>(subview.getSource().getDefiningOp()) &&
         "Expecting subview at this stage to be from alloc");
  SmallVector<OpFoldResult> offsets, sizes, strides;
  Attribute cst1Attr = builder.getI64IntegerAttr(1);
  offsets.push_back(iv);
  offsets.append(subview.getMixedOffsets());
  sizes.push_back(cst1Attr);
  sizes.append(subview.getMixedSizes());
  strides.push_back(cst1Attr);
  strides.append(subview.getMixedStrides());
  // Set up dynamic stride
  int64_t offset;
  auto targetTy = cast<MemRefType>(initOperand->get().getType());
  SmallVector<int64_t> layoutStrides;
  if (getStridesAndOffset(targetTy, layoutStrides, offset).failed())
    llvm_unreachable("Unexpected memref layout");
  auto layout = StridedLayoutAttr::get(builder.getContext(),
                                       ShapedType::kDynamic, layoutStrides);
  auto finalTy = MemRefType::Builder(targetTy).setLayout(layout);
  auto newSubView = builder.create<memref::SubViewOp>(loc, finalTy, expanded,
                                                      offsets, sizes, strides);
  subview->replaceAllUsesWith(newSubView);
  return newSubView;
}

/// Actually migrate/clone each op for each work item
void CVPipelineImpl::migrateOps() {
  for (Operation &op : pipelineLoop.getBody()->getOperations()) {
    auto it = opToWorkItemMap.find(&op);
    if (it == opToWorkItemMap.end()) {
      LLVM_DEBUG(dbgs() << "[migrateOps] Skipping: "; op.dump());
      continue;
    }
    for (WorkItem *target : it->getSecond()) {
      builder.setInsertionPointToEnd(target->forOp.getBody());
      builder.clone(op, target->irMap);
    }
  }
  LLVM_DEBUG(dbgs() << "\n\n[migrateOps] After cloning:\n";
             newLoop->getParentOfType<func::FuncOp>()->dump());

  // Update outputs
  SmallVector<Value> yieldVals;
  for (auto &item : worklist) {
    // Yield outputs come before the local outputs in yield/iter args
    for (auto [orig, argNo] : item->yieldedOutputs) {
      Value newVal = item->irMap.lookup(orig);
      yieldVals.push_back(newVal);
    }

    auto argIt =
        item->forOp.getRegionIterArgs().begin() + item->yieldedOutputs.size();
    auto resIt = item->forOp.getResults().begin() + item->yieldedOutputs.size();
    Value iv = item->forOp.getInductionVar();
    for (auto [orig, expanded] : item->localOutputs) {
      Operation *defining = orig.getDefiningOp();
      assert(defining && "Expecting defining op for output");
      LLVM_DEBUG(dbgs() << "orig: " << orig << "\n\texpanded: " << expanded
                        << '\n');
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(defining)) {
        // Set `defining` to the op that writes to the tensor i.e. the actual
        // defining op for the tensor
        defining = this->outputMemrefMap[toTensor];
      }
      defining = item->irMap.lookup(defining);
      auto dps = dyn_cast<DestinationStyleOpInterface>(defining);
      assert(dps && "expecting destination passing style op for output");
      builder.setInsertionPoint(dps);
      Location loc = dps->getLoc();

      assert(dps.getNumDpsInits() == 1);
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
        Value updatedSubview =
            updateMaskingSubview(builder, loc, expanded, initOperand, iv);
        // Then replace the toTensor operand if it is not updated
        auto innerToTensor =
            dyn_cast<bufferization::ToTensorOp>(internalDef.getDefiningOp());
        assert(innerToTensor &&
               "Expecting memref outputs to be passed as tensors");
        OpOperand *memrefOperand = &innerToTensor.getMemrefMutable();
        if (memrefOperand->get() != updatedSubview) {
          Value toTensorSubview = nullptr;
          builder.setInsertionPointToStart(item->forOp.getBody());
          if (!updatedSubview) {
            toTensorSubview = createSubview(builder, loc, expanded,
                                            initOperand->get().getType(), iv);
            initOperand->set(toTensorSubview);
          } else {
            toTensorSubview = createSubview(builder, loc, expanded,
                                            memrefOperand->get().getType(), iv);
          }
          memrefOperand->set(toTensorSubview);
        }
        builder.setInsertionPointAfter(item->forOp);
        newResult = createToTensor(builder, loc, expanded);
      } else
        llvm_unreachable("Unexpected output type that is not tensor or memref");

      // Update outside users
      LLVM_DEBUG(dbgs() << "[migrateOps] Updating user of "; orig.dump());
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
  builder.setInsertionPointToEnd(newLoop.getBody());
  builder.clone(*pipelineLoop.getBody()->getTerminator(), globalIRMap);
}

/// Main method of the pass
LogicalResult CVPipelineImpl::run() {
  if (createWorkItems().failed())
    return failure();
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
    }
  });
  createNewLoops();
  migrateOps();

  pipelineLoop.replaceAllUsesWith(newLoop.getResults());

  { // FIXME: This really isn't safe generally, but we hack this in for now:
    unsigned i = 0;
    Block *body = newLoop.getBody();
    auto yield = cast<scf::YieldOp>(body->getTerminator());
    while (i < newLoop.getNumRegionIterArgs()) {
      Value yieldVal = yield->getOperand(i);
      Operation *defining = yieldVal.getDefiningOp();
      if (defining && pipelineLoop->isAncestor(defining)) {
        builder.setInsertionPoint(yield);
        Operation *newOp = builder.clone(*defining, globalIRMap);
        defining->replaceAllUsesWith(newOp->getResults());
      }
      ++i;
    }
  }

  LLVM_DEBUG(dbgs() << "\n\nAfter everything:\n";
             newLoop->getParentOfType<func::FuncOp>()->dump());
  pipelineLoop->erase();
  return success();
}

void CVPipeliningPass::runOnOperation() {
  // First find loop to operate on
  func::FuncOp func = getOperation();
  DenseSet<scf::ForOp> pipelinedLoops;

  // Disabled via options
  if (this->pipelineDepth == 1 || this->pipelineDepth == 0)
    return;

  func->walk<WalkOrder::PreOrder>([&pipelinedLoops, this](scf::ForOp loop) {
    auto parentLoop = loop->getParentOfType<scf::ForOp>();

    // Check if this is a part of pipelined loop already
    while (parentLoop) {
      if (pipelinedLoops.contains(parentLoop))
        return;
      parentLoop = parentLoop->getParentOfType<scf::ForOp>();
    }

    CVPipelineImpl impl(loop, this->pipelineDepth);
    if (impl.run().succeeded())
      pipelinedLoops.insert(loop);
  });
}

std::unique_ptr<Pass>
hivm::createCVPipeliningPass(const CVPipeliningOptions &options) {
  return std::make_unique<CVPipeliningPass>(options);
}
} // namespace mlir

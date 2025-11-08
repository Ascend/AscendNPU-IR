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

#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/SetOperations.h"

#define DEBUG_TYPE "cv-pipelining"

using llvm::dbgs;
namespace mlir {
using namespace hivm;

#define GEN_PASS_DEF_CVPIPELINING
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

using bishengir::memref_ext::AllocWorkspaceOp;
using hivm::detail::queryCoreTypeHelper;

namespace {

struct WorkspaceAllocParams {
  unsigned multibuffer;
  annotation::MarkOp marker;
  bufferization::ToTensorOp toTensor;
};

struct WorkItem {
  // Since insertion is not in lexigraphical order, use a set here so that we
  // can iterate over original block while looking up to preserve order
  SetVector<Operation *> ops;
  // Guesstimate of runtime cost of this work item.
  float cost;
  // Values that are referred by other work items later will be stored in this
  // list. Everything here requires the tensor types to be expanded by
  // Multibuffer times, e.g. <16xf16> into <2x16xf16>
  SmallVector<Value> localOutputs;
  // Alloc->Load->ToTensor does not need iterargs, so we have to keep track of
  // other localOutputs
  unsigned numLocalYields;
  SmallVector<Operation *> workspaceOutputs;
  // Values that are yielded in the parent for loop
  SmallVector<Value> yieldedOutputs;
  // Vector or Cube, other types shouldn't end up in here
  TCoreType core;
  // Number of multibuffer
  unsigned multibuffer;
  // The containing for loop
  scf::ForOp parentFor;
  // After unrolling the parent for loop, the upper bound for "reroll"ed loops
  // are computed and inserted here. Created in "unrollOuterLoop"
  Value upperBound;
  // The for op corresponding to the multibuffering, constructed in
  // "constructPipelineLoop"
  scf::ForOp forOp;
  // Reconstructed original induction variable
  Value reconstructedIV;
#ifndef NDEBUG
  int id;
#endif
};

struct CVPipeliningPass
    : public ::mlir::impl::CVPipeliningBase<CVPipeliningPass> {
public:
  using Base::Base;
  void runOnOperation() final;

private:
  // Essentially a work list for DFS
  DenseSet<Block *> blocksContainingMultibuffer_;

  // Since work items need to be referenced in multiple locations, we need to
  // use its reference in multiple places. In order for references to not be
  // destroyed by vector reallocations, we use shared_ptr
  SmallVector<std::shared_ptr<WorkItem>> worklist_;

  DenseMap<AllocWorkspaceOp, WorkspaceAllocParams> workspaceAllocs_;

  DenseMap<scf::ForOp, Value> originalStepMap_;

  // We will have to expand the alloc workspace ops, so this will map from the
  // original to the new ones
  DenseMap<Value, Value> expandedWorkspaceMap_;

  // Reverse mapping from op to work items containing the op
  DenseMap<Operation *, WorkItem *> opToWorkItemMap_;

  DenseSet<Operation *> toErase_;

  IRMapping irMap_;

  // Clear all data structures
  void init();

  LogicalResult constructWorkItems(Block *containingBlk, unsigned multibuffer);

  LogicalResult fillSingleWorkItem(WorkItem *item,
                                   SetVector<Operation *> &vecVisit,
                                   SetVector<Operation *> &cubeVisit,
                                   DenseSet<Operation *> &visited, bool doCube);

  void balanceVectorWorkItems(ArrayRef<std::shared_ptr<WorkItem>> list) const;

  void expandWorkspace(OpBuilder &builder);

  LogicalResult traceAllUseDef(OpBuilder &builder);

  Value unrollOuterLoop(OpBuilder &builder, scf::ForOp forOp, int unrollFactor);

  void constructInnerForOpIterArgs(OpBuilder &builder, WorkItem *task,
                                   SmallVector<Value, 4> &innerForIterArgs,
                                   scf::ForOp forOp);

  void constructPipelineLoop(OpBuilder &builder, WorkItem *item);

  void migrateOps(OpBuilder &builder);

  void printWorkItem(WorkItem *item);
  void printWorkList(ArrayRef<std::shared_ptr<WorkItem>> worklist);
};
} // namespace

static FailureOr<TCoreType> getCoreGrouping(Operation *op);

static FailureOr<TCoreType> scfOpWalkerHelper(Operation *op) {
  TCoreType result = TCoreType::CUBE_OR_VECTOR;
  auto wr = op->walk([&result](Operation *innerOp) {
    // Skip checking for scf ops
    if (isa<scf::ForOp, scf::IfOp>(innerOp))
      return WalkResult::advance();
    // Illegal - skip
    if (isa<AllocWorkspaceOp>(innerOp))
      return WalkResult::interrupt();

    auto maybeCore = getCoreGrouping(innerOp);
    if (failed(maybeCore))
      return WalkResult::interrupt();
    TCoreType coreTy = maybeCore.value();
    if (coreTy != hivm::TCoreType::CUBE && coreTy != hivm::TCoreType::VECTOR)
      return WalkResult::advance();
    if (result == hivm::TCoreType::CUBE_OR_VECTOR)
      result = coreTy;
    else if (result != coreTy)
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (wr.wasInterrupted())
    return op->emitWarning(
        "Unsupported nested structure for pipelining, skipping");
  return result;
}

/// Special case handling for deciding whether a op should be grouped with a
/// vector or cube op
FailureOr<TCoreType> getCoreGrouping(Operation *op) {
  // If not special case, just query the op itself for the core type
  auto maybeCore = queryCoreTypeHelper(op);
  if (maybeCore.has_value())
    return maybeCore.value();

  // For scf.for and scf.if, they are considered cube/vector ops if they only
  // contain cube/vector ops and does not allocate workspace within them
  if (isa<scf::ForOp, scf::IfOp>(op)) {
    return scfOpWalkerHelper(op);
  }

  // If not a for or if, we don't support regioned ops
  if (op->getNumRegions() > 0)
    return failure();

  if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
    auto userRange = alloc.getResult().getUsers();
    SmallVector<Operation *> users;
    users.append(userRange.begin(), userRange.end());
    while (!users.empty()) {
      Operation *usr = users.pop_back_val();
      if (isa<LoadOp>(usr))
        return getCoreGrouping(usr);
      if (isa<bufferization::ToTensorOp>(usr))
        continue;
      auto innerUsrRange = usr->getUsers();
      users.append(innerUsrRange.begin(), innerUsrRange.end());
    }
    return op->emitWarning("Alloc not used by load");
  }

  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
    Operation *defining = toTensor.getMemref().getDefiningOp();
    if (isa_and_nonnull<memref::AllocOp>(defining))
      return getCoreGrouping(defining);
  }

  // Otherwise if an op has tensor operands, group it into cube/vector,
  // depending on where the operand came from, when possible
  for (Value operand : op->getOperands()) {
    if (!isa<TensorType>(operand.getType()))
      continue;
    Operation *defining = operand.getDefiningOp();
    if (!defining)
      continue;
    // Do not trace stores - otherwise this might interfere with output marking
    if (isa<StoreOp, FixpipeOp>(defining))
      break;
    auto maybeCore = getCoreGrouping(defining);
    if (failed(maybeCore))
      return maybeCore;
    auto core = maybeCore.value();
    if (core == TCoreType::VECTOR || core == TCoreType::CUBE)
      return core;
  }

  return TCoreType::CUBE_OR_VECTOR;
}

static float estimateVecCost(ArrayRef<Operation *> opList) {
  float cost = 0;
  for (Operation *op : opList) {
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
    if (!dpsOp)
      continue;

    // Guesstimate cost model - numbers are pretty arbitrary
    for (auto init : dpsOp.getDpsInits()) {
      auto resTy = cast<ShapedType>(init.getType());
      float thisCost = resTy.getElementTypeBitWidth();
      for (int64_t dim : resTy.getShape()) {
        // Abort on dynamic shapes
        // TODO: Maybe try to determine size based on alloc/subview/tl.assume
        if (dim == ShapedType::kDynamic)
          return -1;
        thisCost *= dim;
      }
      // Fixed cost of executing instruction
      cost += thisCost + 256;
    }
  }
  return cost;
}

static bool pruneUsedOps(SetVector<Operation *> &independents,
                         DenseSet<Operation *> &dependents, WorkItem *item) {
  SmallVector<Operation *> toErase;
  for (auto *op : independents) {
    for (Value operand : op->getOperands()) {
      auto blkArg = dyn_cast<BlockArgument>(operand);
      if (!blkArg || blkArg.getArgNumber() == 0 ||
          blkArg.getOwner()->getParentOp() != item->parentFor)
        continue;
      Operation *defining =
          item->parentFor.getYieldedValues()[blkArg.getArgNumber() - 1]
              .getDefiningOp();
      if (dependents.contains(defining)) {
        toErase.push_back(op);
        break;
      }
    }
  }
  if (!toErase.empty()) {
    for (Operation *op : toErase)
      independents.remove(op);
    return true;
  }
  return false;
}

static SetVector<Operation *> independentSubgraph(WorkItem *item) {
  SmallVector<Operation *> stores, stack;
  DenseSet<Operation *> dependents, visited;

  for (auto *op : item->ops)
    if (isa<StoreOp>(op))
      stores.push_back(op);

  for (auto *store : stores) {
    stack.push_back(store);
    while (!stack.empty()) {
      auto *op = stack.pop_back_val();
      if (!item->ops.contains(op) || !item->parentFor->isAncestor(op) ||
          visited.contains(op))
        continue;
      visited.insert(op);
      dependents.insert(op);
      for (Value operand : op->getOperands()) {
        if (auto arg = dyn_cast<BlockArgument>(operand)) {
          if (arg.getArgNumber() == 0 ||
              arg.getOwner()->getParentOp() != item->parentFor)
            continue;
          auto *defining =
              item->parentFor.getYieldedValues()[arg.getArgNumber() - 1]
                  .getDefiningOp();
          stack.push_back(defining);
          continue;
        }
        auto *defining = operand.getDefiningOp();
        stack.push_back(defining);
      }
    }
  }

  auto independents = llvm::set_difference(item->ops, dependents);

  // Prune ops that cannot be moved due to iter args
  bool changed = true;
  while (changed)
    changed = pruneUsedOps(independents, dependents, item);

  return independents;
}

void CVPipeliningPass::init() {
  blocksContainingMultibuffer_.clear();
  worklist_.clear();
  workspaceAllocs_.clear();
  originalStepMap_.clear();
  expandedWorkspaceMap_.clear();
  opToWorkItemMap_.clear();
  irMap_.clear();
  toErase_.clear();
}

void CVPipeliningPass::printWorkItem(WorkItem *item) {
  LLVM_DEBUG({
    dbgs() << "Work Item ID: " << item->id << '\n';
    dbgs() << "Total Ops: " << item->ops.size() << "\n";
    for (auto *op : item->ops)
      dbgs() << "  - " << *op << "\n";
    dbgs() << "UpperBound: " << item->upperBound << '\n';
    dbgs() << "Local outputs: " << item->localOutputs.size() << "\n";
    for (auto val : item->localOutputs)
      dbgs() << "  - " << val << "\n";
    dbgs() << "WS outputs: " << item->workspaceOutputs.size() << "\n";
    for (auto val : item->workspaceOutputs)
      dbgs() << "  - " << *val << "\n";
    dbgs() << "Outputs: " << item->yieldedOutputs.size() << "\n";
    for (auto output : item->yieldedOutputs) {
      dbgs() << "  - " << output << "\n";
    }
    dbgs() << "Core: " << stringifyTCoreType(item->core) << "\n";
    dbgs() << "Guesstimate Cost: " << item->cost << "\n";
    dbgs() << "\n\n";
  });
}

void CVPipeliningPass::printWorkList(
    ArrayRef<std::shared_ptr<WorkItem>> worklist) {
  for (auto &workItem : worklist) {
    printWorkItem(workItem.get());
  }
}

static int getMultibufferCount(annotation::MarkOp marker) {
  auto multibufferAttr = llvm::cast_if_present<IntegerAttr>(
      marker->getAttr(MultiBufferAttr::name));
  if (!multibufferAttr)
    return -1;
  return multibufferAttr.getInt();
}

static std::optional<int> getMultibufferCount(Block *blk) {
  int result = -1;
  constexpr int uninitializedCount = -1;
  for (const Operation &op : blk->getOperations()) {
    auto annotation = dyn_cast<annotation::MarkOp>(&op);
    if (!annotation) {
      continue;
    }
    int mbCount = getMultibufferCount(annotation);
    if (result == uninitializedCount)
      result = mbCount;
    // All counts must be consistant within a block
    else if (result != mbCount)
      return std::nullopt;
  }
  return result;
}

/// Expand workspace by taking original workspace, and adding a multibuffer dim
/// in front: if multibuffer is 2, original workspace is <16x16xf16>, then new
/// expanded workspace is <2x16x16xf16>
void CVPipeliningPass::expandWorkspace(OpBuilder &builder) {
  OpBuilder::InsertionGuard g(builder);
  for (auto [alloc, info] : workspaceAllocs_) {
    builder.setInsertionPointToStart(
        alloc->getParentOfType<scf::ForOp>().getBody());
    Location loc = alloc.getLoc();
    MemRefType origType = alloc.getType();
    ArrayRef<int64_t> origShape = origType.getShape();
    SmallVector<int64_t> newShape = {info.multibuffer};
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

    toErase_.insert(alloc);
    toErase_.insert(info.toTensor);
  }
}

/// Traces the def chain to see if this value is supposed to be in workspace,
/// returns the original alloc_workspace op
static AllocWorkspaceOp getAllocWorkspace(Value v) {
  Operation *defining = v.getDefiningOp();
  // We are not tracing past iter args since workspace alloc should be in the
  // same scope
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

/// Only works for outputs of DPS ops
static SmallVector<Value> getDynamicShapes(OpBuilder &builder, Value val) {
  Operation *defining = val.getDefiningOp();
  if (!defining)
    llvm::report_fatal_error("Need to support tracing shapes past single "
                             "region for cv-pipelining", /*gen_crash_diag*/
                             false);

  auto dps = dyn_cast<DestinationStyleOpInterface>(defining);
  if (!dps)
    llvm::report_fatal_error(
        "Need to support tracing dynamic shape past ops other than DPS",
        /*gen_crash_diag*/ false);
  Value initOperand =
      dps.getDpsInitOperand(cast<OpResult>(val).getResultNumber())->get();

  defining = initOperand.getDefiningOp();
  if (auto emptyOp = dyn_cast<tensor::EmptyOp>(defining))
    return emptyOp.getDynamicSizes();

  // For non-empty dps inits:
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(dps);
  auto shapedTy = cast<ShapedType>(initOperand.getType());
  SmallVector<Value> returnVec;
  for (unsigned i = 0; i < shapedTy.getNumDynamicDims(); ++i)
    returnVec.push_back(
        builder.create<tensor::DimOp>(dps->getLoc(), initOperand, i));
  return returnVec;
}

// This function is used to construct the iterArgs of the inner forOp.
// Each output of the task needs to have a iterArg.  Firstly add from the outer
// for loop's iterArgs Then add the iterArgs for newly yielded ops
//   - Only support tensor.empty for tensor type ops
void CVPipeliningPass::constructInnerForOpIterArgs(
    OpBuilder &builder, WorkItem *task, SmallVector<Value, 4> &innerForIterArgs,
    scf::ForOp forOp) {
  // Outputs are yielded, get the corresponding iterarg
  ValueRange yielded = forOp.getYieldedValues();
  for (Value output : task->yieldedOutputs) {
    unsigned opNumber = static_cast<unsigned>(
        std::distance(yielded.begin(), llvm::find(yielded, output)));
    BlockArgument iterArg = forOp.getRegionIterArg(opNumber);
    innerForIterArgs.push_back(iterArg);
  }
  // Also yield the expanded local outputs
  for (Value workspace : task->localOutputs) {
    auto shapedTy = dyn_cast<TensorType>(workspace.getType());
    assert(shapedTy && "Expecting workspace outputs to be only tensor types");
    AllocWorkspaceOp alloc = getAllocWorkspace(workspace);
    if (alloc) {
      innerForIterArgs.push_back(expandedWorkspaceMap_[alloc]);
    } else if (isa_and_nonnull<bufferization::ToTensorOp>(
                   workspace.getDefiningOp())) {
      // to tensor ops are handled differently elsewhere, no need to add to iter
      // args
      continue;
    } else {
      // Expand the type by multibuffer and pass in a tensor.empty
      SmallVector<int64_t> newShape({task->multibuffer});
      auto shapeArray = shapedTy.getShape();
      newShape.append(shapeArray.begin(), shapeArray.end());
      ValueRange dynamicDims;

      if (shapedTy.getNumDynamicDims() > 0)
        dynamicDims = getDynamicShapes(builder, workspace);

      auto newType = RankedTensorType::get(newShape, shapedTy.getElementType());
      auto empty = builder.create<tensor::EmptyOp>(forOp->getLoc(), newType,
                                                   dynamicDims);
      innerForIterArgs.push_back(empty);
    }
  }
}

static Value createSubview(OpBuilder &builder, Location loc, Value from,
                           Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  auto newType = cast<MemRefType>(from.getType());

  // Set up offsets
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
  SmallVector<ReassociationIndices> reass{{0, 1}};
  for (unsigned i = 2; i < subview.getType().getRank(); ++i)
    reass.push_back({i});
  return builder.create<memref::CollapseShapeOp>(loc, subview, reass);
}

static void
expandMemrefLoadOutputs(OpBuilder &builder, WorkItem *item, scf::ForOp forOp,
                        IRMapping &irMap, DenseSet<Operation *> &toErase,
                        DenseMap<Value, Value> &expandedWorkspaceMap) {
  for (auto output : item->localOutputs) {
    auto toTensor =
        dyn_cast_or_null<bufferization::ToTensorOp>(output.getDefiningOp());
    if (!toTensor)
      continue;
    auto alloc =
        dyn_cast_or_null<memref::AllocOp>(toTensor.getMemref().getDefiningOp());
    assert(alloc &&
           "Expecting local outputing ToTensor ops to originate from alloc");
    Location loc = alloc->getLoc();
    builder.setInsertionPoint(forOp);
    MemRefType origType = alloc.getType();
    SmallVector<int64_t> newShape = {item->multibuffer};
    newShape.append(origType.getShape().begin(), origType.getShape().end());
    auto newType = MemRefType::get(newShape, origType.getElementType());
    auto newAlloc = builder.create<memref::AllocOp>(
        loc, newType, alloc.getDynamicSizes(), alloc.getAlignmentAttr());
    builder.setInsertionPointToStart(forOp.getBody());
    Value replacement =
        createSubview(builder, loc, newAlloc, forOp.getInductionVar());
    irMap.map(alloc, replacement);
    expandedWorkspaceMap[output] = newAlloc;
    toErase.insert(alloc);
    item->ops.remove(alloc);
  }
}

static Value reconstructIV(OpBuilder &builder, Location loc,
                           scf::ForOp parentFor, scf::ForOp innerForOp,
                           DenseMap<scf::ForOp, Value> &originalStepMap) {
  AffineExpr symA, symB;
  mlir::bindSymbols(builder.getContext(), symA, symB);
  auto dim = builder.getAffineDimExpr(0);
  auto ivMap = AffineMap::get(1, 2, dim * symA + symB, builder.getContext());
  Block *body = innerForOp.getBody();
  builder.setInsertionPointToStart(body);

  Value parentIV = parentFor.getInductionVar();

  Type parentTy = parentIV.getType();
  bool isIndex = parentTy.isIndex();
  Value pipelineIV = innerForOp.getInductionVar();
  Value originalStep = originalStepMap[parentFor];

  Type indexTy = builder.getIndexType();
  if (!isIndex) {
    originalStep =
        builder.create<arith::IndexCastOp>(loc, indexTy, originalStep);
    parentIV = builder.create<arith::IndexCastOp>(loc, indexTy, parentIV);
  }

  auto affineApplyOp = builder.create<affine::AffineApplyOp>(
      loc, ivMap, ValueRange{pipelineIV, originalStep, parentIV});
  Value reconstructedIV = affineApplyOp->getResult(0);
  if (!isIndex) {
    reconstructedIV =
        builder.create<arith::IndexCastOp>(loc, parentTy, reconstructedIV);
  }
  return reconstructedIV;
}

/// Special unroll tranformation, which instead of unrolling the loop body,
/// Adding a new loop inside to replace the unrolled body.
/// From:
/// for (10) {
///  add
/// }
/// To:
/// for (5) {
///   scf.for (2) {
///     add
///   }
/// }
void CVPipeliningPass::constructPipelineLoop(OpBuilder &builder,
                                             WorkItem *item) {
  LLVM_DEBUG(dbgs() << "Creating loop for work item #" << item->id << '\n');
  scf::ForOp parentFor = item->parentFor;

  builder.setInsertionPoint(parentFor.getBody()->getTerminator());

  Location loc = parentFor->getLoc();
  // Create fixed lower/upper/step for inner loop.
  Value innerLowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value innerUpperBound = item->upperBound;
  Value innerStep = builder.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value, 4> innerForIterArgs;
  // form the inner for loop's iterArgs
  constructInnerForOpIterArgs(builder, item, innerForIterArgs, parentFor);

  // Create the for loop itself, and attach various attributes
  auto innerForOp = builder.create<scf::ForOp>(
      loc, innerLowerBound, innerUpperBound, innerStep, innerForIterArgs);
  innerForOp->setAttr(
      kMultibufferUnrollAttrName,
      IntegerAttr::get(IntegerType::get(innerForOp->getContext(), 32),
                       item->multibuffer));
  innerForOp->setAttr(kPipelinedLoopCoreTypeAttrName,
                      TCoreTypeAttr::get(&getContext(), item->core));

  // Reconstruct the would-be induction variables of the original output
  item->reconstructedIV =
      reconstructIV(builder, loc, parentFor, innerForOp, originalStepMap_);

  // Local buffers that are loaded but used in more than one work item also
  // needs to be expanded, but it's memref instead of tensors
  expandMemrefLoadOutputs(builder, item, innerForOp, irMap_, toErase_,
                          expandedWorkspaceMap_);

  // Construct dummy yield for now. Since the for op may or may not come with a
  // terminator, we check and make sure to build one if its not present
  Block *body = innerForOp.getBody();
  if (!body->mightHaveTerminator()) {
    builder.setInsertionPointToEnd(body);
    builder.create<scf::YieldOp>(loc);
  }

  item->forOp = innerForOp;
}

Value CVPipeliningPass::unrollOuterLoop(OpBuilder &builder, scf::ForOp forOp,
                                        int unrollFactor) {
  assert(unrollFactor > 1);
  assert(llvm::range_size(forOp.getBody()->getOperations()) > 1 &&
         "Expecting non-dead scf.for");
  Location loc = forOp->getLoc();
  Value step = forOp.getStep();
  Value iv = forOp.getInductionVar();
  Value upperBound = forOp.getUpperBound();
  originalStepMap_[forOp] = step;

  // Whether the loop use index as lower/upper bound and step.
  // Otherwise it has to be i32.
  Type originalType = step.getType();
  auto indexTy = builder.getIndexType();
  bool isIndex = originalType.isIndex();

  builder.setInsertionPoint(forOp);
  AffineExpr d0, d1, d2;
  AffineExpr s0 = builder.getAffineSymbolExpr(0);
  bindDims(&getContext(), d0, d1, d2);

  // Unroll outer loop by multiplying step by number of multibuffer
  AffineExpr newStepExpr = unrollFactor * d0;
  if (!isIndex)
    step = builder.create<arith::IndexCastOp>(loc, indexTy, step);

  Value newStep = builder.create<affine::AffineApplyOp>(
      loc, AffineMap::get(1, 0, newStepExpr), step);

  if (!isIndex)
    forOp.setStep(
        builder.create<arith::IndexCastOp>(loc, originalType, newStep));
  else
    forOp.setStep(newStep);

  // Calculate upper bound for inner "rerolled" loops:
  // (min(iv + newStep, outerUB) - iv) / step

  builder.setInsertionPointToStart(forOp.getBody());
  if (!isIndex) {
    iv = builder.create<arith::IndexCastOp>(loc, indexTy, iv);
    upperBound = builder.create<arith::IndexCastOp>(loc, indexTy, upperBound);
  }
  // d0 = iv, d1 = newstep
  AffineExpr currentUBExpr = d0 + d1;
  Value minVal = builder.create<affine::AffineMinOp>(
      loc, AffineMap::get(3, 0, {currentUBExpr, d2}, &getContext()),
      ValueRange{iv, newStep, upperBound});

  // d0 = minVal, d1 = iv, s0 = oldStep
  AffineExpr newUBExpr = (d0 - d1).ceilDiv(s0);
  Value innerUB = builder.create<affine::AffineApplyOp>(
      loc, AffineMap::get(2, 1, newUBExpr), ValueRange{minVal, iv, step});
  return innerUB;
}

static std::shared_ptr<WorkItem> makeWorkItem(scf::ForOp parentFor,
                                              unsigned multibuffer) {
  auto item = std::make_shared<WorkItem>();
  item->parentFor = parentFor;
  item->multibuffer = multibuffer;
  item->numLocalYields = 0;
#ifndef NDEBUG
  static unsigned id = 0;
  item->id = id++;
#endif
  return item;
}

static FailureOr<bool> isSCFUnrealized(scf::ForOp parentFor, WorkItem *item,
                                       DenseSet<Operation *> &visited,
                                       DenseSet<Operation *> &trace,
                                       Operation *op);

static FailureOr<bool> isOperandUnrealized(scf::ForOp parentFor, WorkItem *item,
                                           DenseSet<Operation *> &visited,
                                           DenseSet<Operation *> &trace,
                                           Operation *op) {
  if (visited.contains(op) || trace.contains(op) ||
      !parentFor->isAncestor(op) || isa<tensor::EmptyOp>(op))
    return false;

  auto maybeUnrealized = isSCFUnrealized(parentFor, item, visited, trace, op);
  if (failed(maybeUnrealized))
    return failure();
  if (maybeUnrealized.value())
    return true;

  // If the source of the load is within scope of pipeline but not yet
  // processed, then we have to defer processing this operation
  if (auto load = dyn_cast<LoadOp>(op)) {
    Operation *srcDef = load.getSrc().getDefiningOp();
    return llvm::isa_and_nonnull<StoreOp, FixpipeOp>(srcDef) &&
           parentFor->isAncestor(srcDef) && !visited.contains(srcDef);
  }

  // This is to prevent infinite loops in cases where indirect dependencies on
  // oneself occurs
  trace.insert(op);

  for (Value input : op->getOperands()) {
    // We only care about tensor types for legality checks
    if (!isa<TensorType>(input.getType()))
      continue;
    Operation *defining = nullptr;
    if (auto blkArg = dyn_cast<BlockArgument>(input)) {
      // Induction variable always legal
      if (blkArg.getArgNumber() == 0)
        continue;
      // We only care about the loop being pipelined
      if (blkArg.getOwner()->getParentOp() != parentFor)
        continue;
      Value yielded = parentFor.getYieldedValues()[blkArg.getArgNumber() - 1];
      defining = yielded.getDefiningOp();
      // If the definition of this loop carried tensor is outside scope, or if
      // its within the current work item, it is legal
      if (!defining || !parentFor->isAncestor(defining) ||
          item->ops.contains(defining))
        continue;

      // This is illegal and we can't transform, because the defining op of this
      // block argument is in another work item... Performing our pipeline
      // transformation would violate dependency
      if (visited.contains(defining))
        return defining->emitWarning(
            "Cannot pipeline this op due to dependency\n");
    } else {
      defining = input.getDefiningOp();
      assert(defining &&
             "Expecting non-block-argument to be defined by an operation");
      if (!parentFor->isAncestor(defining))
        continue;
    }

    auto result =
        isOperandUnrealized(parentFor, item, visited, trace, defining);
    if (failed(result))
      return result;
    if (result.value())
      return true;
  }

  return false;
}

FailureOr<bool> isSCFUnrealized(scf::ForOp parentFor, WorkItem *item,
                                DenseSet<Operation *> &visited,
                                DenseSet<Operation *> &trace, Operation *op) {

  if (isa<scf::ForOp, scf::IfOp>(op)) {
    bool err = false;
    WalkResult wr = op->walk(
        [parentFor, item, &visited, &trace, &err, op](Operation *innerOp) {
          if (innerOp == op)
            return WalkResult::advance();
          auto maybeRes =
              isOperandUnrealized(parentFor, item, visited, trace, innerOp);
          if (failed(maybeRes)) {
            err = true;
            return WalkResult::interrupt();
          }
          if (maybeRes.value())
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
    if (wr.wasInterrupted()) {
      if (err)
        return failure();
      return true;
    }
  }
  return false;
}

static void traceOperand(scf::ForOp forOp, SmallVector<Operation *> &stack,
                         Value operand) {
  if (auto blkArg = dyn_cast<BlockArgument>(operand)) {
    if (blkArg.getOwner() != forOp.getBody() || blkArg.getArgNumber() == 0)
      return;
    // Subtract the induction variable from blk arg
    operand = forOp.getYieldedValues()[blkArg.getArgNumber() - 1];
  }
  if (Operation *defining = operand.getDefiningOp())
    stack.push_back(defining);
}

// Trace strictly cube or strictly vector ops, as well as store/fixpipe ops. Do
// not trace loads - they should be pulled in along with the compute on demand
template <TCoreType CoreTy> static bool isOpOfType(Operation *op) {
  if (isa<LoadOp>(op))
    return false;
  return queryCoreTypeHelper(op).value_or(TCoreType::CUBE_OR_VECTOR) == CoreTy;
}

static LogicalResult validateSCFOp(Operation *op) {
  if (!isa<scf::ForOp, scf::IfOp>(op))
    return success();
  bool cube = false;
  bool vector = false;

  WalkResult res = op->walk([&cube, &vector](Operation *innerOp) {
    if (isa<AllocWorkspaceOp>(innerOp))
      return WalkResult::interrupt();
    if (isOpOfType<hivm::TCoreType::CUBE>(innerOp)) {
      if (vector)
        return WalkResult::interrupt();
      cube = true;
    } else if (isOpOfType<hivm::TCoreType::VECTOR>(innerOp)) {
      if (cube)
        return WalkResult::interrupt();
      vector = true;
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return op->emitWarning("cv-pipelining cannot handle nested scf ops with "
                           "mixed core ops or workspace allocs, skipping");
  return success();
}

static LogicalResult checkDpsTensorSemantics(Operation *op) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (dpsOp && !dpsOp.hasPureTensorSemantics())
    return op->emitWarning("Expecting compute and store ops to be pure "
                           "tensors at cv-pipelining");
  return success();
}

static LogicalResult
markWorkspaceOps(Operation *op, DenseSet<Operation *> &visited,
                 DenseMap<AllocWorkspaceOp, WorkspaceAllocParams> &allocs,
                 unsigned multibuffer) {
  if (isa<AllocWorkspaceOp>(op)) {
    visited.insert(op);
    return success();
  }

  if (auto mark = dyn_cast<annotation::MarkOp>(op)) {
    if (auto alloc = llvm::dyn_cast_if_present<AllocWorkspaceOp>(
            mark.getSrc().getDefiningOp())) {
      if (allocs.contains(alloc)) {
        allocs[alloc].multibuffer = multibuffer;
        allocs[alloc].marker = mark;
      } else
        allocs[alloc] = {multibuffer, mark, nullptr};
      visited.insert(op);
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

    visited.insert(op);
  }
  return success();
}

/// A few caveats with the initial visit list:
static LogicalResult populateVisitList(
    Block *blk, SetVector<Operation *> &vecVisit,
    SetVector<Operation *> &cubeVisit, DenseSet<Operation *> &visited,
    DenseMap<AllocWorkspaceOp, WorkspaceAllocParams> &workspaceAllocs,
    unsigned multibuffer, bool &cubeFirst) {
  DenseMap<AllocWorkspaceOp, WorkspaceAllocParams> allocs;
  for (Operation &op : llvm::reverse(blk->getOperations())) {
    if (isOpOfType<TCoreType::CUBE>(&op)) {
      if (checkDpsTensorSemantics(&op).failed())
        return failure();
      cubeVisit.insert(&op);
      cubeFirst = true;
      continue;
    }
    if (isOpOfType<TCoreType::VECTOR>(&op)) {
      if (checkDpsTensorSemantics(&op).failed())
        return failure();
      vecVisit.insert(&op);
      cubeFirst = false;
      continue;
    }

    if (isa<scf::ForOp, scf::IfOp>(&op)) {
      if (validateSCFOp(&op).failed())
        return failure();
      auto maybeCore = getCoreGrouping(&op);
      if (failed(maybeCore))
        return failure();
      TCoreType core = maybeCore.value();
      if (core == TCoreType::VECTOR) {
        vecVisit.insert(&op);
        cubeFirst = false;
        continue;
      }
      if (core == TCoreType::CUBE) {
        cubeVisit.insert(&op);
        cubeFirst = true;
        continue;
      }
    }

    // If workspace related op, construct related info and skip dfs
    if (markWorkspaceOps(&op, visited, allocs, multibuffer).failed())
      return failure();
  }
  for (auto [key, val] : allocs)
    workspaceAllocs[key] = val;
  return success();
}

static LogicalResult
markOutputs(ArrayRef<std::shared_ptr<WorkItem>> worklist, Operation *terminator,
            DenseMap<Operation *, WorkItem *> &opToWorkItemMap) {
  auto yieldedRange = terminator->getOperands();
  DenseSet<Value> yieldedVals(yieldedRange.begin(), yieldedRange.end());
  for (const auto &item : worklist) {
    for (Operation *op : item->ops) {
      if (isa<StoreOp, FixpipeOp>(op) &&
          getAllocWorkspace(cast<DestinationStyleOpInterface>(op)
                                .getDpsInitOperand(0)
                                ->get())) {
        item->workspaceOutputs.push_back(op);
        continue;
      }
      if (isa<tensor::EmptyOp>(op))
        continue;
      for (Value result : op->getResults()) {
        if (yieldedVals.contains(result)) {
          // Non tensor type or tensors that are only used by yield and other
          // ops within the same work item gets added to output
          if (llvm::any_of(result.getUsers(), [&](Operation *usr) {
                auto it = opToWorkItemMap.find(usr);
                return it != opToWorkItemMap.end() && it->second != item.get();
              }))
            return op->emitError("Unable to pipeline op whose value is yielded "
                                 "but used in different group");

          item->yieldedOutputs.push_back(result);
          continue;
        }
        // For workspace outputs, we only care about tensor values, since
        // others will be duplicated
        if (!isa<TensorType>(result.getType()))
          continue;

        for (Operation *usr : result.getUsers()) {
          if (opToWorkItemMap.contains(usr) && !item->ops.contains(usr)) {
            item->localOutputs.push_back(result);
            if (!isa<bufferization::ToTensorOp>(op))
              ++item->numLocalYields;
            break;
          }
          // End loop over result.users
        }
        // End loop over op->results
      }
      // End loop over item->ops
    }
    // End loop over worklist
  }
  return success();
}

static void traceConnectingOps(Operation *op, scf::ForOp parentFor,
                               SmallVector<Operation *> &workingStack) {
  // If load is from a tensor, then it should be from another workitem
  auto load = dyn_cast<LoadOp>(op);
  if (load && isa<TensorType>(load.getSrc().getType())) {
    // Do not trace ins()
    traceOperand(parentFor, workingStack, load.getDst());
    return;
  }

  // Trace potential dependencies of ops within the nested control flow
  if (isa<LoopLikeOpInterface, scf::IfOp>(op)) {
    op->walk([&](Operation *innerOp) {
      if (innerOp == op)
        return;
      traceConnectingOps(innerOp, parentFor, workingStack);
    });
  }

  for (Value operand : op->getOperands())
    traceOperand(parentFor, workingStack, operand);

  // Special handling for memref - Some load ops generated by triton is of
  // pattern: alloc->load->toTensor, where the load would fail to be traced
  // due to loading memref
  for (Value result : op->getResults()) {
    if (isa<MemRefType>(result.getType())) {
      auto userRange = result.getUsers();
      workingStack.append(userRange.begin(), userRange.end());
    }
  }
}

static LogicalResult
specializedDFS(SmallVector<Operation *> &workingStack, WorkItem *item,
               DenseSet<Operation *> &visited,
               DenseSet<Operation *> &deferredOps,
               DenseMap<Operation *, WorkItem *> &opToWorkItemMap) {
  DenseSet<Operation *> legalityTrace;
  scf::ForOp parentFor = item->parentFor;

  LLVM_DEBUG(dbgs() << "\n\nBeginning DFS\n");
  while (!workingStack.empty()) {
    Operation *op = workingStack.pop_back_val();

    auto maybeCore = getCoreGrouping(op);
    if (failed(maybeCore))
      return failure();
    TCoreType core = maybeCore.value();
    bool vector = core == TCoreType::VECTOR;
    bool cube = core == TCoreType::CUBE;

    // If op is in different core type, or if already visited, skip:
    if ((item->core == hivm::TCoreType::CUBE && vector) ||
        (item->core == hivm::TCoreType::VECTOR && cube) ||
        visited.contains(op) || item->ops.contains(op))
      continue;

    // Confine DFS to current loop scope only
    if (!parentFor->isAncestor(op) || op->getParentOp() != parentFor)
      continue;

    // If the input value has not been stored yet, skip - this is why we
    // reverse the iteration when populating toVisit
    legalityTrace.clear();
    FailureOr<bool> unrealized =
        isOperandUnrealized(parentFor, item, visited, legalityTrace, op);
    if (failed(unrealized))
      return failure();
    if (unrealized.value()) {
      if (vector || cube)
        deferredOps.insert(op);
      continue;
    }

    // Only insert vector or cube ops into the visited - this is due to us
    // wanting to duplicate potentially reused scalar operations
    if (vector || cube) {
      visited.insert(op);
      deferredOps.erase(op);
      opToWorkItemMap[op] = item;
    }
    // Finally DFS - trace operands and results:
    item->ops.insert(op);
    LLVM_DEBUG(dbgs() << "Inserted "; op->dump());

    traceConnectingOps(op, parentFor, workingStack);
  }
  return success();
}

LogicalResult CVPipeliningPass::fillSingleWorkItem(
    WorkItem *item, SetVector<Operation *> &vecVisit,
    SetVector<Operation *> &cubeVisit, DenseSet<Operation *> &visited,
    bool doCube) {
  DenseSet<Operation *> deferredOps;
  SmallVector<Operation *> workingStack;
  if (doCube) {
    item->core = TCoreType::CUBE;
    workingStack.append(cubeVisit.begin(), cubeVisit.end());
  } else {
    item->core = TCoreType::VECTOR;
    workingStack.append(vecVisit.begin(), vecVisit.end());
  }
  if (specializedDFS(workingStack, item, visited, deferredOps, opToWorkItemMap_)
          .failed())
    return failure();

  if (doCube)
    llvm::set_intersect(cubeVisit, deferredOps);
  else
    llvm::set_intersect(vecVisit, deferredOps);

  return success();
}

LogicalResult CVPipeliningPass::constructWorkItems(Block *containingBlk,
                                                   unsigned multibuffer) {
  Operation *parentOp = containingBlk->getParentOp();
  auto parentFor = dyn_cast<scf::ForOp>(parentOp);
  assert(parentFor && "expecting parent op to be a scf.for");
  Operation *terminator = containingBlk->getTerminator();
  unsigned workItems = 0;

  SetVector<Operation *> vecVisit, cubeVisit;
  bool doCube;
  DenseSet<Operation *> visited;
  // First pass - group known vector/cube operations. Reverse the iteration
  // order here, so that the DFS below will group items in (roughly)
  // lexicographical order. This will become important later.
  if (populateVisitList(containingBlk, vecVisit, cubeVisit, visited,
                        workspaceAllocs_, multibuffer, doCube)
          .failed())
    return failure();

  // Second pass - DFS on all operand/results to construct subgraph before sync
  // is needed

  // Do not DFS to the yield op
  visited.insert(terminator);

  unsigned totalOps = vecVisit.size() + cubeVisit.size();
  while (!(vecVisit.empty() && cubeVisit.empty())) {
    auto item = makeWorkItem(parentFor, multibuffer);
    if (fillSingleWorkItem(item.get(), vecVisit, cubeVisit, visited, doCube)
            .failed())
      return failure();

    if (!item->ops.empty()) {
      ++workItems;
      worklist_.push_back(std::move(item));
      doCube = !doCube;
    }
    unsigned newTotal = vecVisit.size() + cubeVisit.size();
    if (newTotal == totalOps)
      return failure();
    totalOps = newTotal;
  } // end while Visit not empty

  // If there are only two work items, no need to pipeline, as the double
  // buffering pass should handle it just fine
  if (workItems <= 2) {
    worklist_.pop_back_n(workItems);
    return success();
  }

  auto curWorklist = ArrayRef(worklist_).take_back(workItems);

  for (const auto &item : curWorklist)
    item->cost = estimateVecCost(item->ops.getArrayRef());

  if (this->enableAutoBalance.getValue())
    balanceVectorWorkItems(curWorklist);

  // Final pass - get outputs from each work item
  if (failed(markOutputs(curWorklist, terminator, opToWorkItemMap_)))
    return failure();
  printWorkList(worklist_);
  return success();
}

/// Rules:
/// * If op is outside the scope of the parent of of the current work item,
///   then it will not be considered for pipelining.
/// * Trace will end with the following, with output being yielded:
///   - hivm.hir.load - where the input is not yet stored
///   - hivm.hir.fixpipe
///   - hivm.hir.store
///   - terminator
LogicalResult CVPipeliningPass::traceAllUseDef(OpBuilder &builder) {
  // Vector used as a stack for dfs
  LogicalResult anySucceeded = failure();
  unsigned workItemSize = 0;
  for (Block *containingBlk : blocksContainingMultibuffer_) {
    auto maybeMultibuffer = getMultibufferCount(containingBlk);
    int multibuffer = maybeMultibuffer.value_or(-1);
    if (multibuffer < 2) {
      containingBlk->getParentOp()->emitWarning(
          "Invalid multibuffer count, skipping pipelining");
      continue;
    }

    LogicalResult result = constructWorkItems(containingBlk, multibuffer);
    unsigned numWorkitems = worklist_.size() - workItemSize;
    if (numWorkitems == 0)
      continue;
    if (result.succeeded()) {
      Value upperBound = unrollOuterLoop(
          builder, cast<scf::ForOp>(containingBlk->getParentOp()),
          worklist_.back()->multibuffer);
      for (const auto &item : ArrayRef(worklist_).take_back(numWorkitems))
        item->upperBound = upperBound;

      anySucceeded = result;
      workItemSize = worklist_.size();
    } else
      worklist_.pop_back_n(numWorkitems);
  }
  LLVM_DEBUG(dbgs() << "\nTotal " << worklist_.size() << " work items\n";);
  return anySucceeded;
}

void CVPipeliningPass::balanceVectorWorkItems(
    ArrayRef<std::shared_ptr<WorkItem>> list) const {
  SmallVector<WorkItem *> vecItems;
  for (const auto &item : list) {
    if (item->core == hivm::TCoreType::VECTOR)
      vecItems.push_back(item.get());
  }
  if (vecItems.size() < 2)
    return;

  using FloatingItem = std::pair<float, SetVector<Operation *>>;
  SmallVector<FloatingItem> independents;

  for (WorkItem *item : vecItems) {
    // Skip if contains dynamic shapes
    if (item->cost < 0)
      continue;
    auto isolated = independentSubgraph(item);
    if (isolated.empty() || isolated.size() == item->ops.size())
      continue;
    float movableCost = estimateVecCost(isolated.getArrayRef());
    item->cost -= movableCost;
    independents.emplace_back(movableCost, isolated);
    for (auto *op : isolated)
      item->ops.remove(op);
  }

  auto sortByCost = [&vecItems]() {
    llvm::sort(vecItems, [](const WorkItem *lhs, const WorkItem *rhs) {
      return lhs->cost < rhs->cost;
    });
  };
  sortByCost();
  llvm::sort(independents,
             [](const FloatingItem &lhs, const FloatingItem &rhs) {
               return lhs.first > rhs.first;
             });

  // NOTE: Very naive/rudimentry balance algorithm
  // We want to minimize the difference between the max and the min cost within
  // vecItems.
  while (!independents.empty()) {
    auto item = independents.pop_back_val();
    vecItems.back()->ops.insert(item.second.begin(), item.second.end());
    sortByCost();
  }
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

static tensor::ExtractSliceOp createExtractSlice(OpBuilder &builder,
                                                 Location loc, Value from,
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
  return builder.create<tensor::ExtractSliceOp>(loc, cast<RankedTensorType>(to),
                                                from, offsets, sizes, strides);
}

static void processLocalOutputs(OpBuilder &builder, WorkItem *item,
                                IRMapping &loopMap,
                                SmallVector<Value> &yieldValues) {
  auto forOp = item->forOp;
  auto iterArgs = forOp.getRegionIterArgs().take_back(item->numLocalYields);
  Operation *terminator = forOp.getBody()->getTerminator();
  Value iv = forOp.getInductionVar();

  unsigned idx = 0;
  for (Value wsOutput : item->localOutputs) {
    Operation *defining = wsOutput.getDefiningOp();
    if (!defining)
      llvm::report_fatal_error("Error in cv-pipelining", /*print_diag*/ false);
    if (llvm::isa<bufferization::ToTensorOp>(defining))
      continue;
    Value wsIterArg = iterArgs[idx++];
    builder.setInsertionPoint(terminator);
    Value mappedWS = loopMap.lookupOrDefault(wsOutput);
    auto originalType = cast<TensorType>(wsOutput.getType());
    Location loc = defining->getLoc();

    // Yield the value in the original workspace tensor slice
    Value result = createInsertSlice(builder, loc, mappedWS, wsIterArg, iv);
    yieldValues.push_back(result);

    // Change the output dps init to tensor.extract_slice
    auto outputing =
        dyn_cast<DestinationStyleOpInterface>(mappedWS.getDefiningOp());
    assert(outputing && "Expecting outputing op to be DPS op");
    if (!outputing)
      llvm::report_fatal_error("Error in cv-pipelining", /*print_diag*/ false);
    builder.setInsertionPoint(outputing);
    Value extractSlice =
        createExtractSlice(builder, loc, wsIterArg, originalType, iv);
    assert(outputing.getNumDpsInits() == 1 &&
           "Expecting outputing DPS op to only have one dps output");
    if (outputing.getNumDpsInits() != 1)
      llvm::report_fatal_error("Error in cv-pipelining", /*print_diag*/ false);
    OpOperand *operand = outputing.getDpsInitOperand(0);
    operand->set(extractSlice);
  }
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
        createSubview(builder, loc, newAlloc, forOp.getInductionVar());
    if (auto storeOp = dyn_cast<StoreOp>(store))
      builder.create<StoreOp>(loc, TypeRange{}, storeOp.getSrc(), newDst);
    else if (auto fixpipe = dyn_cast<FixpipeOp>(store))
      builder.create<FixpipeOp>(
          loc, TypeRange{}, fixpipe.getSrc(), newDst,
          fixpipe.getEnableNz2ndAttr(), fixpipe.getPreQuantAttr(),
          fixpipe.getPreReluAttr(), fixpipe.getChannelSplitAttr());
    store->erase();

    // Create new toTensor
    builder.setInsertionPointAfter(forOp);
    expandedWorkspaceMap[original] =
        builder.create<bufferization::ToTensorOp>(loc, newAlloc,
                                                  /*restrict*/ true);
  }
}

static void processWorkspaceOutputUsers(
    OpBuilder &builder, WorkItem *item,
    const DenseMap<Operation *, WorkItem *> &opToWorkItemMap,
    const DenseMap<Value, Value> &expandedWorkspaceMap,
    SmallVector<std::pair<OpOperand *, Value>> &replaces) {
  for (Operation *output : item->workspaceOutputs) {
    for (OpOperand &operand : output->getUses()) {
      Operation *owner = operand.getOwner();
      if (opToWorkItemMap.contains(owner))
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

static void
processOutputUsers(OpBuilder &builder,
                   ArrayRef<std::shared_ptr<WorkItem>> worklist,
                   const DenseMap<Operation *, WorkItem *> &opToWorkItemMap,
                   const DenseMap<Value, Value> &expandedWorkspaceMap) {
  SmallVector<std::pair<OpOperand *, Value>> replaces;
  for (auto &item : worklist) {
    unsigned idx = item->yieldedOutputs.size();
    for (Value origVal : item->localOutputs) {
      Value extractFrom;
      if (isa<bufferization::ToTensorOp>(origVal.getDefiningOp())) {
        Value newAlloc = expandedWorkspaceMap.lookup(origVal);
        builder.setInsertionPointAfter(item->forOp);
        extractFrom = builder.create<bufferization::ToTensorOp>(
            origVal.getDefiningOp()->getLoc(), newAlloc, /*restrict*/ true,
            /*writable*/ true);
      } else {
        extractFrom = item->forOp.getResult(idx++);
      }
      for (OpOperand &operand : origVal.getUses()) {
        Operation *owner = operand.getOwner();
        // If still original operation (not wrapped in for op) no need to do
        // anything since they will be erased
        if (opToWorkItemMap.contains(owner))
          continue;
        Value sliceIdx;
        builder.setInsertionPoint(owner);
        if (isa<scf::YieldOp>(owner)) {
          sliceIdx = builder.create<arith::ConstantIndexOp>(
              owner->getLoc(), item->multibuffer - 1);
        } else
          sliceIdx = owner->getParentOfType<scf::ForOp>().getInductionVar();
        Value slice = createExtractSlice(builder, owner->getLoc(), extractFrom,
                                         origVal.getType(), sliceIdx);
        replaces.emplace_back(&operand, slice);
      }
    }
    processWorkspaceOutputUsers(builder, item.get(), opToWorkItemMap,
                                expandedWorkspaceMap, replaces);
  }
  for (auto [operand, replace] : replaces)
    operand->set(replace);
}

static void cloneOps(OpBuilder &builder, WorkItem *item, IRMapping &loopMap,
                     DenseSet<Operation *> &toErase) {
  scf::ForOp parentFor = item->parentFor;
  scf::ForOp forOp = item->forOp;
  loopMap.map(parentFor.getInductionVar(), item->reconstructedIV);
  // Map iter args of parent loop as well
  for (const auto [init, arg] :
       llvm::zip(forOp.getInits(), forOp.getRegionIterArgs())) {
    auto parentArg = dyn_cast<BlockArgument>(init);
    if (!parentArg ||
        parentArg.getParentBlock()->getParentOp() != parentFor.getOperation())
      continue;
    loopMap.map(parentArg, arg);
  }

  builder.setInsertionPoint(item->forOp.getBody()->getTerminator());
  for (Operation &op : item->parentFor.getBody()->getOperations()) {
    if (!item->ops.contains(&op))
      continue;
    Operation *newOp = nullptr;
    // Hack to remove potential issues with mismatch memref types
    if (auto subview = dyn_cast<memref::SubViewOp>(&op)) {
      Value src = loopMap.lookupOrDefault(subview.getSource());
      auto offsets = subview.getMixedOffsets();
      auto sizes = subview.getMixedSizes();
      auto strides = subview.getMixedStrides();
      for (OpFoldResult &fold :
           llvm::concat<OpFoldResult>(offsets, sizes, strides)) {
        if (auto val = dyn_cast<Value>(fold))
          fold = loopMap.lookupOrDefault(val);
      }
      newOp = builder.create<memref::SubViewOp>(subview->getLoc(), src, offsets,
                                                sizes, strides);
      loopMap.map(subview.getResult(), newOp->getResult(0));
    } else
      newOp = builder.clone(op, loopMap);
    loopMap.map(&op, newOp);
    toErase.insert(&op);
  }
}

/// Migrate all ops from block into loops for each work item
void CVPipeliningPass::migrateOps(OpBuilder &builder) {
  SmallVector<Value> yieldValues;
  for (const auto &item : worklist_) {
    IRMapping loopMap(irMap_);
    yieldValues.clear();
    auto forOp = item->forOp;
    LLVM_DEBUG(dbgs() << "\n\nMigrating ops for work item #" << item->id
                      << "\n");

    cloneOps(builder, item.get(), loopMap, toErase_);

    // Replace the use of the initial arg with iter arg if within the same for
    // loop
    for (auto [initVal, iterArg] :
         llvm::zip_equal(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
      initVal.replaceUsesWithIf(iterArg, [&](OpOperand &operand) {
        return operand.getOwner()->getParentOp() == forOp;
      });
    }

    // Yield should be taking outputs and workspace outputs in order
    unsigned numWorkspaceOuts = item->numLocalYields;

    unsigned totalOutputs = item->yieldedOutputs.size() + numWorkspaceOuts;
    ValueRange forOutputs =
        forOp->getResults().take_back(totalOutputs).drop_back(numWorkspaceOuts);
    ValueRange iterArgs = forOp.getRegionIterArgs()
                              .take_back(totalOutputs)
                              .drop_back(numWorkspaceOuts);

    Operation *terminator = forOp.getBody()->getTerminator();

    // Replace item->output
    for (auto [origOutput, forOutput, iterArg] :
         llvm::zip_equal(item->yieldedOutputs, forOutputs, iterArgs)) {
      Value mappedOutput = loopMap.lookupOrDefault(origOutput);
      yieldValues.push_back(mappedOutput);
      origOutput.replaceUsesWithIf(iterArg, [&](OpOperand &operand) {
        return operand.getOwner()->getParentOp() == forOp;
      });
      item->parentFor.getBody()->getTerminator()->replaceUsesOfWith(origOutput,
                                                                    forOutput);
    }

    // Replace workspace stores
    processWorkspaceOutputs(builder, item.get(), expandedWorkspaceMap_,
                            loopMap);

    // Replace "localOutputs", defined as tensors that are being used
    // outside the scope of the current work item
    processLocalOutputs(builder, item.get(), loopMap, yieldValues);

    // Finally replace the dummy yield op
    builder.setInsertionPoint(terminator);
    builder.create<scf::YieldOp>(terminator->getLoc(), yieldValues);
    terminator->erase();
  }

  // Final pass - this needed to be done after all the ops have been migrated
  // into their own loops: update user of local outputs to extract slices of
  // the for output
  processOutputUsers(builder, worklist_, opToWorkItemMap_,
                     expandedWorkspaceMap_);
}

void CVPipeliningPass::runOnOperation() {
  init();

  // First find the scope of which we want to operate on, currently we limit
  // this to a single region, i.e. a single scf.for, also only support
  // straight line code for now, i.e. nested for and if's are not allowed. In
  // such cases, we will trigger pass failure.
  func::FuncOp func = getOperation();
  func->walk([this](annotation::MarkOp annotation) {
    Attribute multiBufferAttr = annotation->getAttr(MultiBufferAttr::name);
    if (!multiBufferAttr)
      return;
    blocksContainingMultibuffer_.insert(annotation->getBlock());
  });

  OpBuilder builder(&getContext());

  // Step 1: Trace out and extract connected vector/cube ops into work items
  // NOTE: this will also unroll the surrounding for loop by the multibuffer
  //       count
  if (traceAllUseDef(builder).failed()) {
    // If tracing failed means condition necessary for pipelining is
    // unsatisfied, move on without pipelining
    if (!blocksContainingMultibuffer_.empty()) {
      func->emitWarning("Failed to pipelinine function");
    }
    return;
  }
  printWorkList(worklist_);

  // Step 2: Expand all alloc_workspace ops by number of multibuffers
  expandWorkspace(builder);

  // Step 3: construct the loop around each work item.
  for (auto &item : worklist_)
    constructPipelineLoop(builder, item.get());

  // Step 4: Move work items into respective loops, while updating their
  // input/outputs to match expanded workspace
  migrateOps(builder);

  // Clean up
  Operation *eraseOp = *toErase_.begin();
  while (!toErase_.empty()) {
    if (eraseOp == nullptr)
      eraseOp = *toErase_.begin();
    auto usrBegin = eraseOp->user_begin();
    if (usrBegin == eraseOp->user_end()) {
      eraseOp->erase();
      toErase_.erase(eraseOp);
      eraseOp = nullptr;
      continue;
    }
    Operation *usrOp = *usrBegin;
    Operation *usrParent = usrOp->getParentOp();
    while (!isa<func::FuncOp>(usrParent)) {
      if (toErase_.contains(usrParent)) {
        usrOp = usrParent;
        break;
      }
      usrParent = usrParent->getParentOp();
    }

    if (!toErase_.contains(usrOp)) {
      LLVM_DEBUG(dbgs() << func << "\n\nDef: " << *eraseOp
                        << "\nUser: " << *usrOp << '\n');
      usrOp->emitWarning(
          "Cannot erase user of pipelined op, aborting pipelining pass");
      signalPassFailure();
      return;
    }
    eraseOp = usrOp;
  }
}

std::unique_ptr<Pass>
hivm::createCVPipeliningPass(const CVPipeliningOptions &options) {
  return std::make_unique<CVPipeliningPass>(options);
}
} // namespace mlir

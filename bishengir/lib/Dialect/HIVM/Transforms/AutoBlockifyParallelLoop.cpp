//===- AutoBlockifyParallelLoop.cpp - Auto blockify loop ------------------===//
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
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/MemrefAliasAnalysisState.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_AUTOBLOCKIFYPARALLELLOOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "auto-blockify-parallel-loop"

using namespace mlir;
using namespace mlir::hivm;

static constexpr llvm::StringLiteral BlockifyLoopAttrName =
    "autoblockify.subloop";

namespace {
/// This pass will add a loop over the blocks when the logical block num is
/// larger than physical one.
///
/// for outer from 0,...,ceildiv(logical_block_dim,physical_block_dim)
/// 	 for inner from 0,...,physical_block_dim  <- get as block.idx
///    use(min(outer*physical_block_dim + inner, logical_block_dim))
struct AutoBlockifyParallelLoopPass
    : public impl::AutoBlockifyParallelLoopBase<AutoBlockifyParallelLoopPass> {
  using AutoBlockifyParallelLoopBase<
      AutoBlockifyParallelLoopPass>::AutoBlockifyParallelLoopBase;

public:
  void runOnOperation() override;
};

void traceExceptions(Value input, SmallPtrSet<Operation *, 4> &exceptions) {
  if (isa<BlockArgument>(input)) {
    return;
  }
  Operation *curOp = input.getDefiningOp();
  if (!curOp)
    return;
  exceptions.insert(curOp);
  for (auto opr : curOp->getOperands()) {
    traceExceptions(opr, exceptions);
  }
}

FailureOr<int> getPhysicalBlockNum(func::FuncOp funcOp) {
  auto maybeFuncCoreType = queryFuncCoreType(funcOp);
  if (!maybeFuncCoreType.has_value())
    return failure();
  TFuncCoreType funcCoreType = maybeFuncCoreType.value();
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  auto maybeSpecInterface = hacc::utils::getNPUTargetSpec(moduleOp);
  if (funcCoreType == TFuncCoreType::AIC_OR_AIV ||
      !maybeSpecInterface.has_value())
    return failure();
  auto specInterface = maybeSpecInterface.value();

  bool oneVectorCorePerBlock = funcCoreType == TFuncCoreType::AIV;
  if (funcCoreType == TFuncCoreType::MIX) {
    auto coreRatio = hivm::getCoreRatioAttr(funcOp);
    oneVectorCorePerBlock = coreRatio && coreRatio.getCube() == 0;
  }
  auto aPhysicalBlockNum = oneVectorCorePerBlock
                               ? specInterface.getSpecForIdentifierEnum(
                                     hacc::DeviceSpec::VECTOR_CORE_COUNT)
                               : specInterface.getSpecForIdentifierEnum(
                                     hacc::DeviceSpec::CUBE_CORE_COUNT);
  IntegerAttr castedAttr = cast<IntegerAttr>(aPhysicalBlockNum.getValue());
  int kPhysicalBlockNum = castedAttr.getValue().getSExtValue();
  return kPhysicalBlockNum;
}

enum class ProducerState { Visiting, Hoistable, NotHoistable };

// Collect the transitive producer closure that can be moved before `loop`.
// This is deliberately conservative: every value defined in the loop body must
// be produced by a side-effect-free, speculatable operation (or an allocation),
// and all operands and region captures must satisfy the same condition.
static bool collectHoistableProducers(Value value, scf::ForOp loop,
                                      SetVector<Operation *> &producers,
                                      DenseMap<Value, ProducerState> &cache) {
  // A shared producer DAG can be reached through multiple operands. Cache the
  // result for this loop to avoid repeatedly traversing the same subgraph.
  auto it = cache.find(value);
  if (it != cache.end())
    return it->second == ProducerState::Hoistable;

  // Mark the value before visiting its dependencies. A back-edge in malformed
  // or cyclic value dependencies is not safe to hoist, so treat it as failed.
  cache[value] = ProducerState::Visiting;

  auto finish = [&cache, value](bool hoistable) {
    cache[value] =
        hoistable ? ProducerState::Hoistable : ProducerState::NotHoistable;
    return hoistable;
  };
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    // Arguments of this loop body vary between iterations and cannot be moved
    // outside it. Arguments belonging to an enclosing region are invariant.
    return finish(blockArg.getOwner() != loop.getBody());

  Operation *def = value.getDefiningOp();
  if (!def || !loop->isAncestor(def))
    // Values defined outside the loop already dominate every iteration.
    return finish(true);
  if (def->getBlock() != loop.getBody())
    return finish(false);
  if (!isa<memref::AllocOp>(def) &&
      (!isMemoryEffectFree(def) || !isSpeculatable(def)))
    // Allocations are allowed because the allocation itself is the producer to
    // be shared; other producers must be safe to execute before the loop.
    return finish(false);
  for (Value operand : def->getOperands())
    if (!collectHoistableProducers(operand, loop, producers, cache))
      return finish(false);
  // Operations with regions can capture values without listing them as
  // operands. Those captures must also be available before the loop.
  SetVector<Value> capturedValues;
  getUsedValuesDefinedAbove(def->getRegions(), capturedValues);
  for (Value capturedValue : capturedValues)
    if (!collectHoistableProducers(capturedValue, loop, producers, cache))
      return finish(false);
  producers.insert(def);
  return finish(true);
}

// Build the HIVM-specific alias relations as a supplement to MLIR's general
// AliasAnalysis. The two analyses are both consulted before moving a load.
static void buildMemrefAliasState(Operation *root,
                                  utils::MemrefAliasAnalysisState &state) {
  root->walk([&](Operation *op) {
    for (auto [lhs, rhs] : hivm::getOperationAliasInfo(op))
      state.unionAliasSets(lhs, rhs);
  });
}

static bool hasAliasingWrite(scf::ForOp loop, Value source,
                             utils::MemrefAliasAnalysisState &aliasState,
                             AliasAnalysis &aliasAnalysis,
                             ArrayRef<Operation *> ignoredOps) {
  // Moving a load before the loop is only valid when no iteration can write
  // the source or destination buffer in between. Scan nested operations too,
  // since their effects are part of the loop's memory behavior.
  bool hasWrite = false;
  loop.getBody()->walk([&](Operation *op) {
    // The load and its movable producer closure are checked separately and
    // must not make their own effects look like loop-carried conflicts.
    if (hasWrite || llvm::is_contained(ignoredOps, op))
      return;

    SmallVector<MemoryEffects::EffectInstance> effects;
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(op)) {
      interface.getEffects(effects);
    } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      // Unknown effects cannot be proven independent of the moved buffers.
      hasWrite = true;
      return;
    }
    // Nested operations are inspected by the walk, not recursively collected
    // here, so the ignored producer closure stays excluded at every depth.
    for (const auto &effect : effects) {
      if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect()))
        continue;
      Value value = effect.getValue();
      // An unbound or non-memref write cannot be proven independent of the
      // buffer, so reject the hoist rather than making an unsafe assumption.
      if (!value || !isa<MemRefType>(value.getType())) {
        hasWrite = true;
        return;
      }
      // The custom alias state catches HIVM-specific relations. MLIR alias
      // analysis must also explicitly prove NoAlias; all other results are
      // conservatively treated as a possible write to `source`.
      if (aliasState.isAlias(source, value) ||
          !aliasAnalysis.alias(source, value).isNo()) {
        hasWrite = true;
        return;
      }
    }
  });
  return hasWrite;
}

// Hoist one invariant HIVM buffer load and the movable dependency closure
// immediately before `loop`.
static bool hoistInvariantLoad(hivm::LoadOp load, scf::ForOp loop,
                               utils::MemrefAliasAnalysisState &aliasState,
                               AliasAnalysis &aliasAnalysis) {
  // This pass implements a restricted LICM for pure buffer loads rather than
  // a general-purpose load-motion transformation.
  if (!load.hasPureBufferSemantics() || load->getNumResults() != 0)
    return false;

  Value source = load.getSrc();
  Value destination = load.getDst();
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  auto destinationType = dyn_cast<MemRefType>(destination.getType());
  if (!sourceType || !destinationType)
    return false;

  // Do not move a load if its address is explicitly local rather than GM.
  if (auto sourceSpace = getOptionalHIVMAddressSpace(source.getType());
      sourceSpace && *sourceSpace != AddressSpace::GM)
    return false;
  if (auto destinationSpace =
          getOptionalHIVMAddressSpace(destination.getType());
      destinationSpace && *destinationSpace == AddressSpace::GM)
    return false;

  // Every operand used by the load must remain valid after moving the load
  // before the loop. The producer cache is local to this loop because
  // hoistability depends on the loop boundary being considered.
  SetVector<Operation *> producers;
  DenseMap<Value, ProducerState> producerCache;
  if (!llvm::all_of(load->getOperands(), [&](Value operand) {
        return collectHoistableProducers(operand, loop, producers,
                                         producerCache);
      }))
    return false;

  // Restrict sharing to a fresh allocation in this body. Together with the
  // user checks below, this excludes pre-existing contents and readers through
  // aliases that could observe the destination before the load.
  auto destinationAlloc = destination.getDefiningOp<memref::AllocOp>();
  if (!destinationAlloc || destinationAlloc->getBlock() != loop.getBody())
    return false;

  // Move the complete dependency closure in dominance order. A buffer load
  // may have a to_tensor user; move that read with the load so that the UB
  // buffer and its tensor view are both shared by all iterations.
  SmallVector<Operation *> movedOps(producers.begin(), producers.end());
  movedOps.push_back(load);

  for (Operation *user : destination.getUsers()) {
    if (user == load)
      continue;
    auto toTensor = dyn_cast<bufferization::ToTensorOp>(user);
    // Moving a view from outside this body can place its definition after an
    // existing consumer. A view before the load can observe the old contents.
    // Views after the load in this body retain dominance over their consumers
    // when moved before the loop together with the allocation and load.
    if (!toTensor || toTensor->getBlock() != loop.getBody() ||
        !load->isBeforeInBlock(toTensor))
      return false;
    movedOps.push_back(toTensor);
  }

  // Do not move an allocation while leaving one of its lifetime or alias users
  // behind in the loop. In particular, a dealloc is a Free effect rather than
  // a Write effect and therefore is not caught by the memory-conflict scan.
  for (Operation *producer : producers) {
    auto alloc = dyn_cast<memref::AllocOp>(producer);
    if (!alloc)
      continue;
    for (Operation *user : alloc->getUsers())
      if (!llvm::is_contained(movedOps, user))
        return false;
  }

  // Check both buffers: writes to the source can change what the load reads,
  // while writes to the destination can invalidate the shared load result.
  if (hasAliasingWrite(loop, source, aliasState, aliasAnalysis, movedOps) ||
      hasAliasingWrite(loop, destination, aliasState, aliasAnalysis, movedOps))
    return false;

  Operation *beforeLoop = loop.getOperation();
  for (Operation *op : movedOps)
    op->moveBefore(beforeLoop);
  return true;
}

static void hoistInvariantLoads(scf::ForOp blockifyLoop,
                                utils::MemrefAliasAnalysisState &aliasState,
                                AliasAnalysis &aliasAnalysis) {
  // Hoisting from an inner loop can expose the load as invariant in its
  // enclosing loop. Rebuild the loop list after each round so newly exposed
  // loads are reconsidered at every level, including the blockify loop.
  bool changed;
  do {
    changed = false;
    SmallVector<scf::ForOp> loops;
    // `walk` includes the blockify loop and all loops nested below it. Only
    // direct body operations are inspected below; nested loops are handled as
    // separate entries in this list.
    blockifyLoop.walk([&](scf::ForOp loop) { loops.push_back(loop); });

    // Process inner loops first. A successful hoist moves the load and its
    // dependencies to the immediately enclosing block. Each successful hoist
    // crosses one loop boundary, and repeated rounds allow further outward
    // motion regardless of when the enclosing loop is visited.
    for (scf::ForOp loop : llvm::reverse(loops)) {
      SmallVector<hivm::LoadOp> loads;
      for (Operation &op : loop.getBody()->getOperations())
        if (auto load = dyn_cast<hivm::LoadOp>(&op))
          loads.push_back(load);
      for (hivm::LoadOp load : loads)
        changed |= hoistInvariantLoad(load, loop, aliasState, aliasAnalysis);
    }
    // Stop when a complete round moves no load; this is the LICM fixpoint.
  } while (changed);
}

static void replaceBlockIdUsers(IRRewriter &rewriter,
                                hivm::GetBlockIdxOp getBlockIdxOp, Value iv,
                                Value logicBlockNum, Operation *castedBlockID,
                                Value blockifyV2) {
  // block idx returns i64 meanwhile all other args are i32 so we cast it
  rewriter.setInsertionPointAfterValue(iv);
  auto loc = getBlockIdxOp->getLoc();
  auto mulOp = rewriter.create<arith::MulIOp>(loc, iv, blockifyV2);
  auto castedMulOp =
      rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), mulOp);
  rewriter.replaceAllUsesExcept(getBlockIdxOp, castedMulOp, castedBlockID);
}

static LogicalResult loopOnLogicBlock(func::FuncOp funcOp,
                                      IRRewriter &rewriter) {
  auto &entryBlock = funcOp.getBody().front();
  mlir::Location loc = entryBlock.front().getLoc();
  hivm::GetBlockIdxOp getBlockIdxOp;
  Value logicBlockNum;
  SmallPtrSet<Operation *, 4> exceptions;
  SmallVector<Operation *> opsToMove;
  for (auto &op : entryBlock) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(op)) {
      if (markOp->hasAttr(kLogicalBlockNumAttr)) {
        logicBlockNum = markOp->getOperand(0);
        continue;
      }
    }
    if (!isa<func::ReturnOp>(op)) {
      opsToMove.push_back(&op);
    }
    if (isa<hivm::GetBlockIdxOp>(op)) {
      getBlockIdxOp = dyn_cast<hivm::GetBlockIdxOp>(op);
      rewriter.setInsertionPointAfter(getBlockIdxOp);
    }
  }
  if (!logicBlockNum)
    return funcOp->emitError("Logical Block number not found");
  if (!getBlockIdxOp) {
    return success();
  }

  traceExceptions(logicBlockNum, exceptions);
  exceptions.insert(getBlockIdxOp);
  const int intBits = 32;
  auto kPhysicalBlockNum = getPhysicalBlockNum(funcOp);
  if (failed(kPhysicalBlockNum))
    return funcOp->emitError("Physical block num cannot be inferred");
  Value physicalBlockNum = rewriter.create<arith::ConstantIntOp>(
      loc, kPhysicalBlockNum.value(), intBits);
  Value upperBound = logicBlockNum;
  int blockifyNum = 1;
  if (auto blockifyAttr = funcOp->getAttr("auto_blockify_size"))
    blockifyNum = cast<IntegerAttr>(blockifyAttr).getInt();
  Value blockifyV2 =
      rewriter.create<arith::ConstantIntOp>(loc, blockifyNum, intBits);
  if (blockifyNum > 1)
    upperBound =
        rewriter.create<arith::CeilDivSIOp>(loc, upperBound, blockifyV2);
  auto blockID = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(),
                                                  getBlockIdxOp);
  auto forOp =
      rewriter.create<scf::ForOp>(loc, blockID, upperBound, physicalBlockNum);

  Block *loopBody = forOp.getBody();
  Operation *yieldOp = loopBody->getTerminator();
  for (Operation *op : opsToMove) {
    if (exceptions.count(op) == 0) {
      op->moveBefore(yieldOp);
    }
  }
  replaceBlockIdUsers(rewriter, getBlockIdxOp, forOp.getInductionVar(),
                      logicBlockNum, blockID, blockifyV2);
  auto unit = UnitAttr::get(forOp->getContext());
  forOp->setAttr(BlockifyLoopAttrName, unit);
  return success();
}
} // namespace

void AutoBlockifyParallelLoopPass::runOnOperation() {
  auto funOp = dyn_cast<func::FuncOp>(getOperation());
  if (!funOp) {
    return;
  }
  if (!hacc::utils::isDeviceEntry(funOp)) {
    return;
  }
  MLIRContext *ctx = funOp->getContext();
  IRRewriter rewriter(ctx);
  if (failed(loopOnLogicBlock(funOp, rewriter))) {
    return signalPassFailure();
  }
  // Build both alias views once for the function. LICM uses the custom HIVM
  // relations together with MLIR's conservative alias analysis for every
  // automatically created blockify loop.
  utils::MemrefAliasAnalysisState aliasState(funOp);
  buildMemrefAliasState(funOp, aliasState);
  AliasAnalysis aliasAnalysis(funOp);
  SmallVector<scf::ForOp> blockifyLoops;
  funOp.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(BlockifyLoopAttrName))
      blockifyLoops.push_back(loop);
  });
  // Only loops marked by loopOnLogicBlock are transformed; user-written
  // scf.for operations are intentionally outside this restricted LICM pass.
  for (scf::ForOp loop : blockifyLoops)
    hoistInvariantLoads(loop, aliasState, aliasAnalysis);
}

std::unique_ptr<Pass> mlir::hivm::createAutoBlockifyParallelLoopPass() {
  return std::make_unique<AutoBlockifyParallelLoopPass>();
}

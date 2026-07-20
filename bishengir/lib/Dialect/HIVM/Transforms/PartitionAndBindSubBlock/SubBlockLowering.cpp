//----------------------------SubBlockLowering.cpp----------------------------//
//
// Lowers {sub_block} scopes and core-bound ops into
// scf.if(get_sub_block_idx()==n) guards.
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

#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/SubBlockLowering.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/PartitionTypes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hivm {
namespace partition_and_bind {

namespace {

Value buildSubBlockEqCond(OpBuilder &builder, Location loc, int64_t core) {
  auto subBlockIdxOp =
      builder.create<hivm::GetSubBlockIdxOp>(loc, builder.getI64Type());
  Value subBlockIndex =
      builder
          .create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                      subBlockIdxOp.getResult())
          .getResult();
  Value coreConst = builder.create<arith::ConstantIndexOp>(loc, core);
  return builder.create<arith::CmpIOp>(loc, builder.getI1Type(),
                                       arith::CmpIPredicate::eq, subBlockIndex,
                                       coreConst);
}

/// Stamp the durable already-bound marker on `op` iff it is one of the op types
/// TileAndBindSubBlock re-guards. Normal computation is never re-guarded, so it
/// is left untagged.
void markIfSubBlockReguarded(Operation &op) {
  if (isa<hivm::StoreOp, hivm::IndirectStoreOp, hivm::StrideStoreOp,
          hivm::CopyOp, hivm::CustomOp, hivm::CustomMacroOp>(&op))
    op.setAttr(kSubBlockBoundOpAttrName, UnitAttr::get(op.getContext()));
}

Value deriveElseInit(OpBuilder &builder, Location loc, Value yieldedValue,
                     Type resultType, scf::IfOp ifOp) {
  if (Operation *def = yieldedValue.getDefiningOp()) {
    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(def)) {
      if (auto opResult = dyn_cast<OpResult>(yieldedValue)) {
        OpOperand *initOperand =
            dpsOp.getDpsInitOperand(opResult.getResultNumber());
        if (initOperand) {
          Value init = initOperand->get();
          if (init.getType() == resultType) {
            Operation *initDef = init.getDefiningOp();
            if (!initDef || !ifOp->isAncestor(initDef))
              return init;
            bool hoistable =
                llvm::all_of(initDef->getOperands(), [ifOp](Value operand) {
                  Operation *operandDef = operand.getDefiningOp();
                  return !operandDef || !ifOp->isAncestor(operandDef);
                });
            if (hoistable) {
              initDef->moveBefore(ifOp);
              return init;
            }
          }
        }
      }
    }
  }

  // Fallback: an fresh empty tensor of the result type.
  auto tensorType = dyn_cast<RankedTensorType>(resultType);
  if (!tensorType || !tensorType.hasStaticShape())
    return nullptr;
  return builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                         tensorType.getElementType());
}

/// Trace a memref value back through view-like ops and memory_space_cast to the
/// underlying buffer value (the alloc / block arg it ultimately aliases).
Value traceBufferBase(Value v) {
  for (;;) {
    if (auto view = v.getDefiningOp<ViewLikeOpInterface>()) {
      v = view.getViewSource();
      continue;
    }
    if (auto cast = v.getDefiningOp<memref::MemorySpaceCastOp>()) {
      v = cast.getSource();
      continue;
    }
    break;
  }
  return v;
}

bool isSharedDst(Value writeTarget) {
  Value base = traceBufferBase(writeTarget);
  if (isa<BlockArgument>(base))
    return true; // func-arg / GM buffer.

  std::optional<hivm::AddressSpace> as =
      hivm::getOptionalHIVMAddressSpace(base.getType());
  return as.has_value() && *as != hivm::AddressSpace::UB;
}

bool writesNonPrivateBuffer(Operation &op) {
  if (auto effOp = dyn_cast<MemoryEffectOpInterface>(&op)) {
    llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
    effOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &eff : effects) {
      if (!isa<MemoryEffects::Write>(eff.getEffect()))
        continue;
      Value v = eff.getValue();
      if (!v)
        return true; // write with no pinned value -> conservative, guard once.
      if (isSharedDst(v))
        return true;
    }
    return false;
  }

  for (Value operand : op.getOperands()) {
    if (!isa<ShapedType>(operand.getType()))
      continue;
    if (isa<BlockArgument>(traceBufferBase(operand)))
      return true;
  }
  return false;
}

bool writesAnyBuffer(Operation *op) {
  auto effOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effOp)
    return false;
  llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
  effOp.getEffects(effects);
  for (const MemoryEffects::EffectInstance &eff : effects) {
    if (!isa<MemoryEffects::Write>(eff.getEffect()))
      continue;
    Value v = eff.getValue();
    if (v && isa<ShapedType>(v.getType()))
      return true;
  }
  return false;
}

bool splitDoubleWriteFixpipe(hivm::FixpipeOp fp,
                             const CoreAssignment &assignment) {
  Value dstBuf = fp.getDst();
  Operation *allocOp = dstBuf.getDefiningOp();
  if (!allocOp)
    return false;

  // Trace dst -> memory_space_cast -> to_tensor to reach the readable tensor.
  memref::MemorySpaceCastOp castOp;
  for (Operation *user : dstBuf.getUsers())
    if (auto c = dyn_cast<memref::MemorySpaceCastOp>(user)) {
      castOp = c;
      break;
    }
  if (!castOp)
    return false;

  bufferization::ToTensorOp toTensorOp;
  for (Operation *user : castOp.getResult().getUsers())
    if (auto t = dyn_cast<bufferization::ToTensorOp>(user)) {
      toTensorOp = t;
      break;
    }
  if (!toTensorOp)
    return false;
  Value ubTensor = toTensorOp.getResult();

  // Clone the buffer chain + fixpipe to build the sub-block-1 copy. The mapping
  // threads the new alloc through the cast, to_tensor and fixpipe operands.
  OpBuilder builder(fp);
  builder.setInsertionPointAfter(fp);
  IRMapping map;
  builder.clone(*allocOp, map);
  builder.clone(*castOp.getOperation(), map);
  Operation *toTensorClone = builder.clone(*toTensorOp.getOperation(), map);
  Operation *fpClone = builder.clone(*fp.getOperation(), map);
  Value ubTensor1 = toTensorClone->getResult(0);

  // Pin each fixpipe to its sub-block (single-destination writes).
  fp.setSubBlockIdx(hivm::FixpipeSubBlock::SUB_BLOCK_0);
  cast<hivm::FixpipeOp>(fpClone).setSubBlockIdx(
      hivm::FixpipeSubBlock::SUB_BLOCK_1);

  // Route sub-block-1 readers onto the sub-block-1 buffer; sub-block-0 (and any
  // unassigned) readers keep the original.
  Block *entry = fp->getBlock();
  for (OpOperand &use : llvm::make_early_inc_range(ubTensor.getUses())) {
    Operation *top = use.getOwner();
    while (top && top->getBlock() != entry)
      top = top->getParentOp();
    if (top && assignment.coreOf(top) == Core::V1)
      use.set(ubTensor1);
  }
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// SubBlockLowering::run
//===----------------------------------------------------------------------===//

LogicalResult SubBlockLowering::run() {

  // Point the fixpipes at their sub-block destinations before lowering scopes.
  if (failed(setFixpipeDestinations()))
    return failure();

  for (const Supernode &node : assignment.supernodes) {
    if (failed(lowerSupernode(node)))
      return failure();
  }

  llvm::SmallVector<std::pair<Operation *, Core>, 8> outsideWork;
  for (Operation &op : func.getFunctionBody().front()) {

    if (isa<scope::ScopeOp>(op))
      continue;

    if (op.hasTrait<OpTrait::IsTerminator>() ||
        loweredScopeGuards.contains(&op))
      continue;
    if (isCubeOrSharedOp(&op))
      continue;

    if (op.getResults().empty() && !writesAnyBuffer(&op))
      continue;
    Core core = assignment.coreOf(&op);
    if (!isSingleCore(core)) {
      // A result-less op that writes a non-private (shared/GM) buffer must be
      // bound to a single core
      if (op.getResults().empty() && writesNonPrivateBuffer(op))
        return failure();
      continue;
    }
    outsideWork.emplace_back(&op, core);
  }

  for (auto &[op, core] : outsideWork) {
    if (failed(guardOutsideScopeOp(*op, core)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SubBlockLowering::setFixpipeDestinations
//===----------------------------------------------------------------------===//

LogicalResult SubBlockLowering::setFixpipeDestinations() {
  for (const auto &[op, subBlock] : assignment.fixpipeSubBlock)
    if (subBlock != 0)
      if (auto fixpipe = dyn_cast<hivm::FixpipeOp>(op))
        fixpipe.setSubBlockIdx(static_cast<hivm::FixpipeSubBlock>(subBlock));

  // A fixpipe whose result is needed on both sub-blocks is split into two
  // per-sub-block single-destination fixpipes writing separate buffers (see
  // splitDoubleWriteFixpipe).
  for (Operation *fixpipe : assignment.doubleWriteFixpipes) {
    auto fp = dyn_cast<hivm::FixpipeOp>(fixpipe);
    if (!fp)
      continue;
    if (!splitDoubleWriteFixpipe(fp, assignment))
      return fp.emitError("cannot split double-write fixpipe: unexpected "
                          "destination buffer chain");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SubBlockLowering::inlineSubBlockScopesAsFallback
//===----------------------------------------------------------------------===//

void SubBlockLowering::inlineSubBlockScopesAsFallback(func::FuncOp funcOp) {
  llvm::SmallVector<scope::ScopeOp, 4> scopes;
  funcOp.walk([&](scope::ScopeOp scopeOp) {
    if (isSingleCore(getSubBlockCoreOf(scopeOp.getOperation())))
      scopes.push_back(scopeOp);
  });

  for (scope::ScopeOp scopeOp : scopes) {
    Operation *op = scopeOp.getOperation();
    Block &body = scopeOp.getRegion().front();
    Operation *terminator = body.getTerminator();

    for (Operation &inner : llvm::make_early_inc_range(
             llvm::make_range(body.begin(), terminator->getIterator())))
      inner.moveBefore(op);

    for (auto [res, yielded] :
         llvm::zip_equal(scopeOp.getResults(), terminator->getOperands()))
      res.replaceAllUsesWith(yielded);

    scopeOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// SubBlockLowering::lowerSupernode
//===----------------------------------------------------------------------===//

LogicalResult SubBlockLowering::lowerSupernode(const Supernode &node) {
  scope::ScopeOp scopeOp = node.outerScope;
  if (!scopeOp)
    return failure();

  std::optional<int64_t> coreIdx = coreIndex(node.core);
  if (!coreIdx)
    return failure(); // a discovered supernode must carry a single core.

  Location loc = scopeOp.getLoc();
  OpBuilder builder(scopeOp);

  // (1) Build the `get_sub_block_idx() == n` predicate just before the scope.
  Value cond = buildSubBlockEqCond(builder, loc, *coreIdx);

  // (2) Grab the scope body block + its scope.return terminator. The single
  //     block holds the payload
  Block &scopeBlock = scopeOp.getRegion().front();
  auto returnOp = cast<scope::ReturnOp>(scopeBlock.getTerminator());
  llvm::SmallVector<Value, 4> thenYields(returnOp.getResults().begin(),
                                         returnOp.getResults().end());

  // (3) Create the scf.if with the scope's result types and an else region.
  auto ifOp = builder.create<scf::IfOp>(loc, scopeOp.getResultTypes(), cond,
                                        /*withElseRegion=*/true);

  // (4) Move the scope body ops into the then block
  Block *thenBlock = ifOp.thenBlock();
  thenBlock->getOperations().splice(
      thenBlock->end(), scopeBlock.getOperations(), scopeBlock.begin(),
      returnOp->getIterator());

  OpBuilder thenBuilder = OpBuilder::atBlockEnd(thenBlock);
  thenBuilder.create<scf::YieldOp>(loc, thenYields);

  thenBlock->walk([](Operation *op) { markIfSubBlockReguarded(*op); });

  // (5) yield an init per result (DPS init or fresh tensor.empty).
  Block *elseBlock = ifOp.elseBlock();
  OpBuilder elseBuilder = OpBuilder::atBlockEnd(elseBlock);
  llvm::SmallVector<Value, 4> elseYields;
  elseYields.reserve(thenYields.size());
  for (auto [yielded, resTy] :
       llvm::zip(thenYields, scopeOp.getResultTypes())) {
    Value init = deriveElseInit(elseBuilder, loc, yielded, resTy, ifOp);
    if (!init)
      return failure(); // unsupported (non-tensor / mismatched) result.
    elseYields.push_back(init);
  }
  elseBuilder.create<scf::YieldOp>(loc, elseYields);

  // (6) Replace the scope results over to the scf.if results. Stage-3 cleanup
  // re-identifies the guard structurally by its get_sub_block_idx predicate.
  loweredScopeGuards.insert(ifOp);
  scopeOp.getResults().replaceAllUsesWith(ifOp.getResults());
  scopeOp.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// SubBlockLowering::guardOutsideScopeOp
//===----------------------------------------------------------------------===//

LogicalResult SubBlockLowering::guardOutsideScopeOp(Operation &op,
                                                    Core core) const {
  std::optional<int64_t> coreIdx = coreIndex(core);
  if (!coreIdx)
    return failure();

  Location loc = op.getLoc();
  OpBuilder builder(&op);

  // Result-less op: wrap in a no-else guard,
  if (op.getResults().empty()) {
    Value cond = buildSubBlockEqCond(builder, loc, *coreIdx);
    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange(), cond,
                                          /*withElseRegion=*/false);
    OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
    Operation *cloned = thenBuilder.clone(op);
    markIfSubBlockReguarded(*cloned);
    op.erase();
    return success();
  }

  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(&op);
  if (!dpsOp)
    return success();

  llvm::SmallVector<Value, 4> elseYields;
  elseYields.reserve(op.getNumResults());
  for (OpResult result : op.getResults()) {
    OpOperand *initOperand = dpsOp.getDpsInitOperand(result.getResultNumber());
    if (!initOperand)
      return success();
    Value init = initOperand->get();
    if (init.getType() != result.getType())
      return success();
    elseYields.push_back(init);
  }

  Value cond = buildSubBlockEqCond(builder, loc, *coreIdx);
  auto ifOp = builder.create<scf::IfOp>(loc, op.getResultTypes(), cond,
                                        /*withElseRegion=*/true);
  {
    OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
    Operation *cloneOp = thenBuilder.clone(op);
    markIfSubBlockReguarded(*cloneOp);
    thenBuilder.create<scf::YieldOp>(loc, cloneOp->getResults());
  }
  {
    OpBuilder elseBuilder(&ifOp.getElseRegion().front(),
                          ifOp.getElseRegion().front().end());
    elseBuilder.create<scf::YieldOp>(loc, elseYields);
  }
  op.replaceAllUsesWith(ifOp.getResults());
  op.erase();
  return success();
}

} // namespace partition_and_bind
} // namespace hivm
} // namespace mlir

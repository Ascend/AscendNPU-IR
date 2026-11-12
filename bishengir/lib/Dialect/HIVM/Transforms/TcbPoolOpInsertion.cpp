//===- TcbPoolOpInsertion.cpp - Insert planned TCB pool operations --------===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

namespace mlir {
#define GEN_PASS_DEF_TCBPOOLOPINSERTION
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
// Resource configuration shared by the Cube and Vector rewrites.
struct PoolInsertionConfig {
  int64_t metadataWords() const {
    constexpr int64_t kHeaderWords = 18;
    constexpr int64_t kAddressQueueWords = 2 * 3 * 2;
    constexpr int64_t kReleaseEntryWords = 3;
    return kHeaderWords + poolBytes / slotBytes + kAddressQueueWords +
           kReleaseEntryWords * flagIds.size();
  }

  SmallVector<int64_t, 2> queueBytes;
  SmallVector<int32_t, 2> searchBegin{0, 0}, searchEnd;
  SmallVector<int32_t, 5> flagIds;
  int64_t slotBytes = 0;
  int64_t poolBytes = 0, metadataOffset = 0;
};

// Evaluate planned SSBUF addresses for each subblock to locate free metadata
// storage. The memory planner has already established the address
// types/layouts.
std::optional<int64_t> scalarAddress(Value value, int64_t subBlock) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto c = dyn_cast<arith::ConstantOp>(op))
    return cast<IntegerAttr>(c.getValue()).getInt();
  if (isa<GetSubBlockIdxOp>(op))
    return subBlock;
  if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(
          op))
    return scalarAddress(op->getOperand(0), subBlock);
  if (!isa<arith::AddIOp, arith::MulIOp>(op))
    return std::nullopt;
  auto lhs = scalarAddress(op->getOperand(0), subBlock);
  auto rhs = scalarAddress(op->getOperand(1), subBlock);
  if (!lhs || !rhs)
    return std::nullopt;
  int64_t address;
  bool overflow = isa<arith::AddIOp>(op)
                      ? llvm::AddOverflow(*lhs, *rhs, address)
                      : llvm::MulOverflow(*lhs, *rhs, address);
  return overflow ? std::nullopt : std::optional<int64_t>(address);
}

Value rootBuffer(Value value) {
  // Stop at if/select so callers can inspect each selected member separately.
  auto roots = utils::tracebackMemRefVecByTargetFn(value, [](Value source) {
    return !source.getDefiningOp<ViewLikeOpInterface>();
  });
  return roots.size() == 1 ? roots.front() : value;
}

static bool isPoolAddress(Value address, ArrayRef<Value> members) {
  address = rootBuffer(address);
  if (llvm::is_contained(members, address))
    return true;
  if (auto select = address.getDefiningOp<arith::SelectOp>())
    return isPoolAddress(select.getTrueValue(), members) &&
           isPoolAddress(select.getFalseValue(), members);
  if (auto branch = address.getDefiningOp<scf::IfOp>()) {
    unsigned index = cast<OpResult>(address).getResultNumber();
    return isPoolAddress(branch.thenYield().getOperand(index), members) &&
           isPoolAddress(branch.elseYield().getOperand(index), members);
  }
  return false;
}

static Block *findInitBlock(func::FuncOp func, Value member) {
  for (Block *block = member.getParentBlock(); block;) {
    for (auto pointer : block->getOps<PointerCastOp>())
      if (getHIVMAddressSpace(pointer.getType()) == AddressSpace::SSBUF)
        return block;
    Operation *parent = block->getParentOp();
    if (parent == func.getOperation())
      break;
    block = parent->getBlock();
  }
  return nullptr;
}

static std::optional<unsigned> getPoolQueue(annotation::MarkOp mark,
                                            ArrayAttr groups) {
  auto tcb = mark->getAttrOfType<HIVMTightlyCoupledBufferAttr>(
      HIVMTightlyCoupledBufferAttr::name);
  if (!tcb || !tcb.getId() ||
      getOptionalHIVMAddressSpace(mark.getSrc().getType()) != AddressSpace::UB)
    return std::nullopt;
  for (auto [q, group] : llvm::enumerate(groups))
    if (llvm::is_contained(cast<DenseI32ArrayAttr>(group).asArrayRef(),
                           *tcb.getId()))
      return q;
  return std::nullopt;
}

static LogicalResult reserveMetadata(ModuleOp module,
                                     PoolInsertionConfig &config) {
  SmallVector<std::pair<int64_t, int64_t>> ranges;
  // The native per-AIV address expressions establish the partition boundary.
  // Keep shared metadata in the first partition, beyond its existing words.
  std::optional<int64_t> partition;
  for (auto func : module.getOps<func::FuncOp>()) {
    auto result = func.walk([&](PointerCastOp pointer) -> WalkResult {
      if (getHIVMAddressSpace(pointer.getType()) != AddressSpace::SSBUF)
        return WalkResult::advance();
      auto a = scalarAddress(pointer.getAddrs().front(), 0);
      auto b = scalarAddress(pointer.getAddrs().front(), 1);
      int64_t bytes =
          llvm::divideCeil(*GetBufferBitSize(pointer.getResult()), int64_t{8});
      if (!a || !b)
        return pointer.emitError("cannot evaluate existing SSBUF addresses"),
               WalkResult::interrupt();
      ranges.emplace_back(*a, *a + bytes);
      ranges.emplace_back(*b, *b + bytes);
      if (*b > *a) {
        int64_t stride = *b - *a;
        partition = partition ? std::min(*partition, stride) : stride;
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
  }
  // The existing per-AIV stride supplies the available partition size.
  if (!partition)
    return module.emitError("cannot determine the SSBUF partition size");
  llvm::sort(ranges);
  int64_t bytes = config.metadataWords() * 4;
  int64_t address = 0;
  for (auto [start, end] : ranges) {
    if (address + bytes <= start)
      break;
    if (address < end)
      address = llvm::alignTo(end, int64_t(4));
  }
  if (address + bytes > *partition)
    return module.emitError("no free SSBUF interval for TCB metadata");
  config.metadataOffset = address;
  return success();
}

// Separate search regions contribute their own buffer capacities. For a shared
// region, count small, small, small, large, small, large, ... until it is full.
static unsigned countRequiredFlags(const PoolInsertionConfig &config) {
  const int64_t demands[] = {config.queueBytes[0] / config.slotBytes,
                             config.queueBytes[1] / config.slotBytes};
  if (config.searchEnd[0] <= config.searchBegin[1] ||
      config.searchEnd[1] <= config.searchBegin[0])
    return (config.searchEnd[0] - config.searchBegin[0]) / demands[0] +
           (config.searchEnd[1] - config.searchBegin[1]) / demands[1];

  const int64_t slotCount = config.poolBytes / config.slotBytes;
  const int64_t small = std::min(demands[0], demands[1]);
  const int64_t large = std::max(demands[0], demands[1]);
  if (small == large)
    return slotCount / small;

  const int64_t initialSmallCount = std::min<int64_t>(3, slotCount / small);
  const int64_t remaining = slotCount - initialSmallCount * small;
  const int64_t pairSlots = large + small;
  return initialSmallCount + 2 * (remaining / pairSlots) +
         (remaining % pairSlots >= large ? 1 : 0);
}

static LogicalResult setQueueSearchRanges(ModuleOp module,
                                          PoolInsertionConfig &config) {
  int64_t slots = config.poolBytes / config.slotBytes;
  config.searchEnd.assign(2, slots);
  int preferred = -1;
  if (auto hint = module->getAttrOfType<IntegerAttr>("hivm.tcb_pool_hint")) {
    if (!hint.getValue().isSignedIntN(32) || hint.getInt() < -1 ||
        hint.getInt() > 1)
      return module.emitError("hivm.tcb_pool_hint must be -1 (shared), 0 or 1");
    preferred = hint.getInt();
  } else if (module->hasAttr("hivm.tcb_pool_hint")) {
    return module.emitError("hivm.tcb_pool_hint must be an integer");
  } else if (config.queueBytes[0] != config.queueBytes[1]) {
    preferred = config.queueBytes[0] > config.queueBytes[1] ? 0 : 1;
  }
  if (preferred != -1) {
    int other = 1 - preferred;
    int64_t boundary = 2 * (config.queueBytes[preferred] / config.slotBytes);
    if (boundary + config.queueBytes[other] / config.slotBytes > slots)
      return module.emitError("preferred TCB queue needs two buffers and space "
                              "for the other queue");
    config.searchEnd[preferred] = boundary;
    config.searchBegin[other] = boundary;
  }
  return success();
}

static bool isFixVectorSync(Operation *op) {
  if (!isa<SyncBlockSetOp, SyncBlockWaitOp>(op))
    return false;
  auto src = op->getAttrOfType<PipeAttr>("tpipe").getPipe();
  auto dst = op->getAttrOfType<PipeAttr>("pipe").getPipe();
  return (src == PIPE::PIPE_FIX && dst == PIPE::PIPE_V) ||
         (src == PIPE::PIPE_V && dst == PIPE::PIPE_FIX);
}

// Collect old FIX/V notifications and available IDs independently of rewriting.
// Once legacy notifications disappear upstream, this analysis only needs to
// reserve IDs used by the remaining notifications.
static LogicalResult collectSynchronization(ModuleOp module,
                                            PoolInsertionConfig &config) {
  SmallVector<int32_t> recovered;
  DenseSet<int32_t> reserved;
  auto collectIds = [&](auto &&self, Value value,
                        SmallVectorImpl<int32_t> &ids) -> LogicalResult {
    if (auto c = value.getDefiningOp<arith::ConstantOp>())
      if (auto id = dyn_cast<IntegerAttr>(c.getValue())) {
        ids.push_back(id.getInt());
        return success();
      }
    if (auto select = value.getDefiningOp<arith::SelectOp>()) {
      if (failed(self(self, select.getTrueValue(), ids)))
        return failure();
      return self(self, select.getFalseValue(), ids);
    }
    return module.emitError(
        "cannot determine existing synchronization flag IDs");
  };
  auto result = module.walk([&](Operation *op) -> WalkResult {
    if (!isa<SyncBlockSetOp, SyncBlockWaitOp>(op))
      return WalkResult::advance();
    SmallVector<int32_t> ids;
    if (auto id = op->getAttrOfType<IntegerAttr>("static_flag_id"))
      ids.push_back(id.getInt());
    else {
      Value value = isa<SyncBlockSetOp>(op)
                        ? cast<SyncBlockSetOp>(op).getDynamicFlagId()
                        : cast<SyncBlockWaitOp>(op).getDynamicFlagId();
      if (failed(collectIds(collectIds, value, ids)))
        return WalkResult::interrupt();
    }
    if (isFixVectorSync(op)) {
      llvm::append_range(recovered, ids);
    } else {
      reserved.insert(ids.begin(), ids.end());
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  unsigned required = countRequiredFlags(config);
  if (!required || required > 16)
    return module.emitError(
        "TCB pool needs more synchronization IDs than supported");
  auto addId = [&](int32_t id) {
    if (id >= 0 && id < 16 && !reserved.contains(id) &&
        !llvm::is_contained(config.flagIds, id) &&
        config.flagIds.size() < required)
      config.flagIds.push_back(id);
  };
  for (int32_t id : recovered)
    addId(id);
  for (int32_t id = 0; id < 16; ++id)
    addId(id);
  if (config.flagIds.size() != required)
    return module.emitError(
        "TCB pool has insufficient available synchronization IDs");
  return success();
}

// Step 1 can be removed once upstream no longer emits legacy FIX/V sync.
static void removeLegacySynchronization(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  module.walk([&](Operation *op) {
    if (isFixVectorSync(op))
      rewriter.eraseOp(op);
  });
}

// Each Q is one logical transfer: a single FIX, or two mutually exclusive
// FIX operations selecting the original ping-pong destinations.
static LogicalResult rewriteCubeQueue(func::FuncOp function,
                                      ArrayRef<Value> members, unsigned queue,
                                      Value metadata, Value pool,
                                      IRRewriter &rewriter) {
  SmallVector<FixpipeOp, 2> fixes;
  function.walk([&](FixpipeOp fix) {
    if (isPoolAddress(fix.getDst(), members))
      fixes.push_back(fix);
  });
  if (fixes.empty())
    return function.emitError("cannot find FIX for TCB queue ") << queue;
  Operation *site = fixes.front();
  if (fixes.size() != 1) {
    auto branch = dyn_cast<scf::IfOp>(site->getParentOp());
    if (fixes.size() != 2 || !branch || fixes.back()->getParentOp() != branch ||
        fixes.front()->getParentRegion() == fixes.back()->getParentRegion())
      return function.emitError("expected one FIX or a pair of opposite FIX "
                                "branches for TCB queue ")
             << queue;
    site = branch;
  }
  rewriter.setInsertionPoint(site);
  auto alloc = rewriter.create<TcbPoolAllocOp>(
      site->getLoc(), rewriter.getI64Type(), rewriter.getI32Type(), metadata,
      pool, rewriter.getI32IntegerAttr(queue));
  for (FixpipeOp fix : fixes) {
    rewriter.setInsertionPoint(fix);
    Value address = rewriter.create<PointerCastOp>(
        fix.getLoc(), fix.getDst().getType(), ValueRange{alloc.getAddress()},
        ValueRange{});
    rewriter.modifyOpInPlace(fix, [&] { fix.getDstMutable().assign(address); });
  }
  return success();
}

// A selected address identifies a consumption, not the Q number alone. Never
// reuse a get from an unrelated branch or loop just because it has the same Q.
static LogicalResult rewriteVectorQueue(func::FuncOp function,
                                        ArrayRef<Value> members, unsigned queue,
                                        Value metadata, Value pool,
                                        IRRewriter &rewriter) {
  Operation *site = nullptr;
  auto result = function.walk<WalkOrder::PreOrder>([&](Operation *op)
                                                       -> WalkResult {
    if (!isa<arith::SelectOp, scf::IfOp>(op))
      return WalkResult::advance();
    TcbPoolGetOp get;
    for (Value selected : op->getResults()) {
      if (!isa<MemRefType>(selected.getType()) || selected.use_empty() ||
          !isPoolAddress(selected, members))
        continue;
      if (site && site != op)
        return op->emitError("expected one address selection for TCB queue ")
                   << queue,
               WalkResult::interrupt();
      if (!isMemoryEffectFree(op))
        return op->emitError("TCB address selection must have no side effects"),
               WalkResult::interrupt();
      site = op;
      rewriter.setInsertionPoint(op);
      if (!get)
        get = rewriter.create<TcbPoolGetOp>(
            op->getLoc(), rewriter.getI64Type(), rewriter.getI32Type(),
            metadata, pool, rewriter.getI32IntegerAttr(queue));
      Value address = rewriter.create<PointerCastOp>(
          op->getLoc(), selected.getType(), ValueRange{get.getAddress()},
          ValueRange{});
      rewriter.replaceAllUsesWith(selected, address);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  if (!site)
    return function.emitError("cannot find address selection for TCB queue ")
           << queue;
  return success();
}

// Scan the original annotations once for this side. All insertion positions and
// new operands stay local; no tags or cross-pattern state are needed.
static LogicalResult rewritePoolSide(func::FuncOp function, ArrayAttr groups,
                                     const PoolInsertionConfig &config) {
  SmallVector<Value, 2> members[2];
  PointerCastOp declaration;
  auto result = function.walk([&](annotation::MarkOp mark) -> WalkResult {
    auto queue = getPoolQueue(mark, groups);
    if (!queue)
      return WalkResult::advance();
    Value member = rootBuffer(mark.getSrc());
    auto pointer = member.getDefiningOp<PointerCastOp>();
    if (!pointer)
      return mark.emitError("expected a planned TCB pointer_cast"),
             WalkResult::interrupt();
    if (declaration && pointer->getBlock() != declaration->getBlock())
      return mark.emitError("expected TCB declarations in the same block"),
             WalkResult::interrupt();
    if (!declaration || pointer->isBeforeInBlock(declaration))
      declaration = pointer;
    if (!llvm::is_contained(members[*queue], member))
      members[*queue].push_back(member);
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  if (!declaration)
    return success();
  if (members[0].empty() || members[1].empty())
    return function.emitError("missing members of a planned TCB queue");
  Block *body = findInitBlock(function, declaration.getResult());
  if (!body)
    return function.emitError("cannot find the enclosing SSBUF block");

  IRRewriter rewriter(function.getContext());
  // Start alongside the existing SSBUF declarations, after the entry setup.
  auto firstSSBuffer =
      llvm::find_if(body->getOps<PointerCastOp>(), [](auto op) {
        return getHIVMAddressSpace(op.getType()) == AddressSpace::SSBUF;
      });
  rewriter.setInsertionPoint(*firstSSBuffer);
  auto metaType = MemRefType::get(
      {config.metadataWords()}, rewriter.getI32Type(),
      MemRefLayoutAttrInterface{},
      AddressSpaceAttr::get(rewriter.getContext(), AddressSpace::SSBUF));
  Value address = rewriter.create<arith::ConstantIntOp>(
      declaration.getLoc(), config.metadataOffset, 64);
  Value metadata = rewriter.create<PointerCastOp>(
      declaration.getLoc(), metaType, ValueRange{address}, ValueRange{});
  bool isCube = queryFuncCoreType(function) == TFuncCoreType::AIC;
  if (isCube) {
    uint32_t flagMask = 0;
    for (int32_t id : config.flagIds)
      flagMask |= uint32_t(1) << id;
    SmallVector<int32_t, 2> demands;
    for (int64_t bytes : config.queueBytes)
      demands.push_back(bytes / config.slotBytes);
    rewriter.create<TcbPoolInitOp>(
        declaration.getLoc(), metadata,
        rewriter.getI32IntegerAttr(config.slotBytes),
        rewriter.getI32IntegerAttr(config.poolBytes / config.slotBytes),
        rewriter.getDenseI32ArrayAttr(demands),
        rewriter.getDenseI32ArrayAttr(config.searchBegin),
        rewriter.getDenseI32ArrayAttr(config.searchEnd),
        rewriter.getI32IntegerAttr(flagMask));
    // Finish before the trailing entry-control restoration, if present.
    Operation *finish = body->getTerminator();
    while (finish->getPrevNode() && isa<SetCtrlOp>(finish->getPrevNode()))
      finish = finish->getPrevNode();
    rewriter.setInsertionPoint(finish);
    rewriter.create<TcbPoolReleaseOp>(declaration.getLoc(), metadata);
  }
  rewriter.setInsertionPoint(declaration);
  auto poolType = MemRefType::get(
      {config.poolBytes}, rewriter.getI8Type(), MemRefLayoutAttrInterface{},
      AddressSpaceAttr::get(rewriter.getContext(), AddressSpace::UB));
  Value pool = rewriter.create<PointerCastOp>(
      declaration.getLoc(), poolType, declaration.getAddrs(), ValueRange{});
  for (unsigned q = 0; q < 2; ++q) {
    auto status = isCube ? rewriteCubeQueue(function, members[q], q, metadata,
                                            pool, rewriter)
                         : rewriteVectorQueue(function, members[q], q, metadata,
                                              pool, rewriter);
    if (failed(status))
      return failure();
  }
  return success();
}

// Follow the newly returned address through pointer casts, views and if yields.
// Keep real accesses; address-only operations do not need a data-ready wait.
static FailureOr<SmallVector<Operation *>> findPoolBufferUsers(Value address) {
  Block *block = address.getParentBlock();
  SmallVector<Value> worklist{address};
  DenseSet<Value> seen;
  SmallVector<Operation *> accesses;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        if (auto branch = dyn_cast<scf::IfOp>(yield->getParentOp())) {
          for (auto [index, operand] : llvm::enumerate(yield.getOperands()))
            if (operand == value)
              worklist.push_back(branch.getResult(index));
          continue;
        }
      }
      if (isa<PointerCastOp, ViewLikeOpInterface, arith::SelectOp>(user)) {
        for (Value result : user->getResults())
          if (isa<MemRefType>(result.getType()))
            worklist.push_back(result);
        continue;
      }
      Operation *top = user;
      while (top && top->getBlock() != block)
        top = top->getParentOp();
      if (!top || top == block->getTerminator()) {
        user->emitError("TCB address escapes its allocation/get block");
        return failure();
      }
      if (!llvm::is_contained(accesses, user))
        accesses.push_back(user);
    }
  }
  return accesses;
}

// Step 3 derives synchronization sites from the new address uses, so neither
// FIX operations nor first/last consumer accesses need temporary tags.
static LogicalResult insertPoolSynchronization(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  auto cube = TCoreTypeAttr::get(rewriter.getContext(), TCoreType::CUBE);
  auto vector = TCoreTypeAttr::get(rewriter.getContext(), TCoreType::VECTOR);
  auto fixPipe = PipeAttr::get(rewriter.getContext(), PIPE::PIPE_FIX);
  auto vectorPipe = PipeAttr::get(rewriter.getContext(), PIPE::PIPE_V);
  auto result = module.walk([&](Operation *transfer) -> WalkResult {
    auto alloc = dyn_cast<TcbPoolAllocOp>(transfer);
    auto get = dyn_cast<TcbPoolGetOp>(transfer);
    if (!alloc && !get)
      return WalkResult::advance();
    Value id = alloc ? alloc.getId() : get.getId();
    Value address = alloc ? alloc.getAddress() : get.getAddress();
    auto accesses = findPoolBufferUsers(address);
    if (failed(accesses))
      return WalkResult::interrupt();
    rewriter.setInsertionPointAfter(transfer);
    Value flag = rewriter.create<arith::ExtUIOp>(transfer->getLoc(),
                                                 rewriter.getI64Type(), id);
    if (alloc) {
      for (Operation *access : *accesses)
        if (isa<FixpipeOp>(access)) {
          rewriter.setInsertionPointAfter(access);
          rewriter.create<SyncBlockSetOp>(access->getLoc(), cube, fixPipe,
                                          vectorPipe, OpFoldResult(flag));
        }
      return WalkResult::advance();
    }
    Operation *first = nullptr, *last = nullptr;
    for (Operation *access : *accesses) {
      while (access->getBlock() != transfer->getBlock())
        access = access->getParentOp();
      if (!first || access->isBeforeInBlock(first))
        first = access;
      if (!last || last->isBeforeInBlock(access))
        last = access;
    }
    if (first)
      rewriter.setInsertionPoint(first);
    auto wait = rewriter.create<SyncBlockWaitOp>(
        transfer->getLoc(), vector, fixPipe, vectorPipe, OpFoldResult(flag));
    rewriter.setInsertionPointAfter(last ? last : wait.getOperation());
    rewriter.create<SyncBlockSetOp>(transfer->getLoc(), vector, vectorPipe,
                                    fixPipe, OpFoldResult(flag));
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

// Remove only the replaced address chains and their TCB annotations. Other
// operations, loop arguments and scheduler results are left unchanged.
static LogicalResult cleanupReplacedBuffers(ModuleOp module, ArrayAttr groups) {
  IRRewriter rewriter(module.getContext());
  SmallVector<Operation *> candidates;
  DenseSet<Operation *> seen;
  auto add = [&](Operation *op) {
    if (seen.insert(op).second)
      candidates.push_back(op);
  };
  SmallVector<Operation *> declarations;
  SmallVector<Value> worklist;
  module.walk([&](annotation::MarkOp mark) {
    if (!getPoolQueue(mark, groups))
      return;
    Value member = rootBuffer(mark.getSrc());
    declarations.push_back(member.getDefiningOp());
    worklist.push_back(member);
  });
  DenseSet<Value> seenValues;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seenValues.insert(value).second)
      continue;
    add(value.getDefiningOp());
    for (Operation *user : llvm::make_early_inc_range(value.getUsers())) {
      if (isa<annotation::MarkOp>(user)) {
        rewriter.eraseOp(user);
        continue;
      }
      // A yield keeps a pure address-selection if alive; erase the if as a
      // unit once its selected result is unused, never erase the terminator.
      Operation *forwarder = user;
      if (isa<scf::YieldOp>(user))
        forwarder = user->getParentOp();
      if (!isa<ViewLikeOpInterface, arith::SelectOp, scf::IfOp>(forwarder) ||
          !isMemoryEffectFree(forwarder))
        continue;
      add(forwarder);
      for (Value result : forwarder->getResults())
        if (isa<MemRefType>(result.getType()))
          worklist.push_back(result);
    }
  }
  bool changed;
  do {
    changed = false;
    for (Operation *&op : candidates) {
      // PointerCastOp is an address declaration, but its generic effects are
      // conservative. An unused, replaced declaration can be erased here.
      if (!op || !op->use_empty() ||
          (!isa<PointerCastOp>(op) && !isMemoryEffectFree(op)))
        continue;
      Operation *dead = op;
      dead->walk([&](Operation *nested) {
        for (Operation *&candidate : candidates)
          if (candidate == nested)
            candidate = nullptr;
      });
      rewriter.eraseOp(dead);
      changed = true;
    }
  } while (changed);
  for (Operation *declaration : declarations)
    if (llvm::is_contained(candidates, declaration))
      return module.emitError("replaced TCB buffer still has uses");
  return success();
}

static LogicalResult preparePoolConfig(ModuleOp module, DictionaryAttr plan,
                                       PoolInsertionConfig &config) {
  config.poolBytes = plan.getAs<IntegerAttr>("pool_bytes").getInt();
  config.slotBytes = plan.getAs<IntegerAttr>("slot_bytes").getInt();
  auto groups = plan.getAs<ArrayAttr>("queues");
  config.queueBytes.assign(2, 0);
  module.walk([&](annotation::MarkOp mark) {
    auto queue = getPoolQueue(mark, groups);
    if (queue && queryFuncCoreType(mark->getParentOfType<func::FuncOp>()) ==
                     TFuncCoreType::AIV)
      config.queueBytes[*queue] = llvm::divideCeil(
          *GetBufferBitSize(rootBuffer(mark.getSrc())), int64_t{8});
  });
  if (!config.queueBytes[0] || !config.queueBytes[1])
    return module.emitError("missing Vector TCB queue sizes");
  if (failed(setQueueSearchRanges(module, config)) ||
      failed(collectSynchronization(module, config)))
    return failure();
  return reserveMetadata(module, config);
}

struct TcbPoolOpInsertionPass
    : impl::TcbPoolOpInsertionBase<TcbPoolOpInsertionPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto attr = module->getAttrOfType<DictionaryAttr>("hivm.tcb_pool_plan");
    if (!attr)
      return;
    PoolInsertionConfig config;
    if (failed(preparePoolConfig(module, attr, config))) {
      signalPassFailure();
      return;
    }

    removeLegacySynchronization(module);
    auto groups = attr.getAs<ArrayAttr>("queues");
    for (auto function : module.getOps<func::FuncOp>())
      if (failed(rewritePoolSide(function, groups, config))) {
        signalPassFailure();
        return;
      }
    if (failed(insertPoolSynchronization(module)) ||
        failed(cleanupReplacedBuffers(module, groups))) {
      signalPassFailure();
      return;
    }
    module->removeAttr("hivm.tcb_pool_plan");
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createTcbPoolOpInsertionPass() {
  return std::make_unique<TcbPoolOpInsertionPass>();
}

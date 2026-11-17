//===- TcbPoolPlan.cpp - TCB storage planning fallback --------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/TcbPoolPlan.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/PlanMemory.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

using namespace mlir;
using namespace mlir::hivm;

namespace {
// Trace view-like operations back to their storage source. Keep if/select
// results intact so queue analysis can inspect their alternatives separately.
Value rootBuffer(Value value) {
  auto roots = utils::tracebackMemRefVecByTargetFn(value, [](Value source) {
    return !source.getDefiningOp<ViewLikeOpInterface>();
  });
  return roots.size() == 1 ? roots.front() : value;
}

} // namespace

// Check pool eligibility without analyzing members or selecting a capacity.
bool TcbPoolPlan::canTryPool(func::FuncOp function) {
  auto module = function->getParentOfType<ModuleOp>();
  if (!module || !isMixModule(module))
    return false;
  auto enabled = module->getAttrOfType<BoolAttr>("hivm.dynamic_plan");
  return queryFuncCoreType(function) == TFuncCoreType::AIV && enabled &&
         enabled.getValue();
}

// Validate UB TCB marks on the AIC/AIV pair and trace their alloc/pointer_cast
// sources. Count AIC declarations and record AIV member IDs and sizes for UB
// planning. Require four static contiguous members per core without automatic
// multibuffering.
LogicalResult TcbPoolPlan::collectMembers(ModuleOp module) {
  unsigned cubeMemberCount = 0;
  auto result = module.walk([&](annotation::MarkOp mark) -> WalkResult {
    auto tcb = mark->getAttrOfType<HIVMTightlyCoupledBufferAttr>(
        HIVMTightlyCoupledBufferAttr::name);
    if (!tcb || getOptionalHIVMAddressSpace(mark.getSrc().getType()) !=
                    AddressSpace::UB)
      return WalkResult::advance();
    auto func = mark->getParentOfType<func::FuncOp>();
    auto core = queryFuncCoreType(func);
    Value buffer = rootBuffer(mark.getSrc());
    Operation *declaration = buffer.getDefiningOp();
    if (!declaration || !isa<memref::AllocOp, PointerCastOp>(declaration) ||
        !tcb.getId())
      return mark.emitError(
                 "TCB member must have an ID and originate from alloc or "
                 "pointer_cast"),
             WalkResult::interrupt();
    auto type = cast<MemRefType>(buffer.getType());
    if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
        mark->hasAttr(MultiBufferAttr::name))
      return mark.emitError("TCB members must have static contiguous storage "
                            "without automatic multibuffering"),
             WalkResult::interrupt();
    if (core == TFuncCoreType::AIC) {
      ++cubeMemberCount;
    } else if (core == TFuncCoreType::AIV) {
      int64_t bytes = llvm::divideCeil(*GetBufferBitSize(buffer), int64_t{8});
      vectorMembers.push_back({mark, buffer, *tcb.getId(), bytes});
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  if (cubeMemberCount != 4 || vectorMembers.size() != 4)
    return module.emitError("dynamic plan requires exactly four logical UB TCB "
                            "buffer IDs on AIC/AIV");
  return success();
}

// Infer the two ping-pong queues from AIV if/select alternatives. Each queue
// records its two member IDs and the size of one member. Repeated selections
// identify the same queue; overlapping pairs are rejected because their queue
// ownership is ambiguous.
LogicalResult TcbPoolPlan::buildQueues(ModuleOp module) {
  DenseMap<Value, const Member *> members;
  for (const Member &member : vectorMembers)
    members[member.buffer] = &member;
  // Add this selection to its existing pair, or create a new queue for it.
  auto pair = [&](Value a, Value b, Value selected) -> LogicalResult {
    const Member *first = members.lookup(rootBuffer(a));
    const Member *second = members.lookup(rootBuffer(b));
    // Ignore unrelated selections and selections of the same member.
    if (first == second)
      return success();
    if (!first || !second)
      return selected.getDefiningOp()->emitError(
          "TCB selection must pair two members");
    if (selected.getType() != first->buffer.getType() ||
        first->buffer.getType() != second->buffer.getType() ||
        !isMemoryEffectFree(selected.getDefiningOp()))
      return selected.getDefiningOp()->emitError(
          "TCB selection must preserve whole storage without side effects");
    SmallVector<int32_t, 2> group{first->id, second->id};
    llvm::sort(group);
    for (const Queue &q : queues) {
      if (q.ids == group)
        return success();
      if (llvm::is_contained(q.ids, first->id) ||
          llvm::is_contained(q.ids, second->id))
        return selected.getDefiningOp()->emitError(
            "ambiguous TCB queue grouping");
    }
    Queue q;
    q.ids = group;
    q.bytes = first->bytes;
    queues.push_back(std::move(q));
    return success();
  };
  auto vectorFunction =
      vectorMembers.front().mark->getParentOfType<func::FuncOp>();
  auto result = vectorFunction.walk([&](Operation *op) -> WalkResult {
    if (auto select = dyn_cast<arith::SelectOp>(op))
      return failed(pair(select.getTrueValue(), select.getFalseValue(), select))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (!branch.getElseRegion().empty())
        for (unsigned i = 0; i < branch.getNumResults(); ++i)
          if (failed(pair(branch.thenYield().getOperand(i),
                          branch.elseYield().getOperand(i),
                          branch.getResult(i))))
            return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  if (queues.size() != 2)
    return module.emitError(
        "dynamic plan requires two pairs selected by if/select");
  return success();
}

// Prepare a fresh plan: collect members and infer their queues. Use the GCD
// of the two queue member sizes as the allocation slot size, requiring it to
// be a multiple of 32 bytes. Address-ring depth is a runtime protocol detail.
LogicalResult TcbPoolPlan::analyze(ModuleOp module) {
  if (failed(collectMembers(module)) || failed(buildQueues(module)))
    return failure();
  slotBytes = std::gcd(queues[0].bytes, queues[1].bytes);
  if (slotBytes % 32)
    return module.emitError("TCB GCD slot size must be 32-byte aligned");
  return success();
}

// Compute one capacity on demand after analyze(): attempt zero removes the
// smallest member's size from the original total, attempt one removes the two
// smallest, and so on. Return no value if no positive capacity remains or any
// queue's member is larger than that capacity. This does not update poolBytes.
std::optional<int64_t> TcbPoolPlan::getCandidateSize(unsigned attempt) const {
  // Each side contains the same logical IDs; count the AIV members once.
  SmallVector<int64_t, 4> sizes;
  for (const Member &member : vectorMembers)
    sizes.push_back(member.bytes);
  llvm::sort(sizes);

  if (attempt >= sizes.size())
    return std::nullopt;
  // Each attempt drops one more member, starting with the smallest.
  auto remaining = ArrayRef<int64_t>(sizes).drop_front(attempt + 1);
  int64_t bytes =
      std::accumulate(remaining.begin(), remaining.end(), int64_t(0));
  // The pool must fit at least one allocation from each queue individually.
  // Later attempts only shrink the pool, so none can fit once this fails.
  if (bytes <= 0 ||
      llvm::any_of(queues, [&](const Queue &q) { return q.bytes > bytes; }))
    return std::nullopt;
  return bytes;
}

LogicalResult TcbPoolPlan::prepareAttempt(func::FuncOp function,
                                          unsigned attempt) {
  if (attempt == 0 && failed(analyze(function->getParentOfType<ModuleOp>())))
    return failure();
  auto bytes = getCandidateSize(attempt);
  if (!bytes)
    return function.emitError(
               "no valid TCB pool capacity for fallback attempt ")
           << attempt + 1;
  poolBytes = *bytes;
  return success();
}

// Prepare AIV liveness for the capacity selected by prepareAttempt(). Size
// all members to that capacity, remove their old inplace relationships and
// multibuffer counts, and link them to one anchor. The planner subsequently
// merges their storage entries and lifetimes, assigning every member the same
// base address. Preserve unique-storage requirements from any member; leave
// the IR itself unchanged. An unselected pool leaves liveness unchanged too.
LogicalResult TcbPoolPlan::adaptLiveness(MemLivenessAnalysis &liveness) {
  if (!poolBytes)
    return success();
  DenseSet<Value> members;
  bool memoryUnique = false;
  for (const Member &member : vectorMembers) {
    auto it = liveness.bufferInfos.find(member.buffer);
    if (it == liveness.bufferInfos.end() ||
        !liveness.buffer2Life.count(member.buffer))
      return member.mark->emitError("TCB member is missing from liveness");
    memoryUnique |= it->second.memoryUnique;
    members.insert(member.buffer);
  }
  llvm::erase_if(liveness.inplacePairList, [&](const ValuePair &p) {
    return members.count(p.first) || members.count(p.second);
  });
  Value anchor = vectorMembers.front().buffer;
  for (Value buffer : members) {
    auto &info = liveness.bufferInfos[buffer];
    info.constBits = poolBytes * 8;
    info.ignoreInplace = true;
    info.memoryUnique = memoryUnique;
    info.cvMixId = -1;
    liveness.buffer2MultiNum.erase(buffer);
    if (buffer != anchor)
      liveness.inplacePairList.emplace_back(buffer, anchor);
  }
  return success();
}

// Publish a self-contained planning result after fallback succeeds. The queues
// array order defines queue indices; each entry contains its two logical TCB
// IDs. Insertion can locate the planned declarations by ID without depending on
// this class or repeating membership and queue analysis.
void TcbPoolPlan::recordPlan(ModuleOp module) const {
  Builder b(module.getContext());
  SmallVector<Attribute, 2> queueIds;
  for (const Queue &queue : queues)
    queueIds.push_back(b.getDenseI32ArrayAttr(queue.ids));
  module->setAttr(
      "hivm.tcb_pool_plan",
      b.getDictionaryAttr(
          {b.getNamedAttr("pool_bytes", b.getI64IntegerAttr(poolBytes)),
           b.getNamedAttr("slot_bytes", b.getI64IntegerAttr(slotBytes)),
           b.getNamedAttr("queues", b.getArrayAttr(queueIds))}));
}

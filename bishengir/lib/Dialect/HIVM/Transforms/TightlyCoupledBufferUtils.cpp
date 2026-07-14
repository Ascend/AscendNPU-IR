//===- TightlyCoupledBufferUtils.cpp --------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025~2026. All rights reserved.
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

#include "bishengir/Dialect/HIVM/Transforms/TightlyCoupledBufferUtils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::hivm;

bool mlir::hivm::isL1OrUBAlloc(memref::AllocOp alloc) {
  auto addressSpace = getOptionalHIVMAddressSpace(alloc.getMemref().getType());
  return addressSpace == AddressSpace::L1 || addressSpace == AddressSpace::UB;
}

std::optional<Operation *> mlir::hivm::getTightlyCoupledMark(Value memref) {
  return utils::getAnnotateOpWithAttr(memref,
                                      HIVMTightlyCoupledBufferAttr::name);
}

std::optional<int64_t>
mlir::hivm::getTightlyCoupledBufferId(memref::AllocOp alloc) {
  auto maybeMarked = getTightlyCoupledMark(alloc.getMemref());
  if (!maybeMarked.has_value())
    return std::nullopt;

  auto markOp = dyn_cast<annotation::MarkOp>(*maybeMarked);
  if (!markOp)
    return std::nullopt;

  auto tcbAttr = dyn_cast_if_present<HIVMTightlyCoupledBufferAttr>(
      markOp->getAttr(HIVMTightlyCoupledBufferAttr::name));
  if (!tcbAttr || !tcbAttr.getId().has_value())
    return std::nullopt;
  return tcbAttr.getId().value();
}

llvm::DenseSet<int64_t>
mlir::hivm::collectUsedTightlyCoupledIds(ArrayRef<memref::AllocOp> allocs) {
  llvm::DenseSet<int64_t> usedIds;
  for (memref::AllocOp alloc : allocs) {
    if (auto id = getTightlyCoupledBufferId(alloc))
      usedIds.insert(*id);
  }
  return usedIds;
}

int64_t
mlir::hivm::allocateNextTightlyCoupledId(llvm::DenseSet<int64_t> &usedIds) {
  int64_t nextId = 0;
  while (usedIds.contains(nextId))
    ++nextId;
  usedIds.insert(nextId);
  return nextId;
}

annotation::MarkOp
mlir::hivm::createTightlyCoupledBufferMark(OpBuilder &builder,
                                           memref::AllocOp alloc, int64_t id) {
  builder.setInsertionPointAfter(alloc);
  auto mark =
      builder.create<annotation::MarkOp>(alloc.getLoc(), alloc.getMemref());
  // Preserve the f9/RegBase printing convention used by lit tests and
  // PlanMemory consumers (`effects = ["write", "read"]` in the attr-dict).
  mark->setAttr("effects", builder.getStrArrayAttr(
                               llvm::ArrayRef<StringRef>{"write", "read"}));
  mark->setAttr(HIVMTightlyCoupledBufferAttr::name,
                HIVMTightlyCoupledBufferAttr::get(alloc->getContext(),
                                                  static_cast<int32_t>(id)));
  return mark;
}

Value mlir::hivm::getTightlyCoupledViewSource(Operation *op) {
  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op))
    return toTensor.getMemref();
  if (auto spaceCast = dyn_cast<memref::MemorySpaceCastOp>(op))
    return spaceCast.getSource();
  if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(op))
    return viewLikeOp.getViewSource();
  if (auto memCast = dyn_cast<memref::CastOp>(op))
    return memCast.getSource();
  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op))
    return expand.getSrc();
  if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op))
    return collapse.getSrc();
  if (auto tensorCast = dyn_cast<tensor::CastOp>(op))
    return tensorCast.getSource();
  return {};
}

bool mlir::hivm::tracesToTightlyCoupledValue(Value v, Value root) {
  while (v) {
    if (v == root)
      return true;
    Operation *def = v.getDefiningOp();
    if (!def)
      return false;
    v = getTightlyCoupledViewSource(def);
  }
  return false;
}

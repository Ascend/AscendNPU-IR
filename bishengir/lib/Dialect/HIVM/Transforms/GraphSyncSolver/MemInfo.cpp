//===------------- MemInfo.cpp ---- Graph Sync Solver ---------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/MemInfo.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace mlir;
using namespace hivm::syncsolver;

namespace mlir::hivm::syncsolver {

llvm::SmallVector<int64_t> getAddresses(const llvm::SmallVector<Value> &addrs) {
  llvm::SmallVector<int64_t> offsets;
  for (auto addr : addrs) {
    auto constOp =
        llvm::dyn_cast_if_present<arith::ConstantOp>(addr.getDefiningOp());
    if (!constOp) {
      offsets.push_back(ShapedType::kDynamic);
      continue;
    }
    auto baseAddr =
        static_cast<int64_t>(cast<IntegerAttr>(constOp.getValue()).getInt());
    int64_t baseAddrInBits = baseAddr * utils::kBitsToByte;
    offsets.push_back(baseAddrInBits);
  }
  return offsets;
}

PointerLikeInfo getPointerLikeInfo(hivm::PointerCastOp pointerCastOp) {
  PointerLikeInfo pointerLikeInfo(pointerCastOp);
  pointerLikeInfo.addresses = getAddresses(pointerCastOp.getAddrs());
  pointerLikeInfo.allocateSize = GetBufferBitSize(pointerCastOp.getResult());
  if (!pointerLikeInfo.allocateSize.has_value()) {
    pointerCastOp.emitError("unknown buffer size");
    llvm_unreachable("unknown buffer size");
  }
  if (auto spaceAttr = GetBufferSpaceAttr(pointerCastOp.getResult())) {
    pointerLikeInfo.addressSpace = spaceAttr->getAddressSpace();
  }
  if (auto parentLoop = pointerCastOp->getParentOfType<LoopLikeOpInterface>()) {
    pointerLikeInfo.parentLoop = parentLoop;
  }
  return pointerLikeInfo;
}

PointerLikeInfo
getPointerLikeInfo(bishengir::memref_ext::AllocWorkspaceOp allocWorkspaceOp) {
  PointerLikeInfo pointerLikeInfo(allocWorkspaceOp);
  pointerLikeInfo.addresses = getAddresses(allocWorkspaceOp.getOffset());
  pointerLikeInfo.allocateSize = GetBufferBitSize(allocWorkspaceOp.getResult());
  if (!pointerLikeInfo.allocateSize.has_value()) {
    allocWorkspaceOp.emitError("unknown buffer size");
    llvm_unreachable("unknown buffer size");
  }
  pointerLikeInfo.addressSpace = hivm::AddressSpace::GM;
  if (auto parentLoop =
          allocWorkspaceOp->getParentOfType<LoopLikeOpInterface>()) {
    pointerLikeInfo.parentLoop = parentLoop;
  }
  return pointerLikeInfo;
}

MemInfo getMemInfo(Value val) {
  if (auto *defOp = val.getDefiningOp()) {
    if (auto allocWorkSpaceOp =
            llvm::dyn_cast<bishengir::memref_ext::AllocWorkspaceOp>(defOp)) {
      return MemInfo(val, getPointerLikeInfo(allocWorkSpaceOp),
                     /*isWorkSpace=*/true);
    }
    if (auto pointerCastOp = llvm::dyn_cast<hivm::PointerCastOp>(defOp)) {
      return MemInfo(val, getPointerLikeInfo(pointerCastOp));
    }
  }
  return MemInfo(val, isWorkSpaceFuncArgument(val));
}

MemInfo getMemInfo(const llvm::SmallVector<int64_t> &addrs) {
  MemInfo memInfo;
  memInfo.pointerLikeInfo = PointerLikeInfo();
  memInfo.pointerLikeInfo->addresses = addrs;
  memInfo.pointerLikeInfo->allocateSize = 1;
  memInfo.pointerLikeInfo->addressSpace = hivm::AddressSpace::Zero;
  return memInfo;
}

bool PointerLikeInfo::checkConflict(const PointerLikeInfo &pointerLikeInfo1,
                                    const PointerLikeInfo &pointerLikeInfo2,
                                    std::optional<int64_t> lcmLen,
                                    std::optional<int64_t> eventIdNum) {
  if (!pointerLikeInfo1.addressSpace.has_value() ||
      !pointerLikeInfo2.addressSpace.has_value()) {
    return false;
  }
  if (pointerLikeInfo1.addressSpace.value() !=
      pointerLikeInfo2.addressSpace.value()) {
    return false;
  }

  auto &offsets1 = pointerLikeInfo1.addresses;
  auto &offsets2 = pointerLikeInfo2.addresses;
  auto sz1 = static_cast<int64_t>(offsets1.size());
  auto sz2 = static_cast<int64_t>(offsets2.size());

  int64_t len1 = sz1;
  int64_t len2 = sz2;
  if (lcmLen.has_value()) {
    len1 = lcmLen.value();
    len2 = lcmLen.value();
  }

  for (int64_t i = 0; i < len1; i++) {
    for (int64_t j = 0; j < len2; j++) {
      if (eventIdNum.has_value()) {
        if ((i % eventIdNum.value()) == (j % eventIdNum.value())) {
          continue;
        }
      }

      auto offset1 = offsets1[i % sz1];
      auto offset2 = offsets2[j % sz2];
      if (offset1 == ShapedType::kDynamic || offset2 == ShapedType::kDynamic) {
        return true;
      }

      assert(pointerLikeInfo1.allocateSize.has_value());
      assert(pointerLikeInfo2.allocateSize.has_value());
      auto allocSz1 = pointerLikeInfo1.allocateSize.value();
      auto allocSz2 = pointerLikeInfo2.allocateSize.value();

      if ((allocSz1 != ShapedType::kDynamic) &&
          (offset1 + allocSz1 < offset2 + 1)) {
        continue;
      }
      if ((allocSz2 != ShapedType::kDynamic) &&
          (offset2 + allocSz2 < offset1 + 1)) {
        continue;
      }
      return true;
    }
  }
  return false;
}

bool MemInfo::checkConflict(const MemInfo &memInfo1, const MemInfo &memInfo2,
                            std::optional<int64_t> lcmLen,
                            std::optional<int64_t> eventIdNum) {
  if (memInfo1.pointerLikeInfo.has_value() &&
      memInfo2.pointerLikeInfo.has_value()) {
    return PointerLikeInfo::checkConflict(memInfo1.pointerLikeInfo.value(),
                                          memInfo2.pointerLikeInfo.value(),
                                          lcmLen, eventIdNum);
  }
  return memInfo1.value == memInfo2.value;
}

bool isWorkSpaceFuncArgument(Value value) {
  auto blockArg = dyn_cast_if_present<BlockArgument>(value);
  if (!blockArg) {
    return false;
  }
  auto *block = blockArg.getOwner();
  if (!block) {
    return false;
  }
  auto *region = block->getParent();
  if (!region) {
    return false;
  }
  auto *parentOp = region->getParentOp();
  if (!parentOp) {
    return false;
  }
  auto funcOp = dyn_cast<func::FuncOp>(parentOp);
  if (!funcOp) {
    return false;
  }
  return hacc::utils::isKernelArg(funcOp, blockArg.getArgNumber(),
                                  hacc::KernelArgType::kWorkspace);
}

} // namespace mlir::hivm::syncsolver

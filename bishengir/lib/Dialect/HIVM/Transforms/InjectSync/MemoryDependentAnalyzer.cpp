//===-------- MemoryDependentAnalyzer.h ----Sync dependency analysis ------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/MemoryDependentAnalyzer.h"

#define DEBUG_TYPE "hivm-inject-sync"

using namespace mlir;
using namespace mlir::hivm;

bool MemoryDependentAnalyzer::DepBetween(
    const SmallVector<const BaseMemInfo *> &a,
    const SmallVector<const BaseMemInfo *> &b,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  bool hasAlias = false;
  for (auto &i : a) {
    for (auto &j : b) {
      if (MemAlias(i, j)) {
        // Found dependency conflict between i and j, record the pair.
        depBaseMemInfosVec.push_back(std::make_pair(i, j));
        hasAlias = true;
      }
    }
  }
  return hasAlias;
}

bool MemoryDependentAnalyzer::MemAlias(const BaseMemInfo *a,
                                       const BaseMemInfo *b) {
  assert(a != nullptr && b != nullptr);
  hivm::AddressSpace addressSpaceA = a->addressSpace;
  hivm::AddressSpace addressSpaceB = b->addressSpace;
  if (addressSpaceA != addressSpaceB) {
    return false;
  }
  if (addressSpaceA == hivm::AddressSpace::GM &&
      addressSpaceB == hivm::AddressSpace::GM) {
    return isGMBufferOverlap(a, b);
  }
  if (a->rootBuffer == b->rootBuffer) {
    return true;
  }
  return isBufferAddressRangeOverlap(a, b);
}

bool MemoryDependentAnalyzer::isGMBufferOverlap(const BaseMemInfo *a,
                                                const BaseMemInfo *b) {
  assert(a != nullptr && b != nullptr);
  if (a->rootBuffer != b->rootBuffer) {
    // Different buffers on GM have no dependencies.
    // TODO: handle gm alias cases like inplace
    return false;
  }
  if (a->allocWorkspaceOp.has_value() && b->allocWorkspaceOp.has_value()) {
    return isBufferAddressRangeOverlap(a, b);
  }
  return true;
}

bool MemoryDependentAnalyzer::isBufferAddressRangeOverlap(
    const BaseMemInfo *a, const BaseMemInfo *b) {
  assert(a != nullptr && b != nullptr);
  if (a->hasVariableAddress || b->hasVariableAddress) {
    // conservatively assume overlap if any buffer has variable address
    return true;
  }
  size_t baseAddressesSizeA = a->baseAddresses.size();
  size_t baseAddressesSizeB = b->baseAddresses.size();
  for (size_t i = 0; i < baseAddressesSizeA; i++) {
    for (size_t j = 0; j < baseAddressesSizeB; j++) {
      if (isBufferOverlap(a, b, i, j)) {
        return true;
      }
    }
  }
  return false;
}

bool MemoryDependentAnalyzer::isBufferOverlap(const BaseMemInfo *a,
                                              const BaseMemInfo *b,
                                              uint32_t aIndex,
                                              uint32_t bIndex) {
  assert(a != nullptr && b != nullptr);
  assert(aIndex < a->baseAddresses.size());
  assert(bIndex < b->baseAddresses.size());
  /*
  (size_a != ShapedType::kDynamic && (a+size_a <= b)) || 
  (size_b != ShapedType::kDynamic && (b+size_b <= a)) ==> no overlap
  else overlap
  */
  if((a->allocateSize != ShapedType::kDynamic) &&
     (a->baseAddresses[aIndex] + a->allocateSize <= b->baseAddresses[bIndex])) {
    return false;
  }
  if((b->allocateSize != ShapedType::kDynamic) &&
     (b->baseAddresses[bIndex] + b->allocateSize <= a->baseAddresses[aIndex])) {
    return false;
  }
  return true;
}
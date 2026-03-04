//===-------- MemoryDependentAnalyzer.h ----Sync dependency analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/MemoryDependentAnalyzer.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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
        // Update the current sync dependency buffer.
        depBaseMemInfosVec.push_back(std::make_pair(i, j));
        hasAlias = true;
      }
    }
  }
  return hasAlias;
}

bool MemoryDependentAnalyzer::MemAlias(const BaseMemInfo *a,
                                       const BaseMemInfo *b) {
  hivm::AddressSpace as = a->scope;
  hivm::AddressSpace bs = b->scope;
  if (as == hivm::AddressSpace::GM && bs == hivm::AddressSpace::GM) {
    return isGMBufferOverlap(a, b);
  }
  // different scope, just return no dependency.
  if (as != bs) {
    return false;
  }
  if (a->rootBuffer == b->rootBuffer) {
    return true;
  }
  if (isa<memref::AllocOp>(a->rootBuffer.getDefiningOp()) &&
      isa<memref::AllocOp>(b->rootBuffer.getDefiningOp())) {
    return false;
  }
  return isBufferAddressRangeOverlap(a, b);
}

bool MemoryDependentAnalyzer::isGMBufferOverlap(const BaseMemInfo *a,
                                                const BaseMemInfo *b) {
  if (a->rootBuffer != b->rootBuffer) {
    // Different buffers on GM have no dependencies.
    // TODO: handle gm alias cases like inplace
    return false;
  } else {
    if (a->allocWorkspaceOp.has_value() && b->allocWorkspaceOp.has_value() &&
        !isBufferAddressRangeOverlap(a, b)) {
      return false;
    }
    return true;
  }
}

bool MemoryDependentAnalyzer::isBufferAddressRangeOverlap(
    const BaseMemInfo *a, const BaseMemInfo *b) {
  int aBaseAddressesSize = static_cast<int>(a->baseAddresses.size());
  int bBaseAddressesSize = static_cast<int>(b->baseAddresses.size());
  for (int i = 0; i < aBaseAddressesSize; i++) {
    for (int j = 0; j < bBaseAddressesSize; j++) {
      if (isBufferOverlap(a, b, i, j)) {
        return true;
      }
    }
  }
  return false;
}

bool MemoryDependentAnalyzer::isBufferOverlap(const BaseMemInfo *a,
                                              const BaseMemInfo *b, int aIndex,
                                              int bIndex) {
  if (a->baseAddresses[aIndex] == b->baseAddresses[bIndex]) {
    return true;
  }
  // There are overlapping dependency conflicts.
  if ((a->baseAddresses[aIndex] > b->baseAddresses[bIndex] &&
       a->baseAddresses[aIndex] < b->baseAddresses[bIndex] + b->allocateSize) ||
      (b->baseAddresses[bIndex] > a->baseAddresses[aIndex] &&
       b->baseAddresses[bIndex] < a->baseAddresses[aIndex] + a->allocateSize)) {
    return true;
  }
  return false;
}
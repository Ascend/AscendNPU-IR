//===- UnionFind.h - Union Find/Disjoint Set Impl. --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Utils/UnionFind.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "UnionFind"

int UnionFindBase::find(int x) {
  allocateMinimum(x);
  if (parent_[x] < 0)
    return x;
  return parent_[x] = find(parent_[x]);
}

bool UnionFindBase::join(int a, int b) {
  allocateMinimum(std::max(a, b));

  a = find(a);
  b = find(b);
  if (a != b) {
    if (parent_[a] > parent_[b])
      std::swap(a, b);
    parent_[a] += parent_[b];
    parent_[b] = a;
    minIndex[a] = std::min(minIndex[b], minIndex[a]);
  }
  return true;
}

void UnionFindBase::allocateMinimum(size_t n) {
  if (n + 1 > parent_.size()) {
    parent_.resize(n + 1, -1);
    size_t oldSize = minIndex.size();
    minIndex.resize(n + 1, -1);
    for (size_t i = oldSize; i < n + 1; ++i) {
      assert(minIndex[i] == -1);
      minIndex[i] = static_cast<int>(i);
    }
  }
}

void UnionFindBase::dump() {
  LLVM_DEBUG(
    llvm::dbgs() << "[UnionFind] parent_(" << parent_.size() << " entries):\n";
    for (size_t i = 0; i < parent_.size() - 1; ++i) {
      llvm::dbgs() << (parent_[i]<0 ? "" : " ") << parent_[i] << ", ";
    }
    llvm::dbgs() << (parent_[parent_.size()-1]<0 ? "" : " ") << parent_[parent_.size()-1] << "\n";
    llvm::dbgs() << "[UnionFind] minIndex(" << minIndex.size() << " entries):\n";
    for (size_t i = 0; i < minIndex.size() - 1; ++i) {
      llvm::dbgs() << (minIndex[i]<0 ? "" : " ") << minIndex[i] << ", ";
    }
    llvm::dbgs() << (minIndex[minIndex.size()-1]<0 ? "" : " ") << minIndex[minIndex.size()-1] << "\n";
  );
}

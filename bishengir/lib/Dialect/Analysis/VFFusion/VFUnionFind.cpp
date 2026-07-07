//===- VFUnionFind.cpp - Implementation of VF Union Find ------------------===//
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

#include "bishengir/Dialect/Analysis/VFFusion/VFUnionFind.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "vf-fusion-union-find"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {

VFUnionFind::VFUnionFind(ArrayRef<Operation *> opsInBlock)
    : UnionFindBase(opsInBlock.size()), totalSize(opsInBlock.size(), 0),
      maxIndex(opsInBlock.size(), 0) {
  std::iota(maxIndex.begin(), maxIndex.end(), 0);

  for (auto [i, op] : llvm::enumerate(opsInBlock)) {
    size_t cntNumberOps = 0;
    op->walk([&cntNumberOps](Operation *const _) { ++cntNumberOps; });
    totalSize[i] = cntNumberOps;
  }
}

size_t VFUnionFind::getSizeUnion(int x) {
  return -this->parent_[this->find(x)];
}

size_t VFUnionFind::getTotalSizeUnion(int x) {
  return totalSize[this->find(x)];
}

size_t VFUnionFind::getMaxIndexUnion(int x) { return maxIndex[this->find(x)]; }

bool VFUnionFind::join(int a, int b) {
  a = find(a);
  b = find(b);
  if (a != b) {
    if (parent_[a] < parent_[b])
      std::swap(a, b);
    parent_[a] += parent_[b];
    parent_[b] = a;
    minIndex[a] = std::min(minIndex[b], minIndex[a]);
    maxIndex[a] = std::max(maxIndex[b], maxIndex[a]);
    totalSize[a] += totalSize[b];
    totalSize[b] = 0;
    return true;
  }
  return false;
}

bool VFUnionFind::isConnected(int x, int y) {
  int px = this->find(x);
  int py = this->find(y);
  return px == py;
}

void VFUnionFind::allocateMinimum(size_t n) {
  if (n + 1 > parent_.size())
    llvm_unreachable("shouldn't allocate any new indices.");
}

} // namespace mlir::analysis
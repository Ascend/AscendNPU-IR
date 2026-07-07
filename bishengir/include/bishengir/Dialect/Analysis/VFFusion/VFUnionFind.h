//===- VFUnionFind.h ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_UNION_FIND_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_UNION_FIND_H

#include "bishengir/Dialect/Utils/UnionFind.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace analysis {

// TODO: refactor VFUnionFind to be template so that it can be specified for
// each fusion kind
class VFUnionFind : public UnionFindBase {
public:
  VFUnionFind() = default;
  explicit VFUnionFind(ArrayRef<Operation *> opsInBlock);

  // get the number of nodes of a union only count the outtermost layer
  size_t getSizeUnion(int x);

  // get the number of nodes of a union including the operations inside of its
  // region
  size_t getTotalSizeUnion(int x);

  // get max index from union index
  size_t getMaxIndexUnion(int x);

  // check if two nodes are connected
  bool isConnected(int x, int y);

  // return false if it has been fused.
  bool join(int a, int b) override;

  // shouldn't allocate any new indices.
  void allocateMinimum(size_t n) override;

protected:
  SmallVector<size_t> totalSize;

private:
  SmallVector<size_t> maxIndex;
};

} // namespace analysis
} // namespace mlir
#endif
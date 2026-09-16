//===- UnionFind.h - Union Find/Disjoint Set Impl. --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_UTILS_UNIONFIND_H
#define BISHENGIR_DIALECT_UTILS_UNIONFIND_H

#include <numeric>
#include <vector>

class UnionFindBase {
public:
  UnionFindBase(std::size_t n = 0) : minIndex(n), parent_(n, -1) {
    std::iota(minIndex.begin(), minIndex.end(), 0);
  }
  virtual ~UnionFindBase() = default;

  int find(int x);
  virtual bool join(int a, int b);
  virtual void allocateMinimum(std::size_t n);

public:
  std::vector<int> minIndex;

  void dump();

protected:
  std::vector<int> parent_;
};

#endif // BISHENGIR_DIALECT_UTILS_UNIONFIND_H

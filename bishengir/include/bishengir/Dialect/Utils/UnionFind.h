//===- UnionFind.h - Union Find/Disjoint Set Impl. --------------*- C++ -*-===//
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

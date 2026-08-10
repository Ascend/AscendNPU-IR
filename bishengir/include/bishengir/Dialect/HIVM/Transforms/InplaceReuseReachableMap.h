//===- InplaceReuseReachableMap.h -------------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INPLACE_REUSE_REACHABLE_MAP_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INPLACE_REUSE_REACHABLE_MAP_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace hivm {

/// Memoization cache intended to prevent recomputation of
/// IsInplaceReuseReachable when called with the same value.
class InplaceReuseReachableMap {
public:
  template <PIPE Pipe> void put(Value key, bool val) {
    key = find(key);
    if constexpr (Pipe == PIPE::PIPE_MTE3) {
      mte3Reachable[key] = val;
    } else if constexpr (Pipe == PIPE::PIPE_MTE2) {
      mte2Reachable[key] = val;
    } else {
      llvm::report_fatal_error("Unsupported pipe type");
    }
  }

  template <PIPE Pipe> std::optional<bool> get(Value key) {
    key = find(key);
    if constexpr (Pipe == PIPE::PIPE_MTE3) {
      auto iter = mte3Reachable.find(key);
      if (iter != mte3Reachable.end()) {
        return iter->second;
      }
    } else if constexpr (Pipe == PIPE::PIPE_MTE2) {
      auto iter = mte2Reachable.find(key);
      if (iter != mte2Reachable.end()) {
        return iter->second;
      }
    } else {
      llvm::report_fatal_error("Unsupported pipe type");
    }
    return std::nullopt;
  }

  void unite(Value val1, Value val2) {
    Value genRoot = find(val1);
    Value killRoot = find(val2);
    if (genRoot == killRoot)
      return;

    // propagate killRoot's reachability to genRoot before union
    auto killLoad = get<PIPE::PIPE_MTE2>(killRoot);
    auto killStore = get<PIPE::PIPE_MTE3>(killRoot);
    if (killLoad && killLoad.value())
      put<PIPE::PIPE_MTE2>(genRoot, true);
    if (killStore && killStore.value())
      put<PIPE::PIPE_MTE3>(genRoot, true);

    parent[killRoot] = genRoot;
  }

private:
  DenseMap<Value, bool> mte3Reachable;
  DenseMap<Value, bool> mte2Reachable;

  DenseMap<Value, Value> parent;

  Value find(Value val) {
    auto iter = parent.find(val);
    if (iter == parent.end()) {
      return parent[val] = val;
    }
    Value p = iter->getSecond();
    if (val == p)
      return p;
    return parent[val] = find(p);
  }
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INPLACE_REUSE_REACHABLE_MAP_H

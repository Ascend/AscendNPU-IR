//===- VFInplaceReuseReachability.h -----------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_VF_INPLACE_REUSE_REACHABILITY_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_VF_INPLACE_REUSE_REACHABILITY_H

#include "bishengir/Dialect/HIVM/Analysis/VFInplaceReuseAnalyzer.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace hivm {

/// Memoization cache intended to prevent recomputation of
/// IsInplaceReuseReachable when called with the same value.
class InplaceReuseReachableMap {
public:
  template <typename DstOpType>
  void put(Value key, bool val) {
    key = find(key);
    if constexpr (std::is_same_v<DstOpType, hivm::StoreOp>) {
      storeReachable[key] = val;
    } else if constexpr (std::is_same_v<DstOpType, hivm::LoadOp>) {
      loadReachable[key] = val;
    } else {
      llvm::report_fatal_error("Unsupported op type");
    }
  }

  template <typename DstOpType>
  std::optional<bool> get(Value key) {
    key = find(key);
    if constexpr (std::is_same_v<DstOpType, hivm::StoreOp>) {
      auto iter = storeReachable.find(key);
      if (iter != storeReachable.end()) {
        return iter->second;
      }
    } else if constexpr (std::is_same_v<DstOpType, hivm::LoadOp>) {
      auto iter = loadReachable.find(key);
      if (iter != loadReachable.end()) {
        return iter->second;
      }
    } else {
      llvm::report_fatal_error("Unsupported op type");
    }
    return std::nullopt;
  }

  void unite(Value val1, Value val2) {
    Value genRoot = find(val1);
    Value killRoot = find(val2);
    if (genRoot == killRoot)
      return;

    // propagate killRoot's reachability to genRoot before union
    auto killLoad = get<hivm::LoadOp>(killRoot);
    auto killStore = get<hivm::StoreOp>(killRoot);
    if (killLoad && killLoad.value())
      put<hivm::LoadOp>(genRoot, true);
    if (killStore && killStore.value())
      put<hivm::StoreOp>(genRoot, true);

    parent[killRoot] = genRoot;
  }

private:
  DenseMap<Value, bool> storeReachable;
  DenseMap<Value, bool> loadReachable;

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

inline bool defaultVFReachableOpCheck(Operation *op) {
  // Extra check for the reusable of UB
  // 1. Only reuse UB when VF is not in loop block.
  //    If VF appears in the loop block, membar(load wait store) is needed to
  //    insert before VF to reuse UB.
  return op->getParentOfType<LoopLikeOpInterface>();
}

inline bool noneVFReachableOpCheck(Operation *op) { return false; }

template <typename DstOpType>
inline bool VisitInplaceReuseReachable(
    Value src, VFCallInplaceReuseInfo *vfInfo, DenseSet<Value> &visited,
    InplaceReuseReachableMap &reachableMap,
    llvm::function_ref<bool(Operation *)> extraCheck = defaultVFReachableOpCheck) {
  // don't perform traversal if we already know that `src` is reachable or
  // unreachable
  if (auto computedReachable = reachableMap.get<DstOpType>(src)) {
    return computedReachable.value();
  }

  // skip `src` if it was already processed in previous recursion steps,
  // mark it as processed otherwise
  if (visited.contains(src)) {
    return false;
  }
  visited.insert(src);

  for (Operation *user : src.getUsers()) {
    if (isa<DstOpType>(user) && extraCheck(user)) {
      reachableMap.put<DstOpType>(src, true);
      return true;
    }

    if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
      // Recursively visit subview result users. If this path does not reach
      // the target, keep checking the other users of `src`.
      if (VisitInplaceReuseReachable<DstOpType>(
              subview.getResult(), vfInfo, visited, reachableMap, extraCheck)) {
        return true;
      }
      continue;
    }
    if (auto reshapeOp = dyn_cast<memref::CollapseShapeOp>(user)) {
      if (VisitInplaceReuseReachable<DstOpType>(reshapeOp.getResult(), vfInfo,
                                                visited, reachableMap,
                                                extraCheck)) {
        return true;
      }
      continue;
    }
    if (auto reshapeOp = dyn_cast<memref::ExpandShapeOp>(user)) {
      if (VisitInplaceReuseReachable<DstOpType>(reshapeOp.getResult(), vfInfo,
                                                visited, reachableMap,
                                                extraCheck)) {
        return true;
      }
      continue;
    }
  }

  reachableMap.put<DstOpType>(src, false);
  return false;
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_VF_INPLACE_REUSE_REACHABILITY_H

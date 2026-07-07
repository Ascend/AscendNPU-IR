//===--ValueDependencyAnalyzer.h ---------------------------------*- C++-*-===//
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

#ifndef BISHENGIR_DIALECT_UTILS_VALUEDEPENDENCYANALYZER_H
#define BISHENGIR_DIALECT_UTILS_VALUEDEPENDENCYANALYZER_H

#include "bishengir/Dialect/Utils/UnionFind.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"


namespace mlir {
namespace utils {
using ValueToIndexMap = DenseMap<Value, int>;

class ValueDependencyAnalyzer {

  using ValueToIndexMap = DenseMap<Value, int>;

public:
  // Construct the value dependency to the allocation for all operands including
  // function block arguments This things can works inter through blocks as well

  // to build value dependency of each value to its allocation (block arguments
  // or memref.alloc).
  void buildValueDependency(Operation *parent);
  // get the allocaction of a value
  Value getAllocOf(Value value);

  // topologically sorted index of the values
  ValueToIndexMap valueToIndexMap;
  SmallVector<Value> valueList;

  // vector of operations to an allocation chronologically
  SmallVector<SmallVector<Operation *>> OpsOfAlloc;

private:
  // push effects from MemoryEffectOpInterface operation
  void pushRelatedOpsOfAlloc(Operation *const op);

  // push values/memrefs and effects
  void pushAllValues(Operation *parent);

  // reset all values
  void reset();

  UnionFindBase dsu;
};

} // namespace utils
} // namespace mlir

#endif
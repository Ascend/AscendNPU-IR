//===--ValueDependencyAnalyzer.h ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

private:
  void pushAllValues(Operation *parent);
  // reset all values
  void reset();

  UnionFindBase dsu;
};

} // namespace utils
} // namespace mlir

#endif
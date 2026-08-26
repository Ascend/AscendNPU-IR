//===- ReachabilityAnalyzer.h -- Reachability Analyzer header ---*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_UTILS_REACHABILITYANALYZER_H
#define BISHENGIR_DIALECT_UTILS_REACHABILITYANALYZER_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace utils {
using OperationToIndexMap = DenseMap<Operation *, int>;
class ReachabilityAnalyzer {
  // Initialize reachabilityMatrix with max value
public:
  const int64_t kMaxDistance = std::numeric_limits<int64_t>::max();
  OperationToIndexMap opToIndexMap;
  SmallVector<Operation *> operationList;
  // This matrix is the distance between two operation,
  // and -1 if it's not reachable
  SmallVector<SmallVector<int64_t>> reachabilityMatrix;
  // Underlying edge of the graph
  SmallVector<SmallVector<int64_t>> edge;
  DenseMap<Operation *, DenseSet<Value>> cachedMemref;
  DenseMap<Value, int> lastOperationOnMemref;
  Operation *rootOp;

public:
  // Construct the reachability, this will loop all the operations (1 level only
  // below this parent), for example if the parent is a func, it will loop only
  // the body, if its a scf it will obly loop the body etc
  //
  // This things can works inter through blocks as well
  ReachabilityAnalyzer(Operation *parent);
  void getMemrefFromOp(Operation *op);
  int64_t getIndex(Operation *op);
  Operation *getOperation(int64_t index);
  inline SmallVector<int> getUsersOrRoot(Operation *op);
  // Returns the reachibility between two nodes
  bool isReachable(Operation *start, Operation *dest) const;
  bool isReverseReachable(Operation *start, Operation *dest) const;

  bool hasDataDependency(Operation *op1, Operation *op2) const;

  // Returns the reachability distance between two nodes
  int64_t getReachabilityDistance(Operation *start, Operation *dest) const;

  // Returns the list of lowest common ancestor in the DAG
  SmallVector<int> getLCA(Operation *start, Operation *dest) const;
  int64_t getShortestPathFromAncestor(Operation *start, Operation *dest) const;
  inline void getAndSetMemrefEdge(Operation *op, Value ref);

private:
  void computeReachabilityMatrix(size_t numOps);
  void initializeAdjacencyList(size_t numOps);
};

} // namespace utils
} // namespace mlir

#endif

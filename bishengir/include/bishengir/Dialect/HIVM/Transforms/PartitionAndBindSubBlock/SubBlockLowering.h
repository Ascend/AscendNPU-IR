//-----------------------------SubBlockLowering.h-----------------------------//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PARTITIONANDBINDSUBBLOCK_SUBBLOCKLOWERING_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PARTITIONANDBINDSUBBLOCK_SUBBLOCKLOWERING_H

#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/CoreDependencyAnalysis.h"
#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/PartitionTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace hivm {
namespace partition_and_bind {

class SubBlockLowering {
public:
  SubBlockLowering(func::FuncOp func, const CoreAssignment &assignment)
      : func(func), assignment(assignment) {}

  /// Lower every supernode then guard outside-scope single-core ops. 
  LogicalResult run();

  static void inlineSubBlockScopesAsFallback(func::FuncOp funcOp);

private:
  LogicalResult setFixpipeDestinations();

  LogicalResult lowerSupernode(const Supernode &node);

  LogicalResult guardOutsideScopeOp(Operation &op, Core core) const;

  func::FuncOp func;
  const CoreAssignment &assignment;

  llvm::DenseSet<Operation *> loweredScopeGuards;
};

} // namespace partition_and_bind
} // namespace hivm
} // namespace mlir

#endif // ...SUBBLOCKLOWERING_H

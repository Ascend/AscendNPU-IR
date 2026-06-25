//--------------------------CoreDependencyAnalysis.h--------------------------//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PARTITIONANDBINDSUBBLOCK_COREDEPENDENCYANALYSIS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PARTITIONANDBINDSUBBLOCK_COREDEPENDENCYANALYSIS_H

#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/PartitionTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hivm {
namespace partition_and_bind {

/// Decides which core a free node is placed on.
class FreeNodePlacementPolicy {
public:
  virtual ~FreeNodePlacementPolicy() = default;

  /// Choose a single core for a free top-level op.
  virtual Core placeFreeNode(Operation *op, unsigned currentV0Load,
                             unsigned currentV1Load) = 0;
};

/// Default policy: every free node goes to core 0.
class DefaultFreeNodePlacementPolicy : public FreeNodePlacementPolicy {
public:
  Core placeFreeNode(Operation *op, unsigned currentV0Load,
                     unsigned currentV1Load) override;
};

/// Load-balancing policy: place each free node on the currently lighter
/// sub-core.
class LoadBalancedFreeNodePlacementPolicy : public FreeNodePlacementPolicy {
public:
  Core placeFreeNode(Operation *op, unsigned currentV0Load,
                     unsigned currentV1Load) override;
};

struct CoreAssignment {
  llvm::DenseMap<const Operation *, Core> opCore;
  llvm::SmallVector<Supernode, 4> supernodes;
  /// The sub-block each single-destination fixpipe must deposit its cube result
  /// into, so its post-cube consumer reads the right UB. Absent => default
  /// sub-block 0.
  llvm::DenseMap<Operation *, unsigned> fixpipeSubBlock;

  llvm::SmallPtrSet<Operation *, 4> doubleWriteFixpipes;

  Core coreOf(Operation *op) const;
};

class CoreDependencyAnalysis {
public:
  CoreDependencyAnalysis(func::FuncOp func, FreeNodePlacementPolicy &policy)
      : func(func), policy(policy) {}

  CoreAssignment run();

private:
  void discoverSupernodes(CoreAssignment &out);

  void buildDependencyDag();

  void recordUse(Operation *consumer, Value operand,
                 llvm::SmallPtrSet<Operation *, 8> &seenProducers);

  void propagateCores(CoreAssignment &out);

  void placeFreeNodes(CoreAssignment &out);

  void deriveFixpipeDestinations(CoreAssignment &out);

  Core fixpipeDestination(Operation *fixpipe, const CoreAssignment &out) const;

  Operation *definingTopLevelOp(Value v) const;

  func::FuncOp func;
  FreeNodePlacementPolicy &policy;

  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *, 4>> producers;

  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *, 4>> consumers;
};

} // namespace partition_and_bind
} // namespace hivm
} // namespace mlir

#endif // COREDEPENDENCYANALYSIS_H

//===- TcbPoolPlan.h - TCB storage planning fallback ------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_TCBPOOLPLAN_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_TCBPOOLPLAN_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <optional>

namespace mlir::hivm {
class MemLivenessAnalysis;

/// TCB membership analysis and pool capacity planning for memory fallback.
/// Runtime scheduling and guards belong to a later pass.
class TcbPoolPlan {
public:
  /// Check whether the function is AIV and its module enables pool fallback.
  static bool canTryPool(func::FuncOp function);
  /// Analyze on the first fallback attempt and select this attempt's capacity.
  /// Attempts are zero-based and must be prepared in order.
  LogicalResult prepareAttempt(func::FuncOp function, unsigned attempt);
  /// Link AIV members for storage/lifetime merging at the selected capacity.
  LogicalResult adaptLiveness(MemLivenessAnalysis &liveness);
  /// Record the selected capacity, slot size and queue member IDs.
  /// Later passes consume the IR metadata without rerunning this analysis.
  void recordPlan(ModuleOp module) const;

private:
  /// Collect members and queues, and derive the slot size.
  LogicalResult analyze(ModuleOp module);
  /// Return the capacity for a zero-based fallback attempt, or no value when
  /// no smaller valid capacity remains. Does not change the selected pool.
  std::optional<int64_t> getCandidateSize(unsigned attempt) const;

  struct Member {
    annotation::MarkOp mark;
    Value buffer;
    int32_t id;
    int64_t bytes;
  };
  struct Queue {
    SmallVector<int32_t, 2> ids;
    int64_t bytes;
  };

  /// Count AIC TCB declarations and collect the four AIV members and sizes.
  LogicalResult collectMembers(ModuleOp module);
  /// Group AIV members into two queues using their if/select selections.
  LogicalResult buildQueues(ModuleOp module);

  SmallVector<Member, 4> vectorMembers;
  SmallVector<Queue, 2> queues;
  int64_t slotBytes = 0;
  int64_t poolBytes = 0;
};
} // namespace mlir::hivm

#endif

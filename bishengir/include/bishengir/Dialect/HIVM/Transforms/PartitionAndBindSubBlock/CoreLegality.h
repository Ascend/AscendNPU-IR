//-------------------------------CoreLegality.h-------------------------------//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PARTITIONANDBINDSUBBLOCK_CORELEGALITY_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PARTITIONANDBINDSUBBLOCK_CORELEGALITY_H

#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/PartitionTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace hivm {
namespace partition_and_bind {

/// Outcome of legality. On failure carries the first offending op so it
/// can emit a precise diagnostic and leave the func unchanged.
struct LegalityResult {
  bool ok = true;
  Operation *offendingOp = nullptr;
  std::string message;

  static LegalityResult success() { return {true, nullptr, {}}; }
  static LegalityResult failure(Operation *op, std::string msg) {
    return {false, op, std::move(msg)};
  }
  explicit operator bool() const { return ok; }
};

class MergeOnConflictHook {
public:
  virtual ~MergeOnConflictHook() = default;

  /// Default policy: always reject.
  virtual LogicalResult tryResolve(Operation *conflictOp) { return failure(); }
};

class CoreLegalityChecker {
public:
  CoreLegalityChecker(func::FuncOp func, MergeOnConflictHook &hook)
      : func(func), hook(hook) {}

  /// Run the forward origin dataflow and return the verdict. Independent of the
  /// core assignment -- seeded only from the user `{sub_block}` scopes.
  LegalityResult check();

private:
  Core originOf(Operation &op,
                const llvm::DenseMap<Operation *, Core> &origins) const;

  func::FuncOp func;
  MergeOnConflictHook &hook;
};

} // namespace partition_and_bind
} // namespace hivm
} // namespace mlir

#endif // ...CORELEGALITY_H

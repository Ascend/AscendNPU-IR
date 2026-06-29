//===- DistributedTransformUtils.h - Utilities for Distributed HIVM Ops ---===//
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

#ifndef BISHENGIR_DIALECT_HIVM_IR_CUSTOMOP_DISTRIBUTED_TRANSFORM_UTILS_H
#define BISHENGIR_DIALECT_HIVM_IR_CUSTOMOP_DISTRIBUTED_TRANSFORM_UTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "llvm/ADT/StringSet.h"

#include <string>

namespace mlir {
namespace hivm {

/// Known distributed shmem interpreter custom op names.
extern const llvm::StringSet<> shmemIntp;

/// Symbol names for distributed shmem layout inference.
extern const std::string kShmemPtr;
extern const std::string kShmemConsumeToken;

/// Returns true when `op` is a distributed-marked `hivm.hir.custom`.
inline bool isDistributedTypeCustomOp(Operation *op) {
  return op->hasAttr(hivm::IsDistributedAttr::name) &&
         llvm::isa<hivm::CustomOp>(op);
}

/// Returns true when `op` is a shmem notify custom op.
inline bool isShmemNotifyCustomOp(CustomOp op) {
  return shmemIntp.contains(op.getName());
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_CUSTOMOP_DISTRIBUTED_TRANSFORM_UTILS_H

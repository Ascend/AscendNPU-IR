//===- DistributedTransformUtils.h - Utilities for Distributed HIVM Ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_DISTRIBUTED_TRANSFORM_UTILS_H
#define BISHENGIR_DIALECT_HIVM_DISTRIBUTED_TRANSFORM_UTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace hivm {

// HIVM Distributed CustomOp Types
const StringSet<> shmemIntp = {"dist.aclshmem_int64_p", "dist.aclshmem_int32_p",
                               "dist.aclshmem_int16_p", "dist.aclshmem_int8_p"};

const std::string kShmemPtr = "aclshmem_ptr";
const std::string kShmemConsumeToken = "aclshmem_consume_token";

// Helper Functions
inline bool isDistributedTypeCustomOp(Operation *op) {
  return op->hasAttr(hivm::IsDistributedAttr::name) &&
         llvm::isa<hivm::CustomOp>(op);
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_DISTRIBUTED_TRANSFORM_UTILS_H

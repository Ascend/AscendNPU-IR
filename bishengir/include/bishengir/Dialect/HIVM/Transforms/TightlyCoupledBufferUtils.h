//===- TightlyCoupledBufferUtils.h ------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025~2026. All rights reserved.
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
//
// Shared helpers for MarkTightlyCoupledBuffer / HoistTightlyCoupledAlloc.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_TIGHTLYCOUPLEDBUFFERUTILS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_TIGHTLYCOUPLEDBUFFERUTILS_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace hivm {

/// True when `alloc` targets L1 (cbuf) or UB address space.
bool isL1OrUBAlloc(memref::AllocOp alloc);

/// Return the `annotation.mark` carrying `hivm.tightly_coupled_buffer`, if any.
std::optional<Operation *> getTightlyCoupledMark(Value memref);

/// Read an existing tightly-coupled-buffer id from `alloc`, if present.
std::optional<int64_t> getTightlyCoupledBufferId(memref::AllocOp alloc);

/// Collect used TCB ids among `allocs` that already carry the attribute.
llvm::DenseSet<int64_t>
collectUsedTightlyCoupledIds(ArrayRef<memref::AllocOp> allocs);

/// Allocate the next unused non-negative id given a set of already-used ids.
int64_t allocateNextTightlyCoupledId(llvm::DenseSet<int64_t> &usedIds);

/// Create an `annotation.mark` with `hivm.tightly_coupled_buffer = id` after
/// `alloc`. Returns the created mark op.
annotation::MarkOp createTightlyCoupledBufferMark(OpBuilder &builder,
                                                  memref::AllocOp alloc,
                                                  int64_t id);

/// If `op` just forwards/views another value (without changing the underlying
/// buffer), return that source value; otherwise return a null Value.
Value getTightlyCoupledViewSource(Operation *op);

/// True when `v` is derived from `root` only through forwarding/view ops.
bool tracesToTightlyCoupledValue(Value v, Value root);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_TIGHTLYCOUPLEDBUFFERUTILS_H

//===- AllocToPointerCast.h --Convert memref.AllocOp to hivm.pointercastOp-===//
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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace hivm {
#ifndef LLVM_PROJECT_ALLOCTOPOINTERCAST_H
#define LLVM_PROJECT_ALLOCTOPOINTERCAST_H

/// Walk every `memref.alloc` in `funcOp` and convert it into an
/// `hivm.hir.pointer_cast` bound to its planned address(es). Interrupts and
/// returns failure with a diagnostic if any alloc is not covered by the memory
/// plan (read before first write).
LogicalResult walkAllocToPointerCast(
    func::FuncOp funcOp,
    const DenseMap<Value, SmallVector<uint64_t>> &buffer2Offsets);

/// Walk every `memref_ext.alloc_workspace` in `funcOp` and attach its planned
/// offset(s). Interrupts and returns failure with a diagnostic if any workspace
/// alloc is not covered by the memory plan.
LogicalResult walkUpdateAllocWorkspaceOffset(
    func::FuncOp funcOp,
    const DenseMap<Value, SmallVector<uint64_t>> &buffer2Offsets);

} // namespace hivm
} // namespace mlir

#endif // LLVM_PROJECT_ALLOCTOPOINTERCAST_H

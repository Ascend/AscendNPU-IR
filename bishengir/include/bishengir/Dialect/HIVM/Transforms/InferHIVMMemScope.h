//===- InferHIVMMemScope.h --Infer Memory Scope for HIVM Ops ----*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INFERHIVMMEMSCOPE_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INFERHIVMMEMSCOPE_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace hivm {

class MemScopeInferAndPropagateHelper {
public:
  LogicalResult Run(Value operand, const AddressSpaceAttr &targetMemScope);

private:
  /// Propagate the memory scope change to users of the value.
  LogicalResult propagateMemScopeToUsers(Value val);

  /// Set memory scope for the root alloc op.
  void setMemRefAllocScope(memref::AllocOp op,
                           const AddressSpaceAttr &newScope);
  /// Set memory scope for the block argument.
  void setBlockArgumentScope(BlockArgument operand,
                             const AddressSpaceAttr &targetMemScope);
};

/// Infer, propagate, and set memory scope information to MmadL1Op.
/// \note MmadL1Op should be bufferized beforehand.
LogicalResult inferAndPropagateMemScopeForMmadL1(MmadL1Op op);

/// Infer, propagate, and set memory scope information to FuncOp.
/// \note FuncOp should be bufferized beforehand.
LogicalResult inferAndPropagateMemScopeForFunc(func::FuncOp op);

/// Infer, propagate, and set memory scope information to PointerCastOp.
LogicalResult inferAndPropagateMemScopeForPointerCast(hivm::PointerCastOp op);

/// Infer, propagate, and set memory scope information to AllocOp.
/// \note Set alloc on aic op memory scope to L1. And set aiv alloc memory scope to ub
LogicalResult inferAndPropagateMemScopeForAlloc(memref::AllocOp op, hivm::AddressSpace space);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INFERHIVMMEMSCOPE_H

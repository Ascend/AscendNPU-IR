//===- InferHIVMMemScope.h --Infer Memory Scope for HIVM Ops ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INFERHIVMMEMSCOPE_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INFERHIVMMEMSCOPE_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace hivm {

class MemScopeInferAndPropagateHelper {
public:
  LogicalResult Run(Value operand, const AddressSpaceAttr &targetMemScope);

  /// Propagate the memory scope change to users of the value.
  LogicalResult propagateMemScopeToUsers(Value val);

private:
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
/// \note Set alloc memory scope to ub/l1.
LogicalResult inferAndPropagateMemScopeForAlloc(memref::AllocOp op, hivm::AddressSpace space);

/// Infer, propagate, and set memory scope information to GPUFuncOp.
LogicalResult inferAndPropagateMemScopeForGpuFunc(gpu::GPUFuncOp op);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INFERHIVMMEMSCOPE_H

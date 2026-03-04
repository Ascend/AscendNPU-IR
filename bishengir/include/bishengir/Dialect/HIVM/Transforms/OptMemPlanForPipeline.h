//===- OptMemPlanForPipeline.h --Pipeline optimization for plan memory------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace hivm {
#ifndef BISHENG_DIALECT_HIVM_OPT_MEM_PLAN_FOR_PIPELINE_H
#define BISHENG_DIALECT_HIVM_OPT_MEM_PLAN_FOR_PIPELINE_H

class OptMemPlanForDma {
public:
  OptMemPlanForDma(){};

  /// Main interface for OptMemPlanForDma.
  void build(func::FuncOp func);

  /// Check if buf1 and buf2 is dma and scalar pipe conflict.
  bool BufferPipeConflict(const Value buf1, const Value buf2) const;

  /// Is the current buffer used by DMA instructions.
  bool IsDmaBuffer(const Value buf) const;

  bool IsScalarBuffer(const Value buf) const;

private:
  /// Verify that HIVMOpPipe has a pipe type.
  LogicalResult VerifyExistHivmPipe(hivm::OpPipeInterface hivmPipeOp) const;

  /// Update the buffers for MTE2 and MTE3.
  void UpdateDmaBuffers(SmallVector<Value> dpsOperand);

  template <typename OP>
  typename std::enable_if<std::is_same_v<OP, memref::LoadOp> ||
                              std::is_same_v<OP, memref::StoreOp>,
                          void>::type
  UpdateScalarBuffers(OP op);

  void UpdateScalarBuffersForLowerToLoops(Operation *operands);

  /// Buffer in MTE2 and MTE3.
  DenseSet<Value> DmaBuffers;

  /// Buffer in Scalar.
  DenseSet<Value> ScalarBuffers;
};
} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_OPT_MEM_PLAN_FOR_PIPELINE_H

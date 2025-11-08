//===- OptMemPlanForPipeline.h --Pipeline optimization for plan memory------==//
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

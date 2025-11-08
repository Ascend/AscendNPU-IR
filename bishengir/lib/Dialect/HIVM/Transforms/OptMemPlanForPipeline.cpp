//==- OptMemPlanForPipeline.cpp --Pipeline optimization for plan memory------=//
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

#include "bishengir/Dialect/HIVM/Transforms/OptMemPlanForPipeline.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::detail;
using namespace mlir::hivm;

void OptMemPlanForDma::build(func::FuncOp func) {
  auto result = func->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto implByScalarOp =
            dyn_cast<mlir::hivm::ImplByScalarOpInterface>(op)) {
      if (implByScalarOp.shouldLowerToScalarLoops()) {
        UpdateScalarBuffersForLowerToLoops(op);
        return WalkResult::advance();
      }
    }
    if (auto hivmStructuredOp = dyn_cast<HIVMStructuredOp>(op)) {
      auto hivmPipeOp = dyn_cast<hivm::OpPipeInterface>(op);
      assert(hivmPipeOp != nullptr);
      if (!hivmPipeOp.isSinglePipeOp()) {
        return WalkResult::skip();
      }
      if (failed(VerifyExistHivmPipe(hivmPipeOp))) {
        return WalkResult::interrupt();
      }
      if (hivmPipeOp.getPipe() == hivm::PIPE::PIPE_MTE2) {
        UpdateDmaBuffers(hivmStructuredOp.getDpsInits());
      } else if (hivmPipeOp.getPipe() == hivm::PIPE::PIPE_MTE3) {
        UpdateDmaBuffers(hivmStructuredOp.getDpsInputs());
      }
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      UpdateScalarBuffers(loadOp);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      UpdateScalarBuffers(storeOp);
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("OptMemPlanForLoop Traverse IR Failed! ");
  }
}

LogicalResult
OptMemPlanForDma::VerifyExistHivmPipe(hivm::OpPipeInterface hivmPipeOp) const {
  hivm::PIPE curPipe = hivmPipeOp.getPipe();
  if (curPipe == hivm::PIPE::PIPE_UNASSIGNED) {
    hivmPipeOp.getOperation()->emitError(
        "OptMemPlanForLoop failed to recognize hivmPipeOp! ");
    return failure();
  }
  return success();
}

void OptMemPlanForDma::UpdateDmaBuffers(SmallVector<Value> dpsOperand) {
  for (Value operand : dpsOperand) {
    auto memorySpaceAttr = GetBufferSpaceAttr(operand);
    if (!isLocalBuffer(memorySpaceAttr)) {
      continue;
    }
    DmaBuffers.insert(utils::tracebackMemRef(operand));
  }
}

bool OptMemPlanForDma::IsDmaBuffer(const Value buf) const {
  if (DmaBuffers.empty()) {
    return false;
  }
  for (auto buffer : DmaBuffers) {
    if (buffer == buf) {
      return true;
    }
  }
  return false;
}

bool OptMemPlanForDma::BufferPipeConflict(const Value buf1,
                                          const Value buf2) const {
  if (IsScalarBuffer(buf1) && IsScalarBuffer(buf2)) {
    return false;
  }

  if (IsScalarBuffer(buf1) || IsScalarBuffer(buf2)) {
    return true;
  }

  if (IsDmaBuffer(buf1) || IsDmaBuffer(buf2)) {
    // Process the operation of ForOp as follows:
    // scf.for %arg4 = %c0 to %c1024 step %c128 ->
    //   alloca %allocA
    //   gm2ub(allocA, gm)
    //   ...
    //   alloca %allocB
    //   ub2gm(gm, allocB)
    // There is a conflict in the reuse of allocA and allocB here.
    // MTE3 and MTE3, MTE2 and MTE2 also have similar conflicts.
    return true;
  }
  return false;
}

template <typename OP>
typename std::enable_if<std::is_same_v<OP, memref::LoadOp> ||
                            std::is_same_v<OP, memref::StoreOp>,
                        void>::type
OptMemPlanForDma::UpdateScalarBuffers(OP op) {
  auto memorySpaceAttr = GetBufferSpaceAttr(op.getMemRef());
  if (!isLocalBuffer(memorySpaceAttr)) {
    return;
  }
  ScalarBuffers.insert(utils::tracebackMemRef(op.getMemRef()));
}

bool OptMemPlanForDma::IsScalarBuffer(const Value buf) const {
  if (ScalarBuffers.empty()) {
    return false;
  }
  for (auto buffer : ScalarBuffers) {
    if (buffer == buf) {
      return true;
    }
  }
  return false;
}

void OptMemPlanForDma::UpdateScalarBuffersForLowerToLoops(Operation *op) {
  for (Value operand : op->getOperands()) {
    auto memorySpaceAttr = GetBufferSpaceAttr(operand);
    if (!isLocalBuffer(memorySpaceAttr)) {
      continue;
    }
    ScalarBuffers.insert(utils::tracebackMemRef(operand));
  }
}
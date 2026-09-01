//===- AutoScheduleBuilder.cpp -- HFusion auto-schedule builder -*- C++ -*-===//
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
//
// This file implements HFusion-specific schedule primitives composed on top
// of the dialect-neutral ScheduleBuilder.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBuilder.h"
#include "bishengir/Dialect/Transform/IR/TransformOps.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "AutoScheduleAttrDefs.h"

#define DEBUG_TYPE "hfusion-auto-schedule"
#define DBGS()                                                                 \
  (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Auto Schedule Builder] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

//===----------------------------------------------------------------------===//
// Kernel IO.
//===----------------------------------------------------------------------===//

ValueHandles
AutoScheduleBuilder::getKernelOutputs(OpBuilder &opBuilder,
                                      const KernelInfo &kernelInfo,
                                      const GetKernelIOOptions &options) {
  if (options.isInverted) {
    assert(options.findReshapePosition.empty() &&
           "isInverted cannot be used with findReshapePosition");
  }
  auto funcValue = getFuncValue(opBuilder);
  ValueHandles handles;
  for (size_t operandIdx : kernelInfo.outputOrdering) {
    // TODO: The result is matched one-by-one because split handle op cannot
    // split transform any value typed inputs.
    auto resultHandle = opBuilder.create<transform::GetFuncResultOp>(
        funcValue.getLoc(),
        /*outputs=*/opBuilder.getType<transform::AnyValueType>(),
        /*target=*/funcValue,
        /*raw_position_list=*/
        opBuilder.getDenseI64ArrayAttr({static_cast<int64_t>(operandIdx)}),
        /*is_inverted=*/options.isInverted,
        /*is_all=*/false,
        /*find_reshape_producer=*/
        options.findReshapePosition.contains(operandIdx));
    handles.push_back(
        record<RegularValueHandle>(resultHandle.getResult(), opBuilder));
  }
  return handles;
}

ValueHandles
AutoScheduleBuilder::getKernelInputs(OpBuilder &opBuilder,
                                     const GetKernelIOOptions &options) {
  if (options.isInverted) {
    assert(options.findReshapePosition.empty() &&
           "isInverted cannot be used with findReshapePosition");
  }
  auto funcValue = getFuncValue(opBuilder);
  ValueHandles handles;
  // TODO: The result is matched one-by-one because merge handle op cannot
  // merge transform any value typed inputs.
  for (auto operandIdx : options.positionList) {
    auto funcArgHandle = opBuilder.create<transform::GetFuncArgumentOp>(
        funcValue.getLoc(),
        /*outputs=*/opBuilder.getType<transform::AnyValueType>(),
        /*target=*/funcValue,
        /*raw_position_list=*/
        opBuilder.getDenseI64ArrayAttr({operandIdx}),
        /*is_inverted=*/options.isInverted,
        /*is_all=*/false,
        /*find_reshape_consumer*/
        options.findReshapePosition.contains(operandIdx));
    handles.push_back(
        record<RegularValueHandle>(funcArgHandle.getResult(), opBuilder));
  }
  return handles;
}

AutoScheduleBuilder::CacheIOResult
AutoScheduleBuilder::cacheRead(OpBuilder &opBuilder,
                               const KernelInfo &kernelInfo) {
  auto kernelInputsHandles = getKernelInputs(
      opBuilder,
      GetKernelIOOptions{/*positionList=*/
                         llvm::to_vector(kernelInfo.cacheReadFuncArgIndices),
                         /*isInverted=*/false,
                         /*findReshapePosition=*/
                         kernelInfo.funcArgWithReshapeIndices});

  for (auto [idx, kernelInputsHandle] : llvm::enumerate(kernelInputsHandles)) {
    Value inputs = getValue(kernelInputsHandle, opBuilder);
    auto cachedOp =
        opBuilder
            .create<transform::CacheReadOp>(
                inputs.getLoc(),
                /*cached=*/opBuilder.getType<transform::AnyOpType>(),
                /*targets=*/inputs)
            .getCached();
    ScheduleBuilder::annotateByAttr(
        cachedOp, hfusion::LoadOp::getOperationName(), opBuilder);
    ScheduleBuilder::annotateByAttr(cachedOp, getCacheReadTag(idx), opBuilder);
    kernelInputsHandle->invalidate();
  }
  auto matchTarget = getTransformSeqHandle();
  auto cachedOps = matchByIdentifier(
      matchTarget, OperationIdentifier(hfusion::LoadOp::getOperationName()),
      opBuilder);
  // TODO: needsReverse = true is a temporary solution to the problem that
  // cache reads are done in the order of they appear in the function arguments,
  // but that the order they appear in the IR is in reverse order. We shouldn't
  // depend on the ordering.
  return CacheIOResult{
      /*cachedOps=*/
      record<NamedValueHandle>(
          cachedOps, opBuilder,
          NamedValueHandleArgs{hfusion::LoadOp::getOperationName(),
                               IdentifierType::kOperation,
                               /*needsAnnotate=*/false,
                               /*needsReverse=*/true})};
}

AutoScheduleBuilder::CacheIOResult
AutoScheduleBuilder::cacheWrite(OpBuilder &opBuilder,
                                const KernelInfo &kernelInfo) {
  auto kernelOutputHandles = getKernelOutputs(
      opBuilder, kernelInfo,
      GetKernelIOOptions{/*positionList=*/
                         kernelInfo.outputOrdering,
                         /*isInverted=*/false,
                         /*findReshapePosition=*/
                         kernelInfo.returnValueWithReshapeIndices});

  SmallVector<Value> cacheWriteOriginalOps;
  for (auto [originalResultIdx, outputHandle] :
       llvm::zip_equal(kernelInfo.outputOrdering, kernelOutputHandles)) {
    auto output = getValue(outputHandle, opBuilder);
    auto cachedWriteOp =
        opBuilder
            .create<transform::CacheWriteOp>(
                output.getLoc(),
                /*cached=*/opBuilder.getType<transform::AnyOpType>(),
                /*targets=*/output,
                /*output_only=*/true,
                /*cache_write_to_output_init=*/
                kernelInfo.returnValueIdx2TiedFuncArg.contains(
                    originalResultIdx))
            .getCached();
    ScheduleBuilder::annotateByAttr(
        cachedWriteOp, hfusion::StoreOp::getOperationName(), opBuilder);
    outputHandle->invalidate();
  }
  auto matchTarget = getTransformSeqHandle();
  auto cachedOps = matchByIdentifier(
      matchTarget, OperationIdentifier(hfusion::StoreOp::getOperationName()),
      opBuilder);
  return CacheIOResult{/*cachedOps=*/record<NamedValueHandle>(
      cachedOps, opBuilder,
      NamedValueHandleArgs{hfusion::StoreOp::getOperationName(),
                           IdentifierType::kOperation,
                           /*needsAnnotate=*/false})};
}

//===----------------------------------------------------------------------===//
// Tiling helpers.
//===----------------------------------------------------------------------===//

AutoScheduleBuilder::ForallTilingResult
AutoScheduleBuilder::tileUsingForAll(ValueHandles &targets, int64_t blockDim,
                                     OpBuilder &opBuilder) {
  // The block axis is tied to `hivm.block<x>`.
  auto mapping = opBuilder.getArrayAttr(
      {hivm::HIVMBlockMappingAttr::get(opBuilder.getContext())});
  return ScheduleBuilder::tileUsingForAll(targets, blockDim, mapping,
                                          opBuilder);
}

ValueHandles
AutoScheduleBuilder::getTilingStructHandles(SmallVector<TilingData *> s,
                                            OpBuilder &opBuilder) {
  return llvm::map_to_vector(s, [this, &opBuilder](TilingData *td) {
    return this->getTilingDataHandle(td, opBuilder);
  });
}

ValueHandle *AutoScheduleBuilder::getTilingDataHandle(TilingData *d,
                                                      OpBuilder &opBuilder) {
  if (d->getHandle())
    return d->getHandle();

  auto funcValue = getFuncValue(opBuilder);
  size_t posWithInFunc = d->getPos();
  auto funcArgHandles = opBuilder.create<transform::GetFuncArgumentOp>(
      funcValue.getLoc(),
      /*outputs=*/opBuilder.getType<transform::AnyValueType>(),
      /*target=*/funcValue,
      /*raw_position_list=*/
      SmallVector<int64_t>{static_cast<int64_t>(posWithInFunc)},
      /*is_inverted=*/false);
  auto *handle = getHandleRecord()->record<FuncArgHandle>(FuncArgHandle(
      funcArgHandles.getOutputs(), posWithInFunc, HandleStatus::kValid));
  d->setHandle(handle);
  return handle;
}

void AutoScheduleBuilder::setBufferSize(ValueHandles &targets,
                                        int64_t bufferSize,
                                        OpBuilder &opBuilder,
                                        const SetBufferSizeOptions &options) {
  std::vector<int64_t> bufferSizes(targets.size(), bufferSize);
  auto targetValues = getValues(targets, opBuilder);
  opBuilder.create<transform::SetBufferSizeOp>(
      targetValues.front().getLoc(),
      /*target=*/targetValues,
      /*static_buffer_sizes=*/bufferSizes,
      /*unit_mode=*/options.mode,
      /*reference_type=*/TypeAttr::get(options.referenceType));
}

std::string AutoScheduleBuilder::getCacheReadTag(size_t funcArgIdx) {
  return llvm::formatv(kFuncArgIdxFormat, funcArgIdx).str();
}

ValueHandle *
AutoScheduleBuilder::getIntermediateProducers(OpBuilder &opBuilder) {
  MatchOptions matchOptions;
  matchOptions.needsReverse = true;
  return getOpsWithAttr(kIntermediateProducerTagName, opBuilder, Attribute(),
                        matchOptions);
}

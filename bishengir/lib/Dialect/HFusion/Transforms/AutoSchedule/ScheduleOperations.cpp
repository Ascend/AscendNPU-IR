//===- ScheduleOperations.cpp -- Auto-schedule operation Impl.---*- C++ -*-===//
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
// This file implements auto scheduler's HFusion-specific schedule operations.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/Transforms/TransformOps.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
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
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Base Scheduler] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {
/// Update handles to the containing loops after fusion.
void updateHandleToContainingLoops(
    ValueHandles &containingLoops,
    const SmallVector<Value> &containingLoopValues,
    bool applyCanonicalizeAfterEachFusion) {
  for (auto it : llvm::enumerate(containingLoops)) {
    if (applyCanonicalizeAfterEachFusion) {
      // Currently, for ForeachOp, the payload ops of the corresponding
      // YieldOp operand are merged and mapped to the same resulting handle.
      // Therefore, the result value of ForeachOp corresponding to the new
      // containing op will map to the same containing op many times. This is
      // a bit confusing for downstream users. So we invalidate the handles
      // for now.
      it.value()->invalidate();
    } else {
      it.value()->setHandle(containingLoopValues[it.index()]);
    }
  }
}

transform::ExtendedFuseIntoContainingOp
createFuseIntoContainingOp(Value producerOp,
                           const SmallVector<Value> &containingLoopValues,
                           bool duplicateProducers, size_t numContainingLoop,
                           OpBuilder &opBuilder, Location loc) {
  return opBuilder.create<transform::ExtendedFuseIntoContainingOp>(
      loc,
      /*fused_op=*/
      std::vector<Type>(numContainingLoop,
                        opBuilder.getType<transform::AnyOpType>()),
      /*new_containing_op=*/
      std::vector<Type>(numContainingLoop,
                        opBuilder.getType<transform::AnyOpType>()),
      /*producer_op=*/producerOp,
      /*containing_op=*/containingLoopValues,
      /*duplicate_producer=*/
      BoolAttr::get(opBuilder.getContext(), duplicateProducers));
}
} // namespace

//===----------------------------------------------------------------------===//
// Value handle materialization.
//===----------------------------------------------------------------------===//

SchedulerBase::ForReductionTilingResult SchedulerBase::tileReductionUsingFor(
    ValueHandles &targets, ValueHandleFoldResults &tileSizes,
    OpBuilder &opBuilder, int64_t multiReduceNum) {
  auto [staticTileSizes, dynamicTileSizes] =
      unpackFoldResults(tileSizes, opBuilder);

  ForReductionTilingResult result;

  auto mapFnForInit = [this, &opBuilder](Value init) -> ValueHandle * {
    return record<NamedValueHandle>(
        init, opBuilder,
        NamedValueHandleArgs{kTileReductionInitOpTagName,
                             IdentifierType::kAttribute});
  };

  for (auto *targetHandle : targets) {
    auto targetValue = getValue(targetHandle, opBuilder);
    auto tileReductionOp = opBuilder.create<transform::TileReductionUsingForOp>(
        targetValue.getLoc(),
        /*fill_op=*/
        SmallVector<Type>(multiReduceNum,
                          opBuilder.getType<transform::AnyOpType>()),
        /*split_linalg_op=*/opBuilder.getType<transform::AnyOpType>(),
        /*combining_linalg_op=*/opBuilder.getType<transform::AnyOpType>(),
        /*for_op=*/opBuilder.getType<transform::AnyOpType>(),
        /*target=*/targetValue,
        /*tile_sizes=*/dynamicTileSizes,
        /*static_tile_sizes=*/opBuilder.getDenseI64ArrayAttr(staticTileSizes));

    LDBG("tileReductionUsingFor result");
    LDBG(tileReductionOp.getSplitLinalgOp());
    LDBG(tileReductionOp.getCombiningLinalgOp());
#ifndef NDEBUG
    for (auto fillOp : tileReductionOp.getFillOp())
      LDBG(fillOp);
#endif
    LDBG(tileReductionOp.getForOp());
    result.partialReductionOp.emplace_back(record<NamedValueHandle>(
        tileReductionOp.getSplitLinalgOp(), opBuilder,
        NamedValueHandleArgs{kTileReductionPartialReductionOpTagName,
                             IdentifierType::kAttribute}));

    result.finalReductionOp.emplace_back(record<NamedValueHandle>(
        tileReductionOp.getCombiningLinalgOp(), opBuilder,
        NamedValueHandleArgs{kTileReductionFinalReductionOpTagName,
                             IdentifierType::kAttribute}));

    result.reductionInitOp.emplace_back(
        llvm::map_to_vector(tileReductionOp.getFillOp(), mapFnForInit));

    result.loops.emplace_back(record<NamedValueHandle>(
        tileReductionOp.getForOp(), opBuilder,
        NamedValueHandleArgs{kTileReductionLoopTagName,
                             IdentifierType::kAttribute}));

    // The original reduction op is decomposed into multiple ops, so the
    // handle should be invalidated.
    targetHandle->invalidate();
  }
  return result;
}

void SchedulerBase::fuseIntoContaining(ValueHandles &targetOps,
                                       ValueHandles &containingLoops,
                                       OpBuilder &opBuilder,
                                       bool duplicateProducers,
                                       bool applyCanonicalizeAfterEachFusion) {
  SmallVector<Value> containingLoopValues =
      getValues(containingLoops, opBuilder);
  size_t numContainingLoop = containingLoopValues.size();

  SmallVector<Value> fusedLoops;
  for (auto *targetHandle : targetOps) {
    auto targetValue = getValue(targetHandle, opBuilder);
    Location loc = targetValue.getLoc();
    if (applyCanonicalizeAfterEachFusion) {
      // Construct `transform::ForeachOp` to perform canonicalization before
      // fusing each target op into the containing op.
      // This is necessarily for complicated cases where the target ops
      // are used multiple times in the containing op.
      auto forEachRegionBuilderFn = [&](ImplicitLocOpBuilder &opBuilder,
                                        Block &block) -> void {
        auto blockArg = block.getArgument(0);

        // disabled patterns:
        //   a) kSimplifyTrivialLoops: in case trivial loops is simplified and
        //      lead to invalid loop handles
        applyPatterns(
            getFuncHandle(opBuilder),
            /*patterns=*/
            SmallVector<TransformPatternKind>{
                TransformPatternKind::CSE,
                TransformPatternKind::CANONICALIZATION,
                TransformPatternKind::MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE,
                TransformPatternKind::RESOLVE_RANKED_SHAPED_TYPE_RESULT_DIMS},
            opBuilder,
            /*disablePatterns=*/
            SmallVector<CanonicalizationPatternKind>{
                CanonicalizationPatternKind::kSimplifyTrivialLoops});
        auto op = createFuseIntoContainingOp(blockArg, containingLoopValues,
                                             duplicateProducers,
                                             numContainingLoop, opBuilder, loc);
        opBuilder.create<transform::YieldOp>(op.getLoc(), op->getResults());
      };

      std::vector<Type> forEachResultTypes(
          numContainingLoop * 2, opBuilder.getType<transform::AnyOpType>());
      auto forEachResults = createForEachOp(targetValue, forEachResultTypes,
                                            forEachRegionBuilderFn, opBuilder);
      fusedLoops = {forEachResults.begin(),
                    forEachResults.begin() + numContainingLoop};
      // TODO: Update containingLoopValue to ForeachOp's result
    } else {
      auto op = createFuseIntoContainingOp(targetValue, containingLoopValues,
                                           duplicateProducers,
                                           numContainingLoop, opBuilder, loc);
      fusedLoops = op.getFusedOp();
      containingLoopValues = op.getNewContainingOp();
    }
    if (numContainingLoop == 1) {
      targetHandle->setHandle(fusedLoops.front());
    } else {
      targetHandle->invalidate();
    }
  }
  updateHandleToContainingLoops(containingLoops, containingLoopValues,
                                applyCanonicalizeAfterEachFusion);
}

//===----------------------------------------------------------------------===//
// Kernel IO.
//===----------------------------------------------------------------------===//

ValueHandles
SchedulerBase::getKernelOutputs(OpBuilder &opBuilder,
                                const GetKernelIOOptions &options) {
  if (options.isInverted) {
    assert(options.findReshapePosition.empty() &&
           "isInverted cannot be used with findReshapePosition");
  }
  auto funcValue = getFuncValue(opBuilder);
  ValueHandles handles;
  for (size_t operandIdx : getKernelInfo()->outputOrdering) {
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

ValueHandles SchedulerBase::getKernelInputs(OpBuilder &opBuilder,
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

SchedulerBase::CacheIOResult SchedulerBase::cacheRead(OpBuilder &opBuilder) {
  auto kernelInputsHandles = getKernelInputs(
      opBuilder, GetKernelIOOptions{
                     /*positionList=*/
                     llvm::to_vector(getKernelInfo()->cacheReadFuncArgIndices),
                     /*isInverted=*/false,
                     /*findReshapePosition=*/
                     getKernelInfo()->funcArgWithReshapeIndices});

  for (auto [idx, kernelInputsHandle] : llvm::enumerate(kernelInputsHandles)) {
    Value inputs = getValue(kernelInputsHandle, opBuilder);
    auto cachedOp =
        opBuilder
            .create<transform::CacheReadOp>(
                inputs.getLoc(),
                /*cached=*/opBuilder.getType<transform::AnyOpType>(),
                /*targets=*/inputs)
            .getCached();
    annotateByAttr(cachedOp, hfusion::LoadOp::getOperationName(), opBuilder);
    annotateByAttr(cachedOp, getCacheReadTag(idx), opBuilder);
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

SchedulerBase::CacheIOResult SchedulerBase::cacheWrite(OpBuilder &opBuilder) {
  auto kernelOutputHandles = getKernelOutputs(
      opBuilder,
      GetKernelIOOptions{/*positionList=*/
                         getKernelInfo()->outputOrdering,
                         /*isInverted=*/false,
                         /*findReshapePosition=*/
                         getKernelInfo()->returnValueWithReshapeIndices});

  SmallVector<Value> cacheWriteOriginalOps;
  for (auto [originalResultIdx, outputHandle] :
       llvm::zip_equal(getKernelInfo()->outputOrdering, kernelOutputHandles)) {
    auto output = getValue(outputHandle, opBuilder);
    auto cachedWriteOp =
        opBuilder
            .create<transform::CacheWriteOp>(
                output.getLoc(),
                /*cached=*/opBuilder.getType<transform::AnyOpType>(),
                /*targets=*/output,
                /*output_only=*/true,
                /*cache_write_to_output_init=*/
                getKernelInfo()->returnValueIdx2TiedFuncArg.contains(
                    originalResultIdx))
            .getCached();
    annotateByAttr(cachedWriteOp, hfusion::StoreOp::getOperationName(),
                   opBuilder);
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

SchedulerBase::ForallTilingResult
SchedulerBase::tileUsingForAll(ValueHandles &targets, int64_t blockDim,
                               OpBuilder &opBuilder) {
  // The block axis is tied to `hivm.block<x>`.
  auto mapping =
      opBuilder.getArrayAttr({hivm::HIVMBlockMappingAttr::get(getContext())});
  return schedule::ScheduleBuilder::tileUsingForAll(targets, blockDim, mapping,
                                                    opBuilder);
}

ValueHandles SchedulerBase::getTilingStructHandles(SmallVector<TilingData *> s,
                                                   OpBuilder &opBuilder) {
  return llvm::map_to_vector(s, [this, &opBuilder](TilingData *td) {
    return this->getTilingDataHandle(td, opBuilder);
  });
}

ValueHandle *SchedulerBase::getTilingDataHandle(TilingData *d,
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

void SchedulerBase::setBufferSize(ValueHandles &targets, int64_t bufferSize,
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

std::string SchedulerBase::getCacheReadTag(size_t funcArgIdx) {
  return llvm::formatv(kFuncArgIdxFormat, funcArgIdx).str();
}

ValueHandle *SchedulerBase::getIntermediateProducers(OpBuilder &opBuilder) {
  MatchOptions matchOptions;
  matchOptions.needsReverse = true;
  return getOpsWithAttr(kIntermediateProducerTagName, opBuilder, Attribute(),
                        matchOptions);
}

//===- AutoScheduleBuilder.h -- HFusion auto-schedule builder ---*- C++ -*-===//
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
// HFusion-specific schedule primitives extending the dialect-neutral
// ScheduleBuilder: kernel IO (cache read/write), tiling data handles,
// reduction tiling, loop fusion and buffer size setting.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBUILDER_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBUILDER_H

#include "bishengir/Dialect/Analysis/Schedule/Builder.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/TilingUtils.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace hfusion {
using namespace mlir::schedule;

namespace detail {
/// Struct to return the result of cache read/write.
struct CacheIOResult {
  ValueHandle *cachedOps;
};

/// Struct for specifying options when getting kernel inputs/outputs.
struct GetKernelIOOptions {
  /// The positions of the kernel input/output.
  SmallVector<int64_t> positionList{};
  /// Whether the raw position is the kernel input/output to exclude.
  bool isInverted{false};
  /// For getting kernel inputs, this is the positions of the input arguments
  /// that are reshaped. If set, the return handle points to the reshaped kernel
  /// input.
  /// For getting kernel outputs, this is the positions of the kernel
  /// outputs that are reshape op's results. If set, the return handle points to
  /// the value before reshaping.
  /// \Note Cannot be used when \c isInverted is set to true.
  SetVector<int64_t> findReshapePosition{};
};

/// Struct for specifying options for set buffer size.
struct SetBufferSizeOptions {
  transform::SetBufferSizeMode mode{transform::SetBufferSizeMode::kPerByte};
  Type referenceType{Type()};
};
} // namespace detail

/// Builder type names re-exported for unqualified use by schedule
/// clients. Import with `using namespace mlir::hfusion::schedule;`.
namespace schedule {
using ForallTilingResult = ScheduleBuilder::ForallTilingResult;
using ForTilingResult = ScheduleBuilder::ForTilingResult;
using ForReductionTilingResult = ScheduleBuilder::ForReductionTilingResult;
using TransformPatternKind = ScheduleBuilder::TransformPatternKind;
using CanonicalizationPatternKind =
    ScheduleBuilder::CanonicalizationPatternKind;
using LoopTileResult = ScheduleBuilder::LoopTileResult;
using LoopTileMode = ScheduleBuilder::LoopTileMode;
using LoopTileOptions = ScheduleBuilder::LoopTileOptions;
using MapForToForallOptions = ScheduleBuilder::MapForToForallOptions;
using MatchOptions = ScheduleBuilder::MatchOptions;
using NamedValueHandleArgs = ScheduleBuilder::NamedValueHandleArgs;
using Identifier = ScheduleBuilder::Identifier;
using OperationIdentifier = ScheduleBuilder::OperationIdentifier;
using AttributeIdentifier = ScheduleBuilder::AttributeIdentifier;
using RegionBuilderFn = ScheduleBuilder::RegionBuilderFn;
using CacheIOResult = detail::CacheIOResult;
using GetKernelIOOptions = detail::GetKernelIOOptions;
using SetBufferSizeOptions = detail::SetBufferSizeOptions;
using SetBufferSizeMode = transform::SetBufferSizeMode;
} // namespace schedule

/// HFusion-specific schedule primitives extending the dialect-neutral
/// \c ScheduleBuilder.
class AutoScheduleBuilder : public ScheduleBuilder {
public:
  using CacheIOResult = detail::CacheIOResult;
  using GetKernelIOOptions = detail::GetKernelIOOptions;
  using SetBufferSizeOptions = detail::SetBufferSizeOptions;
  using SetBufferSizeMode = transform::SetBufferSizeMode;

  explicit AutoScheduleBuilder(MLIRContext *ctx) : ScheduleBuilder(ctx) {}
  ~AutoScheduleBuilder() override = default;

  //===--------------------------------------------------------------------===//
  // HFusion-specific schedule primitives.
  //===--------------------------------------------------------------------===//

  /// Get handles to the outputs of the kernel.
  ///
  /// \param kernelInfo Information regarding the to-be-scheduled kernel.
  /// \param options Options for getting kernel outputs.
  /// \return RegularValueHandles to the producing op of kernel function's
  ///         return values.
  ValueHandles
  getKernelOutputs(const KernelInfo &kernelInfo,
                   const GetKernelIOOptions &options = GetKernelIOOptions());

  /// Get handles to the inputs of the kernel.
  ///
  /// \param options Options for getting kernel inputs.
  /// \return RegularValueHandles to the kernel function's input block argument.
  ValueHandles
  getKernelInputs(const GetKernelIOOptions &options = GetKernelIOOptions());

  /// Get handle to the tiling data.
  ///
  /// \param d Tiling data pointer.
  /// \return FuncArgHandle to the kernel function's block argument that
  ///         corresponds to the tiling data.
  ValueHandle *getTilingDataHandle(TilingData *d);

  /// Get handles to each tiling data in tiling struct \c s.
  ///
  /// \param s A series of tiling data pointer.
  /// \return FuncArgHandles to the kernel function's block arguments that
  ///         correspond to the tiling data in tiling struct.
  ValueHandles getTilingStructHandles(SmallVector<TilingData *> s);

  /// Perform cache read on kernel inputs.
  ///
  /// After cache read, an unique tag name will be added to the cached op.
  /// For example:
  /// ```
  /// func.func @foo(%arg0):
  ///   linalg.copy ins(%arg0) outs(...) {__arg0__}
  /// ```
  ///
  /// \param kernelInfo Information regarding the to-be-scheduled kernel.
  /// \return NamedValueHandle to cached ops. Note that the handle points to
  ///         ALL cached ops. If you wish to obtain a more fine-grained control
  ///         over each ops, you can match by the attributed name returned by
  ///         `getCacheReadTag`.
  CacheIOResult cacheRead(const KernelInfo &kernelInfo);

  /// Get a unique identifier to the cached op by the function argument index.
  std::string getCacheReadTag(size_t funcArgIdx);

  /// Perform cache write on kernel outputs.
  ///
  /// \param kernelInfo Information regarding the to-be-scheduled kernel.
  /// \return NamedValueHandle to the cached ops.
  CacheIOResult cacheWrite(const KernelInfo &kernelInfo);

  /// Tile the target linalg ops using \c scf.forall ops by a
  /// factor of \c blockDim. The block axis is tied to \c hivm.block<x>
  ///
  /// Before tiling:
  ///   linalg.op
  ///
  /// After tiling:
  ///   scf.forall %arg in (blockDim):
  ///     tiled linalg.op
  ///   mapping [hivm.block<x>]
  ///
  /// \param targets Value handles to linalg ops.
  /// \param blockDim Number of blocks.
  /// \return NamedValueHandles to `scf.forall` ops.
  /// \note The input `targets` handles are updated to the tiled linalg ops
  ///       and can be reused without invalidation.
  ForallTilingResult tileUsingForAll(ValueHandles &targets, int64_t blockDim);

  /// Set the size of the `targets` to `bufferSize`.
  ///
  /// If the payload operation is `memref.alloc` or `memeref.alloca`, the
  /// transformation takes place immediately.
  /// Otherwise, the target op is only annotated with the `bufferSize`, and
  /// the actual transformation will happen later on.
  ///
  /// \param targets Value handles to target ops.
  /// \param bufferSize Static buffer size.
  /// \param options
  /// \note The input `targets` handles are invalidated.
  void
  setBufferSize(ValueHandles &targets, int64_t bufferSize,
                const SetBufferSizeOptions &options = SetBufferSizeOptions());

  /// Get handle to all intermediate producers.
  ValueHandle *getIntermediateProducers();
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBUILDER_H

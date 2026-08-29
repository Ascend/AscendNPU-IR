//===- AutoScheduleBase.h -- Auto scheduler basic definitions ---*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBASE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBASE_H

#include "bishengir/Dialect/Analysis/Schedule/Builder.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/FusibleProducerAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/TilingUtils.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {
class Location;
class OpBuilder;

namespace transform {
class AnyOpType;
class AnyValueType;
class OperationType;
class NamedSequenceOp;
class TransformHandleTypeInterface;
} // namespace transform

namespace func {
class FuncOp;
} // namespace func

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

//===----------------------------------------------------------------------===//
// SchedulerBase
//===----------------------------------------------------------------------===//

/// Base class for auto scheduler.
/// Work flow:
///                          +---------------+
///                          | target kernel |
///                          |  fusion_kind  |
///                          +---------------+
///                                 |           @analyzeAndVerifyKernel
///  |----------------------------------------------------------------------|
///  |                            /  \          @calculateTiling            |
///  |                            ....                                      |
///  |     +-------------------+        +------------------+                |
///  |     |  tiling case #0  |         |  tiling case #N  |                |
///  |     +-------------------+        +------------------+                |
///  |                 \                        /    @selectTiling          |
///  |                 |                       |     @createScheduleImpl    |
///  |          +-------------+           +-------------+                   |
///  |          | schedule #i |           | schedule #k |                   |
///  |          +-------------+           +-------------+                   |
///  |                |                          |      @applyScheduleImpl  |
///  |     +---------------------+      +---------------------+             |
///  |     | scheduled kernel #0 |      | scheduled kernel #N |             |
///  |     +---------------------+      +---------------------+             |
///  |----------------------------------------------------------------------|
class SchedulerBase : public schedule::ScheduleBuilder {
public:
  explicit SchedulerBase(func::FuncOp f, FusionKind kind);

  explicit SchedulerBase(func::FuncOp f,
                         std::unique_ptr<KernelInfo> &&kernelInfo,
                         std::unique_ptr<TilingInfo> &&tilingInfo);

  virtual ~SchedulerBase();

  /// Main entry point to do auto-scheduling.
  virtual LogicalResult runOnOperation(OpBuilder &opBuilder);

  /// Apply schedule to outlineFunc
  static LogicalResult applySchedule(func::FuncOp &funcOp,
                                     OpBuilder &opBuilder);

  /// Get and set auto schedule options.
  static AutoScheduleOptions getAutoScheduleOptions() { return options_; }
  static void setAutoScheduleOptions(const AutoScheduleOptions &options) {
    options_ = options;
  }

protected:
  //===--------------------------------------------------------------------===//
  // Type defs.
  //===--------------------------------------------------------------------===//
  using CacheIOResult = detail::CacheIOResult;
  using GetKernelIOOptions = detail::GetKernelIOOptions;
  using SetBufferSizeOptions = detail::SetBufferSizeOptions;
  using SetBufferSizeMode = transform::SetBufferSizeMode;

  /// Implementation of kernel analysis and verification.
  virtual LogicalResult analyzeAndVerifyKernelImpl();

  /// Implementation of host tiling calculation logic.
  virtual TilingComputeFn calculateTilingImpl() = 0;

  /// Implementation of creating a schedule from the input tiling key.
  virtual LogicalResult createScheduleImpl(TilingKey key,
                                           OpBuilder &opBuilder) = 0;

  /// Run pre-schedule procedure (e.g., kernel info collection and
  /// verification).
  virtual LogicalResult runPreScheduleProcedure(OpBuilder &opBuilder);

  /// Run post-schedule procedure (e.g., tiling pack).
  virtual LogicalResult runPostScheduleProcedure(OpBuilder &opBuilder);

  /// Run schedule procedure (including tiling calculation and schedule
  /// operation).
  LogicalResult runScheduleProcedure(OpBuilder &opBuilder);

  /// Run analysis on kernel function and verify constraints.
  LogicalResult analyzeAndVerifyKernel();
  void analyzeKernelForInterchangeAndDimensions();

  //===--------------------------------------------------------------------===//
  // Basic Schedule API (HFusion-specific part).
  //===--------------------------------------------------------------------===//

  /// Get handles to the outputs of the kernel.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Options for getting kernel outputs.
  /// \return RegularValueHandles to the producing op of kernel function's
  ///         return values.
  ValueHandles
  getKernelOutputs(OpBuilder &opBuilder,
                   const GetKernelIOOptions &options = GetKernelIOOptions());

  /// Get handles to the inputs of the kernel.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Options for getting kernel inputs.
  /// \return RegularValueHandles to the kernel function's input block argument.
  ValueHandles
  getKernelInputs(OpBuilder &opBuilder,
                  const GetKernelIOOptions &options = GetKernelIOOptions());

  /// Get handle to the tiling data.
  ///
  /// \param d Tiling data pointer.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return FuncArgHandle to the kernel function's block argument that
  ///         corresponds to the tiling data.
  ValueHandle *getTilingDataHandle(TilingData *d, OpBuilder &opBuilder);

  /// Get handles to each tiling data in tiling struct \c s.
  ///
  /// \param s A series of tiling data pointer.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return FuncArgHandles to the kernel function's block arguments that
  ///         correspond to the tiling data in tiling struct.
  ValueHandles getTilingStructHandles(SmallVector<TilingData *> s,
                                      OpBuilder &opBuilder);

  /// Tile the target linalg reduction op using \c scf.for ops by \c
  /// tileSizes.
  ///
  /// \param targets Value handles to \c linalg.reduce ops.
  /// \param tileSizes Value handles to mixed tile sizes.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param multiReduceNum The number of multi-reduced tensors.
  /// \return ForReductionTilingResult
  /// \note The input \c targets handles are invalidated.
  ForReductionTilingResult
  tileReductionUsingFor(ValueHandles &targets,
                        ValueHandleFoldResults &tileSizes, OpBuilder &opBuilder,
                        int64_t multiReduceNum = 1);

  /// Perform cache read on kernel inputs.
  ///
  /// After cache read, an unique tag name will be added to the cached op.
  /// For example:
  /// ```
  /// func.func @foo(%arg0):
  ///   linalg.copy ins(%arg0) outs(...) {__arg0__}
  /// ```
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandle to cached ops. Note that the handle points to
  ///         ALL cached ops. If you wish to obtain a more fine-grained control
  ///         over each ops, you can match by the attributed name returned by
  ///         `getCacheReadTag`.
  CacheIOResult cacheRead(OpBuilder &opBuilder);

  /// Get a unique identifier to the cached op by the function argument index.
  std::string getCacheReadTag(size_t funcArgIdx);

  /// Perform cache write on kernel outputs.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandle to the cached ops.
  CacheIOResult cacheWrite(OpBuilder &opBuilder);

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
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandles to `scf.forall` ops.
  /// \note The input `targets` handles are updated to the tiled linalg ops
  ///       and can be reused without invalidation.
  ForallTilingResult tileUsingForAll(ValueHandles &targets, int64_t blockDim,
                                     OpBuilder &opBuilder);

  /// TODO: Add return value to this API.
  /// Fuse target ops into containing ops one by one.
  ///
  /// When target op has multiple users in the containing op, the producer
  /// will be tiled according to the union of the users.
  ///
  /// \param targetOps Handles to fuse.
  /// \param containingLoops Handles to the initial containing ops.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param duplicateProducers Whether to duplicate producer when it is used
  ///        in multiple containing ops.
  /// \param applyCanonicalizeAfterEachFusion Whether to apply canonicalize
  ///        patterns to the IR after each fusion.
  /// \note If `applyCanonicalizeAfterEachFusion` is set to true, all input
  ///       handles are invalidated.
  ///       Otherwise, the handles in `containingLoop` are automatically
  ///       updated. The handles in `targetOps` are automatically updated if
  ///       and only if `len(containingLoop) == 1`.
  void fuseIntoContaining(ValueHandles &targetOps, ValueHandles &containingLoop,
                          OpBuilder &opBuilder, bool duplicateProducers = false,
                          bool applyCanonicalizeAfterEachFusion = true);

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
  /// \param opBuilder Reference to IRBuilder instance.
  /// \note The input `targets` handles are invalidated.
  void
  setBufferSize(ValueHandles &targets, int64_t bufferSize, OpBuilder &opBuilder,
                const SetBufferSizeOptions &options = SetBufferSizeOptions());

  /// Get handle to all intermediate producers.
  ValueHandle *getIntermediateProducers(OpBuilder &opBuilder);

  //===--------------------------------------------------------------------===//
  // APIs to run pre/post process passes.
  //===--------------------------------------------------------------------===//

  /// Apply target patterns
  LogicalResult applyPatternSets(Operation *op,
                                 const FrozenRewritePatternSet &patterns) const;

  /// Apply op flattening pass to \c target.
  LogicalResult applyOpFlattenPass(Operation *target,
                                   const FlattenOpsOptions &options = {}) const;

  /// Apply op fusion and outline pass to \c target.
  FailureOr<SmallVector<func::FuncOp>>
  applyOpFusionOutline(func::FuncOp target,
                       const HFusionOpFusionOptions &options = {}) const;

  /// Apply a pass to move the init operands corresponding to the \c target
  /// function results to the function arguments.
  /// \note This pass applies to the whole function.
  LogicalResult applyTensorResultToOutParamsPass(func::FuncOp target);

  /// Apply a pass to re-cache io to \c target.
  LogicalResult applyReCacheIOPass(func::FuncOp target) const;

  /// Apply a pass to apply symbol analysis to \c target.
  LogicalResult applySymbolAnalysisPass(func::FuncOp target) const;

  /// Apply a pass to aggressively bubble up extract slice to \c target
  LogicalResult applyAggressiveBubbleUpExtractSlice(func::FuncOp target) const;

  /// Apply a pass to merge consecutive insert extract slice to \c target
  LogicalResult
  applyMergeConsecutiveInsertExtractSlice(func::FuncOp target) const;

  /// Apply a pass to pack tiling data corresponding to the \c target
  /// function.
  /// \note This pass applies to the whole function.
  LogicalResult applyPackTilingDataPass(func::FuncOp target);

  /// Apply a pass to cse && canonicalize corresponding to the \c target
  /// function
  LogicalResult applyCSEAndCanonicalizePass(
      func::FuncOp target, ArrayRef<std::string> disabledPatterns = {}) const;

  //===--------------------------------------------------------------------===//
  // Getter methods.
  //===--------------------------------------------------------------------===//

  /// Get the enclosing module of the kernel function.
  ModuleOp getModule() { return module_; }
  /// Get a pointer to kernel info.
  KernelInfo *getKernelInfo() const { return kernelInfo_.get(); }
  /// Get pointer to the tiling info.
  TilingInfo *getTilingInfo() const { return tilingInfo_.get(); };
  /// Get MLIR Context.
  MLIRContext *getContext() const { return module_->getContext(); };
  /// Get the original kernel.
  func::FuncOp getOriginalKernel() {
    assert(originalKernel_);
    return originalKernel_;
  }
  /// Get the to-be-scheduled kernel.
  func::FuncOp getToBeScheduledKernel() {
    assert(toBeScheduledKernel_);
    return toBeScheduledKernel_;
  }
  /// Get the name to the original kernel.
  std::string getOriginalKernelName() {
    return getOriginalKernel().getSymName().str();
  }
  /// Get the name to the to-be-scheduled kernel.
  std::string getToBeScheduledKernelName() {
    return getToBeScheduledKernel().getSymName().str();
  }
  /// Getters for pass options.
  unsigned getBlockDim() { return options_.blockDim; }
  bool getEnableAutoMultiBuffer() { return options_.enableAutoMultiBuffer; }
  bool getEnableHostResourceMgmt() {
    return options_.enableManageHostResources;
  }
  int64_t getMaxBufferCntTuning() { return options_.maxBufferCntTuning; }
  ArrayRef<int64_t> getCubeTilingTuning() { return options_.cubeTilingTuning; }

  /// Getter KernelTilingMap
  IRMapping *getKernelTilingMap() const { return kernelTilingMap_.get(); }

  //===--------------------------------------------------------------------===//
  // Setter methods.
  //===--------------------------------------------------------------------===//

  /// Set the to-be-scheduled kernel.
  void setToBeScheduledKernel(func::FuncOp f) { toBeScheduledKernel_ = f; }
  /// Set tiling info.
  void setTilingInfo(TilingInfo &&info) {
    tilingInfo_ = std::make_unique<TilingInfo>(std::move(info));
  }
  /// Set the original kernel.
  void setOriginalKernel(func::FuncOp f) { originalKernel_ = f; }

private:
  using TilingIdx2TilingData = DenseMap<size_t, Value>;
  using CallSite2TilingIdx2TilingData =
      DenseMap<func::CallOp, TilingIdx2TilingData>;
  using CallerInfo = tiling::CallerInfo;

  /// Information needed to construct callee's arguments.
  struct CallSiteArgBuilderInfo {
    /// Mapping from tiling index (in ordered present in tiling struct) to the
    /// tiling data.
    TilingIdx2TilingData tilingIdx2TilingData{};
    /// Whether callee is the original kernel.
    bool calleeIsOriginalKernel{false};
  };

private:
  //===--------------------------------------------------------------------===//
  // Utility functions for Schedule APIs.
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Utility functions for Schedule.
  //===--------------------------------------------------------------------===//

  /// Check whether the schedule is nop.
  bool isNopSchedule() const;

  /// Run necessary procedures (such as generating an empty tiling function)
  /// even if the schedule is nop.
  LogicalResult runNopScheduleProcedure(OpBuilder &opBuilder);

  /// Cache input and output values.
  LogicalResult cacheIO(OpBuilder &opBuilder);

  /// Mark `hacc` input-related attributes to the kernel function.
  static LogicalResult markHACCInputArgAttr(func::FuncOp func);

  /// Calculate tiling struct for all tiling cases.
  LogicalResult calculateTiling(OpBuilder &opBuilder);

  /// Prune and select tiling cases if possible.
  LogicalResult selectTiling() const;

  /// Create one or more tiling cases and apply schedules.
  LogicalResult createAndApplySchedules(OpBuilder &opBuilder);

  /// Apply one specific schedule according to the input tiling info.
  LogicalResult applyScheduleImpl(OpBuilder &opBuilder);

  /// Prepare kernel function for scheduling and init schedule sequence.
  LogicalResult initSchedule(TilingKey key, OpBuilder &opBuilder);

  /// Reset things after doing schedule.
  void cleanUpAfterSchedule();

  /// Create switch cases for entry function to call scheduled functions
  /// according to tiling key and the callers of device kernels.
  LogicalResult fixCallSitesAndCaller(OpBuilder &opBuilder);

  /// Fix the call sites by replacing arguments.
  void doFixCallSite(CallerInfo &callerInfo, func::CallOp callSite,
                     CallSiteArgBuilderInfo &builderInfo,
                     DenseMap<Operation *, Operation *> &irMap,
                     OpBuilder &opBuilder) const;

  /// Generate callers for scheduled device functions.
  void generateDeviceCallers(func::CallOp callSite, Value tilingKey,
                             const SmallVector<Value> &newCallArgs,
                             DenseMap<Operation *, Operation *> &irMap,
                             OpBuilder &opBuilder) const;

  /// Construct new call site arguments.
  static SmallVector<Value>
  getNewArgsForCallSite(func::FuncOp caller, func::CallOp oldCallSite,
                        const CallSiteArgBuilderInfo &info,
                        OpBuilder &opBuilder);

  /// Get the tiling data arguments for the call sites.
  CallSite2TilingIdx2TilingData
  getTilingDataForCallSite(func::FuncOp caller, TilingInfo *tilingInfo,
                           const CallerInfo &callerInfo, OpBuilder &opBuilder);

  /// Dump current schedule and kernel function for debugging purposes.
  void dumpKernelAndSchedule();

private:
  /// Module enclosing the to-be-scheduled kernel.
  ModuleOp module_{nullptr};
  /// Original kernel function without scheduling.
  func::FuncOp originalKernel_{nullptr};
  /// Kernel function that will be scheduled.
  func::FuncOp toBeScheduledKernel_{nullptr};
  /// Information regarding the to-be-scheduled kernel.
  std::unique_ptr<KernelInfo> kernelInfo_{nullptr};
  /// Information regarding the tiling.
  std::unique_ptr<TilingInfo> tilingInfo_{nullptr};
  /// Underlying fusion kind.
  FusionKind kind_;
  /// Schedule options.
  static AutoScheduleOptions options_;

  /// Map between kernel function ops and tiling function ops
  std::unique_ptr<IRMapping> kernelTilingMap_;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBASE_H

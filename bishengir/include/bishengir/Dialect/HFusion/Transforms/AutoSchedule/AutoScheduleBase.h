//===- AutoScheduleBase.h -- Auto scheduler basic definitions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBASE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBASE_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/FusibleProducerAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/TilingUtils.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

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
namespace detail {
/// Struct to return the result of cache read/write.
struct CacheIOResult {
  ValueHandle *cachedOps;
};

/// Struct to return the result of tiling ops using forall.
struct ForallTilingResult {
  ValueHandles loops;
};

/// Struct to return the result of tiling ops using for.
struct ForTilingResult {
  // When tiling ops using for, the number of loops returned depends
  // on the number of "tile-able" axes.
  SmallVector<ValueHandles> loops;
};

/// Struct to return the result of tiling reduction ops using for.
struct ForReductionTilingResult {
  /// The partial reduction tiled op generated.
  ValueHandles partialReductionOp;
  /// The final reduction operation merging all the partial reductions.
  ValueHandles finalReductionOp;
  /// The fill op used to initialize the neutral element.
  /// We support tiling multi-reduce ops (i.e., reduce with multiple results),
  /// each reduction will have its own init op.
  SmallVector<ValueHandles> reductionInitOp;
  /// The loop operations that iterate over the tiles.
  ValueHandles loops;
};

/// Struct to return the results of tiling a loop.
struct LoopTileResult {
  ValueHandle *outerLoop;
  ValueHandle *innerLoop;
};

/// Enum class for holding transform patterns.
enum class TransformPatternKind : uint8_t {
  CSE = 0,                                // ApplyPatternsOp {apply_cse}
  CANONICALIZATION,                       // ApplyCanonicalizationPatternsOp
  MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE, // ApplyMergeConsecutiveInsertExtractSlicePatternsOp
  RESOLVE_RANKED_SHAPED_TYPE_RESULT_DIMS // ApplyResolveRankedShapedTypeResultDimsPatternsOp
};

/// Enum class for holding canonicalization patterns.
enum class CanonicalizationPatternKind : uint8_t {
  kSimplifyTrivialLoops = 0,      // SimplifyTrivialLoops
  kFoldTransposeWithTranspose = 1 // FoldTransposeWithTranspose Pattern
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

/// Struct for specifying options for matching IR values.
struct MatchOptions {
  /// Whether to reverse the order of payload objects in \c target.
  bool needsReverse{false};
  /// If set, will only match operations that are ancestors of
  /// \c childHandleOrValue.
  std::optional<std::variant<ValueHandle *, Value>> childHandleOrValue{};
};

/// Enum class for loop tile mode.
enum class LoopTileMode : uint8_t { kFactorMode = 0, kNPartMode };

/// Struct for specifying options for loop tiling.
struct LoopTileOptions {
  /// The tiling mode.
  LoopTileMode mode{LoopTileMode::kFactorMode};
  /// Whether reorder the tiled axis.
  bool isReorderMode{false};
};

/// Struct for specifying options for mapping scf.for to scf.forall.
struct MapForToForallOptions {
  /// Device mapping attribute for the `scf.forall` op.
  std::optional<DeviceMappingAttrInterface> mapping{std::nullopt};
  /// Whether the transformation is effectively immediately. If not, only an
  /// attribute is added to the `scf.for` op.
  bool annotateOnly{false};
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
class SchedulerBase {
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
  using ForallTilingResult = detail::ForallTilingResult;
  using ForTilingResult = detail::ForTilingResult;
  using ForReductionTilingResult = detail::ForReductionTilingResult;
  using TransformPatternKind = detail::TransformPatternKind;
  using CanonicalizationPatternKind = detail::CanonicalizationPatternKind;
  using SetBufferSizeMode = transform::SetBufferSizeMode;
  using GetKernelIOOptions = detail::GetKernelIOOptions;
  using MatchOptions = detail::MatchOptions;
  using NamedValueHandleArgs = detail::NamedValueHandleArgs;
  using LoopTileResult = detail::LoopTileResult;
  using LoopTileMode = detail::LoopTileMode;
  using LoopTileOptions = detail::LoopTileOptions;
  using MapForToForallOptions = detail::MapForToForallOptions;
  using SetBufferSizeOptions = detail::SetBufferSizeOptions;
  using Identifier = detail::Identifier;
  using OperationIdentifier = detail::OperationIdentifier;
  using AttributeIdentifier = detail::AttributeIdentifier;

  /// Implementation of kernel analysis and verification.
  virtual LogicalResult analyzeAndVerifyKernelImpl();

  /// Implementation of host tiling calculation logic.
  virtual TilingComputeFn calculateTilingImpl() = 0;

  /// Implementation of creating a schedule from the input tiling key.
  virtual LogicalResult createScheduleImpl(TilingKey key,
                                           OpBuilder &opBuilder) = 0;

  /// Mark tiling data according to tiling info and op relations.
  virtual LogicalResult markTilingData();

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
  // Basic Schedule API.
  //===--------------------------------------------------------------------===//

  /// Get handle to the kernel function.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return RegularValueHandle to `func.func` op.
  ValueHandle *getFuncHandle(OpBuilder &opBuilder);

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

  /// Get handle to ops with the \c opName in the kenrel, with additional
  /// constraints/options specified in \c options.
  ///
  /// \param opName Target op name.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Match options.
  /// \return NamedValueHandle to the target ops.
  ValueHandle *getOpsWithName(StringRef opName, OpBuilder &opBuilder,
                              const MatchOptions &options = MatchOptions());

  /// Get handle to ops with given attribute in the kernel, with additional
  /// constraints/options specified in \c options.
  ///
  /// \param attrName Attribute name.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param attrValue Attribute value.
  /// \param options Match options.
  /// \return NamedValueHandle to the target ops.
  ValueHandle *getOpsWithAttr(StringRef attrName, OpBuilder &opBuilder,
                              Attribute attrValue = Attribute(),
                              const MatchOptions &options = MatchOptions());

  /// Get handle to ops with given attributes in the kernel, with additional
  /// constraints/options specified in \c options.
  ///
  /// \param requiredAttrs List of required attributes.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param optionalAttrs List of optional attributes. If provided, at least
  ///                      one of them have to match.
  /// \param options Match options.
  /// \return NamedValueHandle to the target ops.
  ValueHandle *
  getOpsWithAttrs(const SmallVector<NamedAttribute> &requiredAttrs,
                  OpBuilder &opBuilder,
                  const SmallVector<NamedAttribute> &optionalAttrs = {},
                  const MatchOptions &options = MatchOptions());

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

  /// Tile the target linalg ops using \c scf.for ops by \c tileSizes.
  ///
  /// \param targets Value handles to linalg ops.
  /// \param tileSize Value handles to mixed tile sizes.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandles to `scf.for` ops.
  /// \note The input `targets` handles are updated to the tiled linalg ops
  /// and
  ///       can be reused without invalidation.
  ForTilingResult
  tileUsingFor(ValueHandles &targets, ValueHandleFoldResults &tileSizes,
               OpBuilder &opBuilder,
               ArrayRef<int64_t> interchangeAxis = ArrayRef<int64_t>{});

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

  /// Fuse independent loops together.
  ///
  /// \param loops Value handles to loops of the same type (i.e., all
  ///        `scf.for` or all `scf.forall`)
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandles to the fused loop.
  /// \note The input `loops` handles are invalidated.
  ValueHandle *fuseLoops(ValueHandles &loops, OpBuilder &opBuilder);

  /// Fuse independent loops for each dim together.
  ///
  /// \param loops Value handles to loops for each dimension
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return vector of NamedValueHandles to the fused loop.
  /// \note The input `loops` handles are invalidated.
  ValueHandles fuseLoopsForEachDim(ArrayRef<ValueHandles> tiledLoopsForEachDim,
                                   OpBuilder &builder);

  /// Coalesces the perfect loop nest enclosed by \c outerMostLoop
  ///
  /// \param outerMostLoop Value handle to the outer most loop (must be either
  ///                      `scf.for` or `affine.for` loop)
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandles to the coalesced loop.
  /// \note The input \c outerMostLoop handle is invalidated.
  ValueHandle *coalesceLoops(ValueHandle *outerMostLoop, OpBuilder &opBuilder);

  /// Tile the given loop by a factor of \c tileSize.
  ///
  /// IR before tiling:
  ///   for i : l to u step s
  ///     use(i)
  ///
  /// When tiling mode is `LoopTile::kFactor` (default):
  ///
  /// IR after tiling loop i by a factor of x:
  ///   for i.o : l to u step x
  ///    for i.i : 0 to min(u - i.o, x) step s
  ///      use(i.o + i.i)
  ///
  /// When no-min-max-bounds option is enabled:
  ///   for i.o : l to u step x
  ///    for i.i : 0 to x step s
  ///     if (i.o + i.i < u)
  ///       use(i.o + i.i)
  ///
  /// When tiling mode is `LoopTile::kNPart`:
  ///
  /// IR after tiling loop i by a factor of x:
  ///   for i.o 0 to x step 1
  ///     for i.i 0 to min(ceilDiv(u, x), u - i.o*(ceilDiv(u, x))) step 1
  ///       use(i.o*(ceilDiv(u, x)) + i.i)
  /// Note that this requires the loop to be normalized before tiling. And it
  /// cannot be used together with no-min-max-bounds option.
  ///
  /// \param targetLoop Value handle to the target `scf.for` loop.
  /// \param tileSize ValueHandleFoldResult to the Size of tiling
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Loop tiling options.
  /// \return Handles to the outer and inner loop after tiling.
  /// \note The input \c targetLoop handle is invalidated.
  LoopTileResult tileLoop(ValueHandle *targetLoop,
                          ValueHandleFoldResult tileSize, OpBuilder &opBuilder,
                          const LoopTileOptions &options = LoopTileOptions());

  /// Normalize the given loop (i.e., has step 1 while preserving trip count)
  ///
  /// \param targetLoop Value handle to the target `scf.for` loop.
  /// \note The input \c targetLoop handle is updated to the loop after being
  ///       normalized.
  void normalizeLoop(ValueHandle *targetLoop, OpBuilder &opBuilder);

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

  /// Split `handle` into `splitSize` parts.
  ///
  /// \param handle Target value handle.
  /// \param splitSize Number of parts to split.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return RegularValueHandles to the splitted handles.
  /// \note Runtime error will occur if the handle cannot be split into the
  ///       request parts.
  ValueHandles splitHandle(ValueHandle *handle, size_t splitSize,
                           OpBuilder &opBuilder);

  /// Apply canonicalize pass.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \note This function resets all handles.
  void applyCanonicalization(OpBuilder &opBuilder);

  /// Apply common subexpression elimination pass.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \note This function resets all handles.
  void applyCSE(OpBuilder &opBuilder);

  /// Apply `patterns` to `target`.
  ///
  /// \param target Target handle to apply patterns.
  /// \param patterns List of `TransformPatternKind` to apply.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param disablePatterns List of `CanonicalizationPatternKind` to disable.
  void applyPatterns(
      ValueHandle *target, const SmallVector<TransformPatternKind> &patterns,
      OpBuilder &opBuilder,
      const SmallVector<CanonicalizationPatternKind> &disablePatterns = {});

  /// Apply one-shot-bufferization to the kernel function.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  void applyOneShotBufferization(OpBuilder &opBuilder);

  /// Map `scf.forall` to HIVM Block ops.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  void mapForallToHIVMBlocks(OpBuilder &opBuilder);

  /// Map `scf.for` to `scf.forall` op, with optional mapping.
  ///
  /// \param targetLoop Handle to `scf.for` op.
  /// \param mapping Optional mapping for`scf.forall` op.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Map for to forall options.
  /// \return NamedValueHandles to the `scf.forall` op if `options.annotateOnly`
  ///         is set to true. Otherwise, its the NamedValueHandles to the
  ///         `scf.for` op after annotation.
  /// \note The input `targetLoop` handle is invalidated.
  ValueHandle *mapForToForall(
      ValueHandle *targetLoop, OpBuilder &opBuilder,
      const MapForToForallOptions &options = MapForToForallOptions());

  using RegionBuilderFn =
      llvm::function_ref<void(ImplicitLocOpBuilder &, Block &)>;

  /// Construct `transform.foreachOp` and return its results.
  ///
  /// \param target Target to apply `transform.foreachOp`
  /// \param resultTypes `transform.foreachOp`'s result types.
  /// \param regionBuilder Lambda to build `transform.foreachOp`'s body.
  /// \param opBuilder Reference to IRBuilder instance.
  ResultRange createForEachOp(Value target, TypeRange resultTypes,
                              RegionBuilderFn regionBuilder,
                              OpBuilder &opBuilder);

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

  /// Apply a pass to remove cache io to \c target.
  LogicalResult applyRemoveCacheIOPass(func::FuncOp target);

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
  // Value Handle related API.
  //===--------------------------------------------------------------------===//

  /// Create and record handle.
  template <class T, class... Args>
  T *record(Value v, OpBuilder &b, Args &&...args) {
    return handleRecord_->record<T>(
        recordImpl(v, b, std::forward<Args>(args)...));
  }

  template <class T, class... Args>
  std::optional<T *> tryFetchRecord(Args &&...args) {
    static_assert(std::is_same_v<T, NamedValueHandle> &&
                  "Only support fetching NamedValueHandle");
    return handleRecord_->tryFetchRecordImpl(std::forward<Args>(args)...);
  }

  /// Reset all recorded handles.
  /// \note Different value handle kind have different implementation.
  void resetAllHandles() { return handleRecord_->resetAllHandles(); }

  //===--------------------------------------------------------------------===//
  // Getter methods.
  //===--------------------------------------------------------------------===//

  /// Get the handle to transform sequence's block argument.
  Value getTransformSeqHandle() { return transformSeqBlockHandle_; }
  /// Get the enclosing module of the kernel function.
  ModuleOp getModule() { return module_; }
  /// Get a pointer to kernel info.
  KernelInfo *getKernelInfo() const { return kernelInfo_.get(); }
  /// Get pointer to the tiling info.
  TilingInfo *getTilingInfo() const { return tilingInfo_.get(); };
  /// Get MLIR Context.
  MLIRContext *getContext() const { return module_->getContext(); };
  /// Get reference to the handle record.
  HandleRecord *getHandleRecord() { return handleRecord_.get(); }
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

  /// Update the handle to transform sequence's block argument.
  void setTransformSeqHandle(Value newHandle) {
    transformSeqBlockHandle_ = newHandle;
  }
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

  /// Get value from handle.
  ///
  /// \param handle Pointer to a value handle instance.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return Value corresponding to the input handle.
  /// \note User should guarantee that the input handle is valid, otherwise
  ///       a runtime error is produced.
  Value getValue(ValueHandle *handle, OpBuilder &opBuilder);

  /// Get values from handles.
  ///
  /// \param handles Vector of pointer to a value handle instance.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return Values corresponding to the input handles.
  SmallVector<Value> getValues(const ValueHandles &handle,
                               OpBuilder &opBuilder);

  /// Match and return IR values with \c identifier of type \c type, with
  /// additional constraints/options specified in \c options.
  ///
  /// \param target Target to perform matching.
  /// \param identifier Identifier information.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Match options.
  /// \return Values corresponding to the input handles.
  Value matchByIdentifier(Value target, const Identifier &identifier,
                          OpBuilder &opBuilder,
                          const MatchOptions &options = MatchOptions());

  /// Merge handles whose type is `handleType` and return the merged
  /// handle's value.
  ///
  /// \param handles Vector of handle values to merge.
  /// \param handleType Handle's type.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return Value holding the merged handles.
  Value mergeHandles(const SmallVectorImpl<Value> &handles,
                     transform::TransformHandleTypeInterface handleType,
                     OpBuilder &opBuilder);

  /// Annotate the IR values corresponding to \c target with \c attrName.
  ///
  /// \param target Target value to annotate.
  /// \param attrName Attribute name to add to operation's attribute list.
  /// \param opBuilder Reference to IRBuilder instance.
  void annotateByAttr(Value target, StringRef attrName, OpBuilder &opBuilder);

  /// Get handle value to the kernel function.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return Value to a handle of `func.func` op.
  Value getFuncValue(OpBuilder &opBuilder);

  /// Get handle to ops with the specified identifier information in the kernel,
  /// with additional constraints/options specified in \c options.
  ///
  /// \param identifier Identifier info.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Match options.
  /// \return NamedValueHandle to the target ops.
  ValueHandle *
  getOpsWithIdentifier(const Identifier &identifier, OpBuilder &opBuilder,
                       const MatchOptions &options = MatchOptions());

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

  /// Helper function to convert `tensor.empty` to
  /// `bufferization.alloc_tensor`.
  ///
  /// \note This function is invoked in `applyOneShotBufferization` and should
  ///       not be called separately.
  void bufferizeEmptyTensor(OpBuilder &opBuilder);

  /// Create and record NamedValueHandle.
  NamedValueHandle recordImpl(Value target, OpBuilder &opBuilder,
                              const NamedValueHandleArgs &args);

  /// Create and record RegularValueHandle.
  RegularValueHandle recordImpl(Value target, OpBuilder &opBuilder);

  /// Create and record FuncArgHandle.
  FuncArgHandle recordImpl(Value target, OpBuilder &opBuilder,
                           size_t funcArgNum);

  std::pair<SmallVector<int64_t>, SmallVector<Value>>
  unpackFoldResults(ValueHandleFoldResults &values, OpBuilder &opBuilder);

private:
  /// Module enclosing the to-be-scheduled kernel.
  ModuleOp module_{nullptr};
  /// Original kernel function without scheduling.
  func::FuncOp originalKernel_{nullptr};
  /// Kernel function that will be scheduled.
  func::FuncOp toBeScheduledKernel_{nullptr};
  /// The transform sequence block argument value.
  Value transformSeqBlockHandle_;
  /// Information regarding the to-be-scheduled kernel.
  std::unique_ptr<KernelInfo> kernelInfo_{nullptr};
  /// Information regarding the tiling.
  std::unique_ptr<TilingInfo> tilingInfo_{nullptr};
  /// Record keeping all allocated value handles.
  std::unique_ptr<HandleRecord> handleRecord_{nullptr};
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

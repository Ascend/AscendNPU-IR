//===--------- Builder.h - Transform op wrapper for schedules ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect-neutral wrapper around common transform ops, integrated with the
// ValueHandle system: builders create handles, update them in place where the
// transform op yields a new payload op, and invalidate or re-match them where
// the payload IR is structurally rewritten.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_SCHEDULE_BUILDER_H
#define BISHENGIR_DIALECT_ANALYSIS_SCHEDULE_BUILDER_H

#include "bishengir/Dialect/Analysis/Schedule/ValueHandle.h"

#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <optional>
#include <variant>

namespace mlir {
class Block;
class Location;
class OpBuilder;

namespace func {
class FuncOp;
} // namespace func

namespace schedule {

/// Default identifiers used to annotate payload ops and re-match handles.
inline constexpr llvm::StringLiteral kTiledForTagName = "__tiled_for__";
inline constexpr llvm::StringLiteral kTiledForAllTagName = "__tiled_forall__";
inline constexpr llvm::StringLiteral kFusedLoopTagName = "__fused_loop__";
inline constexpr llvm::StringLiteral kForallLoopTagName = "__forall__";
inline constexpr llvm::StringLiteral kCoalescedLoopTagName =
    "__coalesced_loop__";
inline constexpr llvm::StringLiteral kTileReductionPartialReductionOpTagName =
    "__partial_reduction_op__";
inline constexpr llvm::StringLiteral kTileReductionFinalReductionOpTagName =
    "__final_reduction_op__";
inline constexpr llvm::StringLiteral kTileReductionInitOpTagName =
    "__reduction_init_op__";
inline constexpr llvm::StringLiteral kTileReductionLoopTagName =
    "__reduction_loop__";

namespace detail {

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

/// Enum class for loop tile mode.
enum class LoopTileMode : uint8_t { kFactorMode = 0, kNPartMode };

/// Struct for specifying options for tiling a single loop.
struct LoopTileOptions {
  /// Tiling mode.
  LoopTileMode mode{LoopTileMode::kFactorMode};
  /// Whether to reorder the loop iterators when tiling.
  bool isReorderMode{false};
};

/// Struct for specifying options for mapping scf.for to scf.forall.
struct MapForToForallOptions {
  /// Device mapping attribute for the `scf.forall` op.
  std::optional<DeviceMappingAttrInterface> mapping{std::nullopt};
  /// Whether the transformation is effectively immediate. If not, only an
  /// attribute is added to the `scf.for` op.
  bool annotateOnly{false};
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

} // namespace detail

/// Wraps common transform ops with handle bookkeeping.
///
/// Each method emits transform IR into the current transform sequence and
/// returns/updates value handles. Handles that survive the transform keep
/// their identity (e.g. the tiled op replaces the original target handle);
/// handles whose payload ops are structurally rewritten are invalidated or
/// marked for re-matching.
class ScheduleBuilder {
public:
  using ForallTilingResult = detail::ForallTilingResult;
  using ForTilingResult = detail::ForTilingResult;
  using ForReductionTilingResult = detail::ForReductionTilingResult;
  using TransformPatternKind = detail::TransformPatternKind;
  using CanonicalizationPatternKind = detail::CanonicalizationPatternKind;
  using LoopTileResult = detail::LoopTileResult;
  using LoopTileMode = detail::LoopTileMode;
  using LoopTileOptions = detail::LoopTileOptions;
  using MapForToForallOptions = detail::MapForToForallOptions;
  using MatchOptions = detail::MatchOptions;
  using NamedValueHandleArgs = detail::NamedValueHandleArgs;
  using Identifier = detail::Identifier;
  using OperationIdentifier = detail::OperationIdentifier;
  using AttributeIdentifier = detail::AttributeIdentifier;
  using RegionBuilderFn =
      llvm::function_ref<void(ImplicitLocOpBuilder &, Block &)>;

  ScheduleBuilder() : handleRecord_(std::make_unique<HandleRecord>()) {}
  virtual ~ScheduleBuilder() = default;

  //===--------------------------------------------------------------------===//
  // Handle record management.
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
  void resetAllHandles() { handleRecord_->resetAllHandles(); }

  HandleRecord *getHandleRecord() { return handleRecord_.get(); }

  /// Get the handle to transform sequence's block argument.
  Value getTransformSeqHandle() { return transformSeqBlockHandle_; }
  /// Update the handle to transform sequence's block argument.
  void setTransformSeqHandle(Value newHandle) {
    transformSeqBlockHandle_ = newHandle;
  }

  //===--------------------------------------------------------------------===//
  // Handle materialization.
  //===--------------------------------------------------------------------===//

  /// Get value from handle.
  ///
  /// \note User should guarantee that the input handle is valid, otherwise
  ///       a runtime error is produced.
  Value getValue(ValueHandle *handle, OpBuilder &opBuilder);

  /// Get values from handles.
  SmallVector<Value> getValues(const ValueHandles &handles,
                               OpBuilder &opBuilder);

  //===--------------------------------------------------------------------===//
  // Matching and annotating.
  //===--------------------------------------------------------------------===//

  /// Get handle value to the kernel function.
  Value getFuncValue(OpBuilder &opBuilder);

  /// Get handle to the kernel function.
  ValueHandle *getFuncHandle(OpBuilder &opBuilder);

  /// Get handle to ops with the \c opName in the kernel, with additional
  /// constraints/options specified in \c options.
  ValueHandle *getOpsWithName(StringRef opName, OpBuilder &opBuilder,
                              const MatchOptions &options = MatchOptions());

  /// Get handle to ops with given attribute in the kernel, with additional
  /// constraints/options specified in \c options.
  ValueHandle *getOpsWithAttr(StringRef attrName, OpBuilder &opBuilder,
                              Attribute attrValue = Attribute(),
                              const MatchOptions &options = MatchOptions());

  /// Get handle to ops with given attributes in the kernel, with additional
  /// constraints/options specified in \c options.
  ValueHandle *
  getOpsWithAttrs(const SmallVector<NamedAttribute> &requiredAttrs,
                  OpBuilder &opBuilder,
                  const SmallVector<NamedAttribute> &optionalAttrs = {},
                  const MatchOptions &options = MatchOptions());

  /// Match and return IR values with \c identifier of type \c type, with
  /// additional constraints/options specified in \c options.
  Value matchByIdentifier(Value target, const Identifier &identifier,
                          OpBuilder &opBuilder,
                          const MatchOptions &options = MatchOptions());

  /// Split `handle` into `splitSize` parts.
  ///
  /// \note Runtime error will occur if the handle cannot be split into the
  ///       request parts.
  ValueHandles splitHandle(ValueHandle *handle, size_t splitSize,
                           OpBuilder &opBuilder);

  //===--------------------------------------------------------------------===//
  // Stateless transform helpers.
  //
  // These methods only wrap a transform op around their arguments and never
  // read or update ScheduleBuilder state (no handle record, no transform
  // sequence entry), so they can be called without an instance, e.g.
  // `ScheduleBuilder::createReverseOp(...)`.
  //===--------------------------------------------------------------------===//

  /// Create the op reversing the payload object order in `target`.
  /// Creates the dialect-neutral `transform.reverse` op
  /// (Dialect/Transform).
  static Value createReverseOp(Value target, OpBuilder &opBuilder);

  /// Annotate the IR values corresponding to \c target with \c attrName.
  static void annotateByAttr(Value target, StringRef attrName,
                             OpBuilder &opBuilder);

  /// Merge handles whose type is `handleType` and return the merged
  /// handle's value.
  static Value mergeHandles(const SmallVectorImpl<Value> &handles,
                            transform::TransformHandleTypeInterface handleType,
                            OpBuilder &opBuilder);

  /// Construct `transform.foreachOp` and return its results.
  static ResultRange createForEachOp(Value target, TypeRange resultTypes,
                                     RegionBuilderFn regionBuilder,
                                     OpBuilder &opBuilder);

  //===--------------------------------------------------------------------===//
  // Transform op wrappers.
  //===--------------------------------------------------------------------===//

  /// Tile the target linalg ops using \c scf.forall ops by \c blockDim, with
  /// the given device mapping attributes.
  ///
  /// \param targets Value handles to linalg ops.
  /// \param staticNumThreads Number of threads.
  /// \param mapping Array of device mapping attributes for the `scf.forall`.
  /// \return NamedValueHandles to `scf.forall` ops.
  /// \note The input `targets` handles are updated to the tiled linalg ops
  ///       and can be reused without invalidation.
  ForallTilingResult tileUsingForAll(ValueHandles &targets,
                                     int64_t staticNumThreads,
                                     ArrayAttr mapping, OpBuilder &opBuilder);

  /// Tile the target linalg ops using \c scf.for ops by \c tileSizes.
  ///
  /// \param targets Value handles to linalg ops.
  /// \param tileSize Value handles to mixed tile sizes.
  /// \param interchangeAxis Interchange axis for tiling.
  /// \return NamedValueHandles to `scf.for` ops.
  /// \note The input `targets` handles are updated to the tiled linalg ops
  ///       and can be reused without invalidation.
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
  /// \return NamedValueHandles to the fused loop.
  /// \note The input `loops` handles are invalidated.
  ValueHandle *fuseLoops(ValueHandles &loops, OpBuilder &opBuilder);

  /// Fuse independent loops for each dim together.
  ///
  /// \param loops Value handles to loops for each dimension
  /// \return vector of NamedValueHandles to the fused loop.
  /// \note The input `loops` handles are invalidated.
  ValueHandles fuseLoopsForEachDim(ArrayRef<ValueHandles> tiledLoopsForEachDim,
                                   OpBuilder &builder);

  /// Coalesces the perfect loop nest enclosed by \c outerMostLoop.
  ///
  /// \param outerMostLoop Value handle to the outer most loop (must be either
  ///                      `scf.for` or `affine.for` loop)
  /// \return NamedValueHandles to the coalesced loop.
  /// \note The input \c outerMostLoop handle is invalidated.
  ValueHandle *coalesceLoops(ValueHandle *outerMostLoop, OpBuilder &opBuilder);

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

  /// Tile the given loop by a factor of \c tileSize.
  ///
  /// IR before tiling:
  ///     scf.for %i = 0 to 100 step 1 {
  ///         ...
  ///     }
  /// IR after tiling:
  ///     scf.for %i = 0 to 50 step 1 {
  ///         scf.for %j = 0 to 2 step 1 {
  ///             ...
  ///         }
  ///     }
  ///
  /// \param targetLoop Value handle to the target loop.
  /// \param tileSize Tile size (static or handle to dynamic size).
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Loop tiling options.
  /// \return Handles to the outer and inner loops.
  /// \note The input \c targetLoop handle is invalidated.
  LoopTileResult tileLoop(ValueHandle *targetLoop,
                          ValueHandleFoldResult tileSize, OpBuilder &opBuilder,
                          const LoopTileOptions &options);

  /// Normalize the given loop.
  ///
  /// \param targetLoop Value handle to the target loop.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \note The input \c targetLoop handle is updated to the normalized loop.
  void normalizeLoop(ValueHandle *targetLoop, OpBuilder &opBuilder);

  /// Map the given `scf.for` loop to an `scf.forall` loop.
  ///
  /// \param targetLoop Value handle to the target loop.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Mapping options.
  /// \return Handle to the `scf.forall` loop, or the input loop handle when
  ///         \c annotateOnly is set.
  /// \note The input \c targetLoop handle is invalidated unless
  ///       \c annotateOnly is set.
  ValueHandle *mapForToForall(ValueHandle *targetLoop, OpBuilder &opBuilder,
                              const MapForToForallOptions &options);

  /// Apply canonicalize pass.
  /// \note This function resets all handles.
  void applyCanonicalization(OpBuilder &opBuilder);

  /// Apply common subexpression elimination pass.
  /// \note This function resets all handles.
  void applyCSE(OpBuilder &opBuilder);

  /// Apply `patterns` to `target`.
  ///
  /// \param target Target handle to apply patterns.
  /// \param patterns List of `TransformPatternKind` to apply.
  /// \param disablePatterns List of `CanonicalizationPatternKind` to disable.
  void applyPatterns(
      ValueHandle *target, const SmallVector<TransformPatternKind> &patterns,
      OpBuilder &opBuilder,
      const SmallVector<CanonicalizationPatternKind> &disablePatterns = {});

  /// Unpack fold results into static and dynamic tile sizes.
  std::pair<SmallVector<int64_t>, SmallVector<Value>>
  unpackFoldResults(ValueHandleFoldResults &values, OpBuilder &opBuilder);

private:
  /// Create and record NamedValueHandle.
  NamedValueHandle recordImpl(Value target, OpBuilder &opBuilder,
                              const NamedValueHandleArgs &args);

  /// Create and record RegularValueHandle.
  static RegularValueHandle recordImpl(Value target, OpBuilder &opBuilder);

  /// Create and record FuncArgHandle.
  static FuncArgHandle recordImpl(Value target, OpBuilder &opBuilder,
                                  size_t funcArgNum);

  /// Get handle to ops with the specified identifier information in the
  /// kernel, with additional constraints/options specified in \c options.
  ValueHandle *
  getOpsWithIdentifier(const Identifier &identifier, OpBuilder &opBuilder,
                       const MatchOptions &options = MatchOptions());

private:
  /// Record keeping all allocated value handles.
  std::unique_ptr<HandleRecord> handleRecord_;
  /// The transform sequence block argument value.
  Value transformSeqBlockHandle_;
};

} // namespace schedule
} // namespace mlir

#endif // BISHENGIR_DIALECT_ANALYSIS_SCHEDULE_BUILDER_H

//===- AnyPBRSchedule.h -- Any Pointwise/Broadcast/Reduce Schedule --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBRSCHEDULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBRSCHEDULE_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRKernelInfo.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRKernelInfoCollector.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <utility>

namespace mlir {
class OpBuilder;

namespace hfusion {

/// Scheduler for kernels with any axis reduction operations and other
/// elemwise/broadcast operations.
class AnyPBRScheduler : public SchedulerBase {
public:
  explicit AnyPBRScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(
            funcOpIn,
            std::make_unique<AnyPBRKernelInfo>(funcOpIn->getContext()),
            std::make_unique<TilingInfo>()){};

  /// Implementation of kernel analysis and verification.
  LogicalResult analyzeAndVerifyKernelImpl() override;

  /// Implementation of tiling case and schedule sequence generation.
  ///
  /// Tiling Case #0 ~ #M-2 #M ~ #N-1 (Split-Parallel):
  /// Only tile the parallel dims.
  ///
  /// \code
  /// for block.idx in block.dim:
  ///   for ub.idx in ub_loop_cnt:
  ///     copyIn(ub_buffer_size, D)
  ///     compute(ub_buffer_size, D)
  ///     copyOut(ub_buffer_size, D)
  /// \endcode
  ///
  /// Tiling Case #M-1 (Split-Reduction):
  ///
  /// \code
  /// for block.idx in block.dim:
  ///   for ub.idx in ub_n_loop_cnt:
  ///     for r.idx in ub_d_loop_cnt:
  ///       copyIn(1, rfactor_size)
  ///       compute(1, rfactor_size)
  ///
  ///     reduce(1, rfactor_size)
  ///     compute(1, 1)
  ///     copyOut(1, 1)
  ///
  ///     for d.idx in ub_d_loop_cnt:
  ///       copyIn(1, rfactor_size)
  ///       compute(1, rfactor_size)
  ///       copyOut(1, rfactor_size)
  /// \endcode
  ///
  /// Tiling Data is organized as:
  ///   1.   Tiling Key
  ///   2.   UB Tile Size in Parallel Dim 0
  ///   3.   UB Tile Size in Parallel Dim 1
  ///   ...
  ///   M.   UB Tile Size in Reduction Dim M-2
  ///   ...
  ///   N.   UB Tile Size in Parallel Dim N-2
  ///   N+1. Buffer size in bytes
  TilingComputeFn calculateTilingImpl() override;
  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override;

protected:
  //===--------------------------------------------------------------------===//
  // Getter methods
  //===--------------------------------------------------------------------===//

  /// Return whether reduce axis should be split.
  ///
  /// e.g. suppose tiling key is dim_{N-2}
  /// - reduce dim is dim_{N-3}, then we should tile one size from reduce axis.
  /// - reduce dim is dim_{N-1}, then no need to tile the reduce axis.
  /// the tile sizes will be [1, ..., 1, ubAvailableNum / dim_{N-2}, dim_{N-1}]
  bool needToSplitReduction(TilingKey key) const;

  /// Return reduction related information.
  bool hasReduceOp() const;

  /// Return vector of tile factors for the current op based on the \c axisMask
  /// and \c tilingMask.
  ///
  /// \param tilingKey The current tiling case's key.
  /// \param tilingData The tiling data.
  /// \param axisMask A bit vector indicating the axis relationship of the
  ///                 current target w.r.t. the anchor value.
  /// \param tilingMask A bit vector indicating which axis to tile in terms of
  ///                   the anchor value.
  ValueHandleFoldResults getTilingFactors(
      TilingKey tilingKey, const SmallVector<TilingData *> &tilingData,
      const BitVector &axisMask, const BitVector &tilingMask,
      ArrayRef<int64_t> tileSizeInterchange = ArrayRef<int64_t>{}) const;

  /// Get the interchange for tiling.
  SmallVector<int64_t>
  getOpInterchangeAxes(SmallVector<int64_t> normalizedInterchange) const;

  //===--------------------------------------------------------------------===//
  // Helper functions for schedule implementation
  //===--------------------------------------------------------------------===//

  /// Apply canonicalization patterns.
  ///
  /// Disabled `kSimplifyTrivialLoops` because loop handles might be invalidate
  /// if the tiled loop is trivial during compile-time
  void applyCanonicalization(OpBuilder &opBuilder);

  /// Set buffer size for targets based on tiling info.
  LogicalResult setBufferSize(const TilingInfo *tilingInfo,
                              ValueHandles &targetsToSetBufferSize,
                              OpBuilder &opBuilder);

  //===--------------------------------------------------------------------===//
  // Tile and fuse into related functions.
  //===--------------------------------------------------------------------===//

  /// Tile `hfusion.store`'s parallel axes and fuse producers into the tiled
  /// loop.
  ///
  /// \result the tiled and coalesed loop; nullptr if the number of parallel
  /// axis is zero.
  ValueHandle *
  tileParallelAxesAndFuseProducers(TilingKey tilingKey, TilingInfo &tilingInfo,
                                   const AnyPBRKernelInfo &kernelInfo,
                                   OpBuilder &opBuilder);

  /// Tile reduction axes for `hfusion.store` or `linalg.reduce`, and fuse
  /// producers into the tiled loop.
  ///
  /// \result the tiled and coalesed loop; nullptr if the number of reduction
  /// axis is zero.
  ValueHandle *tileReduceAxesAndFuseProducers(
      TilingKey key, TilingInfo &tilingInfo, const AnyPBRKernelInfo &anyPBRInfo,
      OpBuilder &opBuilder,
      std::optional<ValueHandles> targetsToSetBufferSize = std::nullopt);

  /// Merge the producer handles and return an unique collection of producers to
  /// fuse into.
  ValueHandles
  mergeProducerHandles(const SmallVector<NamedAttribute> &producerIdentifiers,
                       const MatchOptions &options, OpBuilder &opBuilder);

  //===--------------------------------------------------------------------===//
  // Bind multi-core axes related functions.
  //===--------------------------------------------------------------------===//

  void bindLoopToMulticore(ValueHandle *loop, AnyPBRKernelInfo &kernelInfo,
                           OpBuilder &opBuilder,
                           const TilingData *numOfCores = nullptr);

  /// Get multi-core num expr for Parallel and Reduce axis respectively.
  std::pair<Expr, Expr> getMultiCoreNum(Expr totalCores,
                                        const SetVector<int64_t> &reduceDims,
                                        const SmallVector<Expr> &tileSizes,
                                        const SmallVector<Expr> &dimSizes,
                                        StmtExprBuilder *opBuilder) const;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBRSCHEDULE_H

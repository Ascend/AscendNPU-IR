//===- PureElemwiseSchedule.h -- Schedule for PureElemwise Op ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_PUREELEMWISESCHEDULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_PUREELEMWISESCHEDULE_H

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
class Location;
class OpBuilder;
namespace hfusion {

constexpr size_t kPureElemwiseTilingCnt = 3;

/// Scheduler for pure element wise kernels.
class PureElemwiseScheduler : public SchedulerBase {
public:
  explicit PureElemwiseScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(funcOpIn,
                      std::make_unique<KernelInfo>(FusionKind::PureElemwise,
                                                   funcOpIn->getContext()),
                      std::make_unique<TilingInfo>(kPureElemwiseTilingCnt)){};

private:
  /// Implementation of tiling case and schedule sequence generation.
  ///
  /// Tiling Case #1:
  ///
  /// \code
  /// for block.idx in block.dim:
  ///   for ub.idx in ub_loop_cnt:
  ///     copyIn(ub_buffer_size)
  ///     compute(ub_buffer_size)
  ///     copyOut(ub_buffer_size)
  /// \endcode
  TilingComputeFn calculateTilingImpl() override;
  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_PUREELEMWISESCHEDULE_H

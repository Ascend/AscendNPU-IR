//===- VFSchedule.h -- Vector Function Schedule ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_VFSCHEDULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_VFSCHEDULE_H

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRSchedule.h"

namespace mlir {
class OpBuilder;

namespace hfusion {

/// Vector Function Scheduler (VFScheduler) implementation, serving as a
/// specialized variant of the AnyPBRScheduler.
///
/// VFScheduler introduces several key modifications compared with
/// AnyPBRScheduler:
/// - Adapts tile sizes to match register length (256 bytes on A5 architecture)
/// - Remove Load/Store and add vectorization after main transformations
/// - Disregards unnecessary configurations such as multi-core support and
///   buffer size annotations
class VFScheduler final : public AnyPBRScheduler {
public:
  explicit VFScheduler(func::FuncOp funcOpIn) : AnyPBRScheduler(funcOpIn){};

  TilingComputeFn calculateTilingImpl() override;
  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override;
  LogicalResult markTilingData() override;
  LogicalResult runPostScheduleProcedure(OpBuilder &opBuilder) override;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_VFSCHEDULE_H

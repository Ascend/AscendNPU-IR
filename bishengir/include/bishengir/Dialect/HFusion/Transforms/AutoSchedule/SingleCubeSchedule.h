//===- SingleCubeSchedule.h -- Schedule for Single Cube Op ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SINGLECUBESCHEDULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SINGLECUBESCHEDULE_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Location;
class OpBuilder;

namespace transform {
class NamedSequenceOp;
} // namespace transform

namespace hfusion {

constexpr size_t kSingleCubeTilingCnt = 12;

/// Scheduler for single cube kernels.
class SingleCubeScheduler : public SchedulerBase {
public:
  explicit SingleCubeScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(funcOpIn, std::make_unique<KernelInfo>(),
                      std::make_unique<TilingInfo>(kSingleCubeTilingCnt)){};

  TilingComputeFn calculateTilingImpl() override;

  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override;

  LogicalResult runPreScheduleProcedure(OpBuilder &opBuilder) override;

  LogicalResult runPostScheduleProcedure(OpBuilder &opBuilder) override;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SINGLECUBESCHEDULE_H

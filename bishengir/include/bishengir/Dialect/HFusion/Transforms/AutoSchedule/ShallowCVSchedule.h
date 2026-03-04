//===- ShallowCVSchedule.h -- Schedule for Shallow CV Op --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SHALLOWCVSCHEDULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SHALLOWCVSCHEDULE_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
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

/// Scheduler for shallow cv kernels.
class ShallowCVScheduler : public SchedulerBase {
public:
  explicit ShallowCVScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(funcOpIn, FusionKind::ShallowCV){};

  LogicalResult runOnOperation(OpBuilder &opBuilder) override;

  LogicalResult analyzeAndVerifyKernelImpl() override { return success(); }

  TilingComputeFn calculateTilingImpl() override { return nullptr; };

  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override {
    return success();
  }
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SHALLOWCVSCHEDULE_H

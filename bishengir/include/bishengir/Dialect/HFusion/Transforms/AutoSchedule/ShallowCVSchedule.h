//===- ShallowCVSchedule.h -- Schedule for Shallow CV Op --------*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SHALLOWCVSCHEDULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SHALLOWCVSCHEDULE_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Location;

namespace transform {
class NamedSequenceOp;
} // namespace transform

namespace hfusion {

/// Scheduler for shallow cv kernels.
class ShallowCVScheduler : public SchedulerBase {
public:
  explicit ShallowCVScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(funcOpIn, FusionKind::ShallowCV){};

  LogicalResult runOnOperation() override;

  LogicalResult analyzeAndVerifyKernelImpl() override { return success(); }

  TilingComputeFn calculateTilingImpl() override { return nullptr; };

  LogicalResult createScheduleImpl(TilingKey key) override { return success(); }
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SHALLOWCVSCHEDULE_H

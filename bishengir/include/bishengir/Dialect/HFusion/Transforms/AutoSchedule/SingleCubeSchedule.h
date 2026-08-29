//===- SingleCubeSchedule.h -- Schedule for Single Cube Op ------*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SINGLECUBESCHEDULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_SINGLECUBESCHEDULE_H

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

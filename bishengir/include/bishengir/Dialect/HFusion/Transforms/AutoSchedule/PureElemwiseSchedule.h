//===- PureElemwiseSchedule.h -- Schedule for PureElemwise Op ---*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_PUREELEMWISESCHEDULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_PUREELEMWISESCHEDULE_H

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
class Location;
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
  LogicalResult createScheduleImpl(TilingKey key) override;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_PUREELEMWISESCHEDULE_H

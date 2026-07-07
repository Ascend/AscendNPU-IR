//===- AnyPBRKernelInfoCollector.h -- AnyPBR Kernel Info Collector --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBRKERNELINFOCOLLECTOR_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBRKERNELINFOCOLLECTOR_H

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfoCollector.h"

namespace mlir {
namespace hfusion {

class AnyPBRKernelInfoCollector final : public KernelInfoCollector {
public:
  explicit AnyPBRKernelInfoCollector(KernelInfo *info,
                                     AutoScheduleOptions options)
      : KernelInfoCollector(info, std::move(options)) {}

private:
  LogicalResult visitLinalgOpImpl(Operation *op) override;
  LogicalResult postVisitFuncImpl(func::FuncOp f) override;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBRKERNELINFOCOLLECTOR_H

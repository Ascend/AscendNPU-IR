//===- MSCostModel.h - modulo-schedule linear cost model --------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
//
// The MS (Modulo-Schedule) cost model: per-module linear formulas
// y = a·x + b read exclusively from MSCostParams.h.  This class contains
// no numeric literals, never touches the DDG or MLIR IR (VectorOpDesc::op
// is only read for its op name in notes/diagnostics), and never emits
// diagnostics itself — errors surface as CostResult{-1, note} and the
// caller prints them.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_MSCOSTMODEL_H
#define BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_MSCOSTMODEL_H

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/CostModel.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/CostModelTypes.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/MSCostParams.h"

namespace mlir::hivm::costmodel {

class MSCostModel : public CostModel {
public:
  MSCostModel() = default;

  /// Cube: x = sm·sn·sk taken from the first two fractal dimensions of
  /// the two inputs; params dispatched on dtype class (f16/bf16/i8 vs
  /// f32).  Shape/dtype anomalies → CostResult{-1, note}.
  CostResult getCubeCost(const CubeCostInput &input) const override;

  /// Vector group: per-op x = max element count of the op's inputs,
  /// summed over ops then multiplied by the VF-factor.  Ops missing from
  /// the multiplier tables count via the fallback multipliers and land in
  /// the note; input shape anomalies fail the whole group with
  /// CostResult{-1, note}.
  CostResult
  getVectorGroupCost(const VectorGroupCostInput &input) const override;

  /// Transfer: x = byte size of the moved block; params dispatched on the
  /// (pipeline, storageTag) pair.  Unknown pair or bad shape/dtype →
  /// CostResult{-1, note}.
  CostResult getTransferCost(const TransferCostInput &input) const override;
};

} // namespace mlir::hivm::costmodel

#endif // BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_MSCOSTMODEL_H

//===- CostModel.h - abstract interface for SMS cost models ------*- C++
//-*-===//
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
// Abstract interface between MsDagBuilder and the cost models.  It exists
// so unit tests can inject mock models; the production path instantiates
// MSCostModel directly.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_COSTMODEL_H
#define BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_COSTMODEL_H

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/CostModelTypes.h"

namespace mlir::hivm::costmodel {

class CostModel {
public:
  virtual ~CostModel() = default;

  /// None of the three queries returns FailureOr: when a cost cannot be
  /// computed they return CostResult{costTime = -1, note = specifics} (see
  /// CostModelTypes.h); the caller checks costTime == -1, prints the note,
  /// and fails the build.  The only anomaly that does NOT go through the
  /// -1 sentinel is a vector op outside the accept list: the result is a
  /// normal positive value whose note records the op names so the caller
  /// can emit a warning (§4.4.2 of the design doc).

  /// Cube compute cost: dispatch on dtype class, x = sm·sn·sk.
  virtual CostResult getCubeCost(const CubeCostInput &input) const = 0;

  /// Vector-group cost: sum the per-op latencies then multiply by the
  /// VF-factor.  Ops outside the multiplier tables: the note records the
  /// op names (the caller emits a warning) and the latency still counts
  /// via the fallback multipliers — the group does not fail.
  virtual CostResult
  getVectorGroupCost(const VectorGroupCostInput &input) const = 0;

  /// Transfer (MTE/fixpipe) cost: x = the data block's byte size,
  /// dispatched on (pipeline, direction).
  virtual CostResult getTransferCost(const TransferCostInput &input) const = 0;
};

} // namespace mlir::hivm::costmodel

#endif // BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_COSTMODEL_H

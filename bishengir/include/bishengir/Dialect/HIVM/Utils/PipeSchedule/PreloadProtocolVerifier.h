//===- PreloadProtocolVerifier.h - MS preload protocol verification *- C++
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
//
// Read-only gate between MsPartitioner and scope materialization.
//
// Validates that an MS partition can be expressed by the current CV preload
// lowering (CreatePreload, MarkMultiBuffer, PlanMemory, GSS).  The verifier
// never mutates IR, creates scopes, or invokes WorklistBuilder.
//
// Four verification categories (§4.6 of design doc):
//   1. Coordinate & depth — pipelineDepth, coordinate set, stage grouping
//   2. Ownership & ordering — single-owner, block order, DDG forward
//   3. Carrier & memory — cross-WorkItem values have carriers
//   4. Lowering expressibility — P0 stage-level recurrence contract
//
// Usage:
//   PreloadProtocolVerifier(loop, dagResult, partition).verify()
//
// Returns success() when the partition is safe to install; emits diagnostics
// on the offending operation/loop and returns failure() otherwise.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PRELOADPROTOCOLVERIFIER_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PRELOADPROTOCOLVERIFIER_H

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsPartitioner.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace hivm {

class PreloadProtocolVerifier {
public:
  PreloadProtocolVerifier(scf::ForOp loop, const MsDagBuildResult &dagResult,
                          const MsPartitionResult &partition);

  LogicalResult verify();

private:
  LogicalResult verifyCoordinatesAndDepth();
  LogicalResult verifyOwnershipAndOrdering();
  LogicalResult verifyCarriersAndMemory();
  LogicalResult verifyLoweringExpressibility();

  scf::ForOp pipelineLoop;
  const MsDagBuildResult &dagResult;
  const MsPartitionResult &partition;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PRELOADPROTOCOLVERIFIER_H

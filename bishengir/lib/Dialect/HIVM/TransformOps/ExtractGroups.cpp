//===- ExtractGroups.cpp - Extract selected groups from a plan ------------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace {
FailureOr<PlanAttr> resolvePlan(Operation *user, ArrayRef<Attribute> params) {
  if (params.size() != 1)
    return user->emitOpError() << "expected exactly one plan attribute";
  auto plan = dyn_cast<PlanAttr>(params.front());
  if (!plan)
    return user->emitOpError() << "expected a #hivm_vectorize.plan attribute";
  return plan;
}

} // namespace

DiagnosedSilenceableFailure
HIVMExtractGroupsOp::apply(TransformRewriter &, TransformResults &results,
                           TransformState &state) {
  auto plan = resolvePlan(getOperation(), state.getParams(getPlan()));
  if (failed(plan))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.setParams(cast<OpResult>(getGroups()), {plan->getGroups()});
  return DiagnosedSilenceableFailure::success();
}

void HIVMExtractGroupsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getPlanMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

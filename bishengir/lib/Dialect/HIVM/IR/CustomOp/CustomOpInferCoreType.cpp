//===- CustomOpInferCoreType.cpp - Custom op core type inference ----------===//
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

#include "bishengir/Dialect/HIVM/IR/CustomOp/DistributedTransformUtils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;
using namespace mlir::hivm;

namespace {
template <typename CustomOpT>
std::optional<TCoreType> inferCustomOpCoreTypeFromAttr(CustomOpT op) {
  if (auto coreTypeAttr =
          static_cast<Operation *>(op)->template getAttrOfType<TCoreTypeAttr>(
              TCoreTypeAttr::name)) {
    return coreTypeAttr.getTcoretype();
  }
  return {};
}

TCoreType getPreCoreTypeForNotifyOp(Operation *notifyOp) {
  TCoreType preCoreType = TCoreType::VECTOR;
  func::FuncOp funcOp = notifyOp->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return preCoreType;

  funcOp.walk([&](Operation *op) {
    if (op == notifyOp)
      return WalkResult::interrupt();
    if (auto customOp = dyn_cast<hivm::CustomOp>(op)) {
      if (isShmemNotifyCustomOp(customOp))
        return WalkResult::advance();
    }
    std::optional<TCoreType> tmpCoreType =
        hivm::detail::queryCoreTypeHelper(op);
    if (tmpCoreType != std::nullopt &&
        tmpCoreType != TCoreType::CUBE_AND_VECTOR &&
        tmpCoreType != TCoreType::CUBE_OR_VECTOR) {
      preCoreType = tmpCoreType.value();
    }
    return WalkResult::advance();
  });
  return preCoreType;
}
} // namespace

std::optional<TCoreType> CustomOp::inferCoreType() {
  if (isShmemNotifyCustomOp(*this)) {
    TCoreType preCoreType = getPreCoreTypeForNotifyOp(getOperation());
    if (preCoreType == TCoreType::CUBE_OR_VECTOR)
      return TCoreType::VECTOR;
    return preCoreType;
  }
  return inferCustomOpCoreTypeFromAttr(*this);
}

std::optional<TCoreType> CustomMacroOp::inferCoreType() {
  return inferCustomOpCoreTypeFromAttr(*this);
}

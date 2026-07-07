//===- AnnotationHelpers.cpp ------------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"

namespace mlir::hfusion {

template <typename OpType>
std::optional<StringRef> getAnnotateOverflowMode(OpType op) {
  std::optional<Operation *> overflowMode =
      utils::getAnnotateOpWithAttr(op.getResult(0), "overflow_mode");
  if (!overflowMode.has_value()) {
    return std::nullopt;
  }
  StringAttr overflowAttrVal =
      overflowMode.value()->getAttrOfType<StringAttr>("overflow_mode");
  return overflowAttrVal.getValue();
}

// Explicit template instantiations
template std::optional<StringRef> mlir::hfusion::getAnnotateOverflowMode<CompareOp>(CompareOp);
template std::optional<StringRef> mlir::hfusion::getAnnotateOverflowMode<CastOp>(CastOp);

} // namespace mlir::hfusion

//===- CustomOpStructured.cpp - Custom op structured op interface ---------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;
using namespace mlir::hivm;

namespace {
template <typename CustomOpT> ArrayAttr getCustomOpIndexingMaps(CustomOpT op) {
  if (auto attr = op.getOperation()->template getAttrOfType<ArrayAttr>(
          CustomOpT::kIndexingMapName))
    return attr;
  return ArrayAttr::get(op.getContext(), {});
}

template <typename CustomOpT>
SmallVector<hivm::IteratorType> getCustomOpIteratorTypesArray(CustomOpT op) {
  if (!op.getOperation()->hasAttr(CustomOpT::kIteratorTypesName))
    return {};

  return llvm::to_vector(
      llvm::map_range(op.getOperation()->template getAttrOfType<ArrayAttr>(
                          CustomOpT::kIteratorTypesName),
                      [](Attribute attr) {
                        return cast<hivm::IteratorTypeAttr>(attr).getValue();
                      }));
}
} // namespace

ArrayAttr CustomOp::getIndexingMaps() { return getCustomOpIndexingMaps(*this); }

SmallVector<hivm::IteratorType> CustomOp::getIteratorTypesArray() {
  return getCustomOpIteratorTypesArray(*this);
}

ArrayAttr CustomMacroOp::getIndexingMaps() {
  return getCustomOpIndexingMaps(*this);
}

SmallVector<hivm::IteratorType> CustomMacroOp::getIteratorTypesArray() {
  return getCustomOpIteratorTypesArray(*this);
}

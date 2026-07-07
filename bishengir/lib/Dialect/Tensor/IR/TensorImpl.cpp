//===- TensorImpl.cpp -----------------------------------------------------===//
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

#include "bishengir/Config/bishengir-config.h"

#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace mlir {
namespace tensor {

Value createTensorEmptyOpWithTargetElemType(OpBuilder &builder, Location loc,
                                            Value source, Type targetElemType) {
  auto shapedType = cast<ShapedType>(source.getType());
  ArrayRef<int64_t> staticShapes = shapedType.getShape();
  llvm::SmallVector<Value, 2> dynamicSizes;
  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic) {
      Operation *dynDimOp = builder.create<tensor::DimOp>(loc, source, i);
      dynamicSizes.push_back(dynDimOp->getResults()[0]);
    }
  }
  return builder.create<tensor::EmptyOp>(loc, staticShapes, targetElemType,
                                         dynamicSizes);
}

Value createTensorEmptyOp(OpBuilder &builder, Location loc, Value source) {
  auto elementType = getElementTypeOrSelf(source);
  auto emptyOp =
      createTensorEmptyOpWithTargetElemType(builder, loc, source, elementType);
  return emptyOp;
}

} // namespace tensor
} // namespace mlir

#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY

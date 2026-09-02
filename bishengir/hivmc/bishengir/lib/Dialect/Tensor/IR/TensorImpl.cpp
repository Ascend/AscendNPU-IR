//===- TensorImpl.cpp -----------------------------------------------------===//
//
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

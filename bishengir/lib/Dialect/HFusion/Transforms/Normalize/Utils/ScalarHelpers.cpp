//===- ScalarHelpers.cpp --------------------------------------------*- C++ -*-===//
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

/// Convert dense tensor/memref with only 1 element to scalar.
std::optional<Value>
getScalarFromConstantOp(PatternRewriter &rewriter, Location loc,
                        arith::ConstantOp constant) {
  auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(constant.getValue());
  if (!denseAttr) {
    return std::nullopt;
  }

  auto elemType = denseAttr.getElementType();
  if (!elemType.isIntOrIndexOrFloat()) {
    return std::nullopt;
  }

  TypedAttr typedAttr =
      elemType.isIntOrIndex()
          ? (TypedAttr)*denseAttr.getValues<IntegerAttr>().begin()
          : (TypedAttr)*denseAttr.getValues<FloatAttr>().begin();

  return rewriter.create<arith::ConstantOp>(loc, elemType, typedAttr);
}

/// Convert dense tensor/memref with only 1 element to scalar.
std::optional<Value>
singleElemDenseTensorToScalar(Value operand, PatternRewriter &rewriter) {
  auto constantOp = operand.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return std::nullopt;

  auto shapedType = dyn_cast<ShapedType>(constantOp.getType());
  if (!shapedType)
    return std::nullopt;

  auto shape = shapedType.getShape();
  if (shape.size() > 1 || (!shape.empty() && shape[0] > 1))
    return std::nullopt;

  return getScalarFromConstantOp(rewriter, operand.getLoc(), constantOp);
}

} // namespace mlir::hfusion

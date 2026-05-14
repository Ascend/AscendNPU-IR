//===- ScalarTemplateHelpers.h ---------------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_SCALARTEMPLATEHELPERS_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_SCALARTEMPLATEHELPERS_H

#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {

/// Convert dense tensor/memref with only 1 element to scalar.
inline std::optional<Value> getScalarFromConstantOp(PatternRewriter &rewriter,
                                                    Location loc,
                                                    arith::ConstantOp constant) {
  auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(constant.getValue());
  if (!denseAttr)
    return std::nullopt;

  auto elemType = denseAttr.getElementType();
  if (!elemType.isIntOrIndexOrFloat())
    return std::nullopt;

  TypedAttr typedAttr =
      elemType.isIntOrIndex()
          ? static_cast<TypedAttr>(*denseAttr.getValues<IntegerAttr>().begin())
          : static_cast<TypedAttr>(*denseAttr.getValues<FloatAttr>().begin());

  return rewriter.create<arith::ConstantOp>(loc, elemType, typedAttr);
}

/// Convert dense tensor/memref with only 1 element to scalar.
inline std::optional<Value>
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

/// Convert a dense constant tensor with exactly one static element to a scalar.
///
/// This is broader than singleElemDenseTensorToScalar: it also accepts
/// multi-rank unit tensors such as tensor<1x1xf32>. That shape appears on the
/// HIVM path when a scalar-like broadcast source has already been rank-expanded
/// during HFusion-to-HIVM conversion.
inline std::optional<Value> unitDenseTensorToScalar(Value operand,
                                                    PatternRewriter &rewriter) {
  auto constantOp = operand.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return std::nullopt;

  auto shapedType = dyn_cast<ShapedType>(constantOp.getType());
  if (!shapedType || !shapedType.hasStaticShape() ||
      shapedType.getNumElements() != 1)
    return std::nullopt;

  return getScalarFromConstantOp(rewriter, operand.getLoc(), constantOp);
}

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_SCALARTEMPLATEHELPERS_H

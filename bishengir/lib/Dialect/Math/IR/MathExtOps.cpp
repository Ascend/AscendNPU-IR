//===- MathExtOps.cpp - Implementation of Math Ext dialect and types ------===//
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

#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

using namespace mlir;
using namespace mlir::mathExt;
using namespace mlir::arith;

#define GET_OP_CLASSES
#include "bishengir/Dialect/MathExt/IR/MathExtOps.cpp.inc"

//===----------------------------------------------------------------------===//
// IlogbOp folder
//===----------------------------------------------------------------------===//
OpFoldResult mathExt::IlogbOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOpConditional<FloatAttr>(
      adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
        if (a.isNegative())
          return {};

        if (a.getSizeInBits(a.getSemantics()) == 64)
          return APFloat(log(a.convertToDouble()));

        if (a.getSizeInBits(a.getSemantics()) == 32)
          return APFloat(logf(a.convertToFloat()));

        return {};
      });
}

//===----------------------------------------------------------------------===//
// LdexpOp folder
//===----------------------------------------------------------------------===//
OpFoldResult mathExt::LdexpOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOpConditional<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
        if (a.getSizeInBits(a.getSemantics()) == 64 &&
            b.getSizeInBits(b.getSemantics()) == 64)
          return APFloat(ldexp(a.convertToDouble(),
                               static_cast<int>(b.convertToDouble())));

        if (a.getSizeInBits(a.getSemantics()) == 32 &&
            b.getSizeInBits(b.getSemantics()) == 32)
          return APFloat(
              ldexpf(a.convertToFloat(), static_cast<int>(b.convertToFloat())));

        return {};
      });
}

//===----------------------------------------------------------------------===//
// DivFHPOp folder
//===----------------------------------------------------------------------===//
OpFoldResult mathExt::DivFHPOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOpConditional<FloatAttr>(
      adaptor.getOperands(),
      /*calculate=*/
      [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
        APFloat out = a; // IMPORTANT: make a mutable copy
        out.divide(b, APFloat::rmNearestTiesToEven); // divide mutates 'out'
        return out;
      });
}
/// Materialize an integer or floating point constant.
Operation *mathExt::MathExtDialect::materializeConstant(OpBuilder &builder,
                                                        Attribute value,
                                                        Type type,
                                                        Location loc) {
  if (auto poison = dyn_cast<ub::PoisonAttr>(value))
    return builder.create<ub::PoisonOp>(loc, type, poison);

  return arith::ConstantOp::materialize(builder, value, type, loc);
}

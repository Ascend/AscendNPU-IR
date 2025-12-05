//===- HFusionImpl.h - HFusion implementation -------------------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONIMPL_H
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONIMPL_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace mlir {
namespace hfusion {

template <typename BnaryOp, typename OpFun, typename OpFunAttr>
Operation *createBinaryOp(OpBuilder &builder, Location loc, OpFun opFn,
                          ValueRange inputs, ValueRange out) {
  auto attr = builder.getAttr<OpFunAttr>(opFn);
  auto fnAttr = builder.getNamedAttr("fun", attr);
  return builder.create<BnaryOp>(loc, inputs, out, fnAttr);
}

template <typename UnaryOp, typename OpFun, typename OpFunAttr>
Operation *createUnaryOp(OpBuilder &builder, Location loc, OpFun opFn,
                         ValueRange inputs, ValueRange outs) {
  auto attr = builder.getAttr<OpFunAttr>(opFn);
  auto fnAttr = builder.getNamedAttr("fun", attr);
  return builder.create<UnaryOp>(loc, inputs, outs, fnAttr);
}

/// Cast `src` value to the specified element type and rounding mode.
///
/// `src` can be either tensor or scalar.
/// If it's a scalar, casting is done by arith dialect ops.
/// If it's a tensor, casting is done by `hfusion.cast` op. If `dst` is not
/// provided, the init value is a `tensor.empty` op. Otherwise, it's written
/// to `dst`.
Value castTo(OpBuilder &builder, Value src, Type targetElemType,
  hfusion::RoundMode roundMode,
  std::optional<Value> dst = std::nullopt,
  bool enableOverflow = true,
  hfusion::TypeFn castIntegerType = hfusion::TypeFn::cast_signed);

/// Cast `src` value to the specified element type.
/// Select rounding mode inside.
Value castTo(OpBuilder &builder, Value src, Type targetElemType,
  hfusion::TypeFn castIntegerType = hfusion::TypeFn::cast_signed);

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONIMPL_H

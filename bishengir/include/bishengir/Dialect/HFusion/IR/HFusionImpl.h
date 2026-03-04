//===- HFusionImpl.h - HFusion implementation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONIMPL_H

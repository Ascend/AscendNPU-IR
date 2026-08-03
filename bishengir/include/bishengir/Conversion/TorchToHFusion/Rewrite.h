//===- Rewrite.h - Helper rewriters for Torch To HFusion --------*- C++ -*-===//
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
#ifndef BISHENGIR_CONVERSION_TORCHTOHFUSION_REWTRITE_H
#define BISHENGIR_CONVERSION_TORCHTOHFUSION_REWTRITE_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include <type_traits>

namespace mlir {
template <linalg::BinaryFn linalgFn>
static Value createLinalgBinary(OpBuilder &builder, Location loc, Value lhs,
                                Value rhs, Value out) {
  auto attr = builder.getAttr<linalg::BinaryFnAttr>(linalgFn);
  auto fnAttr = builder.getNamedAttr("fun", attr);
  auto newOp = builder.create<linalg::ElemwiseBinaryOp>(
      loc, ValueRange{lhs, rhs}, ValueRange{out}, fnAttr);
  return newOp->getResult(0);
}

template <hfusion::BinaryFn hfusionFn>
static Value createHFusionBinary(OpBuilder &builder, Location loc, Value lhs,
                                 Value rhs, Value out) {
  auto attr = builder.getAttr<hfusion::BinaryFnAttr>(hfusionFn);
  auto fnAttr = builder.getNamedAttr("fun", attr);
  auto newOp = builder.create<hfusion::ElemwiseBinaryOp>(
      loc, ValueRange{lhs, rhs}, ValueRange{out}, fnAttr);
  return newOp->getResult(0);
}

template <linalg::UnaryFn linalgFn>
static Value createLinalgUnary(OpBuilder &builder, Location loc, Value input,
                               Value out) {
  auto attr = builder.getAttr<linalg::UnaryFnAttr>(linalgFn);
  auto fnAttr = builder.getNamedAttr("fun", attr);
  auto newOp = builder.create<linalg::ElemwiseUnaryOp>(loc, ValueRange{input},
                                                       ValueRange{out}, fnAttr);
  return newOp->getResult(0);
}

template <hfusion::UnaryFn hfusionFn>
static Value createHFusionUnary(OpBuilder &builder, Location loc, Value input,
                                Value out) {
  auto attr = builder.getAttr<hfusion::UnaryFnAttr>(hfusionFn);
  auto fnAttr = builder.getNamedAttr("fun", attr);
  auto newOp = builder.create<hfusion::ElemwiseUnaryOp>(
      loc, ValueRange{input}, ValueRange{out}, fnAttr);
  return newOp->getResult(0);
}

template <hfusion::CompareFn CompareFnTy>
static Value createHFusionCompare(PatternRewriter &rewriter, Location loc,
                                  Value lhs, Value rhs, Value out) {
  auto attr = rewriter.getAttr<hfusion::CompareFnAttr>(CompareFnTy);
  auto fnAttr =
      rewriter.getNamedAttr(hfusion::CompareFnAttr::getMnemonic(), attr);
  auto newOp = rewriter.create<hfusion::CompareOp>(loc, ValueRange{lhs, rhs},
                                                   ValueRange{out}, fnAttr);
  return newOp->getResult(0);
}
} // namespace mlir

#endif // BISHENGIR_CONVERSION_TORCHTOHFUSION_REWTRITE_H
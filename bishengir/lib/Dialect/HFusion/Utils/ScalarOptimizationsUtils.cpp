//===- ScalarOptimizationsUtils.cpp - Scalar optimization utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Utils/ScalarOptimizationsUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace mlir::hfusion;

namespace mlir {
namespace hfusion {
namespace detail {

bool isSingletonTensorType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 1 &&
         tensorType.getDimSize(0) == 1 &&
         tensorType.getElementType().isIntOrFloat();
}

bool isZeroRankTensorType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 0 &&
         tensorType.getElementType().isIntOrFloat();
}

bool isVectorToScalarTransfer(tensor::ExtractOp extractOp) {
  Type type = extractOp.getTensor().getType();
  return isZeroRankTensorType(type) || isSingletonTensorType(type);
}

bool isVectorToScalarTransfer(Operation *op) {
  auto extractOp = dyn_cast<tensor::ExtractOp>(op);
  return extractOp && isVectorToScalarTransfer(extractOp);
}

bool isScalarInsert(tensor::InsertOp insertOp) {
  return isSingletonTensorType(insertOp.getResult().getType());
}

bool isScalarInsert(Operation *op) {
  auto insertOp = dyn_cast<tensor::InsertOp>(op);
  return insertOp && isScalarInsert(insertOp);
}

bool isStaticRankOneFill(Operation *op) {
  auto fillOp = dyn_cast<linalg::FillOp>(op);
  if (!fillOp)
    return false;

  auto type = dyn_cast<RankedTensorType>(fillOp.output().getType());
  return type && type.getRank() == 1 &&
         !ShapedType::isDynamic(type.getDimSize(0));
}

bool isScalarToVectorTransfer(Operation *op) {
  if (auto insertOp = dyn_cast<tensor::InsertOp>(op))
    return isScalarInsert(insertOp);

  return isStaticRankOneFill(op);
}

} // namespace detail
} // namespace hfusion
} // namespace mlir

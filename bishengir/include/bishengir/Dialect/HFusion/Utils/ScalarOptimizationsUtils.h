//===- ScalarOptimizationsUtils.h - Scalar optimization utilities ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_UTILS_SCALAROPTIMIZATIONSUTILS_H
#define BISHENGIR_DIALECT_HFUSION_UTILS_SCALAROPTIMIZATIONSUTILS_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace hfusion {
namespace detail {

bool isSingletonTensorType(Type type);
bool isZeroRankTensorType(Type type);

bool isVectorToScalarTransfer(tensor::ExtractOp extractOp);
bool isVectorToScalarTransfer(Operation *op);

bool isScalarInsert(tensor::InsertOp insertOp);
bool isScalarInsert(Operation *op);

bool isStaticRankOneFill(Operation *op);
bool isScalarToVectorTransfer(Operation *op);

} // namespace detail
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_UTILS_SCALAROPTIMIZATIONSUTILS_H

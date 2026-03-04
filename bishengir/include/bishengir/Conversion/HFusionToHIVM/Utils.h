//===- Utils.h - HFusion to HIVM Conversion Utilities ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HFUSIONTOHIVM_UTILS_H
#define BISHENGIR_CONVERSION_HFUSIONTOHIVM_UTILS_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace hfusion_conversion_utils {

/// Use ins shapedType of linalg op to expand.
/// Insert 1 into shape in axis of dimsArr, and update strides if stridedLayout
/// exits.
///
/// \param shapedType ins shaped type of linalg ops
Type getExpandShapeOpResType(ShapedType shapedType, ArrayRef<int64_t> dimsArr);

template <typename ConcreteOp>
std::enable_if_t<
    std::is_same<ConcreteOp, linalg::ReduceOp>::value ||
        std::is_same<ConcreteOp, linalg::BroadcastOp>::value ||
        std::is_same<ConcreteOp, hfusion::ReduceWithIndexOp>::value,
    Value>
createExpandShapeOp(ConcreteOp op, PatternRewriter &rewriter, Value expandSrc,
                    ShapedType targetType) {
  auto dims = op.getDimensions();
  ShapedType shapedType = cast<ShapedType>(expandSrc.getType());
  Type expandShapeOpResTy = getExpandShapeOpResType(shapedType, dims);
  int64_t outRank = cast<ShapedType>(expandSrc.getType()).getRank();
  if (outRank != 0) {
    // This is not a rank-0 case. Thus turn to the normal case.
    outRank = targetType.getRank();
  }
  auto reassociation = reshape_utils::getReAssociation(dims, outRank);

  const bool hasPureBuffer = op.hasPureBufferSemantics();
  assert(expandSrc && "expandSrc shouldn't null.");
  Value expandShapeOp =
      hasPureBuffer
          ? (Value)rewriter.create<memref::ExpandShapeOp>(
                op.getLoc(), expandShapeOpResTy, expandSrc, reassociation)
          : (Value)rewriter.create<tensor::ExpandShapeOp>(
                op.getLoc(), expandShapeOpResTy, expandSrc, reassociation);
  return expandShapeOp;
}

Value createCollapseShapeOp(PatternRewriter &rewriter, Location loc,
                            Value collapseSrc, Type resultType,
                            SmallVector<SmallVector<int64_t, 2>> collapseDims,
                            bool isPureTensor);

hivm::RoundMode mapRoundModeHFusionToHiVM(hfusion::RoundMode hsRndMode);

} // namespace hfusion_conversion_utils
} // namespace mlir

#endif // BISHENGIR_CONVERSION_HFUSIONTOHIVM_UTILS_H

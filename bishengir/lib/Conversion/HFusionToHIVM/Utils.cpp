//===- Utils.h - HFusion to HIVM Utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HFusionToHIVM/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <set>

namespace mlir {
namespace hfusion_conversion_utils {

Type getExpandShapeOpResType(ShapedType shapedType, ArrayRef<int64_t> dimsArr) {
  SmallVector<int64_t> shape(shapedType.getShape()); // static shape
  for (size_t i = 0; i < dimsArr.size(); ++i) {
    shape.insert(shape.begin() + dimsArr[i], 1);
  }

  const bool isTensor = isa<TensorType>(shapedType);
  if (isTensor) {
    return shapedType.clone(shape);
  }

  auto mem = cast<MemRefType>(shapedType);
  auto stridedLayout = dyn_cast<StridedLayoutAttr>(mem.getLayout());
  if (!stridedLayout) {
    StridedLayoutAttr layout = {};
    return MemRefType::get(shape, mem.getElementType(), layout,
                           mem.getMemorySpace());
  }

  SmallVector<int64_t> strides(stridedLayout.getStrides());
  const int64_t offset = stridedLayout.getOffset();

  for (size_t i = 0; i < dimsArr.size(); ++i) {
    long strideVal;
    if (dimsArr[i] == 0) {
      if (mem.getShape()[0] != ShapedType::kDynamic &&
          strides[0] != ShapedType::kDynamic) {
        strideVal = mem.getShape()[0] * strides[0];
      } else {
        strideVal = ShapedType::kDynamic;
      }
    } else {
      if (static_cast<size_t>(dimsArr[i]) >= strides.size() + 1)
        llvm_unreachable("strides accessed index out-of-bounds");
      strideVal = strides[dimsArr[i] - 1];
    }
    strides.insert(strides.begin() + dimsArr[i], strideVal);
  }

  auto newLayout = StridedLayoutAttr::get(mem.getContext(), offset, strides);
  return MemRefType::get(shape, mem.getElementType(), newLayout,
                         mem.getMemorySpace());
}

Value createCollapseShapeOp(PatternRewriter &rewriter, Location loc,
                            Value collapseSrc, Type resultType,
                            SmallVector<SmallVector<int64_t, 2>> collapseDims,
                            bool isPureTensor) {
  return isPureTensor ? (Value)rewriter.create<tensor::CollapseShapeOp>(
                            loc, resultType, collapseSrc, collapseDims)
                      : (Value)rewriter.create<memref::CollapseShapeOp>(
                            loc, resultType, collapseSrc, collapseDims);
}

hivm::RoundMode mapRoundModeHFusionToHiVM(hfusion::RoundMode hsRndMode) {
  switch (hsRndMode) {
  case (hfusion::RoundMode::RINT):
    return hivm::RoundMode::RINT;
  case (hfusion::RoundMode::ROUND):
    return hivm::RoundMode::ROUND;
  case (hfusion::RoundMode::CEIL):
    return hivm::RoundMode::CEIL;
  case (hfusion::RoundMode::FLOOR):
    return hivm::RoundMode::FLOOR;
  case (hfusion::RoundMode::TRUNC):
    return hivm::RoundMode::TRUNC;
  case (hfusion::RoundMode::ODD):
    return hivm::RoundMode::ODD;
  case (hfusion::RoundMode::TRUNCWITHOVERFLOW):
    return hivm::RoundMode::TRUNCWITHOVERFLOW;
  }
  llvm_unreachable("unsupported hfusion::RoundMode");
}

} // namespace hfusion_conversion_utils
} // namespace mlir

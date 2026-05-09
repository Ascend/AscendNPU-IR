//===- Utils.h - HFusion to HIVM Utilities ----------------------*- C++ -*-===//
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
        llvm::report_fatal_error("strides accessed index out-of-bounds");
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

} // namespace hfusion_conversion_utils
} // namespace mlir

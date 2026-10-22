//===- ComputeOffset.cpp - linear memref offset for AVE ops ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// computeLinearMemRefOffset lives here (in the IR library) rather than in
// Utils.cpp so the AVE dialect can build in standalone IR-only mode, where the
// full HIVMAVE Utils library (which depends on HIVMRegbaseIntrins) is not
// built. The function only needs MLIR core + Arith/MemRef dialects.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

Value hivmave::computeLinearMemRefOffset(OpBuilder &builder, Location loc,
                                         Value memref, ValueRange indices,
                                         Type resultType) {
  assert((resultType.isIndex() || isa<IntegerType>(resultType)) &&
         "result type must be index or integer");

  auto memrefType = cast<MemRefType>(memref.getType());
  int64_t rank = memrefType.getRank();

  auto makeConst = [&](int64_t value) -> Value {
    if (resultType.isIndex())
      return builder.create<arith::ConstantIndexOp>(loc, value);
    return builder.create<arith::ConstantOp>(
        loc, IntegerAttr::get(resultType, value));
  };
  auto castToResultType = [&](Value value) -> Value {
    if (value.getType() == resultType)
      return value;
    return builder.create<arith::IndexCastOp>(loc, resultType, value);
  };

  SmallVector<Value, 4> strides(rank);
  if (auto stridedLayout =
          dyn_cast<StridedLayoutAttr>(memrefType.getLayout())) {
    ArrayRef<int64_t> layoutStrides = stridedLayout.getStrides();
    std::optional<memref::ExtractStridedMetadataOp> metadata;
    for (int64_t i = 0; i < rank; ++i) {
      if (!ShapedType::isDynamic(layoutStrides[i])) {
        strides[i] = makeConst(layoutStrides[i]);
        continue;
      }

      if (!metadata)
        metadata =
            builder.create<memref::ExtractStridedMetadataOp>(loc, memref);
      strides[i] = castToResultType(metadata->getStrides()[i]);
    }
  } else {
    Value strideValue = makeConst(1);
    for (int64_t i = rank - 1; i >= 0; --i) {
      strides[i] = strideValue;
      if (i == 0)
        continue;

      Value dimSize;
      if (memrefType.isDynamicDim(i)) {
        dimSize = builder.create<memref::DimOp>(loc, memref, i);
        dimSize = castToResultType(dimSize);
      } else {
        dimSize = makeConst(memrefType.getDimSize(i));
      }
      strideValue = builder.create<arith::MulIOp>(loc, strideValue, dimSize);
    }
  }

  Value linearOffset = makeConst(0);
  for (auto [idx, index] : llvm::enumerate(indices)) {
    Value indexValue = castToResultType(index);
    Value product =
        builder.create<arith::MulIOp>(loc, indexValue, strides[idx]);
    linearOffset = builder.create<arith::AddIOp>(loc, linearOffset, product);
  }
  return linearOffset;
}

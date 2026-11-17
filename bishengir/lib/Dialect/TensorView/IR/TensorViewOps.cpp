//===- TensorViewOps.cpp - TensorView operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------===//
// TensorViewEncodingOpInterface
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/TensorView/IR/TensorViewInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/TensorView/IR/TensorViewOps.cpp.inc"

namespace mlir::tv {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

// Verify that `res` matches `src` in everything but the encoding, and that
// `src` itself is a base (encoding-free) view.
static LogicalResult verifyEncodingParent(Operation *op, TensorViewType src,
                                          TensorViewType res) {
  if (src.getEncoding())
    return op->emitOpError("source must be a base view (no encoding)");
  if (src.getShape() != res.getShape() ||
      src.getStrides() != res.getStrides() ||
      src.getElementType() != res.getElementType())
    return op->emitOpError("result must match source except for the encoding");
  if (getEncodingPaddingValue(res.getEncoding()) != PaddingValue::ZERO &&
      !isa<FloatType>(res.getElementType()))
    return op->emitOpError(
        "nan and infinity padding require a floating-point element type");
  return success();
}

//===----------------------------------------------------------------------===//
// make_tensor_view
//===----------------------------------------------------------------------===//

LogicalResult MakeTensorViewOp::verify() {
  auto resTy = getResult().getType();
  if (resTy.getEncoding())
    return emitOpError("result must be a base view (no encoding)");
  int64_t rank = resTy.getRank();
  if (static_cast<int64_t>(getSizes().size()) != rank)
    return emitOpError("expected ")
           << rank << " size operands but got " << getSizes().size();
  if (static_cast<int64_t>(getStrides().size()) != rank)
    return emitOpError("expected ")
           << rank << " stride operands but got " << getStrides().size();
  for (auto [dim, values] :
       llvm::enumerate(llvm::zip(getSizes(), getStrides()))) {
    auto [size, stride] = values;
    auto extent = getConstantIntValue(size);
    if (extent && *extent < 0)
      return emitOpError("logical sizes must be nonnegative");
    if (!ShapedType::isDynamic(resTy.getShape()[dim]) &&
        extent != resTy.getShape()[dim])
      return emitOpError("static shape must match the size operands");
    if (!ShapedType::isDynamic(resTy.getStrides()[dim]) &&
        getConstantIntValue(stride) != resTy.getStrides()[dim])
      return emitOpError("static strides must match the stride operands");
  }
  Type srcPointee;
  if (auto tvPtrTy = dyn_cast<PtrType>(getSource().getType()))
    srcPointee = tvPtrTy.getPointeeType();
  else if (auto memref = dyn_cast<MemRefType>(getSource().getType()))
    srcPointee = memref.getElementType();
  else
    srcPointee = resTy.getElementType();
  if (resTy.getElementType() != srcPointee)
    return emitOpError("result element type must match source pointee type");
  return success();
}

//===----------------------------------------------------------------------===//
// make_partition_view / make_strided_view / make_gather_scatter_view
//===----------------------------------------------------------------------===//

LogicalResult MakePartitionViewOp::verify() {
  auto src = getSource().getType();
  auto res = getResult().getType();
  if (failed(verifyEncodingParent(*this, src, res)))
    return failure();
  auto enc = dyn_cast_or_null<PartitionViewAttr>(res.getEncoding());
  if (!enc)
    return emitOpError("result must carry a #tv.partition_view encoding");
  int64_t rank = res.getRank();
  if (static_cast<int64_t>(enc.getTile().size()) != rank)
    return emitOpError("partition tile rank must match the view rank");
  for (auto [dim, mapped] : llvm::enumerate(enc.getDimMap()))
    if (mapped != static_cast<int64_t>(dim))
      return emitOpError("only identity dim_map is supported");
  return success();
}

LogicalResult MakeStridedViewOp::verify() {
  auto src = getSource().getType();
  auto res = getResult().getType();
  if (failed(verifyEncodingParent(*this, src, res)))
    return failure();
  auto enc = dyn_cast_or_null<StridedViewAttr>(res.getEncoding());
  if (!enc)
    return emitOpError("result must carry a #tv.strided_view encoding");
  int64_t rank = res.getRank();
  if (static_cast<int64_t>(enc.getTile().size()) != rank)
    return emitOpError("strided tile rank must match the view rank");
  for (auto [dim, mapped] : llvm::enumerate(enc.getDimMap()))
    if (mapped != static_cast<int64_t>(dim))
      return emitOpError("only identity dim_map is supported");
  return success();
}

LogicalResult MakeGatherScatterViewOp::verify() {
  auto src = getSource().getType();
  auto res = getResult().getType();
  if (failed(verifyEncodingParent(*this, src, res)))
    return failure();
  auto enc = dyn_cast_or_null<GatherScatterViewAttr>(res.getEncoding());
  if (!enc)
    return emitOpError("result must carry a #tv.gather_scatter_view encoding");
  int64_t rank = res.getRank();
  if (static_cast<int64_t>(enc.getTile().size()) != rank)
    return emitOpError("gather/scatter tile rank must match the view rank");
  for (int64_t d : enc.getSparseDim())
    if (d < 0 || d >= rank)
      return emitOpError("sparse_dim entry ")
             << d << " out of range [0, " << rank << ")";
  return success();
}

static LogicalResult verifyViewAccess(Operation *op, TensorViewType view,
                                      ShapedType data, ValueRange indices,
                                      bool bufferized) {
  Attribute enc = view.getEncoding();
  if (!isViewEncoding(enc))
    return op->emitOpError("view must carry an encoding");
  if (indices.size() != getEncodingTileShape(enc).size())
    return op->emitOpError("expected one index per view dimension");
  if (data.getShape() != ArrayRef<int64_t>(getEncodingTileShape(enc)) ||
      data.getElementType() != view.getElementType())
    return op->emitOpError(
        "data type must match the view tile shape and element type");
  auto sparseDims = getEncodingSparseDims(enc);
  auto tile = getEncodingTileShape(enc);
  for (auto [dim, index] : llvm::enumerate(indices)) {
    if (!llvm::is_contained(sparseDims, static_cast<int64_t>(dim))) {
      if (!index.getType().isIndex())
        return op->emitOpError("regular coordinates must be scalar indices");
      continue;
    }
    auto type = dyn_cast<ShapedType>(index.getType());
    if (!type || !type.hasRank() || type.getRank() != 1 ||
        (bufferized ? !isa<MemRefType>(type) : !isa<RankedTensorType>(type)))
      return op->emitOpError(
          "sparse coordinates must be rank-one index tensors or buffers");
    Type elementType = type.getElementType();
    if (!elementType.isIndex() && !elementType.isInteger(32) &&
        !elementType.isInteger(64))
      return op->emitOpError("indices must have index, i32, or i64 elements");
    if (!type.isDynamicDim(0) && type.getDimSize(0) != tile[dim])
      return op->emitOpError(
          "sparse index length must match its tile dimension");
  }
  return success();
}

LogicalResult ViewLoadOp::verify() {
  auto data = cast<ShapedType>(getDst().getType());
  return verifyViewAccess(*this, getView().getType(), data, getIndices(),
                          isa<MemRefType>(data));
}
LogicalResult ViewStoreOp::verify() {
  auto data = cast<ShapedType>(getValue().getType());
  return verifyViewAccess(*this, getView().getType(), data, getIndices(),
                          isa<MemRefType>(data));
}
void ViewLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  if (isa<MemRefType>(getDst().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                         SideEffects::DefaultResource::get());
  for (OpOperand &index : getIndicesMutable())
    if (isa<MemRefType>(index.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &index,
                           SideEffects::DefaultResource::get());
}
void ViewStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
  if (isa<MemRefType>(getValue().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable(),
                         SideEffects::DefaultResource::get());
  for (OpOperand &index : getIndicesMutable())
    if (isa<MemRefType>(index.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &index,
                           SideEffects::DefaultResource::get());
}

} // namespace mlir::tv

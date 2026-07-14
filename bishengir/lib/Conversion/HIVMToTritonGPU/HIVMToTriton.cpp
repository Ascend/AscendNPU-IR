//===- HIVMToTriton.cpp - conversion from HIVM to Triton dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>
#include <numeric>

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::triton;

namespace {
Value castIndexToI32(ConversionPatternRewriter &rewriter, Location loc,
                            Value value) {
  if (value.getType().isInteger(32))
    return value;
  return rewriter.createOrFold<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                   value);
}

FailureOr<Value>
castI32TensorToType(ConversionPatternRewriter &rewriter, Location loc,
                    Value value, Type elementType) {
  auto srcTy = dyn_cast<RankedTensorType>(value.getType());
  if (!srcTy)
    return failure();
  if (srcTy.getElementType() == elementType)
    return value;

  auto dstTy = RankedTensorType::get(srcTy.getShape(), elementType);
  if (auto intTy = dyn_cast<IntegerType>(elementType)) {
    unsigned width = intTy.getWidth();
    if (width < 32)
      return rewriter.create<arith::TruncIOp>(loc, dstTy, value).getResult();
    if (width > 32)
      return rewriter.create<arith::ExtUIOp>(loc, dstTy, value).getResult();
    return rewriter.create<arith::BitcastOp>(loc, dstTy, value).getResult();
  }
  if (isa<FloatType>(elementType))
    return rewriter.create<arith::SIToFPOp>(loc, dstTy, value).getResult();

  return failure();
}

// Lowers hivm.hir.varange to an N-D strided index tensor:
//
//   result[i0, i1, ..., in] = offset + i0 * stride0 + i1 * stride1 + ... +
//                             in * striden
//
// For each dimension k, this builds tt.make_range(0, shape[k]), reshapes it to
// [1, ..., shape[k], ..., 1], broadcasts it to the final result shape, multiplies
// by the splatted stride[k], and accumulates all terms.  The accumulated i32
// tensor is cast to the requested result element type at the end.
FailureOr<Value>
buildArangeTensor(ConversionPatternRewriter &rewriter, Location loc,
                  ArrayRef<int64_t> shape, ValueRange strides, Value offset,
                  Type resultElementType) {
  if (shape.empty() || shape.size() != strides.size())
    return failure();

  auto i32Ty = rewriter.getI32Type();
  auto resultTy = RankedTensorType::get(shape, i32Ty);
  Value result;

  for (auto [dim, stride] : llvm::enumerate(strides)) {
    int64_t dimSize = shape[dim];
    if (dimSize <= 0 || dimSize > std::numeric_limits<int32_t>::max())
      return failure();

    auto dimTy = RankedTensorType::get({dimSize}, i32Ty);
    Value dimRange =
        rewriter.create<triton::MakeRangeOp>(loc, dimTy, 0, dimSize);
    if (shape.size() > 1) {
      // Materialize one per-dimension range and broadcast it to the final
      // result shape so we can accumulate a strided N-D linear index tensor.
      SmallVector<int64_t> reshapeShape(shape.size(), 1);
      reshapeShape[dim] = dimSize;
      auto reshapeTy = RankedTensorType::get(reshapeShape, i32Ty);
      dimRange =
          rewriter.create<triton::ReshapeOp>(loc, reshapeTy, dimRange, false);
      dimRange = rewriter.create<triton::BroadcastOp>(loc, resultTy, dimRange);
    }

    Value strideI32 = castIndexToI32(rewriter, loc, stride);
    Value strideTensor =
        rewriter.create<triton::SplatOp>(loc, resultTy, strideI32);
    Value term = rewriter.create<arith::MulIOp>(loc, dimRange, strideTensor);
    result = result ? rewriter.create<arith::AddIOp>(loc, result, term) : term;
  }

  if (offset) {
    Value offsetI32 = castIndexToI32(rewriter, loc, offset);
    Value offsetTensor =
        rewriter.create<triton::SplatOp>(loc, resultTy, offsetI32);
    result = result ? rewriter.create<arith::AddIOp>(loc, result, offsetTensor)
                    : offsetTensor;
  }

  return castI32TensorToType(rewriter, loc, result, resultElementType);
}

FailureOr<Value> buildReinterpretCastTensorPointers(
    ConversionPatternRewriter &rewriter, Location loc, Value base,
    MemRefType baseMemrefTy, ArrayRef<OpFoldResult> mixedOffsets,
    ArrayRef<OpFoldResult> mixedStrides, ArrayRef<int64_t> shape);

// Linearises multi-dimensional subview offsets into a single flat offset
// using the given per-dimension strides:
//   linearOffset = sum_i (offsets[i] * strides[i])
// via an affine.apply so that symbolic (runtime) offsets are supported.
static OpFoldResult
linearizeSubviewOffsets(ConversionPatternRewriter &rewriter, Location loc,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<int64_t> strides) {
  auto *ctx = rewriter.getContext();
  AffineExpr expr = getAffineConstantExpr(0, ctx);
  SmallVector<OpFoldResult> exprOperands;
  for (size_t i = 0; i < offsets.size(); ++i) {
    expr = expr + getAffineSymbolExpr(i, ctx) * strides[i];
    exprOperands.push_back(offsets[i]);
  }
  return affine::makeComposedFoldedAffineApply(rewriter, loc, expr,
                                               exprOperands);
}

// Extracts the common subview-to-pointer-tensor logic shared by
// HIVMLoalLoadOpPattern and HIVMLoalStoreOpPattern.  Given a memref.subview,
// computes the linearised base offset, the non-unit strides, and the transfer
// shape, then delegates to buildReinterpretCastTensorPointers to produce the
// final pointer tensor.
static FailureOr<Value>
buildSubviewPointerTensor(ConversionPatternRewriter &rewriter, Location loc,
                          memref::SubViewOp subviewOp) {
  auto subviewResultTy = cast<MemRefType>(subviewOp.getResult().getType());
  // SmallVector<int64_t> shape(subviewResultTy.getShape().begin(),
  //                            subviewResultTy.getShape().end());

  Value source = subviewOp.getSource();
  auto baseMemrefTy = dyn_cast<MemRefType>(source.getType());
  if (!baseMemrefTy)
    return failure();

  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset;
  if (failed(getStridesAndOffset(baseMemrefTy, sourceStrides, sourceOffset)))
    return failure();

  SmallVector<OpFoldResult> mixedStrides;
  auto layout = dyn_cast<StridedLayoutAttr>(subviewResultTy.getLayout());
  if (layout) {
    for (int64_t s : layout.getStrides())
      mixedStrides.push_back(rewriter.getIndexAttr(s));
  } else {
    auto sizes = subviewOp.getMixedSizes();
    for (size_t i = 0; i < sizes.size(); ++i) {
      auto sizeOpt = getConstantIntValue(sizes[i]);
      if (!sizeOpt || *sizeOpt != 1)
        mixedStrides.push_back(rewriter.getIndexAttr(sourceStrides[i]));
    }
  }

  auto subviewOffsets = subviewOp.getMixedOffsets();
  OpFoldResult linearOffset =
      linearizeSubviewOffsets(rewriter, loc, subviewOffsets, sourceStrides);
  SmallVector<OpFoldResult> mixedOffsets{linearOffset};

  return buildReinterpretCastTensorPointers(rewriter, loc, source, baseMemrefTy,
                                            mixedOffsets, mixedStrides, subviewResultTy.getShape());
}

class GetBlockIdxOpPattern : public OpRewritePattern<hivm::GetBlockIdxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::GetBlockIdxOp op,
                                PatternRewriter &rewriter) const override {
    // This pattern restores only the canonical form generated by
    // triton-global-kernel-args-to-hivm-op: a single i64 get_block_idx root
    // first truncated to i32, then consumed by the 1D-to-3D div/rem tree.
    SmallVector<arith::TruncIOp> truncUsers;
    SmallVector<Operation *> otherUsers;
    for (Operation *user : op->getUsers()) {
      if (auto truncOp = dyn_cast<arith::TruncIOp>(user))
        truncUsers.push_back(truncOp);
      else
        otherUsers.push_back(user);
    }

    if (!otherUsers.empty() || truncUsers.size() != 1) {
      op.emitOpError("is only supported when produced by "
                     "triton-global-kernel-args-to-hivm-op and consumed via "
                     "its canonical trunci/divsi/remsi decomposition");
      return failure();
    }

    arith::TruncIOp truncOp = truncUsers.front();
    // tt.get_program_id produces i32, so the canonical trunc is the value we
    // replace.  Keeping the match narrow avoids changing arbitrary i64 block-id
    // uses that do not represent Triton program ids.
    if (!truncOp.getType().isInteger(32)) {
      op.emitOpError("expects its canonical i32 truncation before restoring "
                     "tt.get_program_id");
      return failure();
    }

    // The mixed SIMD-SIMT path treats HIVM get_block_idx as the raw linear
    // program id, represented in Triton as program_id x.  The surrounding
    // canonical div/rem users recover logical x/y/z if they are still needed.
    auto pid = rewriter.create<triton::GetProgramIdOp>(op.getLoc(), 0);
    rewriter.replaceOp(truncOp, pid);
    rewriter.eraseOp(op);
    return success();
  }
};

// Convert hivm.hir.gather_load op into tt.load, for example:
// Before:
//  %1 = hivm.hir.gather_load ins(%base, %indices, %burst_len) outs(%dst)
// After:
//  %5 = tt.splat %base : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
//  %6 = tt.addptr %5, %indices : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
//  %7 = tt.load %6 : tensor<16x!tt.ptr<f32>>
class GatherLoadOpPattern : public OpConversionPattern<hivm::GatherLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hivm::GatherLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getResult()) {
      return rewriter.notifyMatchFailure(
          op, "only tensor destination gather_load is supported");
    }

    if (!isa<RankedTensorType>(adaptor.getDst().getType())) {
      return rewriter.notifyMatchFailure(
          op, "destination must be a ranked tensor type");
    }

    auto indices = adaptor.getIndices();
    auto indicesTy = dyn_cast<RankedTensorType>(indices.getType());
    if (!indicesTy) {
      return rewriter.notifyMatchFailure(
          op, "indices must be a ranked tensor type");
    }

    auto shape = indicesTy.getShape();
    auto loc = op.getLoc();
    auto base = adaptor.getBase();
    auto ptrTy = HIVMToTritonTypeConvert(base.getType());
    auto ttPtr = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, base);
    auto splatTy = RankedTensorType::get(shape, ptrTy);

    auto splat =
        rewriter.create<triton::SplatOp>(loc, splatTy, ttPtr.getResult(0));
    auto ptrTensor = splat.getResult();
    auto addptr = rewriter.create<triton::AddPtrOp>(loc, ptrTensor.getType(),
                                                    ptrTensor, indices);
    auto cache = triton::CacheModifier::NONE;
    if (auto res = adaptor.getCacheAttr()) {
      cache = static_cast<triton::CacheModifier>(res.getPolicy());
    }
    auto evict = triton::EvictionPolicy::NORMAL;
    if (auto res = adaptor.getEvictAttr()) {
      evict = static_cast<triton::EvictionPolicy>(res.getPolicy());
    }
    auto isVolatile = false;
    if (auto res = adaptor.getIsVolatile()) {
      isVolatile = res.value();
    }
    auto load = rewriter.create<triton::LoadOp>(
        loc, addptr.getResult(), adaptor.getMask(), adaptor.getOther(),
        llvm::ArrayRef<int32_t>{}, triton::PaddingOptionAttr{}, cache, evict,
        isVolatile);
    rewriter.replaceOp(op, load);

    return success();
  }
};

class HIVMLoalLoadOpPattern : public OpConversionPattern<hivm::LocalLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hivm::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addr = op.getAddr();
    auto ptrTy = HIVMToTritonTypeConvert(addr.getType());
    auto tensorTy = op.getResult().getType();

    if (auto subviewOp = addr.getDefiningOp<memref::SubViewOp>()) {
      auto ptrTensor = buildSubviewPointerTensor(rewriter, loc, subviewOp);
      if (failed(ptrTensor))
        return failure();

      auto loaded = rewriter.create<triton::LoadOp>(
          loc, tensorTy, *ptrTensor, Value{}, Value{},
          llvm::ArrayRef<int32_t>{}, triton::PaddingOptionAttr{});
      rewriter.replaceOp(op, loaded);
      return success();
    }

    auto num = tensorTy.getNumElements();
    auto rangeTensorTy = RankedTensorType::get({num}, rewriter.getI32Type());
    auto mkrng = rewriter.create<triton::MakeRangeOp>(op.getLoc(),
                                                      rangeTensorTy, 0, num);

    mlir::Value offset = mkrng;
    if (tensorTy.getRank() > 1) {
      auto reshapeTensorTy = RankedTensorType::get(
          tensorTy.getShape(), rangeTensorTy.getElementType());
      auto reshape = rewriter.create<triton::ReshapeOp>(op.getLoc(),
                                                        reshapeTensorTy, mkrng);
      offset = reshape;
    }

    auto ttPtr = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, addr);
    auto ptrTensorTy = RankedTensorType::get(tensorTy.getShape(), ptrTy);
    auto splat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTensorTy,
                                                  ttPtr.getResult(0));
    auto addptr = rewriter.create<triton::AddPtrOp>(op.getLoc(), ptrTensorTy,
                                                    splat, offset);

    auto valTensor = rewriter.create<triton::LoadOp>(
        op.getLoc(), tensorTy, addptr, Value{}, Value{},
        llvm::ArrayRef<int32_t>{}, triton::PaddingOptionAttr{});

    rewriter.replaceOp(op, valTensor);
    return success();
  }
};

// Convert hivm.hir.scatter_store op into tt.store, for example:
// Before:
//  hivm.hir.scatter_store ins(%indices, %data, %burst_len) outs(%base)
// After:
//  %5 = tt.splat %base : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
//  %6 = tt.addptr %5, %indices : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
//  tt.store %6, %data : tensor<16x!tt.ptr<f32>>
class ScatterStoreOpPattern : public OpConversionPattern<hivm::ScatterStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hivm::ScatterStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<MemRefType>(adaptor.getBase().getType())) {
      return rewriter.notifyMatchFailure(
          op, "only memref base scatter_store is supported");
    }

    auto indices = adaptor.getIndices();
    auto indicesTy = dyn_cast<RankedTensorType>(indices.getType());
    if (!indicesTy) {
      return rewriter.notifyMatchFailure(
          op, "indices must be a ranked tensor type");
    }
 
    auto shape = indicesTy.getShape();
    auto loc = op.getLoc();
    auto base = adaptor.getBase();
    auto ptrTy = HIVMToTritonTypeConvert(base.getType());
    auto ttPtr = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, base);
    auto splatTy = RankedTensorType::get(shape, ptrTy);
 
    auto splat = rewriter.create<triton::SplatOp>(loc, splatTy, ttPtr.getResult(0));
    auto ptrTensor = splat.getResult();
    auto addptr = rewriter.create<triton::AddPtrOp>(loc, ptrTensor.getType(), ptrTensor, indices);
    auto cache = triton::CacheModifier::NONE;
    if (auto res = adaptor.getCacheAttr()) {
      cache = static_cast<triton::CacheModifier>(res.getPolicy());
    }
    auto evict = triton::EvictionPolicy::NORMAL;
    if (auto res = adaptor.getEvictAttr()) {
      evict = static_cast<triton::EvictionPolicy>(res.getPolicy());
    }
    auto storeOp = rewriter.create<triton::StoreOp>(
        loc, addptr.getResult(), adaptor.getData(), adaptor.getMask(),
        llvm::ArrayRef<int32_t>{}, cache, evict);
    rewriter.replaceOp(op, storeOp);
 
    return success();
  }
};

class HIVMLoalStoreOpPattern : public OpConversionPattern<hivm::LocalStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hivm::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addr = op.getAddr();
    auto data = op.getData();
    auto ptrTy = HIVMToTritonTypeConvert(addr.getType());
    auto tensorTy = data.getType();

    if (auto subviewOp = addr.getDefiningOp<memref::SubViewOp>()) {
      auto ptrTensor = buildSubviewPointerTensor(rewriter, loc, subviewOp);
      if (failed(ptrTensor))
        return failure();

      rewriter.create<triton::StoreOp>(
          op.getLoc(), *ptrTensor, data, Value(), llvm::ArrayRef<int32_t>{},
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
      rewriter.eraseOp(op);
      return success();
    }

    auto num = tensorTy.getNumElements();
    auto rangeTensorTy = RankedTensorType::get({num}, rewriter.getI32Type());
    auto mkrng = rewriter.create<triton::MakeRangeOp>(op.getLoc(),
                                                      rangeTensorTy, 0, num);

    mlir::Value offset = mkrng;
    if (tensorTy.getRank() > 1) {
      auto reshapeTensorTy = RankedTensorType::get(
          tensorTy.getShape(), rangeTensorTy.getElementType());
      auto reshape = rewriter.create<triton::ReshapeOp>(op.getLoc(),
                                                        reshapeTensorTy, mkrng);
      offset = reshape;
    }

    auto ttPtr = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, addr);
    auto ptrTensorTy = RankedTensorType::get(tensorTy.getShape(), ptrTy);
    auto splat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTensorTy,
                                                  ttPtr.getResult(0));
    auto addptr = rewriter.create<triton::AddPtrOp>(op.getLoc(), ptrTensorTy,
                                                    splat, offset);

    rewriter.create<triton::StoreOp>(
        op.getLoc(), addptr, data, Value(), llvm::ArrayRef<int32_t>{},
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Computes the flattened offset tensor for memory accesses based on the target
/// MemRef's strided layout.
///
/// If the memory is perfectly contiguous, it simply returns the fast-path
/// `linearOffsets`. Otherwise, it computes a point-wise index tensor
/// corresponding to the shape scaling each dimension coordinate by the
/// respective stride in the MemRef layout.
///
static Value calcStridedOffsets(ConversionPatternRewriter &rewriter,
                                Location loc, MemRefType memrefTy,
                                ArrayRef<int64_t> shape, Value linearOffsets) {
  auto layout = dyn_cast<StridedLayoutAttr>(memrefTy.getLayout());
  if (!layout)
    return linearOffsets;

  auto strides = layout.getStrides();
  int64_t baseOffset = layout.getOffset();

  bool isContiguous = false;
  if (strides.empty() || (strides.size() == 1 && strides[0] == 1)) {
    isContiguous = true;
  } else if (shape.size() == strides.size() && !shape.empty() &&
             strides.back() == 1) {
    isContiguous = true;
    int64_t expect = 1;
    for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0;
         --dim) {
      if (strides[dim] != expect) {
        isContiguous = false;
        break;
      }
      expect *= shape[dim];
    }
  }

  if (isContiguous)
    return linearOffsets;

  auto i32Ty = rewriter.getI32Type();
  auto ndIndexTy = RankedTensorType::get(shape, i32Ty);

  auto zeroConst =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  Value offsets = rewriter.create<triton::SplatOp>(loc, ndIndexTy, zeroConst);

  for (size_t dim = 0; dim < shape.size(); ++dim) {
    int64_t dimLen = shape[dim];
    auto dimRangeTy = RankedTensorType::get({dimLen}, i32Ty);
    Value dimRange =
        rewriter.create<triton::MakeRangeOp>(loc, dimRangeTy, 0, dimLen);

    SmallVector<int64_t> reshapeShape(shape.size(), 1);
    reshapeShape[dim] = dimLen;
    auto dimReshapeTy = RankedTensorType::get(reshapeShape, i32Ty);
    Value dimReshaped =
        rewriter.create<triton::ReshapeOp>(loc, dimReshapeTy, dimRange, false);
    Value dimBroadcast =
        rewriter.create<triton::BroadcastOp>(loc, ndIndexTy, dimReshaped);

    int64_t stride = strides[dim];
    if (stride != 1) {
      auto strideConst = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(stride));
      auto strideTensor =
          rewriter.create<triton::SplatOp>(loc, ndIndexTy, strideConst);
      dimBroadcast =
          rewriter.create<arith::MulIOp>(loc, dimBroadcast, strideTensor);
    }

    offsets = rewriter.create<arith::AddIOp>(loc, offsets, dimBroadcast);
  }

  if (baseOffset != 0) {
    auto baseConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(baseOffset));
    auto baseSplat =
        rewriter.create<triton::SplatOp>(loc, ndIndexTy, baseConst);
    offsets = rewriter.create<arith::AddIOp>(loc, offsets, baseSplat);
  }

  return offsets;
}

FailureOr<Value> materializeI32Scalar(ConversionPatternRewriter &rewriter,
                                      Location loc, OpFoldResult ofr) {
  if (std::optional<int64_t> constant = getConstantIntValue(ofr)) {
    if (*constant < std::numeric_limits<int32_t>::min() ||
        *constant > std::numeric_limits<int32_t>::max())
      return failure();
    return rewriter
        .create<arith::ConstantOp>(loc,
                                   rewriter.getI32IntegerAttr(*constant))
        .getResult();
  }

  if (auto value = dyn_cast<Value>(ofr))
    return castIndexToI32(rewriter, loc, value);

  return failure();
}

FailureOr<Value> materializeI32Tensor(ConversionPatternRewriter &rewriter,
                                      Location loc, OpFoldResult ofr,
                                      RankedTensorType tensorTy) {
  if (std::optional<int64_t> constant = getConstantIntValue(ofr)) {
    if (*constant < std::numeric_limits<int32_t>::min() ||
        *constant > std::numeric_limits<int32_t>::max())
      return failure();

    auto attr =
        DenseElementsAttr::get(tensorTy, rewriter.getI32IntegerAttr(*constant));
    return rewriter.create<arith::ConstantOp>(loc, tensorTy, attr).getResult();
  }

  FailureOr<Value> scalarValue = materializeI32Scalar(rewriter, loc, ofr);
  if (failed(scalarValue))
    return failure();
  return rewriter.create<triton::SplatOp>(loc, tensorTy, *scalarValue)
      .getResult();
}

SmallVector<int64_t> getDimTermShape(ArrayRef<int64_t> shape, unsigned dim) {
  SmallVector<int64_t> termShape(shape.size(), 1);
  termShape[dim] = shape[dim];
  return termShape;
}

FailureOr<SmallVector<int64_t>> getBroadcastShape(ArrayRef<int64_t> lhs,
                                                  ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size())
    return failure();

  SmallVector<int64_t> result;
  result.reserve(lhs.size());
  for (auto [lhsDim, rhsDim] : llvm::zip_equal(lhs, rhs)) {
    if (lhsDim == rhsDim) {
      result.push_back(lhsDim);
      continue;
    }
    if (lhsDim == 1) {
      result.push_back(rhsDim);
      continue;
    }
    if (rhsDim == 1) {
      result.push_back(lhsDim);
      continue;
    }
    return failure();
  }
  return result;
}

Value broadcastTensor(ConversionPatternRewriter &rewriter, Location loc,
                      Value value, ArrayRef<int64_t> targetShape) {
  auto tensorTy = cast<RankedTensorType>(value.getType());
  if (llvm::equal(tensorTy.getShape(), targetShape))
    return value;

  auto targetTy =
      RankedTensorType::get(targetShape, tensorTy.getElementType());
  return rewriter.create<triton::BroadcastOp>(loc, targetTy, value);
}

FailureOr<Value>
buildDimOffsetTerm(ConversionPatternRewriter &rewriter, Location loc,
                   ArrayRef<int64_t> shape, unsigned dim, OpFoldResult stride,
                   std::optional<OpFoldResult> baseOffset = std::nullopt) {
  auto i32Ty = rewriter.getI32Type();
  int64_t dimLen = shape[dim];
  if (dimLen <= 0 || dimLen > std::numeric_limits<int32_t>::max())
    return failure();

  SmallVector<int64_t> termShape = getDimTermShape(shape, dim);
  auto termTy = RankedTensorType::get(termShape, i32Ty);
  auto dimRangeTy = RankedTensorType::get({dimLen}, i32Ty);
  Value term =
      rewriter.create<triton::MakeRangeOp>(loc, dimRangeTy, 0, dimLen);

  if (shape.size() > 1)
    term = rewriter.create<triton::ReshapeOp>(loc, termTy, term, false);

  if (!isConstantIntValue(stride, 1)) {
    FailureOr<Value> strideTensor =
        materializeI32Tensor(rewriter, loc, stride, termTy);
    if (failed(strideTensor))
      return failure();
    term = rewriter.create<arith::MulIOp>(loc, term, *strideTensor);
  }

  if (baseOffset && !isConstantIntValue(*baseOffset, 0)) {
    FailureOr<Value> offsetTensor =
        materializeI32Tensor(rewriter, loc, *baseOffset, termTy);
    if (failed(offsetTensor))
      return failure();
    term = rewriter.create<arith::AddIOp>(loc, term, *offsetTensor);
  }

  return term;
}

FailureOr<Value> buildReinterpretCastTensorPointers(
    ConversionPatternRewriter &rewriter, Location loc, Value base,
    MemRefType baseMemrefTy, ArrayRef<OpFoldResult> mixedOffsets,
    ArrayRef<OpFoldResult> mixedStrides, ArrayRef<int64_t> shape) {
  // Lower a reinterpret_cast view by expanding its strided descriptor into
  // per-dimension pointer arithmetic. For a 2-D view:
  //
  //   offset: [base_offset], sizes: [M, N], strides: [stride0, stride1]
  //
  // this creates the equivalent of:
  //
  //   row = make_range(0, M) -> tensor<Mx1xi32>
  //   row_offset = row * stride0 + base_offset
  //   row_ptr = addptr(splat(base) : tensor<Mx1xptr>, row_offset)
  //   col = make_range(0, N) -> tensor<1xNxi32>
  //   ptrs = addptr(broadcast(row_ptr) : tensor<MxNxptr>,
  //                 broadcast(col * stride1) : tensor<MxNxi32>)
  //
  // The final result is still a full tensor<...x!tt.ptr<T>> pointer tile for
  // tt.load/tt.store. Only the construction is staged so Triton can see the
  // sliced row-pointer form instead of a single fully-materialized offset tile.
  if (shape.empty() || mixedOffsets.size() != 1 ||
      mixedStrides.size() != shape.size())
    return failure();

  Type ptrTy = HIVMToTritonTypeConvert(baseMemrefTy);
  auto ttBase = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, base);

  Value ptrs;
  SmallVector<int64_t> currentShape;
  for (auto [dim, stride] : llvm::enumerate(mixedStrides)) {
    std::optional<OpFoldResult> baseOffset;
    if (dim == 0)
      baseOffset = mixedOffsets.front();

    FailureOr<Value> maybeTerm =
        buildDimOffsetTerm(rewriter, loc, shape, dim, stride, baseOffset);
    if (failed(maybeTerm))
      return failure();

    Value term = *maybeTerm;
    auto termTy = cast<RankedTensorType>(term.getType());
    SmallVector<int64_t> termShape(termTy.getShape().begin(),
                                   termTy.getShape().end());

    if (!ptrs) {
      auto ptrTensorTy = RankedTensorType::get(termShape, ptrTy);
      Value splat = rewriter.create<triton::SplatOp>(
          loc, ptrTensorTy, ttBase.getResult(0));
      ptrs = rewriter
                 .create<triton::AddPtrOp>(loc, ptrTensorTy, splat, term)
                 .getResult();
      currentShape = std::move(termShape);
      continue;
    }

    FailureOr<SmallVector<int64_t>> commonShape =
        getBroadcastShape(currentShape, termShape);
    if (failed(commonShape))
      return failure();

    ptrs = broadcastTensor(rewriter, loc, ptrs, *commonShape);
    term = broadcastTensor(rewriter, loc, term, *commonShape);

    auto ptrTensorTy = RankedTensorType::get(*commonShape, ptrTy);
    ptrs = rewriter.create<triton::AddPtrOp>(loc, ptrTensorTy, ptrs, term)
               .getResult();
    currentShape = std::move(*commonShape);
  }

  if (!llvm::equal(currentShape, shape))
    ptrs = broadcastTensor(rewriter, loc, ptrs, shape);

  return ptrs;
}

// Resolves a static transfer shape from two candidate MemRefs.
// It prefers the primary MemRef when its shape is static, and falls back
// to the secondary MemRef otherwise.
// Returns `std::nullopt` when neither MemRef provides a static shape.
static std::optional<SmallVector<int64_t>>
resolveStaticTransferShape(MemRefType primaryTy, MemRefType fallbackTy) {
  if (primaryTy && primaryTy.hasStaticShape()) {
    SmallVector<int64_t> shape(primaryTy.getShape().begin(),
                               primaryTy.getShape().end());
    return shape;
  }
  if (fallbackTy && fallbackTy.hasStaticShape()) {
    SmallVector<int64_t> shape(fallbackTy.getShape().begin(),
                               fallbackTy.getShape().end());
    return shape;
  }
  return std::nullopt;
}

// Creates a tensor of linear element offsets for the target shape.
// For 1-D accesses it returns the direct `tt.make_range` result.
// For N-D accesses it reshapes the linear range to the requested tensor shape.
static Value createLinearOffsetTensor(ConversionPatternRewriter &rewriter,
                                      Location loc,
                                      ArrayRef<int64_t> shape) {
  int64_t numElements = std::accumulate(shape.begin(), shape.end(), 1LL,
                                        std::multiplies<int64_t>());
  auto i32Ty = rewriter.getI32Type();
  auto flatIndexTy = RankedTensorType::get({numElements}, i32Ty);
  Value linearRange =
      rewriter.create<triton::MakeRangeOp>(loc, flatIndexTy, 0, numElements);
  if (shape.size() <= 1)
    return linearRange;

  auto ndIndexTy = RankedTensorType::get(shape, i32Ty);
  return rewriter.create<triton::ReshapeOp>(loc, ndIndexTy, linearRange,
                                            false);
}

// Builds a tensor of Triton pointers from a MemRef base value and offsets.
// The base pointer is first converted to a Triton pointer type, then splatted
// to the target tensor shape, and finally offset element-wise with `tt.addptr`.
static Value buildTensorPointers(ConversionPatternRewriter &rewriter,
                                 Location loc, Value base,
                                 MemRefType memrefTy,
                                 ArrayRef<int64_t> shape, Value offsets) {
  Type ptrTy = HIVMToTritonTypeConvert(memrefTy);
  auto ttBase =
      rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, base);
  auto ptrTensorTy = RankedTensorType::get(shape, ptrTy);
  auto splat = rewriter.create<triton::SplatOp>(loc, ptrTensorTy,
                                                ttBase.getResult(0));
  return rewriter
      .create<triton::AddPtrOp>(loc, ptrTensorTy, splat, offsets)
      .getResult();
}

FailureOr<Value>
buildMemRefTensorPointers(ConversionPatternRewriter &rewriter, Location loc,
                          Value originalValue, Value convertedValue,
                          MemRefType memrefTy, ArrayRef<int64_t> shape,
                          llvm::function_ref<Value()> getLinearOffsets) {
  if (auto castOp =
          originalValue.getDefiningOp<memref::ReinterpretCastOp>()) {
    Value base = castOp.getSource();
    auto baseMemrefTy = dyn_cast<MemRefType>(base.getType());
    if (!baseMemrefTy)
      return failure();

    SmallVector<OpFoldResult> mixedOffsets = castOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedStrides = castOp.getMixedStrides();
    return buildReinterpretCastTensorPointers(
        rewriter, loc, base, baseMemrefTy, mixedOffsets, mixedStrides, shape);
  }

  if (auto subviewOp = originalValue.getDefiningOp<memref::SubViewOp>()) {
    // Require unit strides on the subview itself
    for (auto svStride : subviewOp.getMixedStrides()) {
      if (!isConstantIntValue(svStride, 1))
        return failure();
    }

    // Per-dim strides come from the SubView source: prefer the strides of
    // an upstream ReinterpretCast (so the base pointer is the raw memref);
    // otherwise read the strided layout from the source memref type.
    Value source = subviewOp.getSource();
    SmallVector<OpFoldResult> mixedStrides;
    if (auto srcCast = source.getDefiningOp<memref::ReinterpretCastOp>()) {
      mixedStrides = srcCast.getMixedStrides();
      source = srcCast.getSource();
    } else {
      auto sourceMemrefTy = dyn_cast<MemRefType>(source.getType());
      if (!sourceMemrefTy)
        return failure();
      auto layout =
          dyn_cast<StridedLayoutAttr>(sourceMemrefTy.getLayout());
      if (!layout)
        return failure();
      for (int64_t s : layout.getStrides())
        mixedStrides.push_back(rewriter.getIndexAttr(s));
    }

    auto baseMemrefTy = dyn_cast<MemRefType>(source.getType());
    if (!baseMemrefTy)
      return failure();

    auto subviewOffsets = subviewOp.getMixedOffsets();
    if (subviewOffsets.size() != mixedStrides.size())
      return failure();

    SmallVector<int64_t> intStrides;
    intStrides.reserve(mixedStrides.size());
    for (auto ms : mixedStrides) {
      auto s = getConstantIntValue(ms);
      if (!s)
        return failure();
      intStrides.push_back(*s);
    }

    OpFoldResult linearOffset =
        linearizeSubviewOffsets(rewriter, loc, subviewOffsets, intStrides);
    SmallVector<OpFoldResult> mixedOffsets{linearOffset};
    return buildReinterpretCastTensorPointers(rewriter, loc, source,
                                              baseMemrefTy, mixedOffsets,
                                              mixedStrides, shape);
  }

  Value offsets =
      calcStridedOffsets(rewriter, loc, memrefTy, shape, getLinearOffsets());
  return buildTensorPointers(rewriter, loc, convertedValue, memrefTy, shape,
                             offsets);
}

// Maps HIVM atomic kinds to the corresponding Triton RMW operations.
// Returns `std::nullopt` for atomic kinds that do not have a Triton mapping.
static std::optional<triton::RMWOp> toTritonRMWOp(hivm::AtomicKind kind) {
  switch (kind) {
  case hivm::AtomicKind::ADD:
    return triton::RMWOp::ADD;
  case hivm::AtomicKind::MAX:
    return triton::RMWOp::MAX;
  case hivm::AtomicKind::MIN:
    return triton::RMWOp::MIN;
  case hivm::AtomicKind::AND:
    return triton::RMWOp::AND;
  case hivm::AtomicKind::OR:
    return triton::RMWOp::OR;
  case hivm::AtomicKind::XOR:
    return triton::RMWOp::XOR;
  case hivm::AtomicKind::XCHG:
    return triton::RMWOp::XCHG;
  default:
    return std::nullopt;
  }
}

// Convert hivm.load op into Triton arithmetic and memory ops.
// Supported Conversion Scenarios:
// 1. Loads data from source `memref` into Triton registers using `tt.load`.
// 2. Address calculation supports contiguous memory (fast-path) and
//    N-D Strided Layout via `calcStridedOffsets`.
// 3. Constraints: The loaded data must be solely consumed by
// `bufferization.to_tensor`.
//    If other consumers exist, the conversion checks and will emit an error.
//    Otherwise, it replaces the `to_tensor` users with the loaded Triton tensor
//    directly.
class HIVMLoadOpPattern : public OpConversionPattern<hivm::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hivm::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = adaptor.getSrc();
    auto dst = adaptor.getDst();
    auto evict = triton::EvictionPolicy::NORMAL;
    if (auto evictAttr = op.getEvictionPolicy()) {
      switch (evictAttr->getPolicy()) {
      case hivm::EvictionPolicy::EvictNormal:
        evict = triton::EvictionPolicy::NORMAL;
        break;
      case hivm::EvictionPolicy::EvictFirst:
        evict = triton::EvictionPolicy::EVICT_FIRST;
        break;
      case hivm::EvictionPolicy::EvictLast:
        evict = triton::EvictionPolicy::EVICT_LAST;
        break;
      }
    }
    Value other = adaptor.getPadValue();

    // Guard: padded loads cannot be reversed to plain tt.load
    if (op.getPadMode())
      return rewriter.notifyMatchFailure(op, "padded load not converted");

    // === Only support Memref form ===
    auto srcMemrefTy = dyn_cast<MemRefType>(src.getType());
    auto dstMemrefTy = dyn_cast<MemRefType>(dst.getType());
    if (!srcMemrefTy || !dstMemrefTy)
      return failure();

    auto shapeOr = resolveStaticTransferShape(srcMemrefTy, dstMemrefTy);
    if (!shapeOr)
      return rewriter.notifyMatchFailure(op, "cannot resolve shape");
    SmallVector<int64_t> shape = *shapeOr;

    Value linearOffsets;
    auto getLinearOffsets = [&]() -> Value {
      if (!linearOffsets)
        linearOffsets = createLinearOffsetTensor(rewriter, loc, shape);
      return linearOffsets;
    };

    FailureOr<Value> maybeSrcPtrs = buildMemRefTensorPointers(
        rewriter, loc, op.getSrc(), src, srcMemrefTy, shape, getLinearOffsets);
    if (failed(maybeSrcPtrs))
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize source pointers");
    Value srcPtrs = *maybeSrcPtrs;

    // tt.load from GM
    auto loaded = rewriter.create<triton::LoadOp>(
        loc, srcPtrs, Value(), other, llvm::ArrayRef<int32_t>{},
        std::nullopt, triton::CacheModifier::NONE, evict, false);

    Value loadedTensor = loaded.getResult();

    // Scan dst users for to_tensor
    SmallVector<bufferization::ToTensorOp> toTensorUsers;
    bool hasOtherUsers = false;
    bool hasbufferization = false;
    for (Operation *user : op.getDst().getUsers()) {
      if (user == op.getOperation())
        continue;
      if (isa<UnrealizedConversionCastOp>(user))
        continue;
      if (auto tt = dyn_cast<bufferization::ToTensorOp>(user)) {
        hasbufferization = true;
        toTensorUsers.push_back(tt);
      } else
        hasOtherUsers = true;
    }

    // Replace to_tensor users
    if (!toTensorUsers.empty()) {
      for (auto tt : toTensorUsers)
        rewriter.replaceOp(tt, loadedTensor);
    }

    if (hasOtherUsers && hasbufferization) {
      return op->emitError(
          "hivm.load's dst should only be used by bufferization.to_tensor");
    }

    // Store to dst if needed
    if (toTensorUsers.empty()) {
      FailureOr<Value> maybeDstPtrs =
          buildMemRefTensorPointers(rewriter, loc, op.getDst(), dst,
                                    dstMemrefTy, shape, getLinearOffsets);
      if (failed(maybeDstPtrs))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize destination pointers");
      Value dstPtrs = *maybeDstPtrs;

      rewriter.create<triton::StoreOp>(
          loc, dstPtrs, loadedTensor, Value(), llvm::ArrayRef<int32_t>{},
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Convert hivm.store op into Triton memory or atomic operations.
// Supported Conversion Scenarios (in order of matching logic):
// 1. Atomic Store Memory Operations (Branch 1)
//    - Triggered when `atomic_kind` is present.
//    - If source is a `memref`, sequentially loads data into registers via
//    `tt.load`.
//    - Translates `atomic_kind` (add, max, min, etc.) to Triton's `rmw_op`.
//    - Issues `tt.atomic_rmw` to the destination pointer with enforced
//    `ACQUIRE_RELEASE` semantic.
// 2. Direct Tensor-to-MemRef Fast Save (Branch 2)
//    - The primary operand is naturally a `tensor` computed down from previous
//    MLIR calculation loops.
//    - Directly computes strided destination addresses.
//    - Uses `tt.store` to write vector tensor to Global Memory (`memref`).
// 3. MemRef Buffer Transfers (Branch 3)
//    - Plain memref -> memref data move. Generates sequential `tt.load` ->
//    `tt.store`.
class HIVMStoreOpPattern : public OpConversionPattern<hivm::StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hivm::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = adaptor.getSrc();
    auto dst = adaptor.getDst();

    // === Branch 1: Atomic store ===
    if (op.getAtomicKind()) {
      auto rmwOp = toTritonRMWOp(op.getAtomicKind().value());
      if (!rmwOp)
        return rewriter.notifyMatchFailure(op, "unsupported atomic kind");

      // Resolve store value as tensor
      Value storeVal = src;
      if (auto srcMemrefTy = dyn_cast<MemRefType>(src.getType())) {
        SmallVector<int64_t> srcShape(srcMemrefTy.getShape().begin(),
                                      srcMemrefTy.getShape().end());
        int64_t numElems = std::accumulate(srcShape.begin(), srcShape.end(),
                                           1LL, std::multiplies<int64_t>());
        auto srcTensorTy = RankedTensorType::get(srcShape,
                                                 srcMemrefTy.getElementType());
        SmallVector<int64_t> flatShape{numElems};
        Value srcRange = createLinearOffsetTensor(rewriter, loc, flatShape);
        Value srcPtrs = buildTensorPointers(rewriter, loc, src, srcMemrefTy,
                                            flatShape, srcRange);

        auto loaded1D = rewriter.create<triton::LoadOp>(
            loc, srcPtrs, Value(), Value(), llvm::ArrayRef<int32_t>{},
            std::nullopt, triton::CacheModifier::NONE,
            triton::EvictionPolicy::NORMAL, false);

        if (srcTensorTy.getRank() > 1) {
          storeVal = rewriter.create<triton::ReshapeOp>(
              loc, srcTensorTy, loaded1D.getResult(), false);
        } else {
          storeVal = loaded1D.getResult();
        }
      }

      auto valTy = cast<RankedTensorType>(storeVal.getType());
      auto shape = valTy.getShape();
      int64_t numElements = std::accumulate(shape.begin(), shape.end(), 1LL,
                                            std::multiplies<int64_t>());

      auto dstMemrefTy = cast<MemRefType>(dst.getType());
      SmallVector<int64_t> flatShape{numElements};
      Value range = createLinearOffsetTensor(rewriter, loc, flatShape);
      Value ptrs =
          buildTensorPointers(rewriter, loc, dst, dstMemrefTy, flatShape,
                              range);

      // Flatten if rank > 1 
      if (valTy.getRank() > 1) {
        auto flatTy =
            RankedTensorType::get({numElements}, valTy.getElementType());
        storeVal =
            rewriter.create<triton::ReshapeOp>(loc, flatTy, storeVal, false);
      }

      auto rmwAttr = triton::RMWOpAttr::get(rewriter.getContext(), *rmwOp);
      auto semAttr = triton::MemSemanticAttr::get(
          rewriter.getContext(), triton::MemSemantic::ACQUIRE_RELEASE);
      auto scopeAttr = triton::MemSyncScopeAttr::get(rewriter.getContext(),
                                                     triton::MemSyncScope::GPU);
      rewriter.create<triton::AtomicRMWOp>(loc, storeVal.getType(), rmwAttr,
                                           ptrs, storeVal, Value(), semAttr,
                                           scopeAttr);
      rewriter.eraseOp(op);
      return success();
    }

    // === Branch 2: Tensor -> GM memref ===
    if (auto srcTensorTy = dyn_cast<RankedTensorType>(src.getType())) {
      auto dstMemrefTy = dyn_cast<MemRefType>(dst.getType());
      if (!dstMemrefTy)
        return failure();

      auto shape = srcTensorTy.getShape();
      Value linearOffsets;
      auto getLinearOffsets = [&]() -> Value {
        if (!linearOffsets)
          linearOffsets = createLinearOffsetTensor(rewriter, loc, shape);
        return linearOffsets;
      };

      FailureOr<Value> maybeDstPtrs =
          buildMemRefTensorPointers(rewriter, loc, op.getDst(), dst,
                                    dstMemrefTy, shape, getLinearOffsets);
      if (failed(maybeDstPtrs))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize destination pointers");
      Value dstPtrs = *maybeDstPtrs;

      rewriter.create<triton::StoreOp>(
          loc, dstPtrs, src, Value(), llvm::ArrayRef<int32_t>{},
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);

      rewriter.eraseOp(op);
      return success();
    }

    // === Memref src -> memref dst ===
    auto srcMemrefTy = dyn_cast<MemRefType>(src.getType());
    auto dstMemrefTy = dyn_cast<MemRefType>(dst.getType());
    if (!srcMemrefTy || !dstMemrefTy)
      return failure();

    // === Branch 3: Plain memref -> tt.load + tt.store ===
    auto shapeOr = resolveStaticTransferShape(srcMemrefTy, dstMemrefTy);
    if (!shapeOr)
      return rewriter.notifyMatchFailure(op, "cannot resolve shape");
    SmallVector<int64_t> shape = *shapeOr;

    int64_t numElements = std::accumulate(shape.begin(), shape.end(), 1LL,
                                          std::multiplies<int64_t>());
    SmallVector<int64_t> flatShape{numElements};

    Value range = createLinearOffsetTensor(rewriter, loc, flatShape);

    // Load from UB
    Value srcPtrs = buildTensorPointers(rewriter, loc, src, srcMemrefTy,
                                        flatShape, range);
    auto loaded = rewriter.create<triton::LoadOp>(
        loc, srcPtrs, Value(), Value(), llvm::ArrayRef<int32_t>{},
        std::nullopt, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, false);

    // Store to GM
    Value dstPtrs = buildTensorPointers(rewriter, loc, dst, dstMemrefTy,
                                        flatShape, range);

    rewriter.create<triton::StoreOp>(
        loc, dstPtrs, loaded.getResult(), Value(), llvm::ArrayRef<int32_t>{},
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);

    rewriter.eraseOp(op);
    return success();
  }
};

class VArangeOpPattern : public OpConversionPattern<hivm::VArangeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hivm::VArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op->getNumResults() == 0)
      return op.emitOpError("buffer-form varange is not supported");

    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy || !resultTy.hasStaticShape())
      return op.emitOpError("requires a ranked static tensor result");

    auto resultTensor =
        buildArangeTensor(rewriter, loc, resultTy.getShape(),
                          adaptor.getStrides(), adaptor.getOffset(),
                          resultTy.getElementType());
    if (failed(resultTensor))
      return op.emitOpError(
          "unsupported shape, strides, or result element type");

    rewriter.replaceOp(op, resultTensor.value());
    return success();
  }
};

class VBrcOpPattern : public OpConversionPattern<hivm::VBrcOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hivm::VBrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op->getNumResults() == 0)
      return op.emitOpError("buffer-form vbrc is not supported");

    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy || !resultTy.hasStaticShape()) {
      return op.emitOpError("requires a ranked static tensor result");
    }

    Value resultTensor;
    if (isa<RankedTensorType>(adaptor.getSrc().getType())) {
      // HFusion inserts expand_shape before vbrc so the source already matches
      // the destination rank and Triton only needs a pure broadcast here.
      resultTensor =
          rewriter.create<triton::BroadcastOp>(loc, resultTy, adaptor.getSrc());
    } else if (adaptor.getSrc().getType().isIntOrFloat()) {
      // Scalar broadcast is a splat in Triton IR.
      resultTensor =
          rewriter.create<triton::SplatOp>(loc, resultTy, adaptor.getSrc());
    } else {
      return op.emitOpError("only tensor or scalar sources are supported");
    }

    rewriter.replaceOp(op, resultTensor);
    return success();
  }
};

// Convert hivm.hir.vreduce to tt.reduce
// Before: %2 = hivm.hir.vreduce <sum> (%0： tensor<16x16xf32>) outs(%1: tensor<1x16xf32>) unsigned_src = false reduce_dims=[0] ->tensor<16xf32>
// After: %2 = tt.reduce （%0）<{axis=0:i32}> ({
//     ^bb0(%arg0:f32, %arg1:f32){
//          %1 = arith.addf %arg0, %arg1
//           tt.reduce.return %1} }) : (tensor<16x16xf32>) -> tensor<16xf32>
//   %3 = tt.expand_dims ...
struct HIVMToTTReduceOp: public OpRewritePattern<hivm::VReduceOp> {
    using OpRewritePattern<hivm::VReduceOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(hivm::VReduceOp op,
                                PatternRewriter &rewriter) const final {
        auto loc = op.getLoc();
        Value src = op.getSrc();
        
        if (isa<MemRefType>(src.getType())) {
            return op.emitOpError("memref source is not supported currently");
        }
        auto srcType = cast<RankedTensorType>(src.getType());
        auto elemType = srcType.getElementType();
        
        auto reduceDims = op.getReduceDims();
        if (reduceDims.empty()) {
            return failure();
        }
        
        auto arithAttr = op.getArithAttr();
        auto reduceOp = arithAttr.getReduceOp();
        auto dstType = cast<RankedTensorType>(op.getDstValue().getType());

        Value finalResult = src;
        SmallVector<int64_t> currentShape(srcType.getShape().begin(), srcType.getShape().end());
        
        // Currently, we reduce dims in order, and expand dims to match dstType, since triton::ReduceOp only supports reduction in single axis.
        for (auto axis : reduceDims) {
          SmallVector<int64_t> resultShape(currentShape.begin(), currentShape.end());
          resultShape.erase(resultShape.begin() + axis);
          RankedTensorType reduceResultType = RankedTensorType::get(resultShape, elemType);
          
          auto adjustedAxis = axis;
          auto ttReduceOp = rewriter.create<triton::ReduceOp>(
              loc,
              reduceResultType,
              finalResult,
              adjustedAxis
          );
          
          Region &combineRegion = ttReduceOp.getCombineOp();
          rewriter.createBlock(&combineRegion);
          Block &block = combineRegion.front();
          block.addArgument(elemType, loc);
          block.addArgument(elemType, loc);
          
          rewriter.setInsertionPointToEnd(&block);
          Value arg0 = block.getArgument(0);
          Value arg1 = block.getArgument(1);
          Value result;
          
          switch (reduceOp) {
          case hivm::ReduceOperation::sum:
              if (isa<FloatType>(elemType)) {
                  result = rewriter.create<arith::AddFOp>(loc, arg0, arg1);
              } else {
                  result = rewriter.create<arith::AddIOp>(loc, arg0, arg1);
              }
              break;
          case hivm::ReduceOperation::prod:
              if (isa<FloatType>(elemType)) {
                  result = rewriter.create<arith::MulFOp>(loc, arg0, arg1);
              } else {
                  result = rewriter.create<arith::MulIOp>(loc, arg0, arg1);
              }
              break;
          case hivm::ReduceOperation::max:
              if (isa<FloatType>(elemType)) {
                  result = rewriter.create<arith::MaximumFOp>(loc, arg0, arg1);
              } else if (op.getUnsignedSrc()) {
                  result = rewriter.create<arith::MaxUIOp>(loc, arg0, arg1);
              } else {
                  result = rewriter.create<arith::MaxSIOp>(loc, arg0, arg1);
              }
              break;
          case hivm::ReduceOperation::min:
              if (isa<FloatType>(elemType)) {
                  result = rewriter.create<arith::MinimumFOp>(loc, arg0, arg1);
              } else if (op.getUnsignedSrc()) {
                  result = rewriter.create<arith::MinUIOp>(loc, arg0, arg1);
              } else {
                  result = rewriter.create<arith::MinSIOp>(loc, arg0, arg1);
              }
              break;
          case hivm::ReduceOperation::andi:
              result = rewriter.create<arith::AndIOp>(loc, arg0, arg1);
              break;
          case hivm::ReduceOperation::ori:
              result = rewriter.create<arith::OrIOp>(loc, arg0, arg1);
              break;
          case hivm::ReduceOperation::xori:
              result = rewriter.create<arith::XOrIOp>(loc, arg0, arg1);
              break;
          case hivm::ReduceOperation::any:
              result = rewriter.create<arith::OrIOp>(loc, arg0, arg1);
              break;
          case hivm::ReduceOperation::all:
              result = rewriter.create<arith::AndIOp>(loc, arg0, arg1);
              break;
          default:
              return failure();
          }
          
          rewriter.create<triton::ReduceReturnOp>(loc, result);
          rewriter.setInsertionPointAfter(ttReduceOp);
          
          Value reduceResult = ttReduceOp->getResult(0);
          finalResult = reduceResult;
          currentShape = resultShape;
          
          // triton::ReduceOp removes the reduced dimension, but HIVM keeps it as size 1
          auto currentResultType = cast<RankedTensorType>(reduceResult.getType());
          if (currentResultType.getRank() != dstType.getRank()) {
              // Insert dimension of size 1 at the reduced axis position
              SmallVector<int64_t> expandShape(currentShape.begin(), currentShape.end());
              expandShape.insert(expandShape.begin() + axis, 1);
              RankedTensorType finalType = RankedTensorType::get(expandShape, elemType);
              finalResult = rewriter.create<triton::ExpandDimsOp>(loc, finalType, reduceResult, axis);
              currentShape = expandShape;
          }
        }
        
        rewriter.replaceOp(op, finalResult);
        return success();
    }
};

} // namespace

void mlir::hivm::populateHIVMToTritonPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns
      .add<GetBlockIdxOpPattern, GatherLoadOpPattern, ScatterStoreOpPattern,
           HIVMLoadOpPattern, HIVMStoreOpPattern, HIVMLoalLoadOpPattern,
           HIVMLoalStoreOpPattern, VArangeOpPattern, VBrcOpPattern,
           HIVMToTTReduceOp>(context);
}

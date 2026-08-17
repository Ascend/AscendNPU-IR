//===- HIVMToTriton.cpp - conversion from HIVM to Triton dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonUtils.h"
#include "bishengir/Conversion/HIVMToTritonGPU/MemRefDescriptor.h"
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
  auto i32Ty = rewriter.getI32Type();
  if (value.getType().isInteger(32))
    return value;
  if (isa<IndexType>(value.getType()))
    return rewriter.createOrFold<arith::IndexCastOp>(loc, i32Ty, value);
  auto intTy = cast<IntegerType>(value.getType());
  if (intTy.getWidth() > 32)
    return rewriter.createOrFold<arith::TruncIOp>(loc, i32Ty, value);
  return rewriter.createOrFold<arith::ExtSIOp>(loc, i32Ty, value);
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

FailureOr<Value>
buildDescriptorPointerTensor(ConversionPatternRewriter &rewriter, Location loc,
                             const hivm::MemRefDescriptor &desc, Type ptrTy,
                             ArrayRef<int64_t> shape);

// A 1:N adaptor hands every operand back as a ValueRange. Non-MemRef operands
// still expand 1:1, so this picks the single value out
static Value onlyValue(ValueRange range) {
  assert(range.size() <= 1 && "1:N operand truncated; use getMemRefDescriptor");
  return range.empty() ? Value() : range.front();
}

// Returns the scalar base pointer a descriptor addresses, with its composed
// view offset applied.
static Value getDescriptorBasePtr(ConversionPatternRewriter &rewriter,
                                  Location loc,
                                  const hivm::MemRefDescriptor &desc,
                                  Type ptrTy) {
  return rewriter.createOrFold<triton::AddPtrOp>(loc, ptrTy, desc.basePtr(),
                                                 desc.offset);
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
  using OpConversionPattern<hivm::GatherLoadOp>::OneToNOpAdaptor;
  LogicalResult
  matchAndRewrite(hivm::GatherLoadOp op, OneToNOpAdaptor nAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getResult()) {
      return rewriter.notifyMatchFailure(
          op, "only tensor destination gather_load is supported");
    }

    if (!isa<RankedTensorType>(op.getDst().getType())) {
      return rewriter.notifyMatchFailure(
          op, "destination must be a ranked tensor type");
    }

    Value indices = onlyValue(nAdaptor.getIndices());
    auto indicesTy =
        indices ? dyn_cast<RankedTensorType>(indices.getType()) : nullptr;
    if (!indicesTy) {
      return rewriter.notifyMatchFailure(
          op, "indices must be a ranked tensor type");
    }

    auto shape = indicesTy.getShape();
    auto loc = op.getLoc();
    auto ptrTy = HIVMToTritonTypeConvert(op.getBase().getType());
    FailureOr<hivm::MemRefDescriptor> desc =
        hivm::getMemRefDescriptor(
            rewriter, loc, cast<MemRefType>(op.getBase().getType()),
            nAdaptor.getBase());
    if (failed(desc))
      return rewriter.notifyMatchFailure(op, "no descriptor for base");
    Value ttPtr = getDescriptorBasePtr(rewriter, loc, *desc, ptrTy);
    auto splatTy = RankedTensorType::get(shape, ptrTy);

    auto splat = rewriter.create<triton::SplatOp>(loc, splatTy, ttPtr);
    auto ptrTensor = splat.getResult();
    auto addptr = rewriter.create<triton::AddPtrOp>(loc, ptrTensor.getType(),
                                                    ptrTensor, indices);
    auto cache = triton::CacheModifier::NONE;
    if (auto res = op.getCacheAttr()) {
      cache = static_cast<triton::CacheModifier>(res.getPolicy());
    }
    auto evict = triton::EvictionPolicy::NORMAL;
    if (auto res = op.getEvictAttr()) {
      evict = static_cast<triton::EvictionPolicy>(res.getPolicy());
    }
    auto isVolatile = false;
    if (auto res = op.getIsVolatile()) {
      isVolatile = res.value();
    }
    auto load = rewriter.create<triton::LoadOp>(
        loc, addptr.getResult(), onlyValue(nAdaptor.getMask()),
        onlyValue(nAdaptor.getOther()),
        llvm::ArrayRef<int32_t>{}, triton::PaddingOptionAttr{}, cache, evict,
        isVolatile);
    rewriter.replaceOp(op, load);

    return success();
  }
};

class HIVMLoalLoadOpPattern : public OpConversionPattern<hivm::LocalLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  using OpConversionPattern<hivm::LocalLoadOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(hivm::LocalLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addr = op.getAddr();
    auto ptrTy = HIVMToTritonTypeConvert(addr.getType());
    auto tensorTy = op.getResult().getType();

    FailureOr<hivm::MemRefDescriptor> desc =
        hivm::getMemRefDescriptor(
            rewriter, loc, cast<MemRefType>(addr.getType()), adaptor.getAddr());
    if (failed(desc) || desc->getRank() != tensorTy.getRank())
      return rewriter.notifyMatchFailure(op, "addr is not a MemRef descriptor");

    FailureOr<Value> ptrTensor = buildDescriptorPointerTensor(
        rewriter, loc, *desc, ptrTy, tensorTy.getShape());
    if (failed(ptrTensor))
      return rewriter.notifyMatchFailure(op, "cannot build pointer tile");

    auto loaded = rewriter.create<triton::LoadOp>(
        loc, tensorTy, *ptrTensor, Value{}, Value{},
        llvm::ArrayRef<int32_t>{}, triton::PaddingOptionAttr{});
    rewriter.replaceOp(op, loaded);
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
  using OpConversionPattern<hivm::ScatterStoreOp>::OneToNOpAdaptor;
  LogicalResult
  matchAndRewrite(hivm::ScatterStoreOp op, OneToNOpAdaptor nAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<MemRefType>(op.getBase().getType())) {
      return rewriter.notifyMatchFailure(
          op, "only memref base scatter_store is supported");
    }

    Value indices = onlyValue(nAdaptor.getIndices());
    auto indicesTy =
        indices ? dyn_cast<RankedTensorType>(indices.getType()) : nullptr;
    if (!indicesTy) {
      return rewriter.notifyMatchFailure(
          op, "indices must be a ranked tensor type");
    }

    auto shape = indicesTy.getShape();
    auto loc = op.getLoc();
    auto ptrTy = HIVMToTritonTypeConvert(op.getBase().getType());
    FailureOr<hivm::MemRefDescriptor> desc =
        hivm::getMemRefDescriptor(
            rewriter, loc, cast<MemRefType>(op.getBase().getType()),
            nAdaptor.getBase());
    if (failed(desc))
      return rewriter.notifyMatchFailure(op, "no descriptor for base");
    Value ttPtr = getDescriptorBasePtr(rewriter, loc, *desc, ptrTy);
    auto splatTy = RankedTensorType::get(shape, ptrTy);

    auto splat = rewriter.create<triton::SplatOp>(loc, splatTy, ttPtr);
    auto ptrTensor = splat.getResult();
    auto addptr = rewriter.create<triton::AddPtrOp>(loc, ptrTensor.getType(), ptrTensor, indices);
    auto cache = triton::CacheModifier::NONE;
    if (auto res = op.getCacheAttr()) {
      cache = static_cast<triton::CacheModifier>(res.getPolicy());
    }
    auto evict = triton::EvictionPolicy::NORMAL;
    if (auto res = op.getEvictAttr()) {
      evict = static_cast<triton::EvictionPolicy>(res.getPolicy());
    }
    auto storeOp = rewriter.create<triton::StoreOp>(
        loc, addptr.getResult(), onlyValue(nAdaptor.getData()),
        onlyValue(nAdaptor.getMask()),
        llvm::ArrayRef<int32_t>{}, cache, evict);
    rewriter.replaceOp(op, storeOp);

    return success();
  }
};

class HIVMLoalStoreOpPattern : public OpConversionPattern<hivm::LocalStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  using OpConversionPattern<hivm::LocalStoreOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(hivm::LocalStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto addr = op.getAddr();
    auto data = op.getData();
    auto ptrTy = HIVMToTritonTypeConvert(addr.getType());
    auto tensorTy = data.getType();

    FailureOr<hivm::MemRefDescriptor> desc =
        hivm::getMemRefDescriptor(
            rewriter, loc, cast<MemRefType>(addr.getType()), adaptor.getAddr());
    if (failed(desc) || desc->getRank() != tensorTy.getRank())
      return rewriter.notifyMatchFailure(op, "addr is not a MemRef descriptor");

    FailureOr<Value> ptrTensor = buildDescriptorPointerTensor(
        rewriter, loc, *desc, ptrTy, tensorTy.getShape());
    if (failed(ptrTensor))
      return rewriter.notifyMatchFailure(op, "cannot build pointer tile");

    rewriter.create<triton::StoreOp>(
        op.getLoc(), *ptrTensor, data, Value(), llvm::ArrayRef<int32_t>{},
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    rewriter.eraseOp(op);
    return success();
  }
};
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

// Builds the pointer tile a 1:N descriptor describes, for the consumer's own
// transfer `shape`. No view op is traced and no layout is read from a type.
//
// The descriptor is expanded into per-dimension pointer arithmetic. For a 2-D
// tile with offset `off` and strides [stride0, stride1] this is the equivalent
// of:
//
//   row     = make_range(0, M) -> tensor<Mx1xi32>
//   row_off = row * stride0 + off
//   row_ptr = addptr(splat(base) : tensor<Mx1xptr>, row_off)
//   col     = make_range(0, N) -> tensor<1xNxi32>
//   ptrs    = addptr(broadcast(row_ptr) : tensor<MxNxptr>,
//                    broadcast(col * stride1) : tensor<MxNxi32>)
//
// The result is a full tensor<...x!tt.ptr<T>> tile for tt.load/tt.store; only
// the construction is staged per dimension, so Triton sees the sliced
// row-pointer form instead of one fully-materialized offset tile.
//
// Offset and strides go to buildDimOffsetTerm as OpFoldResult so that a
// descriptor field which is already an arith.constant - what the view
// producers emit for a static layout - drops the multiply-by-1 or add-0
// entirely rather than emitting arithmetic over a constant.
FailureOr<Value>
buildDescriptorPointerTensor(ConversionPatternRewriter &rewriter, Location loc,
                             const hivm::MemRefDescriptor &desc, Type ptrTy,
                             ArrayRef<int64_t> shape) {
  if (shape.empty() || desc.getRank() != shape.size())
    return failure();

  int64_t packed = 1;
  bool isPacked = isConstantIntValue(desc.offset, 0);
  for (int i = shape.size() - 1; i >= 0 && isPacked; --i) {
    isPacked = isConstantIntValue(desc.strides[i], packed);
    packed *= shape[i];
  }
  if (isPacked) {
    int64_t numElements = packed;
    auto i32Ty = rewriter.getI32Type();
    if (numElements > 0 &&
        numElements <= std::numeric_limits<int32_t>::max()) {
      Value flat = rewriter.create<triton::MakeRangeOp>(
          loc, RankedTensorType::get({numElements}, i32Ty), 0, numElements);
      Value index = flat;
      if (shape.size() > 1)
        index = rewriter.create<triton::ReshapeOp>(
            loc, RankedTensorType::get(shape, i32Ty), flat, false);
      auto ptrTensorTy = RankedTensorType::get(shape, ptrTy);
      Value splat =
          rewriter.create<triton::SplatOp>(loc, ptrTensorTy, desc.basePtr());
      return rewriter
          .create<triton::AddPtrOp>(loc, ptrTensorTy, splat, index)
          .getResult();
    }
  }


  Value ptrs;
  SmallVector<int64_t> currentShape;
  for (auto [dim, stride] : llvm::enumerate(desc.strides)) {
    std::optional<OpFoldResult> baseOffset;
    if (dim == 0)
      baseOffset = OpFoldResult(desc.offset);

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
      Value splat =
          rewriter.create<triton::SplatOp>(loc, ptrTensorTy, desc.basePtr());
      ptrs = rewriter.create<triton::AddPtrOp>(loc, ptrTensorTy, splat, term)
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

// Convert hivm.load op into Triton arithmetic and memory ops.
// Supported Conversion Scenarios:
// 1. Loads data from source `memref` into Triton registers using `tt.load`.
// 2. Address calculation supports contiguous memory (fast-path) and
//    N-D Strided Layout via the MemRef descriptor's strides.
// 3. Constraints: The loaded data must be solely consumed by
// `bufferization.to_tensor`.
//    If other consumers exist, the conversion checks and will emit an error.
//    Otherwise, it replaces the `to_tensor` users with the loaded Triton tensor
//    directly.
class HIVMLoadOpPattern : public OpConversionPattern<hivm::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpConversionPattern<hivm::LoadOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(hivm::LoadOp op, OneToNOpAdaptor nAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto dst = op.getDst();
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
    Value other = onlyValue(nAdaptor.getPadValue());

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

    FailureOr<hivm::MemRefDescriptor> srcDesc =
        hivm::getMemRefDescriptor(rewriter, loc, srcMemrefTy,
                                  nAdaptor.getSrc());
    if (failed(srcDesc) || srcDesc->getRank() != shape.size())
      return rewriter.notifyMatchFailure(op, "no descriptor for source");
    FailureOr<Value> maybeSrcPtrs = buildDescriptorPointerTensor(
        rewriter, loc, *srcDesc, HIVMToTritonTypeConvert(srcMemrefTy), shape);
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
      FailureOr<hivm::MemRefDescriptor> dstDesc =
          hivm::getMemRefDescriptor(rewriter, loc, dstMemrefTy,
                                  nAdaptor.getDst());
      if (failed(dstDesc) || dstDesc->getRank() != shape.size())
        return rewriter.notifyMatchFailure(op, "no descriptor for destination");
      FailureOr<Value> maybeDstPtrs = buildDescriptorPointerTensor(
          rewriter, loc, *dstDesc, HIVMToTritonTypeConvert(dstMemrefTy), shape);
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
  using OpConversionPattern<hivm::StoreOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(hivm::StoreOp op, OneToNOpAdaptor nAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    bool srcIsMemRef = isa<MemRefType>(op.getSrc().getType());
    Value src = srcIsMemRef ? Value() : onlyValue(nAdaptor.getSrc());
    auto dst = op.getDst();

    // === Branch 1: Atomic store ===
    if (op.getAtomicKind()) {
      auto rmwOp = toTritonRMWOp(op.getAtomicKind().value());
      if (!rmwOp)
        return rewriter.notifyMatchFailure(op, "unsupported atomic kind");

      Value storeVal = src;
      if (auto srcMemrefTy = dyn_cast<MemRefType>(op.getSrc().getType())) {
        SmallVector<int64_t> srcShape(srcMemrefTy.getShape().begin(),
                                      srcMemrefTy.getShape().end());
        FailureOr<hivm::MemRefDescriptor> srcDesc =
            hivm::getMemRefDescriptor(rewriter, loc, srcMemrefTy,
                                  nAdaptor.getSrc());
        if (failed(srcDesc))
          return rewriter.notifyMatchFailure(op, "no descriptor for source");
        Type srcPtrTy = HIVMToTritonTypeConvert(srcMemrefTy);
        FailureOr<Value> maybeSrcPtrs = buildDescriptorPointerTensor(
            rewriter, loc, *srcDesc, srcPtrTy, srcShape);
        if (failed(maybeSrcPtrs))
          return rewriter.notifyMatchFailure(
              op, "failed to materialize source pointers");

        storeVal = rewriter
                       .create<triton::LoadOp>(
                           loc, *maybeSrcPtrs, Value(), Value(),
                           llvm::ArrayRef<int32_t>{}, std::nullopt,
                           triton::CacheModifier::NONE,
                           triton::EvictionPolicy::NORMAL, false)
                       .getResult();
      }

      auto valTy = cast<RankedTensorType>(storeVal.getType());
      auto shape = valTy.getShape();

      auto dstMemrefTy = cast<MemRefType>(dst.getType());
      FailureOr<hivm::MemRefDescriptor> dstDesc =
          hivm::getMemRefDescriptor(rewriter, loc, dstMemrefTy,
                                  nAdaptor.getDst());
      if (failed(dstDesc))
        return rewriter.notifyMatchFailure(op, "no descriptor for destination");
      Type dstPtrTy = HIVMToTritonTypeConvert(dstMemrefTy);
      FailureOr<Value> maybePtrs = buildDescriptorPointerTensor(
          rewriter, loc, *dstDesc, dstPtrTy, shape);
      if (failed(maybePtrs))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize destination pointers");
      Value ptrs = *maybePtrs;

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
    if (auto srcTensorTy = src ? dyn_cast<RankedTensorType>(src.getType())
                               : RankedTensorType()) {
      auto dstMemrefTy = dyn_cast<MemRefType>(dst.getType());
      if (!dstMemrefTy)
        return failure();

      auto shape = srcTensorTy.getShape();
      FailureOr<hivm::MemRefDescriptor> dstDesc =
          hivm::getMemRefDescriptor(rewriter, loc, dstMemrefTy,
                                  nAdaptor.getDst());
      if (failed(dstDesc) || dstDesc->getRank() != shape.size())
        return rewriter.notifyMatchFailure(op, "no descriptor for destination");
      FailureOr<Value> maybeDstPtrs = buildDescriptorPointerTensor(
          rewriter, loc, *dstDesc, HIVMToTritonTypeConvert(dstMemrefTy), shape);
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
    auto srcMemrefTy = dyn_cast<MemRefType>(op.getSrc().getType());
    auto dstMemrefTy = dyn_cast<MemRefType>(dst.getType());
    if (!srcMemrefTy || !dstMemrefTy)
      return failure();

    // === Branch 3: Plain memref -> tt.load + tt.store ===
    auto shapeOr = resolveStaticTransferShape(srcMemrefTy, dstMemrefTy);
    if (!shapeOr)
      return rewriter.notifyMatchFailure(op, "cannot resolve shape");
    SmallVector<int64_t> shape = *shapeOr;

    // Load from UB
    FailureOr<hivm::MemRefDescriptor> srcDesc3 =
        hivm::getMemRefDescriptor(rewriter, loc, srcMemrefTy,
                                  nAdaptor.getSrc());
    if (failed(srcDesc3))
      return rewriter.notifyMatchFailure(op, "no descriptor for source");
    Type srcPtrTy3 = HIVMToTritonTypeConvert(srcMemrefTy);
    FailureOr<Value> maybeSrcPtrs = buildDescriptorPointerTensor(
        rewriter, loc, *srcDesc3, srcPtrTy3, shape);
    if (failed(maybeSrcPtrs))
      return rewriter.notifyMatchFailure(
          op, "failed to materialize source pointers");
    auto loaded = rewriter.create<triton::LoadOp>(
        loc, *maybeSrcPtrs, Value(), Value(), llvm::ArrayRef<int32_t>{},
        std::nullopt, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, false);

    // Store to GM
    FailureOr<hivm::MemRefDescriptor> dstDesc3 =
        hivm::getMemRefDescriptor(rewriter, loc, dstMemrefTy,
                                  nAdaptor.getDst());
    if (failed(dstDesc3))
      return rewriter.notifyMatchFailure(op, "no descriptor for destination");
    Type dstPtrTy3 = HIVMToTritonTypeConvert(dstMemrefTy);
    FailureOr<Value> maybeDstPtrs3 = buildDescriptorPointerTensor(
        rewriter, loc, *dstDesc3, dstPtrTy3, shape);
    if (failed(maybeDstPtrs3))
      return rewriter.notifyMatchFailure(
          op, "failed to materialize destination pointers");
    Value dstPtrs = *maybeDstPtrs3;

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

// Convert hivm.hir.cumsum to tt.scan {add}
// Before:
// %cumsum = hivm.hir.vcumsum ins(%0) outs(%0) cum_dims=[0] reverse = false -> tensor<100xf32>

// After:
// %cumsum = "tt.scan" (%0) <{axis=0:i32, reverse = false}> ({
//     ^bb0(%1,%2):
//      %3 = arith,addf %1,%2
//      tt.scan.return %3
// }): tensor<100xf32> -> tensor<100xf32>
struct HIVMToTTScanOp : public OpRewritePattern<hivm::VCumsumOp> {
    using OpRewritePattern<hivm::VCumsumOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(hivm::VCumsumOp op,
                                  PatternRewriter &rewriter) const final {
      auto loc = op.getLoc();
      Value src = op.getSrc();

      if (isa<MemRefType>(src.getType())) {
          return op.emitOpError("memref source is not supported currently");
      }
      auto srcType = cast<RankedTensorType>(src.getType());
      auto elemType = srcType.getElementType();

      auto cumDims = op.getCumDims();
      if (cumDims.empty()) {
          return failure();
      }

      bool reverse = op.getReverse();

      Value finalResult = src;

      for (auto axis64 : cumDims) {
        int axis = static_cast<int>(axis64);

        auto scanOp = rewriter.create<triton::ScanOp>(
            loc, ValueRange{finalResult}, axis, reverse);

        Region &combineRegion = scanOp.getCombineOp();
        rewriter.createBlock(&combineRegion);
        Block &block = combineRegion.front();
        block.addArgument(elemType, loc);
        block.addArgument(elemType, loc);

        rewriter.setInsertionPointToEnd(&block);
        Value arg0 = block.getArgument(0);
        Value arg1 = block.getArgument(1);
        Value addResult;

        if (isa<FloatType>(elemType)) {
            addResult = rewriter.create<arith::AddFOp>(loc, arg0, arg1);
        } else {
            addResult = rewriter.create<arith::AddIOp>(loc, arg0, arg1);
        }

        rewriter.create<triton::ScanReturnOp>(loc, addResult);
        rewriter.setInsertionPointAfter(scanOp);

        finalResult = scanOp->getResult(0);
      }

      rewriter.replaceOp(op, finalResult);
      return success();
    }
};

} // namespace

FailureOr<Value> mlir::hivm::buildMemRefDescriptorPointers(
    ConversionPatternRewriter &rewriter, Location loc,
    const MemRefDescriptor &desc, Type ptrTy, ArrayRef<int64_t> shape) {
  return buildDescriptorPointerTensor(rewriter, loc, desc, ptrTy, shape);
}

void mlir::hivm::populateHIVMToTritonPatterns(TritonTypeConverter &converter,
                                              RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<HIVMLoalLoadOpPattern, HIVMLoalStoreOpPattern,
               GatherLoadOpPattern, ScatterStoreOpPattern, HIVMLoadOpPattern,
               HIVMStoreOpPattern>(converter, context);

  patterns.add<GetBlockIdxOpPattern, VArangeOpPattern, VBrcOpPattern,
               HIVMToTTReduceOp, HIVMToTTScanOp>(context);
}

//===- HIVMToTriton.cpp - conversion from HIVM to Triton dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <numeric>

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::triton;

namespace {
// Convert hivm.hir.gather_load op into tt.load, for example:
// Before:
//  %1 = hivm.hir.gather_load (%base, %indices, %burst_len)
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
    auto *context = op.getContext();
    auto boundary =
        adaptor.getBoundaryCheck().value_or(llvm::ArrayRef<int32_t>{});
    triton::PaddingOptionAttr padding;
    if (auto res = adaptor.getPaddingAttr()) {
      auto v = static_cast<triton::PaddingOption>(res.getValue());
      padding = triton::PaddingOptionAttr::get(context, v);
    }
    auto cache = triton::CacheModifier::NONE;
    if (auto res = adaptor.getCacheAttr()) {
      cache = static_cast<triton::CacheModifier>(res.getValue());
    }
    auto evict = triton::EvictionPolicy::NORMAL;
    if (auto res = adaptor.getEvictAttr()) {
      evict = static_cast<triton::EvictionPolicy>(res.getEvictionpolicy());
    }
    auto isVolatile = false;
    if (auto res = adaptor.getIsVolatile()) {
      isVolatile = res.value();
    }
    auto load = rewriter.create<triton::LoadOp>(
        loc, addptr.getResult(), adaptor.getMask(), adaptor.getOther(),
        boundary, padding, cache, evict, isVolatile);
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

    // Generate tt.make_range operator to get continuous sequence
    auto num = tensorTy.getNumElements();
    auto rangeTensorTy = RankedTensorType::get({num}, rewriter.getI32Type());
    auto mkrng = rewriter.create<triton::MakeRangeOp>(op.getLoc(),
                                                      rangeTensorTy, 0, num);

    mlir::Value offset = mkrng;
    // Generate tt.reshape operator to get multi-dim continuous sequence tensor
    // of target shape if needed
    if (tensorTy.getRank() > 1) {
      auto reshapeTensorTy = RankedTensorType::get(
          tensorTy.getShape(), rangeTensorTy.getElementType());
      auto reshape = rewriter.create<triton::ReshapeOp>(op.getLoc(),
                                                        reshapeTensorTy, mkrng);
      offset = reshape;
    }

    auto ttPtr = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, addr);

    // Generate tt.splat operator to get pointer tensor of target shape
    auto ptrTensor = RankedTensorType::get(tensorTy.getShape(), ptrTy);
    auto splat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTensor,
                                                  ttPtr.getResult(0));

    // Generate tt.addptr operator to get pointer tensor with offset
    auto addptr = rewriter.create<triton::AddPtrOp>(op.getLoc(), ptrTensor,
                                                    splat, offset);

    // Generate tt.load operator to get value from pointer tensor
    auto valTensor = rewriter.create<triton::LoadOp>(
        op.getLoc(), tensorTy, addptr, Value{}, Value{},
        llvm::ArrayRef<int32_t>{}, triton::PaddingOptionAttr{});

    rewriter.replaceOp(op, valTensor);
    return success();
  }
};

// Convert hivm.hir.scatter_store op into tt.store, for example:
// Before:
//  %1 = hivm.hir.scatter_store (%base, %indices, %data, %burst_len)
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
    auto boundaryCheck = adaptor.getBoundaryCheck().value_or(llvm::ArrayRef<int32_t>{});
    auto cache = triton::CacheModifier::NONE;
    if (auto res = adaptor.getCacheAttr()) {
      cache = static_cast<triton::CacheModifier>(res.getValue());
    }
    auto evict = triton::EvictionPolicy::NORMAL;
    if (auto res = adaptor.getEvictAttr()) {
      evict = static_cast<triton::EvictionPolicy>(res.getEvictionpolicy());
    }
    auto storeOp = rewriter.create<triton::StoreOp>(
        loc, addptr.getResult(), adaptor.getData(), adaptor.getMask(),
        boundaryCheck, cache, evict);
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

    // Generate tt.make_range operator to get continuous sequence
    auto tensorTy = data.getType();
    auto num = tensorTy.getNumElements();
    auto rangeTensorTy = RankedTensorType::get({num}, rewriter.getI32Type());
    auto mkrng = rewriter.create<triton::MakeRangeOp>(op.getLoc(),
                                                      rangeTensorTy, 0, num);

    mlir::Value offset = mkrng;
    // Generate tt.reshape operator to get multi-dim continuous sequence tensor
    // of target shape if needed
    if (tensorTy.getRank() > 1) {
      auto reshapeTensorTy = RankedTensorType::get(
          tensorTy.getShape(), rangeTensorTy.getElementType());
      auto reshape = rewriter.create<triton::ReshapeOp>(op.getLoc(),
                                                        reshapeTensorTy, mkrng);
      offset = reshape;
    }

    auto ttPtr = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, addr);

    // Generate tt.splat operator to get pointer tensor of target shape
    auto ptrTensor = RankedTensorType::get(tensorTy.getShape(), ptrTy);
    auto splat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTensor,
                                                  ttPtr.getResult(0));

    // Generate tt.addptr operator to get pointer tensor with offset
    auto addptr = rewriter.create<triton::AddPtrOp>(op.getLoc(), ptrTensor,
                                                    splat, offset);

    // Generate tt.store operator to set value to pointer tensor
    auto storeOp =
        rewriter.create<triton::StoreOp>(op.getLoc(), addptr, data, Value());

    rewriter.replaceOp(op, storeOp);
    return success();
  }
};

// Convert hivm.load op into tt.make_range + tt.addptr + tt.load + tt.addptr + tt.store.
// If 'dst' is provided, it also generates a tt.store (copy behavior).
// Before:
// hivm.hir.load ins(%arg0 : memref<1024xf16>) outs(%arg1 : memref<1024xf16>)
// After:
//     %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
//     %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
//     %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
//     %3 = tt.load %2 : tensor<1024x!tt.ptr<f16>>
//     %4 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
//     %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
//     tt.store %5, %3 : tensor<1024x!tt.ptr<f16>>
class HIVMLoadOpPattern : public OpConversionPattern<hivm::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hivm::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto dst = op.getDst();

    // 1. Determine the shape of the load operation
    ArrayRef<int64_t> shape;
    if (op.getResultTensor()) {
      auto resultTy =
          dyn_cast<RankedTensorType>(op.getResultTensor().getType());
      if (!resultTy)
        return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
      shape = resultTy.getShape();
    } else {
      // For void returns, infer shape from src or dst MemRef
      if (auto srcTy = dyn_cast<MemRefType>(src.getType());
          srcTy && srcTy.hasStaticShape()) {
        shape = srcTy.getShape();
      } else if (auto dstTy = dyn_cast<MemRefType>(dst.getType());
                 dstTy && dstTy.hasStaticShape()) {
        shape = dstTy.getShape();
      } else {
        return rewriter.notifyMatchFailure(op, "cannot infer shape for load");
      }
    }

    // 2. Convert source to Triton pointer type
    Type srcPtrTy;
    if (auto memRefTy = dyn_cast<MemRefType>(src.getType())) {
      srcPtrTy = HIVMToTritonTypeConvert(memRefTy);
    } else {
      return rewriter.notifyMatchFailure(op, "src must be memref");
    }

    // 3. Generate linear indices using make_range and optional reshape
    int64_t numElements = std::accumulate(shape.begin(), shape.end(), 1LL,
                                          std::multiplies<int64_t>());

    Value range = rewriter.create<triton::MakeRangeOp>(
        loc, RankedTensorType::get({numElements}, rewriter.getI32Type()), 0,
        numElements);

    Value indices = range;
    if (shape.size() != 1 || shape[0] != numElements) {
      indices = rewriter.create<triton::ReshapeOp>(
          loc, RankedTensorType::get(shape, range.getType().mlir::cast<RankedTensorType>().getElementType()), range);
    }

    // 4. Create pointer tensor (Splat + AddPtr)
    auto ttSrcBase =
        rewriter.create<UnrealizedConversionCastOp>(loc, srcPtrTy, src);
    auto srcPtrTensorTy = RankedTensorType::get(shape, srcPtrTy);
    auto srcSplat = rewriter.create<triton::SplatOp>(loc, srcPtrTensorTy,
                                                     ttSrcBase.getResult(0));
    auto srcPtrTensor = rewriter.create<triton::AddPtrOp>(loc, srcPtrTensorTy,
                                                          srcSplat, indices);

    // 5. Extract Load attributes
    auto evict = triton::EvictionPolicy::NORMAL;
    if (auto evictAttr = op.getEvictionPolicy()) {
      switch (evictAttr->getEvictionpolicy()) {
      case hivm::EvictionPolicy::EvictFirst:
        evict = triton::EvictionPolicy::EVICT_FIRST;
        break;
      case hivm::EvictionPolicy::EvictLast:
        evict = triton::EvictionPolicy::EVICT_LAST;
        break;
      }
    }

    Value other = adaptor.getPadValue();

    // 6. Create tt.load
    auto loadOp = rewriter.create<triton::LoadOp>(
        loc, srcPtrTensor, Value(), other, llvm::ArrayRef<int32_t>{},
        std::nullopt, triton::CacheModifier::NONE, evict, false);

    // 7. Handle optional destination (Store if dst exists)
    if (dst) {
      Type dstPtrTy;
      if (auto memRefTy = dyn_cast<MemRefType>(dst.getType())) {
        dstPtrTy = HIVMToTritonTypeConvert(memRefTy);
      } else {
        return rewriter.notifyMatchFailure(op, "dst must be memref");
      }

      auto ttDstBase =
          rewriter.create<UnrealizedConversionCastOp>(loc, dstPtrTy, dst);
      auto dstPtrTensorTy = RankedTensorType::get(shape, dstPtrTy);
      auto dstSplat = rewriter.create<triton::SplatOp>(loc, dstPtrTensorTy,
                                                       ttDstBase.getResult(0));
      auto dstPtrTensor = rewriter.create<triton::AddPtrOp>(loc, dstPtrTensorTy,
                                                            dstSplat, indices);

      rewriter.create<triton::StoreOp>(
          loc, dstPtrTensor, loadOp.getResult(), /*mask=*/Value(),
          /*boundaryCheck=*/llvm::ArrayRef<int32_t>{},
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    }

    // 8. Replace or erase the original op
    if (op.getResultTensor()) {
      rewriter.replaceOp(op, loadOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// Convert hivm.store op into tt.make_range + tt.addptr + tt.store.
// Before:
//  hivm.hir.store ins(%arg0 : tensor<1024xf16>) outs(%arg1 : memref<1024xf16>)
// After:
// %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
// %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
// %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
// tt.store %2, %arg0 : tensor<1024x!tt.ptr<f16>>
class HIVMStoreOpPattern : public OpConversionPattern<hivm::StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hivm::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = adaptor.getSrc();
    auto dst = adaptor.getDst();

    auto srcTensorTy = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTensorTy) {
      return rewriter.notifyMatchFailure(op, "src must be a ranked tensor");
    }

    Type dstPtrTy;
    if (auto memRefTy = dyn_cast<MemRefType>(dst.getType())) {
      dstPtrTy = HIVMToTritonTypeConvert(memRefTy);
    } else {
      return rewriter.notifyMatchFailure(op, "dst must be memref");
    }

    auto shape = srcTensorTy.getShape();
    int64_t numElements = std::accumulate(shape.begin(), shape.end(), 1LL,
                                          std::multiplies<int64_t>());

    Value range = rewriter.create<triton::MakeRangeOp>(
        loc, RankedTensorType::get({numElements}, rewriter.getI32Type()), 0,
        numElements);

    Value indices = range;
    if (shape.size() != 1 || shape[0] != numElements) {
      indices = rewriter.create<triton::ReshapeOp>(
          loc, RankedTensorType::get(shape, range.getType().mlir::cast<RankedTensorType>().getElementType()), range);
    }

    auto ttDstBase =
        rewriter.create<UnrealizedConversionCastOp>(loc, dstPtrTy, dst);
    auto ptrTensorTy = RankedTensorType::get(shape, dstPtrTy);
    auto splat = rewriter.create<triton::SplatOp>(loc, ptrTensorTy,
                                                  ttDstBase.getResult(0));
    auto dstPtrTensor =
        rewriter.create<triton::AddPtrOp>(loc, ptrTensorTy, splat, indices);

    rewriter.create<triton::StoreOp>(
        loc, dstPtrTensor, src, Value(), llvm::ArrayRef<int32_t>{},
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::hivm::populateHIVMToTritonPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns
      .add<GatherLoadOpPattern, ScatterStoreOpPattern, HIVMLoadOpPattern,
           HIVMStoreOpPattern, HIVMLoalLoadOpPattern, HIVMLoalStoreOpPattern>(
          context);
}

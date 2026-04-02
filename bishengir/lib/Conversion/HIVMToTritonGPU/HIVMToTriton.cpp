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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

    // Guard: padded loads cannot be reversed to plain tt.load
    if (op.getPadMode())
      return rewriter.notifyMatchFailure(op, "padded load not converted");

    // // === Branch : Tensor form === delete

    // === Branch 1: Memref form ===
    auto srcMemrefTy = dyn_cast<MemRefType>(src.getType());
    auto dstMemrefTy = dyn_cast<MemRefType>(dst.getType());
    if (!srcMemrefTy || !dstMemrefTy)
      return failure();

    auto shapeOr = resolveStaticTransferShape(srcMemrefTy, dstMemrefTy);
    if (!shapeOr)
      return rewriter.notifyMatchFailure(op, "cannot resolve shape");
    SmallVector<int64_t> shape = *shapeOr;

    Value linearOffsets = createLinearOffsetTensor(rewriter, loc, shape);

    Value srcOffsets =
        calcStridedOffsets(rewriter, loc, srcMemrefTy, shape, linearOffsets);

    Value srcPtrs =
        buildTensorPointers(rewriter, loc, src, srcMemrefTy, shape, srcOffsets);

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
      Value dstOffsets =
          calcStridedOffsets(rewriter, loc, dstMemrefTy, shape, linearOffsets);

      Value dstPtrs = buildTensorPointers(rewriter, loc, dst, dstMemrefTy,
                        shape, dstOffsets);

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
      Value linearOffsets = createLinearOffsetTensor(rewriter, loc, shape);

      Value dstOffsets =
          calcStridedOffsets(rewriter, loc, dstMemrefTy, shape, linearOffsets);

      Value dstPtrs = buildTensorPointers(rewriter, loc, dst, dstMemrefTy,
                                          shape, dstOffsets);

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
} // namespace

void mlir::hivm::populateHIVMToTritonPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns
      .add<GatherLoadOpPattern, ScatterStoreOpPattern, HIVMLoadOpPattern,
           HIVMStoreOpPattern, HIVMLoalLoadOpPattern, HIVMLoalStoreOpPattern>(
          context);
}

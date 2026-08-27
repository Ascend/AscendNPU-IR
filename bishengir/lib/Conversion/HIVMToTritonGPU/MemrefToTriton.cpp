//===------------- Conversion from memref ops to Triton dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Conversion/HIVMToTritonGPU/MemRefDescriptor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;

namespace {
// Materializes a layout value as an i64. Offsets and strides are kept in i64 to
// match the width the MemRef layout stores them in and the widest offset
// tt.addptr accepts.
static Value materializeI64(ConversionPatternRewriter &rewriter, Location loc,
                            OpFoldResult ofr) {
  auto i64Ty = rewriter.getI64Type();
  if (std::optional<int64_t> constant = getConstantIntValue(ofr))
    return rewriter.create<arith::ConstantIntOp>(loc, *constant, 64);

  Value value = cast<Value>(ofr);
  if (isa<IndexType>(value.getType()))
    return rewriter.createOrFold<arith::IndexCastOp>(loc, i64Ty, value);
  if (value.getType() != i64Ty)
    return rewriter.create<arith::ExtSIOp>(loc, i64Ty, value);
  return value;
}

// memref.reinterpret_cast RESETS the offset: the result's element 0 sits at
// `source base + castOffset`, regardless of any offset the source already
// carried. So the descriptor keeps the root pointer and OVERWRITES the offset.
class ReinterpretCastOpConversion
    : public OpConversionPattern<memref::ReinterpretCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpConversionPattern<memref::ReinterpretCastOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<hivm::MemRefDescriptor> src =
        hivm::getMemRefDescriptor(
        rewriter, op.getLoc(), cast<MemRefType>(op.getSource().getType()),
        adaptor.getSource());
    if (failed(src))
      return rewriter.notifyMatchFailure(op, "source is not a descriptor");

    Location loc = op.getLoc();
    hivm::MemRefDescriptor result;
    // Root pointers are inherited; the source's own offset is discarded.
    result.allocPtr = src->allocPtr;
    result.alignedPtr = src->alignedPtr;
    result.offset = materializeI64(rewriter, loc, op.getMixedOffsets().front());
    // reinterpret_cast restates the whole shape, so sizes come from the op
    // rather than being composed onto the source's.
    for (OpFoldResult size : op.getMixedSizes())
      result.sizes.push_back(materializeI64(rewriter, loc, size));
    for (OpFoldResult stride : op.getMixedStrides())
      result.strides.push_back(materializeI64(rewriter, loc, stride));

    SmallVector<Value> flat = result.flatten();
    rewriter.replaceOpWithMultiple(op, {ValueRange(flat)});
    return success();
  }
};

// memref.subview COMPOSES onto its source:
//   offset += sum_d subviewOffset[d] * sourceStride[d]
//   stride[d] *= subviewStride[d]
// Dimensions dropped by rank reduction contribute to the offset but leave no
// stride entry, so the descriptor's stride count matches the RESULT rank.
class SubViewOpConversion : public OpConversionPattern<memref::SubViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpConversionPattern<memref::SubViewOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<hivm::MemRefDescriptor> src =
        hivm::getMemRefDescriptor(
        rewriter, op.getLoc(), cast<MemRefType>(op.getSource().getType()),
        adaptor.getSource());
    if (failed(src))
      return rewriter.notifyMatchFailure(op, "source is not a descriptor");

    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> subStrides = op.getMixedStrides();
    if (offsets.size() != src->getRank() ||
        subStrides.size() != src->getRank())
      return rewriter.notifyMatchFailure(op, "rank mismatch with source");

    // Which source dimensions survive into the result.
    llvm::SmallBitVector dropped = op.getDroppedDims();

    SmallVector<OpFoldResult> subSizes = op.getMixedSizes();

    Location loc = op.getLoc();
    hivm::MemRefDescriptor result;
    result.allocPtr = src->allocPtr;
    result.alignedPtr = src->alignedPtr;
    result.offset = src->offset;

    for (unsigned d = 0, e = src->getRank(); d < e; ++d) {
      Value srcStride = src->strides[d];
      // offset += offsets[d] * srcStride, skipping the identity term.
      if (!isConstantIntValue(offsets[d], 0)) {
        Value off = materializeI64(rewriter, loc, offsets[d]);
        Value term = rewriter.createOrFold<arith::MulIOp>(loc, off, srcStride);
        result.offset =
            rewriter.createOrFold<arith::AddIOp>(loc, result.offset, term);
      }
      if (dropped.test(d))
        continue;
      // The subview's own extent replaces the source's for surviving dims.
      result.sizes.push_back(materializeI64(rewriter, loc, subSizes[d]));
      Value stride = srcStride;
      if (!isConstantIntValue(subStrides[d], 1)) {
        Value sub = materializeI64(rewriter, loc, subStrides[d]);
        stride = rewriter.createOrFold<arith::MulIOp>(loc, stride, sub);
      }
      result.strides.push_back(stride);
    }

    SmallVector<Value> flat = result.flatten();
    rewriter.replaceOpWithMultiple(op, {ValueRange(flat)});
    return success();
  }
};

// Reads the descriptor for one MemRef operand and returns the address of the
// element at `indices`:  basePtr + offset + sum_d indices[d] * strides[d].
// Everything comes from the adaptor - no view op is traced, no layout is read.
static FailureOr<Value>
buildScalarElementPtr(ConversionPatternRewriter &rewriter, Location loc,
                      MemRefType memrefTy, ValueRange descriptorValues,
                      Type ptrTy, ValueRange indices) {
  FailureOr<hivm::MemRefDescriptor> desc =
      hivm::getMemRefDescriptor(rewriter, loc, memrefTy, descriptorValues);
  if (failed(desc))
    return failure();
  if (desc->getRank() != indices.size())
    return failure();

  Value linear = desc->offset;
  for (auto [stride, index] : llvm::zip_equal(desc->strides, indices)) {
    if (isConstantIntValue(index, 0))
      continue;
    Value idx = materializeI64(rewriter, loc, index);
    Value term = rewriter.createOrFold<arith::MulIOp>(loc, idx, stride);
    linear = rewriter.createOrFold<arith::AddIOp>(loc, linear, term);
  }

  return rewriter.createOrFold<triton::AddPtrOp>(loc, ptrTy, desc->basePtr(),
                                                 linear);
}

// Convert `memref.load %memref[%idx0, ...]` into a scalar tt.load.
class MemRefLoadOpPattern : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpConversionPattern<memref::LoadOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptrTy = hivm::HIVMToTritonTypeConvert(op.getMemRefType());
    FailureOr<Value> addr = buildScalarElementPtr(
        rewriter, op.getLoc(), op.getMemRefType(), adaptor.getMemref(), ptrTy,
        op.getIndices());
    if (failed(addr))
      return rewriter.notifyMatchFailure(op, "no descriptor for source");

    auto load = rewriter.create<triton::LoadOp>(
        op.getLoc(), *addr, Value(), Value(), llvm::ArrayRef<int32_t>{},
        std::nullopt, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, false);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

// Convert `memref.store %val, %memref[...]` into a scalar tt.store.
class MemRefStoreOpPattern : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpConversionPattern<memref::StoreOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptrTy = hivm::HIVMToTritonTypeConvert(op.getMemRefType());
    FailureOr<Value> addr = buildScalarElementPtr(
        rewriter, op.getLoc(), op.getMemRefType(), adaptor.getMemref(), ptrTy,
        op.getIndices());
    if (failed(addr))
      return rewriter.notifyMatchFailure(op, "no descriptor for destination");

    rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, *addr, getOneToOneAdaptorOperands(adaptor.getValue()).front(),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    return success();
  }
};

// Convert `memref.extract_aligned_pointer_as_index %src` into:
//   %addr64  = tt.ptr_to_int <aligned ptr> : !tt.ptr<...> -> i64
//   %addridx = arith.index_cast %addr64 : i64 to index
//
// The op is used to inspect whether an optional pointer argument is null.
// Stage1 must lower it so Stage2 (FuncOpPattern) no longer sees a memref op
// referencing a memref-typed block argument.
class ExtractAlignedPointerAsIndexOpPattern
    : public OpConversionPattern<memref::ExtractAlignedPointerAsIndexOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpConversionPattern<
      memref::ExtractAlignedPointerAsIndexOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(memref::ExtractAlignedPointerAsIndexOp op,
                  OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    FailureOr<hivm::MemRefDescriptor> desc = hivm::getMemRefDescriptor(
        rewriter, loc, cast<MemRefType>(op.getSource().getType()),
        adaptor.getSource());
    if (failed(desc))
      return rewriter.notifyMatchFailure(op, "source is not a descriptor");

    Value addr = rewriter.create<triton::PtrToIntOp>(loc, rewriter.getI64Type(),
                                                     desc->alignedPtr);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, rewriter.getIndexType(),
                                                    addr);
    return success();
  }
};
} // namespace

void mlir::hivm::populateMemRefToTritonPatterns(TritonTypeConverter &converter,
                                                RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  // The view producers rewrite the descriptor in place so offsets compose; the
  // scalar accesses read the composed result straight off it.
  patterns.add<ReinterpretCastOpConversion, SubViewOpConversion,
               MemRefLoadOpPattern, MemRefStoreOpPattern,
               ExtractAlignedPointerAsIndexOpPattern>(converter, ctx);
}

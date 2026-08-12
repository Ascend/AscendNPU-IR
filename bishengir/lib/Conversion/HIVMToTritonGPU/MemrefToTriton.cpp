//===------------- Conversion from memref ops to Triton dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;

namespace {
// Reinterpret-cast view semantics are consumed by the memory-access lowering
// patterns.  Keep the cast legal during dialect conversion by preserving its
// base pointer and dynamic layout operands in an unrealized cast.
class ReinterpretCastOpReplacementPattern
    : public OpConversionPattern<memref::ReinterpretCastOp> {

public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<Value> inputs{op.getSource()};
    llvm::append_range(inputs, op.getOffsets());
    llvm::append_range(inputs, op.getStrides());
    auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getResult().getType(), inputs);
    rewriter.replaceOp(op, unrealizedCast.getResult(0));
    return success();
  }
};

class SubViewOpReplacementPattern
    : public OpConversionPattern<memref::SubViewOp> {

public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    // When every user is a memref::LoadOp the scalar-load pattern will pick
    // up the subview parameters (offsets / strides) itself.  Create a simple
    // source→ptr cast so that the load pattern can treat the cast result as
    // the base pointer.  This avoids the irreconcilable memref→memref cast
    // that the generic path would produce.
    if (llvm::all_of(op->getUsers(),
                     [](Operation *user) { return isa<memref::LoadOp>(user); })) {
      Value source = op.getSource();
      MemRefType sourceMemrefTy = op.getSourceType();

      // Look through a zero-offset parent reinterpret_cast.
      if (auto parentCast = source.getDefiningOp<memref::ReinterpretCastOp>()) {
        auto staticOffsets = parentCast.getStaticOffsets();
        if (staticOffsets.size() == 1 && staticOffsets.front() == 0) {
          source = parentCast.getSource();
          sourceMemrefTy = cast<MemRefType>(source.getType());
        }
      }

      Type ptrTy = hivm::HIVMToTritonTypeConvert(sourceMemrefTy);
      auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
          op.getLoc(), ptrTy, source);
      rewriter.replaceOp(op, unrealizedCast.getResult(0));
      return success();
    }

    SmallVector<Value> inputs{op.getSource()};
    llvm::append_range(inputs, op.getOffsets());
    llvm::append_range(inputs, op.getSizes());
    llvm::append_range(inputs, op.getStrides());
    auto unrealizedCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getResult().getType(), inputs);
    rewriter.replaceOp(op, unrealizedCast.getResult(0));
    return success();
  }
};

// Helper: given a SubViewOp, compute the scalar Triton pointer and linear
// offset for a memref.load that reads from the subview.  Returns the base
// pointer (as a !tt.ptr value) and sets `linearOffset` to the sum of the
// subview's base offset and the load indices contribution.
static FailureOr<Value>
buildSubviewLoadPointer(ConversionPatternRewriter &rewriter, Location loc,
                        memref::SubViewOp subviewOp,
                        ArrayRef<Value> loadIndices, Value &linearOffset) {
  auto i64Ty = rewriter.getI64Type();
  auto sourceMemrefTy = subviewOp.getSourceType();

  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset;
  if (failed(getStridesAndOffset(sourceMemrefTy, sourceStrides, sourceOffset)))
    return failure();
  if (llvm::is_contained(sourceStrides, ShapedType::kDynamic))
    return failure();

  // Look through a zero-offset parent reinterpret_cast.
  Value base = subviewOp.getSource();
  MemRefType baseMemrefTy = sourceMemrefTy;
  if (auto parentCast =
          base.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto staticOffsets = parentCast.getStaticOffsets();
    if (staticOffsets.size() == 1 && staticOffsets.front() == 0) {
      base = parentCast.getSource();
      baseMemrefTy = cast<MemRefType>(base.getType());
    }
  }

  Type ptrTy = hivm::HIVMToTritonTypeConvert(baseMemrefTy);
  auto basePtr = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, base);

  // Compute subview linear offset = sum(subview_offsets[i] * sourceStrides[i]).
  auto subviewOffsets = subviewOp.getMixedOffsets();
  for (size_t i = 0; i < subviewOffsets.size(); ++i) {
    Value off;
    if (auto constVal = getConstantIntValue(subviewOffsets[i]))
      off = rewriter.create<arith::ConstantIntOp>(loc, *constVal, 64);
    else
      off = subviewOffsets[i].get<Value>();
    if (isa<IndexType>(off.getType()))
      off = rewriter.createOrFold<arith::IndexCastOp>(loc, i64Ty, off);
    Value strideVal =
        rewriter.create<arith::ConstantIntOp>(loc, sourceStrides[i], 64);
    Value term = rewriter.create<arith::MulIOp>(loc, off, strideVal);
    linearOffset = linearOffset
                       ? rewriter.create<arith::AddIOp>(loc, linearOffset, term)
                       : term;
  }

  // Add load indices offset = sum(load_indices[i] * subview_strides[i]).
  auto subviewStrides = subviewOp.getMixedStrides();
  for (size_t i = 0; i < loadIndices.size(); ++i) {
    Value idx = loadIndices[i];
    if (isa<IndexType>(idx.getType()))
      idx = rewriter.createOrFold<arith::IndexCastOp>(loc, i64Ty, idx);

    Value strideVal;
    if (auto constVal = getConstantIntValue(subviewStrides[i]))
      strideVal = rewriter.create<arith::ConstantIntOp>(loc, *constVal, 64);
    else
      strideVal = subviewStrides[i].get<Value>();
    if (isa<IndexType>(strideVal.getType()))
      strideVal =
          rewriter.createOrFold<arith::IndexCastOp>(loc, i64Ty, strideVal);

    Value term = rewriter.create<arith::MulIOp>(loc, idx, strideVal);
    linearOffset = linearOffset
                       ? rewriter.create<arith::AddIOp>(loc, linearOffset, term)
                       : term;
  }

  if (!linearOffset)
    linearOffset = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

  return basePtr.getResult(0);
}

// Convert `memref.load %memref[%idx0, %idx1, ...]` into:
//   %ptr   = unrealized_conversion_cast %memref : memref<...> to !tt.ptr<...>
//   %off   = <linear offset computed from indices * strides>
//   %addr  = tt.addptr %ptr, %off
//   %r     = tt.load %addr
//
// The unrealized_conversion_cast bridges the memref type to a Triton pointer
// type so that Stage 2 (FuncOpPattern) can map the memref function argument to
// a !tt.ptr value without leaving an unconvertible memref.load in the body.
// After Stage 2 the cast becomes a no-op (same source and target type) and is
// removed by reconcile-unrealized-casts.
//
// When the memref operand is a memref.subview (either the original SubViewOp
// or the unrealized_conversion_cast produced by SubViewOpReplacementPattern),
// the pointer is computed directly from the subview's source and parameters to
// avoid leaving an irreconcilable intermediate memref-typed cast.
class MemRefLoadOpPattern : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto memrefTy = op.getMemRefType();
    auto i64Ty = rewriter.getI64Type();

    Value convertedMemref = adaptor.getMemref();
    Type ptrTy = hivm::HIVMToTritonTypeConvert(memrefTy);
    Value ptr;
    Value linearOffset;

    // Case 1: The memref operand is still the original memref::SubViewOp
    // (load pattern runs before SubViewOpReplacementPattern).
    if (auto subviewOp = convertedMemref.getDefiningOp<memref::SubViewOp>()) {
      SmallVector<Value> indices(op.getIndices().begin(),
                                 op.getIndices().end());
      auto result = buildSubviewLoadPointer(rewriter, loc, subviewOp,
                                            indices, linearOffset);
      if (failed(result))
        return failure();
      ptr = *result;

      Value addr = rewriter.create<triton::AddPtrOp>(loc, ptrTy, ptr,
                                                     linearOffset);
      auto load = rewriter.create<triton::LoadOp>(
          loc, addr, Value(), Value(), llvm::ArrayRef<int32_t>{}, std::nullopt,
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
      rewriter.replaceOp(op, load.getResult());
      return success();
    }

    // Case 2: The memref operand is the unrealized_conversion_cast created
    // by SubViewOpReplacementPattern.
    //
    // When the subview was consumed only by memref.load ops, the replacement
    // pattern creates a simple source→ptr cast (result is !tt.ptr).  The
    // subview's offset/stride info is not in the cast operands, so we look at
    // the ORIGINAL memref.load operand to find the SubViewOp.
    //
    // When the subview had other consumers, the replacement pattern creates a
    // memref→memref cast with {source, offsets, sizes, strides} operands.
    auto memrefCast =
        convertedMemref.getDefiningOp<UnrealizedConversionCastOp>();
    auto originalSubview =
        op.getMemRef().getDefiningOp<memref::SubViewOp>();

    if (originalSubview && memrefCast) {
      // The SubViewOpReplacementPattern created a source→ptr cast for us.
      // Use the cast result as the base pointer and compute the offset from
      // the original subview parameters.
      ptr = convertedMemref;

      auto sourceMemrefTy = originalSubview.getSourceType();
      SmallVector<int64_t> sourceStrides;
      int64_t sourceOffset;
      if (failed(getStridesAndOffset(sourceMemrefTy, sourceStrides,
                                     sourceOffset)))
        return failure();
      if (llvm::is_contained(sourceStrides, ShapedType::kDynamic))
        return failure();

      // Compute subview linear offset =
      //   sum(subview_offsets[i] * source_strides[i]).
      auto subviewOffsets = originalSubview.getMixedOffsets();
      for (size_t i = 0; i < subviewOffsets.size(); ++i) {
        Value off;
        if (auto constVal = getConstantIntValue(subviewOffsets[i]))
          off = rewriter.create<arith::ConstantIntOp>(loc, *constVal, 64);
        else
          off = subviewOffsets[i].get<Value>();
        if (isa<IndexType>(off.getType()))
          off = rewriter.createOrFold<arith::IndexCastOp>(loc, i64Ty, off);
        Value strideVal =
            rewriter.create<arith::ConstantIntOp>(loc, sourceStrides[i], 64);
        Value term = rewriter.create<arith::MulIOp>(loc, off, strideVal);
        linearOffset = linearOffset
                           ? rewriter.create<arith::AddIOp>(loc, linearOffset,
                                                            term)
                           : term;
      }

      // Add load indices offset =
      //   sum(load_indices[i] * subview_strides[i]).
      auto subviewStrides = originalSubview.getMixedStrides();
      auto indices = op.getIndices();
      for (size_t i = 0; i < indices.size(); ++i) {
        Value idx = indices[i];
        if (isa<IndexType>(idx.getType()))
          idx = rewriter.createOrFold<arith::IndexCastOp>(loc, i64Ty, idx);
        Value strideVal;
        if (auto constVal = getConstantIntValue(subviewStrides[i]))
          strideVal = rewriter.create<arith::ConstantIntOp>(loc, *constVal, 64);
        else
          strideVal = subviewStrides[i].get<Value>();
        if (isa<IndexType>(strideVal.getType()))
          strideVal =
              rewriter.createOrFold<arith::IndexCastOp>(loc, i64Ty, strideVal);
        Value term = rewriter.create<arith::MulIOp>(loc, idx, strideVal);
        linearOffset = linearOffset
                           ? rewriter.create<arith::AddIOp>(loc, linearOffset,
                                                            term)
                           : term;
      }

      if (!linearOffset)
        linearOffset = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

      Value addr = rewriter.create<triton::AddPtrOp>(loc, ptrTy, ptr,
                                                     linearOffset);
      auto load = rewriter.create<triton::LoadOp>(
          loc, addr, Value(), Value(), llvm::ArrayRef<int32_t>{}, std::nullopt,
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
      rewriter.replaceOp(op, load.getResult());
      return success();
    }

    if (memrefCast && mlir::isa<MemRefType>(memrefCast.getResultTypes().front())) {
      auto castResultTy = cast<MemRefType>(memrefCast.getResultTypes().front());
      if (mlir::isa<StridedLayoutAttr>(castResultTy.getLayout())) {
        auto castOperands = memrefCast.getOperands();
        if (castOperands.size() >= 4) {
          Value subviewSource = castOperands[0];
          auto sourceMemrefTy = dyn_cast<MemRefType>(subviewSource.getType());

          if (sourceMemrefTy) {
            SmallVector<int64_t> sourceStrides;
            int64_t sourceOffset;
            if (succeeded(getStridesAndOffset(sourceMemrefTy, sourceStrides,
                                              sourceOffset)) &&
                !llvm::is_contained(sourceStrides, ShapedType::kDynamic)) {
              int64_t rank = sourceStrides.size();
              size_t numSubviewOffsets = rank;
              size_t numSubviewSizes = rank;

              if (castOperands.size() ==
                  1 + numSubviewOffsets + numSubviewSizes + rank) {
                // Look through a zero-offset parent reinterpret_cast.
                Value base = subviewSource;
                MemRefType baseMemrefTy = sourceMemrefTy;
                if (auto parentCast =
                        subviewSource.getDefiningOp<memref::ReinterpretCastOp>()) {
                  auto staticOffsets = parentCast.getStaticOffsets();
                  if (staticOffsets.size() == 1 && staticOffsets.front() == 0) {
                    base = parentCast.getSource();
                    baseMemrefTy = cast<MemRefType>(base.getType());
                  }
                }

                ptr = rewriter
                          .create<UnrealizedConversionCastOp>(
                              loc, hivm::HIVMToTritonTypeConvert(baseMemrefTy),
                              base)
                          .getResult(0);

                // Compute subview linear offset =
                //   sum(subview_offsets[i] * source_strides[i]).
                for (size_t i = 0; i < numSubviewOffsets; ++i) {
                  Value off = castOperands[1 + i];
                  if (isa<IndexType>(off.getType()))
                    off = rewriter.createOrFold<arith::IndexCastOp>(
                        loc, i64Ty, off);
                  Value strideVal = rewriter.create<arith::ConstantIntOp>(
                      loc, sourceStrides[i], 64);
                  Value term =
                      rewriter.create<arith::MulIOp>(loc, off, strideVal);
                  linearOffset = linearOffset
                                     ? rewriter.create<arith::AddIOp>(
                                           loc, linearOffset, term)
                                     : term;
                }

                // Add load indices offset =
                //   sum(load_indices[i] * subview_strides[i]).
                auto indices = op.getIndices();
                for (size_t i = 0; i < indices.size(); ++i) {
                  Value subviewStride =
                      castOperands[1 + numSubviewOffsets + numSubviewSizes + i];
                  Value idx = indices[i];
                  if (isa<IndexType>(idx.getType()))
                    idx = rewriter.createOrFold<arith::IndexCastOp>(
                        loc, i64Ty, idx);
                  if (isa<IndexType>(subviewStride.getType()))
                    subviewStride = rewriter.createOrFold<arith::IndexCastOp>(
                        loc, i64Ty, subviewStride);
                  Value term =
                      rewriter.create<arith::MulIOp>(loc, idx, subviewStride);
                  linearOffset = linearOffset
                                     ? rewriter.create<arith::AddIOp>(
                                           loc, linearOffset, term)
                                     : term;
                }

                if (!linearOffset)
                  linearOffset =
                      rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

                Value addr = rewriter.create<triton::AddPtrOp>(
                    loc, ptr.getType(), ptr, linearOffset);
                auto load = rewriter.create<triton::LoadOp>(
                    loc, addr, Value(), Value(), llvm::ArrayRef<int32_t>{},
                    std::nullopt, triton::CacheModifier::NONE,
                    triton::EvictionPolicy::NORMAL, false);
                rewriter.replaceOp(op, load.getResult());
                return success();
              }
            }
          }
        }
      }
    }

    // Default path: memref is a function argument or simple memref.
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy,
                                                             convertedMemref);
    ptr = castOp.getResult(0);

    // Extract strides and offset from the memref layout.
    SmallVector<int64_t> strides;
    int64_t staticOffset;
    if (failed(getStridesAndOffset(memrefTy, strides, staticOffset)))
      return failure();

    // Dynamic strides are unsupported — would require descriptor access.
    if (llvm::is_contained(strides, ShapedType::kDynamic))
      return failure();

    // Add static non-zero offset. Dynamic offset (kDynamic) is handled by
    // FuncOpPattern's AddPtr in Stage 2 via the runtime offset argument.
    if (staticOffset != ShapedType::kDynamic && staticOffset != 0) {
      linearOffset =
          rewriter.create<arith::ConstantIntOp>(loc, staticOffset, 64);
    }

    auto indices = op.getIndices();

    for (size_t i = 0; i < indices.size(); ++i) {
      Value idx = indices[i];
      if (isa<IndexType>(idx.getType())) {
        idx = rewriter.createOrFold<arith::IndexCastOp>(loc, i64Ty, idx);
      }
      Value strideVal =
          rewriter.create<arith::ConstantIntOp>(loc, strides[i], 64);
      Value term = rewriter.create<arith::MulIOp>(loc, idx, strideVal);
      linearOffset = linearOffset
                         ? rewriter.create<arith::AddIOp>(loc, linearOffset,
                                                          term)
                         : term;
    }

    if (!linearOffset)
      linearOffset = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

    // Compute the element address and emit a scalar tt.load.
    Value addr = rewriter.create<triton::AddPtrOp>(loc, ptrTy, ptr,
                                                   linearOffset);
    auto load = rewriter.create<triton::LoadOp>(
        loc, addr, Value(), Value(), llvm::ArrayRef<int32_t>{}, std::nullopt,
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

// Convert `memref.extract_aligned_pointer_as_index %src` into:
//   %ptr    = unrealized_conversion_cast %src : memref<...> to !tt.ptr<...>
//   %addr64 = tt.ptr_to_int %ptr : !tt.ptr<...> -> i64
//   %addridx = arith.index_cast %addr64 : i64 to index
//
// The op is used to inspect whether an optional pointer argument is null.
// Stage1 must lower it so Stage2 (FuncOpPattern) no longer sees a memref op
// that references a memref-typed block argument.
class ExtractAlignedPointerAsIndexOpPattern
    : public OpConversionPattern<memref::ExtractAlignedPointerAsIndexOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractAlignedPointerAsIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto memrefTy = op.getSource().getType();
    Type ptrTy = hivm::HIVMToTritonTypeConvert(memrefTy);

    // Bridge the memref operand to a Triton pointer.
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, ptrTy, adaptor.getSource());
    Value ptr = castOp.getResult(0);

    // Cast the pointer to a 64-bit integer address.
    auto i64Ty = rewriter.getI64Type();
    auto addr64 = rewriter.create<triton::PtrToIntOp>(loc, i64Ty, ptr);

    // Convert the integer address to index to match the original result type.
    auto indexTy = rewriter.getIndexType();
    auto addridx = rewriter.create<arith::IndexCastOp>(loc, indexTy, addr64);
    rewriter.replaceOp(op, addridx);
    return success();
  }
};
} // namespace

void mlir::hivm::populateReinterpretCastToUnrealizedCastPatterns(
    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ReinterpretCastOpReplacementPattern>(ctx);
  patterns.add<SubViewOpReplacementPattern>(ctx);
}

void mlir::hivm::populateMemRefLoadToTritonPatterns(RewritePatternSet &patterns) {
  patterns.add<MemRefLoadOpPattern>(patterns.getContext());
}

void mlir::hivm::populateExtractAlignedPointerToTritonPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExtractAlignedPointerAsIndexOpPattern>(patterns.getContext());
}

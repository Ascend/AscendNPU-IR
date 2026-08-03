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
class MemRefLoadOpPattern : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto memrefTy = op.getMemRefType();

    // Bridge the memref operand to a Triton pointer.
    Type ptrTy = hivm::HIVMToTritonTypeConvert(memrefTy);
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, ptrTy, adaptor.getMemref());
    Value ptr = castOp.getResult(0);

    // Extract strides and offset from the memref layout.
    SmallVector<int64_t> strides;
    int64_t staticOffset;
    if (failed(getStridesAndOffset(memrefTy, strides, staticOffset)))
      return failure();

    // Dynamic strides are unsupported — would require descriptor access.
    if (llvm::is_contained(strides, ShapedType::kDynamic))
      return failure();

    // Compute linear offset = staticOffset + sum(indices[i] * strides[i]).
    auto i64Ty = rewriter.getI64Type();
    Value linearOffset;

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

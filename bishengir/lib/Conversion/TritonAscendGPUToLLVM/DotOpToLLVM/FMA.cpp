//===-- FMA.cpp - Outer-product FMA lowering with minimal live ranges -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::ascend;

namespace {

// ---------------------------------------------------------------------------
// Split factor
//
// In the old per-(m,n) K-loop the split reduced the serial FMA chain depth.
// In the outer-product loop the FMAs for different (m,n) pairs are already
// interleaved, providing natural ILP.
// ---------------------------------------------------------------------------
static unsigned getFMASplitFactor(unsigned K) {
  if (K <= 64)
    return 1;
  if (K <= 128)
    return 2;
  return 4;
}

// ---------------------------------------------------------------------------
// GenericFMAVectorMultiplier
// ---------------------------------------------------------------------------
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  // ------------------------------------------------------------------
  // Type promotion: f16 (and bf16) → f32; everything else is identity.
  // Called once per (mi, k) for A and once per (ni, k) for B in the
  // outer-product loop, so the resulting f32 SSA value is live for only
  // totalN (or totalM) FMAs before it becomes dead.
  // ------------------------------------------------------------------
  Value promoteToAccType(Value v, Type accTy) override {
    Type srcTy = v.getType();
    if (srcTy == accTy)
      return v;
    // f16 / bf16 → f32
    if (isa<FloatType>(srcTy) && isa<FloatType>(accTy))
      return builder.create<LLVM::FPExtOp>(loc, accTy, v);
    // Integer widening (i8/i16 → i32)
    if (isa<IntegerType>(srcTy) && isa<IntegerType>(accTy))
      return builder.create<LLVM::SExtOp>(loc, accTy, v);
    return v; // already compatible
  }

  // ------------------------------------------------------------------
  // Single FMA.  a and b must already be promoted to accTy.
  // ------------------------------------------------------------------
  Value emitSingleFMA(Value a, Value b, Value acc) override {
    Type accTy = acc.getType();
    if (isa<FloatType>(accTy))
      return builder.create<LLVM::FMulAddOp>(loc, a, b, acc);
    // Integer: acc + (a * b)
    return builder.create<LLVM::AddOp>(
        loc, builder.create<LLVM::MulOp>(loc, a, b), acc);
  }

  // ------------------------------------------------------------------
  // Vector multiply-accumulate over K elements.
  //
  // Still available for callers that use the vector interface directly.
  // Internally uses emitSingleFMA so promotion is consistent.
  //
  // Split into independent sub-chains (pairwise tree reduction) to
  // reduce critical-path depth.  In the outer-product loop this is
  // called only when the legacy per-(m,n) path is used; the preferred
  // path calls emitSingleFMA directly from parametricConvertFMADot.
  // ------------------------------------------------------------------
  Value multiplyVectors(const Value *a, const Value *b, Value c,
                        unsigned K) override {
    assert(K > 0);
    Type accTy = c.getType();

    const unsigned desiredSplits = getFMASplitFactor(K);
    const unsigned minChunk = 4;
    unsigned splits = desiredSplits;
    while (splits > 1 && K / splits < minChunk)
      splits /= 2;

    const unsigned chunkSize = (K + splits - 1) / splits;

    Value zero;
    if (splits > 1)
      zero = builder.create<LLVM::ConstantOp>(loc, accTy,
                                              builder.getZeroAttr(accTy));

    SmallVector<Value> partials(splits);
    for (unsigned s = 0; s < splits; ++s) {
      Value accum = (s == 0) ? c : zero;
      const unsigned kStart = s * chunkSize;
      const unsigned kEnd = std::min(kStart + chunkSize, K);
      for (unsigned k = kStart; k < kEnd; ++k) {
        Value ai = promoteToAccType(a[k], accTy);
        Value bi = promoteToAccType(b[k], accTy);
        accum = emitSingleFMA(ai, bi, accum);
      }
      partials[s] = accum;
    }

    // Pairwise tree reduction.
    while (partials.size() > 1) {
      SmallVector<Value> next;
      for (unsigned i = 0; i + 1 < partials.size(); i += 2)
        next.push_back(emitAdd(partials[i], partials[i + 1]));
      if (partials.size() % 2)
        next.push_back(partials.back());
      partials = std::move(next);
    }
    return partials[0];
  }

private:
  Value emitAdd(Value x, Value y) {
    if (isa<FloatType>(x.getType()))
      return builder.create<LLVM::FAddOp>(loc, x, y);
    return builder.create<LLVM::AddOp>(loc, x, y);
  }
};

} // namespace

namespace mlir::triton::ascend {

LogicalResult convertFMADot(DotOp op, DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  GenericFMAVectorMultiplier multiplier(rewriter, loc);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}

} // namespace mlir::triton::ascend
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_DOUBLE_TO_FLOAT_RNE_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_DOUBLE_TO_FLOAT_RNE_H

#include "Utils.h"

#include <cstdint>

// Software IEEE-754 double -> float conversion with round-to-nearest-even.
//
// The vector unit has no fp64 lanes and the device runtime does not provide
// __truncdfsf2, so `(float)someDouble` cannot be used here: it would emit a
// reference to that missing symbol. Instead the conversion is done exactly as
// compiler-rt implements it (lib/builtins/fp_trunc_impl.inc) on the raw
// integer bits, so it only needs integer instructions.
__aiv__ __attribute__((always_inline)) inline uint32_t
truncdfsf2_bits_rn(uint64_t a) {
  constexpr int32_t kSrcExpBits = 11;
  constexpr int32_t kSrcSigFracBits = 52;
  constexpr int32_t kDstExpBits = 8;
  constexpr int32_t kDstSigFracBits = 23;

  const int32_t srcInfExp = (1 << kSrcExpBits) - 1;
  const int32_t srcExpBias = srcInfExp >> 1;
  const uint64_t srcExpMask = (1ull << kSrcExpBits) - 1;
  const uint64_t srcSigFracMask = (1ull << kSrcSigFracBits) - 1;

  constexpr int32_t kSigFracTailBits = kSrcSigFracBits - kDstSigFracBits;
  const uint64_t roundMask = (1ull << kSigFracTailBits) - 1;
  const uint64_t halfway = 1ull << (kSigFracTailBits - 1);

  const uint64_t srcQNaN = 1ull << (kSrcSigFracBits - 1);
  const uint64_t srcNaNCode = srcQNaN - 1;
  const uint32_t dstQNaN = 1u << (kDstSigFracBits - 1);
  const uint32_t dstNaNCode = dstQNaN - 1;

  const int32_t dstInfExp = (1 << kDstExpBits) - 1;
  const int32_t dstExpBias = dstInfExp >> 1;
  const int32_t overflowExponent = srcExpBias + dstInfExp - dstExpBias;

  const uint64_t srcSign = (a >> (kSrcSigFracBits + kSrcExpBits)) & 1ull;
  const uint64_t srcExp = (a >> kSrcSigFracBits) & srcExpMask;
  const uint64_t srcSigFrac = a & srcSigFracMask;

  uint32_t dstExp = 0;
  uint32_t dstSigFrac = 0;

  const int32_t dstExpCandidate =
      (static_cast<int32_t>(srcExp) - srcExpBias) + dstExpBias;
  if (dstExpCandidate >= 1 && dstExpCandidate < dstInfExp) {
    // Normal number in the destination format: shift and round the
    // significand, then adjust the exponent.
    dstExp = static_cast<uint32_t>(dstExpCandidate);
    dstSigFrac = static_cast<uint32_t>(srcSigFrac >> kSigFracTailBits);
    const uint64_t roundBits = srcSigFrac & roundMask;
    if (roundBits > halfway)
      ++dstSigFrac;
    else if (roundBits == halfway)
      dstSigFrac += dstSigFrac & 1u;

    if (dstSigFrac >= (1u << kDstSigFracBits)) {
      ++dstExp;
      dstSigFrac ^= (1u << kDstSigFracBits);
    }
  } else if (srcExp == static_cast<uint64_t>(srcInfExp) && srcSigFrac != 0) {
    // NaN: start from infinity, set the quiet bit and keep the truncated
    // payload.
    dstExp = static_cast<uint32_t>(dstInfExp);
    dstSigFrac =
        dstQNaN |
        (static_cast<uint32_t>((srcSigFrac & srcNaNCode) >> kSigFracTailBits) &
         dstNaNCode);
  } else if (static_cast<int32_t>(srcExp) >= overflowExponent) {
    // Overflow to infinity (this also covers the infinity input).
    dstExp = static_cast<uint32_t>(dstInfExp);
    dstSigFrac = 0;
  } else {
    // Underflow or exact zero: produce a denormal or zero.
    uint64_t significand = srcSigFrac;
    int32_t shift = srcExpBias - dstExpBias - static_cast<int32_t>(srcExp);
    if (srcExp != 0) {
      significand |= (1ull << kSrcSigFracBits);
      ++shift;
    }

    if (shift > kSrcSigFracBits) {
      dstExp = 0;
      dstSigFrac = 0;
    } else {
      const bool sticky = shift != 0 && ((significand << (64 - shift)) != 0);
      const uint64_t denormalized =
          (significand >> shift) | (sticky ? 1ull : 0ull);
      dstSigFrac = static_cast<uint32_t>(denormalized >> kSigFracTailBits);
      const uint64_t roundBits = denormalized & roundMask;
      if (roundBits > halfway)
        ++dstSigFrac;
      else if (roundBits == halfway)
        dstSigFrac += dstSigFrac & 1u;

      if (dstSigFrac >= (1u << kDstSigFracBits)) {
        ++dstExp;
        dstSigFrac ^= (1u << kDstSigFracBits);
      }
    }
  }

  return (static_cast<uint32_t>(srcSign) << 31) | (dstExp << kDstSigFracBits) |
         dstSigFrac;
}

// Move the raw bits across the fp64 <-> fp32 boundary. These are plain 8/4 byte
// moves; no fp64 arithmetic is performed, so no soft-float helper is needed.
__aiv__ __attribute__((always_inline)) inline uint64_t
double_to_bits(double value) {
  union {
    double asDouble;
    uint64_t asBits;
  } raw;
  raw.asDouble = value;
  return raw.asBits;
}

__aiv__ __attribute__((always_inline)) inline float
bits_to_float(uint32_t bits) {
  union {
    uint32_t asBits;
    float asFloat;
  } raw;
  raw.asBits = bits;
  return raw.asFloat;
}

// Element-wise f64 -> f32 cast over a UB memref. The rounding mode carried by
// the vcast op is always `rint` for this pair (it is the only mode accepted by
// the hivm.hir.vcast verifier), which is what truncdfsf2_bits_rn implements.
template <size_t DIM>
__aiv__ __attribute__((always_inline)) inline void
vcast_double_to_float_rn_nd(memref_t<__ubuf__ double, DIM> *src,
                            memref_t<__ubuf__ float, DIM> *dst) {
  int64_t outerSize = 1;
  for (size_t i = 0; i + 1 < DIM; ++i)
    outerSize *= src->sizes[i];
  const int64_t innerSize = src->sizes[DIM - 1];

  __ubuf__ double *srcBase = src->aligned + src->offset;
  __ubuf__ float *dstBase = dst->aligned + dst->offset;

  // The conversion runs on the scalar pipe while the buffers are produced and
  // consumed by the vector pipe; serialize between the two like the other
  // scalar fallbacks in this library do.
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  for (int64_t outer = 0; outer < outerSize; ++outer) {
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    // Decompose the flattened outer index into per-dimension coordinates
    // (row-major, so the innermost outer dimension varies fastest).
    int64_t remaining = outer;
    for (int32_t dim = static_cast<int32_t>(DIM) - 2; dim >= 0; --dim) {
      const int64_t coord = remaining % src->sizes[dim];
      remaining /= src->sizes[dim];
      srcOffset += coord * src->strides[dim];
      dstOffset += coord * dst->strides[dim];
    }

    for (int64_t inner = 0; inner < innerSize; ++inner) {
      const uint64_t bits =
          double_to_bits(srcBase[srcOffset + inner * src->strides[DIM - 1]]);
      dstBase[dstOffset + inner * dst->strides[DIM - 1]] =
          bits_to_float(truncdfsf2_bits_rn(bits));
    }
  }

  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_DOUBLE_TO_FLOAT_RNE_H

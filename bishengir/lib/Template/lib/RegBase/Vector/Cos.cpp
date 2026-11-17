/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "ArchDefs.h"
#include "RegBase/SIMTTrigCommon.h"
#include "RegBase/SimtUtils.h"
#include "RegBase/VecUtils.h"
#include "Utils.h"

#if BISHENG_TPL_REGBASE
template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_cos(TYPE src) {
  float x = src;
  x = __fma(x, 0.0f, x);
  int quadrant = 0;
  float reducedAngle = 0.0f;
  // Uses Payne-Hanek for large angles
  if (__fabsf(x) > SIN_COS_THRESHOLD) {
    // Step 1: Extract raw bits of the input angle
    uint32_t inputBits = reinterpret_cast<uint32_t &>(x);

    // Step 2: Extract exponent and compute index into 2/pi table
    int32_t exponent = ((inputBits & 0x7F800000) >> 23) - 127;
    uint32_t exponentIndex = static_cast<uint32_t>(exponent) >> 5;

    // Step 3: Get the 2/pi table entries for this exponent index
    constexpr uint32_t twoOverPiTable[] = {0x517cc1b7, 0x27220a94, 0xfe13abe8,
                                           0xfa9a6ee0, 0x6db14acc, 0x9e21c820};
    uint32_t highTerm = exponentIndex ? twoOverPiTable[exponentIndex - 1] : 0;
    uint32_t midTerm = twoOverPiTable[exponentIndex];
    uint32_t lowTerm = twoOverPiTable[exponentIndex + 1];
    uint32_t lastTerm = twoOverPiTable[exponentIndex + 2];

    // Step 4: Compute exponent remainder and shift table entries accordingly
    int32_t exponentRemainder = static_cast<uint32_t>(exponent) & 0x1F;
    if (exponentRemainder != 0) {
      highTerm = (highTerm << exponentRemainder) |
                 (midTerm >> (32 - exponentRemainder));
      midTerm = (midTerm << exponentRemainder) |
                (lowTerm >> (32 - exponentRemainder));
      lowTerm = (lowTerm << exponentRemainder) |
                (lastTerm >> (32 - exponentRemainder));
    }

    // Step 5: Extract and normalize the mantissa
    uint32_t mantissa = (inputBits & 0x007FFFFF) | 0x4F000000;
    uint32_t normalizedMantissa =
        static_cast<uint32_t>(reinterpret_cast<float &>(mantissa));

    // Step 6: Compute product = (mantissa * highTerm) << 32 + mantissa *
    // midTerm + mantissa * lowTerm
    uint64_t product = static_cast<uint64_t>(normalizedMantissa) * lowTerm;
    product =
        static_cast<uint64_t>(normalizedMantissa) * midTerm + (product >> 32);
    product =
        (static_cast<uint64_t>(normalizedMantissa * highTerm) << 32) + product;

    // Step 7: Extract quotient and remainder
    int32_t quotient = static_cast<int32_t>(product >> 62);
    product = product & 0x3FFFFFFFFFFFFFFFULL;

    // Step 8: Handle carry
    if (product & 0x2000000000000000ULL) {
      product -= 0x4000000000000000ULL;
      quotient += 1;
    }

    // Step 9: Split product into high and low
    int64_t productInt64 = static_cast<int64_t>(product);
    int64_t highFloat = static_cast<float>(productInt64);
    productInt64 = productInt64 - static_cast<int64_t>(highFloat);
    int64_t lowFloat = static_cast<float>(productInt64);

    // Step 10: Compute final result = (high + low) * pi/2 * 2^-62
    reducedAngle = (highFloat + lowFloat) * PI_OVER_TWO_LOW;

    // Step 11: Handle negative input
    if (x < 0.0f) {
      reducedAngle = -reducedAngle;
      quotient = -quotient;
    }

    quadrant = quotient;
  } else {
    // Uses Cody-Waite for small angles
    float y = __fma(x, TWO_OVER_PI, TRUNCATE_VALUE);
    quadrant = reinterpret_cast<int &>(y);
    y = y - TRUNCATE_VALUE;
    x = __fma(y, NEG_HIGN_OF_PI_OVER_2, x);
    x = __fma(y, NEG_MIDDLE_OF_PI_OVER_2, x);
    reducedAngle = __fma(y, NEG_LOW_OF_PI_OVER_2, x);
  }

  // Step 2: Compute cosine and sine of the reduced angle using polynomial
  // approximations
  // cosPoly
  float c = reducedAngle;
  c = c * c;
  float t = __fma(c, ONE_OVER_8_FACT, NEG_ONE_OVER_6_FACT);
  t = __fma(c, t, ONE_OVER_4_FACT);
  t = __fma(c, t, NEG_ONE_OVER_2_FACT);
  c = __fma(c, t, ONE);

  // sinpoly
  float s = reducedAngle;
  float p = s * s;
  float m = __fma(s, p, 0.0f);

  float z = __fma(p, ONE_OVER_9_FACT, NEG_ONE_OVER_7_FACT);
  z = __fma(p, z, ONE_OVER_5_FACT);
  z = __fma(p, z, NEG_ONE_OVER_3_FACT);

  s = __fma(z, m, s); // * x^3 + x

  // Step 3: Adjust the sine value based on the quadrant
  if (quadrant & 2) { // Quadrants 2 and 3: cos(pi + x) = -cos(x)
    s = -s;
    c = -c;
  }
  if (quadrant & 1) { // Quadrants 1 and 3: cos(pi/2 + x) = -sin(x)
    c = -s;
  }

  return (TYPE)c;
}

// Template-lib SIMT cos (A3: threshold + Payne-Hanek / Cody-Waite + minimax).
// The body mirrors Sin.cpp: it uses the compiler-style Payne-Hanek reduction
// (32-bit limbs, no uint64 math) with the 2/pi table staged into UB by the
// caller. The cosine identity used here is
//   cos(x) = (-1)^parity * sin(pi / 2 - r),  r = frac(x / pi) * pi.
#define COS_BODY(TBL)                                                          \
                                                                               \
  float x = src;                                                               \
  x = __fma(x, 0.0f, x);                                                       \
  int quadrant = 0;                                                            \
  float reducedAngle = 0.0f;                                                   \
                                                                               \
  if (__fabsf(x) > SIN_COS_THRESHOLD) {                                        \
                                                                               \
    uint32_t inputBits = reinterpret_cast<uint32_t &>(x);                      \
    uint32_t expo = (inputBits >> 23) & 0xffu;                                 \
    if (expo == 0xffu) {                                                       \
      return static_cast<TYPE>(x - x);                                         \
    }                                                                          \
    uint32_t mant = (inputBits & 0x007fffffu) | 0x00800000u;                   \
                                                                               \
    uint32_t indexHigh = expo + 8u;                                            \
    uint32_t indexLow = indexHigh + 32u;                                       \
    uint32_t twoInvPiHigh = (TBL)[indexHigh];                                  \
    uint32_t twoInvPiLow = (TBL)[indexLow];                                    \
    uint32_t xHigh = mant >> 16;                                               \
    uint32_t xLow = mant & 0xffffu;                                            \
    uint32_t highHigh = twoInvPiHigh >> 16;                                    \
    uint32_t highLow = twoInvPiHigh & 0xffffu;                                 \
    uint32_t lowHigh = twoInvPiLow >> 16;                                      \
    uint32_t lowLow = twoInvPiLow & 0xffffu;                                   \
    uint32_t term0 = (xLow * lowHigh) >> 16;                                   \
    uint32_t term1 = xLow * highLow;                                           \
    uint32_t term2 = ((xLow * highHigh) & 0xffffu) << 16;                      \
    uint32_t term3 = (xHigh * lowLow) >> 16;                                   \
    uint32_t term4 = xHigh * lowHigh;                                          \
    uint32_t term5 = ((xHigh * highLow) & 0xffffu) << 16;                      \
    uint32_t accumulated = term0 + term1 + term2 + term3 + term4 + term5;      \
    uint32_t parity = accumulated >> 31;                                       \
    uint32_t reducedBits = accumulated & 0x7fffffffu;                          \
                                                                               \
    float pi = 3.1415927f;                                                     \
    float reduced =                                                            \
        static_cast<float>(reducedBits) * 4.656612873077393e-10f * pi;         \
                                                                               \
    /* cos(r) = sin(pi / 2 - r), with r in [0, pi). */                         \
    float a = PI_OVER_2 - reduced;                                             \
    float a2 = a * a;                                                          \
    float a3 = a * a2;                                                         \
                                                                               \
    /* Minimax coefficients for sin on [-pi/2, pi/2]; the odd polynomial       \
       handles the negative half directly. */                                  \
    constexpr float PH_COS_C3 = -1.6666658223e-01f;                            \
    constexpr float PH_COS_C5 = 8.3330580965e-03f;                             \
    constexpr float PH_COS_C7 = -1.9809580408e-04f;                            \
    constexpr float PH_COS_C9 = 2.6065545171e-06f;                             \
    float poly = __fma(a2, PH_COS_C9, PH_COS_C7);                              \
    poly = __fma(a2, poly, PH_COS_C5);                                         \
    poly = __fma(a2, poly, PH_COS_C3);                                         \
    float c = __fma(poly, a3, a);                                              \
    if ((parity & 1u) != 0u) {                                                 \
      c = -c;                                                                  \
    }                                                                          \
    return static_cast<TYPE>(c);                                               \
  } else {                                                                     \
                                                                               \
    float y = __fma(x, TWO_OVER_PI, TRUNCATE_VALUE);                           \
    quadrant = reinterpret_cast<int &>(y);                                     \
    y = y - TRUNCATE_VALUE;                                                    \
    x = __fma(y, NEG_HIGN_OF_PI_OVER_2, x);                                    \
    x = __fma(y, NEG_MIDDLE_OF_PI_OVER_2, x);                                  \
    reducedAngle = __fma(y, NEG_LOW_OF_PI_OVER_2, x);                          \
  }                                                                            \
                                                                               \
  float c = reducedAngle;                                                      \
  c = c * c;                                                                   \
  float t = __fma(c, ONE_OVER_8_FACT, NEG_ONE_OVER_6_FACT);                    \
  t = __fma(c, t, ONE_OVER_4_FACT);                                            \
  t = __fma(c, t, NEG_ONE_OVER_2_FACT);                                        \
  c = __fma(c, t, ONE);                                                        \
                                                                               \
  float s = reducedAngle;                                                      \
  float p = s * s;                                                             \
  float m = __fma(s, p, 0.0f);                                                 \
                                                                               \
  float z = __fma(p, ONE_OVER_9_FACT, NEG_ONE_OVER_7_FACT);                    \
  z = __fma(p, z, ONE_OVER_5_FACT);                                            \
  z = __fma(p, z, NEG_ONE_OVER_3_FACT);                                        \
                                                                               \
  s = __fma(z, m, s);                                                          \
                                                                               \
  if (quadrant & 2) {                                                          \
    s = -s;                                                                    \
    c = -c;                                                                    \
  }                                                                            \
  if (quadrant & 1) {                                                          \
    c = -s;                                                                    \
  }                                                                            \
                                                                               \
  return (TYPE)c;

// __simt_callee__ entry callable from a __simt_vf__ kernel. The 2/pi table is
// passed in (staged into UB by the caller) so the SIMT body does not keep a
// private copy of the 320-word table.
template <typename TYPE>
__simt_callee__ __attribute__((always_inline)) TYPE
simt_cos_callee(TYPE src, const __ubuf__ uint32_t *tbl) {
  COS_BODY(tbl)
}

//===----------------------------------------------------------------------===//
// SIMT wrapper: from the vector/AIV context enter SIMT execution and compute
// cos via the template-lib algorithm (threshold + Payne-Hanek + minimax).
// Entry points are rank-specialized because the memref descriptor layout and
// the element-count derivation differ per rank. The rank suffix must stay in
// sync with COS_SIMT_MAX_RANK / kMaxTemplateLibTrigRank.
//===----------------------------------------------------------------------===//
constexpr uint32_t COS_SIMT_THREAD_NUM = 1024;

template <typename DTYPE> __simt_vf__ LAUNCH_BOUND(COS_SIMT_THREAD_NUM)
__aiv__ __attribute__((always_inline)) void
simtCosCompute(__ubuf__ DTYPE *src, __ubuf__ DTYPE *dst, const int64_t total,
               const __ubuf__ uint32_t *tbl) {
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < total;
       i += static_cast<int64_t>(blockDim.x)) {
    dst[i] = simt_cos_callee<DTYPE>(src[i], tbl);
  }
}

// Rank-generic kernel: the element offset is reconstructed from the memref
// sizes/strides, so non-unit innermost strides and non-contiguous views are
// handled. Only used for rank >= 2; rank 1 keeps the cheaper contiguous
// kernel above. Dimensions above RANK are passed as extent 1 / stride 0.
template <typename DTYPE, int RANK>
__simt_vf__ LAUNCH_BOUND(COS_SIMT_THREAD_NUM)
__aiv__ __attribute__((always_inline)) void
simtCosComputeND(__ubuf__ DTYPE *src, __ubuf__ DTYPE *dst, const int64_t total,
                 const int64_t s0, const int64_t s1, const int64_t s2,
                 const int64_t s3, const int64_t s4, const int64_t ss0,
                 const int64_t ss1, const int64_t ss2, const int64_t ss3,
                 const int64_t ss4, const int64_t ds0, const int64_t ds1,
                 const int64_t ds2, const int64_t ds3, const int64_t ds4,
                 const __ubuf__ uint32_t *tbl) {
  const int64_t sizes[COS_SIMT_MAX_RANK] = {s0, s1, s2, s3, s4};
  const int64_t srcStrides[COS_SIMT_MAX_RANK] = {ss0, ss1, ss2, ss3, ss4};
  const int64_t dstStrides[COS_SIMT_MAX_RANK] = {ds0, ds1, ds2, ds3, ds4};
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < total;
       i += static_cast<int64_t>(blockDim.x)) {
    int64_t linear = i;
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    for (int dim = RANK - 1; dim >= 0; --dim) {
      const int64_t index = linear % sizes[dim];
      linear /= sizes[dim];
      srcOffset += index * srcStrides[dim];
      dstOffset += index * dstStrides[dim];
    }
    dst[dstOffset] = simt_cos_callee<DTYPE>(src[srcOffset], tbl);
  }
}

// Extent of dimension `dim`, with 1 for dimensions the rank does not have.
template <int RANK, typename MemRefT>
__aiv__ __attribute__((always_inline)) int64_t cosSimtExtent(const MemRefT *ref,
                                                             int dim) {
  return dim < RANK ? ref->sizes[dim] : 1;
}

// Stride of dimension `dim`, with 0 for dimensions the rank does not have
// (their extent is 1, so the stride is never used).
template <int RANK, typename MemRefT>
__aiv__ __attribute__((always_inline)) int64_t cosSimtStride(const MemRefT *ref,
                                                             int dim) {
  return dim < RANK ? ref->strides[dim] : 0;
}

template <typename DTYPE, int RANK>
__aiv__ __attribute__((always_inline)) void
cosSimtImpl(memref_t<__ubuf__ DTYPE, RANK> *src,
            memref_t<__ubuf__ int32_t, 1> *tbl,
            memref_t<__ubuf__ DTYPE, RANK> *dst) {
  __ubuf__ DTYPE *srcPtr =
      reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset);
  __ubuf__ DTYPE *dstPtr =
      reinterpret_cast<__ubuf__ DTYPE *>(dst->aligned + dst->offset);
  const __ubuf__ uint32_t *tblPtr =
      reinterpret_cast<const __ubuf__ uint32_t *>(tbl->aligned + tbl->offset);
  if constexpr (RANK == 1) {
    const int64_t total = src->sizes[0];
    cce::async_invoke<simtCosCompute<DTYPE>>(cce::dim3{COS_SIMT_THREAD_NUM},
                                             srcPtr, dstPtr, total, tblPtr);
  } else {
    // The SIMT callee processes one element per index, so the count is the
    // full number of elements, not the extent of the outermost dimension.
    int64_t total = 1;
    for (int dim = 0; dim < RANK; ++dim)
      total *= src->sizes[dim];
    cce::async_invoke<simtCosComputeND<DTYPE, RANK>>(
        cce::dim3{COS_SIMT_THREAD_NUM}, srcPtr, dstPtr, total,
        cosSimtExtent<RANK>(src, 0), cosSimtExtent<RANK>(src, 1),
        cosSimtExtent<RANK>(src, 2), cosSimtExtent<RANK>(src, 3),
        cosSimtExtent<RANK>(src, 4), cosSimtStride<RANK>(src, 0),
        cosSimtStride<RANK>(src, 1), cosSimtStride<RANK>(src, 2),
        cosSimtStride<RANK>(src, 3), cosSimtStride<RANK>(src, 4),
        cosSimtStride<RANK>(dst, 0), cosSimtStride<RANK>(dst, 1),
        cosSimtStride<RANK>(dst, 2), cosSimtStride<RANK>(dst, 3),
        cosSimtStride<RANK>(dst, 4), tblPtr);
  }
}

extern "C" {
REGISTER_SIMT_OP(cos, bfloat16_t, bfloat16_t);
REGISTER_SIMT_OP(cos, half, half);
REGISTER_SIMT_OP(cos, float, float);

#define DEFINE_COS_SIMT_ENTRY(RANK)                                            \
  __aiv__ __attribute__((visibility("default"), always_inline)) void           \
      _mlir_ciface_cos_fp32_simt_##RANK##d(                                    \
          memref_t<__ubuf__ float, RANK> *src,                                 \
          memref_t<__ubuf__ int32_t, 1> *tbl,                                  \
          memref_t<__ubuf__ float, RANK> *dst) {                               \
    cosSimtImpl<float, RANK>(src, tbl, dst);                                   \
  }

#if COS_SIMT_MAX_RANK >= 1
DEFINE_COS_SIMT_ENTRY(1)
#endif
#if COS_SIMT_MAX_RANK >= 2
DEFINE_COS_SIMT_ENTRY(2)
#endif
#if COS_SIMT_MAX_RANK >= 3
DEFINE_COS_SIMT_ENTRY(3)
#endif
#if COS_SIMT_MAX_RANK >= 4
DEFINE_COS_SIMT_ENTRY(4)
#endif
#if COS_SIMT_MAX_RANK >= 5
DEFINE_COS_SIMT_ENTRY(5)
#endif

#undef DEFINE_COS_SIMT_ENTRY
}
#endif

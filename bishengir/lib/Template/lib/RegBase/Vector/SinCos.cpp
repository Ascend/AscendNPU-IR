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

#include "ArchDefs.h"
#include "RegBase/SIMTTrigCommon.h"
#include "RegBase/SimtUtils.h"
#include "RegBase/VecUtils.h"
#include "Utils.h"

#if BISHENG_TPL_REGBASE
template <typename TYPE>
__simt_callee__ __attribute__((always_inline)) void
simt_sincos_callee(TYPE src, const __ubuf__ uint32_t *tbl, TYPE *s, TYPE *c) {
  float x = src;
  x = __fma(x, 0.0f, x);
  int quadrant = 0;

  if (__fabsf(x) > SIN_COS_THRESHOLD) {

    uint32_t inputBits = reinterpret_cast<uint32_t &>(x);
    uint32_t sign = (inputBits >> 31) & 1u;
    uint32_t expo = (inputBits >> 23) & 0xffu;
    if (expo == 0xffu) {
      *s = static_cast<TYPE>(x - x);
      *c = static_cast<TYPE>(x - x);
      return;
    }
    uint32_t mant = (inputBits & 0x007fffffu) | 0x00800000u;

    uint32_t indexHigh = expo + 8u;
    uint32_t indexLow = indexHigh + 32u;
    uint32_t twoInvPiHigh = tbl[indexHigh];
    uint32_t twoInvPiLow = tbl[indexLow];
    uint32_t xHigh = mant >> 16;
    uint32_t xLow = mant & 0xffffu;
    uint32_t highHigh = twoInvPiHigh >> 16;
    uint32_t highLow = twoInvPiHigh & 0xffffu;
    uint32_t lowHigh = twoInvPiLow >> 16;
    uint32_t lowLow = twoInvPiLow & 0xffffu;
    uint32_t term0 = (xLow * lowHigh) >> 16;
    uint32_t term1 = xLow * highLow;
    uint32_t term2 = ((xLow * highHigh) & 0xffffu) << 16;
    uint32_t term3 = (xHigh * lowLow) >> 16;
    uint32_t term4 = xHigh * lowHigh;
    uint32_t term5 = ((xHigh * highLow) & 0xffffu) << 16;
    uint32_t accumulated = term0 + term1 + term2 + term3 + term4 + term5;
    uint32_t parity = accumulated >> 31;
    uint32_t reducedBits = accumulated & 0x7fffffffu;

    float pi = 3.1415927f;
    float r = static_cast<float>(reducedBits) * 4.656612873077393e-10f * pi;

    /* Minimax sin polynomial on [-pi/2, pi/2], shared by both results. */
    constexpr float PH_C3 = -1.6666658223e-01f;
    constexpr float PH_C5 = 8.3330580965e-03f;
    constexpr float PH_C7 = -1.9809580408e-04f;
    constexpr float PH_C9 = 2.6065545171e-06f;

    /* cos(x) = (-1)^parity * sin(pi/2 - r). */
    float ac = PI_OVER_2 - r;
    float ac2 = ac * ac;
    float ac3 = ac * ac2;
    float pc = __fma(ac2, PH_C9, PH_C7);
    pc = __fma(ac2, pc, PH_C5);
    pc = __fma(ac2, pc, PH_C3);
    float cosValue = __fma(pc, ac3, ac);
    if ((parity & 1u) != 0u) {
      cosValue = -cosValue;
    }

    /* sin(x) = (-1)^(parity ^ sign) * sin(min(r, pi - r)). */
    float folded = pi - r;
    float y = r < folded ? r : folded;
    float y2 = y * y;
    float y3 = y * y2;
    float ps = __fma(y2, PH_C9, PH_C7);
    ps = __fma(y2, ps, PH_C5);
    ps = __fma(y2, ps, PH_C3);
    float sinValue = __fma(ps, y3, y);
    if (((parity ^ sign) & 1u) != 0u) {
      sinValue = -sinValue;
    }

    *s = static_cast<TYPE>(sinValue);
    *c = static_cast<TYPE>(cosValue);
    return;
  }

  /* Cody-Waite: reduce once, then evaluate both polynomials on it. */
  float y = __fma(x, TWO_OVER_PI, TRUNCATE_VALUE);
  quadrant = reinterpret_cast<int &>(y);
  y = y - TRUNCATE_VALUE;
  x = __fma(y, NEG_HIGN_OF_PI_OVER_2, x);
  x = __fma(y, NEG_MIDDLE_OF_PI_OVER_2, x);
  x = __fma(y, NEG_LOW_OF_PI_OVER_2, x);

  float sinR = x;
  float sinSq = sinR * sinR;
  float sinM = __fma(sinR, sinSq, 0.0f);
  float sinZ = __fma(sinSq, ONE_OVER_9_FACT, NEG_ONE_OVER_7_FACT);
  sinZ = __fma(sinSq, sinZ, ONE_OVER_5_FACT);
  sinZ = __fma(sinSq, sinZ, NEG_ONE_OVER_3_FACT);
  sinR = __fma(sinZ, sinM, sinR);

  float cosR = x;
  float cosSq = cosR * cosR;
  float cosT = __fma(cosSq, ONE_OVER_8_FACT, NEG_ONE_OVER_6_FACT);
  cosT = __fma(cosSq, cosT, ONE_OVER_4_FACT);
  cosT = __fma(cosSq, cosT, NEG_ONE_OVER_2_FACT);
  cosR = __fma(cosSq, cosT, ONE);

  if (quadrant & 2) {
    sinR = -sinR;
    cosR = -cosR;
  }
  if (quadrant & 1) {
    float swap = -sinR;
    sinR = cosR;
    cosR = swap;
  }

  *s = static_cast<TYPE>(sinR);
  *c = static_cast<TYPE>(cosR);
}

//===----------------------------------------------------------------------===//
// SIMT wrapper: one range reduction feeds both sin and cos. Entry points are
// rank-specialized because the memref descriptor layout differs per rank.
//===----------------------------------------------------------------------===//
constexpr uint32_t SINCOS_SIMT_THREAD_NUM = 1024;

template <typename DTYPE> __simt_vf__ LAUNCH_BOUND(SINCOS_SIMT_THREAD_NUM)
__aiv__ __attribute__((always_inline)) void
simtSincosCompute(__ubuf__ DTYPE *src, __ubuf__ DTYPE *dstSin,
                  __ubuf__ DTYPE *dstCos, const int64_t total,
                  const int64_t srcStride, const int64_t sinStride,
                  const int64_t cosStride, const __ubuf__ uint32_t *tbl) {
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < total;
       i += static_cast<int64_t>(blockDim.x)) {
    DTYPE s;
    DTYPE c;
    simt_sincos_callee<DTYPE>(src[i * srcStride], tbl, &s, &c);
    dstSin[i * sinStride] = s;
    dstCos[i * cosStride] = c;
  }
}

// Rank-generic kernel: element offsets are reconstructed from the memref
// sizes/strides. Only used for rank >= 2; rank 1 keeps the cheaper 1-D kernel,
// which applies the strides directly instead of decomposing an index.
template <typename DTYPE, int RANK>
__simt_vf__ LAUNCH_BOUND(SINCOS_SIMT_THREAD_NUM)
__aiv__ __attribute__((always_inline)) void simtSincosComputeND(
    __ubuf__ DTYPE *src, __ubuf__ DTYPE *dstSin, __ubuf__ DTYPE *dstCos,
    const int64_t total, const int64_t s0, const int64_t s1, const int64_t s2,
    const int64_t s3, const int64_t s4, const int64_t ss0, const int64_t ss1,
    const int64_t ss2, const int64_t ss3, const int64_t ss4, const int64_t ds0,
    const int64_t ds1, const int64_t ds2, const int64_t ds3, const int64_t ds4,
    const int64_t dc0, const int64_t dc1, const int64_t dc2, const int64_t dc3,
    const int64_t dc4, const __ubuf__ uint32_t *tbl) {
  const int64_t sizes[SINCOS_SIMT_MAX_RANK] = {s0, s1, s2, s3, s4};
  const int64_t srcStrides[SINCOS_SIMT_MAX_RANK] = {ss0, ss1, ss2, ss3, ss4};
  const int64_t sinStrides[SINCOS_SIMT_MAX_RANK] = {ds0, ds1, ds2, ds3, ds4};
  const int64_t cosStrides[SINCOS_SIMT_MAX_RANK] = {dc0, dc1, dc2, dc3, dc4};
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < total;
       i += static_cast<int64_t>(blockDim.x)) {
    int64_t linear = i;
    int64_t srcOffset = 0;
    int64_t sinOffset = 0;
    int64_t cosOffset = 0;
    for (int dim = RANK - 1; dim >= 0; --dim) {
      const int64_t index = linear % sizes[dim];
      linear /= sizes[dim];
      srcOffset += index * srcStrides[dim];
      sinOffset += index * sinStrides[dim];
      cosOffset += index * cosStrides[dim];
    }
    DTYPE s;
    DTYPE c;
    simt_sincos_callee<DTYPE>(src[srcOffset], tbl, &s, &c);
    dstSin[sinOffset] = s;
    dstCos[cosOffset] = c;
  }
}

// Extent of dimension `dim`, with 1 for dimensions the rank does not have.
template <int RANK, typename MemRefT>
__aiv__ __attribute__((always_inline)) int64_t
sincosSimtExtent(const MemRefT *ref, int dim) {
  return dim < RANK ? ref->sizes[dim] : 1;
}

// Stride of dimension `dim`, with 0 for dimensions the rank does not have
// (their extent is 1, so the stride is never used).
template <int RANK, typename MemRefT>
__aiv__ __attribute__((always_inline)) int64_t
sincosSimtStride(const MemRefT *ref, int dim) {
  return dim < RANK ? ref->strides[dim] : 0;
}

template <typename DTYPE, int RANK>
__aiv__ __attribute__((always_inline)) void
sincosSimtImpl(memref_t<__ubuf__ DTYPE, RANK> *src,
               memref_t<__ubuf__ int32_t, 1> *tbl,
               memref_t<__ubuf__ DTYPE, RANK> *dstSin,
               memref_t<__ubuf__ DTYPE, RANK> *dstCos) {
  __ubuf__ DTYPE *srcPtr =
      reinterpret_cast<__ubuf__ DTYPE *>(src->aligned + src->offset);
  __ubuf__ DTYPE *dstSinPtr =
      reinterpret_cast<__ubuf__ DTYPE *>(dstSin->aligned + dstSin->offset);
  __ubuf__ DTYPE *dstCosPtr =
      reinterpret_cast<__ubuf__ DTYPE *>(dstCos->aligned + dstCos->offset);
  const __ubuf__ uint32_t *tblPtr =
      reinterpret_cast<const __ubuf__ uint32_t *>(tbl->aligned + tbl->offset);
  if constexpr (RANK == 1) {
    const int64_t total = src->sizes[0];
    cce::async_invoke<simtSincosCompute<DTYPE>>(
        cce::dim3{SINCOS_SIMT_THREAD_NUM}, srcPtr, dstSinPtr, dstCosPtr, total,
        src->strides[0], dstSin->strides[0], dstCos->strides[0], tblPtr);
  } else {
    int64_t total = 1;
    for (int dim = 0; dim < RANK; ++dim)
      total *= src->sizes[dim];
    cce::async_invoke<simtSincosComputeND<DTYPE, RANK>>(
        cce::dim3{SINCOS_SIMT_THREAD_NUM}, srcPtr, dstSinPtr, dstCosPtr, total,
        sincosSimtExtent<RANK>(src, 0), sincosSimtExtent<RANK>(src, 1),
        sincosSimtExtent<RANK>(src, 2), sincosSimtExtent<RANK>(src, 3),
        sincosSimtExtent<RANK>(src, 4), sincosSimtStride<RANK>(src, 0),
        sincosSimtStride<RANK>(src, 1), sincosSimtStride<RANK>(src, 2),
        sincosSimtStride<RANK>(src, 3), sincosSimtStride<RANK>(src, 4),
        sincosSimtStride<RANK>(dstSin, 0), sincosSimtStride<RANK>(dstSin, 1),
        sincosSimtStride<RANK>(dstSin, 2), sincosSimtStride<RANK>(dstSin, 3),
        sincosSimtStride<RANK>(dstSin, 4), sincosSimtStride<RANK>(dstCos, 0),
        sincosSimtStride<RANK>(dstCos, 1), sincosSimtStride<RANK>(dstCos, 2),
        sincosSimtStride<RANK>(dstCos, 3), sincosSimtStride<RANK>(dstCos, 4),
        tblPtr);
  }
}

extern "C" {
#define DEFINE_SINCOS_SIMT_ENTRY(RANK)                                         \
  __aiv__ __attribute__((visibility("default"), always_inline)) void           \
      _mlir_ciface_sincos_fp32_simt_##RANK##d(                                 \
          memref_t<__ubuf__ float, RANK> *src,                                 \
          memref_t<__ubuf__ int32_t, 1> *tbl,                                  \
          memref_t<__ubuf__ float, RANK> *dstSin,                              \
          memref_t<__ubuf__ float, RANK> *dstCos) {                            \
    sincosSimtImpl<float, RANK>(src, tbl, dstSin, dstCos);                     \
  }

#if SINCOS_SIMT_MAX_RANK >= 1
DEFINE_SINCOS_SIMT_ENTRY(1)
#endif
#if SINCOS_SIMT_MAX_RANK >= 2
DEFINE_SINCOS_SIMT_ENTRY(2)
#endif
#if SINCOS_SIMT_MAX_RANK >= 3
DEFINE_SINCOS_SIMT_ENTRY(3)
#endif
#if SINCOS_SIMT_MAX_RANK >= 4
DEFINE_SINCOS_SIMT_ENTRY(4)
#endif
#if SINCOS_SIMT_MAX_RANK >= 5
DEFINE_SINCOS_SIMT_ENTRY(5)
#endif

#undef DEFINE_SINCOS_SIMT_ENTRY
}
#endif

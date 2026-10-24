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
// Template-lib SIMT sin (A3: threshold + Payne-Hanek / Cody-Waite + minimax).
// The body is a macro so the same algorithm can be emitted inside each entry
// function with that function's calling convention (aicore vs simt_callee).
// 2/pi table (32-bit limbs) for the compiler-style Payne-Hanek
// reduction. Defined once so the inline body below can take either this
// constant (aicore entry) or the caller-staged UB pointer (SIMT callee).
static constexpr uint32_t SIN_TWO_OVER_PI_TABLE[320] = {
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000002, 0x00000005,
    0x0000000a, 0x00000014, 0x00000028, 0x00000051, 0x000000a2, 0x00000145,
    0x0000028b, 0x00000517, 0x00000a2f, 0x0000145f, 0x000028be, 0x0000517c,
    0x0000a2f9, 0x000145f3, 0x00028be6, 0x000517cc, 0x000a2f98, 0x00145f30,
    0x0028be60, 0x00517cc1, 0x00a2f983, 0x0145f306, 0x028be60d, 0x0517cc1b,
    0x0a2f9836, 0x145f306d, 0x28be60db, 0x517cc1b7, 0xa2f9836e, 0x45f306dc,
    0x8be60db9, 0x17cc1b72, 0x2f9836e4, 0x5f306dc9, 0xbe60db93, 0x7cc1b727,
    0xf9836e4e, 0xf306dc9c, 0xe60db939, 0xcc1b7272, 0x9836e4e4, 0x306dc9c8,
    0x60db9391, 0xc1b72722, 0x836e4e44, 0x06dc9c88, 0x0db93910, 0x1b727220,
    0x36e4e441, 0x6dc9c882, 0xdb939105, 0xb727220a, 0x6e4e4415, 0xdc9c882a,
    0xb9391054, 0x727220a9, 0xe4e44152, 0xc9c882a5, 0x9391054a, 0x27220a94,
    0x4e441529, 0x9c882a53, 0x391054a7, 0x7220a94f, 0xe441529f, 0xc882a53f,
    0x91054a7f, 0x220a94fe, 0x441529fc, 0x882a53f8, 0x1054a7f0, 0x20a94fe1,
    0x41529fc2, 0x82a53f84, 0x054a7f09, 0x0a94fe13, 0x1529fc27, 0x2a53f84e,
    0x54a7f09d, 0xa94fe13a, 0x529fc275, 0xa53f84ea, 0x4a7f09d5, 0x94fe13ab,
    0x29fc2757, 0x53f84eaf, 0xa7f09d5f, 0x4fe13abe, 0x9fc2757d, 0x3f84eafa,
    0x7f09d5f4, 0xfe13abe8, 0xfc2757d1, 0xf84eafa3, 0xf09d5f47, 0xe13abe8f,
    0xc2757d1f, 0x84eafa3e, 0x09d5f47d, 0x13abe8fa, 0x2757d1f5, 0x4eafa3ea,
    0x9d5f47d4, 0x3abe8fa9, 0x757d1f53, 0xeafa3ea6, 0xd5f47d4d, 0xabe8fa9a,
    0x57d1f534, 0xafa3ea69, 0x5f47d4d3, 0xbe8fa9a6, 0x7d1f534d, 0xfa3ea69b,
    0xf47d4d37, 0xe8fa9a6e, 0xd1f534dd, 0xa3ea69bb, 0x47d4d377, 0x8fa9a6ee,
    0x1f534ddc, 0x3ea69bb8, 0x7d4d3770, 0xfa9a6ee0, 0xf534ddc0, 0xea69bb81,
    0xd4d37703, 0xa9a6ee06, 0x534ddc0d, 0xa69bb81b, 0x4d377036, 0x9a6ee06d,
    0x34ddc0db, 0x69bb81b6, 0xd377036d, 0xa6ee06db, 0x4ddc0db6, 0x9bb81b6c,
    0x377036d8, 0x6ee06db1, 0xddc0db62, 0xbb81b6c5, 0x77036d8a, 0xee06db14,
    0xdc0db629, 0xb81b6c52, 0x7036d8a5, 0xe06db14a, 0xc0db6295, 0x81b6c52b,
    0x036d8a56, 0x06db14ac, 0x0db62959, 0x1b6c52b3, 0x36d8a566, 0x6db14acc,
    0xdb629599, 0xb6c52b32, 0x6d8a5664, 0xdb14acc9, 0xb6295993, 0x6c52b327,
    0xd8a5664f, 0xb14acc9e, 0x6295993c, 0xc52b3278, 0x8a5664f1, 0x14acc9e2,
    0x295993c4, 0x52b32788, 0xa5664f10, 0x4acc9e21, 0x95993c43, 0x2b327887,
    0x5664f10e, 0xacc9e21c, 0x5993c439, 0xb3278872, 0x664f10e4, 0xcc9e21c8,
    0x993c4390, 0x32788720, 0x64f10e41, 0xc9e21c82, 0x93c43904, 0x27887208,
    0x4f10e410, 0x9e21c820,
};

#define SIN_BODY(TBL)                                                          \
                                                                               \
  float x = src;                                                               \
  x = __fma(x, 0.0f, x);                                                       \
  int quadrant = 0;                                                            \
  float reducedAngle = 0.0f;                                                   \
                                                                               \
  if (__fabsf(x) > SIN_COS_THRESHOLD) {                                        \
                                                                               \
    uint32_t inputBits = reinterpret_cast<uint32_t &>(x);                      \
    uint32_t sign = (inputBits >> 31) & 1u;                                    \
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
    float angle =                                                              \
        static_cast<float>(reducedBits) * 4.656612873077393e-10f * pi;         \
    float folded = pi - angle;                                                 \
    float y = angle < folded ? angle : folded;                                 \
                                                                               \
    float y2 = y * y;                                                          \
    float y3 = y * y2;                                                         \
                                                                               \
    constexpr float PH_SIN_C3 = -1.6666658223e-01f;                            \
    constexpr float PH_SIN_C5 = 8.3330580965e-03f;                             \
    constexpr float PH_SIN_C7 = -1.9809580408e-04f;                            \
    constexpr float PH_SIN_C9 = 2.6065545171e-06f;                             \
    float poly = __fma(y2, PH_SIN_C9, PH_SIN_C7);                              \
    poly = __fma(y2, poly, PH_SIN_C5);                                         \
    poly = __fma(y2, poly, PH_SIN_C3);                                         \
    float s = __fma(poly, y3, y);                                              \
    if (((parity ^ sign) & 1u) != 0u) {                                        \
      s = -s;                                                                  \
    }                                                                          \
    return static_cast<TYPE>(s);                                               \
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
    s = c;                                                                     \
  }                                                                            \
                                                                               \
  return (TYPE)s;

template <typename TYPE>
__aicore__ __attribute__((always_inline)) TYPE simt_sin(TYPE src) {
  SIN_BODY(SIN_TWO_OVER_PI_TABLE)
}

// __simt_callee__ entry callable from a __simt_vf__ kernel. The 2/pi table is
// passed in (staged into UB by the caller) so the SIMT body does not keep a
// private copy of the 320-word table.
template <typename TYPE>
__simt_callee__ __attribute__((always_inline)) TYPE
simt_sin_callee(TYPE src, const __ubuf__ uint32_t *tbl) {
  SIN_BODY(tbl)
}

//===----------------------------------------------------------------------===//
// SIMT wrapper: from the vector/AIV context enter SIMT execution and compute
// sin via the template-lib algorithm (threshold + Payne-Hanek + minimax).
// Exposed under a distinct symbol so it does not collide with libdevice's
// `_mlir_ciface_sin_fp32` / `_mlir_ciface_simt_sin_float`.
//
// The entry is selected by memref rank: the descriptor layout and therefore
// the way the element count is derived depend on the rank, so every supported
// rank has its own symbol (`_mlir_ciface_sin_fp32_simt_<rank>d`). The rank
// suffix must stay in sync with kMaxTemplateLibTrigRank in
// lib/Conversion/MathToHFusion/MathToHFusion.cpp.
//
// No pipe_barrier is emitted here, matching the other SIMT wrappers
// (SIMTStrideLoad/Store, SIMTIndexSelect, SIMTGatherUBToUB, ...). Ordering
// against the neighbouring MTE2/MTE3/DMA work is expressed in IR: the caller
// is a hivm.custom with SinglePipeOpTrait and MemoryEffectsOpInterface, so
// GraphSyncSolver models it as a PIPE_V read-write operation and injects the
// set/wait_flag pairs itself. A pipe_barrier in this inline callee would drain
// every pipe on the AIV core while staying invisible to the scheduler.
//===----------------------------------------------------------------------===//
constexpr uint32_t SIN_SIMT_THREAD_NUM = 1024;

template <typename DTYPE> __simt_vf__ LAUNCH_BOUND(SIN_SIMT_THREAD_NUM)
__aiv__ __attribute__((always_inline)) void
simtSinCompute(__ubuf__ DTYPE *src, __ubuf__ DTYPE *dst, const int64_t total,
               const __ubuf__ uint32_t *tbl) {
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < total;
       i += static_cast<int64_t>(blockDim.x)) {
    dst[i] = simt_sin_callee<DTYPE>(src[i], tbl);
  }
}

// Rank-generic kernel: the element offset is reconstructed from the memref
// sizes/strides, so non-unit innermost strides and non-contiguous views are
// handled. Only used for rank >= 2; rank 1 keeps the cheaper contiguous
// kernel above. Dimensions above RANK are passed as extent 1 / stride 0.
template <typename DTYPE, int RANK>
__simt_vf__ LAUNCH_BOUND(SIN_SIMT_THREAD_NUM)
__aiv__ __attribute__((always_inline)) void
simtSinComputeND(__ubuf__ DTYPE *src, __ubuf__ DTYPE *dst, const int64_t total,
                 const int64_t s0, const int64_t s1, const int64_t s2,
                 const int64_t s3, const int64_t s4, const int64_t ss0,
                 const int64_t ss1, const int64_t ss2, const int64_t ss3,
                 const int64_t ss4, const int64_t ds0, const int64_t ds1,
                 const int64_t ds2, const int64_t ds3, const int64_t ds4,
                 const __ubuf__ uint32_t *tbl) {
  const int64_t sizes[SIN_SIMT_MAX_RANK] = {s0, s1, s2, s3, s4};
  const int64_t srcStrides[SIN_SIMT_MAX_RANK] = {ss0, ss1, ss2, ss3, ss4};
  const int64_t dstStrides[SIN_SIMT_MAX_RANK] = {ds0, ds1, ds2, ds3, ds4};
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
    dst[dstOffset] = simt_sin_callee<DTYPE>(src[srcOffset], tbl);
  }
}

// Extent of dimension `dim`, with 1 for dimensions the rank does not have.
template <int RANK, typename MemRefT>
__aiv__ __attribute__((always_inline)) int64_t sinSimtExtent(const MemRefT *ref,
                                                             int dim) {
  return dim < RANK ? ref->sizes[dim] : 1;
}

// Stride of dimension `dim`, with 0 for dimensions the rank does not have
// (their extent is 1, so the stride is never used).
template <int RANK, typename MemRefT>
__aiv__ __attribute__((always_inline)) int64_t sinSimtStride(const MemRefT *ref,
                                                             int dim) {
  return dim < RANK ? ref->strides[dim] : 0;
}

template <typename DTYPE, int RANK>
__aiv__ __attribute__((always_inline)) void
sinSimtImpl(memref_t<__ubuf__ DTYPE, RANK> *src,
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
    cce::async_invoke<simtSinCompute<DTYPE>>(cce::dim3{SIN_SIMT_THREAD_NUM},
                                             srcPtr, dstPtr, total, tblPtr);
  } else {
    // The SIMT callee processes one element per index, so the count is the
    // full number of elements, not the extent of the outermost dimension.
    int64_t total = 1;
    for (int dim = 0; dim < RANK; ++dim)
      total *= src->sizes[dim];
    cce::async_invoke<simtSinComputeND<DTYPE, RANK>>(
        cce::dim3{SIN_SIMT_THREAD_NUM}, srcPtr, dstPtr, total,
        sinSimtExtent<RANK>(src, 0), sinSimtExtent<RANK>(src, 1),
        sinSimtExtent<RANK>(src, 2), sinSimtExtent<RANK>(src, 3),
        sinSimtExtent<RANK>(src, 4), sinSimtStride<RANK>(src, 0),
        sinSimtStride<RANK>(src, 1), sinSimtStride<RANK>(src, 2),
        sinSimtStride<RANK>(src, 3), sinSimtStride<RANK>(src, 4),
        sinSimtStride<RANK>(dst, 0), sinSimtStride<RANK>(dst, 1),
        sinSimtStride<RANK>(dst, 2), sinSimtStride<RANK>(dst, 3),
        sinSimtStride<RANK>(dst, 4), tblPtr);
  }
}

extern "C" {
REGISTER_SIMT_OP(sin, bfloat16_t, bfloat16_t);
REGISTER_SIMT_OP(sin, half, half);
REGISTER_SIMT_OP(sin, float, float);

#define DEFINE_SIN_SIMT_ENTRY(RANK)                                            \
  __aiv__ __attribute__((visibility("default"), always_inline)) void           \
      _mlir_ciface_sin_fp32_simt_##RANK##d(                                    \
          memref_t<__ubuf__ float, RANK> *src,                                 \
          memref_t<__ubuf__ int32_t, 1> *tbl,                                  \
          memref_t<__ubuf__ float, RANK> *dst) {                               \
    sinSimtImpl<float, RANK>(src, tbl, dst);                                   \
  }

#if SIN_SIMT_MAX_RANK >= 1
DEFINE_SIN_SIMT_ENTRY(1)
#endif
#if SIN_SIMT_MAX_RANK >= 2
DEFINE_SIN_SIMT_ENTRY(2)
#endif
#if SIN_SIMT_MAX_RANK >= 3
DEFINE_SIN_SIMT_ENTRY(3)
#endif
#if SIN_SIMT_MAX_RANK >= 4
DEFINE_SIN_SIMT_ENTRY(4)
#endif
#if SIN_SIMT_MAX_RANK >= 5
DEFINE_SIN_SIMT_ENTRY(5)
#endif

#undef DEFINE_SIN_SIMT_ENTRY
}
#endif

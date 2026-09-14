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

#include "DMA/DMAUtils.h"
#include "Utils.h"
#include "Vector/Cumulative/CumOpUtils.h"
#include "Vector/Cumulative/CumsumUtils.h"
#include "Vector/VecUtils.h"

/// cumsum ra op description:
/// Returns the cumulative sum of elements of input in the first axis.
///
/// \param src (type: memref<a x b x T>)
/// \param dst (type: memref<a x b x T>)
///
/// Constraints:
/// 1. cumsum ra op supports int16_t, int32_t, float16 and float32 types.
/// 2. cumsum ra op only accepts 2d type memrefs as src and dst.
/// 3. r axis should be aligned to ub_block_unit.
/// 4. a axis should be continuous.
/// 5. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
template <typename T, bool reverse>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  static_assert_supported_type<T>();
  check_inputs_of_cumop<T, 2>(src, dst);
  vector_cum_op_ra<VectorOpTy::VADD, T, reverse>(src, dst);
}

template <typename T, bool reverse>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_ara(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst) {
  static_assert_supported_type<T>();
  check_inputs_of_cumop<T, 3>(src, dst);
  vector_cum_op_ara<VectorOpTy::VADD, T, reverse>(src, dst);
}

/// cumsum 1d sklansky op description (membase):
/// Computes the inclusive prefix sum of a 1D buffer using a scalar Sklansky
/// scan tree, matching the regbase SIMT Sklansky floating-point addition
/// order to guarantee bit-identical results.
///
/// The SIMT path runs a two-level Sklansky scan inside each 1024-element
/// block: (1) an intra-warp scan over 32 lanes, then (2) a cross-warp scan
/// over the per-warp totals. This scalar port reproduces the exact same
/// parenthesisation by running the same Sklansky tree in a serial loop —
/// 32-element groups play the role of warps, and the staged cross-group
/// carry add mimics the cross-warp __shfl reduction. L > 1024 tiles into
/// 1024-element blocks with a serial cross-block Sklansky carry.
///
/// \param src (type: memref<N x T>)
/// \param dst (type: memref<N x T>)
///
/// Constraints:
/// 1. Currently supports half (f16) and float (f32) types.
/// 2. Only 1D memrefs with cumDim == 0, non-reverse.
/// 3. The start pointer address should be aligned to ub_block_unit.

// ---- Sklansky scan primitives (ported from regbase regbuf path) ----
// These are pure-scalar, no SIMT primitives. They produce the exact same
// addition tree as the regbase _scan{8,16}_sklansky_reg functions.

template <typename T>
__aiv__ __attribute__((always_inline)) void
_scan8_sklansky_scalar(__ubuf__ T *s, __ubuf__ T *d, T &carry) {
  T x0 = s[0], x1 = s[1], x2 = s[2], x3 = s[3], x4 = s[4], x5 = s[5], x6 = s[6],
    x7 = s[7];

  // level 1: groups of 2
  x1 = x1 + x0;
  x3 = x3 + x2;
  x5 = x5 + x4;
  x7 = x7 + x6;

  // level 2: groups of 4
  x2 = x2 + x1;
  x3 = x3 + x1;
  x6 = x6 + x5;
  x7 = x7 + x5;

  // level 3: groups of 8
  x4 = x4 + x3;
  x5 = x5 + x3;
  x6 = x6 + x3;
  x7 = x7 + x3;

  T c = carry;
  x0 = x0 + c;
  x1 = x1 + c;
  x2 = x2 + c;
  x3 = x3 + c;
  x4 = x4 + c;
  x5 = x5 + c;
  x6 = x6 + c;
  x7 = x7 + c;

  d[0] = x0;
  d[1] = x1;
  d[2] = x2;
  d[3] = x3;
  d[4] = x4;
  d[5] = x5;
  d[6] = x6;
  d[7] = x7;

  carry = x7;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
_scan16_sklansky_scalar(__ubuf__ T *s, __ubuf__ T *d, T &carry) {
  T x0 = s[0], x1 = s[1], x2 = s[2], x3 = s[3];
  T x4 = s[4], x5 = s[5], x6 = s[6], x7 = s[7];
  T x8 = s[8], x9 = s[9], x10 = s[10], x11 = s[11];
  T x12 = s[12], x13 = s[13], x14 = s[14], x15 = s[15];

  // level 1: groups of 2
  x1 = x1 + x0;
  x3 = x3 + x2;
  x5 = x5 + x4;
  x7 = x7 + x6;
  x9 = x9 + x8;
  x11 = x11 + x10;
  x13 = x13 + x12;
  x15 = x15 + x14;

  // level 2: groups of 4
  x2 = x2 + x1;
  x3 = x3 + x1;
  x6 = x6 + x5;
  x7 = x7 + x5;
  x10 = x10 + x9;
  x11 = x11 + x9;
  x14 = x14 + x13;
  x15 = x15 + x13;

  // level 3: groups of 8
  x4 = x4 + x3;
  x5 = x5 + x3;
  x6 = x6 + x3;
  x7 = x7 + x3;
  x12 = x12 + x11;
  x13 = x13 + x11;
  x14 = x14 + x11;
  x15 = x15 + x11;

  // level 4: groups of 16
  x8 = x8 + x7;
  x9 = x9 + x7;
  x10 = x10 + x7;
  x11 = x11 + x7;
  x12 = x12 + x7;
  x13 = x13 + x7;
  x14 = x14 + x7;
  x15 = x15 + x7;

  T c = carry;
  x0 = x0 + c;
  x1 = x1 + c;
  x2 = x2 + c;
  x3 = x3 + c;
  x4 = x4 + c;
  x5 = x5 + c;
  x6 = x6 + c;
  x7 = x7 + c;
  x8 = x8 + c;
  x9 = x9 + c;
  x10 = x10 + c;
  x11 = x11 + c;
  x12 = x12 + c;
  x13 = x13 + c;
  x14 = x14 + c;
  x15 = x15 + c;

  d[0] = x0;
  d[1] = x1;
  d[2] = x2;
  d[3] = x3;
  d[4] = x4;
  d[5] = x5;
  d[6] = x6;
  d[7] = x7;
  d[8] = x8;
  d[9] = x9;
  d[10] = x10;
  d[11] = x11;
  d[12] = x12;
  d[13] = x13;
  d[14] = x14;
  d[15] = x15;

  carry = x15;
}

// ---- regbuf scan: chunked 16/8/1 tail (same as regbase sklansky_regbuf_16)
// ----
template <typename T>
__aiv__ __attribute__((always_inline)) void
sklansky_regbuf_16_scalar(__ubuf__ T *src, __ubuf__ T *dst, int sz) {
  if (sz < 1)
    return;
  T carry = 0;
  int n = sz;
  __ubuf__ T *s = src;
  __ubuf__ T *d = dst;
  while (n >= 16) {
    _scan16_sklansky_scalar<T>(s, d, carry);
    s += 16;
    d += 16;
    n -= 16;
  }
  while (n >= 8) {
    _scan8_sklansky_scalar<T>(s, d, carry);
    s += 8;
    d += 8;
    n -= 8;
  }
  while (n-- > 0) {
    carry = carry + *s;
    *d = carry;
    s++;
    d++;
  }
}

// ---- scalar two-level Sklansky block scan ----
// Mirrors the regbase simt_sklansky_scan_1d_block_unroll_p2:
//   stage 1: intra-group inclusive scan (32-element groups, rolled loop
//            mimicking the warp shuffle scan).
//   stage 2: cross-group Sklansky scan on per-group totals (mimicking the
//            cross-warp __shfl reduction), staged so each group picks up
//            the exclusive carry from the boundary group.
// The addition order is identical to the SIMT float path → bit-exact.
constexpr int CUMSUM_SCALAR_GROUP = 32; // mirrors CUMSUM_WARP

template <typename T>
__aiv__ __attribute__((always_inline)) void
scalar_sklansky_scan_block(__ubuf__ T *src, __ubuf__ T *dst, int L,
                           T *warpTotals) {
  if (L < 1)
    return;

  // Small sizes: use the regbuf path (identical addition tree, less overhead).
  if (L <= 64) {
    sklansky_regbuf_16_scalar<T>(src, dst, L);
    return;
  }

  int numGroups = (L + CUMSUM_SCALAR_GROUP - 1) / CUMSUM_SCALAR_GROUP;
  if (numGroups < 1)
    numGroups = 1;

  // stage 1: per-group intra scan + collect group totals.
  // Mirrors the SIMT intra-warp Sklansky: for each group of 32, run the
  // h = 1,2,4,8,16 Sklansky levels in-place on dst, then publish the last
  // element as the group total.
  for (int g = 0; g < numGroups; g++) {
    int base = g * CUMSUM_SCALAR_GROUP;
    int len = (g == numGroups - 1) ? (L - base) : CUMSUM_SCALAR_GROUP;
    __ubuf__ T *s = src + base;
    __ubuf__ T *d = dst + base;

    // Load group into local registers.
    T v[CUMSUM_SCALAR_GROUP];
    for (int i = 0; i < len; i++)
      v[i] = s[i];
    for (int i = len; i < CUMSUM_SCALAR_GROUP; i++)
      v[i] = T(0);

    // Intra-group Sklansky scan (h = 1,2,4,8,16).
    for (int h = 1; h < CUMSUM_SCALAR_GROUP; h <<= 1) {
      int span = h << 1;
      for (int lane = 0; lane < CUMSUM_SCALAR_GROUP; lane++) {
        if ((lane & (span - 1)) >= h) {
          int boundary = (lane & ~(span - 1)) + h - 1;
          v[lane] = v[lane] + v[boundary];
        }
      }
    }

    // Write back scanned values.
    for (int i = 0; i < len; i++)
      d[i] = v[i];

    // Publish group total (last active element's inclusive scan).
    warpTotals[g] = v[len - 1];
  }

  // stage 2: cross-group Sklansky scan on warpTotals (in-place).
  // Mirrors the SIMT cross-warp unrolled reduction. The staged carry is
  // added to each group's elements afterwards.
  for (int h = 1; h < numGroups; h <<= 1) {
    int span = h << 1;
    for (int b = h; b < numGroups; b++) {
      if ((b & (span - 1)) >= h) {
        int boundary = (b & ~(span - 1)) + h - 1;
        warpTotals[b] = warpTotals[b] + warpTotals[boundary];
      }
    }
  }

  // stage 3: propagate exclusive carry to each group.
  // Group 0 has no carry; group g gets warpTotals[g-1] (the inclusive total
  // of the previous group) added to all its elements. This mirrors the SIMT
  // "add exclusive warp carry" step.
  for (int g = 1; g < numGroups; g++) {
    T carry = warpTotals[g - 1];
    int base = g * CUMSUM_SCALAR_GROUP;
    int len = (g == numGroups - 1) ? (L - base) : CUMSUM_SCALAR_GROUP;
    __ubuf__ T *d = dst + base;
    for (int i = 0; i < len; i++)
      d[i] = d[i] + carry;
  }
}

// ---- 1D entry point ----
// L <= 1024: single block, two-level scalar Sklansky.
// L >  1024: tile into 1024-element blocks + serial cross-block Sklansky
//            carry (mirrors regbase simt_sklansky_cumsum_1d tiling).
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_1d_sklansky(memref_t<__ubuf__ T, 1> *src,
                          memref_t<__ubuf__ T, 1> *dst) {
  int L = src->sizes[0];
  __ubuf__ T *sp = src->aligned + src->offset;
  __ubuf__ T *dp = dst->aligned + dst->offset;
  constexpr int BLK = CUMSUM_SCALAR_GROUP * CUMSUM_SCALAR_GROUP; // 1024

  if (L < 1)
    return;
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  // For A2/A3, L > ((192 * 256) // 2) will cause UB_OVERFLOW.
  assert((L < CUMSUM_SCALAR_GROUP * BLK) &&
      "vector_cumsum_1d_sklansky: L must be less than 32*1024=32768");
#endif

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);

  // On-stack scratch for per-group totals (max 32 groups per block).
  T warpTotals[CUMSUM_SCALAR_GROUP];

  if (L <= BLK) {
    scalar_sklansky_scan_block<T>(sp, dp, L, warpTotals);
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  // Multi-block tiling with cross-block Sklansky carry.
  // Forward only (non-reverse), head-anchored: block c covers [c*BLK, ...).
  int C = (L + BLK - 1) / BLK;
  int rem = L - (C - 1) * BLK;

  // step 1: per-block local scan.
  for (int c = 0; c < C; c++) {
    int len = (c < C - 1) ? BLK : rem;
    int base = c * BLK;
    scalar_sklansky_scan_block<T>(sp + base, dp + base, len, warpTotals);
  }

  // step 2: cross-block Sklansky carry propagation.
  // Block b inherits the inclusive total of the boundary block (its LAST
  // element) and adds it to all elements of block b.
  // We reuse dp as the block-total source: forward → last element of block.
  T blockTotals[CUMSUM_SCALAR_GROUP]; // max 32 blocks for L <= 32*1024 = 32768
  // Collect per-block totals.
  for (int c = 0; c < C; c++) {
    int len = (c < C - 1) ? BLK : rem;
    int base = c * BLK;
    blockTotals[c] = dp[base + len - 1];
  }

  // Sklansky scan on block totals.
  for (int h = 1; h < C; h <<= 1) {
    int span = h << 1;
    for (int b = h; b < C; b++) {
      if ((b & (span - 1)) >= h) {
        int boundary = (b & ~(span - 1)) + h - 1;
        blockTotals[b] = blockTotals[b] + blockTotals[boundary];
      }
    }
  }

  // step 3: propagate cross-block exclusive carry to each block.
  // Block 0 has no carry; block b gets blockTotals[b-1] added to all elements.
  for (int b = 1; b < C; b++) {
    T carry = blockTotals[b - 1];
    int len = (b < C - 1) ? BLK : rem;
    int base = b * BLK;
    __ubuf__ T *d = dp + base;
    for (int i = 0; i < len; i++)
      d[i] = d[i] + carry;
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

extern "C" {
//===-------------------------------------------------------------------===//
// cumsum 1d sklansky (membase), dim0
//===-------------------------------------------------------------------===//
REGISTE_CUMSUM_1D_SKLANSKY(float)

//===-------------------------------------------------------------------===//
// cumsum ra, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_CUMSUM(ra, 2, int16_t)
REGISTE_CUMSUM(ra, 2, int32_t)
REGISTE_CUMSUM(ra, 2, half)
REGISTE_CUMSUM(ra, 2, float)

REGISTE_REVERSE_CUMSUM(ra, 2, int16_t)
REGISTE_REVERSE_CUMSUM(ra, 2, int32_t)
REGISTE_REVERSE_CUMSUM(ra, 2, half)
REGISTE_REVERSE_CUMSUM(ra, 2, float)
//===-------------------------------------------------------------------===//
// cumsum ara, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_CUMSUM(ara, 3, int16_t)
REGISTE_CUMSUM(ara, 3, int32_t)
REGISTE_CUMSUM(ara, 3, half)
REGISTE_CUMSUM(ara, 3, float)

REGISTE_REVERSE_CUMSUM(ara, 3, int16_t)
REGISTE_REVERSE_CUMSUM(ara, 3, int32_t)
REGISTE_REVERSE_CUMSUM(ara, 3, half)
REGISTE_REVERSE_CUMSUM(ara, 3, float)
}

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

// SIMT 1D cumsum via the SKLANSKY prefix network.
// Owns _mlir_ciface_cumsum_1d_<dtype>_dim0 — the symbols HIVMToStandard already
// emits for 1D cumsum — so the existing lowering reaches this SIMT path with no
// MLIR changes. One block of up to 32 warps runs a two-level Sklansky scan
// (intra-warp scan + warp-carry scan); L > 1024 tiles into 1024-element blocks
// with a serial cross-block carry.
//
// Scratch / temp: per-warp totals need block-shared storage. The op-allocated
// `temp` buffer (memref<ceil(N/32)xT>, threaded in via the cumsum cwrapper)
// is the scratch; each block gets `scratch + base/32` for non-overlapping
// per-warp slices.

#include "RegBase/Cumulative/Cumulative.h"
#include "Utils.h"

#include <type_traits>

#if defined(__DAV_C310__)

constexpr int CUMSUM_WARP = 32;

// Integer cumsum dtypes: addition is associative, so the fast two-level scan
// (simt_sklansky_scan_1d_block_int) is order-independent and bit-exact. Floats
// (half / float / bfloat16_t) need the layered scan to match the canonical
// Sklansky parenthesization. Listed explicitly via is_same_v (already used in
// this backend) to avoid any ambiguity in classifying the half/bf16 carrier
// types.
template <typename T>
inline constexpr bool kCumsumIsInt =
    std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>;

template <typename T>
__aiv__ __attribute__((always_inline)) void
_scan8_sklansky_reg(__ubuf__ T *s, __ubuf__ T *d, T &carry) {
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
_scan16_sklansky_reg(__ubuf__ T *s, __ubuf__ T *d, T &carry) {
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

template <typename T>
__aiv__ __attribute__((always_inline)) void
_scan8_sklansky_reg_reverse(__ubuf__ T *s, __ubuf__ T *d, T &carry) {
  T x0 = s[0], x1 = s[1], x2 = s[2], x3 = s[3];
  T x4 = s[4], x5 = s[5], x6 = s[6], x7 = s[7];

  // level 1
  x0 = x0 + x1;
  x2 = x2 + x3;
  x4 = x4 + x5;
  x6 = x6 + x7;

  // level 2
  // right-half totals are x2 and x6
  x0 = x0 + x2;
  x1 = x1 + x2;
  x4 = x4 + x6;
  x5 = x5 + x6;

  // level 3
  // right-half total is x4
  x0 = x0 + x4;
  x1 = x1 + x4;
  x2 = x2 + x4;
  x3 = x3 + x4;

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

  carry = x0;
}
template <typename T>
__aiv__ __attribute__((always_inline)) void
_scan16_sklansky_reg_reverse(__ubuf__ T *s, __ubuf__ T *d, T &carry) {
  T x0 = s[0], x1 = s[1], x2 = s[2], x3 = s[3];
  T x4 = s[4], x5 = s[5], x6 = s[6], x7 = s[7];
  T x8 = s[8], x9 = s[9], x10 = s[10], x11 = s[11];
  T x12 = s[12], x13 = s[13], x14 = s[14], x15 = s[15];

  // level 1: groups of 2, suffix within each pair
  // [a0,a1] -> [a0+a1,a1]
  x0 = x0 + x1;
  x2 = x2 + x3;
  x4 = x4 + x5;
  x6 = x6 + x7;
  x8 = x8 + x9;
  x10 = x10 + x11;
  x12 = x12 + x13;
  x14 = x14 + x15;

  // level 2: groups of 4
  // right-half totals are x2, x6, x10, x12
  x0 = x0 + x2;
  x1 = x1 + x2;
  x4 = x4 + x6;
  x5 = x5 + x6;
  x8 = x8 + x10;
  x9 = x9 + x10;
  x12 = x12 + x14;
  x13 = x13 + x14;

  // level 3: groups of 8
  // right-half totals are x4 and x12
  x0 = x0 + x4;
  x1 = x1 + x4;
  x2 = x2 + x4;
  x3 = x3 + x4;

  x8 = x8 + x12;
  x9 = x9 + x12;
  x10 = x10 + x12;
  x11 = x11 + x12;

  // level 4: groups of 16
  // right-half total is x8
  x0 = x0 + x8;
  x1 = x1 + x8;
  x2 = x2 + x8;
  x3 = x3 + x8;
  x4 = x4 + x8;
  x5 = x5 + x8;
  x6 = x6 + x8;
  x7 = x7 + x8;

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

  // suffix scan 后，本 chunk 的总和在 x0
  carry = x0;
}

template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
sklansky_regbuf_16(__ubuf__ PromoteDataType *src, __ubuf__ PromoteDataType *dst,
                   int sz) {
  using T = PromoteDataType;
  if (sz < 1)
    return;
  T carry = 0;
  int n = sz;
  __ubuf__ T *s = src;
  __ubuf__ T *d = dst;
  while (n >= 16) {
    _scan16_sklansky_reg<T>(s, d, carry);
    s += 16;
    d += 16;
    n -= 16;
  }
  while (n >= 8) {
    _scan8_sklansky_reg<T>(s, d, carry);
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

template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
sklansky_regbuf_16_reverse(__ubuf__ PromoteDataType *src,
                           __ubuf__ PromoteDataType *dst, int sz) {
  using T = PromoteDataType;
  if (sz < 1)
    return;
  T carry = 0;
  int n = sz;
  __ubuf__ T *stail = src + sz;
  __ubuf__ T *dtail = dst + sz;
  while (n >= 16) {
    stail -= 16;
    dtail -= 16;
    n -= 16;
    _scan16_sklansky_reg_reverse<T>(stail, dtail, carry);
  }
  while (n >= 8) {
    stail -= 8;
    dtail -= 8;
    n -= 8;
    _scan8_sklansky_reg_reverse<T>(stail, dtail, carry);
  }
  while (n-- > 0) {
    --stail;
    --dtail;
    carry = *stail + carry;
    *dtail = carry;
  }
}

template <typename T> __simt_callee__ inline T wsh_idx(T v, int srcLane) {
  if constexpr (sizeof(T) <= 4) {
    int raw = 0;
    __builtin_memcpy(&raw, &v, sizeof(T));
    raw = __shfl(raw, srcLane);
    T out;
    __builtin_memcpy(&out, &raw, sizeof(T));
    return out;
  } else {
    long long raw = 0;
    __builtin_memcpy(&raw, &v, sizeof(T));
    raw = __shfl(raw, srcLane);
    T out;
    __builtin_memcpy(&out, &raw, sizeof(T));
    return out;
  }
}

// ---- variant: ONLY phase 2 (cross-warp) unrolled; intra-warp scan stays
// rolled.
//      Isolates how much of the unroll win comes from the cross-warp stages.
//      Bit-identical to simt_sklansky_scan_1d_block. ----
template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) __aiv__
    __attribute__((always_inline)) static void simt_sklansky_scan_1d_block_unroll_p2(
        __ubuf__ T *src, __ubuf__ T *dst, int L, int reverse,
        __ubuf__ T *scratch) {
  int lane = threadIdx.x;
  int warpId = threadIdx.y;
  int numWarps = blockDim.y;
  int gid = warpId * CUMSUM_WARP + lane;
  int pos = reverse ? (L - 1 - gid) : gid;
  bool active = gid < L;
  T v = active ? src[pos] : T(0);
  __ubuf__ T *warpStageTotals = scratch;

  // stage 1: intra-warp inclusive scan (ROLLED).
  for (int h = 1; h < CUMSUM_WARP; h <<= 1) {
    int span = h << 1;
    int boundary = (lane & ~(span - 1)) + h - 1;
    T u = wsh_idx(v, boundary);
    if ((lane & (span - 1)) >= h)
      v += u;
  }

  // phase 1: publish each warp's total.
  {
    int lastLane = CUMSUM_WARP - 1;
    if (warpId == numWarps - 1) {
      int activeLanes = L % CUMSUM_WARP;
      if (activeLanes != 0)
        lastLane = activeLanes - 1;
    }
    T warpTotal = wsh_idx(v, lastLane);
    if (lane == 0) {
      warpStageTotals[warpId] = warpTotal;
    }
  }
  __syncthreads(); // the only block barrier

  // phase 2: redundant cross-warp Sklansky, manually unrolled (warpH =
  // 1,2,4,8,16). Each stage: fold the matching boundary warp's running total
  // into v, and replay the same Sklansky level on the warp-total vector t so
  // the next stage's boundary total is correct. Hand-expanded (no macro) so
  // codecheck stays clean.
  {
    T t = (lane < numWarps) ? warpStageTotals[lane] : T(0);

    // warpH = 1, span = 2
    if (numWarps > 1) {
      const int span = 2;
      // addend shuffle is warp-uniform (guard depends only on warpId) ->
      // hoisting it inside the guard keeps all 32 lanes in lock-step (still a
      // valid all-lanes
      // __shfl) while skipping the shuffle for warps that don't need the carry.
      if ((warpId & (span - 1)) >= 1) {
        int boundaryWarp = (warpId & ~(span - 1)) + 1 - 1;
        T addend = wsh_idx(t, boundaryWarp);
        v += addend;
      }
      // the `u` replay shuffle stays UNCONDITIONAL: its guard is lane-varying
      // and its source lane tb is itself a non-participating lane -> a
      // conditional shuffle there would be an undefined partial-warp __shfl.
      int tb = (lane & ~(span - 1)) + 1 - 1;
      T u = wsh_idx(t, tb);
      if ((lane & (span - 1)) >= 1)
        t += u;
    }

    // warpH = 2, span = 4
    if (numWarps > 2) {
      const int span = 4;
      if ((warpId & (span - 1)) >= 2) {
        int boundaryWarp = (warpId & ~(span - 1)) + 2 - 1;
        T addend = wsh_idx(t, boundaryWarp);
        v += addend;
      }
      int tb = (lane & ~(span - 1)) + 2 - 1;
      T u = wsh_idx(t, tb);
      if ((lane & (span - 1)) >= 2)
        t += u;
    }

    // warpH = 4, span = 8
    if (numWarps > 4) {
      const int span = 8;
      if ((warpId & (span - 1)) >= 4) {
        int boundaryWarp = (warpId & ~(span - 1)) + 4 - 1;
        T addend = wsh_idx(t, boundaryWarp);
        v += addend;
      }
      int tb = (lane & ~(span - 1)) + 4 - 1;
      T u = wsh_idx(t, tb);
      if ((lane & (span - 1)) >= 4)
        t += u;
    }

    // warpH = 8, span = 16
    if (numWarps > 8) {
      const int span = 16;
      if ((warpId & (span - 1)) >= 8) {
        int boundaryWarp = (warpId & ~(span - 1)) + 8 - 1;
        T addend = wsh_idx(t, boundaryWarp);
        v += addend;
      }
      int tb = (lane & ~(span - 1)) + 8 - 1;
      T u = wsh_idx(t, tb);
      if ((lane & (span - 1)) >= 8)
        t += u;
    }

    // warpH = 16, span = 32
    if (numWarps > 16) {
      const int span = 32;
      if ((warpId & (span - 1)) >= 16) {
        int boundaryWarp = (warpId & ~(span - 1)) + 16 - 1;
        T addend = wsh_idx(t, boundaryWarp);
        v += addend;
      }
      int tb = (lane & ~(span - 1)) + 16 - 1;
      T u = wsh_idx(t, tb);
      if ((lane & (span - 1)) >= 16)
        t += u;
    }
  }

  if (active)
    dst[pos] = v;
}
// ---- cross-block carry adder ----
// Adds carryPtr[0] to every element of blk[0..len-1] with stride `stride`.
template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) __aiv__
    __attribute__((always_inline)) static void simt_add_carry_block(
        __ubuf__ T *blk, int len, __ubuf__ T *carryPtr) {
  T carry = carryPtr[0];
  int tid = threadIdx.y * CUMSUM_WARP + threadIdx.x;
  if (tid < len)
    blk[tid] += carry;
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) __aiv__
    __attribute__((always_inline)) static void simt_sklansky_scan_1d_block_int(
        __ubuf__ T *src, __ubuf__ T *dst, int L, int reverse,
        __ubuf__ T *scratch) {
  int lane = threadIdx.x;
  int warpId = threadIdx.y;
  int numWarps = blockDim.y;
  int gid = warpId * CUMSUM_WARP + lane;
  int pos = reverse ? (L - 1 - gid) : gid;
  bool active = gid < L;
  T v = active ? src[pos] : T(0);
  // Per-warp scratch: a single slice scratch[0..numWarps-1] holds the per-warp
  // inclusive totals. Integer-only path: addition is associative, so this fast
  // two-level scan yields the exact same result as the layered float path —
  // summation order is irrelevant for integers. Same 32-slot scratch footprint
  // as simt_sklansky_scan_1d_block, so dispatch needs no extra buffer space.
  __ubuf__ T *warpTotals = scratch;

  // stage 1: intra-warp inclusive scan
  for (int h = 1; h < CUMSUM_WARP; h <<= 1) {
    int span = h << 1;
    int boundary = (lane & ~(span - 1)) + h - 1;
    T u = wsh_idx(v, boundary);
    if ((lane & (span - 1)) >= h)
      v += u;
  }
  // Warp total: for a full warp read lane 31; for the LAST partial warp read
  // the last ACTIVE lane (lane 31 may be inactive -> __shfl returns undefined).
  int lastLane = CUMSUM_WARP - 1;
  if (warpId == numWarps - 1) {
    int activeLanes = L % CUMSUM_WARP;
    if (activeLanes != 0)
      lastLane = activeLanes - 1;
  }
  T warpTotal = wsh_idx(v, lastLane);
  if (lane == 0) {
    warpTotals[warpId] = warpTotal;
  }
  __syncthreads();

  // stage 2: warp 0 scans the per-warp totals into inclusive per-warp totals
  // (in-place; the scan runs in registers so there is no read/write hazard).
  if (warpId == 0) {
    T t = (lane < numWarps) ? warpTotals[lane] : T(0);
    for (int h = 1; h < CUMSUM_WARP; h <<= 1) {
      int span = h << 1;
      int boundary = (lane & ~(span - 1)) + h - 1;
      T u = wsh_idx(t, boundary);
      if ((lane & (span - 1)) >= h)
        t += u;
    }
    if (lane < numWarps)
      warpTotals[lane] = t;
  }
  __syncthreads();

  // stage 3: each warp adds the exclusive carry = inclusive total of the
  // previous warp (warp 0 adds nothing). Reads warpTotals[warpId-1] directly —
  // no separate carries buffer needed.
  if (warpId > 0)
    v += warpTotals[warpId - 1];
  if (active)
    dst[pos] = v;
}

// ---- 1D entry point ----
// L ≤ 1024: single block, two-level Sklansky.
// L > 1024: tile into 1024-element blocks + serial cross-block carry.
template <typename T>
__aiv__ __attribute__((always_inline)) void
simt_sklansky_cumsum_1d(memref_t<__ubuf__ T, 1> *src,
                        memref_t<__ubuf__ T, 1> *dst,
                        memref_t<__ubuf__ T, 1> *temp, bool reverse) {
  int L = src->sizes[0];
  __ubuf__ T *sp = src->aligned + src->offset;
  __ubuf__ T *dp = dst->aligned + dst->offset;
  constexpr int BLK = CUMSUM_WARP * CUMSUM_WARP; // 1024 elements per block

  // Per-warp scratch: the op-allocated temp buffer (ceil(N/32) slots).
  pipe_barrier(PIPE_ALL);
  __ubuf__ T *scratch = temp->aligned + temp->offset;
  if (L <= BLK) {
    if (L <= 64) {
      if (reverse)
        sklansky_regbuf_16_reverse<T>(sp, dp, static_cast<int>(L));
      else
        sklansky_regbuf_16<T>(sp, dp, static_cast<int>(L));
      pipe_barrier(PIPE_ALL);
      return;
    }
    unsigned nw = (L + CUMSUM_WARP - 1) / CUMSUM_WARP;
    if (nw < 1)
      nw = 1;
    // Integer dtypes use the fast two-level scan (order-independent); floats
    // use the layered scan to preserve canonical Sklansky parenthesization.
    if constexpr (kCumsumIsInt<T>) {
      cce::async_invoke<simt_sklansky_scan_1d_block_int<T>>(
          cce::dim3{CUMSUM_WARP, nw, 1}, sp, dp, L, reverse ? 1 : 0, scratch);
    } else {
      cce::async_invoke<simt_sklansky_scan_1d_block_unroll_p2<T>>(
          cce::dim3{CUMSUM_WARP, nw, 1}, sp, dp, L, reverse ? 1 : 0, scratch);
    }
    pipe_barrier(PIPE_ALL);
    return;
  }

  int C = (L + BLK - 1) / BLK;
  int rem = L - (C - 1) * BLK; // size of the partial block (== L%BLK, or BLK)

  // Block partition.
  //   Forward: head-anchored — block c covers [c*BLK, ...), the partial block
  //   is LAST. Reverse: TAIL-anchored — full blocks are packed against the tail
  //   and the partial block sits at the HEAD.  Reverse within a block is a
  //   suffix scan, and the global reverse cumsum is canonical flip∘forward∘flip
  //   Sklansky.  When L is not a multiple of BLK, head-anchored blocking puts
  //   the block boundaries at the wrong offsets vs that flip layout, so the
  //   cross-block carry order diverges from the CPU Sklansky golden (verified:
  //   N=8523 reverse hit 19 ULP mismatches).  Tail-anchoring aligns the block
  //   grid with flip∘forward∘flip and reproduces the golden bit-for-bit.
  //   baseOf(c) = forward ? c*BLK : (c < C-1 ? L-(c+1)*BLK : 0);  lenOf(c) =
  //   c<C-1 ? BLK : rem.
  // scratch is sliced per block index (c*CUMSUM_WARP); tail-anchored bases are
  // not 32-aligned so base/CUMSUM_WARP could alias — c*CUMSUM_WARP gives clean
  // 32-slot slices.

  // step1: per-block local scan (forward prefix / reverse suffix within the
  // block).
  for (int c = 0; c < C; c++) {
    int len = (c < C - 1) ? BLK : rem;
    int base = reverse ? ((c < C - 1) ? (L - (c + 1) * BLK) : 0) : (c * BLK);
    unsigned nw = (len + CUMSUM_WARP - 1) / CUMSUM_WARP;
    if constexpr (kCumsumIsInt<T>) {
      cce::async_invoke<simt_sklansky_scan_1d_block_int<T>>(
          cce::dim3{CUMSUM_WARP, nw, 1}, sp + base, dp + base, len,
          reverse ? 1 : 0, scratch + c * CUMSUM_WARP);
    } else {
      cce::async_invoke<simt_sklansky_scan_1d_block_unroll_p2<T>>(
          cce::dim3{CUMSUM_WARP, nw, 1}, sp + base, dp + base, len,
          reverse ? 1 : 0, scratch + c * CUMSUM_WARP);
    }
  }
  pipe_barrier(PIPE_ALL);

  // step2: one forward Sklansky network over the block index for BOTH
  // directions (reverse is already folded into the tail-anchored layout above).
  // Block b inherits the boundary block's scan-completed total: forward → its
  // LAST element, reverse → its FIRST element (= whole-block suffix total).
  for (int h = 1; h < C; h <<= 1) {
    int span = h << 1;
    for (int b = h; b < C; b++) {
      if ((b & (span - 1)) >= (unsigned)h) {
        int boundary = (b & ~(span - 1)) + h - 1;
        int blen = (b < C - 1) ? BLK : rem;
        int bbase =
            reverse ? ((b < C - 1) ? (L - (b + 1) * BLK) : 0) : (b * BLK);
        int obase = reverse
                        ? ((boundary < C - 1) ? (L - (boundary + 1) * BLK) : 0)
                        : (boundary * BLK);
        int olen = (boundary < C - 1) ? BLK : rem;
        __ubuf__ T *carryPtr = reverse ? (dp + obase) : (dp + obase + olen - 1);
        unsigned nwc = (blen + CUMSUM_WARP - 1) / CUMSUM_WARP;
        cce::async_invoke<simt_add_carry_block<T>>(
            cce::dim3{CUMSUM_WARP, nwc, 1}, dp + bbase, blen, carryPtr);
      }
      pipe_barrier(PIPE_ALL);
    }
  }
  pipe_barrier(PIPE_ALL);
}

// ---- register all 9 dtypes via the project cwrapper ----
// Hand-expanded per dtype (no local macro) so codecheck stays clean.
extern "C" {
DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(1, int8_t, 0) {
  simt_sklansky_cumsum_1d<int8_t>(src, dst, temp, reverse);
}
DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(1, uint8_t, 0) {
  simt_sklansky_cumsum_1d<uint8_t>(src, dst, temp, reverse);
}
DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(1, int16_t, 0) {
  simt_sklansky_cumsum_1d<int16_t>(src, dst, temp, reverse);
}
DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(1, uint16_t, 0) {
  simt_sklansky_cumsum_1d<uint16_t>(src, dst, temp, reverse);
}
DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(1, int32_t, 0) {
  simt_sklansky_cumsum_1d<int32_t>(src, dst, temp, reverse);
}
DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(1, uint32_t, 0) {
  simt_sklansky_cumsum_1d<uint32_t>(src, dst, temp, reverse);
}
DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(1, half, 0) {
  simt_sklansky_cumsum_1d<half>(src, dst, temp, reverse);
}
DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(1, float, 0) {
  simt_sklansky_cumsum_1d<float>(src, dst, temp, reverse);
}
DECLARE_CUMSUM_WITHOUT_DEFAULT_VALUE(1, bfloat16_t, 0) {
  simt_sklansky_cumsum_1d<bfloat16_t>(src, dst, temp, reverse);
}
}

#endif // __DAV_C310__

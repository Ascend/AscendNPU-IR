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

#include "DMA/DMAUtils.h"
#include "RegBase/Cumulative/Cumulative.h"
#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"

#if defined(__DAV_C310__)

enum cum_minmax_kind {
  CUM_MM_MAX = 0,
  CUM_MM_MIN = 1,
  CUM_MM_PROP_NAN = 2,
};

template <int kind, typename T>
__aiv__ __attribute__((always_inline)) void
cum_mm_comb(VectorReg<T> &dst, VectorReg<T> &a, VectorReg<T> &b,
            vector_bool &mask) {
  if constexpr (!std::is_integral<T>::value) {
    VectorReg<T> savedA;
    vsel(savedA, a, a, mask);
    vector_bool aNaN, bNaN;
    vcmp_ne(aNaN, a, a, mask);
    vcmp_ne(bNaN, b, b, mask);
    vector_bool cmp;
    if constexpr ((kind & 1) == CUM_MM_MAX) {
      vcmp_ge(cmp, a, b, mask);
    } else {
      vcmp_le(cmp, a, b, mask);
    }
    vsel(dst, a, b, cmp);
    if constexpr ((kind & CUM_MM_PROP_NAN) != 0) {
      vsel(dst, savedA, dst, aNaN);
      vsel(dst, b, dst, bNaN);
    } else {
      vsel(dst, b, dst, aNaN);
      vsel(dst, savedA, dst, bNaN);
    }
  } else {
    vector_bool cmp;
    if constexpr ((kind & 1) == CUM_MM_MAX) {
      vcmp_gt(cmp, a, b, mask);
    } else {
      vcmp_lt(cmp, a, b, mask);
    }
    vsel(dst, a, b, cmp);
  }
}

template <int kind, typename T>
__aiv__ __attribute__((always_inline)) T cum_mm_comb_s(T a, T b) {
  if constexpr (!std::is_integral<T>::value) {
    if constexpr ((kind & CUM_MM_PROP_NAN) != 0) {
      // Propagate NaN (torch.cummax/cummin, arith::Maximum/MinimumFOp):
      // if either operand is NaN, return NaN.
      if (a != a) return a;
      if (b != b) return b;
    } else {
      // Ignore NaN (arith::MaxNum/MinNumFOp, ops-math):
      // if one operand is NaN, return the other.
      if (a != a) return b;
      if (b != b) return a;
    }
  }
  if constexpr ((kind & 1) == CUM_MM_MAX) {
    return (a > b) ? a : b;
  } else {
    return (a < b) ? a : b;
  }
}

//===----------------------------------------------------------------------===//
// 3d scan along dim0 (rows are contiguous tiles, 2-way interleaved for ILP)
//===----------------------------------------------------------------------===//
template <typename T, int kind>
__aiv__ __attribute__((always_inline)) void
oneway_cum_minmax(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                  bool reverse) {
  int32_t rFactor = src->sizes[0];
  int32_t mFactor = src->sizes[1];
  int32_t nFactor = src->sizes[2];
  int32_t dupSize_ = CEIL_FACTOR(nFactor * sizeof(T), 32);
  int32_t nAddFactor = dupSize_ / sizeof(T);
  int32_t realDupSize = nAddFactor * sizeof(T) * mFactor;
  uint16_t nLoop = CEIL_DIV(realDupSize, REG_REGISTER_SIZE);
  uint16_t rLoop = (uint16_t)rFactor;
  int32_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
  int32_t stride0 = src->strides[0];
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  int32_t start_row_offset = reverse ? (rFactor - 1) * stride0 : 0;
  uint32_t totalElements = (uint32_t)nAddFactor * (uint32_t)mFactor;
  // When stride0 < nAddFactor*mFactor (e.g. 2D→3D reshape where stride0 =
  // nFactor, not the padded nAddFactor), the padding lanes in the tile map to
  // the next row's first element(s). Capping totalElements at stride0 ensures
  // the tail-tile mask never overruns into the next row.
  if (totalElements > (uint32_t)stride0) {
    totalElements = (uint32_t)stride0;
  }
  // Only full register tiles may use a full mask: the tail tile must be
  // masked exactly, otherwise its store overruns into the next row and
  // corrupts data that has not been scanned yet.
  uint16_t fullTiles = (uint16_t)(totalElements / (uint32_t)num_per_reg);
  uint16_t jQuad = fullTiles & ~(uint16_t)3; // multiple-of-4 full tiles
  uint32_t fullLanes = (uint32_t)num_per_reg;
  __VEC_SCOPE__ {
    VectorReg<T> a0, a1, a2, a3, x0, x1, x2, x3;
    vector_bool mask;
    CREATE_MASK_BY_SIZE(mask, T, fullLanes);
    // Four independent accumulator chains hide the combine latency.
    for (uint16_t j = 0; j < jQuad; j += 4) {
      int32_t o0 = (int32_t)j * num_per_reg;
      int32_t o1 = o0 + num_per_reg;
      int32_t o2 = o1 + num_per_reg;
      int32_t o3 = o2 + num_per_reg;
      vlds(a0, src_ptr, start_row_offset + o0, NORM);
      vlds(a1, src_ptr, start_row_offset + o1, NORM);
      vlds(a2, src_ptr, start_row_offset + o2, NORM);
      vlds(a3, src_ptr, start_row_offset + o3, NORM);
      vsts(a0, dst_ptr, start_row_offset + o0, NORM_B32, mask);
      vsts(a1, dst_ptr, start_row_offset + o1, NORM_B32, mask);
      vsts(a2, dst_ptr, start_row_offset + o2, NORM_B32, mask);
      vsts(a3, dst_ptr, start_row_offset + o3, NORM_B32, mask);
      for (uint16_t r = 1; r < rLoop; r++) {
        int32_t base = (reverse ? (rFactor - 1 - r) : (int32_t)r) * stride0;
        vlds(x0, src_ptr + base, o0, NORM);
        vlds(x1, src_ptr + base, o1, NORM);
        vlds(x2, src_ptr + base, o2, NORM);
        vlds(x3, src_ptr + base, o3, NORM);
        cum_mm_comb<kind, T>(a0, a0, x0, mask);
        cum_mm_comb<kind, T>(a1, a1, x1, mask);
        cum_mm_comb<kind, T>(a2, a2, x2, mask);
        cum_mm_comb<kind, T>(a3, a3, x3, mask);
        vsts(a0, dst_ptr + base, o0, NORM_B32, mask);
        vsts(a1, dst_ptr + base, o1, NORM_B32, mask);
        vsts(a2, dst_ptr + base, o2, NORM_B32, mask);
        vsts(a3, dst_ptr + base, o3, NORM_B32, mask);
      }
    }
    // Remaining full tiles and the (exactly masked) tail tile.
    for (uint16_t j = jQuad; j < nLoop; j++) {
      int32_t o0 = (int32_t)j * num_per_reg;
      uint32_t lanes = totalElements - (uint32_t)j * (uint32_t)num_per_reg;
      if (lanes > (uint32_t)num_per_reg) {
        lanes = (uint32_t)num_per_reg;
      }
      CREATE_MASK_BY_SIZE(mask, T, lanes);
      vlds(a0, src_ptr, start_row_offset + o0, NORM);
      vsts(a0, dst_ptr, start_row_offset + o0, NORM_B32, mask);
      for (uint16_t r = 1; r < rLoop; r++) {
        int32_t base = (reverse ? (rFactor - 1 - r) : (int32_t)r) * stride0;
        vlds(x0, src_ptr + base, o0, NORM);
        cum_mm_comb<kind, T>(a0, a0, x0, mask);
        vsts(a0, dst_ptr + base, o0, NORM_B32, mask);
      }
    }
  }
}

// Sklansky scan along dim0, in place on dst. max/min are idempotent and
// associative, so the tree reordering is bit-exact; unlike float cumsum we
// can always use it. Work is nLoop*R*log(R)/2 ops but depth is only log(R),
// so it wins over the latency-bound sequential chain when nLoop is small.
template <typename T, int kind>
__aiv__ __attribute__((always_inline)) void
sklansky_cum_minmax_round(memref_t<__ubuf__ T, 3> *dst, sklansky_param_t param) {
  int32_t groupTailCount = param.groupTailCount;
  uint16_t groupTailCountu16 = (uint16_t)groupTailCount;
  uint16_t addTailCountu16 = (uint16_t)param.addTailCount;
  uint16_t addCount = (uint16_t)param.addCount;
  int32_t startOffset = param.startOffset;
  int32_t group_offset = param.group_offset;
  int32_t groupMainCount = param.groupMainCount;
  uint16_t groupMainCountu16 = (uint16_t)groupMainCount;
  int32_t flag = param.flag;
  constexpr uint16_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
  __ubuf__ T *ptr = dst->aligned + dst->offset;
  uint16_t nLoop = CEIL_DIV(param.realDupSize, REG_REGISTER_SIZE);
  uint16_t stride0 = (uint16_t)dst->strides[0];
  uint32_t totalElements = (uint32_t)param.nAddFactor * (uint32_t)param.mFactor;
  if (totalElements > (uint32_t)stride0) {
    totalElements = (uint32_t)stride0;
  }
  __VEC_SCOPE__ {
    VectorReg<T> x1, x2;
    vector_bool mask;
    for (uint16_t j = 0; j < nLoop; j++) {
      // Mask the tail tile exactly: rows are stride-contiguous, so an
      // overrunning store would corrupt the next, not-yet-scanned row.
      uint32_t lanes = totalElements - (uint32_t)j * (uint32_t)num_per_reg;
      if (lanes > (uint32_t)num_per_reg) {
        lanes = (uint32_t)num_per_reg;
      }
      CREATE_MASK_BY_SIZE(mask, T, lanes);
      for (uint16_t m = 0; m < groupMainCountu16; m++) {
        int32_t src_offset = flag * m * group_offset + j * num_per_reg;
        vlds(x1, ptr + startOffset, src_offset, NORM);
        for (uint16_t n = 1; n <= addCount; n++) {
          int32_t dst_offset =
              flag * (m * group_offset + n * stride0) + j * num_per_reg;
          vlds(x2, ptr + startOffset, dst_offset, NORM);
          cum_mm_comb<kind, T>(x2, x1, x2, mask);
          vsts(x2, ptr + startOffset, dst_offset, NORM_B32, mask);
        }
      }
      for (uint16_t m = 0; m < groupTailCountu16; m++) {
        int32_t src_offset = flag * groupMainCount * group_offset + j * num_per_reg;
        vlds(x1, ptr + startOffset, src_offset, NORM);
        for (uint16_t n = 1; n <= addTailCountu16; n++) {
          int32_t dst_offset =
              flag * (groupMainCount * group_offset + n * stride0) +
              j * num_per_reg;
          vlds(x2, ptr + startOffset, dst_offset, NORM);
          cum_mm_comb<kind, T>(x2, x1, x2, mask);
          vsts(x2, ptr + startOffset, dst_offset, NORM_B32, mask);
        }
      }
    }
  }
}

template <typename T, int kind>
__aiv__ __attribute__((always_inline)) void
sklansky_cum_minmax(memref_t<__ubuf__ T, 3> *dst, bool reverse) {
  int32_t rFactor = dst->sizes[0];
  int32_t mFactor = dst->sizes[1];
  int32_t nFactor = dst->sizes[2];
  int32_t dupSize_ = CEIL_FACTOR(nFactor * sizeof(T), 32);
  int32_t nAddFactor = dupSize_ / sizeof(T);
  sklansky_param_t param;
  param.flag = reverse ? -1 : 1;
  param.mFactor = mFactor;
  param.nAddFactor = nAddFactor;
  param.realDupSize = nAddFactor * sizeof(T) * mFactor;
  int32_t rounds = CeilLog2(rFactor);
  for (int32_t k = 0; k < rounds; k++) {
    int32_t addLoopPow = 1 << k;
    int32_t nextAddLoopPow = addLoopPow << 1;
    param.startOffset = (param.flag == -1)
                            ? (rFactor - addLoopPow) * dst->strides[0]
                            : (addLoopPow - 1) * dst->strides[0];
    param.addCount = addLoopPow;
    param.group_offset = nextAddLoopPow * dst->strides[0];
    param.groupMainCount = rFactor / nextAddLoopPow;
    param.groupTailCount = (rFactor % nextAddLoopPow > addLoopPow) ? 1 : 0;
    param.addTailCount =
        rFactor - param.groupMainCount * nextAddLoopPow - addLoopPow;
    sklansky_cum_minmax_round<T, kind>(dst, param);
  }
}

// Direct scan along dim1, no transposes and no extra buffer. Each (r, n-tile)
// pair is an independent chain over m, so chains across r hide the combine
// latency; rows are loaded/stored contiguously.
template <typename T, int kind>
__aiv__ __attribute__((always_inline)) void
dim1_scan_direct(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                 bool reverse) {
  int32_t R = src->sizes[0];
  int32_t M = src->sizes[1];
  int32_t N = src->sizes[2];
  int32_t s0 = src->strides[0];
  int32_t s1 = src->strides[1];
  constexpr int32_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
  uint16_t nTiles = (uint16_t)CEIL_DIV(N, num_per_reg);
  uint16_t Mu = (uint16_t)M;
  uint16_t rQuad = (uint16_t)(R & ~3);
  uint16_t Ru = (uint16_t)R;
  int32_t startM = reverse ? (M - 1) * s1 : 0;
  __ubuf__ T *sp = src->aligned + src->offset;
  __ubuf__ T *dp = dst->aligned + dst->offset;
  __VEC_SCOPE__ {
    VectorReg<T> a0, a1, a2, a3, x0, x1, x2, x3;
    vector_bool mask;
    for (uint16_t j = 0; j < nTiles; j++) {
      int32_t o = (int32_t)j * num_per_reg;
      uint32_t lanes = (uint32_t)((N - o < num_per_reg) ? (N - o) : num_per_reg);
      CREATE_MASK_BY_SIZE(mask, T, lanes);
      // Four independent r-chains per iteration for latency hiding.
      for (uint16_t r = 0; r < rQuad; r += 4) {
        int32_t b0 = (int32_t)r * s0 + o;
        int32_t b1 = b0 + s0;
        int32_t b2 = b1 + s0;
        int32_t b3 = b2 + s0;
        vlds(a0, sp + b0, startM, NORM);
        vlds(a1, sp + b1, startM, NORM);
        vlds(a2, sp + b2, startM, NORM);
        vlds(a3, sp + b3, startM, NORM);
        vsts(a0, dp + b0, startM, NORM_B32, mask);
        vsts(a1, dp + b1, startM, NORM_B32, mask);
        vsts(a2, dp + b2, startM, NORM_B32, mask);
        vsts(a3, dp + b3, startM, NORM_B32, mask);
        for (uint16_t m = 1; m < Mu; m++) {
          int32_t mo = (reverse ? (M - 1 - (int32_t)m) : (int32_t)m) * s1;
          vlds(x0, sp + b0, mo, NORM);
          vlds(x1, sp + b1, mo, NORM);
          vlds(x2, sp + b2, mo, NORM);
          vlds(x3, sp + b3, mo, NORM);
          cum_mm_comb<kind, T>(a0, a0, x0, mask);
          cum_mm_comb<kind, T>(a1, a1, x1, mask);
          cum_mm_comb<kind, T>(a2, a2, x2, mask);
          cum_mm_comb<kind, T>(a3, a3, x3, mask);
          vsts(a0, dp + b0, mo, NORM_B32, mask);
          vsts(a1, dp + b1, mo, NORM_B32, mask);
          vsts(a2, dp + b2, mo, NORM_B32, mask);
          vsts(a3, dp + b3, mo, NORM_B32, mask);
        }
      }
      for (uint16_t r = rQuad; r < Ru; r++) { // 1..3 remainder chains
        int32_t b0 = (int32_t)r * s0 + o;
        vlds(a0, sp + b0, startM, NORM);
        vsts(a0, dp + b0, startM, NORM_B32, mask);
        for (uint16_t m = 1; m < Mu; m++) {
          int32_t mo = (reverse ? (M - 1 - (int32_t)m) : (int32_t)m) * s1;
          vlds(x0, sp + b0, mo, NORM);
          cum_mm_comb<kind, T>(a0, a0, x0, mask);
          vsts(a0, dp + b0, mo, NORM_B32, mask);
        }
      }
    }
  }
}

template <typename T, int dim, int kind>
__aiv__ __attribute__((always_inline)) void
compute_cum_minmax_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                      bool reverse, memref_t<__ubuf__ T, 3> *temp = nullptr) {
  constexpr int32_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
  if constexpr (dim == 0) {
    int32_t rFactor = (int32_t)src->sizes[0];
    int32_t nAligned =
        (int32_t)(CEIL_FACTOR(src->sizes[2] * sizeof(T), 32) / sizeof(T));
    int32_t tiles =
        (int32_t)CEIL_DIV(nAligned * (int32_t)src->sizes[1], num_per_reg);
    copy_for_cum_op(src, dst);
    // Few register tiles -> the sequential chain is latency bound; the
    // Sklansky tree (depth log R) is faster despite the extra work.
    if (tiles <= 2 && rFactor >= 8) {
      sklansky_cum_minmax<T, kind>(dst, reverse);
    } else {
      oneway_cum_minmax<T, kind>(dst, dst, reverse);
    }
  } else if constexpr (dim == 1) {
    int32_t N = (int32_t)src->sizes[2];
    // Direct scan wins unless lane utilization collapses (tiny N, where the
    // transpose packs rows densely); it also needs no temp buffer.
    if (temp == nullptr || N * 4 >= num_per_reg) {
      dim1_scan_direct<T, kind>(src, dst, reverse);
      return;
    }
    int64_t n_aligned = CEIL_FACTOR(temp->sizes[2] * sizeof(T), 32) / sizeof(T);
    memref_t<__ubuf__ T, 3> temp_3d{
        temp->allocated,
        temp->aligned,
        temp->offset,
        {temp->sizes[1], temp->sizes[0], temp->sizes[2]},
        {temp->sizes[0] * n_aligned, n_aligned, 1}};
    transpose_dim_01<T>(src, &temp_3d);
    oneway_cum_minmax<T, kind>(&temp_3d, &temp_3d, reverse);
    transpose_dim_01<T>(&temp_3d, dst);
  }
}

template <typename T, int cum_dim, int kind>
__aiv__ __attribute__((always_inline)) void
vector_cum_minmax_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                     memref_t<__ubuf__ T, 3> *temp, bool reverse) {
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
          std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
          std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
          std::is_same<T, float>::value || std::is_same<T, half>::value ||
          std::is_same<T, bfloat16_t>::value,
      "cummax/cummin op only support uint8/16/32_t, int8/16/32_t, float16/32 "
      "and bfloat16 type operands in template!");
  compute_cum_minmax_3d<T, cum_dim, kind>(src, dst, reverse, temp);
}

//===----------------------------------------------------------------------===//
// 2d: dim0 reshapes to 3d; dim1 transposes then scans dim0
//===----------------------------------------------------------------------===//
template <typename T, int cum_dim, int kind>
__aiv__ __attribute__((always_inline)) void
vector_cum_minmax_2d(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
                     memref_t<__ubuf__ T, 2> *temp, bool reverse) {
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
          std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
          std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
          std::is_same<T, float>::value || std::is_same<T, half>::value ||
          std::is_same<T, bfloat16_t>::value,
      "cummax/cummin op only support uint8/16/32_t, int8/16/32_t, float16/32 "
      "and bfloat16 type operands in template!");

  if constexpr (cum_dim == 0) {
    memref_t<__ubuf__ T, 3> src_3d{
        src->allocated,
        src->aligned,
        src->offset,
        {src->sizes[0], 1, src->sizes[1]},
        {src->strides[0], src->strides[0], src->strides[1]}};
    memref_t<__ubuf__ T, 3> dst_3d{
        dst->allocated,
        dst->aligned,
        dst->offset,
        {dst->sizes[0], 1, dst->sizes[1]},
        {dst->strides[0], dst->strides[0], dst->strides[1]}};
    compute_cum_minmax_3d<T, 0, kind>(&src_3d, &dst_3d, reverse);
  } else {
    memref_t<__ubuf__ T, 2> temp_trans{temp->allocated,
                                       temp->aligned,
                                       temp->offset,
                                       {src->sizes[1], src->sizes[0]},
                                       {temp->strides[0], temp->strides[1]}};
    if constexpr (sizeof(T) == 1) {
      transpose_ar2ra_i8<T>(src, &temp_trans);
      vector_cum_minmax_2d<T, 0, kind>(&temp_trans, &temp_trans, nullptr,
                                       reverse);
      transpose_ar2ra_i8<T>(&temp_trans, dst);
    } else {
      transpose_ar2ra<T>(src, &temp_trans);
      vector_cum_minmax_2d<T, 0, kind>(&temp_trans, &temp_trans, nullptr,
                                       reverse);
      transpose_ar2ra<T>(&temp_trans, dst);
    }
  }
}

//===----------------------------------------------------------------------===//
// 1d
//===----------------------------------------------------------------------===//
template <typename T, int kind>
__aiv__ __attribute__((always_inline)) T
scalar_incl_prefix_mm_4way(__ubuf__ T *src, __ubuf__ T *dst, int32_t N) {
  T acc = src[0];
  dst[0] = acc;
  int32_t i = 1;
  int32_t imain = 1 + ((N - 1) & ~3);
  for (; i < imain; i += 4) {
    T t0 = src[i + 0];
    T t1 = src[i + 1];
    T t2 = src[i + 2];
    T t3 = src[i + 3];
    T s01 = cum_mm_comb_s<kind, T>(t0, t1);
    T s012 = cum_mm_comb_s<kind, T>(s01, t2);
    T s0123 = cum_mm_comb_s<kind, T>(s012, t3);
    dst[i + 0] = cum_mm_comb_s<kind, T>(acc, t0);
    dst[i + 1] = cum_mm_comb_s<kind, T>(acc, s01);
    dst[i + 2] = cum_mm_comb_s<kind, T>(acc, s012);
    dst[i + 3] = cum_mm_comb_s<kind, T>(acc, s0123);
    acc = cum_mm_comb_s<kind, T>(acc, s0123);
  }
  for (; i < N; i++) {
    acc = cum_mm_comb_s<kind, T>(acc, src[i]);
    dst[i] = acc;
  }
  return acc;
}

template <typename T, int kind>
__aiv__ __attribute__((always_inline)) T
scalar_incl_suffix_mm_4way(__ubuf__ T *src, __ubuf__ T *dst, int32_t N) {
  T acc = src[N - 1];
  dst[N - 1] = acc;
  int32_t i = N - 2;
  for (; i >= 3; i -= 4) {
    T t0 = src[i];
    T t1 = src[i - 1];
    T t2 = src[i - 2];
    T t3 = src[i - 3];
    T s01 = cum_mm_comb_s<kind, T>(t0, t1);
    T s012 = cum_mm_comb_s<kind, T>(s01, t2);
    T s0123 = cum_mm_comb_s<kind, T>(s012, t3);
    dst[i] = cum_mm_comb_s<kind, T>(acc, t0);
    dst[i - 1] = cum_mm_comb_s<kind, T>(acc, s01);
    dst[i - 2] = cum_mm_comb_s<kind, T>(acc, s012);
    dst[i - 3] = cum_mm_comb_s<kind, T>(acc, s0123);
    acc = cum_mm_comb_s<kind, T>(acc, s0123);
  }
  for (; i >= 0; i--) {
    acc = cum_mm_comb_s<kind, T>(acc, src[i]);
    dst[i] = acc;
  }
  return acc;
}

// Two-phase 1d scan: scalar per-register prefix + SIMD cross-register carry.
// Phase 1 uses the proven-correct scalar 4-way scan (no vgather2 — the only
// intrinsic that sporadically miscomputes on this hardware). Phase 2 uses only
// aligned vlds/vsts and vdup which are fully reliable.
template <typename T, int kind>
__aiv__ __attribute__((always_inline)) void
unified_scan_mm_fwd_nocopy(__ubuf__ T *srcBase, __ubuf__ T *dstBase,
                           __ubuf__ T * /*tempBase*/, int N, int BpE) {
  using IdxT = std::conditional_t<sizeof(T) == 2, int16_t, int32_t>;
  using uIdxT = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
  int numBlocks = (N + BpE - 1) / BpE;
  int tailSize = N - (numBlocks - 1) * BpE;
  int last = numBlocks - 1;
  // Phase 1: scalar per-register prefix scan. Each register is scanned
  // independently; after this, lane j in register r = op(src[r*BpE..r*BpE+j]).
  for (int r = 0; r < numBlocks; ++r) {
    int sz = (r == last) ? tailSize : BpE;
    (void)scalar_incl_prefix_mm_4way<T, kind>(
        srcBase + r * BpE, dstBase + r * BpE, static_cast<int32_t>(sz));
  }
  // Phase 2: SIMD cross-register carry. carry = running prefix over all earlier
  // registers; one broadcast-combine per register completes the global prefix.
  if (numBlocks > 1) {
    uint32_t bpeU = static_cast<uint32_t>(BpE);
    INTRINSIC(pipe_barrier, PIPE_ALL);
    __VEC_SCOPE__ {
      VectorReg<T> v, carry;
      VectorReg<IdxT> idx;
      vector_bool m, mFull;
      vci(idx, 0);
      CREATE_MASK_BY_SIZE(mFull, T, bpeU);
      // Seed: register 0's highest lane = op(src[0..BpE-1]).
      vlds(v, dstBase, 0, NORM);
      vdup(carry, v, mFull, POS_HIGHEST, MODE_ZEROING);
      uint16_t cntU = static_cast<uint16_t>(numBlocks - 1);
      for (uint16_t k = 0; k < cntU; ++k) {
        int r = 1 + static_cast<int>(k);
        int sz = (r == last) ? tailSize : BpE;
        __ubuf__ T *rb = dstBase + r * BpE;
        if (sz == BpE) {
          vlds(v, rb, 0, NORM);
          cum_mm_comb<kind, T>(v, v, carry, mFull);
          vsts(v, rb, 0, NORM_B32, mFull);
        } else {
          uint32_t szU = static_cast<uint32_t>(sz);
          CREATE_MASK_BY_SIZE(m, T, szU);
          vgather2(v, rb, (VectorReg<uIdxT> &)idx, m);
          cum_mm_comb<kind, T>(v, v, carry, m);
          vscatter(v, rb, (VectorReg<uIdxT> &)idx, m);
        }
        vdup(carry, v, mFull, POS_HIGHEST, MODE_ZEROING);
      }
    }
  }
}

template <typename T, int kind>
__aiv__ __attribute__((always_inline)) void
unified_scan_mm_rev_nocopy(__ubuf__ T *srcBase, __ubuf__ T *dstBase,
                           __ubuf__ T * /*tempBase*/, int N, int BpE) {
  using IdxT = std::conditional_t<sizeof(T) == 2, int16_t, int32_t>;
  using uIdxT = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
  int numBlocks = (N + BpE - 1) / BpE;
  int tailSize = N - (numBlocks - 1) * BpE;
  int last = numBlocks - 1;
  // Phase 1: scalar per-register suffix scan.
  for (int r = 0; r < numBlocks; ++r) {
    int sz = (r == last) ? tailSize : BpE;
    (void)scalar_incl_suffix_mm_4way<T, kind>(
        srcBase + r * BpE, dstBase + r * BpE, static_cast<int32_t>(sz));
  }
  // Phase 2: SIMD cross-register carry (high→low). carry = running suffix over
  // all later registers; one broadcast-combine per register completes the global
  // suffix. The last register is already fully scanned and seeds the carry.
  if (numBlocks > 1) {
    uint32_t bpeU = static_cast<uint32_t>(BpE);
    uint32_t tailU = static_cast<uint32_t>(tailSize);
    INTRINSIC(pipe_barrier, PIPE_ALL);
    __VEC_SCOPE__ {
      VectorReg<T> v, carry;
      VectorReg<IdxT> idx;
      vector_bool m, mFull;
      vci(idx, 0);
      CREATE_MASK_BY_SIZE(mFull, T, bpeU);
      // Seed: last register's lowest lane = op(src[(last)*BpE .. N-1]).
      if (tailSize == BpE) {
        vlds(v, dstBase + last * BpE, 0, NORM);
      } else {
        CREATE_MASK_BY_SIZE(m, T, tailU);
        vgather2(v, dstBase + last * BpE, (VectorReg<uIdxT> &)idx, m);
      }
      vdup(carry, v, mFull, POS_LOWEST, MODE_ZEROING);
      uint16_t cntU = static_cast<uint16_t>(numBlocks - 1);
      for (uint16_t k = 0; k < cntU; ++k) {
        int r = (numBlocks - 2) - static_cast<int>(k);
        __ubuf__ T *rb = dstBase + r * BpE;
        vlds(v, rb, 0, NORM);
        cum_mm_comb<kind, T>(v, v, carry, mFull);
        vsts(v, rb, 0, NORM_B32, mFull);
        vdup(carry, v, mFull, POS_LOWEST, MODE_ZEROING);
      }
    }
  }
}

template <typename T, int cum_dim, int kind>
__aiv__ __attribute__((always_inline)) void
vector_cum_minmax_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                     memref_t<__ubuf__ T, 1> *temp, bool reverse) {
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
          std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
          std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
          std::is_same<T, float>::value || std::is_same<T, half>::value ||
          std::is_same<T, bfloat16_t>::value,
      "cummax/cummin op only support uint8/16/32_t, int8/16/32_t, float16/32 "
      "and bfloat16 type operands in template!");
  constexpr int BpE = REG_REGISTER_SIZE / sizeof(T);
  int64_t N = dst->sizes[0];
  if (N <= 0) {
    return;
  }
  __ubuf__ T *sp = src->aligned + src->offset;
  __ubuf__ T *dp = dst->aligned + dst->offset;
  INTRINSIC(pipe_barrier, PIPE_ALL);
  if (N <= 2 * BpE) {
    if (!reverse) {
      (void)scalar_incl_prefix_mm_4way<T, kind>(sp, dp, (int32_t)N);
    } else {
      (void)scalar_incl_suffix_mm_4way<T, kind>(sp, dp, (int32_t)N);
    }
    INTRINSIC(pipe_barrier, PIPE_ALL);
    return;
  }
  if constexpr (sizeof(T) == 1) {
    if (!reverse) {
      (void)scalar_incl_prefix_mm_4way<T, kind>(sp, dp, (int32_t)N);
    } else {
      (void)scalar_incl_suffix_mm_4way<T, kind>(sp, dp, (int32_t)N);
    }
    INTRINSIC(pipe_barrier, PIPE_ALL);
  } else {
    __ubuf__ T *tp = temp->aligned + temp->offset;
    if (reverse) {
      unified_scan_mm_rev_nocopy<T, kind>(sp, dp, tp, static_cast<int>(N), BpE);
    } else {
      unified_scan_mm_fwd_nocopy<T, kind>(sp, dp, tp, static_cast<int>(N), BpE);
    }
    INTRINSIC(pipe_barrier, PIPE_ALL);
  }
}

//===----------------------------------------------------------------------===//
// Registrations
//===----------------------------------------------------------------------===//
extern "C" {
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 1, int8_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 1, uint8_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 1, int16_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 1, uint16_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 1, int32_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 1, uint32_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 1, half, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 1, float, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 1, bfloat16_t, 0)

REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, int8_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, uint8_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, int16_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, uint16_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, int32_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, uint32_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, half, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, float, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, bfloat16_t, 0)

REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, int8_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, uint8_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, int16_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, uint16_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, int32_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, uint32_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, half, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, float, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 2, bfloat16_t, 1)

REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, int8_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, uint8_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, int16_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, uint16_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, int32_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, uint32_t, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, half, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, float, 0)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, bfloat16_t, 0)

REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, int8_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, uint8_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, int16_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, uint16_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, int32_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, uint32_t, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, half, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, float, 1)
REGISTER_CUM_MINMAX(cummax, CUM_MM_MAX, 3, bfloat16_t, 1)

REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 1, int8_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 1, uint8_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 1, int16_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 1, uint16_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 1, int32_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 1, uint32_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 1, half, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 1, float, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 1, bfloat16_t, 0)

REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, int8_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, uint8_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, int16_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, uint16_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, int32_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, uint32_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, half, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, float, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, bfloat16_t, 0)

REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, int8_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, uint8_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, int16_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, uint16_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, int32_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, uint32_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, half, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, float, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 2, bfloat16_t, 1)

REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, int8_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, uint8_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, int16_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, uint16_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, int32_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, uint32_t, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, half, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, float, 0)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, bfloat16_t, 0)

REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, int8_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, uint8_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, int16_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, uint16_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, int32_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, uint32_t, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, half, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, float, 1)
REGISTER_CUM_MINMAX(cummin, CUM_MM_MIN, 3, bfloat16_t, 1)
}

#endif

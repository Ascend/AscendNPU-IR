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

#include "RegBase/Cumulative/Cumulative.h"
#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/Transpose/TransposeUtils.h"
#include "Vector/VecUtils.h"

#if defined(__DAV_C310__)

template <typename T>
__aiv__ __attribute__((always_inline)) void
cumprod_3d_u8(memref_t<__ubuf__ T, 3> *src,
              memref_t<__ubuf__ T, 3> *dst, bool reverse) {
  auto dst_offset = dst->offset;
  auto src_offset = src->offset;
  auto dst_aligned = dst->aligned;
  auto src_aligned = src->aligned;

  uint16_t src_size = (uint16_t)src->sizes[0];  // 规约轴长度
  uint32_t size0 = (uint32_t)src->sizes[1] * CEIL_FACTOR((uint32_t)src->sizes[2] * sizeof(T), 32);
  int64_t src_stride= src->strides[0];
  int64_t dst_stride= dst->strides[0];

  constexpr int num_per_reg = REG_REGISTER_SIZE / sizeof(T);
  uint16_t repeat_times = CEIL_DIV(size0, num_per_reg);

  uint32_t size0_temp0 = size0;
  int64_t src_first_row = reverse ? (int64_t)(src_size - 1) * src_stride : (int64_t)0;
  int64_t dst_first_row = reverse ? (int64_t)(src_size - 1) * dst_stride : (int64_t)0;
  for (uint16_t i = 0; i < repeat_times; i++) {
    __VEC_SCOPE__ {
      VectorReg<T> reg;
      vector_bool mask;
      CREATE_MASK_BY_SIZE(mask, T, size0_temp0);
      vlds(reg, src_aligned + src_offset + src_first_row, i * num_per_reg, NORM);
      vsts(reg, dst_aligned + dst_offset + dst_first_row, i * num_per_reg, NORM_B32, mask);
    }
  }

  for (uint16_t j = 1; j < src_size; ++j) {
    int64_t curr_idx = reverse ? (int64_t)(src_size - 1 - j) : (int64_t)j;
    int64_t prev_idx = reverse ? (int64_t)(src_size - j)     : (int64_t)(j - 1);
    int64_t curr_src_off = src_offset + curr_idx * src_stride;
    int64_t prev_dst_off = dst_offset + prev_idx * dst_stride;
    int64_t curr_dst_off = dst_offset + curr_idx * dst_stride;

    __VEC_SCOPE__ {
      uint32_t size0_temp1 = size0;
      vector_u16 src0_u16_even, src0_u16_odd;
      vector_u16 src1_u16_even, src1_u16_odd;
      vector_u16 dst_u16_even,  dst_u16_odd;
      VectorReg<T> src0_reg, src1_reg, dst_reg;
      vector_bool full_mask;

      for (uint16_t i = 0; i < repeat_times; i++) {
        // 读当前 src + 上一轮 dst 结果
        CREATE_MASK_BY_SIZE(full_mask, T, size0_temp1);
        vlds(src0_reg, src_aligned + curr_src_off, i * num_per_reg, NORM);
        vlds(src1_reg, dst_aligned + prev_dst_off, i * num_per_reg, NORM);

        // u8 -> u16 奇偶拆分
        vcvt(src0_u16_even, src0_reg, PART_EVEN);
        vcvt(src0_u16_odd,  src0_reg, PART_ODD);
        vcvt(src1_u16_even, src1_reg, PART_EVEN);
        vcvt(src1_u16_odd,  src1_reg, PART_ODD);

        // 乘法
        vmul(dst_u16_even, src0_u16_even, src1_u16_even, full_mask, MODE_ZEROING);
        vmul(dst_u16_odd,  src0_u16_odd,  src1_u16_odd,  full_mask, MODE_ZEROING);

        vcvt(dst_reg, dst_u16_even, PART_EVEN, MODE_MERGING);
        vcvt(dst_reg, dst_u16_odd,  PART_ODD,  MODE_MERGING);

        // 写回当前 dst
        vsts(dst_reg, dst_aligned + curr_dst_off, i * num_per_reg, NORM_B32, full_mask);
      }
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
cumprod_3d_byte(memref_t<__ubuf__ T, 3> *src,
                memref_t<__ubuf__ T, 3> *dst, bool reverse)
{
  if constexpr (std::is_same_v<T, uint8_t>) {
    cumprod_3d_u8(src, dst, reverse);
  } else {
    // Reinterpret the int8_t memref as uint8_t and delegate to the uint8 path
    memref_t<__ubuf__ uint8_t, 3> src_as_uint8;
    memref_t<__ubuf__ uint8_t, 3> dst_as_uint8;
    view_as<int8_t, uint8_t, 3>(src, &src_as_uint8);
    view_as<int8_t, uint8_t, 3>(dst, &dst_as_uint8);
    cumprod_3d_u8(&src_as_uint8, &dst_as_uint8, reverse);
  }
}

template <typename T>
 __aiv__ __attribute__((always_inline)) void
 oneway_cumprod(memref_t<__ubuf__ T, 3> *src,
                memref_t<__ubuf__ T, 3> *dst, bool reverse) {
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

  if constexpr (sizeof(T) == 1) {
    cumprod_3d_byte<T>(src, dst, reverse);
    return;
  }

  __VEC_SCOPE__ {
    // acc caches the cumulative product of the previous row; x2 holds the current row
    VectorReg<T> acc, x2;
    vector_bool mask;
    uint32_t totalElements = (uint32_t)nAddFactor * (uint32_t)mFactor;

    for (uint16_t j = 0; j < nLoop; j++) {
      CREATE_MASK_BY_SIZE(mask, T, totalElements);

      // Row 0 initialisation: load the first row as the starting point for accumulation

      vlds(acc, src_ptr, start_row_offset + j * num_per_reg, NORM);
      // Write the first row directly to the output
      vsts(acc, dst_ptr, start_row_offset + j * num_per_reg, NORM_B32, mask);

      // Iteratively accumulate starting from row 1

      for (uint16_t r = 1; r < rLoop; r++) {
        int32_t r_offset = reverse ? (rFactor - 1 - r) : r;

        // 1. Load the current row into x2

        vlds(x2, src_ptr + r_offset * stride0, j * num_per_reg, NORM);

        // 2. Standard vector multiply: acc = acc * x2 (no error compensation)

        vmul(acc, acc, x2, mask);

        // 3. Store the cumulative product of the current row to output

        vsts(acc, dst_ptr + r_offset * stride0, j * num_per_reg, NORM_B32, mask);
      }
    }
  }
}

/* =========================================================================
 * 2. compute_cumprod_3d
 *    dim=0 : copy then scan rows.
 *    dim=1 : transpose so dim-1 becomes dim-0, scan, transpose back.
 * =========================================================================
 */
template <typename T, int dim>
__aiv__ __attribute__((always_inline)) void
compute_cumprod_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                   bool reverse, memref_t<__ubuf__ T, 3> *temp = nullptr) {
  if constexpr (dim == 0) {
    copy_for_cum_op(src, dst);
    oneway_cumprod<T>(dst, dst, reverse);
  } else if constexpr (dim == 1) {
    int64_t n_aligned = CEIL_FACTOR(temp->sizes[2] * sizeof(T), 32) / sizeof(T);
    memref_t<__ubuf__ T, 3> temp_3d{
        temp->allocated,
        temp->aligned,
        temp->offset,
        {temp->sizes[1], temp->sizes[0], temp->sizes[2]},
        {temp->sizes[0] * n_aligned, n_aligned, 1}};
    transpose_dim_01<T>(src, &temp_3d);
    oneway_cumprod<T>(&temp_3d, &temp_3d, reverse);
    transpose_dim_01<T>(&temp_3d, dst);
  }
}

/* =========================================================================
 * 3. copy_lane_gather  -  helper used by 1-D scan
 * =========================================================================
 */
template <typename T, typename IdxT, typename uIdxT>
__aiv__ __attribute__((always_inline)) void
copy_lane_gather(__ubuf__ T *s, __ubuf__ T *d, int n) {
  uint32_t nn = static_cast<uint32_t>(n);
  __VEC_SCOPE__ {
    VectorReg<T>    a;
    VectorReg<IdxT> idx;
    vector_bool     m;
    CREATE_MASK_BY_SIZE(m, T, nn);
    vci(idx, 0);
    vgather2(a, s, (VectorReg<uIdxT> &)idx, m);
    vscatter(a, d, (VectorReg<uIdxT> &)idx, m);
  }
}

/* =========================================================================
 * 4. unified_scan_fwd_nocopy  -  forward Hillis-Steele product scan
 *    All vadd -> vmul vs. the cumprod version.
 * =========================================================================
 */
template <typename T>
__aiv__ __attribute__((always_inline)) void
unified_scan_fwd_nocopy(__ubuf__ T *srcBase, __ubuf__ T *dstBase, int N,
                        int BpE) {
  using IdxT  = std::conditional_t<sizeof(T) == 2, int16_t,  int32_t>;
  using uIdxT = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;

  int numBlocks = (N + BpE - 1) / BpE;
  int tailSize  = N - (numBlocks - 1) * BpE;
  int fullRegs  = (tailSize < BpE) ? (numBlocks - 1) : numBlocks;
  int last      = numBlocks - 1;

  for (int d = 1; d < N; d <<= 1) {
    __ubuf__ T *rbase = (d == 1) ? srcBase : dstBase;
    __ubuf__ T *wbase = dstBase;
    int  cf       = (d + BpE - 1) / BpE;
    bool tailG    = (tailSize < BpE) && (last * BpE + tailSize > d);
    bool strG     = (cf >= 1) && (cf * BpE > d) && (cf - 1 < fullRegs);
    int  tailKeep = (d > last * BpE) ? (d - last * BpE) : 0;

    __ubuf__ T *tailRd = rbase + last * BpE + tailKeep;
    __ubuf__ T *tailWd = wbase + last * BpE + tailKeep;
    uint32_t    tailN  = static_cast<uint32_t>(tailSize - tailKeep);
    __ubuf__ T *strRd  = rbase + d;
    __ubuf__ T *strWd  = wbase + d;
    uint32_t    strN   = static_cast<uint32_t>(cf * BpE - d);
    uint32_t    bpeU   = static_cast<uint32_t>(BpE);

    int      fwdCnt  = (fullRegs - cf > 0) ? (fullRegs - cf) : 0;
    uint16_t fwdCntU = static_cast<uint16_t>(fwdCnt);

    __VEC_SCOPE__ {
      VectorReg<T>    a, b;
      VectorReg<IdxT> idx;
      vector_align    valign;
      vector_bool     m, mFull;

      vci(idx, 0);
      CREATE_MASK_BY_SIZE(mFull, T, bpeU);

      if (tailG) {
        CREATE_MASK_BY_SIZE(m, T, tailN);
        vgather2(a, tailRd,     (VectorReg<uIdxT> &)idx, m);
        vgather2(b, tailRd - d, (VectorReg<uIdxT> &)idx, m);
        vmul(a, a, b, m);
        vscatter(a, tailWd, (VectorReg<uIdxT> &)idx, m);
      }

      for (uint16_t k = 0; k < fwdCntU; ++k) {
        int r = (fullRegs - 1) - static_cast<int>(k);
        __ubuf__ T *rb = rbase + r * BpE;
        __ubuf__ T *wb = wbase + r * BpE;
        vlds(a, rb, 0, NORM);
        vldas(valign, rb - d);
        vldus(b, valign, rb - d);
        vmul(a, a, b, mFull);
        vsts(a, wb, 0, NORM_B32, mFull);
      }

      if (strG) {
        CREATE_MASK_BY_SIZE(m, T, strN);
        vgather2(a, strRd,     (VectorReg<uIdxT> &)idx, m);
        vgather2(b, strRd - d, (VectorReg<uIdxT> &)idx, m);
        vmul(a, a, b, m);
        vscatter(a, strWd, (VectorReg<uIdxT> &)idx, m);
      }
    }

    if (d == 1) {
      copy_lane_gather<T, IdxT, uIdxT>(srcBase, dstBase, d);
    }
  }
}

/* =========================================================================
 * 5. unified_scan_rev_nocopy  -  reverse (suffix) product scan
 * =========================================================================
 */
template <typename T>
__aiv__ __attribute__((always_inline)) void
unified_scan_rev_nocopy(__ubuf__ T *srcBase, __ubuf__ T *dstBase, int N,
                        int BpE) {
  using IdxT  = std::conditional_t<sizeof(T) == 2, int16_t,  int32_t>;
  using uIdxT = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;

  int numBlocks = (N + BpE - 1) / BpE;
  int tailSize  = N - (numBlocks - 1) * BpE;
  int fullRegs  = (tailSize < BpE) ? (numBlocks - 1) : numBlocks;
  int last      = numBlocks - 1;

  for (int d = 1; d < N; d <<= 1) {
    __ubuf__ T *rbase = (d == 1) ? srcBase : dstBase;
    __ubuf__ T *wbase = dstBase;
    int dq         = d / BpE;
    int rFA        = (N - d >= BpE) ? (N - d - BpE) / BpE : -1;
    int rSF        = fullRegs - 2 - dq;
    int rCHi       = (rFA < rSF) ? rFA : rSF;
    int lastActive = (N - d - 1) / BpE;
    int gLo        = (rCHi + 1 > 0) ? (rCHi + 1) : 0;
    uint32_t bpeU  = static_cast<uint32_t>(BpE);

    int      revCCnt  = (rCHi >= 0) ? (rCHi + 1) : 0;
    uint16_t revCCntU = static_cast<uint16_t>(revCCnt);
    int      revGCnt  = (lastActive - gLo + 1 > 0) ? (lastActive - gLo + 1) : 0;
    uint16_t revGCntU = static_cast<uint16_t>(revGCnt);

    __VEC_SCOPE__ {
      VectorReg<T>    a, b;
      VectorReg<IdxT> idx;
      vector_align    valign;
      vector_bool     m, mFull;

      vci(idx, 0);
      CREATE_MASK_BY_SIZE(mFull, T, bpeU);

      for (uint16_t k = 0; k < revCCntU; ++k) {
        int r = static_cast<int>(k);
        __ubuf__ T *rb = rbase + r * BpE;
        __ubuf__ T *wb = wbase + r * BpE;
        vlds(a, rb, 0, NORM);
        vldas(valign, rb + d);
        vldus(b, valign, rb + d);
        vmul(a, a, b, mFull);
        vsts(a, wb, 0, NORM_B32, mFull);
      }

      for (uint16_t k = 0; k < revGCntU; ++k) {
        int r      = gLo + static_cast<int>(k);
        int rStart = r * BpE;
        int sz     = (r == last) ? tailSize : BpE;
        int hi     = rStart + sz;
        int addEnd = (hi < (N - d)) ? hi : (N - d);
        uint32_t nact = static_cast<uint32_t>(addEnd - rStart);

        CREATE_MASK_BY_SIZE(m, T, nact);
        __ubuf__ T *rd = rbase + rStart;
        __ubuf__ T *wd = wbase + rStart;
        vgather2(a, rd,     (VectorReg<uIdxT> &)idx, m);
        vgather2(b, rd + d, (VectorReg<uIdxT> &)idx, m);
        vmul(a, a, b, m);
        vscatter(a, wd, (VectorReg<uIdxT> &)idx, m);
      }
    }

    if (d == 1) {
      copy_lane_gather<T, IdxT, uIdxT>(srcBase + (N - d), dstBase + (N - d), d);
    }
  }
}

/* =========================================================================
 * 6. unified_scan_gather_nocopy  -  dispatcher
 * =========================================================================
 */
template <typename T>
__aiv__ __attribute__((always_inline)) void
unified_scan_gather_nocopy(__ubuf__ T *srcBase, __ubuf__ T *dstBase, int N,
                           int BpE, bool reverse) {
  if (reverse)
    unified_scan_rev_nocopy<T>(srcBase, dstBase, N, BpE);
  else
    unified_scan_fwd_nocopy<T>(srcBase, dstBase, N, BpE);
}

/* =========================================================================
 * 7. Scalar 4-way unrolled helpers for short / int8 arrays  (seed = 1)
 * =========================================================================
 */
template <typename T>
__aiv__ __attribute__((always_inline)) T
scalar_incl_prefix_prod_4way(__ubuf__ T *src, __ubuf__ T *dst, int32_t N,
                             T seed) {
  T acc     = seed;
  int32_t i = 0;
  int32_t imain = N - (N & 3);
  for (; i < imain; i += 4) {
    T T0 = src[i + 0], T1 = src[i + 1], T2 = src[i + 2], T3 = src[i + 3];
    T p01   = (T)(T0 * T1);
    T p012  = (T)(p01 * T2);
    T p0123 = (T)(p012 * T3);
    dst[i + 0] = (T)(acc * T0);
    dst[i + 1] = (T)(acc * p01);
    dst[i + 2] = (T)(acc * p012);
    dst[i + 3] = (T)(acc * p0123);
    acc = (T)(acc * p0123);
  }
  for (; i < N; i++) { acc = (T)(acc * src[i]); dst[i] = acc; }
  return acc;
}

template <typename T>
__aiv__ __attribute__((always_inline)) T
scalar_incl_suffix_prod_4way(__ubuf__ T *src, __ubuf__ T *dst, int32_t N,
                             T seed) {
  T acc     = seed;
  int32_t i = N - 1;
  for (; i >= 3; i -= 4) {
    T T0 = src[i], T1 = src[i-1], T2 = src[i-2], T3 = src[i-3];
    T p01   = (T)(T0 * T1);
    T p012  = (T)(p01 * T2);
    T p0123 = (T)(p012 * T3);
    dst[i]     = (T)(acc * T0);
    dst[i - 1] = (T)(acc * p01);
    dst[i - 2] = (T)(acc * p012);
    dst[i - 3] = (T)(acc * p0123);
    acc = (T)(acc * p0123);
  }
  for (; i >= 0; i--) { acc = (T)(acc * src[i]); dst[i] = acc; }
  return acc;
}

/* =========================================================================
 * 8. vector_cumprod_1d_twoway  -  1-D entry (seed = 1)
 * =========================================================================
 */
template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
vector_cumprod_1d_twoway(memref_t<__ubuf__ PromoteDataType, 1> *src,
                         memref_t<__ubuf__ PromoteDataType, 1> *dst,
                         bool reverse) {
  constexpr int BpE = REG_REGISTER_SIZE / sizeof(PromoteDataType);
  int64_t N = dst->sizes[0];

  INTRINSIC(pipe_barrier, PIPE_ALL);

  if (N <= 2 * BpE || sizeof(PromoteDataType) < 2) {
    if (!reverse)
      (void)scalar_incl_prefix_prod_4way<PromoteDataType>(
          src->aligned + src->offset, dst->aligned + dst->offset,
          (int32_t)N, (PromoteDataType)1);
    else
      (void)scalar_incl_suffix_prod_4way<PromoteDataType>(
          src->aligned + src->offset, dst->aligned + dst->offset,
          (int32_t)N, (PromoteDataType)1);
    INTRINSIC(pipe_barrier, PIPE_ALL);
    return;
  }

  if constexpr (sizeof(PromoteDataType) >= 2) {
    unified_scan_gather_nocopy<PromoteDataType>(
        src->aligned + src->offset, dst->aligned + dst->offset,
        static_cast<int>(N), BpE, reverse);
    INTRINSIC(pipe_barrier, PIPE_ALL);
  }
}

/* =========================================================================
 * 9. Public entry points
 * =========================================================================
 */
template <typename T, int cum_dim>
__aiv__ __attribute__((always_inline)) void
vector_cumprod_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                  memref_t<__ubuf__ T, 1> *temp, bool reverse) {
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
      std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
      std::is_same<T, int8_t>::value  || std::is_same<T, uint8_t>::value  ||
      std::is_same<T, float>::value   || std::is_same<T, half>::value     ||
      std::is_same<T, bfloat16_t>::value,
      "cumprod op only supports uint8/16/32_t, int8/16/32_t, "
      "float16/32 and bfloat16 type operands in template!");
  vector_cumprod_1d_twoway<T>(src, dst, reverse);
}

template <typename T, int cum_dim>
__aiv__ __attribute__((always_inline)) void
vector_cumprod_2d(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
                  memref_t<__ubuf__ T, 2> *temp, bool reverse) {
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
      std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
      std::is_same<T, int8_t>::value  || std::is_same<T, uint8_t>::value  ||
      std::is_same<T, float>::value   || std::is_same<T, half>::value     ||
      std::is_same<T, bfloat16_t>::value,
      "cumprod op only supports uint8/16/32_t, int8/16/32_t, "
      "float16/32 and bfloat16 type operands in template!");

  if constexpr (cum_dim == 0) {
    memref_t<__ubuf__ T, 3> src_3d{
        src->allocated, src->aligned, src->offset,
        {src->sizes[0], 1, src->sizes[1]},
        {src->strides[0], src->strides[0], src->strides[1]}};
    memref_t<__ubuf__ T, 3> dst_3d{
        dst->allocated, dst->aligned, dst->offset,
        {dst->sizes[0], 1, dst->sizes[1]},
        {dst->strides[0], dst->strides[0], dst->strides[1]}};
    compute_cumprod_3d<T, 0>(&src_3d, &dst_3d, reverse);
  } else {
    if constexpr (sizeof(T) == 1) {
      memref_t<__ubuf__ T, 2> temp_trans{
          temp->allocated, temp->aligned, temp->offset,
          {src->sizes[1], src->sizes[0]},
          {temp->strides[0], temp->strides[1]}};
      transpose_ar2ra_i8<T>(src, &temp_trans);
      vector_cumprod_2d<T, 0>(&temp_trans, &temp_trans, nullptr, reverse);
      transpose_ar2ra_i8<T>(&temp_trans, dst);
    } else {
      memref_t<__ubuf__ T, 2> temp_trans{
          temp->allocated, temp->aligned, temp->offset,
          {src->sizes[1], src->sizes[0]},
          {temp->strides[0], temp->strides[1]}};
      transpose_ar2ra<T>(src, &temp_trans);
      vector_cumprod_2d<T, 0>(&temp_trans, &temp_trans, nullptr, reverse);
      transpose_ar2ra<T>(&temp_trans, dst);
    }
  }
}

template <typename T, int cum_dim>
__aiv__ __attribute__((always_inline)) void
vector_cumprod_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                  memref_t<__ubuf__ T, 3> *temp, bool reverse) {
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
      std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
      std::is_same<T, int8_t>::value  || std::is_same<T, uint8_t>::value  ||
      std::is_same<T, float>::value   || std::is_same<T, half>::value     ||
      std::is_same<T, bfloat16_t>::value,
      "cumprod op only supports uint8/16/32_t, int8/16/32_t, "
      "float16/32 and bfloat16 type operands in template!");
  compute_cumprod_3d<T, cum_dim>(src, dst, reverse, temp);
}

/* =========================================================================
 * 10. Registration
 * =========================================================================
 */
extern "C" {
REGISTER_CUMPROD(1, int8_t,     0);
REGISTER_CUMPROD(1, uint8_t,    0);
REGISTER_CUMPROD(1, int16_t,    0);
REGISTER_CUMPROD(1, uint16_t,   0);
REGISTER_CUMPROD(1, int32_t,    0);
REGISTER_CUMPROD(1, uint32_t,   0);
REGISTER_CUMPROD(1, half,       0);
REGISTER_CUMPROD(1, float,      0);
REGISTER_CUMPROD(1, bfloat16_t, 0);

REGISTER_CUMPROD(2, int8_t,     0);
REGISTER_CUMPROD(2, uint8_t,    0);
REGISTER_CUMPROD(2, int16_t,    0);
REGISTER_CUMPROD(2, uint16_t,   0);
REGISTER_CUMPROD(2, int32_t,    0);
REGISTER_CUMPROD(2, uint32_t,   0);
REGISTER_CUMPROD(2, half,       0);
REGISTER_CUMPROD(2, float,      0);
REGISTER_CUMPROD(2, bfloat16_t, 0);

REGISTER_CUMPROD_WITH_TEMP(2, int8_t,     1);
REGISTER_CUMPROD_WITH_TEMP(2, uint8_t,    1);
REGISTER_CUMPROD_WITH_TEMP(2, int16_t,    1);
REGISTER_CUMPROD_WITH_TEMP(2, uint16_t,   1);
REGISTER_CUMPROD_WITH_TEMP(2, int32_t,    1);
REGISTER_CUMPROD_WITH_TEMP(2, uint32_t,   1);
REGISTER_CUMPROD_WITH_TEMP(2, half,       1);
REGISTER_CUMPROD_WITH_TEMP(2, float,      1);
REGISTER_CUMPROD_WITH_TEMP(2, bfloat16_t, 1);

REGISTER_CUMPROD(3, int8_t,     0);
REGISTER_CUMPROD(3, uint8_t,    0);
REGISTER_CUMPROD(3, int16_t,    0);
REGISTER_CUMPROD(3, uint16_t,   0);
REGISTER_CUMPROD(3, int32_t,    0);
REGISTER_CUMPROD(3, uint32_t,   0);
REGISTER_CUMPROD(3, half,       0);
REGISTER_CUMPROD(3, float,      0);
REGISTER_CUMPROD(3, bfloat16_t, 0);

REGISTER_CUMPROD_WITH_TEMP(3, int8_t,     1);
REGISTER_CUMPROD_WITH_TEMP(3, uint8_t,    1);
REGISTER_CUMPROD_WITH_TEMP(3, int16_t,    1);
REGISTER_CUMPROD_WITH_TEMP(3, uint16_t,   1);
REGISTER_CUMPROD_WITH_TEMP(3, int32_t,    1);
REGISTER_CUMPROD_WITH_TEMP(3, uint32_t,   1);
REGISTER_CUMPROD_WITH_TEMP(3, half,       1);
REGISTER_CUMPROD_WITH_TEMP(3, float,      1);
REGISTER_CUMPROD_WITH_TEMP(3, bfloat16_t, 1);
}

#endif  /* __DAV_C310__ */

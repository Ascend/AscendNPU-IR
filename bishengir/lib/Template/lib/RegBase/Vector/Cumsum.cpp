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
sklansky_calculation (memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst, sklansky_param_t param)
{
  int32_t groupTailCount = param.groupTailCount;
  uint16_t groupTailCountu16 = (uint16_t)groupTailCount;
  int32_t addTailCount = param.addTailCount;
  uint16_t addTailCountu16 = (uint16_t)addTailCount;
  uint16_t addCount = (uint16_t)param.addCount;
  int32_t startOffset = param.startOffset;
  int32_t group_offset = param.group_offset;
  uint16_t group_offsetu16 = (uint16_t)group_offset;
  int32_t groupMainCount = param.groupMainCount;
  uint16_t groupMainCountu16 = (uint16_t)groupMainCount;
  int32_t mFold = param.mFactor;
  int32_t nAddFactor = param.nAddFactor;
  int32_t realDupSize = param.realDupSize;
  int32_t flag = param.flag;
  constexpr uint16_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
  __ubuf__ T *src_ptr = dst->aligned + dst->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  uint16_t nLoop = CEIL_DIV(realDupSize, REG_REGISTER_SIZE);
  uint16_t src_stride0 = (uint16_t)src->strides[0];
  __VEC_SCOPE__ {
    VectorReg<T> x1RegTensor, x2RegTensor;
    vector_bool mask;
    uint32_t totalElements = (uint32_t)nAddFactor * (uint32_t)mFold;
    for (uint16_t j = 0; j < nLoop; j++) {
      CREATE_MASK_BY_SIZE(mask, T, totalElements);
      for (uint16_t m = 0; m < groupMainCountu16; m++) {
          int32_t src_offset = flag * m * group_offset + j * num_per_reg;
          vlds(x1RegTensor, src_ptr + startOffset, src_offset, NORM);
          for (uint16_t n = 1; n <= addCount; n++) {
            int32_t dst_offset = flag * (m * group_offset + n * src_stride0) + j * num_per_reg;
            vlds(x2RegTensor, src_ptr + startOffset, dst_offset, NORM);
            vadd(x2RegTensor, x1RegTensor, x2RegTensor, mask);
            vsts(x2RegTensor, dst_ptr + startOffset, dst_offset, NORM_B32, mask);
          }
      }

      for (uint16_t m = 0; m < groupTailCountu16; m++) {
        int32_t src_offset = flag * groupMainCount * group_offset + j * num_per_reg;
        vlds(x1RegTensor, src_ptr + startOffset, src_offset, NORM);
        for (uint16_t n = 1; n <= addTailCountu16; n++) {
          int32_t dst_offset = flag * (groupMainCount * group_offset + n * src_stride0) + j * num_per_reg;
          vlds(x2RegTensor, src_ptr + startOffset, dst_offset, NORM);
          vadd(x2RegTensor, x1RegTensor, x2RegTensor, mask, MODE_ZEROING);
          vsts(x2RegTensor, dst_ptr + startOffset, dst_offset, NORM_B32, mask);
        }
      }
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
sklansky_loops(memref_t<__ubuf__ T, 3> *src,
               memref_t<__ubuf__ T, 3> *dst,
               int32_t addLoop, int32_t rFactor, sklansky_param_t param) {
  int32_t addLoopPow     = pow(2, addLoop);      // 2^k：源距组起点的行距,
  int32_t nextAddLoopPow = pow(2, addLoop + 1);  // 2^(k+1)：组大小
  // 源行在组内的位置（正向：第 2^k-1 行，反向：倒数第 2^k 行）
  param.startOffset = (param.flag == -1)
        ? (rFactor - addLoopPow) * src->strides[0]
        : (addLoopPow - 1) * src->strides[0];
  param.addCount = addLoopPow;                          // 每源加 2^k 个目标, 每个源行需要加的目标行数 = 2^k
  param.group_offset = nextAddLoopPow * src->strides[0]; // 组间距（字节）
  param.groupMainCount = rFactor / nextAddLoopPow;  // 完整组数
  param.groupTailCount = (rFactor % nextAddLoopPow > addLoopPow) ? 1 : 0;  // 是否有尾组（0 或 1）
  param.addTailCount = rFactor - param.groupMainCount * nextAddLoopPow - addLoopPow;  // 尾组内目标行数
  sklansky_calculation(src, dst, param);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
oneway_sklansky_cumsum(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst, bool reverse) {
  int32_t rFactor = src->sizes[0];
  int32_t mFactor = src->sizes[1];
  int32_t nFactor = src->sizes[2];
  int32_t dupSize_ = CEIL_FACTOR(nFactor * sizeof(T), 32);
  int32_t nAddFactor = dupSize_ / sizeof(T);
  int32_t realDupSize = nAddFactor * sizeof(T) * mFactor;
  int32_t rLoop = CeilLog2(rFactor);  // 迭代轮数 = log2(rFactor_)
  sklansky_param_t param;
  param.flag = reverse ? -1 : 1;
  param.mFactor = mFactor;
  param.realDupSize = realDupSize;
  param.nAddFactor = nAddFactor;
  for (int32_t k = 0; k < rLoop; k++) {
    sklansky_loops(src, dst, k, rFactor, param);
  }
}

template <typename T, int dim>
__aiv__ __attribute__((always_inline)) void
compute_cumsum_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                  bool reverse, memref_t<__ubuf__ T, 3> *temp = nullptr) {
  if constexpr (dim == 0) {
    copy_for_cum_op(src, dst);
    oneway_sklansky_cumsum<T> (dst, dst, reverse);
  } else if constexpr (dim == 1) {
    int64_t n_aligned = CEIL_FACTOR(temp->sizes[2] * sizeof(T), 32) / sizeof(T);
    memref_t<__ubuf__ T, 3> temp_3d{
      temp->allocated, temp->aligned, temp->offset,
      {temp->sizes[1], temp->sizes[0], temp->sizes[2]},
      {temp->sizes[0] * n_aligned, n_aligned, 1}};
    transpose_dim_01<T>(src, &temp_3d);
    oneway_sklansky_cumsum<T> (&temp_3d, &temp_3d, reverse);
    transpose_dim_01<T>(&temp_3d, dst);
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
compute_cumsum_2d_along_dim1(memref_t<__ubuf__ T, 2> *src,
                             memref_t<__ubuf__ T, 2> *dst, bool reverse) {
  int64_t M = src->sizes[0];
  uint16_t N = src->sizes[1];
  int32_t stride0 = src->strides[0];
  uint32_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
  using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int16_t>;
  using IdxUT = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  uint16_t mLoop = CEIL_DIV((uint16_t)M, (uint16_t)num_per_reg);
  uint32_t size0 = M;
  int32_t start_col = reverse ? (N - 1) : 0;
  __VEC_SCOPE__ {
    VectorReg<T> x1RegTensor, x2RegTensor;
    VectorReg<IdxT> idx_reg;
    vector_bool mask;
    uint32_t full_mask_size = num_per_reg;
    CREATE_MASK_BY_SIZE(mask, T, full_mask_size);
    vci(idx_reg, 0);
    vmuls(idx_reg, idx_reg, stride0, mask);
    for (uint16_t m = 0; m < mLoop; m++) {
      __ubuf__ T *src_block = src_ptr + (int32_t)m * num_per_reg * stride0;
      __ubuf__ T *dst_block = dst_ptr + (int32_t)m * num_per_reg * stride0;
      CREATE_MASK_BY_SIZE(mask, T, size0);
      vgather2(x1RegTensor, src_block + start_col, (VectorReg<IdxUT> &)idx_reg,
               mask);
      vscatter(x1RegTensor, dst_block + start_col, (VectorReg<IdxUT> &)idx_reg,
               mask);
      for (uint16_t n = 1; n < N; n++) {
        int32_t nOffset = reverse ? (N - n - 1) : n;
        vgather2(x2RegTensor, src_block + nOffset, (VectorReg<IdxUT> &)idx_reg,
                 mask);
        vadd(x1RegTensor, x1RegTensor, x2RegTensor, mask);
        vscatter(x1RegTensor, dst_block + nOffset, (VectorReg<IdxUT> &)idx_reg,
                 mask);
      }
    }
  }
}

template <typename T, int cum_dim>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                 memref_t<__ubuf__ T, 3> *temp, bool reverse) {
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
          std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
          std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
          std::is_same<T, float>::value || std::is_same<T, half>::value ||
          std::is_same<T, bfloat16_t>::value,
      "cumsum op only uint8/16/32_t, int8/16/32_t, float16/32 and  bfloat16 "
      "type operands in template!");
  compute_cumsum_3d<T, cum_dim>(src, dst, reverse, temp);
}

template <typename T, int cum_dim>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_2d(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
                 memref_t<__ubuf__ T, 2> *temp, bool reverse) {
  static_assert(
    std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
    std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
    std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
    std::is_same<T, float>::value || std::is_same<T, half>::value ||
    std::is_same<T, bfloat16_t>::value,
    "cumsum ra op only uint8/16/32/64_t, int8/16/32/64_t, float16/32 and  bfloat16 type operands in template!");

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
    compute_cumsum_3d<T, 0>(&src_3d, &dst_3d, reverse);
  } else {
    if constexpr (sizeof(T) == 1) {
      memref_t<__ubuf__ T, 2> temp_trans{temp->allocated,
                                         temp->aligned,
                                         temp->offset,
                                         {src->sizes[1], src->sizes[0]},
                                         {temp->strides[0], temp->strides[1]}};
      transpose_ar2ra_i8<T>(src, &temp_trans);
      vector_cumsum_2d<T, 0>(&temp_trans, &temp_trans, nullptr, reverse);
      transpose_ar2ra_i8<T>(&temp_trans, dst);
    } else {
      memref_t<__ubuf__ T, 2> temp_trans{temp->allocated,
                                        temp->aligned,
                                        temp->offset,
                                        {src->sizes[1], src->sizes[0]},
                                        {temp->strides[0], temp->strides[1]}};
      transpose_ar2ra<T>(src, &temp_trans);
      vector_cumsum_2d<T, 0>(&temp_trans, &temp_trans, nullptr, reverse);
      transpose_ar2ra<T>(&temp_trans, dst);
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void _scan8_sklansky_reg(__ubuf__ T *p,
                                                                T &carry) {
  T x0 = p[0], x1 = p[1], x2 = p[2], x3 = p[3], x4 = p[4], x5 = p[5], x6 = p[6],
    x7 = p[7];

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

  p[0] = x0;
  p[1] = x1;
  p[2] = x2;
  p[3] = x3;
  p[4] = x4;
  p[5] = x5;
  p[6] = x6;
  p[7] = x7;

  carry = x7;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void _scan16_sklansky_reg(__ubuf__ T *p,
                                                                 T &carry) {
  T x0 = p[0], x1 = p[1], x2 = p[2], x3 = p[3];
  T x4 = p[4], x5 = p[5], x6 = p[6], x7 = p[7];
  T x8 = p[8], x9 = p[9], x10 = p[10], x11 = p[11];
  T x12 = p[12], x13 = p[13], x14 = p[14], x15 = p[15];

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

  p[0] = x0;
  p[1] = x1;
  p[2] = x2;
  p[3] = x3;
  p[4] = x4;
  p[5] = x5;
  p[6] = x6;
  p[7] = x7;
  p[8] = x8;
  p[9] = x9;
  p[10] = x10;
  p[11] = x11;
  p[12] = x12;
  p[13] = x13;
  p[14] = x14;
  p[15] = x15;

  carry = x15;
}

template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
phaseA_sklansky_regbuf_16(__ubuf__ PromoteDataType *p, int sz) {
  using T = PromoteDataType;
  if (sz < 1)
    return;
  T carry = 0;
  int n = sz;
  while (n >= 16) {
    _scan16_sklansky_reg<T>(p, carry);
    p += 16;
    n -= 16;
  }
  while (n >= 8) {
    _scan8_sklansky_reg<T>(p, carry);
    p += 8;
    n -= 8;
  }
  while (n-- > 0) {
    carry = carry + *p;
    *p = carry;
    p++;
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
_scan8_sklansky_reg_reverse(__ubuf__ T *p, T &carry) {
  T x0 = p[0], x1 = p[1], x2 = p[2], x3 = p[3];
  T x4 = p[4], x5 = p[5], x6 = p[6], x7 = p[7];

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

  p[0] = x0;
  p[1] = x1;
  p[2] = x2;
  p[3] = x3;
  p[4] = x4;
  p[5] = x5;
  p[6] = x6;
  p[7] = x7;

  carry = x0;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
_scan16_sklansky_reg_reverse(__ubuf__ T *p, T &carry) {
  T x0 = p[0], x1 = p[1], x2 = p[2], x3 = p[3];
  T x4 = p[4], x5 = p[5], x6 = p[6], x7 = p[7];
  T x8 = p[8], x9 = p[9], x10 = p[10], x11 = p[11];
  T x12 = p[12], x13 = p[13], x14 = p[14], x15 = p[15];

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

  p[0] = x0;
  p[1] = x1;
  p[2] = x2;
  p[3] = x3;
  p[4] = x4;
  p[5] = x5;
  p[6] = x6;
  p[7] = x7;
  p[8] = x8;
  p[9] = x9;
  p[10] = x10;
  p[11] = x11;
  p[12] = x12;
  p[13] = x13;
  p[14] = x14;
  p[15] = x15;

  carry = x0;
}

template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
phaseA_sklansky_regbuf_16_reverse(__ubuf__ PromoteDataType *p, int sz) {
  using T = PromoteDataType;
  if (sz < 1)
    return;
  T carry = 0;
  int n = sz;
  __ubuf__ T *tail = p + sz;
  while (n >= 16) {
    tail -= 16;
    n -= 16;
    _scan16_sklansky_reg_reverse<T>(tail, carry);
  }
  while (n >= 8) {
    tail -= 8;
    n -= 8;
    _scan8_sklansky_reg_reverse<T>(tail, carry);
  }
  while (n-- > 0) {
    --tail;
    carry = *tail + carry;
    *tail = carry;
  }
}

template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
phaseA_scalar_sklansky_regbuf_16(__ubuf__ PromoteDataType *dpBase,
                                 int numBlocks, int tailSize, int BpE) {
  for (uint16_t b = 0; b < static_cast<uint16_t>(numBlocks); b++) {
    int sz = (b < static_cast<uint16_t>(numBlocks - 1))
                 ? BpE
                 : (tailSize > 0 ? tailSize : BpE);
    phaseA_sklansky_regbuf_16<PromoteDataType>(dpBase + b * BpE, sz);
  }
}

template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
phaseA_scalar_sklansky_regbuf_16_reverse(__ubuf__ PromoteDataType *dpBase,
                                         int numBlocks, int tailSize, int BpE) {
  uint16_t nblk = static_cast<uint16_t>(numBlocks);
  for (uint16_t b = nblk; b > 0; b--) {
    int sz = ((b - 1) < nblk - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);
    phaseA_sklansky_regbuf_16_reverse<PromoteDataType>(dpBase + (b - 1) * BpE,
                                                       sz);
  }
}

template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
phaseA_emulate(__ubuf__ PromoteDataType *dpBase, int numBlocks, int tailSize,
               int BpE, int headScalarBlocks = 1) {
  uint16_t nBlocks = static_cast<uint16_t>(numBlocks);
  uint16_t first = static_cast<uint16_t>(headScalarBlocks);
  if (first < nBlocks) {
    uint16_t mainEnd = first;
    // Byte vectors use 2-lane loop (lower register pressure);
    // other types use 4-lane for throughput.
    if constexpr (sizeof(PromoteDataType) == 1) {
      uint16_t rem = static_cast<uint16_t>(nBlocks - first);
      uint16_t mainBlocks = static_cast<uint16_t>((rem / 2) * 2);
      mainEnd = static_cast<uint16_t>(first + mainBlocks);
      __VEC_SCOPE__ {
        VectorReg<PromoteDataType> v_data0, v_data1, v_tmp0, v_tmp1, v_sel0,
            v_sel1, v_zero;
        vector_align v_align0, v_align1;
        vector_bool mask0, mask1, stepMask;
        vbr(v_zero, 0);
#define EMULATE_STEP(S)                                                        \
  do {                                                                         \
    uint32_t _stepSz = static_cast<uint32_t>(S);                               \
    CREATE_MASK_BY_SIZE(stepMask, PromoteDataType, _stepSz);                   \
    int _bs;                                                                   \
    uint32_t _ms0, _ms1;                                                       \
    __ubuf__ PromoteDataType *_b0, *_b1;                                       \
    __ubuf__ PromoteDataType *_s0, *_s1;                                       \
    for (uint16_t b = first; b < mainEnd; b += 2) {                            \
      _bs = (b < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);         \
      _ms0 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask0, PromoteDataType, _ms0);                       \
      _b0 = dpBase + b * BpE;                                                  \
      _s0 = _b0 - (S);                                                         \
      _bs = (b + 1 < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);     \
      _ms1 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask1, PromoteDataType, _ms1);                       \
      _b1 = dpBase + (b + 1) * BpE;                                            \
      _s1 = _b1 - (S);                                                         \
      vlds(v_data0, _b0, 0, NORM);                                             \
      vldas(v_align0, _s0);                                                    \
      vlds(v_data1, _b1, 0, NORM);                                             \
      vldas(v_align1, _s1);                                                    \
      vldus(v_tmp0, v_align0, _s0);                                            \
      vsel(v_sel0, v_zero, v_tmp0, stepMask);                                  \
      vldus(v_tmp1, v_align1, _s1);                                            \
      vsel(v_sel1, v_zero, v_tmp1, stepMask);                                  \
      vadd(v_data0, v_sel0, v_data0, mask0);                                   \
      vadd(v_data1, v_sel1, v_data1, mask1);                                   \
      vsts(v_data0, _b0, 0, NORM_B32, mask0);                                  \
      vsts(v_data1, _b1, 0, NORM_B32, mask1);                                  \
    }                                                                          \
  } while (0)
        EMULATE_STEP(1);
        EMULATE_STEP(2);
        EMULATE_STEP(4);
        EMULATE_STEP(8);
        EMULATE_STEP(16);
        EMULATE_STEP(32);
        EMULATE_STEP(64);
        EMULATE_STEP(128);
#undef EMULATE_STEP
      }
    } else {
      uint16_t rem = static_cast<uint16_t>(nBlocks - first);
      uint16_t vecBlocks = static_cast<uint16_t>(rem);
      uint16_t mainBlocks = static_cast<uint16_t>((vecBlocks / 4) * 4);
      mainEnd = static_cast<uint16_t>(first + mainBlocks);
      __VEC_SCOPE__ {
        VectorReg<PromoteDataType> v_data0, v_data1, v_data2, v_data3, v_tmp0,
            v_tmp1, v_tmp2, v_tmp3, v_sel0, v_sel1, v_sel2, v_sel3, v_zero;
        vector_align v_align0, v_align1, v_align2, v_align3;
        vector_bool mask0, mask1, mask2, mask3, stepMask;
        vbr(v_zero, 0);
#define EMULATE_STEP(S)                                                        \
  do {                                                                         \
    uint32_t _stepSz = static_cast<uint32_t>(S);                               \
    CREATE_MASK_BY_SIZE(stepMask, PromoteDataType, _stepSz);                   \
    int _bs;                                                                   \
    uint32_t _ms0, _ms1, _ms2, _ms3;                                           \
    __ubuf__ PromoteDataType *_b0, *_b1, *_b2, *_b3;                           \
    __ubuf__ PromoteDataType *_s0, *_s1, *_s2, *_s3;                           \
    for (uint16_t b = first; b < mainEnd; b += 4) {                            \
      _bs = (b < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);         \
      _ms0 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask0, PromoteDataType, _ms0);                       \
      _b0 = dpBase + b * BpE;                                                  \
      _s0 = _b0 - (S);                                                         \
      _bs = (b + 1 < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);     \
      _ms1 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask1, PromoteDataType, _ms1);                       \
      _b1 = dpBase + (b + 1) * BpE;                                            \
      _s1 = _b1 - (S);                                                         \
      _bs = (b + 2 < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);     \
      _ms2 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask2, PromoteDataType, _ms2);                       \
      _b2 = dpBase + (b + 2) * BpE;                                            \
      _s2 = _b2 - (S);                                                         \
      _bs = (b + 3 < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);     \
      _ms3 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask3, PromoteDataType, _ms3);                       \
      _b3 = dpBase + (b + 3) * BpE;                                            \
      _s3 = _b3 - (S);                                                         \
      /* load */                                                               \
      vlds(v_data0, _b0, 0, NORM);                                             \
      vldas(v_align0, _s0);                                                    \
      vlds(v_data1, _b1, 0, NORM);                                             \
      vldas(v_align1, _s1);                                                    \
      vlds(v_data2, _b2, 0, NORM);                                             \
      vldas(v_align2, _s2);                                                    \
      vlds(v_data3, _b3, 0, NORM);                                             \
      vldas(v_align3, _s3);                                                    \
      /* compute */                                                            \
      vldus(v_tmp0, v_align0, _s0);                                            \
      vsel(v_sel0, v_zero, v_tmp0, stepMask);                                  \
      vldus(v_tmp1, v_align1, _s1);                                            \
      vsel(v_sel1, v_zero, v_tmp1, stepMask);                                  \
      vldus(v_tmp2, v_align2, _s2);                                            \
      vsel(v_sel2, v_zero, v_tmp2, stepMask);                                  \
      vldus(v_tmp3, v_align3, _s3);                                            \
      vsel(v_sel3, v_zero, v_tmp3, stepMask);                                  \
      vadd(v_data0, v_sel0, v_data0, mask0);                                   \
      vadd(v_data1, v_sel1, v_data1, mask1);                                   \
      vadd(v_data2, v_sel2, v_data2, mask2);                                   \
      vadd(v_data3, v_sel3, v_data3, mask3);                                   \
      /* store */                                                              \
      vsts(v_data0, _b0, 0, NORM_B32, mask0);                                  \
      vsts(v_data1, _b1, 0, NORM_B32, mask1);                                  \
      vsts(v_data2, _b2, 0, NORM_B32, mask2);                                  \
      vsts(v_data3, _b3, 0, NORM_B32, mask3);                                  \
    }                                                                          \
  } while (0)
        EMULATE_STEP(1);
        EMULATE_STEP(2);
        EMULATE_STEP(4);
        EMULATE_STEP(8);
        EMULATE_STEP(16);
        EMULATE_STEP(32);
        if (BpE > 64) {
          EMULATE_STEP(64);
        }
        if (BpE > 128) {
          EMULATE_STEP(128);
        }
#undef EMULATE_STEP
      }
    }
    /* tail */
    for (uint16_t b = mainEnd; b < nBlocks; b++) {
      int bSize = (b < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);
      uint32_t maskSz = static_cast<uint32_t>(bSize);
      __VEC_SCOPE__ {
        VectorReg<PromoteDataType> v_data, v_tmp, v_sel, v_zero;
        vector_align v_align;
        vector_bool mask, stepMask;
        vbr(v_zero, 0);
        CREATE_MASK_BY_SIZE(mask, PromoteDataType, maskSz);
        __ubuf__ PromoteDataType *bBase = dpBase + b * BpE;
#define TAIL_STEP(S)                                                           \
  do {                                                                         \
    uint32_t _sstepSz = static_cast<uint32_t>(S);                              \
    CREATE_MASK_BY_SIZE(stepMask, PromoteDataType, _sstepSz);                  \
    __ubuf__ PromoteDataType *_src = bBase - (S);                              \
    vlds(v_data, bBase, 0, NORM);                                              \
    vldas(v_align, _src);                                                      \
    vldus(v_tmp, v_align, _src);                                               \
    vsel(v_sel, v_zero, v_tmp, stepMask);                                      \
    vadd(v_data, v_sel, v_data, mask);                                         \
    vsts(v_data, bBase, 0, NORM_B32, mask);                                    \
  } while (0)
        TAIL_STEP(1);
        TAIL_STEP(2);
        TAIL_STEP(4);
        TAIL_STEP(8);
        TAIL_STEP(16);
        TAIL_STEP(32);
        if (BpE > 64) {
          TAIL_STEP(64);
        }
        if (BpE > 128) {
          TAIL_STEP(128);
        }
#undef TAIL_STEP
      }
    }
  }
  if (headScalarBlocks > 0)
    phaseA_scalar_sklansky_regbuf_16<PromoteDataType>(dpBase, headScalarBlocks,
                                                      0, BpE);
  // V/S both done, drain V queue before Phase B+C reads
  INTRINSIC(set_flag, PIPE_S, PIPE_V, EVENT_ID6);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, EVENT_ID6);
}

template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
phaseA_emulate_reverse(__ubuf__ PromoteDataType *dpBase, int numBlocks,
                       int tailSize, int BpE, int tailScalarBlocks = 0) {
  if (numBlocks <= 1)
    return;
  uint16_t nBlocks = static_cast<uint16_t>(numBlocks);
  uint16_t vecEnd = static_cast<uint16_t>(nBlocks - tailScalarBlocks);
  uint16_t lastVec = static_cast<uint16_t>(vecEnd - 1);
  if (vecEnd >= 1) {
    uint16_t vecLeft = static_cast<uint16_t>(lastVec + 1);
    uint16_t mainBlocks = static_cast<uint16_t>((vecLeft / 4) * 4);
    uint16_t mainStart = static_cast<uint16_t>(lastVec + 1 - mainBlocks);
    __VEC_SCOPE__ {
      VectorReg<PromoteDataType> v_data0, v_data1, v_data2, v_data3, v_tmp0,
          v_tmp1, v_tmp2, v_tmp3, v_sel0, v_sel1, v_sel2, v_sel3, v_zero;
      vector_align v_align0, v_align1, v_align2, v_align3;
      vector_bool mask0, mask1, mask2, mask3, keepMask0;
      vbr(v_zero, 0);
#define REV_EMULATE_STEP(S)                                                    \
  do {                                                                         \
    uint32_t _stepSz = static_cast<uint32_t>(S);                               \
    uint32_t _ks = static_cast<uint32_t>(BpE - (S));                           \
    CREATE_MASK_BY_SIZE(keepMask0, PromoteDataType, _ks);                      \
    int _bs;                                                                   \
    uint32_t _ms0, _ms1, _ms2, _ms3;                                           \
    __ubuf__ PromoteDataType *_b0, *_b1, *_b2, *_b3;                           \
    __ubuf__ PromoteDataType *_s0, *_s1, *_s2, *_s3;                           \
    for (uint16_t b = lastVec; b >= mainStart && b <= lastVec; b -= 4) {       \
      _bs = (b < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);         \
      _ms0 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask0, PromoteDataType, _ms0);                       \
      _b0 = dpBase + b * BpE;                                                  \
      _s0 = _b0 + (S);                                                         \
      _bs = (b - 1 < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);     \
      _ms1 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask1, PromoteDataType, _ms1);                       \
      _b1 = dpBase + (b - 1) * BpE;                                            \
      _s1 = _b1 + (S);                                                         \
      _bs = (b - 2 < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);     \
      _ms2 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask2, PromoteDataType, _ms2);                       \
      _b2 = dpBase + (b - 2) * BpE;                                            \
      _s2 = _b2 + (S);                                                         \
      _bs = (b - 3 < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);     \
      _ms3 = static_cast<uint32_t>(_bs);                                       \
      CREATE_MASK_BY_SIZE(mask3, PromoteDataType, _ms3);                       \
      _b3 = dpBase + (b - 3) * BpE;                                            \
      _s3 = _b3 + (S);                                                         \
      /* load */                                                               \
      vlds(v_data0, _b0, 0, NORM);                                             \
      vldas(v_align0, _s0);                                                    \
      vlds(v_data1, _b1, 0, NORM);                                             \
      vldas(v_align1, _s1);                                                    \
      vlds(v_data2, _b2, 0, NORM);                                             \
      vldas(v_align2, _s2);                                                    \
      vlds(v_data3, _b3, 0, NORM);                                             \
      vldas(v_align3, _s3);                                                    \
      /* compute: keep first bSize-S, zero last S */                           \
      vldus(v_tmp0, v_align0, _s0);                                            \
      vsel(v_sel0, v_tmp0, v_zero, keepMask0);                                 \
      vldus(v_tmp1, v_align1, _s1);                                            \
      vsel(v_sel1, v_tmp1, v_zero, keepMask0);                                 \
      vldus(v_tmp2, v_align2, _s2);                                            \
      vsel(v_sel2, v_tmp2, v_zero, keepMask0);                                 \
      vldus(v_tmp3, v_align3, _s3);                                            \
      vsel(v_sel3, v_tmp3, v_zero, keepMask0);                                 \
      vadd(v_data0, v_sel0, v_data0, mask0);                                   \
      vadd(v_data1, v_sel1, v_data1, mask1);                                   \
      vadd(v_data2, v_sel2, v_data2, mask2);                                   \
      vadd(v_data3, v_sel3, v_data3, mask3);                                   \
      /* store */                                                              \
      vsts(v_data0, _b0, 0, NORM_B32, mask0);                                  \
      vsts(v_data1, _b1, 0, NORM_B32, mask1);                                  \
      vsts(v_data2, _b2, 0, NORM_B32, mask2);                                  \
      vsts(v_data3, _b3, 0, NORM_B32, mask3);                                  \
    }                                                                          \
  } while (0)
      REV_EMULATE_STEP(1);
      REV_EMULATE_STEP(2);
      REV_EMULATE_STEP(4);
      REV_EMULATE_STEP(8);
      REV_EMULATE_STEP(16);
      REV_EMULATE_STEP(32);
      if (BpE > 64) {
        REV_EMULATE_STEP(64);
      }
      if (BpE > 128) {
        REV_EMULATE_STEP(128);
      }
#undef REV_EMULATE_STEP
    }
    /* tail — remaining blocks < mainStart, down to vecStart, processed
     * right-to-left */
    for (uint16_t b = mainStart - 1; b < vecEnd; b--) {
      int bSize = (b < nBlocks - 1) ? BpE : (tailSize > 0 ? tailSize : BpE);
      uint32_t maskSz = static_cast<uint32_t>(bSize);
      __VEC_SCOPE__ {
        VectorReg<PromoteDataType> v_data, v_tmp, v_sel, v_zero;
        vector_align v_align;
        vector_bool mask, stepMask;
        vbr(v_zero, 0);
        CREATE_MASK_BY_SIZE(mask, PromoteDataType, maskSz);
        __ubuf__ PromoteDataType *bBase = dpBase + b * BpE;
#define TAIL_REV_STEP(S)                                                       \
  do {                                                                         \
    uint32_t _sstepSz = static_cast<uint32_t>(S);                              \
    uint32_t _keepSz = static_cast<uint32_t>(bSize - (S));                     \
    CREATE_MASK_BY_SIZE(stepMask, PromoteDataType, _keepSz);                   \
    __ubuf__ PromoteDataType *_src = bBase + (S);                              \
    vlds(v_data, bBase, 0, NORM);                                              \
    vldas(v_align, _src);                                                      \
    vldus(v_tmp, v_align, _src);                                               \
    vsel(v_sel, v_tmp, v_zero, stepMask);                                      \
    vadd(v_data, v_sel, v_data, mask);                                         \
    vsts(v_data, bBase, 0, NORM_B32, mask);                                    \
  } while (0)
        TAIL_REV_STEP(1);
        TAIL_REV_STEP(2);
        TAIL_REV_STEP(4);
        TAIL_REV_STEP(8);
        TAIL_REV_STEP(16);
        TAIL_REV_STEP(32);
        if (BpE > 64) {
          TAIL_REV_STEP(64);
        }
        if (BpE > 128) {
          TAIL_REV_STEP(128);
        }
#undef TAIL_REV_STEP
      }
    }
  }
  if (tailScalarBlocks > 0)
    phaseA_scalar_sklansky_regbuf_16_reverse<PromoteDataType>(
        dpBase + vecEnd * BpE, static_cast<uint16_t>(tailScalarBlocks),
        tailSize, BpE);
  INTRINSIC(set_flag, PIPE_S, PIPE_V, EVENT_ID6);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, EVENT_ID6);
}

template <typename PromoteDataType>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_1d_twoway(memref_t<__ubuf__ PromoteDataType, 1> *dst,
                        bool reverse) {

  constexpr int BpE = REG_REGISTER_SIZE / sizeof(PromoteDataType);
  int64_t N = dst->sizes[0];
  int numBlocks = static_cast<int>(CEIL_DIV(N, static_cast<int64_t>(BpE)));
  __ubuf__ PromoteDataType *dpBase = dst->aligned + dst->offset;
  pipe_barrier(PIPE_ALL);
  if (N <= 32) {
    int n = static_cast<int>(N);
    PromoteDataType tmp = 0;
    if (!reverse) {
      for (int i = 0; i < n; i++) {
        tmp += dpBase[i];
        dpBase[i] = tmp;
      }
    } else {
      for (int i = n - 1; i >= 0; i--) {
        tmp += dpBase[i];
        dpBase[i] = tmp;
      }
    }
    pipe_barrier(PIPE_ALL);
    return;
  }
  if (N <= 2 * BpE) {
    if (reverse)
      phaseA_sklansky_regbuf_16_reverse<PromoteDataType>(dpBase,
                                                         static_cast<int>(N));
    else
      phaseA_sklansky_regbuf_16<PromoteDataType>(dpBase, static_cast<int>(N));
    pipe_barrier(PIPE_ALL);
    return;
  }

  int tailSize = static_cast<int>(N % BpE);
  if (tailSize == 0 && N > 0)
    tailSize = BpE;
  INTRINSIC(set_flag, PIPE_S, PIPE_V, EVENT_ID6);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, EVENT_ID6);
  int min_block = 1;
  if (reverse)
    phaseA_emulate_reverse<PromoteDataType>(dpBase, numBlocks, tailSize, BpE,
                                            min_block);
  else
    phaseA_emulate<PromoteDataType>(dpBase, numBlocks, tailSize, BpE,
                                    min_block);
  if (numBlocks <= 1)
    return;

  // Streaming Phase B+C: for each block [1..nBlocks), vadds carry, update carry
  // from block's last element
  {
    uint16_t nblk = static_cast<uint16_t>(numBlocks);
    int lastFull = (tailSize > 0) ? (numBlocks - 2) : (numBlocks - 1);
    __VEC_SCOPE__ {
      VectorReg<PromoteDataType> reg;
      vector_bool mask;
      if (!reverse) {
        // carry = last element of block 0
        int idx0 = (numBlocks == 1) ? (tailSize - 1) : (BpE - 1);
        PromoteDataType carry = dpBase[idx0];
        for (uint16_t b = 1; b < nblk; ++b) {
          int bSize = static_cast<int>(b) <= lastFull ? BpE : tailSize;
          uint32_t ms = static_cast<uint32_t>(bSize);
          CREATE_MASK_BY_SIZE(mask, PromoteDataType, ms);
          __ubuf__ PromoteDataType *p = dpBase + b * BpE;
          vlds(reg, p, 0, NORM);
          vadds(reg, reg, carry, mask, MODE_ZEROING);
          vsts(reg, p, 0, NORM_B32, mask);
          carry = p[bSize - 1];
        }
      } else {
        int lastBlockStart = (numBlocks - 1) * BpE;
        PromoteDataType carry = dpBase[lastBlockStart];
        for (int16_t b = static_cast<int16_t>(numBlocks - 2); b >= 0; b--) {
          int bSize = (b <= lastFull) ? BpE : tailSize;
          uint32_t ms = static_cast<uint32_t>(bSize);
          CREATE_MASK_BY_SIZE(mask, PromoteDataType, ms);
          __ubuf__ PromoteDataType *p = dpBase + b * BpE;
          vlds(reg, p, 0, NORM);
          vadds(reg, reg, carry, mask, MODE_ZEROING);
          vsts(reg, p, 0, NORM_B32, mask);
          carry = p[0];
        }
      }
    }
  }

  pipe_barrier(PIPE_ALL);
  return;
}
template <typename T, int cum_dim>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                 memref_t<__ubuf__ T, 1> *temp, bool reverse) {
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
          std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
          std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
          std::is_same<T, float>::value || std::is_same<T, half>::value ||
          std::is_same<T, bfloat16_t>::value,
      "cumsum op only uint8/16/32/64_t, int8/16/32/64_t, float16/32 and "
      "bfloat16 type operands in template!");
  copy_ubuf_to_ubuf_1d_core_with_contiguous_last_dim(src, dst);
  vector_cumsum_1d_twoway<T>(dst, reverse);
}
extern "C" {
REGISTER_CUMSUM(1, int8_t, 0);
REGISTER_CUMSUM(1, uint8_t, 0);
REGISTER_CUMSUM(1, int16_t, 0);
REGISTER_CUMSUM(1, uint16_t, 0);
REGISTER_CUMSUM(1, int32_t, 0);
REGISTER_CUMSUM(1, uint32_t, 0);
REGISTER_CUMSUM(1, half, 0);
REGISTER_CUMSUM(1, float, 0);
REGISTER_CUMSUM(1, bfloat16_t, 0);

REGISTER_CUMSUM(2, int8_t, 0);
REGISTER_CUMSUM(2, uint8_t, 0);
REGISTER_CUMSUM(2, int16_t, 0);
REGISTER_CUMSUM(2, uint16_t, 0);
REGISTER_CUMSUM(2, int32_t, 0);
REGISTER_CUMSUM(2, uint32_t, 0);
REGISTER_CUMSUM(2, half, 0);
REGISTER_CUMSUM(2, float, 0);
REGISTER_CUMSUM(2, bfloat16_t, 0);

REGISTER_CUMSUM_WITH_TEMP(2, int8_t, 1);
REGISTER_CUMSUM_WITH_TEMP(2, uint8_t, 1);
REGISTER_CUMSUM_WITH_TEMP(2, int16_t, 1);
REGISTER_CUMSUM_WITH_TEMP(2, uint16_t, 1);
REGISTER_CUMSUM_WITH_TEMP(2, int32_t, 1);
REGISTER_CUMSUM_WITH_TEMP(2, uint32_t, 1);
REGISTER_CUMSUM_WITH_TEMP(2, half, 1);
REGISTER_CUMSUM_WITH_TEMP(2, float, 1);
REGISTER_CUMSUM_WITH_TEMP(2, bfloat16_t, 1);

REGISTER_CUMSUM(3, int8_t, 0);
REGISTER_CUMSUM(3, uint8_t, 0);
REGISTER_CUMSUM(3, int16_t, 0);
REGISTER_CUMSUM(3, uint16_t, 0);
REGISTER_CUMSUM(3, int32_t, 0);
REGISTER_CUMSUM(3, uint32_t, 0);
REGISTER_CUMSUM(3, half, 0);
REGISTER_CUMSUM(3, float, 0);
REGISTER_CUMSUM(3, bfloat16_t, 0);

REGISTER_CUMSUM_WITH_TEMP(3, int8_t, 1);
REGISTER_CUMSUM_WITH_TEMP(3, uint8_t, 1);
REGISTER_CUMSUM_WITH_TEMP(3, int16_t, 1);
REGISTER_CUMSUM_WITH_TEMP(3, uint16_t, 1);
REGISTER_CUMSUM_WITH_TEMP(3, int32_t, 1);
REGISTER_CUMSUM_WITH_TEMP(3, uint32_t, 1);
REGISTER_CUMSUM_WITH_TEMP(3, half, 1);
REGISTER_CUMSUM_WITH_TEMP(3, float, 1);
REGISTER_CUMSUM_WITH_TEMP(3, bfloat16_t, 1);
}

#endif

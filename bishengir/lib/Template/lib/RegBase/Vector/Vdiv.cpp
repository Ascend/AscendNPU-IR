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

#include "RegBase/ReductionUtils.h"
#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"
#if defined(__DAV_C310__)

// The software implementation interface vdiv of CCEC supports division
// operations on uint16, uint32, int16, and int32 types with no precision error.

template <typename T, size_t dim>
__aiv__ __attribute__((always_inline)) void
vdiv_int(memref_t<__ubuf__ T, dim> *src0, memref_t<__ubuf__ T, dim> *src1,
         memref_t<__ubuf__ T, dim> *dst) {
  uint16_t inner_size;
  uint16_t outer_size;
  int64_t s0_outer_stride = 0;
  int64_t s1_outer_stride = 0;
  int64_t d_outer_stride = 0;

  if (dim == 1) {
    inner_size = src0->sizes[0];
    outer_size = 1;
  } else {
    inner_size = src0->sizes[dim - 1];
    outer_size = 1;
    for (int i = 0; i < dim - 1; ++i)
      outer_size *= src0->sizes[i];

    s0_outer_stride = src0->strides[dim - 2];
    s1_outer_stride = src1->strides[dim - 2];
    d_outer_stride = dst->strides[dim - 2];
  }

  constexpr int dsize = sizeof(T);
  uint32_t ele_per_VL = VL_IN_BYTE / dsize;

  __ubuf__ T *src0_base = src0->aligned + src0->offset;
  __ubuf__ T *src1_base = src1->aligned + src1->offset;
  __ubuf__ T *dst_base = dst->aligned + dst->offset;

  uint16_t loop_times = CEIL_DIV(inner_size, ele_per_VL);

  for (uint16_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    __ubuf__ T *src0_ptr = src0_base + outer_idx * s0_outer_stride;
    __ubuf__ T *src1_ptr = src1_base + outer_idx * s1_outer_stride;
    __ubuf__ T *dst_ptr = dst_base + outer_idx * d_outer_stride;

    __VEC_SCOPE__ {
      VectorReg<T> src0_val, src1_val, dst_val;

      for (uint16_t loop_idx = 0; loop_idx < loop_times; ++loop_idx) {
        uint32_t remain = inner_size - loop_idx * ele_per_VL;
        uint32_t cur = remain > ele_per_VL ? ele_per_VL : remain;

        vector_bool mask;
        CREATE_MASK_BY_SIZE(mask, T, cur);

        vlds(src0_val, src0_ptr, loop_idx * ele_per_VL, NORM);
        vlds(src1_val, src1_ptr, loop_idx * ele_per_VL, NORM);
        vdiv(dst_val, src0_val, src1_val, mask, MODE_ZEROING);

        if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
          vsts(dst_val, dst_ptr, loop_idx * ele_per_VL, NORM_B16, mask);
        else if constexpr (std::is_same_v<T, int32_t> ||
                           std::is_same_v<T, uint32_t>)
          vsts(dst_val, dst_ptr, loop_idx * ele_per_VL, NORM_B32, mask);
      }
    }
  }
}

template <typename T, size_t dim>
__aiv__ __attribute__((always_inline)) void
vdiv_int_scalar(memref_t<__ubuf__ T, dim> *src0, T src1,
                memref_t<__ubuf__ T, dim> *dst) {
  uint16_t inner_size;
  uint16_t outer_size;
  int64_t s0_outer_stride = 0;
  int64_t d_outer_stride = 0;

  if (dim == 1) {
    inner_size = src0->sizes[0];
    outer_size = 1;
  } else {
    inner_size = src0->sizes[dim - 1];
    outer_size = 1;
    for (int i = 0; i < dim - 1; ++i)
      outer_size *= src0->sizes[i];

    s0_outer_stride = src0->strides[dim - 2];
    d_outer_stride = dst->strides[dim - 2];
  }

  constexpr int dsize = sizeof(T);
  uint32_t ele_per_VL = VL_IN_BYTE / dsize;

  __ubuf__ T *src0_base = src0->aligned + src0->offset;
  __ubuf__ T *dst_base = dst->aligned + dst->offset;

  uint16_t loop_times = CEIL_DIV(inner_size, ele_per_VL);

  for (uint16_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    __ubuf__ T *src0_ptr = src0_base + outer_idx * s0_outer_stride;
    __ubuf__ T *dst_ptr = dst_base + outer_idx * d_outer_stride;

    __VEC_SCOPE__ {
      VectorReg<T> src0_val, src1_val, dst_val;
      vbr(src1_val, src1);
      for (uint16_t loop_idx = 0; loop_idx < loop_times; ++loop_idx) {
        uint32_t remain = inner_size - loop_idx * ele_per_VL;
        uint32_t cur = remain > ele_per_VL ? ele_per_VL : remain;

        vector_bool mask;
        CREATE_MASK_BY_SIZE(mask, T, cur);

        vlds(src0_val, src0_ptr, loop_idx * ele_per_VL, NORM);
        vdiv(dst_val, src0_val, src1_val, mask, MODE_ZEROING);

        if constexpr (std::is_same_v<T, int16_t>)
          vsts(dst_val, dst_ptr, loop_idx * ele_per_VL, NORM_B16, mask);
        else if constexpr (dsize == BYTES_B32)
          vsts(dst_val, dst_ptr, loop_idx * ele_per_VL, NORM_B32, mask);
      }
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) VectorReg<T>
ave_vdiv(VectorReg<T> src0, VectorReg<T> src1, ave_preg preg) {
  vector_bool p = convertAVEPregToVecBool(preg);
  VectorReg<T> sret;
  vdiv(sret, src0, src1, p, MODE_ZEROING);
  return sret;
}

__aiv__ __attribute__((always_inline)) vector_f32
vdiv_float_hp(vector_f32 src0, vector_f32 src1, ave_preg preg) {
  constexpr uint32_t infNanBound = 0xff800000u;
  constexpr uint32_t signBitNum = 0x80000000u;
  VectorReg<uint32_t> regNegZero;
  VectorReg<float> tmpDst, dst;
  VectorReg<float> r, z, y;
  VectorReg<uint32_t> infNan;
  VectorReg<float> rPre, rNext, zPre, zNext;

  vector_bool mask = convertAVEPregToVecBool(preg);
  // vdiv(dst_val, src0, src1, mask, MODE_ZEROING);

  vector_bool cmpMaskReg;
  vector_bool infNanCmp;
  vector_bool zeroCmp;

  vdup((VectorReg<uint32_t> &)regNegZero, signBitNum, mask, MODE_ZEROING);
  vdiv(z, src0, src1, mask, MODE_ZEROING);
  vor(infNan, (VectorReg<uint32_t> &)z, (VectorReg<uint32_t> &)regNegZero, mask,
      MODE_ZEROING);
  tmpDst = z;
  vcmps_eq(zeroCmp, z, 0.0f, mask);
  vcmps_ge(infNanCmp, infNan, infNanBound, mask);
  por(infNanCmp, infNanCmp, zeroCmp, mask);

  vmuls(y, src1, -1.0f, mask, MODE_ZEROING);
  r = src0;
  vmula(r, z, y, mask, MODE_ZEROING);

  vadds((VectorReg<int32_t> &)zPre, (VectorReg<int32_t> &)z, -1, mask,
        MODE_ZEROING);
  vadds((VectorReg<int32_t> &)zNext, (VectorReg<int32_t> &)z, 1, mask,
        MODE_ZEROING);

  rPre = src0;
  rNext = src0;

  vmula(rPre, zPre, y, mask, MODE_ZEROING);
  vmula(rNext, zNext, y, mask, MODE_ZEROING);

  vabs(r, r, mask, MODE_ZEROING);
  vabs(rPre, rPre, mask, MODE_ZEROING);
  vabs(rNext, rNext, mask, MODE_ZEROING);
  vcmp_lt(cmpMaskReg, r, rPre, mask);
  vsel(r, r, rPre, cmpMaskReg);
  vsel(z, z, zPre, cmpMaskReg);

  vcmp_lt(cmpMaskReg, rNext, r, mask);
  vsel(z, zNext, z, cmpMaskReg);
  vsel(dst, tmpDst, z, infNanCmp);
  return dst;
}

extern "C" {
REGISTE_INT_VDIV(vdiv, 1, int16_t);
REGISTE_INT_VDIV(vdiv, 1, int32_t);
REGISTE_INT_VDIV(vdiv, 2, int16_t);
REGISTE_INT_VDIV(vdiv, 2, int32_t);
REGISTE_INT_VDIV(vdiv, 3, int16_t);
REGISTE_INT_VDIV(vdiv, 3, int32_t);
REGISTE_INT_VDIV(vdiv, 4, int16_t);
REGISTE_INT_VDIV(vdiv, 4, int32_t);
REGISTE_INT_VDIV(vdiv, 5, int16_t);
REGISTE_INT_VDIV(vdiv, 5, int32_t);
REGISTE_INT_VDIV(vdiv, 6, int16_t);
REGISTE_INT_VDIV(vdiv, 6, int32_t);
REGISTE_INT_VDIV(vdiv, 7, int16_t);
REGISTE_INT_VDIV(vdiv, 7, int32_t);
REGISTE_INT_VDIV(vdiv, 8, int16_t);
REGISTE_INT_VDIV(vdiv, 8, int32_t);

REGISTE_INT_VDIV_SCALAR(vdiv_vs, 1, int16_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 1, int32_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 2, int16_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 2, int32_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 3, int16_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 3, int32_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 4, int16_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 4, int32_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 5, int16_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 5, int32_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 6, int16_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 6, int32_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 7, int16_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 7, int32_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 8, int16_t);
REGISTE_INT_VDIV_SCALAR(vdiv_vs, 8, int32_t);

REGISTE_INT_VDIV(vdiv, 1, uint16_t);
REGISTE_INT_VDIV(vdiv, 1, uint32_t);

REGISTE_DIV_VV(vdiv, int16_t);
REGISTE_DIV_VV(vdiv, int32_t);
REGISTE_DIV_VV(vdiv, uint16_t);
REGISTE_DIV_VV(vdiv, uint32_t);

RETURN_REGISTE_FLOAT_VDIV_HP(vdivfhp);
}
#endif
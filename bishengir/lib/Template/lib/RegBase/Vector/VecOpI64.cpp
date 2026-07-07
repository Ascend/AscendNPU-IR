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

#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"

#if defined(__DAV_C310__)
__aiv__ __attribute__((always_inline)) vector_f32
ave_cast_i64_to_float(vector_2xvl_s64 *src, ave_preg preg, RoundType rnd) {
  vector_f32 ret{};
  if (rnd == RoundType::RoundR) {
    vcvt(ret, *src, ROUND_R);
  } else if (rnd == RoundType::RoundA) {
    vcvt(ret, *src, ROUND_A);
  } else if (rnd == RoundType::RoundF) {
    vcvt(ret, *src, ROUND_F);
  } else if (rnd == RoundType::RoundC) {
    vcvt(ret, *src, ROUND_C);
  } else if (rnd == RoundType::RoundZ) {
    vcvt(ret, *src, ROUND_Z);
  }
  return ret;
}

__aiv__ __attribute__((always_inline)) vector_f32
ave_cast_u64_to_float(vector_2xvl_s64 *src, ave_preg preg, RoundType rnd) {
  vector_f32 ret, retHighVal, lowPartF32{}, highPartF32{};
  vector_bool p = convertAVEPregToVecBool(preg);
  vector_2xvl_u64 unsignSrc = vector_2xvl_u64(*src);
  vector_2xvl_u64 lowPart, highPart, tmp;
  vector_2xvl_s64 lowPartS64, highPartS64;

  vshrs(highPart, unsignSrc, 32, p);
  vshls(tmp, unsignSrc, 32, p) ;
  vshrs(lowPart, tmp, 32, p);

  vcvt(lowPartS64, lowPart);
  vcvt(highPartS64, highPart);
  if (rnd == RoundType::RoundR) {
    vcvt(lowPartF32, lowPartS64, ROUND_R);
    vcvt(highPartF32, highPartS64, ROUND_R);
  } else if (rnd == RoundType::RoundA) {
    vcvt(lowPartF32, lowPartS64, ROUND_A);
    vcvt(highPartF32, highPartS64, ROUND_A);
  } else if (rnd == RoundType::RoundF) {
    vcvt(lowPartF32, lowPartS64, ROUND_F);
    vcvt(highPartF32, highPartS64, ROUND_F);
  } else if (rnd == RoundType::RoundC) {
    vcvt(lowPartF32, lowPartS64, ROUND_C);
    vcvt(highPartF32, highPartS64, ROUND_C);
  } else if (rnd == RoundType::RoundZ) {
    vcvt(lowPartF32, lowPartS64, ROUND_Z);
    vcvt(highPartF32, highPartS64, ROUND_Z);
  }
  
  vmuls(retHighVal, highPartF32, 1LL << 32, p);
  vadd(ret, retHighVal, lowPartF32, p);
  return ret;
}

__aiv__ __attribute__((always_inline)) void
ave_cast_float_to_i64(vector_2xvl_s64 *sret, vector_f32 src, ave_preg preg,
                      RoundType rnd) {
  vector_bool p = convertAVEPregToVecBool(preg);

  // TODO: Toggle RoundingSaturation
  if (rnd == RoundType::RoundR) {
    vcvt(*sret, src, ROUND_R, RS_ENABLE);
  } else if (rnd == RoundType::RoundA) {
    vcvt(*sret, src, ROUND_A, RS_ENABLE);
  } else if (rnd == RoundType::RoundF) {
    vcvt(*sret, src, ROUND_F, RS_ENABLE);
  } else if (rnd == RoundType::RoundC) {
    vcvt(*sret, src, ROUND_C, RS_ENABLE);
  } else if (rnd == RoundType::RoundZ) {
    vcvt(*sret, src, ROUND_Z, RS_ENABLE);
  }
}

__aiv__ __attribute__((always_inline)) void
ave_cast_float_to_u64(vector_2xvl_u64 *sret, vector_f32 src, ave_preg preg, RoundType rnd) {
  vector_bool p = convertAVEPregToVecBool(preg);
  vector_f32 src_H, highPartF32, highPartF32Mul32, lowPartF32;
  vector_2xvl_s64 highPartI64, lowPartI64;
  vector_2xvl_u64 highPartU64, lowPartU64, retHighVal;
  vector_u32 highPartU32, lowPartU32;

  vmuls(src_H, src, 1.0f / 4294967296.0f, p);
  vtrc(highPartF32, src_H, ROUND_F, p);
  vmuls(highPartF32Mul32, highPartF32, 1LL << 32, p);
  vsub(lowPartF32, src, highPartF32Mul32, p);

  if (rnd == RoundType::RoundR) {
    vcvt(highPartI64, highPartF32, ROUND_R, RS_ENABLE);
    vcvt(highPartU64, highPartI64);
    vcvt(lowPartI64, lowPartF32, ROUND_R, RS_ENABLE);
    vcvt(lowPartU64, lowPartI64);
  } else if (rnd == RoundType::RoundA) {
    vcvt(highPartI64, highPartF32, ROUND_A, RS_ENABLE);
    vcvt(highPartU64, highPartI64);
    vcvt(lowPartI64, lowPartF32, ROUND_A, RS_ENABLE);
    vcvt(lowPartU64, lowPartI64);    
  } else if (rnd == RoundType::RoundF) {
    vcvt(highPartI64, highPartF32, ROUND_F, RS_ENABLE);
    vcvt(highPartU64, highPartI64);
    vcvt(lowPartI64, lowPartF32, ROUND_F, RS_ENABLE);
    vcvt(lowPartU64, lowPartI64);  
  } else if (rnd == RoundType::RoundC) {
    vcvt(highPartI64, highPartF32, ROUND_C, RS_ENABLE);
    vcvt(highPartU64, highPartI64);
    vcvt(lowPartI64, lowPartF32, ROUND_C, RS_ENABLE);
    vcvt(lowPartU64, lowPartI64);
  } else if (rnd == RoundType::RoundZ) {
    vcvt(highPartI64, highPartF32, ROUND_Z, RS_ENABLE);
    vcvt(highPartU64, highPartI64);
    vcvt(lowPartI64, lowPartF32, ROUND_Z, RS_ENABLE);
    vcvt(lowPartU64, lowPartI64);
  }
  vmuls(retHighVal, highPartU64, 1LL << 32, p);
  vadd(*sret, retHighVal, lowPartU64, p);
}

__aiv__ __attribute__((always_inline)) vector_s32
ave_cast_i64_to_int32_t(vector_2xvl_s64 *src, ave_preg preg) {
  vector_s32 ret;
  vcvt(ret, *src);
  return ret;
}

__aiv__ __attribute__((always_inline)) vector_s32
ave_cast_i64_to_i32_sat(vector_2xvl_s64 *src, ave_preg preg) {
  vector_bool p = convertAVEPregToVecBool(preg);
  vector_s32 ret;
  vector_2xvl_s64 s64Sat;
  int64_t I32_MAX_VAL = 2147483647;
  int64_t I32_MIN_VAL = -2147483648;
  vmins(s64Sat, *src, I32_MAX_VAL, p);
  vmaxs(s64Sat, s64Sat, I32_MIN_VAL, p);
  vcvt(ret, s64Sat) ;
  return ret;
}

__aiv__ __attribute__((always_inline)) vector_u32
ave_cast_i64_to_u32_sat(vector_2xvl_s64 *src, ave_preg preg) {
  vector_bool p = convertAVEPregToVecBool(preg);
  vector_u32 ret;
  vector_2xvl_s64 s64Sat;
  vector_2xvl_u64 u64Sat;
  int64_t U32_MAX_VAL = 4294967295;
  vmins(s64Sat, *src, U32_MAX_VAL, p);
  vmaxs(s64Sat, s64Sat, 0, p);
  vcvt(u64Sat, s64Sat) ;
  vcvt(ret, u64Sat) ;
  return ret;
}
 
__aiv__ __attribute__((always_inline)) vector_s32
ave_cast_u64_to_i32_sat(vector_2xvl_u64 *src, ave_preg preg) {
  vector_bool p = convertAVEPregToVecBool(preg);
  vector_s32 ret;
  vector_2xvl_u64 u64Sat;
  vector_u32 u32Sat;
  int64_t U32_MAX_VAL = 4294967295;
  int64_t I32_MAX_VAL = 2147483647;
  vmins(u64Sat, *src, U32_MAX_VAL, p);
  vcvt(u32Sat, u64Sat);
  vmins(u32Sat, u32Sat, I32_MAX_VAL, p);
  ret = (vector_s32)u32Sat;
  return ret;
}
 
__aiv__ __attribute__((always_inline)) vector_u32
ave_cast_u64_to_u32_sat(vector_2xvl_u64 *src, ave_preg preg) {
  vector_bool p = convertAVEPregToVecBool(preg);
  vector_2xvl_u64 u64Sat;
  vector_u32 ret;
  int64_t U32_MAX_VAL = 4294967295;
  vmins(u64Sat, *src, U32_MAX_VAL, p);
  vcvt(ret, u64Sat);
  return ret;
}

__aiv__ __attribute__((always_inline)) void
ave_cast_int32_t_to_i64(vector_2xvl_s64 *sret, vector_s32 src, ave_preg preg) {
  vector_bool p = convertAVEPregToVecBool(preg);

  // TODO: Toggle RoundingSaturation
  vcvt(*sret, src);
}

__aiv__ __attribute__((always_inline)) void
ave_cast_uint32_t_to_i64(vector_2xvl_u64 *sret, vector_u32 src, ave_preg preg) {
  vector_bool p = convertAVEPregToVecBool(preg);

  // TODO: Toggle RoundingSaturation
  vcvt(*sret, src);
}

__aiv__ __attribute__((always_inline)) void ave_vdup_i64(vector_2xvl_s64 *sret,
                                                         vector_2xvl_s64 *src,
                                                         ave_preg preg,
                                                         DupPos pos) {
  vector_bool p = convertAVEPregToVecBool(preg);
  if (pos == DupPos::Lowest) {
    vdup(*sret, *src, p, POS_LOWEST, MODE_ZEROING);
  } else if (pos == DupPos::Highest) {
    vdup(*sret, *src, p, POS_HIGHEST, MODE_ZEROING);
  }
}

__aiv__ __attribute__((always_inline)) void
ave_vci_i64(vector_2xvl_s64 *sret, int64_t index, VCIOrder order) {
  if (order == VCIOrder::Decrease) {
    vci(*sret, index, DEC_ORDER);
  } else {
    vci(*sret, index);
  }
}

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_vload_BRC_B64_int64_t(vector_2xvl_s64 *sret,
                          memref_t<__ubuf__ int64_t, DIM> *base,
                          int64_t offset) {
  __ubuf__ int64_t *base_ptr = base->aligned + base->offset;
  __ubuf__ int64_t *offset_ptr = base_ptr + offset;
  int64_t scalar_value = offset_ptr[0];
  vbr(*sret, scalar_value);
}

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_vload_NORM_B64_int64_t(vector_2xvl_s64 *sret,
                           memref_t<__ubuf__ int64_t, DIM> *base,
                           int64_t offset) {
  __ubuf__ int64_t *base_ptr = base->aligned + base->offset;
  __ubuf__ int64_t *offset_ptr = base_ptr + offset;
  vlds(*sret, offset_ptr, 0);
}

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_vload_NORM_B64_int64_t_unalign(vector_2xvl_s64 *sret,
                           memref_t<__ubuf__ int64_t, DIM> *base,
                           int64_t offset) {
  __ubuf__ int64_t *base_ptr = base->aligned + base->offset;
  __ubuf__ int64_t *offset_ptr = base_ptr + offset;
  vload(*sret, offset_ptr, ADDRESS_UNALIGNED);
}

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_store_ONEPT_B64_int64_t(memref_t<__ubuf__ int64_t, DIM> *base,
                            int64_t offset, ave_preg preg,
                            vector_2xvl_s64 *src) {

  __ubuf__ int64_t *base_ptr = base->aligned + base->offset;
  __ubuf__ int64_t *offset_ptr = base_ptr + offset;
  vstore(offset_ptr, *src, 1, ADDRESS_UNALIGNED);
}

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_store_NORM_B64_int64_t(memref_t<__ubuf__ int64_t, DIM> *base,
                           int64_t offset, ave_preg preg,
                           vector_2xvl_s64 *src) {

  __ubuf__ int64_t *base_ptr = base->aligned + base->offset;
  __ubuf__ int64_t *offset_ptr = base_ptr + offset;
  vector_bool p = convertAVEPregToVecBool(preg);
  vsts(*src, offset_ptr, 0, p);
}

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_store_NORM_B64_int64_t_unalign(memref_t<__ubuf__ int64_t, DIM> *base,
                           int64_t offset, vector_2xvl_s64 *src,
                           uint32_t elementCount) {

  __ubuf__ int64_t *base_ptr = base->aligned + base->offset;
  __ubuf__ int64_t *offset_ptr = base_ptr + offset;
  vstore(offset_ptr, *src, elementCount, ADDRESS_UNALIGNED);
}

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_vgather_int64_t(vector_2xvl_s64 *res,
                    memref_t<__ubuf__ int64_t, DIM> *base,
                    int64_t offset,
                    vector_2xvl_s64 *index_vec,
                    ave_preg mask) {
  int64_t linear = base->offset + offset;
  __ubuf__ int64_t *ptr = base->aligned + linear;
  vector_bool m = convertAVEPregToVecBool(mask);

  vector_2xvl_u64 u64Vec;
  vcvt(u64Vec, *index_vec);
  vector_u32 u32Vec;
  vcvt(u32Vec, u64Vec);

  vgather2(*res, ptr, u32Vec, m);
}

__aiv__ __attribute__((always_inline)) void
ave_vslide_int64_t(vector_2xvl_s64 *ret, vector_2xvl_s64 *src1,
                   vector_2xvl_s64 *src2, int16_t offset) {
  // [To Do] : vslide is not a valid instruction in __DAV_C310__
  // need a way to implement vslide logic
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_offset_vv_1d_vf(memref_t<__ubuf__ T, 1> *src,
                                   memref_t<__ubuf__ T, 1> *dst) {
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  const int64_t size0 = src->sizes[0];
  __VEC_SCOPE__ {
    constexpr int num_per_register = REG_REGISTER_SIZE / sizeof(T);
    uint16_t repeatTimes = size0 / num_per_register;
    uint16_t tailsize = size0 - num_per_register * repeatTimes;
    VectorReg<T> srcReg;
    vector_align ureg0, ureg1;
    vldas(ureg0, src_ptr);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
      vldus(srcReg, ureg0, src_ptr, num_per_register, POST_UPDATE);
      vstus(ureg1, num_per_register, srcReg, dst_ptr, POST_UPDATE);
    }
    if (tailsize > 0) {
      vldus(srcReg, ureg0, src_ptr, tailsize, POST_UPDATE);
      vstus(ureg1, tailsize, srcReg, dst_ptr, POST_UPDATE);
    }
    vstas(ureg1, dst_ptr, 0);
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_offset_vv_1d_vf(memref_t<__ubuf__ bool, 1> *src,
                                   memref_t<__ubuf__ bool, 1> *dst) {
  // convert bool memref to int8 memref
  memref_t<__ubuf__ int8_t, 1> src_as_int8;
  memref_t<__ubuf__ int8_t, 1> dst_as_int8;
  view_as<bool, int8_t, 1>(src, &src_as_int8);
  view_as<bool, int8_t, 1>(dst, &dst_as_int8);

  vector_dma_unalign_offset_vv_1d_vf<int8_t>(&src_as_int8, &dst_as_int8);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_offset_vv_1d_vf(memref_t<__ubuf__ int64_t, 1> *src,
                                   memref_t<__ubuf__ int64_t, 1> *dst) {
  // convert int64 memref to int32 memref
  memref_t<__ubuf__ int32_t, 1> src_as_int32;
  memref_t<__ubuf__ int32_t, 1> dst_as_int32;
  view_as<int64_t, int32_t, 1>(src, &src_as_int32);
  view_as<int64_t, int32_t, 1>(dst, &dst_as_int32);

  vector_dma_unalign_offset_vv_1d_vf<int32_t>(&src_as_int32, &dst_as_int32);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_offset_vv_1d_vf(memref_t<__ubuf__ uint64_t, 1> *src,
                                   memref_t<__ubuf__ uint64_t, 1> *dst) {
  // convert uint64 memref to int32 memref
  memref_t<__ubuf__ int32_t, 1> src_as_int32;
  memref_t<__ubuf__ int32_t, 1> dst_as_int32;
  view_as<uint64_t, int32_t, 1>(src, &src_as_int32);
  view_as<uint64_t, int32_t, 1>(dst, &dst_as_int32);

  vector_dma_unalign_offset_vv_1d_vf<int32_t>(&src_as_int32, &dst_as_int32);
}

template <typename T, bool HasTail, bool HasUnrollTail>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_vv_2d_vf(memref_t<__ubuf__ T, 2> *src,
                            memref_t<__ubuf__ T, 2> *dst) {
  __ubuf__ T *src_base = src->aligned + src->offset;
  __ubuf__ T *dst_base = dst->aligned + dst->offset;
  const int64_t size0 = src->sizes[0];
  const int64_t size1 = src->sizes[1];
  const int64_t src_stride0 = src->strides[0];
  const int64_t dst_stride0 = dst->strides[0];
  __VEC_SCOPE__ {
    constexpr int num_per_register = REG_REGISTER_SIZE / sizeof(T);
    constexpr uint16_t UNROLL = 2;
    uint16_t row_count = (uint16_t)size0;
    uint16_t col_count = (uint16_t)size1;
    uint16_t repeatTimes = col_count / num_per_register;
    uint16_t tailsize = col_count - num_per_register * repeatTimes;
    uint16_t unroll_row_count = row_count / UNROLL * UNROLL;
    uint16_t j_unroll = repeatTimes / UNROLL;
    VectorReg<T> srcReg0, srcReg1, srcReg2, srcReg3;
    vector_align ureg_src0, ureg_src1, ureg_dst0, ureg_dst1;
    for (uint16_t i = 0; i < unroll_row_count; i += UNROLL) {
      __ubuf__ T *row_src0 = src_base + (int64_t)i * src_stride0;
      __ubuf__ T *row_dst0 = dst_base + (int64_t)i * dst_stride0;
      __ubuf__ T *row_src1 = src_base + (int64_t)(i + 1) * src_stride0;
      __ubuf__ T *row_dst1 = dst_base + (int64_t)(i + 1) * dst_stride0;
      vldas(ureg_src0, row_src0);
      vldas(ureg_src1, row_src1);
      for (uint16_t j = 0; j < j_unroll; ++j) {
        vldus(srcReg0, ureg_src0, row_src0, num_per_register, POST_UPDATE);
        vldus(srcReg1, ureg_src1, row_src1, num_per_register, POST_UPDATE);
        vldus(srcReg2, ureg_src0, row_src0, num_per_register, POST_UPDATE);
        vldus(srcReg3, ureg_src1, row_src1, num_per_register, POST_UPDATE);
        vstus(ureg_dst0, num_per_register, srcReg0, row_dst0, POST_UPDATE);
        vstus(ureg_dst1, num_per_register, srcReg1, row_dst1, POST_UPDATE);
        vstus(ureg_dst0, num_per_register, srcReg2, row_dst0, POST_UPDATE);
        vstus(ureg_dst1, num_per_register, srcReg3, row_dst1, POST_UPDATE);
      }
      if constexpr (HasUnrollTail) {
        vldus(srcReg0, ureg_src0, row_src0, num_per_register, POST_UPDATE);
        vldus(srcReg1, ureg_src1, row_src1, num_per_register, POST_UPDATE);
        vstus(ureg_dst0, num_per_register, srcReg0, row_dst0, POST_UPDATE);
        vstus(ureg_dst1, num_per_register, srcReg1, row_dst1, POST_UPDATE);
      }
      if constexpr (HasTail) {
        vldus(srcReg0, ureg_src0, row_src0, tailsize, POST_UPDATE);
        vldus(srcReg1, ureg_src1, row_src1, tailsize, POST_UPDATE);
        vstus(ureg_dst0, tailsize, srcReg0, row_dst0, POST_UPDATE);
        vstus(ureg_dst1, tailsize, srcReg1, row_dst1, POST_UPDATE);
      }
      vstas(ureg_dst0, row_dst0, 0);
      vstas(ureg_dst1, row_dst1, 0);
    }
    for (uint16_t i = unroll_row_count; i < row_count; ++i) {
      __ubuf__ T *row_src = src_base + (int64_t)i * src_stride0;
      __ubuf__ T *row_dst = dst_base + (int64_t)i * dst_stride0;
      vldas(ureg_src0, row_src);
      for (uint16_t j = 0; j < repeatTimes; ++j) {
        vldus(srcReg0, ureg_src0, row_src, num_per_register, POST_UPDATE);
        vstus(ureg_dst0, num_per_register, srcReg0, row_dst, POST_UPDATE);
      }
      if constexpr (HasTail) {
        vldus(srcReg0, ureg_src0, row_src, tailsize, POST_UPDATE);
        vstus(ureg_dst0, tailsize, srcReg0, row_dst, POST_UPDATE);
      }
      vstas(ureg_dst0, row_dst, 0);
    }
  }
}

#define SPECIALIZE_DMA_UNALIGN_2D(OrigT, ConvT)                                \
  template <> __aiv__ __attribute__((always_inline)) void                      \
  vector_dma_unalign_vv_2d_vf<OrigT, true, true>(                             \
      memref_t<__ubuf__ OrigT, 2> *src, memref_t<__ubuf__ OrigT, 2> *dst) {   \
    memref_t<__ubuf__ ConvT, 2> src_as;                                        \
    memref_t<__ubuf__ ConvT, 2> dst_as;                                        \
    view_as<OrigT, ConvT, 2>(src, &src_as);                                    \
    view_as<OrigT, ConvT, 2>(dst, &dst_as);                                    \
    vector_dma_unalign_vv_2d_vf<ConvT, true, true>(&src_as, &dst_as);          \
  }                                                                            \
  template <> __aiv__ __attribute__((always_inline)) void                      \
  vector_dma_unalign_vv_2d_vf<OrigT, true, false>(                            \
      memref_t<__ubuf__ OrigT, 2> *src, memref_t<__ubuf__ OrigT, 2> *dst) {   \
    memref_t<__ubuf__ ConvT, 2> src_as;                                        \
    memref_t<__ubuf__ ConvT, 2> dst_as;                                        \
    view_as<OrigT, ConvT, 2>(src, &src_as);                                    \
    view_as<OrigT, ConvT, 2>(dst, &dst_as);                                    \
    vector_dma_unalign_vv_2d_vf<ConvT, true, false>(&src_as, &dst_as);         \
  }                                                                            \
  template <> __aiv__ __attribute__((always_inline)) void                      \
  vector_dma_unalign_vv_2d_vf<OrigT, false, true>(                            \
      memref_t<__ubuf__ OrigT, 2> *src, memref_t<__ubuf__ OrigT, 2> *dst) {   \
    memref_t<__ubuf__ ConvT, 2> src_as;                                        \
    memref_t<__ubuf__ ConvT, 2> dst_as;                                        \
    view_as<OrigT, ConvT, 2>(src, &src_as);                                    \
    view_as<OrigT, ConvT, 2>(dst, &dst_as);                                    \
    vector_dma_unalign_vv_2d_vf<ConvT, false, true>(&src_as, &dst_as);         \
  }                                                                            \
  template <> __aiv__ __attribute__((always_inline)) void                      \
  vector_dma_unalign_vv_2d_vf<OrigT, false, false>(                           \
      memref_t<__ubuf__ OrigT, 2> *src, memref_t<__ubuf__ OrigT, 2> *dst) {   \
    memref_t<__ubuf__ ConvT, 2> src_as;                                        \
    memref_t<__ubuf__ ConvT, 2> dst_as;                                        \
    view_as<OrigT, ConvT, 2>(src, &src_as);                                    \
    view_as<OrigT, ConvT, 2>(dst, &dst_as);                                    \
    vector_dma_unalign_vv_2d_vf<ConvT, false, false>(&src_as, &dst_as);        \
  }

SPECIALIZE_DMA_UNALIGN_2D(bool, int8_t);
SPECIALIZE_DMA_UNALIGN_2D(int64_t, int32_t);
SPECIALIZE_DMA_UNALIGN_2D(uint64_t, int32_t);

#undef SPECIALIZE_DMA_UNALIGN_2D

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_size_vv_1d_vf(memref_t<__ubuf__ T, 1> *src,
                                 memref_t<__ubuf__ T, 1> *dst) {
  bool useVMax =
      (std::is_same<float, T>::value || std::is_same<half, T>::value);
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  const int64_t size0 = src->sizes[0];
  __VEC_SCOPE__ {
    constexpr int num_per_block = REG_REGISTER_SIZE / sizeof(T);
    uint16_t repeatTimes = CEIL_DIV(size0, num_per_block);
    VectorReg<T> srcReg;
    uint32_t sreg = size0;
    vector_bool preg;
    using StorePattern = std::conditional_t< sizeof(T) == 1, NORM_B8_Type, 
        std::conditional_t< sizeof(T) == 2, NORM_B16_Type, 
        std::conditional_t< sizeof(T) == 4, NORM_B32_Type, void>>>;
    static_assert(!std::is_same_v<StorePattern, void>,
        "Unsupported element size");
    for (uint16_t i = 0; i < repeatTimes; ++i) {
      CREATE_MASK_BY_SIZE(preg, T, sreg);
      vlds(srcReg, src_ptr, i * num_per_block, NORM);
      vsts(srcReg, dst_ptr, i * num_per_block, StorePattern{}, preg);
    }
  }
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_size_vv_1d_vf(memref_t<__ubuf__ bool, 1> *src,
                                 memref_t<__ubuf__ bool, 1> *dst) {
  // convert bool memref to int8 memref
  memref_t<__ubuf__ int8_t, 1> src_as_int8;
  memref_t<__ubuf__ int8_t, 1> dst_as_int8;
  view_as<bool, int8_t, 1>(src, &src_as_int8);
  view_as<bool, int8_t, 1>(dst, &dst_as_int8);

  vector_dma_unalign_size_vv_1d_vf<int8_t>(&src_as_int8, &dst_as_int8);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_size_vv_1d_vf(memref_t<__ubuf__ int64_t, 1> *src,
                                 memref_t<__ubuf__ int64_t, 1> *dst) {
  // convert int64 memref to int32 memref
  memref_t<__ubuf__ int32_t, 1> src_as_int32;
  memref_t<__ubuf__ int32_t, 1> dst_as_int32;
  view_as<int64_t, int32_t, 1>(src, &src_as_int32);
  view_as<int64_t, int32_t, 1>(dst, &dst_as_int32);

  vector_dma_unalign_size_vv_1d_vf<int32_t>(&src_as_int32, &dst_as_int32);
}

template <>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_size_vv_1d_vf(memref_t<__ubuf__ uint64_t, 1> *src,
                                 memref_t<__ubuf__ uint64_t, 1> *dst) {
  // convert uint64 memref to int32 memref
  memref_t<__ubuf__ int32_t, 1> src_as_int32;
  memref_t<__ubuf__ int32_t, 1> dst_as_int32;
  view_as<uint64_t, int32_t, 1>(src, &src_as_int32);
  view_as<uint64_t, int32_t, 1>(dst, &dst_as_int32);

  vector_dma_unalign_size_vv_1d_vf<int32_t>(&src_as_int32, &dst_as_int32);
}

__aiv__ __attribute__((always_inline)) void vector_vmodui(vector_2xvl_s64 *sret,
                                                          vector_2xvl_s64 *src0,
                                                          vector_2xvl_s64 *src1,
                                                          ave_preg preg) {
  vector_bool p = convertAVEPregToVecBool(preg);
  vector_2xvl_u64 *sret_u = (vector_2xvl_u64 *)sret;
  vector_2xvl_u64 *src0_u = (vector_2xvl_u64 *)src0;
  vector_2xvl_u64 *src1_u = (vector_2xvl_u64 *)src1;
  vmod(*sret_u, *src0_u, *src1_u, p, MODE_ZEROING);
}

__aiv__ __attribute__((always_inline)) void vector_vmod(vector_2xvl_s64 *sret,
                                                        vector_2xvl_s64 *src0,
                                                        vector_2xvl_s64 *src1,
                                                        ave_preg preg) {
  vector_bool p = convertAVEPregToVecBool(preg);
  vmod<DivisionMode::DIV_TRUNCATE>(*sret, *src0, *src1, p, MODE_ZEROING);
}

extern "C" {
// Elementwise Ops Definition
REGISTE_BINARY_VV(vadd);
REGISTE_BINARY_VV(vsub);
REGISTE_BINARY_VV(vmul);
REGISTE_BINARY_VV(vdiv);
REGISTE_BINARY_VV(vmax);
REGISTE_BINARY_VV(vmin);
REGISTE_BINARY_VV(vand);
REGISTE_BINARY_VV(vor);
REGISTE_BINARY_VV(vxor);

REGISTE_BINARY_VV_U64(vmax);
REGISTE_BINARY_VV_U64(vmin);

REGISTE_BINARY_VS(vadds);
REGISTE_BINARY_VS(vmuls);
REGISTE_BINARY_VS(vmaxs);
REGISTE_BINARY_VS(vmins);

REGISTE_BINARY_VS_U64(vmaxs);
REGISTE_BINARY_VS_U64(vmins);

REGISTE_UNARY_V(vabs);
REGISTE_UNARY_V(vneg);
REGISTE_UNARY_V(vnot);

REGISTE_VSEL(vsel);

REGISTE_CMP_VV(eq);
REGISTE_CMP_VV(ne);
REGISTE_CMP_VV(gt);
REGISTE_CMP_VV(ge);
REGISTE_CMP_VV(lt);
REGISTE_CMP_VV(le);

REGISTE_CMP_VV_U64(le);
REGISTE_CMP_VV_U64(lt);
REGISTE_CMP_VV_U64(ge);
REGISTE_CMP_VV_U64(gt);

REGISTE_DIV_VV_U64(vdiv);

REGISTE_CMP_VS(vcmps_eq);
REGISTE_CMP_VS(vcmps_ne);
REGISTE_CMP_VS(vcmps_gt);
REGISTE_CMP_VS(vcmps_ge);
REGISTE_CMP_VS(vcmps_lt);
REGISTE_CMP_VS(vcmps_le);

// Vector Load/Store Ops Definition
REGISTE_VEC_LOAD_BRC(vload_BRC_B64, 0);
REGISTE_VEC_LOAD_BRC(vload_BRC_B64, 1);
REGISTE_VEC_LOAD_BRC(vload_BRC_B64, 2);
REGISTE_VEC_LOAD_BRC(vload_BRC_B64, 3);
REGISTE_VEC_LOAD_BRC(vload_BRC_B64, 4);
REGISTE_VEC_LOAD_BRC(vload_BRC_B64, 5);
REGISTE_VEC_LOAD_BRC(vload_BRC_B64, 6);
REGISTE_VEC_LOAD_BRC(vload_BRC_B64, 7);
REGISTE_VEC_LOAD_BRC(vload_BRC_B64, 8);
REGISTE_VEC_LOAD_NORM(vload_NORM, 0);
REGISTE_VEC_LOAD_NORM(vload_NORM, 1);
REGISTE_VEC_LOAD_NORM(vload_NORM, 2);
REGISTE_VEC_LOAD_NORM(vload_NORM, 3);
REGISTE_VEC_LOAD_NORM(vload_NORM, 4);
REGISTE_VEC_LOAD_NORM(vload_NORM, 5);
REGISTE_VEC_LOAD_NORM(vload_NORM, 6);
REGISTE_VEC_LOAD_NORM(vload_NORM, 7);
REGISTE_VEC_LOAD_NORM(vload_NORM, 8);
REGISTE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 0);
REGISTE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 1);
REGISTE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 2);
REGISTE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 3);
REGISTE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 4);
REGISTE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 5);
REGISTE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 6);
REGISTE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 7);
REGISTE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 8);
REGISTE_VEC_STORE_OPT(masked_store_ONEPT_B64, 0);
REGISTE_VEC_STORE_OPT(masked_store_ONEPT_B64, 1);
REGISTE_VEC_STORE_OPT(masked_store_ONEPT_B64, 2);
REGISTE_VEC_STORE_OPT(masked_store_ONEPT_B64, 3);
REGISTE_VEC_STORE_OPT(masked_store_ONEPT_B64, 4);
REGISTE_VEC_STORE_OPT(masked_store_ONEPT_B64, 5);
REGISTE_VEC_STORE_OPT(masked_store_ONEPT_B64, 6);
REGISTE_VEC_STORE_OPT(masked_store_ONEPT_B64, 7);
REGISTE_VEC_STORE_OPT(masked_store_ONEPT_B64, 8);
REGISTE_VEC_STORE_NORM(masked_store_NORM_B64, 0);
REGISTE_VEC_STORE_NORM(masked_store_NORM_B64, 1);
REGISTE_VEC_STORE_NORM(masked_store_NORM_B64, 2);
REGISTE_VEC_STORE_NORM(masked_store_NORM_B64, 3);
REGISTE_VEC_STORE_NORM(masked_store_NORM_B64, 4);
REGISTE_VEC_STORE_NORM(masked_store_NORM_B64, 5);
REGISTE_VEC_STORE_NORM(masked_store_NORM_B64, 6);
REGISTE_VEC_STORE_NORM(masked_store_NORM_B64, 7);
REGISTE_VEC_STORE_NORM(masked_store_NORM_B64, 8);
REGISTE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 0);
REGISTE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 1);
REGISTE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 2);
REGISTE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 3);
REGISTE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 4);
REGISTE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 5);
REGISTE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 6);
REGISTE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 7);
REGISTE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 8);

REGISTER_VEC_GATHER(vgather, 0);
REGISTER_VEC_GATHER(vgather, 1);
REGISTER_VEC_GATHER(vgather, 2);
REGISTER_VEC_GATHER(vgather, 3);
REGISTER_VEC_GATHER(vgather, 4);
REGISTER_VEC_GATHER(vgather, 5);
REGISTER_VEC_GATHER(vgather, 6);
REGISTER_VEC_GATHER(vgather, 7);
REGISTER_VEC_GATHER(vgather, 8);

REGISTE_CAST_I64_TO_F32();
REGISTE_CAST_U64_TO_F32();
REGISTE_CAST_F32_TO_I64();
REGISTE_CAST_F32_TO_U64();
REGISTE_CAST_I64_TO_I32();
REGISTE_CAST_I32_TO_I64();
REGISTE_CAST_U32_TO_I64();

REGISTE_CAST_I64_TO_I32_SAT();
REGISTE_CAST_I64_TO_U32_SAT();
REGISTE_CAST_U64_TO_I32_SAT();
REGISTE_CAST_U64_TO_U32_SAT();

REGISTE_VBR(vbr);
REGISTE_VDUPS(vdups);
REGISTE_VDUP(vdup);

REGISTE_REDUCTION_I64(vcadd);
REGISTE_REDUCTION_I64(vcmax);
REGISTE_REDUCTION_I64(vcmin);

REGISTE_REDUCTION_U64(vcmax);
REGISTE_REDUCTION_U64(vcmin);

REGISTE_VCI(vci);
REGISTE_VSLIDE(vslide);

REGISTE_VINTLV(vintlv);
REGISTE_VDINTLV(vdintlv);

REGISTE_BINARY_DMA_UNALIGN_SIZE(float);
REGISTE_BINARY_DMA_UNALIGN_SIZE(half);
REGISTE_BINARY_DMA_UNALIGN_SIZE(bfloat16_t);
REGISTE_BINARY_DMA_UNALIGN_SIZE(float8_e4m3_t);
REGISTE_BINARY_DMA_UNALIGN_SIZE(float8_e5m2_t);
REGISTE_BINARY_DMA_UNALIGN_SIZE(int8_t);
REGISTE_BINARY_DMA_UNALIGN_SIZE(uint8_t);
REGISTE_BINARY_DMA_UNALIGN_SIZE(int16_t);
REGISTE_BINARY_DMA_UNALIGN_SIZE(uint16_t);
REGISTE_BINARY_DMA_UNALIGN_SIZE(int32_t);
REGISTE_BINARY_DMA_UNALIGN_SIZE(uint32_t);


REGISTE_BINARY_DMA_UNALIGN_OFFSET(float);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(half);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(bfloat16_t);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(float8_e4m3_t);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(float8_e5m2_t);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(int8_t);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(uint8_t);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(int16_t);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(uint16_t);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(int32_t);
REGISTE_BINARY_DMA_UNALIGN_OFFSET(uint32_t);

REGISTE_BINARY_DMA_UNALIGN_2D(float);
REGISTE_BINARY_DMA_UNALIGN_2D(half);
REGISTE_BINARY_DMA_UNALIGN_2D(bfloat16_t);
REGISTE_BINARY_DMA_UNALIGN_2D(float8_e4m3_t);
REGISTE_BINARY_DMA_UNALIGN_2D(float8_e5m2_t);
REGISTE_BINARY_DMA_UNALIGN_2D(int8_t);
REGISTE_BINARY_DMA_UNALIGN_2D(uint8_t);
REGISTE_BINARY_DMA_UNALIGN_2D(int16_t);
REGISTE_BINARY_DMA_UNALIGN_2D(uint16_t);
REGISTE_BINARY_DMA_UNALIGN_2D(int32_t);
REGISTE_BINARY_DMA_UNALIGN_2D(uint32_t);
REGISTE_BINARY_DMA_UNALIGN_2D(int64_t);
REGISTE_BINARY_DMA_UNALIGN_2D(uint64_t);
REGISTE_BINARY_DMA_UNALIGN_2D(bool);


REGISTE_SHIFT_VV(vshr);
REGISTE_SHIFT_VV(vshl);
REGISTE_SHIFT_VS(vshrs);
REGISTE_SHIFT_VS(vshls);

REGISTE_VMOD_VV(vmod);
REGISTE_VMOD_VV(vmodui);
}
#endif

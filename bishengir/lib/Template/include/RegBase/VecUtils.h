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

#ifndef HIVMAVE_MLIR_TEMPLATE_VECTOR_UTILS_H
#define HIVMAVE_MLIR_TEMPLATE_VECTOR_UTILS_H

#include "Utils.h"
#include <type_traits>

#define AVE_PREDICATE_LENGTH 256

template <typename T>
struct VectorTy;

// Helper macro to define specializations
#define DEFINE_VECTOR_MAPPING(ScalarType, VectorType)                          \
  template <>                                                                  \
  struct VectorTy<ScalarType> {                                                \
    using type = VectorType;                                                   \
  }

// Floating-point
DEFINE_VECTOR_MAPPING(float, vector_f32);
DEFINE_VECTOR_MAPPING(half, vector_f16);
DEFINE_VECTOR_MAPPING(bfloat16_t, vector_bf16);
#if defined(__DAV_C310__)
DEFINE_VECTOR_MAPPING(float8_e4m3_t, vector_f8e4m3);
DEFINE_VECTOR_MAPPING(float8_e5m2_t, vector_f8e5m2);
// 64-bit integers
DEFINE_VECTOR_MAPPING(int64_t, vector_2xvl_s64);
DEFINE_VECTOR_MAPPING(uint64_t, vector_2xvl_u64);
#endif

// 32-bit integers
DEFINE_VECTOR_MAPPING(int32_t, vector_s32);
DEFINE_VECTOR_MAPPING(uint32_t, vector_u32);

// 16-bit integers
DEFINE_VECTOR_MAPPING(int16_t, vector_s16);
DEFINE_VECTOR_MAPPING(uint16_t, vector_u16);

// 8-bit integers
DEFINE_VECTOR_MAPPING(int8_t, vector_s8);
DEFINE_VECTOR_MAPPING(uint8_t, vector_u8);

#undef DEFINE_VECTOR_MAPPING // Clean up macro

template <typename T>
using VectorReg = typename VectorTy<T>::type;

#define CREATE_MASK_BY_SIZE(MASK, Type, VALUE)                                 \
  if constexpr (std::is_same<Type, int16_t>::value ||                          \
                std::is_same<Type, uint16_t>::value ||                         \
                std::is_same<Type, half>::value ||                             \
                std::is_same<Type, bfloat16_t>::value)                         \
    (MASK) = plt_b16(VALUE, POST_UPDATE);                                      \
  else if constexpr (std::is_same<Type, int32_t>::value ||                     \
                     std::is_same<Type, uint32_t>::value ||                    \
                     std::is_same<Type, float>::value)                         \
    (MASK) = plt_b32(VALUE, POST_UPDATE);                                      \
  else if constexpr (std::is_same<Type, int8_t>::value ||                      \
                     std::is_same<Type, uint8_t>::value ||                     \
                     std::is_same<Type, float8_e4m3_t>::value ||               \
                     std::is_same<Type, float8_e5m2_t>::value)                 \
    (MASK) = plt_b8(VALUE, POST_UPDATE);                                       \
  else                                                                         \
    (MASK) = plt_b32(VALUE, POST_UPDATE)

#define CREATE_MASK_BY_PAT(MASK, Type, PAT)                                    \
  if constexpr (std::is_same<Type, int16_t>::value ||                          \
                std::is_same<Type, uint16_t>::value ||                         \
                std::is_same<Type, half>::value)                               \
    (MASK) = pset_b16(PAT);                                                     \
  else if constexpr (std::is_same<Type, int32_t>::value ||                     \
                     std::is_same<Type, uint32_t>::value ||                    \
                     std::is_same<Type, float>::value)                         \
    (MASK) = pset_b32(PAT);                                                     \
  else if constexpr (std::is_same<Type, int8_t>::value ||                      \
                     std::is_same<Type, uint8_t>::value)                       \
    (MASK) = pset_b8(PAT);                                                      \
  else                                                                         \
    (MASK) = pset_b32(PAT)

#define DEFINE_AVE_VECTOR_TYPE(BASE_TYPE, NAME, SIZE)                          \
  typedef BASE_TYPE NAME __attribute__((ext_vector_type(SIZE)))

#define DEFINE_AVE_PREG_TYPE(BASE_TYPE, NAME, SIZE)                            \
  typedef BASE_TYPE NAME __attribute__((ext_vector_type(SIZE)))

DEFINE_AVE_VECTOR_TYPE(long long, ave_vector_i64, 64); // <64xi64>
DEFINE_AVE_PREG_TYPE(bool, ave_preg, AVE_PREDICATE_LENGTH);

#if defined(__DAV_C310__)
// helper Micros to convert between ave types and backend defined types
#define convertAVEVec64To2xvl64(v) (*reinterpret_cast<vector_2xvl_s64 *>(&(v)))
#define convert2xvl64ToAVEVec64(v) (*reinterpret_cast<ave_vector_i64 *>(&(v)))
#define convertAVEPregToVecBool(p) (*reinterpret_cast<vector_bool *>(&(p)))
#define convertVecBoolToAVEPreg(p) (*reinterpret_cast<ave_preg *>(&(p)))

enum class RoundType : uint32_t {
  RoundR = 0,
  RoundA = 1,
  RoundF = 2,
  RoundC = 3,
  RoundZ = 4,
  RoundO = 5,
  RoundH = 6
};

// aligned with HIVM_VFBroadcastVectorOp
enum class DupPos : uint32_t { Highest = 0, Lowest = 1 };

enum class VCIOrder : uint32_t { Increase = 0, Decrease = 1 };

__aiv__ __attribute__((always_inline)) void
ave_vslide_int64_t(vector_2xvl_s64 *ret, vector_2xvl_s64 *src1,
                   vector_2xvl_s64 *src2, int16_t offset);

__aiv__ __attribute__((always_inline)) vector_f32
ave_cast_i64_to_float(vector_2xvl_s64 *src, ave_preg preg, RoundType rnd);

__aiv__ __attribute__((always_inline)) void
ave_cast_float_to_i64(vector_2xvl_s64 *sret, vector_f32 src, ave_preg preg,
                      RoundType rnd);

__aiv__ __attribute__((always_inline)) void
ave_cast_float_to_u64(vector_2xvl_u64 *sret, vector_f32 src, ave_preg preg,
                      RoundType rnd);

__aiv__ __attribute__((always_inline)) vector_s32
ave_cast_i64_to_int32_t(vector_2xvl_s64 *src, ave_preg preg);

__aiv__ __attribute__((always_inline)) void
ave_cast_int32_t_to_i64(vector_2xvl_s64 *sret, vector_s32 src, ave_preg preg);

__aiv__ __attribute__((always_inline)) void
ave_cast_uint32_t_to_i64(vector_2xvl_u64 *sret, vector_u32 src,
                             ave_preg preg);

__aiv__ __attribute__((always_inline)) void ave_vdup_i64(vector_2xvl_s64 *sret,
                                                         vector_2xvl_s64 *src,
                                                         ave_preg preg,
                                                         DupPos pos);

template <typename T>
__aiv__ __attribute__((always_inline)) VectorReg<T>
ave_vdiv(VectorReg<T> src0, VectorReg<T> src1, ave_preg preg);

__aiv__ __attribute__((always_inline)) void
ave_vci_i64(vector_2xvl_s64 *sret, int64_t index, VCIOrder order);

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_vload_BRC_B64_int64_t(vector_2xvl_s64 *sret,
                          memref_t<__ubuf__ int64_t, DIM> *base,
                          int64_t offset);

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_vload_NORM_B64_int64_t(vector_2xvl_s64 *sret,
                           memref_t<__ubuf__ int64_t, DIM> *base,
                           int64_t offset);

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_vload_NORM_B64_int64_t_unalign(vector_2xvl_s64 *sret,
                           memref_t<__ubuf__ int64_t, DIM> *base,
                           int64_t offset);

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_store_NORM_B64_int64_t(memref_t<__ubuf__ int64_t, DIM> *base,
                           int64_t offset, ave_preg preg, vector_2xvl_s64 *src);

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_store_NORM_B64_int64_t_unalign(memref_t<__ubuf__ int64_t, DIM> *base,
                           int64_t offset, vector_2xvl_s64 *src,
                           uint32_t elementCount);

template <int DIM>
__aiv__ __attribute__((always_inline)) void
ave_store_ONEPT_B64_int64_t(memref_t<__ubuf__ int64_t, DIM> *base,
                            int64_t offset, ave_preg preg,
                            vector_2xvl_s64 *src);

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_size_vv_1d_vf(memref_t<__ubuf__ T, 1> *src,
                                 memref_t<__ubuf__ T, 1> *dst);

template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_offset_vv_1d_vf(memref_t<__ubuf__ T, 1> *src,
                                   memref_t<__ubuf__ T, 1> *dst);

template <typename T, bool HasTail = true, bool HasUnrollTail = true>
__aiv__ __attribute__((always_inline)) void
vector_dma_unalign_vv_2d_vf(memref_t<__ubuf__ T, 2> *src,
                             memref_t<__ubuf__ T, 2> *dst);

template <typename T, size_t dim>
__aiv__ __attribute__((always_inline)) void
vdiv_int(memref_t<__ubuf__ T, dim> *src0,
                 memref_t<__ubuf__ T, dim> *src1,
                 memref_t<__ubuf__ T, dim> *dst);

template <typename T, size_t dim>
__aiv__ __attribute__((always_inline)) void
vdiv_int_scalar(memref_t<__ubuf__ T, dim> *src0,
                 T src1,
                 memref_t<__ubuf__ T, dim> *dst);

// elementwise binary vector-vector (e.g vadd(...))
#define DECLARE_BINARY_VV(op_name)                                             \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret,                      \
                                   vector_2xvl_s64 *src0,                      \
                                   vector_2xvl_s64 *src1, ave_preg preg)

#define DECLARE_BINARY_VV_U64(op_name)                                         \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_uint64_t(vector_2xvl_u64 *sret,                     \
                                    vector_2xvl_u64 *src0,                     \
                                    vector_2xvl_u64 *src1, ave_preg preg)

#define REGISTE_BINARY_VV(op_name)                                             \
  DECLARE_BINARY_VV(op_name) {                                                 \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src0, *src1, p, MODE_ZEROING);                             \
  }

#define REGISTE_BINARY_VV_U64(op_name)                                         \
  DECLARE_BINARY_VV_U64(op_name) {                                             \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src0, *src1, p, MODE_ZEROING);                             \
  }

// elementwise binary vector-scalar(e.g vadds(...))
#define DECLARE_BINARY_VS(op_name)                                             \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret,                      \
                                   vector_2xvl_s64 *src0, int64_t src1,        \
                                   ave_preg preg)

#define DECLARE_BINARY_VS_U64(op_name)                                         \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_uint64_t(vector_2xvl_u64 *sret,                     \
                                   vector_2xvl_u64 *src0, uint64_t src1,       \
                                   ave_preg preg)

#define REGISTE_BINARY_VS(op_name)                                             \
  DECLARE_BINARY_VS(op_name) {                                                 \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src0, src1, p, MODE_ZEROING);                              \
  }

#define REGISTE_BINARY_VS_U64(op_name)                                         \
  DECLARE_BINARY_VS_U64(op_name) {                                             \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src0, src1, p, MODE_ZEROING);                              \
  }

// Elementwise unary (e.g., vabs(...))
#define DECLARE_UNARY_V(op_name)                                               \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret,                      \
                                   vector_2xvl_s64 *src, ave_preg preg)

#define REGISTE_UNARY_V(op_name)                                               \
  DECLARE_UNARY_V(op_name) {                                                   \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src, p, MODE_ZEROING);                                     \
  }

// Elementwise ternary vector-vector (e.g., vfma(...))
#define DECLARE_TERNARY(op_name)                                               \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret,                      \
                                   vector_2xvl_s64 *src0,                      \
                                   vector_2xvl_s64 *src1, ave_preg preg)

#define REGISTE_TERNARY(op_name)                                               \
  DECLARE_TERNARY(op_name) {                                                   \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src0, *src1, p);                                           \
  }

#define DECLARE_VSEL(op_name)                                                  \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret, ave_preg preg,       \
                                   vector_2xvl_s64 *src0,                      \
                                   vector_2xvl_s64 *src1)

#define REGISTE_VSEL(op_name)                                                  \
  DECLARE_VSEL(op_name) {                                                      \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src0, *src1, p);                                           \
  }

#define DECLARE_CMP_VV(op)                                                     \
  __aiv__ __attribute__((always_inline))                                       \
  ave_preg _mlir_ciface_vcmp_##op##_int64_t(                                   \
      vector_2xvl_s64 *src0, vector_2xvl_s64 *src1, ave_preg preg)

#define DECLARE_DIV_VV_U64(op_name)                                            \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_uint64_t(vector_2xvl_u64 *sret,                     \
                                   vector_2xvl_u64 *src0,                      \
                                   vector_2xvl_u64 *src1, ave_preg preg)
 
#define REGISTE_DIV_VV_U64(op_name)                                            \
  DECLARE_DIV_VV_U64(op_name) {                                                \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src0, *src1, p, MODE_ZEROING);                             \
  }

#define DECLARE_DIV_VV(op_name, dtype)                                         \
  __aiv__ __attribute__((always_inline)) VectorReg<dtype>                      \
  _mlir_ciface_##op_name##_##dtype(VectorReg<dtype> src0, VectorReg<dtype> src1, \
                                 ave_preg preg)

#define REGISTE_DIV_VV(op_name, dtype)                                         \
  DECLARE_DIV_VV(op_name, dtype) { return ave_vdiv<dtype>(src0, src1, preg); }

#define REGISTE_CMP_VV(op)                                                     \
  DECLARE_CMP_VV(op) {                                                         \
    vector_bool returnBool;                                                    \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    vcmp_##op(returnBool, *src0, *src1, p);                                    \
    ave_preg ret = convertVecBoolToAVEPreg(returnBool);                        \
    return ret;                                                                \
  }

// FIXME uint version uses s64 vectors since bishengir utilizes only int64 in
//       pipeline. After support of uint64 in MLIR pipeline we should switch to
//       u64 vectors and remove casts
#define DECLARE_CMP_VV_U64(op_name)                                            \
  __aiv__ __attribute__((always_inline))                                       \
  ave_preg _mlir_ciface_vcmp_u##op_name##_uint64_t(                            \
      vector_2xvl_s64 *src0, vector_2xvl_s64 *src1, ave_preg preg)

#define REGISTE_CMP_VV_U64(op)                                                 \
  DECLARE_CMP_VV_U64(op) {                                                     \
    vector_2xvl_u64 unsignedSrc0 = vector_2xvl_u64(*src0);                     \
    vector_2xvl_u64 unsignedSrc1 = vector_2xvl_u64(*src1);                     \
    vector_bool returnBool;                                                    \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    vcmp_##op(returnBool, unsignedSrc0, unsignedSrc1, p);                      \
    ave_preg ret = convertVecBoolToAVEPreg(returnBool);                        \
    return ret;                                                                \
  }

#define DECLARE_CMP_VS(op_name)                                                \
  __aiv__ __attribute__((always_inline)) ave_preg                              \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *src0, int64_t src1,        \
                                   ave_preg preg)

#define REGISTE_CMP_VS(op_name)                                                \
  DECLARE_CMP_VS(op_name) {                                                    \
    vector_bool returnBool;                                                    \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(returnBool, *src0, src1, p);                                       \
    ave_preg ret = convertVecBoolToAVEPreg(returnBool);                        \
    return ret;                                                                \
  }

#define DECLARE_VBR(op_name)                                                   \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret, int64_t src)

#define REGISTE_VBR(op_name)                                                   \
  DECLARE_VBR(op_name) { op_name(*sret, src); }

#define DECLARE_VDUPS(op_name)                                                 \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret, int64_t src,         \
                                   ave_preg preg)

#define REGISTE_VDUPS(op_name)                                                 \
  DECLARE_VDUPS(op_name) { vbr(*sret, src); }

#define DECLARE_VDUP(op_name)                                                  \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(                                            \
      vector_2xvl_s64 *sret, vector_2xvl_s64 *src, ave_preg preg, DupPos pos)

#define REGISTE_VDUP(op_name)                                                  \
  DECLARE_VDUP(op_name) { ave_vdup_i64(sret, src, preg, pos); }

#define DECLARE_VEC_LOAD_BRC(op_name, DIM)                                     \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t_rank##DIM(                                  \
      vector_2xvl_s64 *sret, memref_t<__ubuf__ int64_t, DIM> *base,            \
      int64_t offset)

#define REGISTE_VEC_LOAD_BRC(op_name, DIM)                                     \
  DECLARE_VEC_LOAD_BRC(op_name, DIM) {                                         \
    ave_vload_BRC_B64_int64_t<DIM>(sret, base, offset);                        \
  }

#define DECLARE_VEC_LOAD_NORM(op_name, DIM)                                    \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t_rank##DIM(                                  \
      vector_2xvl_s64 *sret, memref_t<__ubuf__ int64_t, DIM> *base,            \
      int64_t offset)

#define REGISTE_VEC_LOAD_NORM(op_name, DIM)                                    \
  DECLARE_VEC_LOAD_NORM(op_name, DIM) {                                        \
    ave_vload_NORM_B64_int64_t<DIM>(sret, base, offset);                       \
  }

#define DECLARE_VEC_LOAD_NORM_UNALIGN(op_name, DIM)                            \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t_unalign_rank##DIM(                          \
      vector_2xvl_s64 *sret, memref_t<__ubuf__ int64_t, DIM> *base,            \
      int64_t offset)

#define REGISTE_VEC_LOAD_NORM_UNALIGN(op_name, DIM)                            \
  DECLARE_VEC_LOAD_NORM_UNALIGN(op_name, DIM) {                                \
    ave_vload_NORM_B64_int64_t_unalign<DIM>(sret, base, offset);               \
  }

#define DECLARE_VEC_STORE_OPT(op_name, DIM)                                    \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t_rank##DIM(                                  \
      memref_t<__ubuf__ int64_t, DIM> *base, int64_t offset, ave_preg preg,    \
      vector_2xvl_s64 *src)

#define REGISTE_VEC_STORE_OPT(op_name, DIM)                                    \
  DECLARE_VEC_STORE_OPT(op_name, DIM) {                                        \
    ave_store_ONEPT_B64_int64_t<DIM>(base, offset, preg, src);                 \
  }

#define DECLARE_VEC_STORE_NORM(op_name, DIM)                                   \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t_rank##DIM(                                  \
      memref_t<__ubuf__ int64_t, DIM> *base, int64_t offset, ave_preg preg,    \
      vector_2xvl_s64 *src)

#define REGISTE_VEC_STORE_NORM(op_name, DIM)                                   \
  DECLARE_VEC_STORE_NORM(op_name, DIM) {                                       \
    ave_store_NORM_B64_int64_t<DIM>(base, offset, preg, src);                  \
  }

#define DECLARE_VEC_STORE_NORM_UNALIGN(op_name, DIM)                           \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t_unalign_rank##DIM(                          \
      memref_t<__ubuf__ int64_t, DIM> *base, int64_t offset,                   \
      vector_2xvl_s64 *src, uint32_t elementCount)

#define REGISTE_VEC_STORE_NORM_UNALIGN(op_name, DIM)                           \
  DECLARE_VEC_STORE_NORM_UNALIGN(op_name, DIM) {                               \
    ave_store_NORM_B64_int64_t_unalign<DIM>(base, offset, src,                 \
                                    elementCount);                             \
  }

#define DECLARE_VEC_GATHER(op_name, DIM)                                       \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t_rank##DIM(                                  \
      vector_2xvl_s64 *res,                                                    \
      memref_t<__ubuf__ int64_t, DIM> *base,                                   \
      int64_t offset,                                                          \
      vector_2xvl_s64 *index_vec,                                              \
      ave_preg mask)

#define REGISTER_VEC_GATHER(op_name, DIM)                                      \
  DECLARE_VEC_GATHER(op_name, DIM) {                                           \
    ave_vgather_int64_t<DIM>(res, base, offset, index_vec, mask);              \
  }

#define DECLARE_CAST_I64_TO_F32()                                              \
  __aiv__ __attribute__((always_inline)) vector_f32                            \
  _mlir_ciface_cast_int64_t_to_float(vector_2xvl_s64 *src, ave_preg preg,      \
                                     RoundType rnd)

#define REGISTE_CAST_I64_TO_F32()                                              \
  DECLARE_CAST_I64_TO_F32() { return ave_cast_i64_to_float(src, preg, rnd); }

#define DECLARE_CAST_U64_TO_F32()                                              \
  __aiv__ __attribute__((always_inline)) vector_f32                            \
  _mlir_ciface_cast_uint64_t_to_float(vector_2xvl_s64 *src, ave_preg preg,     \
                                      RoundType rnd)

#define REGISTE_CAST_U64_TO_F32()                                              \
  DECLARE_CAST_U64_TO_F32() { return ave_cast_u64_to_float(src, preg, rnd); }

#define DECLARE_CAST_F32_TO_I64()                                              \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_cast_float_to_int64_t(vector_2xvl_s64 *sret, vector_f32 src,    \
                                     ave_preg preg, RoundType rnd)

#define REGISTE_CAST_F32_TO_I64()                                              \
  DECLARE_CAST_F32_TO_I64() { ave_cast_float_to_i64(sret, src, preg, rnd); }

#define DECLARE_CAST_F32_TO_U64()                                              \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_cast_float_to_uint64_t(vector_2xvl_u64 *sret, vector_f32 src,   \
                                     ave_preg preg, RoundType rnd)

#define REGISTE_CAST_F32_TO_U64()                                              \
  DECLARE_CAST_F32_TO_U64() { ave_cast_float_to_u64(sret, src, preg, rnd); }

#define DECLARE_CAST_I64_TO_I32()                                              \
  __aiv__ __attribute__((always_inline)) vector_s32                            \
  _mlir_ciface_cast_int64_t_to_int32_t(vector_2xvl_s64 *src, ave_preg preg)

#define REGISTE_CAST_I64_TO_I32()                                              \
  DECLARE_CAST_I64_TO_I32() { return ave_cast_i64_to_int32_t(src, preg); }

#define DECLARE_CAST_I64_TO_I32_SAT()                                          \
  __aiv__ __attribute__((always_inline)) vector_s32                            \
  _mlir_ciface_cast_int64_t_to_int32_t_sat(vector_2xvl_s64 *src, ave_preg preg)
 
#define REGISTE_CAST_I64_TO_I32_SAT()                                          \
  DECLARE_CAST_I64_TO_I32_SAT() { return ave_cast_i64_to_i32_sat(src, preg); }

#define DECLARE_CAST_I64_TO_U32_SAT()                                          \
  __aiv__ __attribute__((always_inline)) vector_u32                            \
  _mlir_ciface_cast_int64_t_to_uint32_t_sat(vector_2xvl_s64 *src, ave_preg preg)
 
#define REGISTE_CAST_I64_TO_U32_SAT()                                          \
  DECLARE_CAST_I64_TO_U32_SAT() { return ave_cast_i64_to_u32_sat(src, preg); }
 
#define DECLARE_CAST_U64_TO_I32_SAT()                                          \
  __aiv__ __attribute__((always_inline)) vector_s32                            \
  _mlir_ciface_cast_uint64_t_to_int32_t_sat(vector_2xvl_u64 *src, ave_preg preg)
 
#define REGISTE_CAST_U64_TO_I32_SAT()                                          \
  DECLARE_CAST_U64_TO_I32_SAT() { return ave_cast_u64_to_i32_sat(src, preg); }
 
#define DECLARE_CAST_U64_TO_U32_SAT()                                          \
  __aiv__ __attribute__((always_inline)) vector_u32                            \
  _mlir_ciface_cast_uint64_t_to_uint32_t_sat(vector_2xvl_u64 *src, ave_preg preg)
 
#define REGISTE_CAST_U64_TO_U32_SAT()                                          \
  DECLARE_CAST_U64_TO_U32_SAT() { return ave_cast_u64_to_u32_sat(src, preg); }

#define DECLARE_CAST_I32_TO_I64()                                              \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_cast_int32_t_to_int64_t(vector_2xvl_s64 *sret, vector_s32 src,  \
                                       ave_preg preg)

#define REGISTE_CAST_I32_TO_I64()                                              \
  DECLARE_CAST_I32_TO_I64() { ave_cast_int32_t_to_i64(sret, src, preg); }

#define DECLARE_CAST_U32_TO_I64()                                              \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_cast_uint32_t_to_int64_t(vector_2xvl_u64 *sret,                 \
                                            vector_u32 src, ave_preg preg)

#define REGISTE_CAST_U32_TO_I64()                                         \
  DECLARE_CAST_U32_TO_I64() { ave_cast_uint32_t_to_i64(sret, src, preg); }

#define DECLARE_REDUCTION_I64(op_name)                                         \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret,                      \
                                   vector_2xvl_s64 *src, ave_preg preg)

#define DECLARE_REDUCTION_U64(op_name)                                         \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_uint64_t(vector_2xvl_u64 *sret,                     \
                                    vector_2xvl_u64 *src, ave_preg preg)

#define REGISTE_REDUCTION_I64(op_name)                                         \
  DECLARE_REDUCTION_I64(op_name) {                                             \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src, p);                                                   \
  }

#define REGISTE_REDUCTION_U64(op_name)                                         \
  DECLARE_REDUCTION_U64(op_name) {                                             \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    op_name(*sret, *src, p);                                                   \
  }

#define DECLARE_VCI(op_name)                                                   \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret, int64_t index,       \
                                   VCIOrder order)

#define REGISTE_VCI(op_name)                                                   \
  DECLARE_VCI(op_name) { ave_vci_i64(sret, index, order); }

#define DECLARE_VSLIDE(op_name)                                                \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *ret,                       \
                                   vector_2xvl_s64 *src1,                      \
                                   vector_2xvl_s64 *src2, int16_t offset)

#define REGISTE_VSLIDE(op_name)                                                \
  DECLARE_VSLIDE(op_name) { ave_vslide_int64_t(ret, src1, src2, offset); }

#define DECLARE_VINTLV(op_name)                                                \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(                                            \
      vector_2xvl_s64 *ret1, vector_2xvl_s64 *ret2, vector_2xvl_s64 *src1,     \
      vector_2xvl_s64 *src2)

#define REGISTE_VINTLV(op_name)                                                \
  DECLARE_VINTLV(op_name) { op_name(*ret1, *ret2, *src1, *src2); }

#define DECLARE_VDINTLV(op_name)                                               \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(                                            \
      vector_2xvl_s64 *ret1, vector_2xvl_s64 *ret2, vector_2xvl_s64 *src1,     \
      vector_2xvl_s64 *src2)

#define REGISTE_VDINTLV(op_name)                                               \
  DECLARE_VDINTLV(op_name) { op_name(*ret1, *ret2, *src1, *src2); }

// copy ub to ub when src/dst size is not align block.
#define DECLARE_BINARY_DMA_UNALIGN_SIZE(T)                                     \
  __aiv__ __attribute__((always_inline)) void                                  \
  vector_dma_unalign_size_vv_1d_vf_##T(memref_t<__ubuf__ T, 1> *src,           \
                                       memref_t<__ubuf__ T, 1> *dst)

#define REGISTE_BINARY_DMA_UNALIGN_SIZE(T)                                     \
  DECLARE_BINARY_DMA_UNALIGN_SIZE(T) {                                         \
    vector_dma_unalign_size_vv_1d_vf<T>(src, dst);                             \
  }

// copy ub to ub when src/dst offset is not align block.
#define DECLARE_BINARY_DMA_UNALIGN_OFFSET(T)                                   \
  __aiv__ __attribute__((always_inline)) void                                  \
  vector_dma_unalign_offset_vv_1d_vf_##T(memref_t<__ubuf__ T, 1> *src,         \
                                         memref_t<__ubuf__ T, 1> *dst)

#define REGISTE_BINARY_DMA_UNALIGN_OFFSET(T)                                   \
  DECLARE_BINARY_DMA_UNALIGN_OFFSET(T) {                                       \
    vector_dma_unalign_offset_vv_1d_vf<T>(src, dst);                           \
  }

// copy ub to ub 2d when src/dst is not align block.
#define DECLARE_BINARY_DMA_UNALIGN_2D(T)                                       \
  __aiv__ __attribute__((always_inline)) void                                  \
  vector_dma_unalign_vv_2d_vf_##T(memref_t<__ubuf__ T, 2> *src,               \
                                   memref_t<__ubuf__ T, 2> *dst)

#define REGISTE_BINARY_DMA_UNALIGN_2D(T)                                       \
  DECLARE_BINARY_DMA_UNALIGN_2D(T) {                                           \
    vector_dma_unalign_vv_2d_vf<T, true, true>(src, dst);                      \
    vector_dma_unalign_vv_2d_vf<T, true, false>(src, dst);                     \
    vector_dma_unalign_vv_2d_vf<T, false, true>(src, dst);                     \
    vector_dma_unalign_vv_2d_vf<T, false, false>(src, dst);                    \
  }

#define DECLARE_EMBEDDINGGATHER(dim, embty, idxty, ...)                        \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_embedding_gather_##dim##d_##embty##_##idxty(                    \
      memref_t<__gm__ embty, 1> *emb, memref_t<__ubuf__ idxty, dim> *idx,      \
      memref_t<__ubuf__ embty, (dim) + 1> *dst, idxty bound, __VA_ARGS__)

#define DECLARE_EMBEDDINGGATHER_1D(embty, idxty)                               \
  DECLARE_EMBEDDINGGATHER(1, embty, idxty, idxty offset0, idxty offset1,       \
                          idxty numel0, idxty numel1)

#define DECLARE_EMBEDDINGGATHER_2D(embty, idxty)                               \
  DECLARE_EMBEDDINGGATHER(2, embty, idxty, idxty offset0, idxty offset1,       \
                          idxty offset2, idxty numel0, idxty numel1,           \
                          idxty numel2)

#define REGISTE_EMBEDDINGGATHER_1D(embty, idxty)                               \
  DECLARE_EMBEDDINGGATHER_1D(embty, idxty) {                                   \
    idxty off_num[4] = {offset0, numel0, offset1, numel1};                     \
    simt_embedding_gather<1, embty, idxty>(emb, idx, dst, bound, off_num);     \
  }

#define REGISTE_EMBEDDINGGATHER_2D(embty, idxty)                               \
  DECLARE_EMBEDDINGGATHER_2D(embty, idxty) {                                   \
    idxty off_num[6] = {offset0, numel0, offset1, numel1, offset2, numel2};    \
    simt_embedding_gather<2, embty, idxty>(emb, idx, dst, bound, off_num);     \
  }

#define DECLARE_INDEX_PUT(dim, srcty, idxty, ...)                              \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_index_put_##dim##d_##srcty##_##idxty(                           \
      memref_t<__gm__ srcty, 1> *dst, memref_t<__ubuf__ idxty, 1> *idx,        \
      memref_t<__ubuf__ srcty, dim> *val, const int32_t scatterDim,            \
      const int64_t indexBoundary, __VA_ARGS__)

#define DECLARE_INDEX_PUT_2D(srcty, idxty)                                     \
  DECLARE_INDEX_PUT(2, srcty, idxty,                                           \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty startOffset0, const idxty startOffset1,        \
                    const idxty dstStride0, const idxty dstStride1)

#define DECLARE_INDEX_PUT_3D(srcty, idxty)                                     \
  DECLARE_INDEX_PUT(3, srcty, idxty,                                           \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2, const idxty startOffset0,          \
                    const idxty startOffset1, const idxty startOffset2,        \
                    const idxty dstStride0, const idxty dstStride1,            \
                    const idxty dstStride2)

#define DECLARE_INDEX_PUT_4D(srcty, idxty)                                     \
  DECLARE_INDEX_PUT(4, srcty, idxty,                                           \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2, const idxty endOffset3,            \
                    const idxty startOffset0, const idxty startOffset1,        \
                    const idxty startOffset2, const idxty startOffset3,        \
                    const idxty dstStride0, const idxty dstStride1,            \
                    const idxty dstStride2, const idxty dstStride3)

#define DECLARE_INDEX_PUT_5D(srcty, idxty)                                     \
  DECLARE_INDEX_PUT(5, srcty, idxty,                                           \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2, const idxty endOffset3,            \
                    const idxty endOffset4, const idxty startOffset0,          \
                    const idxty startOffset1, const idxty startOffset2,        \
                    const idxty startOffset3, const idxty startOffset4,        \
                    const idxty dstStride0, const idxty dstStride1,            \
                    const idxty dstStride2, const idxty dstStride3,            \
                    const idxty dstStride4)

#define REGISTER_INDEX_PUT_2D(srcty, idxty)                                    \
  DECLARE_INDEX_PUT_2D(srcty, idxty) {                                         \
    const idxty endOffsets[2] = {endOffset0, endOffset1};                      \
    const idxty startOffsets[2] = {startOffset0, startOffset1};                \
    const idxty dstStrides[2] = {dstStride0, dstStride1};                      \
    simt_index_put<2, srcty, idxty>(val, idx, dst, scatterDim, indexBoundary,  \
                                    endOffsets, startOffsets, dstStrides);     \
  }

#define REGISTER_INDEX_PUT_3D(srcty, idxty)                                    \
  DECLARE_INDEX_PUT_3D(srcty, idxty) {                                         \
    const idxty endOffsets[3] = {endOffset0, endOffset1, endOffset2};          \
    const idxty startOffsets[3] = {startOffset0, startOffset1, startOffset2};  \
    const idxty dstStrides[3] = {dstStride0, dstStride1, dstStride2};          \
    simt_index_put<3, srcty, idxty>(val, idx, dst, scatterDim, indexBoundary,  \
                                    endOffsets, startOffsets, dstStrides);     \
  }

#define REGISTER_INDEX_PUT_4D(srcty, idxty)                                    \
  DECLARE_INDEX_PUT_4D(srcty, idxty) {                                         \
    const idxty endOffsets[4] = {endOffset0, endOffset1, endOffset2,           \
                                endOffset3};                                   \
    const idxty startOffsets[4] = {startOffset0, startOffset1, startOffset2,   \
                                startOffset3};                                 \
    const idxty dstStrides[4] = {dstStride0, dstStride1, dstStride2,           \
                                dstStride3};                                   \
    simt_index_put<4, srcty, idxty>(val, idx, dst, scatterDim, indexBoundary,  \
                                    endOffsets, startOffsets, dstStrides);     \
  }

#define REGISTER_INDEX_PUT_5D(srcty, idxty)                                    \
  DECLARE_INDEX_PUT_5D(srcty, idxty) {                                         \
    const idxty endOffsets[5] = {endOffset0, endOffset1, endOffset2,           \
                                endOffset3, endOffset4};                       \
    const idxty startOffsets[5] = {startOffset0, startOffset1, startOffset2,   \
                                startOffset3, startOffset4};                   \
    const idxty dstStrides[5] = {dstStride0, dstStride1, dstStride2,           \
                                dstStride3, dstStride4};                       \
    simt_index_put<5, srcty, idxty>(val, idx, dst, scatterDim, indexBoundary,  \
                                    endOffsets, startOffsets, dstStrides);     \
  }

#define DECLARE_INDEX_SELECT(srcDim, srcty, idxty, idxDim, valDim, ...)        \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_index_select_##srcDim##d_##srcty##_##idxDim##d_##idxty(         \
      memref_t<__gm__ srcty, 1> *src, memref_t<__ubuf__ idxty, idxDim> *idx,   \
      const int32_t gatherDim, const int64_t indexBoundary,                    \
      __VA_ARGS__, memref_t<__ubuf__ srcty, valDim> *val)

#define DECLARE_INDEX_SELECT_2D_1D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT(2, srcty, idxty, 1, 2,                                  \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty startOffset0, const idxty startOffset1,        \
                    const idxty srcStride0, const idxty srcStride1)

#define DECLARE_INDEX_SELECT_2D_2D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT(2, srcty, idxty, 2, 3,                                  \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2,                                    \
                    const idxty startOffset0, const idxty startOffset1,        \
                    const idxty srcStride0, const idxty srcStride1)

#define DECLARE_INDEX_SELECT_3D_1D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT(3, srcty, idxty, 1, 3,                                  \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2, const idxty startOffset0,          \
                    const idxty startOffset1, const idxty startOffset2,        \
                    const idxty srcStride0, const idxty srcStride1,            \
                    const idxty srcStride2)

#define DECLARE_INDEX_SELECT_3D_2D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT(3, srcty, idxty, 2, 4,                                  \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2, const idxty endOffset3,            \
                    const idxty startOffset0, const idxty startOffset1,        \
                    const idxty startOffset2, const idxty srcStride0,          \
                    const idxty srcStride1, const idxty srcStride2)

#define DECLARE_INDEX_SELECT_4D_1D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT(4, srcty, idxty, 1, 4,                                  \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2, const idxty endOffset3,            \
                    const idxty startOffset0, const idxty startOffset1,        \
                    const idxty startOffset2, const idxty startOffset3,        \
                    const idxty srcStride0, const idxty srcStride1,            \
                    const idxty srcStride2, const idxty srcStride3)

#define DECLARE_INDEX_SELECT_4D_2D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT(4, srcty, idxty, 2, 5,                                  \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2, const idxty endOffset3,            \
                    const idxty endOffset4, const idxty startOffset0,          \
                    const idxty startOffset1, const idxty startOffset2,        \
                    const idxty startOffset3, const idxty srcStride0,          \
                    const idxty srcStride1, const idxty srcStride2,            \
                    const idxty srcStride3)

#define DECLARE_INDEX_SELECT_5D_1D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT(5, srcty, idxty, 1, 5,                                  \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2, const idxty endOffset3,            \
                    const idxty endOffset4, const idxty startOffset0,          \
                    const idxty startOffset1, const idxty startOffset2,        \
                    const idxty startOffset3, const idxty startOffset4,        \
                    const idxty srcStride0, const idxty srcStride1,            \
                    const idxty srcStride2, const idxty srcStride3,            \
                    const idxty srcStride4)

#define DECLARE_INDEX_SELECT_5D_2D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT(5, srcty, idxty, 2, 6,                                  \
                    const idxty endOffset0, const idxty endOffset1,            \
                    const idxty endOffset2, const idxty endOffset3,            \
                    const idxty endOffset4, const idxty endOffset5,            \
                    const idxty startOffset0, const idxty startOffset1,        \
                    const idxty startOffset2, const idxty startOffset3,        \
                    const idxty startOffset4, const idxty srcStride0,          \
                    const idxty srcStride1, const idxty srcStride2,            \
                    const idxty srcStride3, const idxty srcStride4)

#define REGISTER_INDEX_SELECT_2D_1D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT_2D_1D(srcty, idxty) {                                   \
    const idxty endOffsets[2] = {endOffset0, endOffset1};                      \
    const idxty startOffsets[2] = {startOffset0, startOffset1};                \
    const idxty srcStrides[2] = {srcStride0, srcStride1};                      \
    simt_index_select<2, 1, 2, srcty, idxty>(src, idx, val, gatherDim,         \
                    indexBoundary, endOffsets, startOffsets, srcStrides);      \
  }

#define REGISTER_INDEX_SELECT_2D_2D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT_2D_2D(srcty, idxty) {                                   \
    const idxty endOffsets[3] = {endOffset0, endOffset1, endOffset2};          \
    const idxty startOffsets[2] = {startOffset0, startOffset1};                \
    const idxty srcStrides[2] = {srcStride0, srcStride1};                      \
    simt_index_select<2, 2, 3, srcty, idxty>(src, idx, val, gatherDim,         \
                    indexBoundary, endOffsets, startOffsets, srcStrides);      \
  }

#define REGISTER_INDEX_SELECT_3D_1D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT_3D_1D(srcty, idxty) {                                   \
    const idxty endOffsets[3] = {endOffset0, endOffset1, endOffset2};          \
    const idxty startOffsets[3] = {startOffset0, startOffset1, startOffset2};  \
    const idxty srcStrides[3] = {srcStride0, srcStride1, srcStride2};          \
    simt_index_select<3, 1, 3, srcty, idxty>(src, idx, val, gatherDim,         \
                    indexBoundary, endOffsets, startOffsets, srcStrides);      \
  }

#define REGISTER_INDEX_SELECT_3D_2D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT_3D_2D(srcty, idxty) {                                   \
    const idxty endOffsets[4] = {endOffset0, endOffset1, endOffset2,           \
                                endOffset3};                                   \
    const idxty startOffsets[3] = {startOffset0, startOffset1, startOffset2};  \
    const idxty srcStrides[3] = {srcStride0, srcStride1, srcStride2};          \
    simt_index_select<3, 2, 4, srcty, idxty>(src, idx, val, gatherDim,         \
                    indexBoundary, endOffsets, startOffsets, srcStrides);      \
  }

#define REGISTER_INDEX_SELECT_4D_1D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT_4D_1D(srcty, idxty) {                                   \
    const idxty endOffsets[4] = {endOffset0, endOffset1, endOffset2,           \
                                endOffset3};                                   \
    const idxty startOffsets[4] = {startOffset0, startOffset1, startOffset2,   \
                                  startOffset3};                               \
    const idxty srcStrides[4] = {srcStride0, srcStride1, srcStride2,           \
                                srcStride3};                                   \
    simt_index_select<4, 1, 4, srcty, idxty>(src, idx, val, gatherDim,         \
                    indexBoundary, endOffsets, startOffsets, srcStrides);      \
  }

#define REGISTER_INDEX_SELECT_4D_2D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT_4D_2D(srcty, idxty) {                                   \
    const idxty endOffsets[5] = {endOffset0, endOffset1, endOffset2,           \
                                endOffset3, endOffset4};                       \
    const idxty startOffsets[4] = {startOffset0, startOffset1, startOffset2,   \
                                  startOffset3};                               \
    const idxty srcStrides[4] = {srcStride0, srcStride1, srcStride2,           \
                                srcStride3};                                   \
    simt_index_select<4, 2, 5, srcty, idxty>(src, idx, val, gatherDim,         \
                    indexBoundary, endOffsets, startOffsets, srcStrides);      \
  }

#define REGISTER_INDEX_SELECT_5D_1D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT_5D_1D(srcty, idxty) {                                   \
    const idxty endOffsets[5] = {endOffset0, endOffset1, endOffset2,           \
                                endOffset3, endOffset4};                       \
    const idxty startOffsets[5] = {startOffset0, startOffset1, startOffset2,   \
                                startOffset3, startOffset4};                   \
    const idxty srcStrides[5] = {srcStride0, srcStride1, srcStride2,           \
                                srcStride3, srcStride4};                       \
    simt_index_select<5, 1, 5, srcty, idxty>(src, idx, val, gatherDim,         \
                    indexBoundary, endOffsets, startOffsets, srcStrides);      \
  }

#define REGISTER_INDEX_SELECT_5D_2D(srcty, idxty)                               \
  DECLARE_INDEX_SELECT_5D_2D(srcty, idxty) {                                   \
    const idxty endOffsets[6] = {endOffset0, endOffset1, endOffset2,           \
                                endOffset3, endOffset4, endOffset5};           \
    const idxty startOffsets[5] = {startOffset0, startOffset1, startOffset2,   \
                                startOffset3, startOffset4};                   \
    const idxty srcStrides[5] = {srcStride0, srcStride1, srcStride2,           \
                                srcStride3, srcStride4};                       \
    simt_index_select<5, 2, 6, srcty, idxty>(src, idx, val, gatherDim,         \
                    indexBoundary, endOffsets, startOffsets, srcStrides);      \
  }

#define DECLARE_INDIRECT_LOAD(dim, dtype, itype)                               \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_indirect_load_##dim##d_##dtype##_##itype(                       \
      memref_t<__gm__ dtype, 1> *src, memref_t<__ubuf__ itype, dim> *idx,      \
      memref_t<__ubuf__ dtype, dim> *dst, memref_t<__ubuf__ bool, dim> *mask,  \
      memref_t<__ubuf__ dtype, dim> *other)

#define DECLARE_INDIRECT_LOAD_NONVOLATILE(dim, dtype, itype)                   \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_indirect_load_nonvolatile_##dim##d_##dtype##_##itype(           \
      memref_t<__gm__ dtype, 1> *src, memref_t<__ubuf__ itype, dim> *idx,      \
      memref_t<__ubuf__ dtype, dim> *dst, memref_t<__ubuf__ bool, dim> *mask,  \
      memref_t<__ubuf__ dtype, dim> *other)

#define REGISTE_INDIRECT_LOAD(dim, dtype, itype)                               \
  DECLARE_INDIRECT_LOAD(dim, dtype, itype) {                                   \
    indirect_load<dim, dtype, itype, true>(src, idx, dst, mask, other);        \
  }                                                                            \
  DECLARE_INDIRECT_LOAD_NONVOLATILE(dim, dtype, itype) {                       \
    indirect_load<dim, dtype, itype, false>(src, idx, dst, mask, other);       \
  }

#define DECLARE_SHIFT_VV(op_name)                                              \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(                                            \
      vector_2xvl_s64 *sret, vector_2xvl_s64 *src0, vector_2xvl_s64 *src1,     \
      ave_preg preg, bool sign)

#define REGISTE_SHIFT_VV(op_name)                                              \
  DECLARE_SHIFT_VV(op_name) {                                                  \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    vector_s32 shift;                                                          \
    vector_f32 tmp;                                                            \
    vcvt(tmp, *src1, ROUND_C);                                                 \
    vcvt(shift, tmp, p, ROUND_C, RS_ENABLE);                                   \
    if (!sign) {                                                               \
      vector_2xvl_u64 unsignSrc0 = vector_2xvl_u64(*src0);                     \
      vector_2xvl_u64 unsignSret;                                              \
      op_name(unsignSret, unsignSrc0, shift, p, MODE_ZEROING);                 \
      *sret = vector_2xvl_s64(unsignSret);                                     \
    } else {                                                                   \
      op_name(*sret, *src0, shift, p, MODE_ZEROING);                           \
    }                                                                          \
  }

#define DECLARE_SHIFT_VS(op_name)                                              \
  __aiv__ __attribute__((always_inline)) void                                  \
  _mlir_ciface_##op_name##_int64_t(vector_2xvl_s64 *sret,                      \
                                   vector_2xvl_s64 *src0, int64_t shift,       \
                                   ave_preg preg, bool sign)

#define REGISTE_SHIFT_VS(op_name)                                              \
  DECLARE_SHIFT_VS(op_name) {                                                  \
    vector_bool p = convertAVEPregToVecBool(preg);                             \
    shift = shift > 64 ?  64 : shift;                                        \
    if (!sign) {                                                               \
      vector_2xvl_u64 unsignSrc0 = vector_2xvl_u64(*src0);                     \
      vector_2xvl_u64 unsignSret;                                              \
      op_name(unsignSret, unsignSrc0, (int32_t)shift, p, MODE_ZEROING);        \
      *sret = vector_2xvl_s64(unsignSret);                                     \
    } else {                                                                   \
      op_name(*sret, *src0, (int32_t)shift, p, MODE_ZEROING);                  \
    }                                                                          \
  }

#define DECLARE_INDIRECT_STORE(dim, dtype, itype)                               \
  __aiv__ __attribute__((always_inline)) void                                                         \
  _mlir_ciface_indirect_store_##dim##d_##dtype##_##itype(                       \
      memref_t<__gm__ dtype, 1> *dst, memref_t<__ubuf__ itype, dim> *idx,      \
      memref_t<__ubuf__ dtype, dim> *src, memref_t<__ubuf__ bool, dim> *mask   \
     )

#define REGISTER_INDIRECT_STORE(dim, dtype, itype)                               \
  DECLARE_INDIRECT_STORE(dim, dtype, itype) {                                   \
    indirect_store<dim, dtype, itype>(dst, idx, src, mask);              \
  }

#define DECLARE_INDIRECT_STORE_NO_MASK(dim, dtype, itype)                        \
  __aiv__ __attribute__((always_inline)) void                                                         \
  _mlir_ciface_indirect_store_no_mask_##dim##d_##dtype##_##itype(               \
      memref_t<__gm__ dtype, 1> *dst,                                          \
      memref_t<__ubuf__ itype, dim> *idx,                                      \
      memref_t<__ubuf__ dtype, dim> *src)

#define REGISTER_INDIRECT_STORE_NO_MASK(dim, dtype, itype)                        \
  DECLARE_INDIRECT_STORE_NO_MASK(dim, dtype, itype) {                            \
    indirect_store_no_mask<dim, dtype, itype>(dst, idx, src);                    \
  }

#define REGISTE_INDIRECT_ATOMIC_CAS(dtype, itype)                              \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_indirect_atomic_cas_##dtype##_##itype(                      \
          memref_t<__gm__ dtype, 1> *src,                                      \
          memref_t<__ubuf__ itype, 1> *offsets,                                \
          memref_t<__ubuf__ dtype, 1> *comp,                                   \
          memref_t<__ubuf__ dtype, 1> *value,                                  \
          memref_t<__ubuf__ dtype, 1> *out) {                                  \
    indirect_atomic_cas<dtype, itype>(src, offsets, comp, value, out);         \
  }

#define DECLARE_INDIRECT_ATOMIC(op, dtype, itype)                              \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_indirect_atomic_##op##_##dtype##_##itype(                   \
          memref_t<__gm__ dtype, 1> *src,                                      \
          memref_t<__ubuf__ itype, 1> *offsets,                                \
          memref_t<__ubuf__ dtype, 1> *value,                                  \
          memref_t<__ubuf__ int8_t, 1> *mask,                                  \
          memref_t<__ubuf__ dtype, 1> *out)

#define REGISTE_INDIRECT_ATOMIC(op, dtype, itype)                              \
  DECLARE_INDIRECT_ATOMIC(op, dtype, itype) {                                  \
    indirect_atomic_##op<dtype, itype>(src, offsets, value, mask, out);        \
  }

#define DECLARE_INDIRECT_ATOMIC_NO_MASK(op, dtype, itype)                      \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_indirect_atomic_no_mask_##op##_##dtype##_##itype(           \
          memref_t<__gm__ dtype, 1> *src,                                      \
          memref_t<__ubuf__ itype, 1> *offsets,                                \
          memref_t<__ubuf__ dtype, 1> *value,                                  \
          memref_t<__ubuf__ dtype, 1> *out)

#define REGISTE_INDIRECT_ATOMIC_NO_MASK(op, dtype, itype)                      \
  DECLARE_INDIRECT_ATOMIC_NO_MASK(op, dtype, itype) {                          \
    indirect_atomic_no_mask_##op<dtype, itype>(src, offsets, value, out);      \
  }

#define DECLARE_SCATTER_UB_TO_OUT(DIM, dataty, idxty, ...)                     \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_scatter_ub_to_out_##DIM##d_##dataty##_##idxty(              \
          memref_t<__gm__ dataty, 1> *dst,                                     \
          memref_t<__ubuf__ dataty, DIM> *src,                                 \
          memref_t<__ubuf__ idxty, DIM> *index, const int64_t indexBoundary,   \
          const int32_t dim, __VA_ARGS__)

#define DECLARE_SCATTER_UB_TO_OUT_1D(dataty, idxty)                            \
  DECLARE_SCATTER_UB_TO_OUT(1, dataty, idxty, const int64_t dstStride0,        \
                            const int32_t endOffset0,                          \
                            const int32_t startOffset0)

#define REGISTER_SCATTER_UB_TO_OUT_1D(dataty, idxty)                           \
  DECLARE_SCATTER_UB_TO_OUT_1D(dataty, idxty) {                                \
    int64_t dstStrides[1] = {dstStride0};                                      \
    int32_t endOffsets[1] = {endOffset0};                                      \
    int32_t startOffsets[1] = {startOffset0};                                  \
    SimtScatteUBToOut<1, dataty, idxty>(dst, src, index, indexBoundary, dim,   \
                                        dstStrides, endOffsets, startOffsets); \
  }

#define DECLARE_SCATTER_UB_TO_OUT_2D(dataty, idxty)                            \
  DECLARE_SCATTER_UB_TO_OUT(                                                   \
      2, dataty, idxty, const int64_t dstStride0, const int64_t dstStride1,    \
      const int32_t endOffset0, const int32_t endOffset1,                      \
      const int32_t startOffset0, const int32_t startOffset1)

#define REGISTER_SCATTER_UB_TO_OUT_2D(dataty, idxty)                           \
  DECLARE_SCATTER_UB_TO_OUT_2D(dataty, idxty) {                                \
    int64_t dstStrides[2] = {dstStride0, dstStride1};                          \
    int32_t endOffsets[2] = {endOffset0, endOffset1};                          \
    int32_t startOffsets[2] = {startOffset0, startOffset1};                    \
    SimtScatteUBToOut<2, dataty, idxty>(dst, src, index, indexBoundary, dim,   \
                                        dstStrides, endOffsets, startOffsets); \
  }

#define DECLARE_SCATTER_UB_TO_OUT_3D(dataty, idxty)                            \
  DECLARE_SCATTER_UB_TO_OUT(                                                   \
      3, dataty, idxty, const int64_t dstStride0, const int64_t dstStride1,    \
      const int64_t dstStride2, const int32_t endOffset0,                      \
      const int32_t endOffset1, const int32_t endOffset2,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2)

#define REGISTER_SCATTER_UB_TO_OUT_3D(dataty, idxty)                           \
  DECLARE_SCATTER_UB_TO_OUT_3D(dataty, idxty) {                                \
    int64_t dstStrides[3] = {dstStride0, dstStride1, dstStride2};              \
    int32_t endOffsets[3] = {endOffset0, endOffset1, endOffset2};              \
    int32_t startOffsets[3] = {startOffset0, startOffset1, startOffset2};      \
    SimtScatteUBToOut<3, dataty, idxty>(dst, src, index, indexBoundary, dim,   \
                                        dstStrides, endOffsets, startOffsets); \
  }
#define DECLARE_SCATTER_UB_TO_OUT_4D(dataty, idxty)                            \
  DECLARE_SCATTER_UB_TO_OUT(                                                   \
      4, dataty, idxty, const int64_t dstStride0, const int64_t dstStride1,    \
      const int64_t dstStride2, const int64_t dstStride3,                      \
      const int32_t endOffset0, const int32_t endOffset1,                      \
      const int32_t endOffset2, const int32_t endOffset3,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, const int32_t startOffset3)

#define REGISTER_SCATTER_UB_TO_OUT_4D(dataty, idxty)                           \
  DECLARE_SCATTER_UB_TO_OUT_4D(dataty, idxty) {                                \
    int64_t dstStrides[4] = {dstStride0, dstStride1, dstStride2, dstStride3};  \
    int32_t endOffsets[4] = {endOffset0, endOffset1, endOffset2, endOffset3};  \
    int32_t startOffsets[4] = {startOffset0, startOffset1, startOffset2,       \
                               startOffset3};                                  \
    SimtScatteUBToOut<4, dataty, idxty>(dst, src, index, indexBoundary, dim,   \
                                        dstStrides, endOffsets, startOffsets); \
  }

#define DECLARE_SCATTER_UB_TO_OUT_5D(dataty, idxty)                            \
  DECLARE_SCATTER_UB_TO_OUT(                                                   \
      5, dataty, idxty, const int64_t dstStride0, const int64_t dstStride1,    \
      const int64_t dstStride2, const int64_t dstStride3,                      \
      const int64_t dstStride4, const int32_t endOffset0,                      \
      const int32_t endOffset1, const int32_t endOffset2,                      \
      const int32_t endOffset3, const int32_t endOffset4,                      \
      const int32_t startOffset0, const int32_t startOffset1,                  \
      const int32_t startOffset2, const int32_t startOffset3,                  \
      const int32_t startOffset4)

#define REGISTER_SCATTER_UB_TO_OUT_5D(dataty, idxty)                           \
  DECLARE_SCATTER_UB_TO_OUT_5D(dataty, idxty) {                                \
    int64_t dstStrides[5] = {dstStride0, dstStride1, dstStride2, dstStride3,   \
                             dstStride4};                                      \
    int32_t endOffsets[5] = {endOffset0, endOffset1, endOffset2, endOffset3,   \
                             endOffset4};                                      \
    int32_t startOffsets[5] = {startOffset0, startOffset1, startOffset2,       \
                               startOffset3, startOffset4};                    \
    SimtScatteUBToOut<5, dataty, idxty>(dst, src, index, indexBoundary, dim,   \
                                        dstStrides, endOffsets, startOffsets); \
  }

#define DECLARE_INT_VDIV(op_name, dim, dtype)                                  \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##dim##d##_##dtype(                             \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *src1,                                 \
          memref_t<__ubuf__ dtype, dim> *dst_value)

#define REGISTE_INT_VDIV(op_name, dim, dtype)                                  \
  DECLARE_INT_VDIV(op_name, dim, dtype) {                                      \
    vdiv_int<dtype, dim>(src0, src1, dst_value);                               \
  }

#define DECLARE_INT_VDIV_SCALAR(op_name, dim, dtype)                           \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##dim##d##_##dtype(                             \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          dtype src1,                                                          \
          memref_t<__ubuf__ dtype, dim> *dst_value)

#define REGISTE_INT_VDIV_SCALAR(op_name, dim, dtype)                                  \
  DECLARE_INT_VDIV_SCALAR(op_name, dim, dtype) {                                      \
    vdiv_int_scalar<dtype, dim>(src0, src1, dst_value);                               \
  }

#define DECLARE_FLOAT_VDIV_HP(op_name)                                         \
  __aiv__ __attribute__((always_inline))                                       \
      vector_f32 _mlir_ciface_##op_name##_float(                               \
          vector_f32 src0, vector_f32 src1, ave_preg mask)

#define RETURN_REGISTE_FLOAT_VDIV_HP(op_name)                                  \
  DECLARE_FLOAT_VDIV_HP(op_name) { return vdiv_float_hp(src0, src1, mask); }

#define DECLARE_VMODUI(op_name, dtype)                                         \
  __aiv__ __attribute__((always_inline)) VectorReg<dtype>                      \
      _mlir_ciface_##op_name##_##dtype(VectorReg<dtype> src0,                  \
                                       VectorReg<dtype> src1, ave_preg mask)

#define RETURN_REGISTE_VMODUI(op_name, dtype_signed, dtype_unsigned)           \
  DECLARE_VMODUI(op_name, dtype_signed) {                                      \
    return vmodui<dtype_signed, dtype_unsigned>(src0, src1, mask);             \
  }

#define DECLARE_VMOD(op_name, dtype)                                           \
  __aiv__ __attribute__((always_inline)) VectorReg<dtype>                      \
      _mlir_ciface_##op_name##_##dtype(VectorReg<dtype> src0,                  \
                                       VectorReg<dtype> src1, ave_preg mask)

#define RETURN_REGISTE_VMOD(op_name, dtype)                                    \
  DECLARE_VMOD(op_name, dtype) { return vmodsi<dtype>(src0, src1, mask); }

#define REGISTE_VMOD_VV(op_name)                                               \
  DECLARE_BINARY_VV(op_name) { vector_##op_name(sret, src0, src1, preg); }

extern "C" {

DECLARE_BINARY_VV(vadd);
DECLARE_BINARY_VV(vsub);
DECLARE_BINARY_VV(vmul);
DECLARE_BINARY_VV(vdiv);
DECLARE_BINARY_VV(vmod);
DECLARE_BINARY_VV(vmax);
DECLARE_BINARY_VV(vmin);
DECLARE_BINARY_VV(vand);
DECLARE_BINARY_VV(vor);
DECLARE_BINARY_VV(vxor);

DECLARE_BINARY_VV(vmod);
DECLARE_BINARY_VV(vmodui);

DECLARE_BINARY_VS(vadds);
DECLARE_BINARY_VS(vmuls);
DECLARE_BINARY_VS(vmaxs);
DECLARE_BINARY_VS(vmins);

DECLARE_UNARY_V(vabs);
DECLARE_UNARY_V(vneg);
DECLARE_UNARY_V(vnot);

DECLARE_VSEL(vsel);

DECLARE_CMP_VV(vcmp_eq);
DECLARE_CMP_VV(vcmp_ne);
DECLARE_CMP_VV(vcmp_gt);
DECLARE_CMP_VV(vcmp_ge);
DECLARE_CMP_VV(vcmp_lt);
DECLARE_CMP_VV(vcmp_le);

DECLARE_CMP_VV_U64(le);
DECLARE_CMP_VV_U64(lt);
DECLARE_CMP_VV_U64(ge);
DECLARE_CMP_VV_U64(gt);

DECLARE_DIV_VV_U64(vdiv);
DECLARE_DIV_VV(vdiv, int16_t);
DECLARE_DIV_VV(vdiv, uint16_t);
DECLARE_DIV_VV(vdiv, int32_t);
DECLARE_DIV_VV(vdiv, uint32_t);

DECLARE_CMP_VS(vcmps_eq);
DECLARE_CMP_VS(vcmps_ne);
DECLARE_CMP_VS(vcmps_gt);
DECLARE_CMP_VS(vcmps_ge);
DECLARE_CMP_VS(vcmps_lt);
DECLARE_CMP_VS(vcmps_le);

DECLARE_VEC_LOAD_BRC(vload_BRC_B64, 0);
DECLARE_VEC_LOAD_BRC(vload_BRC_B64, 1);
DECLARE_VEC_LOAD_BRC(vload_BRC_B64, 2);
DECLARE_VEC_LOAD_BRC(vload_BRC_B64, 3);
DECLARE_VEC_LOAD_BRC(vload_BRC_B64, 4);
DECLARE_VEC_LOAD_BRC(vload_BRC_B64, 5);
DECLARE_VEC_LOAD_BRC(vload_BRC_B64, 6);
DECLARE_VEC_LOAD_BRC(vload_BRC_B64, 7);
DECLARE_VEC_LOAD_BRC(vload_BRC_B64, 8);
DECLARE_VEC_LOAD_NORM(vload_NORM, 0);
DECLARE_VEC_LOAD_NORM(vload_NORM, 1);
DECLARE_VEC_LOAD_NORM(vload_NORM, 2);
DECLARE_VEC_LOAD_NORM(vload_NORM, 3);
DECLARE_VEC_LOAD_NORM(vload_NORM, 4);
DECLARE_VEC_LOAD_NORM(vload_NORM, 5);
DECLARE_VEC_LOAD_NORM(vload_NORM, 6);
DECLARE_VEC_LOAD_NORM(vload_NORM, 7);
DECLARE_VEC_LOAD_NORM(vload_NORM, 8);
DECLARE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 0);
DECLARE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 1);
DECLARE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 2);
DECLARE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 3);
DECLARE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 4);
DECLARE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 5);
DECLARE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 6);
DECLARE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 7);
DECLARE_VEC_LOAD_NORM_UNALIGN(vload_NORM, 8);
DECLARE_VEC_STORE_OPT(masked_store_ONEPT_B64, 0);
DECLARE_VEC_STORE_OPT(masked_store_ONEPT_B64, 1);
DECLARE_VEC_STORE_OPT(masked_store_ONEPT_B64, 2);
DECLARE_VEC_STORE_OPT(masked_store_ONEPT_B64, 3);
DECLARE_VEC_STORE_OPT(masked_store_ONEPT_B64, 4);
DECLARE_VEC_STORE_OPT(masked_store_ONEPT_B64, 5);
DECLARE_VEC_STORE_OPT(masked_store_ONEPT_B64, 6);
DECLARE_VEC_STORE_OPT(masked_store_ONEPT_B64, 7);
DECLARE_VEC_STORE_OPT(masked_store_ONEPT_B64, 8);
DECLARE_VEC_STORE_NORM(masked_store_NORM_B64, 0);
DECLARE_VEC_STORE_NORM(masked_store_NORM_B64, 1);
DECLARE_VEC_STORE_NORM(masked_store_NORM_B64, 2);
DECLARE_VEC_STORE_NORM(masked_store_NORM_B64, 3);
DECLARE_VEC_STORE_NORM(masked_store_NORM_B64, 4);
DECLARE_VEC_STORE_NORM(masked_store_NORM_B64, 5);
DECLARE_VEC_STORE_NORM(masked_store_NORM_B64, 6);
DECLARE_VEC_STORE_NORM(masked_store_NORM_B64, 7);
DECLARE_VEC_STORE_NORM(masked_store_NORM_B64, 8);
DECLARE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 0);
DECLARE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 1);
DECLARE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 2);
DECLARE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 3);
DECLARE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 4);
DECLARE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 5);
DECLARE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 6);
DECLARE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 7);
DECLARE_VEC_STORE_NORM_UNALIGN(masked_store_NORM_B64, 8);

DECLARE_VEC_GATHER(vgather, 0);
DECLARE_VEC_GATHER(vgather, 1);
DECLARE_VEC_GATHER(vgather, 2);
DECLARE_VEC_GATHER(vgather, 3);
DECLARE_VEC_GATHER(vgather, 4);
DECLARE_VEC_GATHER(vgather, 5);
DECLARE_VEC_GATHER(vgather, 6);
DECLARE_VEC_GATHER(vgather, 7);
DECLARE_VEC_GATHER(vgather, 8);

DECLARE_CAST_I64_TO_F32();
DECLARE_CAST_U64_TO_F32();
DECLARE_CAST_F32_TO_I64();
DECLARE_CAST_F32_TO_U64();
DECLARE_CAST_I64_TO_I32();
DECLARE_CAST_I32_TO_I64();
DECLARE_CAST_U32_TO_I64();

DECLARE_CAST_I64_TO_I32_SAT();
DECLARE_CAST_I64_TO_U32_SAT();
DECLARE_CAST_U64_TO_I32_SAT();
DECLARE_CAST_U64_TO_U32_SAT();

DECLARE_VBR(vbr);
DECLARE_VDUPS(vdups);
DECLARE_VDUP(vdup);

DECLARE_REDUCTION_I64(vcadd);
DECLARE_REDUCTION_I64(vcmax);
DECLARE_REDUCTION_I64(vcmin);

DECLARE_REDUCTION_U64(vcmax);
DECLARE_REDUCTION_U64(vcmin);

DECLARE_VCI(vci);
DECLARE_VSLIDE(vslide);

DECLARE_VINTLV(vintlv);
DECLARE_VDINTLV(vdintlv);

DECLARE_BINARY_DMA_UNALIGN_SIZE(bool);
DECLARE_BINARY_DMA_UNALIGN_SIZE(float);
DECLARE_BINARY_DMA_UNALIGN_SIZE(half);
DECLARE_BINARY_DMA_UNALIGN_SIZE(bfloat16_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(float8_e4m3_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(float8_e5m2_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(int8_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(uint8_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(int16_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(uint16_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(int32_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(uint32_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(int64_t);
DECLARE_BINARY_DMA_UNALIGN_SIZE(uint64_t);


DECLARE_BINARY_DMA_UNALIGN_OFFSET(bool);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(float);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(half);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(bfloat16_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(float8_e4m3_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(float8_e5m2_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(int8_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(uint8_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(int16_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(uint16_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(int32_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(uint32_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(int64_t);
DECLARE_BINARY_DMA_UNALIGN_OFFSET(uint64_t);

DECLARE_EMBEDDINGGATHER_1D(float, int64_t);
DECLARE_EMBEDDINGGATHER_2D(float, int64_t);
DECLARE_EMBEDDINGGATHER_1D(float, int32_t);
DECLARE_EMBEDDINGGATHER_2D(float, int32_t);

DECLARE_INDEX_PUT_2D(float, int32_t);
DECLARE_INDEX_PUT_2D(half, int32_t);
DECLARE_INDEX_PUT_2D(bfloat16_t, int32_t);
DECLARE_INDEX_PUT_2D(float, int64_t);
DECLARE_INDEX_PUT_2D(half, int64_t);
DECLARE_INDEX_PUT_2D(bfloat16_t, int64_t);

DECLARE_INDEX_PUT_3D(float, int32_t);
DECLARE_INDEX_PUT_3D(half, int32_t);
DECLARE_INDEX_PUT_3D(bfloat16_t, int32_t);
DECLARE_INDEX_PUT_3D(float, int64_t);
DECLARE_INDEX_PUT_3D(half, int64_t);
DECLARE_INDEX_PUT_3D(bfloat16_t, int64_t);

DECLARE_INDEX_PUT_4D(float, int32_t);
DECLARE_INDEX_PUT_4D(half, int32_t);
DECLARE_INDEX_PUT_4D(bfloat16_t, int32_t);
DECLARE_INDEX_PUT_4D(float, int64_t);
DECLARE_INDEX_PUT_4D(half, int64_t);
DECLARE_INDEX_PUT_4D(bfloat16_t, int64_t);

DECLARE_INDEX_PUT_5D(float, int32_t);
DECLARE_INDEX_PUT_5D(half, int32_t);
DECLARE_INDEX_PUT_5D(bfloat16_t, int32_t);
DECLARE_INDEX_PUT_5D(float, int64_t);
DECLARE_INDEX_PUT_5D(half, int64_t);
DECLARE_INDEX_PUT_5D(bfloat16_t, int64_t);

DECLARE_INDEX_SELECT_2D_1D(float, int32_t);
DECLARE_INDEX_SELECT_2D_1D(half, int32_t);
DECLARE_INDEX_SELECT_2D_1D(bfloat16_t, int32_t);
DECLARE_INDEX_SELECT_2D_1D(int8_t, int32_t);
DECLARE_INDEX_SELECT_2D_1D(int16_t, int32_t);
DECLARE_INDEX_SELECT_2D_1D(int32_t, int32_t);
DECLARE_INDEX_SELECT_2D_1D(int64_t, int32_t);
DECLARE_INDEX_SELECT_2D_1D(uint8_t, int32_t);
DECLARE_INDEX_SELECT_2D_1D(uint16_t, int32_t);
DECLARE_INDEX_SELECT_2D_1D(uint32_t, int32_t);
DECLARE_INDEX_SELECT_2D_1D(uint64_t, int32_t);
DECLARE_INDEX_SELECT_2D_1D(float, int64_t);
DECLARE_INDEX_SELECT_2D_1D(half, int64_t);
DECLARE_INDEX_SELECT_2D_1D(bfloat16_t, int64_t);
DECLARE_INDEX_SELECT_2D_1D(int8_t, int64_t);
DECLARE_INDEX_SELECT_2D_1D(int16_t, int64_t);
DECLARE_INDEX_SELECT_2D_1D(int32_t, int64_t);
DECLARE_INDEX_SELECT_2D_1D(int64_t, int64_t);
DECLARE_INDEX_SELECT_2D_1D(uint8_t, int64_t);
DECLARE_INDEX_SELECT_2D_1D(uint16_t, int64_t);
DECLARE_INDEX_SELECT_2D_1D(uint32_t, int64_t);
DECLARE_INDEX_SELECT_2D_1D(uint64_t, int64_t);

DECLARE_INDEX_SELECT_2D_2D(float, int32_t);
DECLARE_INDEX_SELECT_2D_2D(half, int32_t);
DECLARE_INDEX_SELECT_2D_2D(bfloat16_t, int32_t);
DECLARE_INDEX_SELECT_2D_2D(int8_t, int32_t);
DECLARE_INDEX_SELECT_2D_2D(int16_t, int32_t);
DECLARE_INDEX_SELECT_2D_2D(int32_t, int32_t);
DECLARE_INDEX_SELECT_2D_2D(int64_t, int32_t);
DECLARE_INDEX_SELECT_2D_2D(uint8_t, int32_t);
DECLARE_INDEX_SELECT_2D_2D(uint16_t, int32_t);
DECLARE_INDEX_SELECT_2D_2D(uint32_t, int32_t);
DECLARE_INDEX_SELECT_2D_2D(uint64_t, int32_t);
DECLARE_INDEX_SELECT_2D_2D(float, int64_t);
DECLARE_INDEX_SELECT_2D_2D(half, int64_t);
DECLARE_INDEX_SELECT_2D_2D(bfloat16_t, int64_t);
DECLARE_INDEX_SELECT_2D_2D(int8_t, int64_t);
DECLARE_INDEX_SELECT_2D_2D(int16_t, int64_t);
DECLARE_INDEX_SELECT_2D_2D(int32_t, int64_t);
DECLARE_INDEX_SELECT_2D_2D(int64_t, int64_t);
DECLARE_INDEX_SELECT_2D_2D(uint8_t, int64_t);
DECLARE_INDEX_SELECT_2D_2D(uint16_t, int64_t);
DECLARE_INDEX_SELECT_2D_2D(uint32_t, int64_t);
DECLARE_INDEX_SELECT_2D_2D(uint64_t, int64_t);

DECLARE_INDEX_SELECT_3D_1D(float, int32_t);
DECLARE_INDEX_SELECT_3D_1D(half, int32_t);
DECLARE_INDEX_SELECT_3D_1D(bfloat16_t, int32_t);
DECLARE_INDEX_SELECT_3D_1D(int8_t, int32_t);
DECLARE_INDEX_SELECT_3D_1D(int16_t, int32_t);
DECLARE_INDEX_SELECT_3D_1D(int32_t, int32_t);
DECLARE_INDEX_SELECT_3D_1D(int64_t, int32_t);
DECLARE_INDEX_SELECT_3D_1D(uint8_t, int32_t);
DECLARE_INDEX_SELECT_3D_1D(uint16_t, int32_t);
DECLARE_INDEX_SELECT_3D_1D(uint32_t, int32_t);
DECLARE_INDEX_SELECT_3D_1D(uint64_t, int32_t);
DECLARE_INDEX_SELECT_3D_1D(float, int64_t);
DECLARE_INDEX_SELECT_3D_1D(half, int64_t);
DECLARE_INDEX_SELECT_3D_1D(bfloat16_t, int64_t);
DECLARE_INDEX_SELECT_3D_1D(int8_t, int64_t);
DECLARE_INDEX_SELECT_3D_1D(int16_t, int64_t);
DECLARE_INDEX_SELECT_3D_1D(int32_t, int64_t);
DECLARE_INDEX_SELECT_3D_1D(int64_t, int64_t);
DECLARE_INDEX_SELECT_3D_1D(uint8_t, int64_t);
DECLARE_INDEX_SELECT_3D_1D(uint16_t, int64_t);
DECLARE_INDEX_SELECT_3D_1D(uint32_t, int64_t);
DECLARE_INDEX_SELECT_3D_1D(uint64_t, int64_t);

DECLARE_INDEX_SELECT_3D_2D(float, int32_t);
DECLARE_INDEX_SELECT_3D_2D(half, int32_t);
DECLARE_INDEX_SELECT_3D_2D(bfloat16_t, int32_t);
DECLARE_INDEX_SELECT_3D_2D(int8_t, int32_t);
DECLARE_INDEX_SELECT_3D_2D(int16_t, int32_t);
DECLARE_INDEX_SELECT_3D_2D(int32_t, int32_t);
DECLARE_INDEX_SELECT_3D_2D(int64_t, int32_t);
DECLARE_INDEX_SELECT_3D_2D(uint8_t, int32_t);
DECLARE_INDEX_SELECT_3D_2D(uint16_t, int32_t);
DECLARE_INDEX_SELECT_3D_2D(uint32_t, int32_t);
DECLARE_INDEX_SELECT_3D_2D(uint64_t, int32_t);
DECLARE_INDEX_SELECT_3D_2D(float, int64_t);
DECLARE_INDEX_SELECT_3D_2D(half, int64_t);
DECLARE_INDEX_SELECT_3D_2D(bfloat16_t, int64_t);
DECLARE_INDEX_SELECT_3D_2D(int8_t, int64_t);
DECLARE_INDEX_SELECT_3D_2D(int16_t, int64_t);
DECLARE_INDEX_SELECT_3D_2D(int32_t, int64_t);
DECLARE_INDEX_SELECT_3D_2D(int64_t, int64_t);
DECLARE_INDEX_SELECT_3D_2D(uint8_t, int64_t);
DECLARE_INDEX_SELECT_3D_2D(uint16_t, int64_t);
DECLARE_INDEX_SELECT_3D_2D(uint32_t, int64_t);
DECLARE_INDEX_SELECT_3D_2D(uint64_t, int64_t);

DECLARE_INDEX_SELECT_4D_1D(float, int32_t);
DECLARE_INDEX_SELECT_4D_1D(half, int32_t);
DECLARE_INDEX_SELECT_4D_1D(bfloat16_t, int32_t);
DECLARE_INDEX_SELECT_4D_1D(int8_t, int32_t);
DECLARE_INDEX_SELECT_4D_1D(int16_t, int32_t);
DECLARE_INDEX_SELECT_4D_1D(int32_t, int32_t);
DECLARE_INDEX_SELECT_4D_1D(int64_t, int32_t);
DECLARE_INDEX_SELECT_4D_1D(uint8_t, int32_t);
DECLARE_INDEX_SELECT_4D_1D(uint16_t, int32_t);
DECLARE_INDEX_SELECT_4D_1D(uint32_t, int32_t);
DECLARE_INDEX_SELECT_4D_1D(uint64_t, int32_t);
DECLARE_INDEX_SELECT_4D_1D(float, int64_t);
DECLARE_INDEX_SELECT_4D_1D(half, int64_t);
DECLARE_INDEX_SELECT_4D_1D(bfloat16_t, int64_t);
DECLARE_INDEX_SELECT_4D_1D(int8_t, int64_t);
DECLARE_INDEX_SELECT_4D_1D(int16_t, int64_t);
DECLARE_INDEX_SELECT_4D_1D(int32_t, int64_t);
DECLARE_INDEX_SELECT_4D_1D(int64_t, int64_t);
DECLARE_INDEX_SELECT_4D_1D(uint8_t, int64_t);
DECLARE_INDEX_SELECT_4D_1D(uint16_t, int64_t);
DECLARE_INDEX_SELECT_4D_1D(uint32_t, int64_t);
DECLARE_INDEX_SELECT_4D_1D(uint64_t, int64_t);

DECLARE_INDEX_SELECT_4D_2D(float, int32_t);
DECLARE_INDEX_SELECT_4D_2D(half, int32_t);
DECLARE_INDEX_SELECT_4D_2D(bfloat16_t, int32_t);
DECLARE_INDEX_SELECT_4D_2D(int8_t, int32_t);
DECLARE_INDEX_SELECT_4D_2D(int16_t, int32_t);
DECLARE_INDEX_SELECT_4D_2D(int32_t, int32_t);
DECLARE_INDEX_SELECT_4D_2D(int64_t, int32_t);
DECLARE_INDEX_SELECT_4D_2D(uint8_t, int32_t);
DECLARE_INDEX_SELECT_4D_2D(uint16_t, int32_t);
DECLARE_INDEX_SELECT_4D_2D(uint32_t, int32_t);
DECLARE_INDEX_SELECT_4D_2D(uint64_t, int32_t);
DECLARE_INDEX_SELECT_4D_2D(float, int64_t);
DECLARE_INDEX_SELECT_4D_2D(half, int64_t);
DECLARE_INDEX_SELECT_4D_2D(bfloat16_t, int64_t);
DECLARE_INDEX_SELECT_4D_2D(int8_t, int64_t);
DECLARE_INDEX_SELECT_4D_2D(int16_t, int64_t);
DECLARE_INDEX_SELECT_4D_2D(int32_t, int64_t);
DECLARE_INDEX_SELECT_4D_2D(int64_t, int64_t);
DECLARE_INDEX_SELECT_4D_2D(uint8_t, int64_t);
DECLARE_INDEX_SELECT_4D_2D(uint16_t, int64_t);
DECLARE_INDEX_SELECT_4D_2D(uint32_t, int64_t);
DECLARE_INDEX_SELECT_4D_2D(uint64_t, int64_t);

DECLARE_INDEX_SELECT_5D_1D(float, int32_t);
DECLARE_INDEX_SELECT_5D_1D(half, int32_t);
DECLARE_INDEX_SELECT_5D_1D(bfloat16_t, int32_t);
DECLARE_INDEX_SELECT_5D_1D(int8_t, int32_t);
DECLARE_INDEX_SELECT_5D_1D(int16_t, int32_t);
DECLARE_INDEX_SELECT_5D_1D(int32_t, int32_t);
DECLARE_INDEX_SELECT_5D_1D(int64_t, int32_t);
DECLARE_INDEX_SELECT_5D_1D(uint8_t, int32_t);
DECLARE_INDEX_SELECT_5D_1D(uint16_t, int32_t);
DECLARE_INDEX_SELECT_5D_1D(uint32_t, int32_t);
DECLARE_INDEX_SELECT_5D_1D(uint64_t, int32_t);
DECLARE_INDEX_SELECT_5D_1D(float, int64_t);
DECLARE_INDEX_SELECT_5D_1D(half, int64_t);
DECLARE_INDEX_SELECT_5D_1D(bfloat16_t, int64_t);
DECLARE_INDEX_SELECT_5D_1D(int8_t, int64_t);
DECLARE_INDEX_SELECT_5D_1D(int16_t, int64_t);
DECLARE_INDEX_SELECT_5D_1D(int32_t, int64_t);
DECLARE_INDEX_SELECT_5D_1D(int64_t, int64_t);
DECLARE_INDEX_SELECT_5D_1D(uint8_t, int64_t);
DECLARE_INDEX_SELECT_5D_1D(uint16_t, int64_t);
DECLARE_INDEX_SELECT_5D_1D(uint32_t, int64_t);
DECLARE_INDEX_SELECT_5D_1D(uint64_t, int64_t);

DECLARE_INDEX_SELECT_5D_2D(float, int32_t);
DECLARE_INDEX_SELECT_5D_2D(half, int32_t);
DECLARE_INDEX_SELECT_5D_2D(bfloat16_t, int32_t);
DECLARE_INDEX_SELECT_5D_2D(int8_t, int32_t);
DECLARE_INDEX_SELECT_5D_2D(int16_t, int32_t);
DECLARE_INDEX_SELECT_5D_2D(int32_t, int32_t);
DECLARE_INDEX_SELECT_5D_2D(int64_t, int32_t);
DECLARE_INDEX_SELECT_5D_2D(uint8_t, int32_t);
DECLARE_INDEX_SELECT_5D_2D(uint16_t, int32_t);
DECLARE_INDEX_SELECT_5D_2D(uint32_t, int32_t);
DECLARE_INDEX_SELECT_5D_2D(uint64_t, int32_t);
DECLARE_INDEX_SELECT_5D_2D(float, int64_t);
DECLARE_INDEX_SELECT_5D_2D(half, int64_t);
DECLARE_INDEX_SELECT_5D_2D(bfloat16_t, int64_t);
DECLARE_INDEX_SELECT_5D_2D(int8_t, int64_t);
DECLARE_INDEX_SELECT_5D_2D(int16_t, int64_t);
DECLARE_INDEX_SELECT_5D_2D(int32_t, int64_t);
DECLARE_INDEX_SELECT_5D_2D(int64_t, int64_t);
DECLARE_INDEX_SELECT_5D_2D(uint8_t, int64_t);
DECLARE_INDEX_SELECT_5D_2D(uint16_t, int64_t);
DECLARE_INDEX_SELECT_5D_2D(uint32_t, int64_t);
DECLARE_INDEX_SELECT_5D_2D(uint64_t, int64_t);

// indirectLoad 1D
DECLARE_INDIRECT_LOAD(1, float, int32_t);
DECLARE_INDIRECT_LOAD(1, half, int32_t);
DECLARE_INDIRECT_LOAD(1, bfloat16_t, int32_t);
DECLARE_INDIRECT_LOAD(1, int8_t, int32_t);
DECLARE_INDIRECT_LOAD(1, int16_t, int32_t);
DECLARE_INDIRECT_LOAD(1, int32_t, int32_t);
DECLARE_INDIRECT_LOAD(1, int64_t, int32_t);
DECLARE_INDIRECT_LOAD(1, uint8_t, int32_t);
DECLARE_INDIRECT_LOAD(1, uint16_t, int32_t);
DECLARE_INDIRECT_LOAD(1, uint32_t, int32_t);
DECLARE_INDIRECT_LOAD(1, uint64_t, int32_t);
DECLARE_INDIRECT_LOAD(1, bool, int32_t);
DECLARE_INDIRECT_LOAD(1, hifloat8_t, int32_t);
DECLARE_INDIRECT_LOAD(1, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_LOAD(1, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_LOAD(1, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_LOAD(1, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_LOAD(1, float4_e1m2_t, int32_t);

DECLARE_INDIRECT_LOAD(1, float, int64_t);
DECLARE_INDIRECT_LOAD(1, half, int64_t);
DECLARE_INDIRECT_LOAD(1, bfloat16_t, int64_t);
DECLARE_INDIRECT_LOAD(1, int8_t, int64_t);
DECLARE_INDIRECT_LOAD(1, int16_t, int64_t);
DECLARE_INDIRECT_LOAD(1, int32_t, int64_t);
DECLARE_INDIRECT_LOAD(1, int64_t, int64_t);
DECLARE_INDIRECT_LOAD(1, uint8_t, int64_t);
DECLARE_INDIRECT_LOAD(1, uint16_t, int64_t);
DECLARE_INDIRECT_LOAD(1, uint32_t, int64_t);
DECLARE_INDIRECT_LOAD(1, uint64_t, int64_t);
DECLARE_INDIRECT_LOAD(1, bool, int64_t);
DECLARE_INDIRECT_LOAD(1, hifloat8_t, int64_t);
DECLARE_INDIRECT_LOAD(1, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_LOAD(1, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_LOAD(1, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_LOAD(1, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_LOAD(1, float4_e1m2_t, int64_t);

DECLARE_INDIRECT_LOAD(2, float, int32_t);
DECLARE_INDIRECT_LOAD(2, half, int32_t);
DECLARE_INDIRECT_LOAD(2, bfloat16_t, int32_t);
DECLARE_INDIRECT_LOAD(2, int8_t, int32_t);
DECLARE_INDIRECT_LOAD(2, int16_t, int32_t);
DECLARE_INDIRECT_LOAD(2, int32_t, int32_t);
DECLARE_INDIRECT_LOAD(2, int64_t, int32_t);
DECLARE_INDIRECT_LOAD(2, uint8_t, int32_t);
DECLARE_INDIRECT_LOAD(2, uint16_t, int32_t);
DECLARE_INDIRECT_LOAD(2, uint32_t, int32_t);
DECLARE_INDIRECT_LOAD(2, uint64_t, int32_t);
DECLARE_INDIRECT_LOAD(2, bool, int32_t);
DECLARE_INDIRECT_LOAD(2, hifloat8_t, int32_t);
DECLARE_INDIRECT_LOAD(2, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_LOAD(2, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_LOAD(2, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_LOAD(2, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_LOAD(2, float4_e1m2_t, int32_t);

DECLARE_INDIRECT_LOAD(2, float, int64_t);
DECLARE_INDIRECT_LOAD(2, half, int64_t);
DECLARE_INDIRECT_LOAD(2, bfloat16_t, int64_t);
DECLARE_INDIRECT_LOAD(2, int8_t, int64_t);
DECLARE_INDIRECT_LOAD(2, int16_t, int64_t);
DECLARE_INDIRECT_LOAD(2, int32_t, int64_t);
DECLARE_INDIRECT_LOAD(2, int64_t, int64_t);
DECLARE_INDIRECT_LOAD(2, uint8_t, int64_t);
DECLARE_INDIRECT_LOAD(2, uint16_t, int64_t);
DECLARE_INDIRECT_LOAD(2, uint32_t, int64_t);
DECLARE_INDIRECT_LOAD(2, uint64_t, int64_t);
DECLARE_INDIRECT_LOAD(2, bool, int64_t);
DECLARE_INDIRECT_LOAD(2, hifloat8_t, int64_t);
DECLARE_INDIRECT_LOAD(2, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_LOAD(2, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_LOAD(2, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_LOAD(2, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_LOAD(2, float4_e1m2_t, int64_t);

DECLARE_INDIRECT_LOAD(3, float, int32_t);
DECLARE_INDIRECT_LOAD(3, half, int32_t);
DECLARE_INDIRECT_LOAD(3, bfloat16_t, int32_t);
DECLARE_INDIRECT_LOAD(3, int8_t, int32_t);
DECLARE_INDIRECT_LOAD(3, int16_t, int32_t);
DECLARE_INDIRECT_LOAD(3, int32_t, int32_t);
DECLARE_INDIRECT_LOAD(3, int64_t, int32_t);
DECLARE_INDIRECT_LOAD(3, uint8_t, int32_t);
DECLARE_INDIRECT_LOAD(3, uint16_t, int32_t);
DECLARE_INDIRECT_LOAD(3, uint32_t, int32_t);
DECLARE_INDIRECT_LOAD(3, uint64_t, int32_t);
DECLARE_INDIRECT_LOAD(3, bool, int32_t);
DECLARE_INDIRECT_LOAD(3, hifloat8_t, int32_t);
DECLARE_INDIRECT_LOAD(3, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_LOAD(3, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_LOAD(3, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_LOAD(3, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_LOAD(3, float4_e1m2_t, int32_t);

DECLARE_INDIRECT_LOAD(3, float, int64_t);
DECLARE_INDIRECT_LOAD(3, half, int64_t);
DECLARE_INDIRECT_LOAD(3, bfloat16_t, int64_t);
DECLARE_INDIRECT_LOAD(3, int8_t, int64_t);
DECLARE_INDIRECT_LOAD(3, int16_t, int64_t);
DECLARE_INDIRECT_LOAD(3, int32_t, int64_t);
DECLARE_INDIRECT_LOAD(3, int64_t, int64_t);
DECLARE_INDIRECT_LOAD(3, uint8_t, int64_t);
DECLARE_INDIRECT_LOAD(3, uint16_t, int64_t);
DECLARE_INDIRECT_LOAD(3, uint32_t, int64_t);
DECLARE_INDIRECT_LOAD(3, uint64_t, int64_t);
DECLARE_INDIRECT_LOAD(3, bool, int64_t);
DECLARE_INDIRECT_LOAD(3, hifloat8_t, int64_t);
DECLARE_INDIRECT_LOAD(3, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_LOAD(3, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_LOAD(3, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_LOAD(3, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_LOAD(3, float4_e1m2_t, int64_t);

DECLARE_INDIRECT_LOAD(4, float, int32_t);
DECLARE_INDIRECT_LOAD(4, half, int32_t);
DECLARE_INDIRECT_LOAD(4, bfloat16_t, int32_t);
DECLARE_INDIRECT_LOAD(4, int8_t, int32_t);
DECLARE_INDIRECT_LOAD(4, int16_t, int32_t);
DECLARE_INDIRECT_LOAD(4, int32_t, int32_t);
DECLARE_INDIRECT_LOAD(4, int64_t, int32_t);
DECLARE_INDIRECT_LOAD(4, uint8_t, int32_t);
DECLARE_INDIRECT_LOAD(4, uint16_t, int32_t);
DECLARE_INDIRECT_LOAD(4, uint32_t, int32_t);
DECLARE_INDIRECT_LOAD(4, uint64_t, int32_t);
DECLARE_INDIRECT_LOAD(4, bool, int32_t);
DECLARE_INDIRECT_LOAD(4, hifloat8_t, int32_t);
DECLARE_INDIRECT_LOAD(4, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_LOAD(4, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_LOAD(4, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_LOAD(4, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_LOAD(4, float4_e1m2_t, int32_t);

DECLARE_INDIRECT_LOAD(4, float, int64_t);
DECLARE_INDIRECT_LOAD(4, half, int64_t);
DECLARE_INDIRECT_LOAD(4, bfloat16_t, int64_t);
DECLARE_INDIRECT_LOAD(4, int8_t, int64_t);
DECLARE_INDIRECT_LOAD(4, int16_t, int64_t);
DECLARE_INDIRECT_LOAD(4, int32_t, int64_t);
DECLARE_INDIRECT_LOAD(4, int64_t, int64_t);
DECLARE_INDIRECT_LOAD(4, uint8_t, int64_t);
DECLARE_INDIRECT_LOAD(4, uint16_t, int64_t);
DECLARE_INDIRECT_LOAD(4, uint32_t, int64_t);
DECLARE_INDIRECT_LOAD(4, uint64_t, int64_t);
DECLARE_INDIRECT_LOAD(4, bool, int64_t);
DECLARE_INDIRECT_LOAD(4, hifloat8_t, int64_t);
DECLARE_INDIRECT_LOAD(4, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_LOAD(4, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_LOAD(4, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_LOAD(4, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_LOAD(4, float4_e1m2_t, int64_t);

DECLARE_INDIRECT_LOAD(5, float, int32_t);
DECLARE_INDIRECT_LOAD(5, half, int32_t);
DECLARE_INDIRECT_LOAD(5, bfloat16_t, int32_t);
DECLARE_INDIRECT_LOAD(5, int8_t, int32_t);
DECLARE_INDIRECT_LOAD(5, int16_t, int32_t);
DECLARE_INDIRECT_LOAD(5, int32_t, int32_t);
DECLARE_INDIRECT_LOAD(5, int64_t, int32_t);
DECLARE_INDIRECT_LOAD(5, uint8_t, int32_t);
DECLARE_INDIRECT_LOAD(5, uint16_t, int32_t);
DECLARE_INDIRECT_LOAD(5, uint32_t, int32_t);
DECLARE_INDIRECT_LOAD(5, uint64_t, int32_t);
DECLARE_INDIRECT_LOAD(5, bool, int32_t);
DECLARE_INDIRECT_LOAD(5, hifloat8_t, int32_t);
DECLARE_INDIRECT_LOAD(5, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_LOAD(5, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_LOAD(5, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_LOAD(5, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_LOAD(5, float4_e1m2_t, int32_t);

DECLARE_INDIRECT_LOAD(5, float, int64_t);
DECLARE_INDIRECT_LOAD(5, half, int64_t);
DECLARE_INDIRECT_LOAD(5, bfloat16_t, int64_t);
DECLARE_INDIRECT_LOAD(5, int8_t, int64_t);
DECLARE_INDIRECT_LOAD(5, int16_t, int64_t);
DECLARE_INDIRECT_LOAD(5, int32_t, int64_t);
DECLARE_INDIRECT_LOAD(5, int64_t, int64_t);
DECLARE_INDIRECT_LOAD(5, uint8_t, int64_t);
DECLARE_INDIRECT_LOAD(5, uint16_t, int64_t);
DECLARE_INDIRECT_LOAD(5, uint32_t, int64_t);
DECLARE_INDIRECT_LOAD(5, uint64_t, int64_t);
DECLARE_INDIRECT_LOAD(5, bool, int64_t);
DECLARE_INDIRECT_LOAD(5, hifloat8_t, int64_t);
DECLARE_INDIRECT_LOAD(5, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_LOAD(5, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_LOAD(5, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_LOAD(5, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_LOAD(5, float4_e1m2_t, int64_t);

DECLARE_SHIFT_VV(vshr);
DECLARE_SHIFT_VV(vshl);
DECLARE_SHIFT_VS(vshrs);
DECLARE_SHIFT_VS(vshls);

DECLARE_INDIRECT_STORE(1, float, int32_t);
DECLARE_INDIRECT_STORE(1, half, int32_t);
DECLARE_INDIRECT_STORE(1, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE(1, int8_t, int32_t);
DECLARE_INDIRECT_STORE(1, int16_t, int32_t);
DECLARE_INDIRECT_STORE(1, int32_t, int32_t);
DECLARE_INDIRECT_STORE(1, int64_t, int32_t);
DECLARE_INDIRECT_STORE(1, uint8_t, int32_t);
DECLARE_INDIRECT_STORE(1, uint16_t, int32_t);
DECLARE_INDIRECT_STORE(1, uint32_t, int32_t);
DECLARE_INDIRECT_STORE(1, uint64_t, int32_t);
DECLARE_INDIRECT_STORE(1, bool, int32_t);
DECLARE_INDIRECT_STORE(1, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE(1, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE(1, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE(1, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE(1, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE(1, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE(1, float, int64_t);
DECLARE_INDIRECT_STORE(1, half, int64_t);
DECLARE_INDIRECT_STORE(1, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE(1, int8_t, int64_t);
DECLARE_INDIRECT_STORE(1, int16_t, int64_t);
DECLARE_INDIRECT_STORE(1, int32_t, int64_t);
DECLARE_INDIRECT_STORE(1, int64_t, int64_t);
DECLARE_INDIRECT_STORE(1, uint8_t, int64_t);
DECLARE_INDIRECT_STORE(1, uint16_t, int64_t);
DECLARE_INDIRECT_STORE(1, uint32_t, int64_t);
DECLARE_INDIRECT_STORE(1, uint64_t, int64_t);
DECLARE_INDIRECT_STORE(1, bool, int64_t);
DECLARE_INDIRECT_STORE(1, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE(1, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE(1, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE(1, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE(1, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE(1, float4_e1m2_t, int64_t);
 
DECLARE_INDIRECT_STORE(2, float, int32_t);
DECLARE_INDIRECT_STORE(2, half, int32_t);
DECLARE_INDIRECT_STORE(2, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE(2, int8_t, int32_t);
DECLARE_INDIRECT_STORE(2, int16_t, int32_t);
DECLARE_INDIRECT_STORE(2, int32_t, int32_t);
DECLARE_INDIRECT_STORE(2, int64_t, int32_t);
DECLARE_INDIRECT_STORE(2, uint8_t, int32_t);
DECLARE_INDIRECT_STORE(2, uint16_t, int32_t);
DECLARE_INDIRECT_STORE(2, uint32_t, int32_t);
DECLARE_INDIRECT_STORE(2, uint64_t, int32_t);
DECLARE_INDIRECT_STORE(2, bool, int32_t);
DECLARE_INDIRECT_STORE(2, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE(2, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE(2, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE(2, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE(2, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE(2, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE(2, float, int64_t);
DECLARE_INDIRECT_STORE(2, half, int64_t);
DECLARE_INDIRECT_STORE(2, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE(2, int8_t, int64_t);
DECLARE_INDIRECT_STORE(2, int16_t, int64_t);
DECLARE_INDIRECT_STORE(2, int32_t, int64_t);
DECLARE_INDIRECT_STORE(2, int64_t, int64_t);
DECLARE_INDIRECT_STORE(2, uint8_t, int64_t);
DECLARE_INDIRECT_STORE(2, uint16_t, int64_t);
DECLARE_INDIRECT_STORE(2, uint32_t, int64_t);
DECLARE_INDIRECT_STORE(2, uint64_t, int64_t);
DECLARE_INDIRECT_STORE(2, bool, int64_t);
DECLARE_INDIRECT_STORE(2, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE(2, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE(2, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE(2, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE(2, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE(2, float4_e1m2_t, int64_t);
 
DECLARE_INDIRECT_STORE(3, float, int32_t);
DECLARE_INDIRECT_STORE(3, half, int32_t);
DECLARE_INDIRECT_STORE(3, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE(3, int8_t, int32_t);
DECLARE_INDIRECT_STORE(3, int16_t, int32_t);
DECLARE_INDIRECT_STORE(3, int32_t, int32_t);
DECLARE_INDIRECT_STORE(3, int64_t, int32_t);
DECLARE_INDIRECT_STORE(3, uint8_t, int32_t);
DECLARE_INDIRECT_STORE(3, uint16_t, int32_t);
DECLARE_INDIRECT_STORE(3, uint32_t, int32_t);
DECLARE_INDIRECT_STORE(3, uint64_t, int32_t);
DECLARE_INDIRECT_STORE(3, bool, int32_t);
DECLARE_INDIRECT_STORE(3, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE(3, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE(3, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE(3, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE(3, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE(3, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE(3, float, int64_t);
DECLARE_INDIRECT_STORE(3, half, int64_t);
DECLARE_INDIRECT_STORE(3, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE(3, int8_t, int64_t);
DECLARE_INDIRECT_STORE(3, int16_t, int64_t);
DECLARE_INDIRECT_STORE(3, int32_t, int64_t);
DECLARE_INDIRECT_STORE(3, int64_t, int64_t);
DECLARE_INDIRECT_STORE(3, uint8_t, int64_t);
DECLARE_INDIRECT_STORE(3, uint16_t, int64_t);
DECLARE_INDIRECT_STORE(3, uint32_t, int64_t);
DECLARE_INDIRECT_STORE(3, uint64_t, int64_t);
DECLARE_INDIRECT_STORE(3, bool, int64_t);
DECLARE_INDIRECT_STORE(3, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE(3, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE(3, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE(3, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE(3, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE(3, float4_e1m2_t, int64_t);
 
DECLARE_INDIRECT_STORE(4, float, int32_t);
DECLARE_INDIRECT_STORE(4, half, int32_t);
DECLARE_INDIRECT_STORE(4, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE(4, int8_t, int32_t);
DECLARE_INDIRECT_STORE(4, int16_t, int32_t);
DECLARE_INDIRECT_STORE(4, int32_t, int32_t);
DECLARE_INDIRECT_STORE(4, int64_t, int32_t);
DECLARE_INDIRECT_STORE(4, uint8_t, int32_t);
DECLARE_INDIRECT_STORE(4, uint16_t, int32_t);
DECLARE_INDIRECT_STORE(4, uint32_t, int32_t);
DECLARE_INDIRECT_STORE(4, uint64_t, int32_t);
DECLARE_INDIRECT_STORE(4, bool, int32_t);
DECLARE_INDIRECT_STORE(4, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE(4, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE(4, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE(4, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE(4, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE(4, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE(4, float, int64_t);
DECLARE_INDIRECT_STORE(4, half, int64_t);
DECLARE_INDIRECT_STORE(4, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE(4, int8_t, int64_t);
DECLARE_INDIRECT_STORE(4, int16_t, int64_t);
DECLARE_INDIRECT_STORE(4, int32_t, int64_t);
DECLARE_INDIRECT_STORE(4, int64_t, int64_t);
DECLARE_INDIRECT_STORE(4, uint8_t, int64_t);
DECLARE_INDIRECT_STORE(4, uint16_t, int64_t);
DECLARE_INDIRECT_STORE(4, uint32_t, int64_t);
DECLARE_INDIRECT_STORE(4, uint64_t, int64_t);
DECLARE_INDIRECT_STORE(4, bool, int64_t);
DECLARE_INDIRECT_STORE(4, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE(4, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE(4, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE(4, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE(4, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE(4, float4_e1m2_t, int64_t);
 
DECLARE_INDIRECT_STORE(5, float, int32_t);
DECLARE_INDIRECT_STORE(5, half, int32_t);
DECLARE_INDIRECT_STORE(5, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE(5, int8_t, int32_t);
DECLARE_INDIRECT_STORE(5, int16_t, int32_t);
DECLARE_INDIRECT_STORE(5, int32_t, int32_t);
DECLARE_INDIRECT_STORE(5, int64_t, int32_t);
DECLARE_INDIRECT_STORE(5, uint8_t, int32_t);
DECLARE_INDIRECT_STORE(5, uint16_t, int32_t);
DECLARE_INDIRECT_STORE(5, uint32_t, int32_t);
DECLARE_INDIRECT_STORE(5, uint64_t, int32_t);
DECLARE_INDIRECT_STORE(5, bool, int32_t);
DECLARE_INDIRECT_STORE(5, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE(5, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE(5, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE(5, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE(5, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE(5, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE(5, float, int64_t);
DECLARE_INDIRECT_STORE(5, half, int64_t);
DECLARE_INDIRECT_STORE(5, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE(5, int8_t, int64_t);
DECLARE_INDIRECT_STORE(5, int16_t, int64_t);
DECLARE_INDIRECT_STORE(5, int32_t, int64_t);
DECLARE_INDIRECT_STORE(5, int64_t, int64_t);
DECLARE_INDIRECT_STORE(5, uint8_t, int64_t);
DECLARE_INDIRECT_STORE(5, uint16_t, int64_t);
DECLARE_INDIRECT_STORE(5, uint32_t, int64_t);
DECLARE_INDIRECT_STORE(5, uint64_t, int64_t);
DECLARE_INDIRECT_STORE(5, bool, int64_t);
DECLARE_INDIRECT_STORE(5, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE(5, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE(5, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE(5, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE(5, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE(5, float4_e1m2_t, int64_t);

// indirectSTORE 1D
DECLARE_INDIRECT_STORE_NO_MASK(1, float, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, half, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, int8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, int16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, int32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, int64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, uint8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, uint16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, uint32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, uint64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, bool, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(1, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(1, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE_NO_MASK(1, float, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, half, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, int8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, int16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, int32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, int64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, uint8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, uint16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, uint32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, uint64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, bool, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(1, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(1, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(1, float4_e1m2_t, int64_t);
 
DECLARE_INDIRECT_STORE_NO_MASK(2, float, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, half, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, int8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, int16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, int32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, int64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, uint8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, uint16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, uint32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, uint64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, bool, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(2, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(2, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE_NO_MASK(2, float, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, half, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, int8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, int16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, int32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, int64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, uint8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, uint16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, uint32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, uint64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, bool, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(2, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(2, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(2, float4_e1m2_t, int64_t);
 
DECLARE_INDIRECT_STORE_NO_MASK(3, float, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, half, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, int8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, int16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, int32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, int64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, uint8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, uint16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, uint32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, uint64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, bool, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(3, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(3, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE_NO_MASK(3, float, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, half, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, int8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, int16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, int32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, int64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, uint8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, uint16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, uint32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, uint64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, bool, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(3, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(3, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(3, float4_e1m2_t, int64_t);
 
DECLARE_INDIRECT_STORE_NO_MASK(4, float, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, half, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, int8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, int16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, int32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, int64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, uint8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, uint16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, uint32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, uint64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, bool, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(4, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(4, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE_NO_MASK(4, float, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, half, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, int8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, int16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, int32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, int64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, uint8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, uint16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, uint32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, uint64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, bool, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(4, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(4, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(4, float4_e1m2_t, int64_t);
 
DECLARE_INDIRECT_STORE_NO_MASK(5, float, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, half, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, bfloat16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, int8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, int16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, int32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, int64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, uint8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, uint16_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, uint32_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, uint64_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, bool, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, hifloat8_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, hifloat4x2_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, float8_e4m3_t, int32_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, float8_e5m2_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(5, float4_e2m1_t, int32_t);
// DECLARE_INDIRECT_STORE_NO_MASK(5, float4_e1m2_t, int32_t);
 
DECLARE_INDIRECT_STORE_NO_MASK(5, float, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, half, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, bfloat16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, int8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, int16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, int32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, int64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, uint8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, uint16_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, uint32_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, uint64_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, bool, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, hifloat8_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, hifloat4x2_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, float8_e4m3_t, int64_t);
DECLARE_INDIRECT_STORE_NO_MASK(5, float8_e5m2_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(5, float4_e2m1_t, int64_t);
// DECLARE_INDIRECT_STORE_NO_MASK(5, float4_e1m2_t, int64_t);
DECLARE_INT_VDIV(vdiv, 1, int16_t);
DECLARE_INT_VDIV(vdiv, 1, int32_t);
DECLARE_INT_VDIV(vdiv, 2, int16_t);
DECLARE_INT_VDIV(vdiv, 2, int32_t);
DECLARE_INT_VDIV(vdiv, 3, int16_t);
DECLARE_INT_VDIV(vdiv, 3, int32_t);
DECLARE_INT_VDIV(vdiv, 4, int16_t);
DECLARE_INT_VDIV(vdiv, 4, int32_t);
DECLARE_INT_VDIV(vdiv, 5, int16_t);
DECLARE_INT_VDIV(vdiv, 5, int32_t);
DECLARE_INT_VDIV(vdiv, 6, int16_t);
DECLARE_INT_VDIV(vdiv, 6, int32_t);
DECLARE_INT_VDIV(vdiv, 7, int16_t);
DECLARE_INT_VDIV(vdiv, 7, int32_t);
DECLARE_INT_VDIV(vdiv, 8, int16_t);
DECLARE_INT_VDIV(vdiv, 8, int32_t);

DECLARE_INT_VDIV_SCALAR(vdiv_vs, 1, int16_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 1, int32_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 2, int16_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 2, int32_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 3, int16_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 3, int32_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 4, int16_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 4, int32_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 5, int16_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 5, int32_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 6, int16_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 6, int32_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 7, int16_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 7, int32_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 8, int16_t);
DECLARE_INT_VDIV_SCALAR(vdiv_vs, 8, int32_t);
DECLARE_INT_VDIV(vdiv, 1, uint16_t);
DECLARE_INT_VDIV(vdiv, 1, uint32_t);

DECLARE_FLOAT_VDIV_HP(vdivfhp);

DECLARE_VMODUI(vmodui, int16_t);
DECLARE_VMODUI(vmodui, int32_t);
DECLARE_VMOD(vmod, int16_t);
DECLARE_VMOD(vmod, int32_t);
}

#endif

#endif // HIVMAVE_MLIR_TEMPLATE_VECTOR_UTILS_H

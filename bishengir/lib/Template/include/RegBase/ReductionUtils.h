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

#ifndef HIVMAVE_MLIR_TEMPLATE_VECTOR_REDUCTION_UTILS_H
#define HIVMAVE_MLIR_TEMPLATE_VECTOR_REDUCTION_UTILS_H

#include "./VecUtils.h"
#include <cstdint>
#include <type_traits>

#define VL_IN_BYTE CCE_VL

enum class ReduceOpTy : uint32_t {
  REDUCE_BEGIN = 100,
  REDUCE_MAX_WITH_INDEX = 101,
  REDUCE_MIN_WITH_INDEX = 102,
  REDUCE_END = 199,
};

enum class TieBreak : uint32_t {
  LEFT = 300,
  RIGHT = 301,
};

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
__aiv__ __attribute__((always_inline)) void
reduce_r_with_index(memref_t<__ubuf__ T0, 1> *src0,
                    memref_t<__ubuf__ T0, 1> *dst_value,
                    memref_t<__ubuf__ int32_t, 1> *dst_index, T0 initvalue);

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
__aiv__ __attribute__((always_inline)) void
reduce_ar_with_index(memref_t<__ubuf__ T0, 2> *src0,
                     memref_t<__ubuf__ T0, 2> *dst_value,
                     memref_t<__ubuf__ int32_t, 2> *dst_index, T0 initvalue);

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
__aiv__ __attribute__((always_inline)) void
reduce_ra_with_index(memref_t<__ubuf__ T0, 2> *src0,
                     memref_t<__ubuf__ T0, 2> *dst_value,
                     memref_t<__ubuf__ int32_t, 2> *dst_index, T0 initvalue);

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
__aiv__ __attribute__((always_inline)) void
reduce_ra0a1_with_index(memref_t<__ubuf__ T0, 3> *src0,
                        memref_t<__ubuf__ T0, 3> *dst_value,
                        memref_t<__ubuf__ int32_t, 3> *dst_index, T0 initvalue);

/// idxtype refers to the datatype of the register where we store index.
/// Currently it can be either b16/b32 to align with the length of input data
/// however, using b16 register to store index has risk of int overflow.
/// it's better to refactor the implementation to always use b32 to store index
#define DECLARE_REDUCE_R_WITH_INDEX_TIEBREAK(                                  \
    op_name, tie_break_macro, op_type, dim, dtype, idxtype)                    \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##tie_break_macro##_##r##_##dtype(              \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index, dtype initvalue)

#define REGISTE_REDUCE_R_WITH_INDEX_TIEBREAK(                                  \
    op_name, tie_break_macro, tie_break, op_type, dim, dtype, idxtype)         \
  DECLARE_REDUCE_R_WITH_INDEX_TIEBREAK(                                        \
      op_name, tie_break_macro, op_type, dim, dtype, idxtype) {                \
    reduce_r_with_index<op_type, tie_break, dtype, idxtype>(                   \
        src0, dst_value, dst_index, initvalue);                                \
  }

#define DECLARE_REDUCE_R_WITH_INDEX(op_name, op_type, dim, dtype, idxtype)     \
  DECLARE_REDUCE_R_WITH_INDEX_TIEBREAK(                                        \
      op_name, left, op_type, dim, dtype, idxtype);                            \
  DECLARE_REDUCE_R_WITH_INDEX_TIEBREAK(                                        \
      op_name, right, op_type, dim, dtype, idxtype)

#define REGISTE_REDUCE_R_WITH_INDEX(op_name, op_type, dim, dtype, idxtype)     \
  REGISTE_REDUCE_R_WITH_INDEX_TIEBREAK(                                        \
      op_name, left, TieBreak::LEFT, op_type, dim, dtype, idxtype)             \
  REGISTE_REDUCE_R_WITH_INDEX_TIEBREAK(                                        \
      op_name, right, TieBreak::RIGHT, op_type, dim, dtype, idxtype)

#define DECLARE_REDUCE_AR_WITH_INDEX_TIEBREAK(                                 \
    op_name, tie_break_macro, op_type, dim, dtype, idxtype)                    \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##tie_break_macro##_##ar##_##dtype(             \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index, dtype initvalue)

#define REGISTE_REDUCE_AR_WITH_INDEX_TIEBREAK(                                 \
    op_name, tie_break_macro, tie_break, op_type, dim, dtype, idxtype)         \
  DECLARE_REDUCE_AR_WITH_INDEX_TIEBREAK(                                       \
      op_name, tie_break_macro, op_type, dim, dtype, idxtype) {                \
    reduce_ar_with_index<op_type, tie_break, dtype, idxtype>(                  \
        src0, dst_value, dst_index, initvalue);                                \
  }

#define DECLARE_REDUCE_AR_WITH_INDEX(op_name, op_type, dim, dtype, idxtype)    \
  DECLARE_REDUCE_AR_WITH_INDEX_TIEBREAK(                                       \
      op_name, left, op_type, dim, dtype, idxtype);                            \
  DECLARE_REDUCE_AR_WITH_INDEX_TIEBREAK(                                       \
      op_name, right, op_type, dim, dtype, idxtype)

#define REGISTE_REDUCE_AR_WITH_INDEX(op_name, op_type, dim, dtype, idxtype)    \
  REGISTE_REDUCE_AR_WITH_INDEX_TIEBREAK(                                       \
      op_name, left, TieBreak::LEFT, op_type, dim, dtype, idxtype)             \
  REGISTE_REDUCE_AR_WITH_INDEX_TIEBREAK(                                       \
      op_name, right, TieBreak::RIGHT, op_type, dim, dtype, idxtype)

#define DECLARE_REDUCE_RA_WITH_INDEX_TIEBREAK(                                 \
    op_name, tie_break_macro, op_type, dim, dtype, idxtype)                    \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##tie_break_macro##_##ra##_##dtype(             \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index, dtype initvalue)

#define REGISTE_REDUCE_RA_WITH_INDEX_TIEBREAK(                                 \
    op_name, tie_break_macro, tie_break, op_type, dim, dtype, idxtype)         \
  DECLARE_REDUCE_RA_WITH_INDEX_TIEBREAK(                                       \
      op_name, tie_break_macro, op_type, dim, dtype, idxtype) {                \
    reduce_ra_with_index<op_type, tie_break, dtype, idxtype>(                  \
        src0, dst_value, dst_index, initvalue);                                \
  }

#define DECLARE_REDUCE_RA_WITH_INDEX(op_name, op_type, dim, dtype, idxtype)    \
  DECLARE_REDUCE_RA_WITH_INDEX_TIEBREAK(                                       \
      op_name, left, op_type, dim, dtype, idxtype);                            \
  DECLARE_REDUCE_RA_WITH_INDEX_TIEBREAK(                                       \
      op_name, right, op_type, dim, dtype, idxtype)

#define REGISTE_REDUCE_RA_WITH_INDEX(op_name, op_type, dim, dtype, idxtype)    \
  REGISTE_REDUCE_RA_WITH_INDEX_TIEBREAK(                                       \
      op_name, left, TieBreak::LEFT, op_type, dim, dtype, idxtype)             \
  REGISTE_REDUCE_RA_WITH_INDEX_TIEBREAK(                                       \
      op_name, right, TieBreak::RIGHT, op_type, dim, dtype, idxtype)

#define DECLARE_REDUCE_RA0A1_WITH_INDEX_TIEBREAK(                              \
    op_name, tie_break_macro, op_type, dim, dtype, idxtype)                    \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_##op_name##_##tie_break_macro##_##ra0a1##_##dtype(          \
          memref_t<__ubuf__ dtype, dim> *src0,                                 \
          memref_t<__ubuf__ dtype, dim> *dst_value,                            \
          memref_t<__ubuf__ int32_t, dim> *dst_index, dtype initvalue)

#define REGISTE_REDUCE_RA0A1_WITH_INDEX_TIEBREAK(                              \
    op_name, tie_break_macro, tie_break, op_type, dim, dtype, idxtype)         \
  DECLARE_REDUCE_RA0A1_WITH_INDEX_TIEBREAK(                                    \
      op_name, tie_break_macro, op_type, dim, dtype, idxtype) {                \
    reduce_ra0a1_with_index<op_type, tie_break, dtype, idxtype>(               \
        src0, dst_value, dst_index, initvalue);                                \
  }

#define DECLARE_REDUCE_RA0A1_WITH_INDEX(op_name, op_type, dim, dtype, idxtype) \
  DECLARE_REDUCE_RA0A1_WITH_INDEX_TIEBREAK(                                    \
      op_name, left, op_type, dim, dtype, idxtype);                            \
  DECLARE_REDUCE_RA0A1_WITH_INDEX_TIEBREAK(                                    \
      op_name, right, op_type, dim, dtype, idxtype)

#define REGISTE_REDUCE_RA0A1_WITH_INDEX(op_name, op_type, dim, dtype, idxtype) \
  REGISTE_REDUCE_RA0A1_WITH_INDEX_TIEBREAK(                                    \
      op_name, left, TieBreak::LEFT, op_type, dim, dtype, idxtype)             \
  REGISTE_REDUCE_RA0A1_WITH_INDEX_TIEBREAK(                                    \
      op_name, right, TieBreak::RIGHT, op_type, dim, dtype, idxtype)

extern "C" {
//===-------------------------------------------------------------------===//
// reduce r with index, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, half,
                            int16_t);
DECLARE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, float,
                            int32_t);
DECLARE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, int16_t,
                            int16_t);
DECLARE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, int32_t,
                            int32_t);
DECLARE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, int64_t,
                            int32_t);

DECLARE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, half,
                            int16_t);
DECLARE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, float,
                            int32_t);
DECLARE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, int16_t,
                            int16_t);
DECLARE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, int32_t,
                            int32_t);
DECLARE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, int64_t,
                            int32_t);

//===-------------------------------------------------------------------===//
// reduce ar with index, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, half,
                             int16_t);
DECLARE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, float,
                             int32_t);
DECLARE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int16_t,
                             int16_t);
DECLARE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int32_t,
                             int32_t);
DECLARE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int64_t,
                             int32_t);

DECLARE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, half,
                             int16_t);
DECLARE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, float,
                             int32_t);
DECLARE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int16_t,
                             int16_t);
DECLARE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int32_t,
                             int32_t);
DECLARE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int64_t,
                             int32_t);

//===-------------------------------------------------------------------===//
// reduce ra with index, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, half,
                             int16_t);
DECLARE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, float,
                             int32_t);
DECLARE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int16_t,
                             int16_t);
DECLARE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int32_t,
                             int32_t);
DECLARE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int64_t,
                             int32_t);

DECLARE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, half,
                             int16_t);
DECLARE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, float,
                             int32_t);
DECLARE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int16_t,
                             int16_t);
DECLARE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int32_t,
                             int32_t);
DECLARE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int64_t,
                             int32_t);

//===-------------------------------------------------------------------===//
// reduce ra0a1 with index, 3 dim
//===-------------------------------------------------------------------===//
DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, half,
                                int16_t);
DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, float,
                                int32_t);
DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, int16_t,
                                int16_t);
DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, int32_t,
                                int32_t);
DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, int64_t,
                                int32_t);

DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, half,
                                int16_t);
DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, float,
                                int32_t);
DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, int16_t,
                                int16_t);
DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, int32_t,
                                int32_t);
DECLARE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, int64_t,
                                int32_t);
}

#endif
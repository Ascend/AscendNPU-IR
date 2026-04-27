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

#ifndef HIVM_MLIR_TEMPLATE_FIXPIPE_UTILS_H
#define HIVM_MLIR_TEMPLATE_FIXPIPE_UTILS_H
#include "Utils.h"

enum class TransformMode : uint32_t {
  NORMAL = 0,
  NZ_2_ND = 1,
  NZ_2_DN = 2,
  NC1HWC0_2_NHWC = 3,
};

enum DualDstMode {
  NO_DUAL = 0,
  ROW_SPLIT = 1,
  COLUMN_SPLIT = 2,
};

// accept certain declarations in this architecture
#if defined(__DAV_C310__)
#define D_C310(f, ...) f(__VA_ARGS__)
#else
#define D_C310(f, ...)
#endif
#if defined(__DAV_M300__)
#define D_M300(f, ...) f(__VA_ARGS__)
#else
#define D_M300(f, ...)
#endif

#define WRAP(...) __VA_ARGS__

#define REQUIRE_1(x, D1) D1(x)
#define REQUIRE_2(x, D1, D2) D1(x) D2(x)
#define REQUIRE_3(x, D1, D2, D3) D1(x) D2(x) D3(x)

#define FIXPIPE_XT1_ARG_LIST_TO_UB(f, DUAL_DST_INIT) \
  REQUIRE_1(WRAP(f, uint8_t, dual_dst_ctl, DUAL_DST_INIT), D_C310) \
  REQUIRE_1(WRAP(f, bool, sub_blockid, false), D_C310) \
  REQUIRE_2(WRAP(f, uint8_t, clip_relu_pre, 0), D_C310, D_M300)

#define FIXPIPE_XT1_ARG_LIST_TO_L1(f) \
  REQUIRE_1(WRAP(f, uint8_t, l2_cache_ctl, 0), D_C310) \
  REQUIRE_2(WRAP(f, uint8_t, clip_relu_pre, 0), D_C310, D_M300)

#define FIXPIPE_XT1_ARG_LIST_TO_GM(f) \
  REQUIRE_1(WRAP(f, uint8_t, l2_cache_ctl, 0), D_C310) \
  REQUIRE_2(WRAP(f, uint8_t, clip_relu_pre, 0), D_C310, D_M300)

#define FIXPIPE_XT2_ARG_LIST(f, NZ2DN_INIT) \
  REQUIRE_2(WRAP(f, uint64_t, quant_post, 0), D_C310, D_M300) \
  REQUIRE_2(WRAP(f, uint8_t, relu_post, 0), D_C310, D_M300) \
  REQUIRE_2(WRAP(f, bool, clip_relu_post, false), D_C310, D_M300) \
  REQUIRE_1(WRAP(f, bool, loop_enhance_en, false), D_C310) \
  REQUIRE_2(WRAP(f, uint8_t, eltwise_op, 0), D_C310, D_M300) \
  REQUIRE_1(WRAP(f, uint8_t, eltwise_antq_cfg, 0), D_M300) \
  REQUIRE_1(WRAP(f, bool, eltwise_antq_en, false), D_C310) \
  REQUIRE_1(WRAP(f, bool, loop_enhance_merge_en, false), D_C310) \
  REQUIRE_2(WRAP(f, bool, C0_pad_en, false), D_C310, D_M300) \
  REQUIRE_1(WRAP(f, bool, wino_post_en, false), D_C310) \
  REQUIRE_1(WRAP(f, bool, broadcast_en, false), D_C310) \
  REQUIRE_1(WRAP(f, bool, NZ2DN_en, NZ2DN_INIT), D_C310) \

#define STRUCTDEF(type, name, initval) type name;
struct copy_matrix_cc_to_gm_ub_intrin_args_xt_1_ub {
  FIXPIPE_XT1_ARG_LIST_TO_UB(STRUCTDEF, 0 /*dual_dst*/)
};
struct copy_matrix_cc_to_gm_ub_intrin_args_xt_1_gm {
  FIXPIPE_XT1_ARG_LIST_TO_GM(STRUCTDEF)
};
struct copy_matrix_cc_to_gm_ub_intrin_args_xt_1_cb {
  FIXPIPE_XT1_ARG_LIST_TO_L1(STRUCTDEF)
};
struct copy_matrix_cc_to_gm_ub_intrin_args_xt_2 {
  FIXPIPE_XT2_ARG_LIST(STRUCTDEF, false /*nz2dn_en*/)
};
#undef STRUCTDEF

#define FIXPIPE_ARG_LIST_HANDLER_XT1(type, name, initval) args.xt1. name,
#define FIXPIPE_ARG_LIST_HANDLER_XT2(type, name, initval) , args.xt2. name
#define FIXPIPE_ARGS_XT1_TO_GM FIXPIPE_XT1_ARG_LIST_TO_GM(FIXPIPE_ARG_LIST_HANDLER_XT1)
#define FIXPIPE_ARGS_XT1_TO_UB FIXPIPE_XT1_ARG_LIST_TO_UB(FIXPIPE_ARG_LIST_HANDLER_XT1, 0)
#define FIXPIPE_ARGS_XT1_TO_L1 FIXPIPE_XT1_ARG_LIST_TO_L1(FIXPIPE_ARG_LIST_HANDLER_XT1)
#define FIXPIPE_ARGS_XT2 FIXPIPE_XT2_ARG_LIST(FIXPIPE_ARG_LIST_HANDLER_XT2, false)

#define FIXPIPE_ARG_GET_VALUE(type, name, initval) initval,
#define FIXPIPE_ARGS_XT1_VALUES_TO_GM { FIXPIPE_XT1_ARG_LIST_TO_GM(FIXPIPE_ARG_GET_VALUE) },
#define FIXPIPE_ARGS_XT1_VALUES_TO_UB(dual_init) { FIXPIPE_XT1_ARG_LIST_TO_UB(FIXPIPE_ARG_GET_VALUE, dual_init) },
#define FIXPIPE_ARGS_XT1_VALUES_TO_L1 { FIXPIPE_XT1_ARG_LIST_TO_L1(FIXPIPE_ARG_GET_VALUE) },
#define FIXPIPE_ARGS_XT2_VALUES(nz2dn_init) ,{ FIXPIPE_XT2_ARG_LIST(FIXPIPE_ARG_GET_VALUE, nz2dn_init) }

template <typename SRC_TYPE, typename DST_TYPE>
struct copy_matrix_cc_to_gm_intrin_args {
  __gm__ DST_TYPE *dst;
  __cc__ SRC_TYPE *src;
  uint8_t sid;
  uint16_t n_size;
  uint16_t m_size;
  uint32_t dst_D;
  uint16_t src_stride;
  copy_matrix_cc_to_gm_ub_intrin_args_xt_1_gm xt1;
  uint8_t unit_flag_mode;
  QuantMode_t pre_quant;
  uint8_t pre_relu;
  bool channel_split;
  bool nz2nd_en;
  copy_matrix_cc_to_gm_ub_intrin_args_xt_2 xt2;
};

template <typename SRC_TYPE, typename DST_TYPE>
struct copy_matrix_cc_to_ubuf_intrin_args {
  __ubuf__ DST_TYPE *dst;
  __cc__   SRC_TYPE *src;
  uint8_t sid;
  uint16_t n_size;
  uint16_t m_size;
  uint32_t dst_D;
  uint16_t src_stride;
  copy_matrix_cc_to_gm_ub_intrin_args_xt_1_ub xt1;
  uint8_t unit_flag_mode;
  QuantMode_t pre_quant;
  uint8_t pre_relu;
  bool channel_split;
  bool nz2nd_en;
  copy_matrix_cc_to_gm_ub_intrin_args_xt_2 xt2;
};

template <typename SRC_TYPE, typename DST_TYPE>
struct copy_matrix_cc_to_cbuf_intrin_args {
  __cbuf__ DST_TYPE *dst;
  __cc__   SRC_TYPE *src;
  uint8_t sid;
  uint16_t n_size;
  uint16_t m_size;
  uint32_t dst_D;
  uint16_t src_stride;
  copy_matrix_cc_to_gm_ub_intrin_args_xt_1_cb xt1;
  uint8_t unit_flag_mode;
  QuantMode_t pre_quant;
  uint8_t pre_relu;
  bool channel_split;
  bool nz2nd_en;
  copy_matrix_cc_to_gm_ub_intrin_args_xt_2 xt2;
};

__aicore__ __attribute__((always_inline)) QuantMode_t
get_quant_mode(int64_t pre_quant) {
  if (pre_quant == static_cast<int64_t>(QuantMode_t::F322F16)) {
    return QuantMode_t::F322F16;
  }
  #if defined(__DAV_C310__)
  if (pre_quant == static_cast<int64_t>(QuantMode_t::QF322F32_PRE)) {
    return QuantMode_t::QF322F32_PRE;
  }
  #endif
  if (pre_quant == static_cast<int64_t>(QuantMode_t::F322BF16)) {
    return QuantMode_t::F322BF16;
  }
  if (pre_quant == static_cast<int64_t>(QuantMode_t::REQ8)) {
    return QuantMode_t::REQ8;
  }
  return QuantMode_t::NoQuant;
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void copy_matrix_cc_to_gm_intrin(
    copy_matrix_cc_to_gm_intrin_args<SRC_TYPE, DST_TYPE> args) {
#define FIXPIPE_ARGS_XT1 FIXPIPE_ARGS_XT1_TO_GM
#if !defined(__DAV_M300__) && !defined(__DAV_C310__)
#define FIXPIPE_ARGS                                                           \
  args.dst, args.src, args.sid, args.n_size, args.m_size, args.dst_D,          \
      args.src_stride, args.unit_flag_mode, args.pre_quant, args.pre_relu,     \
      args.channel_split, args.nz2nd_en
#else
#define FIXPIPE_ARGS                                                           \
  args.dst, args.src, args.sid, args.n_size, args.m_size, args.dst_D,          \
      args.src_stride, FIXPIPE_ARGS_XT1 args.unit_flag_mode,                   \
      static_cast<uint64_t>(args.pre_quant), args.pre_relu,                    \
      args.channel_split, args.nz2nd_en FIXPIPE_ARGS_XT2
#endif

  INTRINSIC(copy_matrix_cc_to_gm, FIXPIPE_ARGS);
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void copy_matrix_cc_to_cbuf_intrin(
     copy_matrix_cc_to_cbuf_intrin_args<SRC_TYPE, DST_TYPE> args) {
#undef FIXPIPE_ARGS_XT1
#define FIXPIPE_ARGS_XT1 FIXPIPE_ARGS_XT1_TO_L1
  INTRINSIC(copy_matrix_cc_to_cbuf, FIXPIPE_ARGS);
#undef FIXPIPE_ARGS_XT1
     }

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void copy_matrix_cc_to_ubuf_intrin(
    copy_matrix_cc_to_ubuf_intrin_args<SRC_TYPE, DST_TYPE> args) {
#undef FIXPIPE_ARGS_XT1
#define FIXPIPE_ARGS_XT1 FIXPIPE_ARGS_XT1_TO_UB
  INTRINSIC(copy_matrix_cc_to_ub, FIXPIPE_ARGS);
#undef FIXPIPE_ARGS_XT1
#undef FIXPIPE_ARGS
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_normal_2d_to_2d_core(memref_t<__cc__ SRC_TYPE, 2> *l0c,
                                          memref_t<__gm__ DST_TYPE, 2> *gm,
                                          int64_t pre_quant,
                                          float32_t quant_scale,
                                          int64_t pre_relu, bool channel_split,
                                          uint8_t unit_flag);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_normal_2d_to_2d_core(
    memref_t<__cc__ SRC_TYPE, 2> *l0c, memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
    int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,
    bool channel_split, uint8_t unit_flag);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_nz2nd_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                         memref_t<__gm__ DST_TYPE, 2> *gm,
                                         int64_t pre_quant,
                                         float32_t quant_scale,
                                         int64_t pre_relu, bool channel_split,
                                         uint8_t unit_flag);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_nz2nd_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                           memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
                                           int64_t pre_quant,
                                           float32_t quant_scale,
                                           int64_t pre_relu, bool channel_split,
                                           uint8_t unit_flag);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_cbuf_nz2nd_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                           memref_t<__cbuf__ DST_TYPE, 2> *cbuf,
                                           int64_t pre_quant,
                                           float32_t quant_scale,
                                           int64_t pre_relu, bool channel_split,
                                           uint8_t unit_flag);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_nz2dn_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                         memref_t<__ubuf__ DST_TYPE, 2> *gm,
                                         int64_t pre_quant,
                                         float32_t quant_scale,
                                         int64_t pre_relu, bool channel_split,
                                         uint8_t unit_flag);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_nz2dn_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                           memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
                                           int64_t pre_quant,
                                           float32_t quant_scale,
                                           int64_t pre_relu, bool channel_split,
                                           uint8_t unit_flag);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_cbuf_nz2dn_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                           memref_t<__cbuf__ DST_TYPE, 2> *cbuf,
                                           int64_t pre_quant,
                                           float32_t quant_scale,
                                           int64_t pre_relu, bool channel_split,
                                           uint8_t unit_flag);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_normal_2d_to_2d_core(
    memref_t<__cc__ SRC_TYPE, 2> *l0c, memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
    int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,
    bool channel_split, uint8_t unit_flag, uint8_t dual_dst = 0);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_ubuf_nz2nd_4d_to_2d_core(
    memref_t<__cc__ SRC_TYPE, 4> *l0c, memref_t<__ubuf__ DST_TYPE, 2> *ubuf,
    int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,
    bool channel_split, uint8_t unit_flag, uint8_t dual_dst = 0);

#define DECLARE_FIXPIPE(src_scope, dst_scope, src_dim, dst_dim, src_type,                                         \
                        dst_type, mode_name, transform_mode)                                                      \
  __aicore__ __attribute__((always_inline)) void                                                                  \
      _mlir_ciface_fixpipe_##mode_name##_##src_type##_to_##dst_type##_##src_dim##d_to_##dst_dim##d##_##dst_scope( \
          memref_t<__##src_scope##__ src_type, src_dim> *src,                                                     \
          memref_t<__##dst_scope##__ dst_type, dst_dim> *dst,                                                     \
          int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,                                             \
          bool channel_split, uint8_t unit_flag)

#define REGISTE_FIXPIPE(src_scope, dst_scope, src_dim, dst_dim, src_type,         \
                        dst_type, mode_name, transform_mode)                      \
  DECLARE_FIXPIPE(src_scope, dst_scope, src_dim, dst_dim, src_type, dst_type,     \
                  mode_name, transform_mode) {                                    \
    copy_matrix_##src_scope##_to_##dst_scope##_##src_dim##d_to_##dst_dim##d_core< \
        src_type, dst_type, transform_mode>(                                      \
        src, dst, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);    \
  }

#define DECLARE_FIXPIPE_NOSUFFIX(src_scope, dst_scope, src_dim, dst_dim,                            \
                                 src_type, dst_type, mode_name,                                     \
                                 transform_mode)                                                    \
  __aicore__ __attribute__((always_inline)) void                                                    \
      _mlir_ciface_fixpipe_##mode_name##_##src_type##_to_##dst_type##_##src_dim##d_to_##dst_dim##d( \
          memref_t<__##src_scope##__ src_type, src_dim> *src,                                       \
          memref_t<__##dst_scope##__ dst_type, dst_dim> *dst,                                       \
          int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,                               \
          bool channel_split, uint8_t unit_flag, uint8_t dual_dst)

#define REGISTE_FIXPIPE_NOSUFFIX(src_scope, dst_scope, src_dim, dst_dim,                                        \
                                 src_type, dst_type, mode_name,                                                 \
                                 transform_mode)                                                                \
  DECLARE_FIXPIPE_NOSUFFIX(src_scope, dst_scope, src_dim, dst_dim, src_type,                                    \
                           dst_type, mode_name, transform_mode) {                                               \
    _mlir_ciface_fixpipe_##mode_name##_##src_type##_to_##dst_type##_##src_dim##d_to_##dst_dim##d##_##dst_scope( \
        src, dst, pre_quant, quant_scale, pre_relu, channel_split, unit_flag);                                  \
  }

#define DECLARE_FIXPIPE_DUAL(src_scope, dst_scope, src_dim, dst_dim, src_type,                                         \
                             dst_type, mode_name, transform_mode)                                                      \
  __aicore__ __attribute__((always_inline)) void                                                                       \
      _mlir_ciface_fixpipe_##mode_name##_dual_##src_type##_to_##dst_type##_##src_dim##d_to_##dst_dim##d##_##dst_scope( \
          memref_t<__##src_scope##__ src_type, src_dim> *src,                                                          \
          memref_t<__##dst_scope##__ dst_type, dst_dim> *dst,                                                          \
          int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,                                                  \
          bool channel_split, uint8_t unit_flag, uint8_t dual_dst)

#define REGISTE_FIXPIPE_DUAL(src_scope, dst_scope, src_dim, dst_dim, src_type,      \
                             dst_type, mode_name, transform_mode)                   \
  DECLARE_FIXPIPE_DUAL(src_scope, dst_scope, src_dim, dst_dim, src_type,            \
                       dst_type, mode_name, transform_mode) {                       \
    copy_matrix_##src_scope##_to_##dst_scope##_##src_dim##d_to_##dst_dim##d##_core< \
        src_type, dst_type, transform_mode>(src, dst, pre_quant, quant_scale,       \
                                            pre_relu, channel_split,                \
                                            unit_flag, dual_dst);                   \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// fixpipe, 2 dim to 2 dim, normal
//===-------------------------------------------------------------------===//
DECLARE_FIXPIPE(cc, gm, 2, 2, float, half, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, gm, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE_NOSUFFIX(cc, gm, 2, 2, float, half, normal,
                         TransformMode::NORMAL);
DECLARE_FIXPIPE_NOSUFFIX(cc, gm, 2, 2, float, bfloat16_t, normal,
                         TransformMode::NORMAL);

//===-------------------------------------------------------------------===//
// fixpipe, 4 dim to 2 dim, nz2nd
//===-------------------------------------------------------------------===//
DECLARE_FIXPIPE(cc, gm, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND);

DECLARE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, float, half, nz2nd,
                         TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, float, bfloat16_t, nz2nd,
                         TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, float, float, nz2nd,
                         TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, int32_t, int32_t, nz2nd,
                         TransformMode::NZ_2_ND);

#if !defined(__DAV_M300__)
DECLARE_FIXPIPE(cc, gm, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND);

DECLARE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND);
#if !defined(__DAV_C310__)
DECLARE_FIXPIPE(cc, gm, 4, 2, int32_t, int16_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_NOSUFFIX(cc, gm, 4, 2, int32_t, int16_t, nz2nd, TransformMode::NZ_2_ND);
#endif // !defined(__DAV_C310__)
#endif // !defined(__DAV_M300__)

#if defined(__DAV_M300__) || defined(__DAV_C310__)
DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, int8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, uint8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND);

DECLARE_FIXPIPE(cc, ubuf, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, int32_t, uint8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND);
#endif // defined(__DAV_M300__)

#if defined(__DAV_C310__)
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, int8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, uint8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND);

DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, uint8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND);

DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, half, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, bfloat16_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, int8_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, uint8_t, nz2dn, TransformMode::NZ_2_DN);

DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, float, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, half, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, int8_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, uint8_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, int32_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, cbuf, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, cbuf, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, cbuf, 4, 2, float, int8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, cbuf, 4, 2, float, uint8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, cbuf, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND);

DECLARE_FIXPIPE(cc, cbuf, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, cbuf, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, cbuf, 4, 2, int32_t, uint8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, cbuf, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND);

//===-------------------------------------------------------------------===//
// fixpipe, 2 dim to 2 dim, nz2nz normal
//===-------------------------------------------------------------------===//
DECLARE_FIXPIPE(cc, ubuf, 2, 2, float, half, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, ubuf, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL);

DECLARE_FIXPIPE_DUAL(cc, ubuf, 2, 2, float, half, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL);

DECLARE_FIXPIPE(cc, cbuf, 2, 2, float, half, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, cbuf, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL);

//===-------------------------------------------------------------------===//
// fixpipe, 4 dim to 4 dim, nz2nz normal
//===-------------------------------------------------------------------===//
DECLARE_FIXPIPE(cc, gm, 4, 4, float, half, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, gm, 4, 4, float, bfloat16_t, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, gm, 4, 4, float, float, normal, TransformMode::NORMAL);

DECLARE_FIXPIPE(cc, ubuf, 4, 4, float, half, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, ubuf, 4, 4, float, bfloat16_t, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, ubuf, 4, 4, float, float, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, ubuf, 4, 4, int32_t, int32_t, normal,
                TransformMode::NORMAL);

DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 4, float, float, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE_DUAL(cc, ubuf, 4, 4, int32_t, int32_t, normal, TransformMode::NORMAL);

DECLARE_FIXPIPE(cc, cbuf, 4, 4, float, half, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, cbuf, 4, 4, float, bfloat16_t, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, cbuf, 4, 4, float, float, normal, TransformMode::NORMAL);

//===-------------------------------------------------------------------===//
// fixpipe, 4 dim to 2 dim, nz2dn
//===-------------------------------------------------------------------===//
DECLARE_FIXPIPE(cc, gm, 4, 2, float, half, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, gm, 4, 2, float, bfloat16_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, gm, 4, 2, float, float, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, gm, 4, 2, int32_t, int32_t, nz2dn, TransformMode::NZ_2_DN);

DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, half, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, bfloat16_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, int8_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, uint8_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, float, float, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, int32_t, half, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, int32_t, int8_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, int32_t, uint8_t, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, ubuf, 4, 2, int32_t, int32_t, nz2dn, TransformMode::NZ_2_DN);

DECLARE_FIXPIPE(cc, cbuf, 4, 2, float, half, nz2dn, TransformMode::NZ_2_DN);
DECLARE_FIXPIPE(cc, cbuf, 4, 2, float, bfloat16_t, nz2dn, TransformMode::NZ_2_DN);
#endif // defined(__DAV_C310__)
}
#endif // HIVM_MLIR_TEMPLATE_FIXPIPE_UTILS_H
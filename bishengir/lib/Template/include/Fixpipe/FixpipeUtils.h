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
  NC1HWC0_2_NHWC = 2,
};

template <typename SRC_TYPE, typename DST_TYPE>
struct copy_matrix_cc_to_gm_intrin_args {
  __gm__ DST_TYPE *dst;
  __cc__ SRC_TYPE *src;
  uint8_t sid;
  uint16_t n_size;
  uint16_t m_size;
  uint32_t dst_D;
  uint16_t src_stride;
  uint8_t unit_flag_mode;
  QuantMode_t pre_quant;
  uint8_t pre_relu;
  bool channel_split;
  bool nz2nd_en;
};

__aicore__ __attribute__((always_inline)) QuantMode_t
get_quant_mode(int64_t pre_quant) {
  if (pre_quant == static_cast<int64_t>(QuantMode_t::F322F16)) {
    return QuantMode_t::F322F16;
  }
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
#define FIXPIPE_ARGS                                                           \
  args.dst, args.src, args.sid, args.n_size, args.m_size, args.dst_D,          \
      args.src_stride, args.unit_flag_mode, args.pre_quant, args.pre_relu,     \
      args.channel_split, args.nz2nd_en

  INTRINSIC(copy_matrix_cc_to_gm, FIXPIPE_ARGS);
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_normal_2d_to_2d_core(memref_t<__cc__ SRC_TYPE, 2> *l0c,
                                          memref_t<__gm__ DST_TYPE, 2> *gm,
                                          int64_t pre_quant, int64_t pre_relu,
                                          bool channel_split,
                                          uint8_t unit_flag);

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
copy_matrix_cc_to_gm_nz2nd_4d_to_2d_core(memref_t<__cc__ SRC_TYPE, 4> *l0c,
                                         memref_t<__gm__ DST_TYPE, 2> *gm,
                                         int64_t pre_quant, int64_t pre_relu,
                                         bool channel_split, uint8_t unit_flag);

#define DECLARE_FIXPIPE(src_scope, dst_scope, src_dim, dst_dim, src_type,                           \
                        dst_type, mode_name, transform_mode)                                        \
  __aicore__ __attribute__((always_inline)) void                                                    \
      _mlir_ciface_fixpipe_##mode_name##_##src_type##_to_##dst_type##_##src_dim##d_to_##dst_dim##d( \
          memref_t<__##src_scope##__ src_type, src_dim> *src,                                       \
          memref_t<__##dst_scope##__ dst_type, dst_dim> *dst,                                       \
          int64_t pre_quant, int64_t pre_relu, bool channel_split,                                  \
          uint8_t unit_flag)

#define REGISTE_FIXPIPE(src_scope, dst_scope, src_dim, dst_dim, src_type,         \
                        dst_type, mode_name, transform_mode)                      \
  DECLARE_FIXPIPE(src_scope, dst_scope, src_dim, dst_dim, src_type, dst_type,     \
                  mode_name, transform_mode) {                                    \
    copy_matrix_##src_scope##_to_##dst_scope##_##src_dim##d_to_##dst_dim##d_core< \
        src_type, dst_type, transform_mode>(src, dst, pre_quant, pre_relu,        \
                                            channel_split, unit_flag);            \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// fixpipe, 2 dim to 2 dim, normal
//===-------------------------------------------------------------------===//
DECLARE_FIXPIPE(cc, gm, 2, 2, float, half, normal, TransformMode::NORMAL);
DECLARE_FIXPIPE(cc, gm, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL);

//===-------------------------------------------------------------------===//
// fixpipe, 4 dim to 2 dim, nz2nd
//===-------------------------------------------------------------------===//
DECLARE_FIXPIPE(cc, gm, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, int32_t, int16_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND);
DECLARE_FIXPIPE(cc, gm, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND);
}
#endif // HIVM_MLIR_TEMPLATE_FIXPIPE_UTILS_H
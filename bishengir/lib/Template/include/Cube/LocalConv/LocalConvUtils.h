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

#ifndef HIVM_MLIR_TEMPLATE_CONV2D_GROUP_UTILS_H
#define HIVM_MLIR_TEMPLATE_CONV2D_GROUP_UTILS_H
#include "../Utils.h"

#define DECLARE_CONV2D_GROUP(src_scope, dst_scope, dim, src_type, dst_type)    \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_conv2d_group_##src_type##_to_##dst_type(                    \
          memref_t<__##src_scope##__ src_type, dim> *src0,                     \
          memref_t<__##src_scope##__ src_type, dim> *src1, bool init,          \
          memref_t<__##dst_scope##__ dst_type, 4> *dst, int64_t G,             \
          int64_t padT, int64_t padB, int64_t padL, int64_t padR,              \
          int64_t strideH, int64_t strideW, int64_t dilationH,                 \
          int64_t dilationW, int64_t conv_l1_wait_l1a_event,                   \
          int64_t conv_l1_wait_l1b_event, int64_t l1a_wait_conv_l1_event,      \
          int64_t l1b_wait_conv_l1_event,                                      \
          int64_t back_pipe_m_pipe_mte1_db_event0,                             \
          int64_t back_pipe_m_pipe_mte1_db_event1)

#define REGISTER_CONV2D_GROUP(src_scope, dst_scope, dim, src_type, dst_type)   \
  DECLARE_CONV2D_GROUP(src_scope, dst_scope, dim, src_type, dst_type) {        \
    conv2d_group<src_type, dst_type>(                                          \
        src0, src1, init, dst, G, padT, padB, padL, padR, strideH, strideW,    \
        dilationH, dilationW, conv_l1_wait_l1a_event, conv_l1_wait_l1b_event,  \
        l1a_wait_conv_l1_event, l1b_wait_conv_l1_event,                        \
        back_pipe_m_pipe_mte1_db_event0, back_pipe_m_pipe_mte1_db_event1);     \
  }

extern "C" {
DECLARE_CONV2D_GROUP(cbuf, cc, 5, float, float);
DECLARE_CONV2D_GROUP(cbuf, cc, 5, half, float);
DECLARE_CONV2D_GROUP(cbuf, cc, 5, bfloat16_t, float);
}
#endif // HIVM_MLIR_TEMPLATE_CONV2D_GROUP_UTILS_H

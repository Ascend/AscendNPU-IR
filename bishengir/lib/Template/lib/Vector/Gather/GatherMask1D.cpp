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

#include "Vector/Gather/GatherMaskUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
gather_mask_1d(memref_t<__ubuf__ T, 1> *src,
               memref_t<__ubuf__ bool, 1> *mask,
               memref_t<__ubuf__ T, 1> *dst,
               memref_t<__ubuf__ int64_t, 1> *dst_size)
{
    int64_t element_count = src->sizes[0];
    int64_t mask_count = mask->sizes[0];

    if (element_count <= 0 || element_count != mask_count) {
        INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
        return;
    }

    __ubuf__ T *src_ptr = src->aligned + src->offset;
    __ubuf__ bool *mask_ptr = mask->aligned + mask->offset;
    __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
    __ubuf__ int64_t *dst_size_ptr = dst_size->aligned + dst_size->offset;

    constexpr bool is_16bit = (sizeof(T) == 2);
    constexpr bool is_32bit = (sizeof(T) == 4);

    static_assert(is_16bit || is_32bit, "Only 16-bit or 32-bit types are supported");

    if constexpr (is_16bit) {
        INTRINSIC_NO_ARGS(set_mask_count);
        INTRINSIC(set_vector_mask, 0, element_count);

        INTRINSIC(vreducev2,
                  (__ubuf__ uint16_t*)dst_ptr,
                  (__ubuf__ uint16_t*)src_ptr,
                  (__ubuf__ uint16_t*)mask_ptr,
                  static_cast<uint16_t>(1),
                  static_cast<uint8_t>(1),
                  static_cast<uint8_t>(PatternMode::NORMAL_MODE),
                  static_cast<uint16_t>(8),
                  static_cast<uint8_t>(1));

    } else if constexpr (is_32bit) {
        INTRINSIC_NO_ARGS(set_mask_count);
        INTRINSIC(set_vector_mask, 0, element_count);

        INTRINSIC(vreducev2,
                  (__ubuf__ uint32_t*)dst_ptr,
                  (__ubuf__ uint32_t*)src_ptr,
                  (__ubuf__ uint32_t*)mask_ptr,
                  static_cast<uint16_t>(1),
                  static_cast<uint8_t>(1),
                  static_cast<uint8_t>(PatternMode::NORMAL_MODE),
                  static_cast<uint16_t>(8),
                  static_cast<uint8_t>(1));
    }

    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    *(__ubuf__ int64_t *)(dst_size_ptr) = get_rsvd_cnt();
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
 
    INTRINSIC_NO_ARGS(set_mask_norm);
    INTRINSIC(pipe_barrier, PIPE_V);
}

extern "C" {
REGISTE_GATHER_MASK(1, uint16_t)
REGISTE_GATHER_MASK(1, uint32_t)
REGISTE_GATHER_MASK(1, int16_t)
REGISTE_GATHER_MASK(1, int32_t)
REGISTE_GATHER_MASK(1, half)
REGISTE_GATHER_MASK(1, float)
REGISTE_GATHER_MASK(1, bfloat16_t)
}
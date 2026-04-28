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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_DEINTERLEAVE_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_DEINTERLEAVE_UTILS_H

#include "Utils.h"
#include "Vector/VecUtils.h"
#include <type_traits>

enum class DeinterleaveMode : uint32_t {
  // TODO merge channel_from_2 to channel_from_N
  CHANNEL_0_FROM_2_CHANNELS = 0,
  CHANNEL_1_FROM_2_CHANNELS = 1,
  CHANNEL_0_FROM_N_CHANNELS = 3,
};

template <typename T, DeinterleaveMode deinterleave_mode>
__aiv__ __attribute__((always_inline)) void
vector_deinterleave_1d(memref_t<__ubuf__ T, 1> *src,
                       memref_t<__ubuf__ T, 1> *dst);

#define DECLARE_DEINTERLEAVE(mode_name, deinterleave_mode, dim, dtype)         \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_deinterleave_##mode_name##_##dim##d_##dtype(                \
          memref_t<__ubuf__ dtype, dim> *src,                                  \
          memref_t<__ubuf__ dtype, dim> *dst)

#define REGISTER_DEINTERLEAVE(mode_name, deinterleave_mode, dim, dtype)        \
  DECLARE_DEINTERLEAVE(mode_name, deinterleave_mode, dim, dtype) {             \
    vector_deinterleave_##dim##d<deinterleave_mode, dtype>(src, dst);          \
  }

extern "C" {
//===-------------------------------------------------------------------===//
// deinterleave, 1 dim
//===-------------------------------------------------------------------===//
DECLARE_DEINTERLEAVE(channel_0_from_2_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, half);
DECLARE_DEINTERLEAVE(channel_0_from_2_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1,
                     bfloat16_t);
DECLARE_DEINTERLEAVE(channel_0_from_2_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, float);
DECLARE_DEINTERLEAVE(channel_0_from_2_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, int16_t);
DECLARE_DEINTERLEAVE(channel_0_from_2_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, int32_t);
DECLARE_DEINTERLEAVE(channel_0_from_2_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, uint16_t);
DECLARE_DEINTERLEAVE(channel_0_from_2_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_2_CHANNELS, 1, uint32_t);

DECLARE_DEINTERLEAVE(channel_1_from_2_channels,
                     DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, half);
DECLARE_DEINTERLEAVE(channel_1_from_2_channels,
                     DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1,
                     bfloat16_t);
DECLARE_DEINTERLEAVE(channel_1_from_2_channels,
                     DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, float);
DECLARE_DEINTERLEAVE(channel_1_from_2_channels,
                     DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, int16_t);
DECLARE_DEINTERLEAVE(channel_1_from_2_channels,
                     DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, int32_t);
DECLARE_DEINTERLEAVE(channel_1_from_2_channels,
                     DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, uint16_t);
DECLARE_DEINTERLEAVE(channel_1_from_2_channels,
                     DeinterleaveMode::CHANNEL_1_FROM_2_CHANNELS, 1, uint32_t);

//===-------------------------------------------------------------------===//
// deinterleave, 2 dim
//===-------------------------------------------------------------------===//
DECLARE_DEINTERLEAVE(channel_0_from_n_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, half);
DECLARE_DEINTERLEAVE(channel_0_from_n_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2,
                     bfloat16_t);
DECLARE_DEINTERLEAVE(channel_0_from_n_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, float);
DECLARE_DEINTERLEAVE(channel_0_from_n_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, int16_t);
DECLARE_DEINTERLEAVE(channel_0_from_n_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, int32_t);
DECLARE_DEINTERLEAVE(channel_0_from_n_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, uint16_t);
DECLARE_DEINTERLEAVE(channel_0_from_n_channels,
                     DeinterleaveMode::CHANNEL_0_FROM_N_CHANNELS, 2, uint32_t);
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_DEINTERLEAVE_UTILS_H
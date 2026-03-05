/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#pragma once

#include "Macro/common.h"

template <ArchType ArchTag, typename ElementA, typename ElementB, bool WithBias>
struct L1BufferOffset;

template <> struct L1BufferOffset<ArchType::ASCEND_V220, int8_t, int8_t, true> {
  static const uint32_t L1_BIAS_OFFSET = 0;
  static const uint32_t L1_SCALE_OFFSET = 2048;
  static const uint32_t L1_A_OFFSET = 6144;
};

template <>
struct L1BufferOffset<ArchType::ASCEND_V220, int8_t, int8_t, false> {
  static const uint32_t L1_BIAS_OFFSET = 0;
  static const uint32_t L1_SCALE_OFFSET = 0;
  static const uint32_t L1_A_OFFSET = 4096;
};

template <> struct L1BufferOffset<ArchType::ASCEND_V220, half, half, true> {
  static const uint32_t L1_BIAS_OFFSET = 0;
  static const uint32_t L1_SCALE_OFFSET = 2048;
  static const uint32_t L1_A_OFFSET = 2048;
};

template <> struct L1BufferOffset<ArchType::ASCEND_V220, half, half, false> {
  static const uint32_t L1_BIAS_OFFSET = 0;
  static const uint32_t L1_SCALE_OFFSET = 0;
  static const uint32_t L1_A_OFFSET = 0;
};

// BiasTableOffset
template <ArchType ArchTag> struct BiasTableOffset;

template <> struct BiasTableOffset<ArchType::ASCEND_V220> {
  static const uint32_t BIAS_BT_OFFSET = 0;
};
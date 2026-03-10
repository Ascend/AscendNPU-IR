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

#define CONST_2 2

enum class CubeFormat : uint32_t { ND = 0, NZ, ZN, ZZ, NN, VECTOR };

enum class ArchType { ASCEND_V220, ASCEND_V200, ASCEND_M200 };

template <ArchType ArchTag> struct HardwareInfo;

template <> struct HardwareInfo<ArchType::ASCEND_V220> {
  static uint32_t const l2BW = 5;
  static uint32_t const hbmBW = 1;
  static uint32_t const supportMix = 0;
  static uint32_t const l1Size = 512 * 1024;
  static uint32_t const l0ASize = 64 * 1024;
  static uint32_t const l0BSize = 64 * 1024;
  static uint32_t const l0CSize = 128 * 1024;
  static uint32_t const l2Size = 192 * 1024 * 1024;
  static uint32_t const biasSize = 1024;
  static uint32_t const fixBufSize = 7 * 1024;
  static uint32_t const ubSize = 192 * 1024;
  static uint32_t const fractalSize = 512;
  static uint32_t const l1l0BlockSize = 32;
  static uint32_t const btBlockSize = 64;
  static uint32_t const fbBlockSize = 128;
};

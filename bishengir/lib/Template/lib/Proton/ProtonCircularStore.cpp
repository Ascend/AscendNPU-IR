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

#include "Utils.h"

namespace {
constexpr uint32_t kRecordWords = 2;
constexpr int64_t kOffCountVector = 10;
constexpr int64_t kOffDataSegment = 11;
constexpr uint32_t kEndBit = 0x80000000u;
constexpr uint32_t kScopeIdMask = 0xFFu;
constexpr uint32_t kScopeIdShift = 23;
constexpr uint64_t kCycleUpperMask = 0x7FFu;
} // namespace

extern "C" {
__aiv__ __attribute__((always_inline)) void
proton_circular_store(memref_t<__gm__ int32_t, 1> *buffer,
                      int64_t section_offset, int64_t clock, int32_t scope_id,
                      int32_t is_start, int32_t data_segment_words) {
  __gm__ int32_t *buffer_ptr = buffer->aligned + buffer->offset;
  const int64_t stride = buffer->strides[0];
  const int64_t count_index = (section_offset + kOffCountVector) * stride;
  const uint32_t write_pointer = static_cast<uint32_t>(buffer_ptr[count_index]);
  const uint32_t physical_word =
      write_pointer % static_cast<uint32_t>(data_segment_words);
  const int64_t record_index =
      (section_offset + kOffDataSegment + physical_word) * stride;

  const uint64_t cycle = static_cast<uint64_t>(clock);
  const uint32_t tag =
      (is_start == 0 ? kEndBit : 0u) |
      ((static_cast<uint32_t>(scope_id) & kScopeIdMask) << kScopeIdShift) |
      static_cast<uint32_t>((cycle >> 32) & kCycleUpperMask);

  buffer_ptr[record_index] = static_cast<int32_t>(tag);
  buffer_ptr[record_index + stride] = static_cast<int32_t>(cycle);
  buffer_ptr[count_index] = static_cast<int32_t>(write_pointer + kRecordWords);
}
}

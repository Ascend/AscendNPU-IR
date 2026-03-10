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

struct TilingParams {
  uint32_t b{0};
  uint32_t m{0};
  uint32_t k{0};
  uint32_t n{0};
  uint32_t mTensorA{0};
  uint32_t kTensorA{0};
  uint32_t nTensorB{0};
  uint32_t kTensorB{0};
  uint32_t mTile{0};
  uint32_t kTile{0};
  uint32_t nTile{0};
  uint32_t mLoops{0};
  uint32_t nLoops{0};
  uint32_t splitKSlices{1}; // Indicates the number of partition alone K
                            // dimension while enabling split-k.
  uint32_t swizzleDir{0};
  uint32_t swizzleCnt{0};
  uint32_t shuffleKType{0};
  uint32_t tilingKey{0};
  uint8_t reserved[512 - 18 * sizeof(uint32_t)];
};
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

#include "Synchronization/SyncUtils.h"

__aicore__ uint8_t &getUnitFlagIsBadRef(int64_t unit_flag_group_id) {
  static uint8_t unit_flag_is_bad[MAX_BLOCK_NUM][MAX_UNIT_FLAG_GROUP_ID] = {0};
  return unit_flag_is_bad[get_block_idx() % MAX_BLOCK_NUM][unit_flag_group_id];
}

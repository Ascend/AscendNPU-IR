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

#include "Vector/Reduction/ReduceRAImpl.h"

extern "C" {
//===-------------------------------------------------------------------===//
// reduce ra, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, half)
REGISTE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, float)
REGISTE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_sum, ReduceOpTy::REDUCE_SUM, 2, int16_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, half)
REGISTE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, float)
REGISTE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_max, ReduceOpTy::REDUCE_MAX, 2, int16_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, half)
REGISTE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, float)
REGISTE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_min, ReduceOpTy::REDUCE_MIN, 2, int16_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, half)
REGISTE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, float)
REGISTE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_prod, ReduceOpTy::REDUCE_PROD, 2, int16_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_xori, ReduceOpTy::REDUCE_XOR, 2, int64_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, int64_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_ori, ReduceOpTy::REDUCE_OR, 2, uint64_t)

REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint8_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint16_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint32_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, int64_t)
REGISTE_ENTIRE_REDUCE_RA(reduce_andi, ReduceOpTy::REDUCE_AND, 2, uint64_t)
}

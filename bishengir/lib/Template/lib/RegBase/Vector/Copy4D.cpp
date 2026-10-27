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

#include "ArchDefs.h"
#include "RegBase/DMAUtils.h"

// 4D GM->UB loads exist only where NDDMA does: the descriptor is the
// only DMA form that walks a permuted tile of this rank.
// load_gm_to_ubuf_nd_core in RegBase/DMAUtils.h holds the implementation; this
// file only registers the entry points. Loads that do not need a descriptor
// keep the 3D templates (see InferMaxRankExternalModel<LoadOp>), so they never
// arrive here.
#if BISHENG_TPL_REGBASE

extern "C" {

//===-------------------------------------------------------------------===//
// Load gm to ub, 4 dim
//===-------------------------------------------------------------------===//
REGISTE_DMA_LOAD(4, int8_t);
REGISTE_DMA_LOAD(4, uint8_t);
REGISTE_DMA_LOAD(4, int16_t);
REGISTE_DMA_LOAD(4, uint16_t);
REGISTE_DMA_LOAD(4, int32_t);
REGISTE_DMA_LOAD(4, uint32_t);
REGISTE_DMA_LOAD(4, int64_t);
REGISTE_DMA_LOAD(4, uint64_t);
REGISTE_DMA_LOAD(4, half);
REGISTE_DMA_LOAD(4, float);
REGISTE_DMA_LOAD(4, bfloat16_t);

REGISTE_DMA_LOAD_FP8(4, float8_e4m3_t);
REGISTE_DMA_LOAD_FP8(4, float8_e5m2_t);
}

#endif

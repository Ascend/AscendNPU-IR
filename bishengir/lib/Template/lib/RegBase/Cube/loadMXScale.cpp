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

#include "DMA/loadMXScale.h"
#include "Vector/VecUtils.h"

#if defined(__DAV_C310__)

/**
 Loading Scale Tensor from GM to L1 for `dot_scale` op. Only considered
 fp8e8m0(i8) for now. In dot_scale, for A tensor of shape (M, K), the scaleA
 tensor must be (M, K//32). Assume M=32, K=128, then

 ScaleA In GM:   (fp8e8m0, 32x4)
 0,   1,   2,   3,
 4,   5,   6,   7,
 8,   9,   10,  11,
 ...
 124, 125, 126, 127

 Target ScaleA in L1: (fp8e8m0, 32x4, zZ)
 0,   1, | 2,   3,
 4,   5, | 6,   7,
 8,   9, | 10,  11,
 ...,
 60,  61,| 62,  63,
 --------+--------
 64,  65,| 66,  67,
 68,  69,| 70,  71,
 ...
 124, 125| 126, 127

 To perform this load,
 Step 1. wrap 2 fp8e8m0 as one fp16. scaleA as (fp16, 32x2):
 0,  0,  2,  2
 4,  4,  6,  6
 8,  8,  10, 10
 ...
 124, 124, 126, 126

 Step 2. load dn2nz.fp16 with d = 32, n = 2, (zN with each fractil is 1x16xf16)
 0, 0, 4, 4, 8, 8, ..., 60, 60, | 64, 64, 68, 68, ... 124, 124
 -------------------------------+------------------------------
 2, 2, 6, 6, 10,10,..., 62, 62, | 66, 66, 70, 70, ... 126, 126

 Step 3. View as nZ with each fractil is 16x1xf16
 0,  0, | 2, 2
 4,  4, | 6, 6
 8,  8, | 10,10
 ...    | ...
 60, 60,| 62, 62
 -------+-------
 64, 64 | 66, 66
 68, 68 | 70, 70
 ...    | ...
 124,124| 126,126

 Step 4. Unwrap each fp16 as two fp8e8m0
 0,   1, | 2,   3,
 4,   5, | 6,   7,
 8,   9, | 10,  11,
 ...,
 60,  61,| 62,  63,
 --------+--------
 64,  65,| 66,  67,
 68,  69,| 70,  71,
 ...
 124, 125| 126, 127

 Therefore, scaleA is loaded on l1 as zZ. For scaleB of (N, K//32), it can also
 be loaded as nN under this process.

 Since the process requires two fp8e8m0 be wrapped as one fp16, it requires
 scale to be continuous on K//32 dim.

 */

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_gm_to_cbuf_load_mx_scale_core(memref_t<__gm__ T, 2> *gm,
                                   memref_t<__cbuf__ T, 4> *l1) {
  auto gm_ptr = gm->aligned + gm->offset;
  auto l1_ptr = l1->aligned + l1->offset;
  int64_t d_tile_actual = gm->sizes[0];
  int64_t n_tile_actual = gm->sizes[1];
  constexpr const uint32_t MX_SCALE_COPY_GROUP_NUM = sizeof(half) / sizeof(T);
  uint64_t config =
      ((uint64_t)1) |         // Is the DN matrix number to be moved.
      ((uint64_t)(1)) << 16 | // Is the destination stride of loop2 in unit of
                              // C0 size, loop2 dst stride.
      ((uint64_t)(n_tile_actual / MX_SCALE_COPY_GROUP_NUM))
          << 32 | // Is the destination stride of loop3 in unit of C0 size,
                  // loop3 dst stride.
      ((uint64_t)(0)) << 48; // Is the destination stride of loop4 in unit of C0
                             // size, loop4 dst stride
  INTRINSIC(set_mte2_nz_para, config);
  INTRINSIC(
      copy_gm_to_cbuf_multi_dn2nz, /*dst_ptr*/ (__cbuf__ half *)l1_ptr,
      /*src_ptr*/ (__gm__ half *)gm_ptr,
      /*uint8_t sid*/ 0,
      /*uint64_t loop1_src_stride*/ gm->strides[0],
      /*uint8_t l2_cache_ctrl_mode*/ 0,
      /*nValue*/ static_cast<uint16_t>(n_tile_actual / MX_SCALE_COPY_GROUP_NUM),
      /*dValue*/ static_cast<uint16_t>(d_tile_actual),
      /* loop4 src stride */ 0,
      /*bool smallc0_en*/ false);
}

extern "C" {
REGISTE_LOAD_MX_SCALE(gm, cbuf, 2, 4, int8_t);
}
#endif

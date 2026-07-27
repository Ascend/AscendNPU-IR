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
 * Loading Scale Tensor from GM to L1 for `dot_scale` op.
 * Only fp8e8m0 (i8) is considered for now.
 *
 * Layout definitions used in this process:
 *   nd  – normal row‑major 2‑D layout.
 *   dn  – transposed nd (column‑major).
 *   Zz  – fractal layout: each fractal is 16×2 row‑major (small z),
 *         all fractals are row‑major (big Z).
 *   Nz  – fractal layout: each fractal is k×32byte row‑major (small z),
 *         all fractals are column‑major (big N).
 *         For an fp16 tensor originally in dn of shape (b, a),
 *         Nz shape is (a/16, b/k, k, 16) for arbitrary k,
 *         or the 3‑D version (a/16, b, 16) when k = b.
 *
 * In dot_scale, for A tensor of shape (M, K), scaleA must be (M, K//32).
 * Example: M=32, K=128  =>  scaleA is 32×4 i8.
 *
 * Source in GM (nd, 32×4 i8):
 *   0,   1,   2,   3,
 *   4,   5,   6,   7,
 *   ...
 *   124, 125, 126, 127
 *
 * Target in L1 (Zz, 32×4 i8, i.e. 2×2 fractals of 16×2):
 *   0,   1, | 2,   3,
 *   4,   5, | 6,   7,
 *   ...
 *   60,  61,| 62,  63,
 *   --------+--------
 *   64,  65,| 66,  67,
 *   ...
 *   124, 125| 126, 127
 *
 * There is no direct nd→Zz instruction. We perform nd→Zz via dn2nz.f16:
 *
 * Step 1. Wrap two fp8e8m0 as one fp16.
 *   The 32×4 i8 row‑major tensor is viewed as a 32×2 fp16 row‑major tensor.
 *   Pairs (0,1) → fp16 "0,0", (2,3) → "2,2", etc.
 *   Result (fp16, 32×2 row‑major):
 *     0, 0,   2, 2
 *     4, 4,   6, 6
 *     ...
 *     124,124, 126,126
 *
 * Step 2. Reinterpret as dn and load with dn2nz.f16 (k = 2, i.e. the 3‑D Nz version).
 *   The 32×2 row‑major fp16 tensor is physically identical to a 2×32 column‑major
 *   (dn) fp16 tensor. We pass this dn tensor to dn2nz.f16 with k = 2.
 *   The resulting Nz layout in L1 has shape (a/16=2, b=2, 16):
 *     - Each fractal is 2×16 row‑major (holds both columns of a 16‑row tile).
 *     - The grid of fractals is 2×1, traversed column‑major (i.e. just tile 0,
 *       then tile 1).
 *   Memory order (shown as two fractals side‑by‑side, each 2 rows × 16 cols):
 *
 *     tile 0, col0 (rows 0..15):   0,0  4,4 ... 60,60 | tile 1, col0 (rows 16..31): 64,64 ... 124,124
 *     tile 0, col1 (rows 0..15):   2,2  6,6 ... 62,62 | tile 1, col1 (rows 16..31): 66,66 ... 126,126
 *
 *     (in memory: top‑left → bottom‑left → top‑right → bottom‑right)
 *
 * Step 3. Reinterpret the memory view as Zz (fp16).
 *   The physical byte order is already exactly the desired Zz order.
 *   We logically change the tensor strides to view it as a 2×2 grid of
 *   16‑element blocks, row‑major (big Z), where each block is one fp16 column
 *   of the original tile.
 *
 *   Reinterpreted view:
 *     0,0  4,4 ... 60,60 | 2,2  6,6 ... 62,62
 *     -------------------+--------------------
 *     64,64 ... 124,124  | 66,66 ... 126,126
 *     (memory order: top‑left → top‑right → bottom‑left → bottom‑right)
 *
 * Step 4. Unwrap each fp16 back to two fp8e8m0.
 *   Each fp16 value is reinterpreted as two i8 values in the original order,
 *   yielding the target 32×4 i8 Zz layout exactly as shown above.
 *
 * The same method works for scaleB of shape (N, K//32), loading it as nN layout.
 *
 * Note: The process requires the scale to be contiguous along the K//32 dimension,
 * because adjacent bytes are paired into a single fp16.
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
  // Loop3 dst stride = C0 distance between consecutive outer L1 tiles
  // (tile0-col0 → tile1-col0). Use physical L1 strides, not logical n/2, so a
  // fractal subview into a padded alloc (e.g. 2x3x16x2 of 2x4x16x2) still
  // lands on the parent row pitch. Analogous to nd2nz's
  // n_tile_ceil = strides[0] / strides[2]; here one K-fractal (16x2 i8) is
  // one half C0, so strides[0] / strides[1].
  int64_t n_tile_ceil = l1->strides[0] / l1->strides[1];
  uint64_t config =
      ((uint64_t)1) |         // Is the DN matrix number to be moved.
      ((uint64_t)(1)) << 16 | // Is the destination stride of loop2 in unit of
                              // C0 size, loop2 dst stride.
      ((uint64_t)(n_tile_ceil))
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

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

#include "DMA/NCHW2NC1HWC0.h"

#if defined(__DAV_C310__)

template <typename T>
__aicore__ __attribute__((always_inline)) void
check_nchw2nc1hwc0_inputs(memref_t<__gm__ T, 4> *src,
                          memref_t<__cbuf__ T, 5> *dst, int64_t groups) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  constexpr int64_t c0Size = L1_ALIGN_BYTES / sizeof(T);
  assert(groups > 0 && "Groups must be positive.");
  assert(src->sizes[1] % groups == 0 &&
         "Source C dimension must be divisible by groups.");
  assert(dst->sizes[1] % groups == 0 &&
         "Destination C1 dimension must be divisible by groups.");

  const int64_t channelsPerGroup = src->sizes[1] / groups;
  const int64_t c1PerGroup = dst->sizes[1] / groups;

  assert(src->sizes[0] == dst->sizes[0] &&
         "Source and destination batch dimensions must match.");
  assert(c1PerGroup == CEIL_DIV(channelsPerGroup, c0Size) &&
         "Destination C1 dimension does not match grouped source channels.");
  assert(src->sizes[2] == dst->sizes[2] &&
         src->sizes[3] == dst->sizes[3] &&
         "Source and destination spatial dimensions must match.");
  assert(dst->sizes[4] == c0Size &&
         "Destination C0 dimension must occupy one 32-byte block.");

  assert(src->strides[3] == 1 &&
         src->strides[2] == src->sizes[3] &&
         src->strides[1] == src->sizes[2] * src->sizes[3] &&
         src->strides[0] == src->sizes[1] * src->strides[1] &&
         "Source must have a contiguous NCHW layout.");
  assert(dst->strides[4] == 1 && dst->strides[3] == c0Size &&
         dst->strides[2] == dst->sizes[3] * dst->strides[3] &&
         dst->strides[1] == dst->sizes[2] * dst->strides[2] &&
         dst->strides[0] == dst->sizes[1] * dst->strides[1] &&
         "Destination must have a contiguous NC1HWC0 layout.");

  const int64_t spatialSize = src->sizes[2] * src->sizes[3];
  const int64_t matrixCount = src->sizes[0] * groups;
  const int64_t dstMatrixStride =
      c1PerGroup * dst->strides[1] / c0Size;
  assert(matrixCount <= 4095 && spatialSize <= 16384 &&
         dst->strides[1] / c0Size <= 16384 && dstMatrixStride <= 65535 &&
         "NCHW2NC1HWC0 parameters exceed DN2NZ intrinsic limits.");
#endif
}

/// Convert a contiguous GM tensor from [N, G*C, H, W] to a contiguous L1 tensor
/// with shape [N, G*ceil(C/C0), H, W, C0], where C0 * sizeof(T) is 32 bytes.
/// Each group is converted as an independent DN matrix. The DN2NZ intrinsic
/// pads each group's C tail with zeroes when C is not C0-aligned.
template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_gm_to_cbuf_nchw2nc1hwc0_core(memref_t<__gm__ T, 4> *src,
                                  memref_t<__cbuf__ T, 5> *dst,
                                  int64_t groups) {
  static_assert(std::is_same<T, half>::value ||
                    std::is_same<T, bfloat16_t>::value ||
                    std::is_same<T, float>::value,
                "NCHW2NC1HWC0 only supports half, bfloat16_t, and float.");

  check_nchw2nc1hwc0_inputs(src, dst, groups);

  __gm__ T *srcPtr = src->aligned + src->offset;
  __cbuf__ T *dstPtr = dst->aligned + dst->offset;

  const int64_t matrixCount = src->sizes[0] * groups;
  const int64_t channelsPerGroup = src->sizes[1] / groups;
  const int64_t spatialSize = src->sizes[2] * src->sizes[3];
  constexpr int64_t c0Size = L1_ALIGN_BYTES / sizeof(T);
  const int64_t c1PerGroup = dst->sizes[1] / groups;
  const int64_t dstMatrixStride =
      c1PerGroup * dst->strides[1] / c0Size;

  // MTE2_NZ_PARA fields are measured in 32-byte C0 blocks:
  //   loop2: adjacent HW positions;
  //   loop3: adjacent C1 blocks;
  //   loop4: adjacent group matrices, ordered as [N, G].
  const uint64_t config =
      static_cast<uint64_t>(matrixCount) |
      (static_cast<uint64_t>(dst->strides[3] / c0Size) << 16) |
      (static_cast<uint64_t>(dst->strides[1] / c0Size) << 32) |
      (static_cast<uint64_t>(dstMatrixStride) << 48);
  INTRINSIC(set_mte2_nz_para, config);

  // A physical [C, H*W] row-major matrix is a [H*W, C] DN matrix. DN2NZ
  // therefore places C in the innermost C0 dimension and produces NC1HWC0.
  INTRINSIC(copy_gm_to_cbuf_multi_dn2nz, dstPtr, srcPtr,
            /*sid=*/0,
            /*loop1_src_stride=*/src->strides[1] * sizeof(T),
            /*l2_cache_ctrl_mode=*/0,
            /*nValue=*/static_cast<uint16_t>(spatialSize),
            /*dValue=*/static_cast<uint32_t>(channelsPerGroup),
            /*loop4_src_stride=*/channelsPerGroup * src->strides[1] * sizeof(T),
            /*smallc0_en=*/false);
}

#endif

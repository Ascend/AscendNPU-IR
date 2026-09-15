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

#include "DMA/NCHW2C1HWNC0.h"

#if defined(__DAV_C310__)

template <typename T>
__aicore__ __attribute__((always_inline)) void
check_nchw2c1hwnc0_inputs(memref_t<__gm__ T, 4> *src,
                          memref_t<__cbuf__ T, 5> *dst, int64_t groups) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  constexpr int64_t c0Size = L1_ALIGN_BYTES / sizeof(T);
  assert(groups > 0 && "Groups must be positive.");
  assert(src->sizes[0] % groups == 0 &&
         "Source N dimension must be divisible by groups.");
  assert(dst->sizes[3] % groups == 0 &&
         "Destination N dimension must be divisible by groups.");

  const int64_t outputChannelsPerGroup = src->sizes[0] / groups;
  const int64_t alignedOutputChannelsPerGroup = dst->sizes[3] / groups;

  assert(dst->sizes[0] == CEIL_DIV(src->sizes[1], c0Size) &&
         "Destination C1 dimension does not match source channels.");
  assert(src->sizes[2] == dst->sizes[1] &&
         src->sizes[3] == dst->sizes[2] &&
         "Source and destination spatial dimensions must match.");
  assert(dst->sizes[4] == c0Size &&
         "Destination C0 dimension must occupy one 32-byte block.");
  assert(alignedOutputChannelsPerGroup ==
             CEIL_FACTOR(outputChannelsPerGroup, FRACTAL_BLOCK_NUM) &&
         "Destination N dimension must be group-aligned to 16 elements.");

  assert(src->strides[3] == 1 &&
         src->strides[2] == src->sizes[3] &&
         src->strides[1] == src->sizes[2] * src->strides[2] &&
         src->strides[0] == src->sizes[1] * src->strides[1] &&
         "Source must have a contiguous NCHW layout.");
  assert(dst->strides[4] == 1 && dst->strides[3] == c0Size &&
         dst->strides[2] == dst->sizes[3] * dst->strides[3] &&
         dst->strides[1] == dst->sizes[2] * dst->strides[2] &&
         dst->strides[0] == dst->sizes[1] * dst->strides[1] &&
         "Destination must have a contiguous C1HWNC0 layout.");

  const int64_t spatialSize = src->sizes[2] * src->sizes[3];
  const int64_t dstNStride = dst->strides[2] / c0Size;
  const int64_t dstC1Stride = dst->strides[0] / c0Size;
  assert(outputChannelsPerGroup <= 4095 && spatialSize <= 16384 &&
         dstNStride <= 16384 && dstC1Stride <= 16384 &&
         "NCHW2C1HWNC0 parameters exceed DN2NZ intrinsic limits.");
#endif
}

/// Convert a contiguous GM weight from [N, C, H, W] to a contiguous L1 tensor
/// with shape [ceil(C/C0), H, W, G*ceil((N/G)/16)*16, C0], where
/// C0 * sizeof(T) is 32 bytes. C is already the number of input channels per
/// group. The unwritten output-channel padding in L1 is intentionally left
/// unchanged because downstream convolution discards those channels.
template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_gm_to_cbuf_nchw2c1hwnc0_core(memref_t<__gm__ T, 4> *src,
                                  memref_t<__cbuf__ T, 5> *dst,
                                  int64_t groups) {
  static_assert(std::is_same<T, half>::value ||
                    std::is_same<T, bfloat16_t>::value ||
                    std::is_same<T, float>::value,
                "NCHW2C1HWNC0 only supports half, bfloat16_t, and float.");

  check_nchw2c1hwnc0_inputs(src, dst, groups);

  __gm__ T *srcPtr = src->aligned + src->offset;
  __cbuf__ T *dstPtr = dst->aligned + dst->offset;

  const int64_t outputChannelsPerGroup = src->sizes[0] / groups;
  const int64_t alignedOutputChannelsPerGroup = dst->sizes[3] / groups;
  const int64_t spatialSize = src->sizes[2] * src->sizes[3];
  constexpr int64_t c0Size = L1_ALIGN_BYTES / sizeof(T);

  // MTE2_NZ_PARA fields are measured in 32-byte C0 blocks:
  //   loop2: adjacent HW positions, with the complete aligned N dimension
  //          between them;
  //   loop3: adjacent C1 blocks;
  //   loop4: adjacent output-channel matrices within one group.
  const uint64_t config =
      static_cast<uint64_t>(outputChannelsPerGroup) |
      (static_cast<uint64_t>(dst->strides[2] / c0Size) << 16) |
      (static_cast<uint64_t>(dst->strides[0] / c0Size) << 32) |
      (static_cast<uint64_t>(dst->strides[3] / c0Size) << 48);
  INTRINSIC(set_mte2_nz_para, config);

  // Each output channel is a physical [C, H*W] DN matrix. Process groups
  // separately so loop4 skips the output-channel padding between groups.
  // The destination is not initialized to zero before the copy. Therefore,
  // the unwritten oC padding introduced by 16-element alignment retains stale
  // L1 data; downstream convolution must ignore these padded output channels.
  for (int64_t group = 0; group < groups; ++group) {
    __gm__ T *srcGroupPtr =
        srcPtr + group * outputChannelsPerGroup * src->strides[0];
    __cbuf__ T *dstGroupPtr =
        dstPtr + group * alignedOutputChannelsPerGroup * dst->strides[3];

    INTRINSIC(copy_gm_to_cbuf_multi_dn2nz, dstGroupPtr, srcGroupPtr,
              /*sid=*/0,
              /*loop1_src_stride=*/src->strides[1] * sizeof(T),
              /*l2_cache_ctrl_mode=*/0,
              /*nValue=*/static_cast<uint16_t>(spatialSize),
              /*dValue=*/static_cast<uint32_t>(src->sizes[1]),
              /*loop4_src_stride=*/src->strides[0] * sizeof(T),
              /*smallc0_en=*/false);
  }
}

#endif

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

#if defined(__DAV_C310__)

#include "RegBase/VecUtils.h"

const int32_t MAX_THREAD_NUM = 1024;
constexpr unsigned int VEC_SIZE = 2;

#define Call2DIndexSelect(srcty, isTail, src_ptr, val_ptr)                              \
  cce::async_invoke<simt_index_select_core_2d<srcty, IDXTY, IDXDIM, isTail>>(           \
    cce::dim3{MAX_THREAD_NUM},                                                          \
    src_ptr, idx_ptr, val_ptr,                                                          \
    indexBoundary,                                                                      \
    m1, shift1,                                                                         \
    m2, shift2,                                                                         \
    sizes[0], sizes[1], sizes[2],                                                       \
    mappedStartOffsets[0], mappedStartOffsets[1], mappedStartOffsets[2],                \
    mappedEndOffsets[0], mappedEndOffsets[1], mappedEndOffsets[2],                      \
    srcStrides[0] / downsizeFactor, srcStrides[1],                                      \
    idxStrides[0], idxStrides[1],                                                       \
    valStrides[0], valStrides[1], valStrides[2]                                         \
  );

#define Call3DIndexSelect(srcty, isTail, src_ptr, val_ptr)                              \
  cce::async_invoke<simt_index_select_core_3d<srcty, IDXTY, IDXDIM, isTail>>(           \
        cce::dim3{MAX_THREAD_NUM},                                                      \
        src_ptr, idx_ptr, val_ptr,                                                      \
        gatherDim, indexBoundary,                                                       \
        m1, shift1, m2, shift2, m3, shift3,                                             \
        gDim[0], gDim[1],                                                               \
        sizes[0], sizes[1], sizes[2], sizes[3],                                         \
        mappedStartOffsets[0], mappedStartOffsets[1], mappedStartOffsets[2], mappedStartOffsets[3], \
        mappedEndOffsets[0], mappedEndOffsets[1], mappedEndOffsets[2], mappedEndOffsets[3],         \
        srcStrides[0] / downsizeFactor, srcStrides[1] / downsizeFactor, srcStrides[2],              \
        idxStrides[0], idxStrides[1],                                                   \
        valStrides[0], valStrides[1], valStrides[2], valStrides[3]                      \
  );

#define Call4DIndexSelect(srcty, isTail, src_ptr, val_ptr)                              \
  cce::async_invoke<simt_index_select_core_4d<srcty, IDXTY, IDXDIM, isTail>>(           \
        cce::dim3{MAX_THREAD_NUM},                                                      \
        src_ptr, idx_ptr, val_ptr,                                                      \
        gatherDim, indexBoundary,                                                       \
        m1, shift1, m2, shift2, m3, shift3, m4, shift4,                                 \
        gDim[0], gDim[1], gDim[2],                                                      \
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],                               \
        mappedStartOffsets[0], mappedStartOffsets[1], mappedStartOffsets[2],            \
        mappedStartOffsets[3], mappedStartOffsets[4],                                   \
        mappedEndOffsets[0], mappedEndOffsets[1], mappedEndOffsets[2],                  \
        mappedEndOffsets[3], mappedEndOffsets[4],                                       \
        srcStrides[0] / downsizeFactor, srcStrides[1] / downsizeFactor,                 \
        srcStrides[2] / downsizeFactor, srcStrides[3],                                  \
        idxStrides[0], idxStrides[1],                                                   \
        valStrides[0], valStrides[1], valStrides[2], valStrides[3], valStrides[4]       \
  );

#define Call5DIndexSelect(srcty, isTail, src_ptr, val_ptr)                              \
  cce::async_invoke<simt_index_select_core_5d<srcty, IDXTY, IDXDIM, isTail>>(           \
        cce::dim3{MAX_THREAD_NUM},                                                      \
        src_ptr, idx_ptr, val_ptr,                                                      \
        gatherDim, indexBoundary,                                                       \
        m1, shift1, m2, shift2, m3, shift3, m4, shift4, m5, shift5,                     \
        gDim[0], gDim[1], gDim[2], gDim[3],                                             \
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5],                     \
        mappedStartOffsets[0], mappedStartOffsets[1], mappedStartOffsets[2],            \
        mappedStartOffsets[3], mappedStartOffsets[4], mappedStartOffsets[5],            \
        mappedEndOffsets[0], mappedEndOffsets[1], mappedEndOffsets[2],                  \
        mappedEndOffsets[3], mappedEndOffsets[4], mappedEndOffsets[5],                  \
        srcStrides[0] / downsizeFactor, srcStrides[1] / downsizeFactor, srcStrides[2] / downsizeFactor,   \
        srcStrides[3] / downsizeFactor, srcStrides[4],                                  \
        idxStrides[0], idxStrides[1],                                                   \
        valStrides[0], valStrides[1], valStrides[2], valStrides[3], valStrides[4], valStrides[5]          \
  );

template<typename SRCTY, typename IDXTY, int IDXDIM, bool isTail>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void simt_index_select_core_2d(
  __gm__ const SRCTY *src, __ubuf__ const IDXTY *idx, __ubuf__ SRCTY *val,
  const uint32_t indexBoundary,
  const uint32_t m1, const uint32_t shift1,
  const uint32_t m2, const uint32_t shift2,
  const uint32_t size0, const uint32_t size1, const uint32_t size2,
  const uint32_t startOffset0, const uint32_t startOffset1, const uint32_t startOffset2,
  const uint32_t endOffset0, const uint32_t endOffset1, const uint32_t endOffset2,
  const uint32_t src_stride0, const uint32_t src_stride1,
  const uint32_t idx_stride0, const uint32_t idx_stride1,
  const uint32_t val_stride0, const uint32_t val_stride1, const uint32_t val_stride2) {

  // when idx_dim_num = 1, input: size2 = 1, startOffset2,endOffset2 = 0,1, val_stride2 = 0, idx_stride1 = 0
  const uint32_t total_size = size0 * size1 * size2;
  IDXTY index;
  IDXTY src_pos, val_pos;

  for (uint32_t pos = threadIdx.x; pos < total_size; pos += blockDim.x) {
    // when idx_dim_num = 1, vidx2Div = pos // 1 = pos, vidx2 = pos % 1 = 0
    uint32_t vidx2Div = UintDivImpl<uint32_t>(pos, m2, shift2);
    uint32_t vidx2 = pos - vidx2Div * size2;
    uint32_t vidx1Div = UintDivImpl<uint32_t>(vidx2Div, m1, shift1);
    uint32_t vidx1 = vidx2Div - vidx1Div * size1;
    uint32_t vidx0 = vidx1Div;
    if constexpr (isTail){
      // when idx_dim_num = 1, vidx2 = 0, end2=1, start2=0, vidx2 < endOffset2 - startOffset2
      if (vidx0 >= endOffset0 - startOffset0 || vidx1 >= endOffset1 - startOffset1 || 
          vidx2 >= endOffset2 - startOffset2) {
        continue;
      }
    }
    uint32_t sidx1;
    if constexpr (IDXDIM == 1) {
      sidx1 = vidx1 + startOffset1;
    } else if constexpr (IDXDIM == 2) {
      sidx1 = vidx2 + startOffset2;
    }
    // when idx_dim_num = 1, idx_stride1 = 0, index = idx[vidx0 * idx_stride0]
    index = idx[vidx0 * idx_stride0 + vidx1 * idx_stride1];
    if (index < 0) index += indexBoundary;
    if (index < 0 || index >= indexBoundary) continue;
    uint32_t sidx0 = index;

    // when idx_dim_num = 1, vidx2 = 0, val_stride2 = 0, val_pos = vidx0 * val_stride0 + vidx1 * val_stride1
    src_pos = sidx0 * src_stride0 + sidx1 * src_stride1;
    val_pos = vidx0 * val_stride0 + vidx1 * val_stride1 + vidx2 * val_stride2;

    val[val_pos] = src[src_pos];
  }
}

template<typename SRCTY, typename IDXTY, int IDXDIM, bool isTail>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void simt_index_select_core_3d(
  __gm__ const SRCTY *src, __ubuf__ const IDXTY *idx, __ubuf__ SRCTY *val,
  const int32_t gatherDim, const int64_t indexBoundary,
  const uint32_t m1, const uint32_t shift1,
  const uint32_t m2, const uint32_t shift2,
  const uint32_t m3, const uint32_t shift3,
  const uint32_t gDim0, const uint32_t gDim1,
  const uint32_t size0, const uint32_t size1, const uint32_t size2, const uint32_t size3,
  const int32_t startOffset0, const int32_t startOffset1, const int32_t startOffset2, const int32_t startOffset3,
  const int32_t endOffset0, const int32_t endOffset1, const int32_t endOffset2, const int32_t endOffset3,
  const int32_t src_stride0, const int32_t src_stride1, const int32_t src_stride2,
  const int32_t idx_stride0, const int32_t idx_stride1,
  const int32_t val_stride0, const int32_t val_stride1, const int32_t val_stride2, const int32_t val_stride3) {

  // when idx_dim_num = 1, input: size3 = 1, startOffset3,endOffset3 = 0,1, val_stride3 = 0, idx_stride1 = 0
  const uint32_t total_size = size0 * size1 * size2 * size3;
  IDXTY index;
  IDXTY src_pos, val_pos;
  for (uint32_t pos = threadIdx.x; pos < total_size; pos += blockDim.x) {
    // when idx_dim_num = 1, vidx3Div = pos // 1 = pos, vidx3 = pos % 1 = 0
    uint32_t vidx3Div = UintDivImpl<uint32_t>(pos, m3, shift3);
    uint32_t vidx3 = pos - vidx3Div * size3;
    uint32_t vidx2Div = UintDivImpl<uint32_t>(vidx3Div, m2, shift2);
    uint32_t vidx2 = vidx3Div - vidx2Div * size2;
    uint32_t vidx1Div = UintDivImpl<uint32_t>(vidx2Div, m1, shift1);
    uint32_t vidx1 = vidx2Div - vidx1Div * size1;
    uint32_t vidx0 = vidx1Div;
    if constexpr (isTail) {
      // when idx_dim_num = 1, vidx3 = 0, end3=1, start3=0, vidx3 < endOffset3 - startOffset3
      if (vidx0 >= endOffset0 - startOffset0 || vidx1 >= endOffset1 - startOffset1 || 
          vidx2 >= endOffset2 - startOffset2 || vidx3 >= endOffset3 - startOffset3) {
        continue;
      }
    }
    uint32_t sidx0, sidx1, sidx2;
    if constexpr (IDXDIM == 1) {
      sidx0 = vidx0 + startOffset0;
      sidx1 = vidx1 + startOffset1;
      sidx2 = vidx2 + startOffset2;
    } else if constexpr (IDXDIM == 2) {
      // gather dim = 0, dim num = 2, s0->index, s1->v2, s2->v3
      // gather dim = 1, dim num = 2, s0->v0, s1->index, s2->v3
      sidx0 = vidx0 + startOffset0;
      sidx1 = vidx2 + startOffset2;
      sidx2 = vidx3 + startOffset3;
    }

    // when gatherDim == i, gDim_i = 1, other gDim = 0
    // when gatherDim = i, index = idx[vidx_i * idx_stride0 + vidx_(i+1) * idx_stride1], sidx_i = index
    const uint32_t gatherI = (vidx0 * idx_stride0 + vidx1 * idx_stride1) * gDim0 +
                              (vidx1 * idx_stride0 + vidx2 * idx_stride1) * gDim1;
    index = idx[gatherI];
    if (index < 0) index += indexBoundary;
    if (index < 0 || index >= indexBoundary) continue;
    sidx0 = index * gDim0 + sidx0 * gDim1;
    sidx1 = sidx1 * gDim0 + index * gDim1;

    // when idx_dim_num = 1, vidx3 = 0, val_stride3 = 0
    src_pos = sidx0 * src_stride0 + sidx1 * src_stride1 + sidx2 * src_stride2;
    val_pos = vidx0 * val_stride0 + vidx1 * val_stride1 + vidx2 * val_stride2 + vidx3 * val_stride3;

    val[val_pos] = src[src_pos];
  }
}

template<typename SRCTY, typename IDXTY, int IDXDIM, bool isTail>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void simt_index_select_core_4d(
  __gm__ const SRCTY *src, __ubuf__ const IDXTY *idx, __ubuf__ SRCTY *val,
  const int32_t gatherDim, const int64_t indexBoundary,
  const uint32_t m1, const uint32_t shift1,
  const uint32_t m2, const uint32_t shift2,
  const uint32_t m3, const uint32_t shift3,
  const uint32_t m4, const uint32_t shift4,
  const uint32_t gDim0, const uint32_t gDim1, const uint32_t gDim2,
  const uint32_t size0, const uint32_t size1, const uint32_t size2, const uint32_t size3, const uint32_t size4,
  const int32_t startOffset0, const int32_t startOffset1, const int32_t startOffset2,
  const int32_t startOffset3, const int32_t startOffset4,
  const int32_t endOffset0, const int32_t endOffset1, const int32_t endOffset2,
  const int32_t endOffset3, const int32_t endOffset4,
  const int32_t src_stride0, const int32_t src_stride1, const int32_t src_stride2, const int32_t src_stride3,
  const int32_t idx_stride0, const int32_t idx_stride1,
  const int32_t val_stride0, const int32_t val_stride1, const int32_t val_stride2,
  const int32_t val_stride3, const int32_t val_stride4) {

  // when idx_dim_num = 1, input: size4 = 1, startOffset4,endOffset4 = 0,1, val_stride4 = 0, idx_stride1 = 0
  const uint32_t total_size = size0 * size1 * size2 * size3 * size4;
  IDXTY index;
  IDXTY src_pos, val_pos;
  for (uint32_t pos = threadIdx.x; pos < total_size; pos += blockDim.x) {
    // when idx_dim_num = 1, vidx4Div = pos // 1 = pos, vidx4 = pos % 1 = 0
    uint32_t vidx4Div = UintDivImpl<uint32_t>(pos, m4, shift4);
    uint32_t vidx4 = pos - vidx4Div * size4;
    uint32_t vidx3Div = UintDivImpl<uint32_t>(vidx4Div, m3, shift3);
    uint32_t vidx3 = vidx4Div - vidx3Div * size3;
    uint32_t vidx2Div = UintDivImpl<uint32_t>(vidx3Div, m2, shift2);
    uint32_t vidx2 = vidx3Div - vidx2Div * size2;
    uint32_t vidx1Div = UintDivImpl<uint32_t>(vidx2Div, m1, shift1);
    uint32_t vidx1 = vidx2Div - vidx1Div * size1;
    uint32_t vidx0 = vidx1Div;
    if constexpr (isTail) {
      // when idx_dim_num = 1, vidx4 = 0, end4=1, start4=0, vidx4 < endOffset4 - startOffset4
      if (vidx0 >= endOffset0 - startOffset0 || vidx1 >= endOffset1 - startOffset1 || 
          vidx2 >= endOffset2 - startOffset2 || vidx3 >= endOffset3 - startOffset3 ||
          vidx4 >= endOffset4 - startOffset4) {
        continue;
      }
    }
    uint32_t sidx0, sidx1, sidx2, sidx3;
    if constexpr (IDXDIM == 1) {
      sidx0 = vidx0 + startOffset0;
      sidx1 = vidx1 + startOffset1;
      sidx2 = vidx2 + startOffset2;
      sidx3 = vidx3 + startOffset3;
    } else if constexpr (IDXDIM == 2){
      // gather dim = 0, dim num = 2, s0->index, s1->v2, s2->v3, s3->v4
      // gather dim = 1, dim num = 2, s0->v0, s1->index, s2->v3, s3->v4
      // gather dim = 2, dim num = 2, s0->v0, s1->v1, s2->index, s3->v4
      sidx0 = vidx0 + startOffset0;
      sidx1 = vidx2 + startOffset2;
      sidx2 = vidx3 + startOffset3;
      sidx3 = vidx4 + startOffset4;
    }

    // when gatherDim == i, gDim_i = 1, other gDim = 0
    // when gatherDim = i, index = idx[vidx_i * idx_stride0 + vidx_(i+1) * idx_stride1], sidx_i = index
    // when idx_dim_num = 1, idx_stride1 = 0, index = idx[vidx_i * idx_stride0]
    const uint32_t gatherI = (vidx0 * idx_stride0 + vidx1 * idx_stride1) * gDim0 +
                              (vidx1 * idx_stride0 + vidx2 * idx_stride1) * gDim1 +
                              (vidx2 * idx_stride0 + vidx3 * idx_stride1) * gDim2;
    index = idx[gatherI];
    if (index < 0) index += indexBoundary;
    if (index < 0 || index >= indexBoundary) continue;
    sidx0 = index * gDim0 + sidx0 * gDim1 + sidx0 * gDim2;
    sidx1 = sidx1 * gDim0 + index * gDim1 + (vidx1 + startOffset1) * gDim2;
    sidx2 = sidx2 * gDim0 + sidx2 * gDim1 + index * gDim2;

    // when idx_dim_num = 1, vidx4 = 0, val_stride4 = 0
    src_pos = sidx0 * src_stride0 + sidx1 * src_stride1 + sidx2 * src_stride2 + sidx3 * src_stride3;
    val_pos = vidx0 * val_stride0 + vidx1 * val_stride1 + vidx2 * val_stride2 + vidx3 * val_stride3 +
              vidx4 * val_stride4;

    val[val_pos] = src[src_pos];
  }
}

template<typename SRCTY, typename IDXTY, int IDXDIM, bool isTail>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void simt_index_select_core_5d(
  __gm__ const SRCTY *src, __ubuf__ const IDXTY *idx, __ubuf__ SRCTY *val,
  const int32_t gatherDim, const int64_t indexBoundary,
  const uint32_t m1, const uint32_t shift1,
  const uint32_t m2, const uint32_t shift2,
  const uint32_t m3, const uint32_t shift3,
  const uint32_t m4, const uint32_t shift4,
  const uint32_t m5, const uint32_t shift5,
  const uint32_t gDim0, const uint32_t gDim1, const uint32_t gDim2, const uint32_t gDim3,
  const uint32_t size0, const uint32_t size1, const uint32_t size2,
  const uint32_t size3, const uint32_t size4, const uint32_t size5,
  const int32_t startOffset0, const int32_t startOffset1, const int32_t startOffset2,
  const int32_t startOffset3, const int32_t startOffset4, const int32_t startOffset5,
  const int32_t endOffset0, const int32_t endOffset1, const int32_t endOffset2,
  const int32_t endOffset3, const int32_t endOffset4, const int32_t endOffset5,
  const int32_t src_stride0, const int32_t src_stride1, const int32_t src_stride2,
  const int32_t src_stride3, const int32_t src_stride4,
  const int32_t idx_stride0,  const int32_t idx_stride1,
  const int32_t val_stride0, const int32_t val_stride1, const int32_t val_stride2,
  const int32_t val_stride3, const int32_t val_stride4, const int32_t val_stride5) {

  // when idx_dim_num = 1, input: size5 = 1, startOffset5,endOffset5 = 0,1, val_stride5 = 0, idx_stride1 = 0
  const uint32_t total_size = size0 * size1 * size2 * size3 * size4 * size5;
  IDXTY index;
  IDXTY src_pos, val_pos;
  for (uint32_t pos = threadIdx.x; pos < total_size; pos += blockDim.x) {
    // when idx_dim_num = 1, vidx5Div = pos // 1 = pos, vidx5 = pos % 1 = 0
    uint32_t vidx5Div = UintDivImpl<uint32_t>(pos, m5, shift5);
    uint32_t vidx5 = pos - vidx5Div * size5;
    uint32_t vidx4Div = UintDivImpl<uint32_t>(vidx5Div, m4, shift4);
    uint32_t vidx4 = vidx5Div - vidx4Div * size4;
    uint32_t vidx3Div = UintDivImpl<uint32_t>(vidx4Div, m3, shift3);
    uint32_t vidx3 = vidx4Div - vidx3Div * size3;
    uint32_t vidx2Div = UintDivImpl<uint32_t>(vidx3Div, m2, shift2);
    uint32_t vidx2 = vidx3Div - vidx2Div * size2;
    uint32_t vidx1Div = UintDivImpl<uint32_t>(vidx2Div, m1, shift1);
    uint32_t vidx1 = vidx2Div - vidx1Div * size1;
    uint32_t vidx0 = vidx1Div;
    if constexpr (isTail) {
      // when idx_dim_num = 1, vidx5 = 0, end5=1, start5=0, vidx5 < endOffset5 - startOffset5
      if (vidx0 >= endOffset0 - startOffset0 || vidx1 >= endOffset1 - startOffset1 || 
          vidx2 >= endOffset2 - startOffset2 || vidx3 >= endOffset3 - startOffset3 ||
          vidx4 >= endOffset4 - startOffset4 || vidx5 >= endOffset5 - startOffset5) {
        continue;
      }
    }
    uint32_t sidx0, sidx1, sidx2, sidx3, sidx4;
    if constexpr (IDXDIM == 1) {
      sidx0 = vidx0 + startOffset0;
      sidx1 = vidx1 + startOffset1;
      sidx2 = vidx2 + startOffset2;
      sidx3 = vidx3 + startOffset3;
      sidx4 = vidx4 + startOffset4;
    } else if constexpr (IDXDIM == 2){
      // gather dim = 0, dim num = 2, s0->index, s1->v2, s2->v3, s3->v4, s4->v5
      // gather dim = 1, dim num = 2, s0->v0, s1->index, s2->v3, s3->v4, s4->v5
      // gather dim = 2, dim num = 2, s0->v0, s1->v1, s2->index, s3->v4, s4->v5
      // gather dim = 3, dim num = 2, s0->v0, s1->v1, s2->v2, s3->index, s4->v5
      sidx0 = vidx0 + startOffset0;
      sidx1 = vidx2 + startOffset2;
      sidx2 = vidx3 + startOffset3;
      sidx3 = vidx4 + startOffset4;
      sidx4 = vidx5 + startOffset5;
    }

    // when gatherDim == i, gDim_i = 1, other gDim = 0
    // when gatherDim = i, index = idx[vidx_i * idx_stride0 + vidx_(i+1) * idx_stride1], sidx_i = index
    // when idx_dim_num = 1, idx_stride1 = 0, index = idx[vidx_i * idx_stride0]
    const uint32_t gatherI = (vidx0 * idx_stride0 + vidx1 * idx_stride1) * gDim0 +
                              (vidx1 * idx_stride0 + vidx2 * idx_stride1) * gDim1 +
                              (vidx2 * idx_stride0 + vidx3 * idx_stride1) * gDim2 +
                              (vidx3 * idx_stride0 + vidx4 * idx_stride1) * gDim3;
    index = idx[gatherI];
    if (index < 0) index += indexBoundary;
    if (index < 0 || index >= indexBoundary) continue;
    sidx0 = index * gDim0 + sidx0 * gDim1 + sidx0 * gDim2 + sidx0 * gDim3;
    sidx1 = sidx1 * gDim0 + index * gDim1 + (vidx1 + startOffset1) * gDim2 + (vidx1 + startOffset1) * gDim3;
    sidx2 = sidx2 * gDim0 + sidx2 * gDim1 + index * gDim2 + (vidx2 + startOffset2) * gDim3;
    sidx3 = sidx3 * gDim0 + sidx3 * gDim1 + sidx3 * gDim2 + index * gDim3;

    // when idx_dim_num = 1, vidx5 = 0, val_stride5 = 0
    src_pos = sidx0 * src_stride0 + sidx1 * src_stride1 + sidx2 * src_stride2 + sidx3 * src_stride3 +
              sidx4 * src_stride4;
    val_pos = vidx0 * val_stride0 + vidx1 * val_stride1 + vidx2 * val_stride2 + vidx3 * val_stride3 +
              vidx4 * val_stride4 + vidx5 * val_stride5;

    val[val_pos] = src[src_pos];
  }
}

template <int SRCDIM, int IDXDIM, int VALDIM, typename SRCTY, typename IDXTY>
__aiv__ __attribute__((always_inline)) void simt_index_select(
  memref_t<__gm__ SRCTY, 1> *src, memref_t<__ubuf__ IDXTY, IDXDIM> *idx,
  memref_t<__ubuf__ SRCTY, VALDIM> *val,
  const int32_t gatherDim,
  const int64_t indexBoundary,
  const IDXTY* endOffsets,
  const IDXTY* startOffsets,
  const IDXTY* srcStrides
) {

  static_assert(SRCDIM <= 5, "SRC DIM must be <= 5");
  static_assert(IDXDIM <= 2, "IDX DIM must be <= 2");
  static_assert(VALDIM <= 6, "VAL DIM must be <= 6");

  static_assert(std::is_same<SRCTY, float>::value || std::is_same<SRCTY, half>::value ||
                std::is_same<SRCTY, bfloat16_t>::value || std::is_same<SRCTY, int8_t>::value ||
                std::is_same<SRCTY, int16_t>::value || std::is_same<SRCTY, int32_t>::value ||
                std::is_same<SRCTY, int64_t>::value || std::is_same<SRCTY, uint8_t>::value ||
                std::is_same<SRCTY, uint16_t>::value || std::is_same<SRCTY, uint32_t>::value ||
                std::is_same<SRCTY, uint64_t>::value,
              "Currently dtype of source supports only float32|half|bfloat16|int8|int16|int32|int64| \
              uint8|uint16|uint32|uint64");

  static_assert(std::is_same<IDXTY, int32_t>::value || std::is_same<IDXTY, int64_t>::value,
              "Currently dtype of index supports only int32|int64");

  uint32_t downsizeFactor = 1;
  if constexpr (std::is_same<SRCTY, float>::value) {
    if (val->sizes[VALDIM-1] % VEC_SIZE == 0 && srcStrides[0] % VEC_SIZE == 0 &&
        startOffsets[SRCDIM-1] % VEC_SIZE ==0 && endOffsets[VALDIM-1] % VEC_SIZE ==0) {
      downsizeFactor = VEC_SIZE;  // need to reinterpret to float2
    }
  }

  constexpr int MAPDIM = SRCDIM + 1;
  // when idxdim == 1, padding startOffset with 0, padding endOffset with 1;
  IDXTY mappedStartOffsets[MAPDIM] = {0};
  IDXTY mappedEndOffsets[MAPDIM] = {1};
  // when idxdim == 1, padding val size with 1, padding idx stride / val stride with 0;
  uint32_t sizes[MAPDIM] = {1};
  uint32_t valStrides[MAPDIM] = {0};
  uint32_t idxStrides[2] = {0};
  uint32_t gDim[SRCDIM] = {0};
  for (int i = 0; i < SRCDIM; i++) {
    gDim[i] = (i == gatherDim) ? 1 : 0;
  }
  // Align the offsets to MAPDIM
  // for example (s -> startOffset, e -> endOffset)
  // when idxdim == 1, we got input          s0, s1, s2, s3;     e0, e1, e2, e3
  // when idxdim == 2, we got input          s0, s1, s2, s3;     e0, e1, e2, e3, e4
  // map the offsets as :
  // when idxdim == 1,                we get s0, s1, s2, s3, 0 ; e0, e1, e2, e3, 1
  // when idxdim == 2, gatherDim = 0, we get s0, s0, s1, s2, s3; e0, e1, e2, e3, e4
  if constexpr (IDXDIM == 1) {
    for (int i = 0; i < SRCDIM; i++) {
      // reinterpret val strides for not lowest dim, lowest dim will restore later
      mappedStartOffsets[i] = startOffsets[i];
      mappedEndOffsets[i] = endOffsets[i];
      sizes[i] = val->sizes[i];
      valStrides[i] = val->strides[i] / downsizeFactor;
    }
    mappedStartOffsets[SRCDIM] = 0;
    mappedEndOffsets[SRCDIM] = 1;
    sizes[SRCDIM] = 1;
    idxStrides[0] = idx->strides[0];
    idxStrides[1] = 0;
    valStrides[SRCDIM] = 0;
  } else if constexpr (IDXDIM == 2) {
    for (int i = 0; i < MAPDIM; i++) {
      // remap startOffset to align endOffset
      mappedStartOffsets[i] = (i <= gatherDim) ? startOffsets[i] : startOffsets[i-1];
      mappedEndOffsets[i] = endOffsets[i];
      sizes[i] = val->sizes[i];
      valStrides[i] = val->strides[i] / downsizeFactor;
    }
    idxStrides[0] = idx->strides[0];
    idxStrides[1] = idx->strides[1];
  }
  // reinterpret offset and size for lowest dim, restore val stride for lowest dim
  mappedStartOffsets[VALDIM-1] = mappedStartOffsets[VALDIM-1] / downsizeFactor;
  mappedEndOffsets[VALDIM-1] = mappedEndOffsets[VALDIM-1] / downsizeFactor;
  sizes[VALDIM-1] = sizes[VALDIM-1] / downsizeFactor;
  valStrides[VALDIM-1] = val->strides[VALDIM-1];

  bool isTail = false;
  for (int i = 0; i < MAPDIM; i++) {
    if (mappedEndOffsets[i] - mappedStartOffsets[i] < sizes[i]) {
      isTail = true;
      break;
    }
  }

  auto src_ptr = src->aligned;
  auto idx_ptr = idx->aligned + idx->offset;
  auto val_ptr = val->aligned + val->offset;

  uint32_t m1 = 0, m2 = 0, m3 = 0, m4 = 0, m5 = 0;
  uint32_t shift1 = 0, shift2 = 0, shift3 = 0, shift4 = 0, shift5 = 0;
  GetUintDivMagicAndShiftImpl<uint32_t>(m1, shift1, sizes[1]);
  if constexpr (SRCDIM == 2) {
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, sizes[2]);
    if constexpr (std::is_same<SRCTY, float>::value) {
      if (downsizeFactor == VEC_SIZE) {
          __gm__ float2 *src_ptr2 = reinterpret_cast<__gm__ float2 *>(src_ptr);
          __ubuf__ float2 *val_ptr2 = reinterpret_cast<__ubuf__ float2 *>(val_ptr);
          if (isTail) {
              Call2DIndexSelect(float2, true, src_ptr2, val_ptr2)
          } else {
              Call2DIndexSelect(float2, false, src_ptr2, val_ptr2)
          }
          return;
      }
    }
    if (isTail) {
        Call2DIndexSelect(SRCTY, true, src_ptr, val_ptr)
    } else {
        Call2DIndexSelect(SRCTY, false, src_ptr, val_ptr)
    }
  } else if constexpr (SRCDIM == 3) {
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, sizes[2]);
    GetUintDivMagicAndShiftImpl<uint32_t>(m3, shift3, sizes[3]);
    if constexpr (std::is_same<SRCTY, float>::value) {
      if (downsizeFactor == VEC_SIZE) {
          __gm__ float2 *src_ptr2 = reinterpret_cast<__gm__ float2 *>(src_ptr);
          __ubuf__ float2 *val_ptr2 = reinterpret_cast<__ubuf__ float2 *>(val_ptr);
          if (isTail) {
            Call3DIndexSelect(float2, true, src_ptr2, val_ptr2)
          } else {
            Call3DIndexSelect(float2, false, src_ptr2, val_ptr2)
          }
          return;
      }
    }
    if (isTail) {
      Call3DIndexSelect(SRCTY, true, src_ptr, val_ptr)
    } else {
      Call3DIndexSelect(SRCTY, false, src_ptr, val_ptr)
    }
  } else if constexpr (SRCDIM == 4) {
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, sizes[2]);
    GetUintDivMagicAndShiftImpl<uint32_t>(m3, shift3, sizes[3]);
    GetUintDivMagicAndShiftImpl<uint32_t>(m4, shift4, sizes[4]);
    if constexpr (std::is_same<SRCTY, float>::value) {
      if (downsizeFactor == VEC_SIZE) {
          __gm__ float2 *src_ptr2 = reinterpret_cast<__gm__ float2 *>(src_ptr);
          __ubuf__ float2 *val_ptr2 = reinterpret_cast<__ubuf__ float2 *>(val_ptr);
          cce::async_invoke<simt_index_select_core_4d<float2, IDXTY, IDXDIM, false>>(
                cce::dim3{MAX_THREAD_NUM},
                src_ptr2, idx_ptr, val_ptr2,
                gatherDim, indexBoundary,
                m1, shift1, m2, shift2, m3, shift3, m4, shift4,
                gDim[0], gDim[1], gDim[2],
                sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
                mappedStartOffsets[0], mappedStartOffsets[1], mappedStartOffsets[2], 
                mappedStartOffsets[3], mappedStartOffsets[4],
                mappedEndOffsets[0], mappedEndOffsets[1], mappedEndOffsets[2],
                mappedEndOffsets[3], mappedEndOffsets[4],
                srcStrides[0] / downsizeFactor, srcStrides[1] / downsizeFactor,
                srcStrides[2] / downsizeFactor, srcStrides[3],
                idxStrides[0], idxStrides[1],
                valStrides[0], valStrides[1], valStrides[2], valStrides[3], valStrides[4]
          );
          return;
      }
    }
    cce::async_invoke<simt_index_select_core_4d<SRCTY, IDXTY, IDXDIM, false>>(
          cce::dim3{MAX_THREAD_NUM},
          src_ptr, idx_ptr, val_ptr,
          gatherDim, indexBoundary,
          m1, shift1, m2, shift2, m3, shift3, m4, shift4,
          gDim[0], gDim[1], gDim[2],
          sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
          mappedStartOffsets[0], mappedStartOffsets[1], mappedStartOffsets[2], 
          mappedStartOffsets[3], mappedStartOffsets[4],
          mappedEndOffsets[0], mappedEndOffsets[1], mappedEndOffsets[2],
          mappedEndOffsets[3], mappedEndOffsets[4],
          srcStrides[0], srcStrides[1], srcStrides[2], srcStrides[3],
          idxStrides[0], idxStrides[1],
          valStrides[0], valStrides[1], valStrides[2], valStrides[3], valStrides[4]
    );
  } else if constexpr (SRCDIM == 5) {
    GetUintDivMagicAndShiftImpl<uint32_t>(m2, shift2, sizes[2]);
    GetUintDivMagicAndShiftImpl<uint32_t>(m3, shift3, sizes[3]);
    GetUintDivMagicAndShiftImpl<uint32_t>(m4, shift4, sizes[4]);
    GetUintDivMagicAndShiftImpl<uint32_t>(m5, shift5, sizes[5]);
    if constexpr (std::is_same<SRCTY, float>::value) {
      if (downsizeFactor == VEC_SIZE) {
          __gm__ float2 *src_ptr2 = reinterpret_cast<__gm__ float2 *>(src_ptr);
          __ubuf__ float2 *val_ptr2 = reinterpret_cast<__ubuf__ float2 *>(val_ptr);
          cce::async_invoke<simt_index_select_core_5d<float2, IDXTY, IDXDIM, false>>(
                cce::dim3{MAX_THREAD_NUM},
                src_ptr2, idx_ptr, val_ptr2,
                gatherDim, indexBoundary,
                m1, shift1, m2, shift2, m3, shift3, m4, shift4, m5, shift5,
                gDim[0], gDim[1], gDim[2], gDim[3],
                sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5],
                mappedStartOffsets[0], mappedStartOffsets[1], mappedStartOffsets[2], 
                mappedStartOffsets[3], mappedStartOffsets[4], mappedStartOffsets[5],
                mappedEndOffsets[0], mappedEndOffsets[1], mappedEndOffsets[2],
                mappedEndOffsets[3], mappedEndOffsets[4], mappedEndOffsets[5],
                srcStrides[0] / downsizeFactor, srcStrides[1] / downsizeFactor, srcStrides[2] / downsizeFactor,
                srcStrides[3] / downsizeFactor, srcStrides[4],
                idxStrides[0], idxStrides[1],
                valStrides[0], valStrides[1], valStrides[2], valStrides[3], valStrides[4], valStrides[5]
          );
          return;
      }
    }
    cce::async_invoke<simt_index_select_core_5d<SRCTY, IDXTY, IDXDIM, false>>(
          cce::dim3{MAX_THREAD_NUM},
          src_ptr, idx_ptr, val_ptr,
          gatherDim, indexBoundary,
          m1, shift1, m2, shift2, m3, shift3, m4, shift4, m5, shift5,
          gDim[0], gDim[1], gDim[2], gDim[3],
          sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5],
          mappedStartOffsets[0], mappedStartOffsets[1], mappedStartOffsets[2], 
          mappedStartOffsets[3], mappedStartOffsets[4], mappedStartOffsets[5],
          mappedEndOffsets[0], mappedEndOffsets[1], mappedEndOffsets[2],
          mappedEndOffsets[3], mappedEndOffsets[4], mappedEndOffsets[5],
          srcStrides[0], srcStrides[1], srcStrides[2],
          srcStrides[3], srcStrides[4],
          idxStrides[0], idxStrides[1],
          valStrides[0], valStrides[1], valStrides[2], valStrides[3], valStrides[4], valStrides[5]
    );
  }
}

extern "C" {
REGISTER_INDEX_SELECT_2D_1D(float, int32_t);
REGISTER_INDEX_SELECT_2D_1D(half, int32_t);
REGISTER_INDEX_SELECT_2D_1D(bfloat16_t, int32_t);
REGISTER_INDEX_SELECT_2D_1D(int8_t, int32_t);
REGISTER_INDEX_SELECT_2D_1D(int16_t, int32_t);
REGISTER_INDEX_SELECT_2D_1D(int32_t, int32_t);
REGISTER_INDEX_SELECT_2D_1D(int64_t, int32_t);
REGISTER_INDEX_SELECT_2D_1D(uint8_t, int32_t);
REGISTER_INDEX_SELECT_2D_1D(uint16_t, int32_t);
REGISTER_INDEX_SELECT_2D_1D(uint32_t, int32_t);
REGISTER_INDEX_SELECT_2D_1D(uint64_t, int32_t);
REGISTER_INDEX_SELECT_2D_1D(float, int64_t);
REGISTER_INDEX_SELECT_2D_1D(half, int64_t);
REGISTER_INDEX_SELECT_2D_1D(bfloat16_t, int64_t);
REGISTER_INDEX_SELECT_2D_1D(int8_t, int64_t);
REGISTER_INDEX_SELECT_2D_1D(int16_t, int64_t);
REGISTER_INDEX_SELECT_2D_1D(int32_t, int64_t);
REGISTER_INDEX_SELECT_2D_1D(int64_t, int64_t);
REGISTER_INDEX_SELECT_2D_1D(uint8_t, int64_t);
REGISTER_INDEX_SELECT_2D_1D(uint16_t, int64_t);
REGISTER_INDEX_SELECT_2D_1D(uint32_t, int64_t);
REGISTER_INDEX_SELECT_2D_1D(uint64_t, int64_t);

REGISTER_INDEX_SELECT_2D_2D(float, int32_t);
REGISTER_INDEX_SELECT_2D_2D(half, int32_t);
REGISTER_INDEX_SELECT_2D_2D(bfloat16_t, int32_t);
REGISTER_INDEX_SELECT_2D_2D(int8_t, int32_t);
REGISTER_INDEX_SELECT_2D_2D(int16_t, int32_t);
REGISTER_INDEX_SELECT_2D_2D(int32_t, int32_t);
REGISTER_INDEX_SELECT_2D_2D(int64_t, int32_t);
REGISTER_INDEX_SELECT_2D_2D(uint8_t, int32_t);
REGISTER_INDEX_SELECT_2D_2D(uint16_t, int32_t);
REGISTER_INDEX_SELECT_2D_2D(uint32_t, int32_t);
REGISTER_INDEX_SELECT_2D_2D(uint64_t, int32_t);
REGISTER_INDEX_SELECT_2D_2D(float, int64_t);
REGISTER_INDEX_SELECT_2D_2D(half, int64_t);
REGISTER_INDEX_SELECT_2D_2D(bfloat16_t, int64_t);
REGISTER_INDEX_SELECT_2D_2D(int8_t, int64_t);
REGISTER_INDEX_SELECT_2D_2D(int16_t, int64_t);
REGISTER_INDEX_SELECT_2D_2D(int32_t, int64_t);
REGISTER_INDEX_SELECT_2D_2D(int64_t, int64_t);
REGISTER_INDEX_SELECT_2D_2D(uint8_t, int64_t);
REGISTER_INDEX_SELECT_2D_2D(uint16_t, int64_t);
REGISTER_INDEX_SELECT_2D_2D(uint32_t, int64_t);
REGISTER_INDEX_SELECT_2D_2D(uint64_t, int64_t);

REGISTER_INDEX_SELECT_3D_1D(float, int32_t);
REGISTER_INDEX_SELECT_3D_1D(half, int32_t);
REGISTER_INDEX_SELECT_3D_1D(bfloat16_t, int32_t);
REGISTER_INDEX_SELECT_3D_1D(int8_t, int32_t);
REGISTER_INDEX_SELECT_3D_1D(int16_t, int32_t);
REGISTER_INDEX_SELECT_3D_1D(int32_t, int32_t);
REGISTER_INDEX_SELECT_3D_1D(int64_t, int32_t);
REGISTER_INDEX_SELECT_3D_1D(uint8_t, int32_t);
REGISTER_INDEX_SELECT_3D_1D(uint16_t, int32_t);
REGISTER_INDEX_SELECT_3D_1D(uint32_t, int32_t);
REGISTER_INDEX_SELECT_3D_1D(uint64_t, int32_t);
REGISTER_INDEX_SELECT_3D_1D(float, int64_t);
REGISTER_INDEX_SELECT_3D_1D(half, int64_t);
REGISTER_INDEX_SELECT_3D_1D(bfloat16_t, int64_t);
REGISTER_INDEX_SELECT_3D_1D(int8_t, int64_t);
REGISTER_INDEX_SELECT_3D_1D(int16_t, int64_t);
REGISTER_INDEX_SELECT_3D_1D(int32_t, int64_t);
REGISTER_INDEX_SELECT_3D_1D(int64_t, int64_t);
REGISTER_INDEX_SELECT_3D_1D(uint8_t, int64_t);
REGISTER_INDEX_SELECT_3D_1D(uint16_t, int64_t);
REGISTER_INDEX_SELECT_3D_1D(uint32_t, int64_t);
REGISTER_INDEX_SELECT_3D_1D(uint64_t, int64_t);

REGISTER_INDEX_SELECT_3D_2D(float, int32_t);
REGISTER_INDEX_SELECT_3D_2D(half, int32_t);
REGISTER_INDEX_SELECT_3D_2D(bfloat16_t, int32_t);
REGISTER_INDEX_SELECT_3D_2D(int8_t, int32_t);
REGISTER_INDEX_SELECT_3D_2D(int16_t, int32_t);
REGISTER_INDEX_SELECT_3D_2D(int32_t, int32_t);
REGISTER_INDEX_SELECT_3D_2D(int64_t, int32_t);
REGISTER_INDEX_SELECT_3D_2D(uint8_t, int32_t);
REGISTER_INDEX_SELECT_3D_2D(uint16_t, int32_t);
REGISTER_INDEX_SELECT_3D_2D(uint32_t, int32_t);
REGISTER_INDEX_SELECT_3D_2D(uint64_t, int32_t);
REGISTER_INDEX_SELECT_3D_2D(float, int64_t);
REGISTER_INDEX_SELECT_3D_2D(half, int64_t);
REGISTER_INDEX_SELECT_3D_2D(bfloat16_t, int64_t);
REGISTER_INDEX_SELECT_3D_2D(int8_t, int64_t);
REGISTER_INDEX_SELECT_3D_2D(int16_t, int64_t);
REGISTER_INDEX_SELECT_3D_2D(int32_t, int64_t);
REGISTER_INDEX_SELECT_3D_2D(int64_t, int64_t);
REGISTER_INDEX_SELECT_3D_2D(uint8_t, int64_t);
REGISTER_INDEX_SELECT_3D_2D(uint16_t, int64_t);
REGISTER_INDEX_SELECT_3D_2D(uint32_t, int64_t);
REGISTER_INDEX_SELECT_3D_2D(uint64_t, int64_t);

REGISTER_INDEX_SELECT_4D_1D(float, int32_t);
REGISTER_INDEX_SELECT_4D_1D(half, int32_t);
REGISTER_INDEX_SELECT_4D_1D(bfloat16_t, int32_t);
REGISTER_INDEX_SELECT_4D_1D(int8_t, int32_t);
REGISTER_INDEX_SELECT_4D_1D(int16_t, int32_t);
REGISTER_INDEX_SELECT_4D_1D(int32_t, int32_t);
REGISTER_INDEX_SELECT_4D_1D(int64_t, int32_t);
REGISTER_INDEX_SELECT_4D_1D(uint8_t, int32_t);
REGISTER_INDEX_SELECT_4D_1D(uint16_t, int32_t);
REGISTER_INDEX_SELECT_4D_1D(uint32_t, int32_t);
REGISTER_INDEX_SELECT_4D_1D(uint64_t, int32_t);
REGISTER_INDEX_SELECT_4D_1D(float, int64_t);
REGISTER_INDEX_SELECT_4D_1D(half, int64_t);
REGISTER_INDEX_SELECT_4D_1D(bfloat16_t, int64_t);
REGISTER_INDEX_SELECT_4D_1D(int8_t, int64_t);
REGISTER_INDEX_SELECT_4D_1D(int16_t, int64_t);
REGISTER_INDEX_SELECT_4D_1D(int32_t, int64_t);
REGISTER_INDEX_SELECT_4D_1D(int64_t, int64_t);
REGISTER_INDEX_SELECT_4D_1D(uint8_t, int64_t);
REGISTER_INDEX_SELECT_4D_1D(uint16_t, int64_t);
REGISTER_INDEX_SELECT_4D_1D(uint32_t, int64_t);
REGISTER_INDEX_SELECT_4D_1D(uint64_t, int64_t);

REGISTER_INDEX_SELECT_4D_2D(float, int32_t);
REGISTER_INDEX_SELECT_4D_2D(half, int32_t);
REGISTER_INDEX_SELECT_4D_2D(bfloat16_t, int32_t);
REGISTER_INDEX_SELECT_4D_2D(int8_t, int32_t);
REGISTER_INDEX_SELECT_4D_2D(int16_t, int32_t);
REGISTER_INDEX_SELECT_4D_2D(int32_t, int32_t);
REGISTER_INDEX_SELECT_4D_2D(int64_t, int32_t);
REGISTER_INDEX_SELECT_4D_2D(uint8_t, int32_t);
REGISTER_INDEX_SELECT_4D_2D(uint16_t, int32_t);
REGISTER_INDEX_SELECT_4D_2D(uint32_t, int32_t);
REGISTER_INDEX_SELECT_4D_2D(uint64_t, int32_t);
REGISTER_INDEX_SELECT_4D_2D(float, int64_t);
REGISTER_INDEX_SELECT_4D_2D(half, int64_t);
REGISTER_INDEX_SELECT_4D_2D(bfloat16_t, int64_t);
REGISTER_INDEX_SELECT_4D_2D(int8_t, int64_t);
REGISTER_INDEX_SELECT_4D_2D(int16_t, int64_t);
REGISTER_INDEX_SELECT_4D_2D(int32_t, int64_t);
REGISTER_INDEX_SELECT_4D_2D(int64_t, int64_t);
REGISTER_INDEX_SELECT_4D_2D(uint8_t, int64_t);
REGISTER_INDEX_SELECT_4D_2D(uint16_t, int64_t);
REGISTER_INDEX_SELECT_4D_2D(uint32_t, int64_t);
REGISTER_INDEX_SELECT_4D_2D(uint64_t, int64_t);

REGISTER_INDEX_SELECT_5D_1D(float, int32_t);
REGISTER_INDEX_SELECT_5D_1D(half, int32_t);
REGISTER_INDEX_SELECT_5D_1D(bfloat16_t, int32_t);
REGISTER_INDEX_SELECT_5D_1D(int8_t, int32_t);
REGISTER_INDEX_SELECT_5D_1D(int16_t, int32_t);
REGISTER_INDEX_SELECT_5D_1D(int32_t, int32_t);
REGISTER_INDEX_SELECT_5D_1D(int64_t, int32_t);
REGISTER_INDEX_SELECT_5D_1D(uint8_t, int32_t);
REGISTER_INDEX_SELECT_5D_1D(uint16_t, int32_t);
REGISTER_INDEX_SELECT_5D_1D(uint32_t, int32_t);
REGISTER_INDEX_SELECT_5D_1D(uint64_t, int32_t);
REGISTER_INDEX_SELECT_5D_1D(float, int64_t);
REGISTER_INDEX_SELECT_5D_1D(half, int64_t);
REGISTER_INDEX_SELECT_5D_1D(bfloat16_t, int64_t);
REGISTER_INDEX_SELECT_5D_1D(int8_t, int64_t);
REGISTER_INDEX_SELECT_5D_1D(int16_t, int64_t);
REGISTER_INDEX_SELECT_5D_1D(int32_t, int64_t);
REGISTER_INDEX_SELECT_5D_1D(int64_t, int64_t);
REGISTER_INDEX_SELECT_5D_1D(uint8_t, int64_t);
REGISTER_INDEX_SELECT_5D_1D(uint16_t, int64_t);
REGISTER_INDEX_SELECT_5D_1D(uint32_t, int64_t);
REGISTER_INDEX_SELECT_5D_1D(uint64_t, int64_t);

REGISTER_INDEX_SELECT_5D_2D(float, int32_t);
REGISTER_INDEX_SELECT_5D_2D(half, int32_t);
REGISTER_INDEX_SELECT_5D_2D(bfloat16_t, int32_t);
REGISTER_INDEX_SELECT_5D_2D(int8_t, int32_t);
REGISTER_INDEX_SELECT_5D_2D(int16_t, int32_t);
REGISTER_INDEX_SELECT_5D_2D(int32_t, int32_t);
REGISTER_INDEX_SELECT_5D_2D(int64_t, int32_t);
REGISTER_INDEX_SELECT_5D_2D(uint8_t, int32_t);
REGISTER_INDEX_SELECT_5D_2D(uint16_t, int32_t);
REGISTER_INDEX_SELECT_5D_2D(uint32_t, int32_t);
REGISTER_INDEX_SELECT_5D_2D(uint64_t, int32_t);
REGISTER_INDEX_SELECT_5D_2D(float, int64_t);
REGISTER_INDEX_SELECT_5D_2D(half, int64_t);
REGISTER_INDEX_SELECT_5D_2D(bfloat16_t, int64_t);
REGISTER_INDEX_SELECT_5D_2D(int8_t, int64_t);
REGISTER_INDEX_SELECT_5D_2D(int16_t, int64_t);
REGISTER_INDEX_SELECT_5D_2D(int32_t, int64_t);
REGISTER_INDEX_SELECT_5D_2D(int64_t, int64_t);
REGISTER_INDEX_SELECT_5D_2D(uint8_t, int64_t);
REGISTER_INDEX_SELECT_5D_2D(uint16_t, int64_t);
REGISTER_INDEX_SELECT_5D_2D(uint32_t, int64_t);
REGISTER_INDEX_SELECT_5D_2D(uint64_t, int64_t);
}

#endif
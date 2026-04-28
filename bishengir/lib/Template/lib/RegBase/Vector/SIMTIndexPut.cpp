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

const int32_t MAX_THREAD_NUM=1024;
constexpr unsigned int VEC_SIZE = 4;

template<typename SRCTY, typename IDXTY>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void 
simt_index_put_core_2d(
  __ubuf__ const SRCTY *val, __ubuf__ const IDXTY *idx, __gm__ SRCTY *dst,
  const int64_t indexBoundary,
  const int32_t size0, const int32_t size1,
  const int32_t startOffset0, const int32_t startOffset1,
  const int32_t endOffset0, const int32_t endOffset1,
  const int32_t dst_stride0, const int32_t dst_stride1,
  const int32_t idx_stride0,
  const int32_t val_stride0, const int32_t val_stride1) {

  const int32_t batchLimit = min(endOffset0 - startOffset0, size0);
  const int32_t xLimit = min(endOffset1 - startOffset1, size1);

  for (int32_t batch_i = threadIdx.y; batch_i < batchLimit; batch_i += blockDim.y) {
    IDXTY index = idx[batch_i * idx_stride0];
    if (index < 0) {
      index += indexBoundary;
    }
    if (index < 0 || index >= indexBoundary) {
      continue;
    }
    for (int32_t x_i = threadIdx.x; x_i < xLimit; x_i += blockDim.x) {
      dst[index * dst_stride0 + (startOffset1 + x_i) * dst_stride1] =
          val[batch_i * val_stride0 + x_i * val_stride1];
    }
  }
}

template<typename SRCTY, typename IDXTY>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_index_put_core_2d_opt_vec(
  __ubuf__ const SRCTY *val, __ubuf__ const IDXTY *idx, __gm__ SRCTY *dst,
  const int64_t indexBoundary,
  const int32_t size0, const int32_t size1,
  const int32_t startOffset0, const int32_t startOffset1,
  const int32_t endOffset0, const int32_t endOffset1,
  const int32_t dst_stride0, const int32_t dst_stride1,
  const int32_t idx_stride0,
  const int32_t val_stride0, const int32_t val_stride1) {

  __gm__ float4 *dst4 = reinterpret_cast<__gm__ float4 *>(dst);
  __ubuf__ const float4 *val4 = reinterpret_cast<__ubuf__ const float4 *>(val);

  const int32_t batchLimit = min(endOffset0 - startOffset0, size0);
  const int32_t xLimit = min(endOffset1 - startOffset1, size1);
  const int32_t startOffset1Div4 = startOffset1 / VEC_SIZE;
  for (int32_t batch_i = threadIdx.y; batch_i < batchLimit; batch_i += blockDim.y) {
    IDXTY index = idx[batch_i * idx_stride0];
    if (index < 0) {
      index += indexBoundary;
    }
    if (index < 0 || index >= indexBoundary) {
      continue;
    }

    for (int32_t x_i = threadIdx.x; x_i * VEC_SIZE < xLimit; x_i += blockDim.x) {
      dst4[index * dst_stride0 + (startOffset1Div4 + x_i) * dst_stride1] =
          val4[batch_i * val_stride0 + x_i * val_stride1];
    }
  }
}

template<typename SRCTY, typename IDXTY>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_index_put_core_3d(
  __ubuf__ const SRCTY *val, __ubuf__ const IDXTY *idx, __gm__ SRCTY *dst,
  const int32_t scatterDim, const int64_t indexBoundary,
  const int32_t size0, const int32_t size1, const int32_t size2,
  const int32_t startOffset0, const int32_t startOffset1, const int32_t startOffset2,
  const int32_t endOffset0, const int32_t endOffset1, const int32_t endOffset2,
  const int32_t dst_stride0, const int32_t dst_stride1, const int32_t dst_stride2,
  const int32_t idx_stride0,
  const int32_t val_stride0, const int32_t val_stride1, const int32_t val_stride2) {

  IDXTY index;
  const int32_t dim0Limit = min(endOffset0 - startOffset0, size0);
  const int32_t dim1Limit = min(endOffset1 - startOffset1, size1);
  const int32_t dim2Limit = min(endOffset2 - startOffset2, size2);

  if (scatterDim == 0){
    for (int32_t i = threadIdx.z; i < dim0Limit; i += blockDim.z) {
      index = idx[i * idx_stride0];
      if (index < 0) {
        index += indexBoundary;
      }
      if (index < 0 || index >= indexBoundary) {
        continue;
      }
      for (int32_t j = threadIdx.y; j < dim1Limit; j += blockDim.y) {
        for (int32_t k = threadIdx.x; k < dim2Limit; k += blockDim.x) {
          dst[index * dst_stride0 + (startOffset1 + j) * dst_stride1 + (startOffset2 + k)] =
            val[i * val_stride0 + j * val_stride1 + k];
        }
      }
    }
  } else {
    for (int32_t j = threadIdx.y; j < dim1Limit; j += blockDim.y) {
      index = idx[j * idx_stride0];
      if (index < 0) {
        index += indexBoundary;
      }
      if (index < 0 || index >= indexBoundary) {
        continue;
      }
      for (int32_t i = threadIdx.z; i < dim0Limit; i += blockDim.z) {
        for (int32_t k = threadIdx.x; k < dim2Limit; k += blockDim.x) {
          dst[(startOffset0 + i) * dst_stride0 + index * dst_stride1 + (startOffset2 + k)] =
            val[i * val_stride0 + j * val_stride1 + k];
        }
      }
    }
  }
}

template<typename SRCTY, typename IDXTY>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void simt_index_put_core_4d(
  __ubuf__ const SRCTY *val, __ubuf__ const IDXTY *idx, __gm__ SRCTY *dst,
  const int32_t scatterDim, const int64_t indexBoundary,
  const int32_t total_size,
  const int32_t size0, const int32_t size1, const int32_t size2, const int32_t size3,
  const int32_t startOffset0, const int32_t startOffset1, const int32_t startOffset2, const int32_t startOffset3,
  const int32_t endOffset0, const int32_t endOffset1, const int32_t endOffset2, const int32_t endOffset3,
  const int32_t dst_stride0, const int32_t dst_stride1, const int32_t dst_stride2, const int32_t dst_stride3,
  const int32_t idx_stride0,
  const int32_t val_stride0, const int32_t val_stride1, const int32_t val_stride2, const int32_t val_stride3) {

  IDXTY index;
  int32_t vidx[4] = {}, tmp;
  int32_t didx[4] = {};
  const int32_t endOffsets[4] = {endOffset0, endOffset1, endOffset2, endOffset3};
  const int32_t val_strides[4] = {val_stride0, val_stride1, val_stride2, val_stride3};
  const int32_t dst_strides[4] = {dst_stride0, dst_stride1, dst_stride2, dst_stride3};

  bool flag=false;
  IDXTY dst_pos, val_pos;

  for (int32_t pos = threadIdx.x; pos < total_size; pos +=blockDim.x) {
    tmp = pos;
    vidx[3] = tmp % size3; tmp /= size3;
    vidx[2] = tmp % size2; tmp /= size2;
    vidx[1] = tmp % size1; tmp /= size1;
    vidx[0] = tmp;

    didx[0] = vidx[0] + startOffset0;
    didx[1] = vidx[1] + startOffset1;
    didx[2] = vidx[2] + startOffset2;
    didx[3] = vidx[3] + startOffset3;

    index = idx[vidx[scatterDim] * idx_stride0];
    if (index < 0) index += indexBoundary;
    if (index < 0) continue;
    didx[scatterDim] = index;

    flag=false;
    dst_pos = val_pos = 0;
    for (int i = 0;i < 4; i++){
      if (didx[i] >= endOffsets[i]) {flag=true; break;}
      dst_pos += didx[i] * dst_strides[i];
      val_pos += vidx[i] * val_strides[i];
    }
    if (flag) continue;

    dst[dst_pos] = val[val_pos];
  }
}

template<typename SRCTY, typename IDXTY>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void simt_index_put_core_5d(
  __ubuf__ const SRCTY *val, __ubuf__ const IDXTY *idx, __gm__ SRCTY *dst,
  const int32_t scatterDim, const int64_t indexBoundary,
  const int32_t total_size,
  const int32_t size0, const int32_t size1, const int32_t size2, const int32_t size3, const int32_t size4,
  const int32_t startOffset0, const int32_t startOffset1, const int32_t startOffset2, const int32_t startOffset3, const int32_t startOffset4,
  const int32_t endOffset0, const int32_t endOffset1, const int32_t endOffset2, const int32_t endOffset3, const int32_t endOffset4,
  const int32_t dst_stride0, const int32_t dst_stride1, const int32_t dst_stride2, const int32_t dst_stride3, const int32_t dst_stride4,
  const int32_t idx_stride0,
  const int32_t val_stride0, const int32_t val_stride1, const int32_t val_stride2, const int32_t val_stride3, const int32_t val_stride4) {

  IDXTY index;
  int32_t vidx[5] = {}, tmp;
  int32_t didx[5] = {};
  const int32_t endOffsets[4] = {endOffset0, endOffset1, endOffset2, endOffset3};
  const int32_t val_strides[5] = {val_stride0, val_stride1, val_stride2, val_stride3, val_stride4};
  const int32_t dst_strides[5] = {dst_stride0, dst_stride1, dst_stride2, dst_stride3, dst_stride4};

  bool flag=false;
  IDXTY dst_pos, val_pos;

  for (int32_t pos = threadIdx.x; pos < total_size; pos +=blockDim.x) {
    tmp = pos;
    vidx[4] = tmp % size4; tmp /= size4;
    vidx[3] = tmp % size3; tmp /= size3;
    vidx[2] = tmp % size2; tmp /= size2;
    vidx[1] = tmp % size1; tmp /= size1;
    vidx[0] = tmp;

    didx[0] = vidx[0] + startOffset0;
    didx[1] = vidx[1] + startOffset1;
    didx[2] = vidx[2] + startOffset2;
    didx[3] = vidx[3] + startOffset3;
    didx[4] = vidx[4] + startOffset4;

    index = idx[vidx[scatterDim] * idx_stride0];
    if (index < 0) index += indexBoundary;
    if (index < 0) continue;
    didx[scatterDim] = index;

    flag=false;
    dst_pos = val_pos = 0;
    for (int i = 0;i < 5; i++){
      if (didx[i] >= endOffsets[i]) {flag=true; break;}
      dst_pos += didx[i] * dst_strides[i];
      val_pos += vidx[i] * val_strides[i];
    }
    if (flag) continue;

    dst[dst_pos] = val[val_pos];
  }
}

template <int DIM, typename SRCTY, typename IDXTY>
__aiv__ __attribute__((always_inline)) void simt_index_put(
  memref_t<__ubuf__ SRCTY, DIM> *val, memref_t<__ubuf__ IDXTY, 1> *idx,
  memref_t<__gm__ SRCTY, 1> *dst,
  const int32_t scatterDim,
  const int64_t indexBoundary,
  const IDXTY* endOffsets,
  const IDXTY* startOffsets,
  const IDXTY* dstStrides
) {

  static_assert(DIM <= 5, "DIM must be <= 5");

  static_assert(std::is_same<SRCTY, float>::value ||
                std::is_same<SRCTY, half>::value ||
                std::is_same<SRCTY, bfloat16_t>::value,
              "Currently dtype of source supports only float32|half|bfloat16");

  static_assert(std::is_same<IDXTY, int32_t>::value ||
                  std::is_same<IDXTY, int64_t>::value,
              "Currently dtype of index supports only int32|int64");

  const IDXTY &startOffset0 = startOffsets[0];
  const IDXTY &endOffset0 = endOffsets[0];
  const IDXTY &startOffset1 = startOffsets[1];
  const IDXTY &endOffset1 = endOffsets[1];

  auto val_ptr = val->aligned + val->offset;
  auto idx_ptr = idx->aligned + idx->offset;
  auto dst_ptr = dst->aligned;
  const IDXTY size0 = val->sizes[0];
  const IDXTY size1 = val->sizes[1];

  const IDXTY idx_stride0 = idx->strides[0];

  const IDXTY val_stride0 = val->strides[0];
  const IDXTY val_stride1 = val->strides[1];

  unsigned int block_dim_x = 1;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;

  if constexpr (DIM == 2) {

    if (std::is_same<SRCTY, float>::value && size1 % VEC_SIZE == 0 && dstStrides[0] % VEC_SIZE ==0
        && startOffset1 % VEC_SIZE ==0 && endOffset1 % VEC_SIZE ==0) {
      if (size1 >= MAX_THREAD_NUM * VEC_SIZE) {
        // note: block_dim_x does not finish the complete axis 1 of size1
        block_dim_x = MAX_THREAD_NUM / VEC_SIZE;
      } else if (size1 * size0 >= MAX_THREAD_NUM * VEC_SIZE) {
        block_dim_x = size1 / VEC_SIZE;
        block_dim_y = MAX_THREAD_NUM / block_dim_x;
      } else {
        block_dim_x = size1 / VEC_SIZE;
        block_dim_y = size0;
      }

      cce::async_invoke<simt_index_put_core_2d_opt_vec<SRCTY, IDXTY> >(
        cce::dim3{block_dim_x, block_dim_y, 1}, 
        val_ptr, idx_ptr, dst_ptr,
        indexBoundary,
        size0, size1,
        startOffset0, startOffset1,
        endOffset0, endOffset1,
        /*dst_stride0=*/dstStrides[0] / VEC_SIZE, 
        /*dst_stride1=*/dstStrides[1],
        idx_stride0,
        val_stride0 / VEC_SIZE, 
        val_stride1
      );

    } else {
      // currently no float4 is used. maybe in future we use float2, float3
      // to handle this situation
      if (size1 >= MAX_THREAD_NUM) {
        // note: block_dim_x does not finish the complete axis 1 of size1
        block_dim_x = MAX_THREAD_NUM;
      } else if (size1 * size0 >= MAX_THREAD_NUM) {
        block_dim_x = size1;
        block_dim_y = MAX_THREAD_NUM / block_dim_x;
      } else {
        block_dim_x = size1;
        block_dim_y = size0;
      }
      cce::async_invoke<simt_index_put_core_2d<SRCTY, IDXTY> >(
            cce::dim3{block_dim_x, block_dim_y, 1},
            val_ptr, idx_ptr, dst_ptr,
            indexBoundary, 
            size0, size1,
            startOffset0, startOffset1,
            endOffset0, endOffset1,
            dstStrides[0], dstStrides[1],
            idx_stride0,
            val_stride0,
            val_stride1
        );
    }
  } else if constexpr (DIM == 3) {
    const IDXTY &startOffset2 = startOffsets[2];
    const IDXTY &endOffset2 = endOffsets[2];

    const IDXTY size2 = val->sizes[2];
    const IDXTY val_stride2 = val->strides[2];
    if (size2 >= MAX_THREAD_NUM) {
      block_dim_x = MAX_THREAD_NUM;
    } else if (size2 * size1 >= MAX_THREAD_NUM) {
      block_dim_x = size2;
      block_dim_y = MAX_THREAD_NUM / block_dim_x;
    } else if (size2 * size1 * size0 >= MAX_THREAD_NUM) {
      block_dim_x = size2;
      block_dim_y = size1;
      block_dim_z = MAX_THREAD_NUM / (block_dim_x * block_dim_y);
    } else {
      block_dim_x = size2;
      block_dim_y = size1;
      block_dim_z = size0;
    }

    cce::async_invoke<simt_index_put_core_3d<SRCTY, IDXTY> >(
          cce::dim3{block_dim_x, block_dim_y, block_dim_z},
          val_ptr, idx_ptr, dst_ptr,
          scatterDim, indexBoundary,
          size0, size1, size2,
          startOffset0, startOffset1, startOffset2,
          endOffset0, endOffset1, endOffset2,
          dstStrides[0], dstStrides[1], dstStrides[2],
          idx_stride0,
          val_stride0,
          val_stride1,
          val_stride2
    );
  } else if constexpr (DIM == 4) {
    const IDXTY &startOffset2 = startOffsets[2];
    const IDXTY &endOffset2 = endOffsets[2];
    const IDXTY size2 = val->sizes[2];
    const IDXTY val_stride2 = val->strides[2];

    const IDXTY &startOffset3 = startOffsets[3];
    const IDXTY &endOffset3 = endOffsets[3];
    const IDXTY size3 = val->sizes[3];
    const IDXTY val_stride3 = val->strides[3];

    const int32_t total_size = size0 * size1 * size2 * size3;
    cce::async_invoke<simt_index_put_core_4d<SRCTY, IDXTY> >(
          cce::dim3{MAX_THREAD_NUM},
          val_ptr, idx_ptr, dst_ptr,
          scatterDim, indexBoundary,
          total_size,
          size0, size1, size2, size3,
          startOffset0, startOffset1, startOffset2, startOffset3,
          endOffset0, endOffset1, endOffset2, endOffset3,
          dstStrides[0], dstStrides[1], dstStrides[2], dstStrides[3],
          idx_stride0,
          val_stride0, val_stride1, val_stride2, val_stride3
    );
  } else if constexpr (DIM == 5) {
    const IDXTY &startOffset2 = startOffsets[2];
    const IDXTY &endOffset2 = endOffsets[2];
    const IDXTY size2 = val->sizes[2];
    const IDXTY val_stride2 = val->strides[2];

    const IDXTY &startOffset3 = startOffsets[3];
    const IDXTY &endOffset3 = endOffsets[3];
    const IDXTY size3 = val->sizes[3];
    const IDXTY val_stride3 = val->strides[3];

    const IDXTY &startOffset4 = startOffsets[4];
    const IDXTY &endOffset4 = endOffsets[4];
    const IDXTY size4 = val->sizes[4];
    const IDXTY val_stride4 = val->strides[4];

    const int32_t total_size = size0 * size1 * size2 * size3 * size4;
    cce::async_invoke<simt_index_put_core_5d<SRCTY, IDXTY> >(
          cce::dim3{MAX_THREAD_NUM},
          val_ptr, idx_ptr, dst_ptr,
          scatterDim, indexBoundary,
          total_size,
          size0, size1, size2, size3, size4,
          startOffset0, startOffset1, startOffset2, startOffset3, startOffset4,
          endOffset0, endOffset1, endOffset2, endOffset3, endOffset4,
          dstStrides[0], dstStrides[1], dstStrides[2], dstStrides[3], dstStrides[4],
          idx_stride0,
          val_stride0, val_stride1, val_stride2, val_stride3, val_stride4
    );
  }
}

extern "C" {
REGISTER_INDEX_PUT_2D(float, int32_t);
REGISTER_INDEX_PUT_2D(half, int32_t);
REGISTER_INDEX_PUT_2D(bfloat16_t, int32_t);
REGISTER_INDEX_PUT_2D(float, int64_t);
REGISTER_INDEX_PUT_2D(half, int64_t);
REGISTER_INDEX_PUT_2D(bfloat16_t, int64_t);

REGISTER_INDEX_PUT_3D(float, int32_t);
REGISTER_INDEX_PUT_3D(half, int32_t);
REGISTER_INDEX_PUT_3D(bfloat16_t, int32_t);
REGISTER_INDEX_PUT_3D(float, int64_t);
REGISTER_INDEX_PUT_3D(half, int64_t);
REGISTER_INDEX_PUT_3D(bfloat16_t, int64_t);

REGISTER_INDEX_PUT_4D(float, int32_t);
REGISTER_INDEX_PUT_4D(half, int32_t);
REGISTER_INDEX_PUT_4D(bfloat16_t, int32_t);
REGISTER_INDEX_PUT_4D(float, int64_t);
REGISTER_INDEX_PUT_4D(half, int64_t);
REGISTER_INDEX_PUT_4D(bfloat16_t, int64_t);

REGISTER_INDEX_PUT_5D(float, int32_t);
REGISTER_INDEX_PUT_5D(half, int32_t);
REGISTER_INDEX_PUT_5D(bfloat16_t, int32_t);
REGISTER_INDEX_PUT_5D(float, int64_t);
REGISTER_INDEX_PUT_5D(half, int64_t);
REGISTER_INDEX_PUT_5D(bfloat16_t, int64_t);

}

#endif
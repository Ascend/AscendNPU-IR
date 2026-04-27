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

constexpr unsigned int MAX_THREAD_NUM = 1024;
// EMBTY = float
constexpr unsigned int VEC_SIZE = 2;

__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_embedding_gather_core_1d_float_int32_t(
    __gm__ const float *emb, __ubuf__ const int32_t *idx, __ubuf__ float *dst,
    const int32_t bound, const int32_t size0, const int32_t size1,
    const int32_t offset0, const int32_t offset1, const int32_t numel0,
    const int32_t numel1, const int32_t emb_stride0, const int32_t idx_stride0,
    const int32_t dst_stride0, const int32_t dst_stride1) {
  constexpr int32_t emb_stride1 = 1;
  for (int32_t batch_i = threadIdx.y;
       (batch_i < size0) && (offset0 + batch_i < numel0);
       batch_i += blockDim.y) {
    int32_t index = idx[batch_i * idx_stride0];
    if (index < 0) {
      index += bound;
    }
    // WORKAOUND
    if (index >= bound) {
      continue;
    }
    if (index < 0) {
      continue;
    }
    for (int32_t x_i = threadIdx.x; (x_i < size1) && (offset1 + x_i < numel1);
         x_i += blockDim.x) {
      dst[batch_i * dst_stride0 + x_i * dst_stride1] =
          emb[index * emb_stride0 + (offset1 + x_i) * emb_stride1];
    }
  }
}

__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_embedding_gather_core_1d_float_int64_t(
    __gm__ const float *emb, __ubuf__ const int64_t *idx, __ubuf__ float *dst,
    const int64_t bound, const int32_t size0, const int32_t size1,
    const int32_t offset0, const int32_t offset1, const int32_t numel0,
    const int32_t numel1, const int32_t emb_stride0, const int32_t idx_stride0,
    const int32_t dst_stride0, const int32_t dst_stride1) {
  constexpr int32_t emb_stride1 = 1;
  for (int32_t batch_i = threadIdx.y;
       (batch_i < size0) && (offset0 + batch_i < numel0);
       batch_i += blockDim.y) {
    int64_t index = idx[batch_i * idx_stride0];
    if (index < 0) {
      index += bound;
    }
    // WORKAOUND
    if (index >= bound) {
      continue;
    }
    if (index < 0) {
      continue;
    }
    for (int32_t x_i = threadIdx.x; (x_i < size1) && (offset1 + x_i < numel1);
         x_i += blockDim.x) {
      dst[batch_i * dst_stride0 + x_i * dst_stride1] =
          emb[index * emb_stride0 + (offset1 + x_i) * emb_stride1];
    }
  }
}

__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_embedding_gather_core_2d_float_int32_t(
    __gm__ const float *emb, __ubuf__ const int32_t *idx, __ubuf__ float *dst,
    const int32_t bound, const int32_t size0, const int32_t size1,
    const int32_t size2, const int32_t offset0, const int32_t offset1,
    const int32_t offset2, const int32_t numel0, const int32_t numel1,
    const int32_t numel2, const int32_t emb_stride0, const int32_t idx_stride0,
    const int32_t idx_stride1, const int32_t dst_stride0,
    const int32_t dst_stride1, const int32_t dst_stride2) {
  constexpr int32_t emb_stride1 = 1;
  for (int32_t batch_i = threadIdx.z;
       (batch_i < size0) && (offset0 + batch_i < numel0);
       batch_i += blockDim.z) {
    int32_t batch_idx = batch_i * idx_stride0;
    for (int32_t reduce_i = threadIdx.y;
         (reduce_i < size1) && (offset1 + reduce_i < numel1);
         reduce_i += blockDim.y) {
      int32_t index = idx[batch_idx + reduce_i * idx_stride1];
      if (index < 0) {
        index += bound;
      }
      // WORKAOUND
      if (index >= bound) {
        continue;
      }
      if (index < 0) {
        continue;
      }
      for (int32_t x_i = threadIdx.x; (x_i < size2) && (offset2 + x_i < numel2);
           x_i += blockDim.x) {
        dst[batch_i * dst_stride0 + reduce_i * dst_stride1 +
            x_i * dst_stride2] =
            emb[index * emb_stride0 + (offset2 + x_i) * emb_stride1];
      }
    }
  }
}

__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_embedding_gather_core_2d_float_int64_t(
    __gm__ const float *emb, __ubuf__ const int64_t *idx, __ubuf__ float *dst,
    const int64_t bound, const int32_t size0, const int32_t size1,
    const int32_t size2, const int32_t offset0, const int32_t offset1,
    const int32_t offset2, const int32_t numel0, const int32_t numel1,
    const int32_t numel2, const int32_t emb_stride0, const int32_t idx_stride0,
    const int32_t idx_stride1, const int32_t dst_stride0,
    const int32_t dst_stride1, const int32_t dst_stride2) {
  constexpr int32_t emb_stride1 = 1;
  for (int32_t batch_i = threadIdx.z;
       (batch_i < size0) && (offset0 + batch_i < numel0);
       batch_i += blockDim.z) {
    int32_t batch_idx = batch_i * idx_stride0;
    for (int32_t reduce_i = threadIdx.y;
         (reduce_i < size1) && (offset1 + reduce_i < numel1);
         reduce_i += blockDim.y) {
      int32_t reduce_idx = batch_idx + reduce_i * idx_stride1;
      int64_t index = idx[reduce_idx];
      if (index < 0) {
        index += bound;
      }
      // WORKAOUND
      if (index >= bound) {
        continue;
      }
      if (index < 0) {
        continue;
      }
      for (int32_t x_i = threadIdx.x; (x_i < size2) && (offset2 + x_i < numel2);
           x_i += blockDim.x) {
        dst[batch_i * dst_stride0 + reduce_i * dst_stride1 +
            x_i * dst_stride2] =
            emb[index * emb_stride0 + (offset2 + x_i) * emb_stride1];
      }
    }
  }
}

__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_embedding_gather_core_1d_float_int32_t_opt_vec(
    __gm__ const float *emb, __ubuf__ const int32_t *idx, __ubuf__ float *dst,
    const int32_t bound, const int32_t size0, const int32_t size1,
    const int32_t offset0, const int32_t offset1, const int32_t numel0,
    const int32_t numel1, const int32_t emb_stride0, const int32_t idx_stride0,
    const int32_t dst_stride0, const int32_t dst_stride1) {
  constexpr int32_t emb_stride1 = 1;
  __gm__ const float2 *emb4 = reinterpret_cast<__gm__ const float2 *>(emb);
  __ubuf__ float2 *dst4 = reinterpret_cast<__ubuf__ float2 *>(dst);
  for (int32_t batch_i = threadIdx.y;
       (batch_i < size0) && (offset0 + batch_i < numel0);
       batch_i += blockDim.y) {
    int32_t index = idx[batch_i * idx_stride0];
    if (index < 0) {
      index += bound;
    }
    // WORKAOUND
    if (index >= bound) {
      continue;
    }
    if (index < 0) {
      continue;
    }
    // assume size1 = numel1 then we could eliminate the bound-check.
    for (int32_t x_i = threadIdx.x;
         (x_i * VEC_SIZE < size1) && (offset1 + x_i * VEC_SIZE < numel1);
         x_i += blockDim.x) {
      dst4[batch_i * dst_stride0 + x_i * dst_stride1] =
          emb4[index * emb_stride0 + (offset1 + x_i) * emb_stride1];
    }
  }
}

__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_embedding_gather_core_1d_float_int64_t_opt_vec(
    __gm__ const float *emb, __ubuf__ const int64_t *idx, __ubuf__ float *dst,
    const int64_t bound, const int32_t size0, const int32_t size1,
    const int32_t offset0, const int32_t offset1, const int32_t numel0,
    const int32_t numel1, const int32_t emb_stride0, const int32_t idx_stride0,
    const int32_t dst_stride0, const int32_t dst_stride1) {
  constexpr int32_t emb_stride1 = 1;
  __gm__ const float2 *emb4 = reinterpret_cast<__gm__ const float2 *>(emb);
  __ubuf__ float2 *dst4 = reinterpret_cast<__ubuf__ float2 *>(dst);
  for (int32_t batch_i = threadIdx.y;
       (batch_i < size0) && (offset0 + batch_i < numel0);
       batch_i += blockDim.y) {
    int64_t index = idx[batch_i * idx_stride0];
    if (index < 0) {
      index += bound;
    }
    // WORKAOUND
    if (index >= bound) {
      continue;
    }
    if (index < 0) {
      continue;
    }
    // assume size1 = numel1 then we could eliminate the bound-check.
    for (int32_t x_i = threadIdx.x;
         (x_i * VEC_SIZE < size1) && (offset1 + x_i * VEC_SIZE < numel1);
         x_i += blockDim.x) {
      dst4[batch_i * dst_stride0 + x_i * dst_stride1] =
          emb4[index * emb_stride0 + (offset1 + x_i) * emb_stride1];
    }
  }
}

__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_embedding_gather_core_2d_float_int32_t_opt_vec(
    __gm__ const float *emb, __ubuf__ const int32_t *idx, __ubuf__ float *dst,
    const int32_t bound, const int32_t size0, const int32_t size1,
    const int32_t size2, const int32_t offset0, const int32_t offset1,
    const int32_t offset2, const int32_t numel0, const int32_t numel1,
    const int32_t numel2, const int32_t emb_stride0, const int32_t idx_stride0,
    const int32_t idx_stride1, const int32_t dst_stride0,
    const int32_t dst_stride1, const int32_t dst_stride2) {
  constexpr int32_t emb_stride1 = 1;
  __gm__ const float2 *emb4 = reinterpret_cast<__gm__ const float2 *>(emb);
  __ubuf__ float2 *dst4 = reinterpret_cast<__ubuf__ float2 *>(dst);
  for (int32_t batch_i = threadIdx.z;
       (batch_i < size0) && (offset0 + batch_i < numel0);
       batch_i += blockDim.z) {
    int32_t batch_idx = batch_i * idx_stride0;
    // assume size1 = numel1 then we could eliminate the bound-check.
    for (int32_t reduce_i = threadIdx.y;
         (reduce_i < size1) && (offset1 + reduce_i < numel1);
         reduce_i += blockDim.y) {
      int32_t index = idx[batch_idx + reduce_i * idx_stride1];
      if (index < 0) {
        index += bound;
      }
      // WORKAOUND
      if (index >= bound) {
        continue;
      }
      if (index < 0) {
        continue;
      }
      // assume size2 = numel2 then we could eliminate the bound-check.
      for (int32_t x_i = threadIdx.x;
           (x_i * VEC_SIZE < size2) && (offset2 + x_i * VEC_SIZE < numel2);
           x_i += blockDim.x) {
        dst4[batch_i * dst_stride0 + reduce_i * dst_stride1 +
             x_i * dst_stride2] =
            emb4[index * emb_stride0 + (offset2 + x_i) * emb_stride1];
      }
    }
  }
}

__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_embedding_gather_core_2d_float_int64_t_opt_vec(
    __gm__ const float *emb, __ubuf__ const int64_t *idx, __ubuf__ float *dst,
    const int64_t bound, const int32_t size0, const int32_t size1,
    const int32_t size2, const int32_t offset0, const int32_t offset1,
    const int32_t offset2, const int32_t numel0, const int32_t numel1,
    const int32_t numel2, const int32_t emb_stride0, const int32_t idx_stride0,
    const int32_t idx_stride1, const int32_t dst_stride0,
    const int32_t dst_stride1, const int32_t dst_stride2) {
  constexpr int32_t emb_stride1 = 1;
  __gm__ const float2 *emb4 = reinterpret_cast<__gm__ const float2 *>(emb);
  __ubuf__ float2 *dst4 = reinterpret_cast<__ubuf__ float2 *>(dst);
  for (int32_t batch_i = threadIdx.z;
       (batch_i < size0) && (offset0 + batch_i < numel0);
       batch_i += blockDim.z) {
    int32_t batch_idx = batch_i * idx_stride0;
    // assume size1 = numel1 then we could eliminate the bound-check.
    for (int32_t reduce_i = threadIdx.y;
         (reduce_i < size1) && (offset1 + reduce_i < numel1);
         reduce_i += blockDim.y) {
      int64_t index = idx[batch_idx + reduce_i * idx_stride1];
      if (index < 0) {
        index += bound;
      }
      // WORKAOUND
      if (index >= bound) {
        continue;
      }
      if (index < 0) {
        continue;
      }
      // assume size2 = numel2 then we could eliminate the bound-check.
      for (int32_t x_i = threadIdx.x;
           (x_i * VEC_SIZE < size2) && (offset2 + x_i * VEC_SIZE < numel2);
           x_i += blockDim.x) {
        dst4[batch_i * dst_stride0 + reduce_i * dst_stride1 +
             x_i * dst_stride2] =
            emb4[index * emb_stride0 + (offset2 + x_i) * emb_stride1];
      }
    }
  }
}

/// Assume
/// - emb is 2D continuous tensor in GM. Thus the stride1 = 1, stride0 = numel
///   of the lowest dim.
/// - emb is of float type
template <int DIM, typename EMBTY, typename IDXTY>
__aiv__ __attribute__((always_inline)) void simt_embedding_gather(
    memref_t<__gm__ EMBTY, 1> *emb, memref_t<__ubuf__ IDXTY, DIM> *idx,
    memref_t<__ubuf__ EMBTY, DIM + 1> *dst, IDXTY bound, IDXTY *offs_nums) {

  static_assert(std::is_same<EMBTY, float>::value,
                "Currently dtype of embedding supports only float");
  static_assert(std::is_same<IDXTY, int32_t>::value ||
                    std::is_same<IDXTY, int64_t>::value,
                "Currently dtype of index supports only int32|int64");

  IDXTY &offset0 = offs_nums[0];
  IDXTY &numel0 = offs_nums[1];
  IDXTY &offset1 = offs_nums[2];
  IDXTY &numel1 = offs_nums[3];

  auto src_ptr = emb->aligned;
  auto idx_ptr = idx->aligned + idx->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  const IDXTY size0 = dst->sizes[0];
  const IDXTY size1 = dst->sizes[1];

  const IDXTY idx_stride0 = idx->strides[0];
  const IDXTY dst_stride0 = dst->strides[0];
  const IDXTY dst_stride1 = dst->strides[1];

  unsigned int block_dim_x = 1;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;

  if constexpr (DIM == 1) {

    if (size1 % VEC_SIZE == 0) {
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

      if constexpr (std::is_same<IDXTY, int32_t>::value) {
        cce::async_invoke<simt_embedding_gather_core_1d_float_int32_t_opt_vec>(
            cce::dim3{block_dim_x, block_dim_y, 1}, src_ptr, idx_ptr, dst_ptr,
            bound, size0, size1, offset0, offset1, numel0, numel1,
            /*emb_stride0=*/numel1 / VEC_SIZE, idx_stride0,
            dst_stride0 / VEC_SIZE, dst_stride1);
      } else if constexpr (std::is_same<IDXTY, int64_t>::value) {
        cce::async_invoke<simt_embedding_gather_core_1d_float_int64_t_opt_vec>(
            cce::dim3{block_dim_x, block_dim_y, 1}, src_ptr, idx_ptr, dst_ptr,
            bound, size0, size1, offset0, offset1, numel0, numel1,
            /*emb_stride0=*/numel1 / VEC_SIZE, idx_stride0,
            dst_stride0 / VEC_SIZE, dst_stride1);
      } // IDXTY
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

      if constexpr (std::is_same<IDXTY, int32_t>::value) {
        cce::async_invoke<simt_embedding_gather_core_1d_float_int32_t>(
            cce::dim3{block_dim_x, block_dim_y, 1}, src_ptr, idx_ptr, dst_ptr,
            bound, size0, size1, offset0, offset1, numel0, numel1,
            /*emb_stride0=*/numel1, idx_stride0, dst_stride0, dst_stride1);
      } else if constexpr (std::is_same<IDXTY, int64_t>::value) {
        cce::async_invoke<simt_embedding_gather_core_1d_float_int64_t>(
            cce::dim3{block_dim_x, block_dim_y, 1}, src_ptr, idx_ptr, dst_ptr,
            bound, size0, size1, offset0, offset1, numel0, numel1,
            /*emb_stride0=*/numel1, idx_stride0, dst_stride0, dst_stride1);
      } // IDXTY
    }   // size1 % VEC_SIZE == 0

  } else if constexpr (DIM == 2) {
    const IDXTY idx_stride1 = idx->strides[1];
    const IDXTY size2 = dst->sizes[2];
    if (size2 % VEC_SIZE == 0) {
      if (size2 >= MAX_THREAD_NUM * VEC_SIZE) {
        block_dim_x = MAX_THREAD_NUM;
      } else if (size2 * size1 >= MAX_THREAD_NUM * VEC_SIZE) {
        block_dim_x = size2 / VEC_SIZE;
        block_dim_y = MAX_THREAD_NUM / block_dim_x;
      } else if (size2 * size1 * size0 >= MAX_THREAD_NUM * VEC_SIZE) {
        block_dim_x = size2 / VEC_SIZE;
        block_dim_y = size1;
        block_dim_z = MAX_THREAD_NUM / (block_dim_x * block_dim_y);
      } else {
        block_dim_x = size2 / VEC_SIZE;
        block_dim_y = size1;
        block_dim_z = size0;
      }

      if constexpr (std::is_same<IDXTY, int32_t>::value) {
        cce::async_invoke<simt_embedding_gather_core_2d_float_int32_t_opt_vec>(
            cce::dim3{block_dim_x, block_dim_y, block_dim_z}, src_ptr, idx_ptr,
            dst_ptr, bound, size0, size1, dst->sizes[2], offset0, offset1,
            offs_nums[4], numel0, numel1, offs_nums[5],
            /*emb_stride0=*/offs_nums[5] / VEC_SIZE, idx_stride0, idx_stride1,
            dst_stride0 / VEC_SIZE, dst_stride1 / VEC_SIZE, dst->strides[2]);
      } else if constexpr (std::is_same<IDXTY, int64_t>::value) {
        cce::async_invoke<simt_embedding_gather_core_2d_float_int64_t_opt_vec>(
            cce::dim3{block_dim_x, block_dim_y, block_dim_z}, src_ptr, idx_ptr,
            dst_ptr, bound, size0, size1, dst->sizes[2], offset0, offset1,
            offs_nums[4], numel0, numel1, offs_nums[5],
            /*emb_stride0=*/offs_nums[5] / VEC_SIZE, idx_stride0, idx_stride1,
            dst_stride0 / VEC_SIZE, dst_stride1 / VEC_SIZE, dst->strides[2]);
      } // IDXTY
    } else {
      // currently no float4 is used. maybe in future we use float2, float3
      // to handle this situation
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

      if constexpr (std::is_same<IDXTY, int32_t>::value) {
        cce::async_invoke<simt_embedding_gather_core_2d_float_int32_t>(
            cce::dim3{block_dim_x, block_dim_y, block_dim_z}, src_ptr, idx_ptr,
            dst_ptr, bound, size0, size1, dst->sizes[2], offset0, offset1,
            offs_nums[4], numel0, numel1, offs_nums[5],
            /*emb_stride0=*/offs_nums[5], idx_stride0, idx_stride1, dst_stride0,
            dst_stride1, dst->strides[2]);
      } else if constexpr (std::is_same<IDXTY, int64_t>::value) {
        cce::async_invoke<simt_embedding_gather_core_2d_float_int64_t>(
            cce::dim3{block_dim_x, block_dim_y, block_dim_z}, src_ptr, idx_ptr,
            dst_ptr, bound, size0, size1, dst->sizes[2], offset0, offset1,
            offs_nums[4], numel0, numel1, offs_nums[5],
            /*emb_stride0=*/offs_nums[5], idx_stride0, idx_stride1, dst_stride0,
            dst_stride1, dst->strides[2]);
      } // IDXTY
    }   // size2 % VEC_SIZE == 0

  } // DIM
}

extern "C" {
REGISTE_EMBEDDINGGATHER_1D(float, int64_t);
REGISTE_EMBEDDINGGATHER_2D(float, int64_t);
REGISTE_EMBEDDINGGATHER_1D(float, int32_t);
REGISTE_EMBEDDINGGATHER_2D(float, int32_t);
}
#endif
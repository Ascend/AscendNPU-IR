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
constexpr unsigned int THREAD_NUM_512 = 512;

template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore1D(
    __gm__ DTYPE *dst,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *src,
    __ubuf__ bool *mask,
    const uint32_t indice_size0,       // no need to use uint64_t, as UBUF is small.
    const uint32_t indice_stride0,
    const uint32_t src_stride0,
    const uint32_t mask_stride0
)
{
    for (uint32_t i0 = threadIdx.x; i0 < indice_size0; i0 += blockDim.x) {
        if (mask[i0 * mask_stride0])
        {
            ITYPE dstIdx = indices[i0 * indice_stride0];
            dst[dstIdx] = src[i0 * src_stride0];
        }
    }
}

template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore1DWithOutMask(
    __gm__ DTYPE *dst,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *src,
    const uint32_t indice_size0,
    const uint32_t indice_stride0,
    const uint32_t src_stride0
)
{
    for (uint32_t i0 = threadIdx.x; i0 < indice_size0; i0 += blockDim.x) {
        uint32_t indiceIdx = i0 * indice_stride0;
        ITYPE dstIdx = indices[indiceIdx];
        dst[dstIdx] =  src[i0 * src_stride0];
    }
}

template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore2D(
    __gm__ DTYPE *dst,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *src,
    __ubuf__ bool *mask,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t src_stride0,
    const uint32_t src_stride1,
    const uint32_t mask_stride0,
    const uint32_t mask_stride1
) {
    for (uint32_t i0 = threadIdx.y; i0 < indice_size0; i0 += blockDim.y) {
        for (uint32_t i1 = threadIdx.x; i1 < indice_size1; i1 += blockDim.x) {
            if (mask[i0 * mask_stride0 + i1 * mask_stride1])
            {
                ITYPE dstIdx = indices[i0 * indice_stride0 + i1* indice_stride1];
                dst[dstIdx] = src[i0 * src_stride0 + i1 * src_stride1];
            }
        }
    }
}


template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore2DWithOutMask(
    __gm__ DTYPE *dst,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *src,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t src_stride0,
    const uint32_t src_stride1
) {
    for (uint32_t i0 = threadIdx.y; i0 < indice_size0; i0 += blockDim.y) {
        for (uint32_t i1 = threadIdx.x; i1 < indice_size1; i1 += blockDim.x) {
        ITYPE dstIdx = indices[i0 * indice_stride0 + i1* indice_stride1];
        dst[dstIdx] = src[i0 * src_stride0 + i1 * src_stride1];
        }
    }
}


template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_512)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore3D(
    __gm__ DTYPE *dst,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *src,
    __ubuf__ bool *mask,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_size2,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t indice_stride2,
    const uint32_t src_stride0,
    const uint32_t src_stride1,
    const uint32_t src_stride2,
    const uint32_t mask_stride0,
    const uint32_t mask_stride1,
    const uint32_t mask_stride2
) {
    const uint32_t num_elems = indice_size0 * indice_size1 * indice_size2;
    for (uint32_t tx = threadIdx.x; tx < num_elems; tx += blockDim.x) {
        uint32_t elem_idx = tx;
        uint32_t i2 = elem_idx % indice_size2; elem_idx /= indice_size2;
        uint32_t i1 = elem_idx % indice_size1; elem_idx /= indice_size1;
        uint32_t i0 = elem_idx;

        if (mask[i0*mask_stride0 + i1*mask_stride1 + i2*mask_stride2])
        {
            ITYPE dstIdx = indices[i0*indice_stride0+ i1*indice_stride1 + i2*indice_stride2];
            dst[dstIdx] = src[i0*src_stride0 + i1*src_stride1 + i2*src_stride2];
        }
    }
}


template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore3DWithOutMask(
    __gm__ DTYPE *dst,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *src,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_size2,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t indice_stride2,
    const uint32_t src_stride0,
    const uint32_t src_stride1,
    const uint32_t src_stride2
) {
    const uint32_t num_elems = indice_size0 * indice_size1 * indice_size2;
    for (uint32_t tx = threadIdx.x; tx < num_elems; tx += blockDim.x) {
        uint32_t elem_idx = tx;
        uint32_t i2 = elem_idx % indice_size2; elem_idx /= indice_size2;
        uint32_t i1 = elem_idx % indice_size1; elem_idx /= indice_size1;
        uint32_t i0 = elem_idx;
        ITYPE dstIdx = indices[i0*indice_stride0+ i1*indice_stride1 + i2*indice_stride2];
        dst[dstIdx] = src[i0*src_stride0 + i1*src_stride1 + i2*src_stride2];
    }
}


template<typename DTYPE, typename ITYPE >
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_512)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore4D(
    __gm__ DTYPE  *dst,
    __ubuf__ ITYPE  *indices,
    __ubuf__ DTYPE  *src,
    __ubuf__ bool *mask,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_size2,
    const uint32_t indice_size3,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t indice_stride2,
    const uint32_t indice_stride3,
    const uint32_t src_stride0,
    const uint32_t src_stride1,
    const uint32_t src_stride2,
    const uint32_t src_stride3,
    const uint32_t mask_stride0,
    const uint32_t mask_stride1,
    const uint32_t mask_stride2,
    const uint32_t mask_stride3
) {
    const uint32_t num_elems = indice_size0 * indice_size1 * indice_size2 * indice_size3;
    for (uint32_t tx = threadIdx.x; tx < num_elems; tx += blockDim.x) {
        uint32_t elem_idx = tx;
        uint32_t i3 = elem_idx % indice_size3; elem_idx /= indice_size3;
        uint32_t i2 = elem_idx % indice_size2; elem_idx /= indice_size2;
        uint32_t i1 = elem_idx % indice_size1; elem_idx /= indice_size1;
        uint32_t i0 = elem_idx;
        
        if (mask[i0*mask_stride0 + i1*mask_stride1 + i2*mask_stride2 + i3*mask_stride3])
        {
            ITYPE dstIdx = indices[i0*indice_stride0 + i1*indice_stride1 + i2*indice_stride2 + i3*indice_stride3];
            dst[dstIdx] = src[i0*src_stride0 + i1*src_stride1 + i2*src_stride2 + i3*src_stride3];
        }
    }
}


template<typename DTYPE, typename ITYPE >
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_512)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore4DWithOutMask(
    __gm__ DTYPE  *dst,
    __ubuf__ ITYPE  *indices,
    __ubuf__ DTYPE  *src,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_size2,
    const uint32_t indice_size3,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t indice_stride2,
    const uint32_t indice_stride3,
    const uint32_t src_stride0,
    const uint32_t src_stride1,
    const uint32_t src_stride2,
    const uint32_t src_stride3
) {
    const uint32_t num_elems = indice_size0 * indice_size1 * indice_size2 * indice_size3;
    for (uint32_t tx = threadIdx.x; tx < num_elems; tx += blockDim.x) {
        uint32_t elem_idx = tx;
        uint32_t i3 = elem_idx % indice_size3; elem_idx /= indice_size3;
        uint32_t i2 = elem_idx % indice_size2; elem_idx /= indice_size2;
        uint32_t i1 = elem_idx % indice_size1; elem_idx /= indice_size1;
        uint32_t i0 = elem_idx;
        ITYPE dstIdx = indices[i0*indice_stride0 + i1*indice_stride1 + i2*indice_stride2 + i3*indice_stride3];
        dst[dstIdx] = src[i0*src_stride0 + i1*src_stride1 + i2*src_stride2 + i3*src_stride3];
    }
}


template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_512)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore5D(
    __gm__ DTYPE *dst,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *src,
    __ubuf__ bool *mask,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_size2,
    const uint32_t indice_size3,
    const uint32_t indice_size4,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t indice_stride2,
    const uint32_t indice_stride3,
    const uint32_t indice_stride4,
    const uint32_t src_stride0,
    const uint32_t src_stride1,
    const uint32_t src_stride2,
    const uint32_t src_stride3,
    const uint32_t src_stride4,
    const uint32_t mask_stride0,
    const uint32_t mask_stride1,
    const uint32_t mask_stride2,
    const uint32_t mask_stride3,
    const uint32_t mask_stride4
)  {
    const uint32_t num_elems = indice_size0 * indice_size1 * indice_size2 * indice_size3 * indice_size4;
    for (uint32_t tx = threadIdx.x; tx < num_elems; tx += blockDim.x) {
        uint32_t elem_idx = tx;
        uint32_t i4 = elem_idx % indice_size4; elem_idx /= indice_size4;
        uint32_t i3 = elem_idx % indice_size3; elem_idx /= indice_size3;
        uint32_t i2 = elem_idx % indice_size2; elem_idx /= indice_size2;
        uint32_t i1 = elem_idx % indice_size1; elem_idx /= indice_size1;
        uint32_t i0 = elem_idx;
        if (mask[i0*mask_stride0 + i1*mask_stride1 + i2*mask_stride2 + i3*mask_stride3 +  + i4*mask_stride4])
        {
            ITYPE dstIdx = indices[i0*indice_stride0 + i1*indice_stride1 + \
                i2*indice_stride2 + i3*indice_stride3 + i4*indice_stride4];
            dst[dstIdx] = src[i0*src_stride0 + i1*src_stride1 + i2*src_stride2 + i3*src_stride3 + i4*src_stride4];
        }
    }
}


template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_512)
__aiv__ __attribute__((always_inline)) static void SIMTIndirectStore5DWithOutMask(
    __gm__ DTYPE *dst,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *src,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_size2,
    const uint32_t indice_size3,
    const uint32_t indice_size4,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t indice_stride2,
    const uint32_t indice_stride3,
    const uint32_t indice_stride4,
    const uint32_t src_stride0,
    const uint32_t src_stride1,
    const uint32_t src_stride2,
    const uint32_t src_stride3,
    const uint32_t src_stride4
)  {
    const uint32_t num_elems = indice_size0 * indice_size1 * indice_size2 * indice_size3 * indice_size4;
    for (uint32_t tx = threadIdx.x; tx < num_elems; tx += blockDim.x) {
        uint32_t elem_idx = tx;
        uint32_t i4 = elem_idx % indice_size4; elem_idx /= indice_size4;
        uint32_t i3 = elem_idx % indice_size3; elem_idx /= indice_size3;
        uint32_t i2 = elem_idx % indice_size2; elem_idx /= indice_size2;
        uint32_t i1 = elem_idx % indice_size1; elem_idx /= indice_size1;
        uint32_t i0 = elem_idx;

        ITYPE dstIdx = indices[i0*indice_stride0 + i1*indice_stride1 + \
            i2*indice_stride2 + i3*indice_stride3 + i4*indice_stride4];

        dst[dstIdx] = src[i0*src_stride0 + i1*src_stride1 + i2*src_stride2 + i3*src_stride3 + i4*src_stride4];
    }
}

template<int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void indirect_store(
    memref_t<__gm__ DTYPE, 1> *dst,
    memref_t<__ubuf__ ITYPE, DIM> *indices,
    memref_t<__ubuf__ DTYPE, DIM> *src,
    memref_t<__ubuf__ bool, DIM> *mask
) {
    static_assert(DIM <= 5, "DIM must be <= 5");
    static_assert(std::is_same<DTYPE, float>::value ||
                  std::is_same<DTYPE, half>::value ||
                  std::is_same<DTYPE, bfloat16_t>::value ||
                  std::is_same<DTYPE, int8_t>::value ||
                  std::is_same<DTYPE, int16_t>::value ||
                  std::is_same<DTYPE, int32_t>::value ||
                  std::is_same<DTYPE, int64_t>::value ||
                  std::is_same<DTYPE, uint8_t>::value ||
                  std::is_same<DTYPE, uint16_t>::value ||
                  std::is_same<DTYPE, uint32_t>::value ||
                  std::is_same<DTYPE, uint64_t>::value ||
                  std::is_same<DTYPE, bool>::value ||
                  std::is_same<DTYPE, hifloat8_t>::value ||
                  std::is_same<DTYPE, hifloat4x2_t>::value ||
                  std::is_same<DTYPE, float8_e4m3_t>::value ||
                  std::is_same<DTYPE, float8_e5m2_t>::value,
                //   std::is_same<DTYPE, float4_e2m1_t>::value ||
                //   std::is_same<DTYPE, float4_e1m2_t>::value,
                  "Currently dtype of data supports only \
                    float|half|bfloat16_t|int8_t|int16_t|int32_t|int64_t| \
                    bool|hifloat8_t|hifloat4x2_t|float8_e4m3_t|float8_e5m2_t|float4_e2m1_t|float4_e1m2_t.");
    static_assert(std::is_same<ITYPE, int32_t>::value ||
                  std::is_same<ITYPE, int64_t>::value,
                  "Currently dtype of index supports only int32|int64");

    unsigned int block_dim_x = 1;
    unsigned int block_dim_y = 1;
    unsigned int block_dim_z = 1;
    
    if constexpr (DIM == 1) {
        cce::async_invoke<SIMTIndirectStore1D<DTYPE, ITYPE>>(cce::dim3{MAX_THREAD_NUM},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            indices->sizes[0],
            indices->strides[0],
            src->strides[0],
            mask->strides[0]);
    }else if constexpr (DIM == 2) {
        const uint32_t size0 = indices->sizes[0];
        const uint32_t size1 = indices->sizes[1];
        const uint32_t stride0 = indices->strides[0];
        const uint32_t stride1 = indices->strides[1];

        block_dim_x = min(size1, MAX_THREAD_NUM);
        block_dim_y = (size1 >= MAX_THREAD_NUM) ? 1 : min(size0, MAX_THREAD_NUM / size1);

        cce::async_invoke<SIMTIndirectStore2D<DTYPE, ITYPE>>(cce::dim3{block_dim_x, block_dim_y},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            size0, size1,
            stride0, stride1,
            src->strides[0], src->strides[1],
            mask->strides[0], mask->strides[1]
        );
    }else if constexpr (DIM == 3) {
        cce::async_invoke<SIMTIndirectStore3D<DTYPE, ITYPE>>(cce::dim3{THREAD_NUM_512},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            indices->sizes[0], indices->sizes[1], indices->sizes[2],
            indices->strides[0], indices->strides[1], indices->strides[2],
            src->strides[0], src->strides[1], src->strides[2],
            mask->strides[0], mask->strides[1],  mask->strides[2]
            );
    }else if constexpr (DIM == 4) {
        cce::async_invoke<SIMTIndirectStore4D<DTYPE, ITYPE>>(cce::dim3{THREAD_NUM_512},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            indices->sizes[0], indices->sizes[1], indices->sizes[2], indices->sizes[3],
            indices->strides[0], indices->strides[1], indices->strides[2], indices->strides[3],
            src->strides[0], src->strides[1], src->strides[2], src->strides[3],
            mask->strides[0], mask->strides[1], mask->strides[2], mask->strides[3]
        );
    }else if constexpr (DIM == 5) {
        cce::async_invoke<SIMTIndirectStore5D<DTYPE, ITYPE>>(cce::dim3{THREAD_NUM_512},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            indices->sizes[0], indices->sizes[1], indices->sizes[2], indices->sizes[3], indices->sizes[4],
            indices->strides[0], indices->strides[1], indices->strides[2], indices->strides[3], indices->strides[4],
            src->strides[0], src->strides[1], src->strides[2], src->strides[3], src->strides[4],
            mask->strides[0], mask->strides[1], mask->strides[2], mask->strides[3], mask->strides[4]
        );
    }
}


template<int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void indirect_store_no_mask(
    memref_t<__gm__ DTYPE, 1> *dst,
    memref_t<__ubuf__ ITYPE, DIM> *indices,
    memref_t<__ubuf__ DTYPE, DIM> *src
) {
    static_assert(DIM <= 5, "DIM must be <= 5");
    static_assert(std::is_same<DTYPE, float>::value ||
                  std::is_same<DTYPE, half>::value ||
                  std::is_same<DTYPE, bfloat16_t>::value ||
                  std::is_same<DTYPE, int8_t>::value ||
                  std::is_same<DTYPE, int16_t>::value ||
                  std::is_same<DTYPE, int32_t>::value ||
                  std::is_same<DTYPE, int64_t>::value ||
                  std::is_same<DTYPE, uint8_t>::value ||
                  std::is_same<DTYPE, uint16_t>::value ||
                  std::is_same<DTYPE, uint32_t>::value ||
                  std::is_same<DTYPE, uint64_t>::value ||
                  std::is_same<DTYPE, bool>::value ||
                  std::is_same<DTYPE, hifloat8_t>::value ||
                  std::is_same<DTYPE, hifloat4x2_t>::value ||
                  std::is_same<DTYPE, float8_e4m3_t>::value ||
                  std::is_same<DTYPE, float8_e5m2_t>::value,
                //   std::is_same<DTYPE, float4_e2m1_t>::value ||
                //   std::is_same<DTYPE, float4_e1m2_t>::value,
                  "Currently dtype of data supports only \
                    float|half|bfloat16_t|int8_t|int16_t|int32_t|int64_t| \
                    bool|hifloat8_t|hifloat4x2_t|float8_e4m3_t|float8_e5m2_t|float4_e2m1_t|float4_e1m2_t.");
    static_assert(std::is_same<ITYPE, int32_t>::value ||
                  std::is_same<ITYPE, int64_t>::value,
                  "Currently dtype of index supports only int32|int64");

    unsigned int block_dim_x = 1;
    unsigned int block_dim_y = 1;
    unsigned int block_dim_z = 1;
    
    if constexpr (DIM == 1) {
        cce::async_invoke<SIMTIndirectStore1DWithOutMask<DTYPE, ITYPE>>(cce::dim3{MAX_THREAD_NUM},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            indices->sizes[0],
            indices->strides[0],
            src->strides[0]);
    }else if constexpr (DIM == 2) {
        const uint32_t size0 = indices->sizes[0];
        const uint32_t size1 = indices->sizes[1];
        const uint32_t stride0 = indices->strides[0];
        const uint32_t stride1 = indices->strides[1];

        block_dim_x = min(size1, MAX_THREAD_NUM);
        block_dim_y = (size1 >= MAX_THREAD_NUM) ? 1 : min(size0, MAX_THREAD_NUM / size1);

        cce::async_invoke<SIMTIndirectStore2DWithOutMask<DTYPE, ITYPE>>(cce::dim3{block_dim_x, block_dim_y},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            size0, size1,
            stride0, stride1,
            src->strides[0], src->strides[1]
        );
    }else if constexpr (DIM == 3) {
        cce::async_invoke<SIMTIndirectStore3DWithOutMask<DTYPE, ITYPE>>(cce::dim3{MAX_THREAD_NUM},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            indices->sizes[0], indices->sizes[1], indices->sizes[2],
            indices->strides[0], indices->strides[1], indices->strides[2],
            src->strides[0], src->strides[1], src->strides[2]
            );
    }else if constexpr (DIM == 4) {
        cce::async_invoke<SIMTIndirectStore4DWithOutMask<DTYPE, ITYPE>>(cce::dim3{THREAD_NUM_512},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            indices->sizes[0], indices->sizes[1], indices->sizes[2], indices->sizes[3],
            indices->strides[0], indices->strides[1], indices->strides[2], indices->strides[3],
            src->strides[0], src->strides[1], src->strides[2], src->strides[3]
        );
    }else if constexpr (DIM == 5) {
        cce::async_invoke<SIMTIndirectStore5DWithOutMask<DTYPE, ITYPE>>(cce::dim3{THREAD_NUM_512},
            reinterpret_cast<__gm__ DTYPE*> (dst->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (src->aligned + src->offset),
            indices->sizes[0], indices->sizes[1], indices->sizes[2], indices->sizes[3], indices->sizes[4],
            indices->strides[0], indices->strides[1], indices->strides[2], indices->strides[3], indices->strides[4],
            src->strides[0], src->strides[1], src->strides[2], src->strides[3], src->strides[4]
        );
    }
}

extern "C" {
REGISTER_INDIRECT_STORE(1, float, int32_t);
REGISTER_INDIRECT_STORE(1, half, int32_t);
REGISTER_INDIRECT_STORE(1, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE(1, int8_t, int32_t);
REGISTER_INDIRECT_STORE(1, int16_t, int32_t);
REGISTER_INDIRECT_STORE(1, int32_t, int32_t);
REGISTER_INDIRECT_STORE(1, int64_t, int32_t);
REGISTER_INDIRECT_STORE(1, uint8_t, int32_t);
REGISTER_INDIRECT_STORE(1, uint16_t, int32_t);
REGISTER_INDIRECT_STORE(1, uint32_t, int32_t);
REGISTER_INDIRECT_STORE(1, uint64_t, int32_t);
REGISTER_INDIRECT_STORE(1, bool, int32_t);
REGISTER_INDIRECT_STORE(1, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE(1, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE(1, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE(1, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE(1, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE(1, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE(1, float, int64_t);
REGISTER_INDIRECT_STORE(1, half, int64_t);
REGISTER_INDIRECT_STORE(1, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE(1, int8_t, int64_t);
REGISTER_INDIRECT_STORE(1, int16_t, int64_t);
REGISTER_INDIRECT_STORE(1, int32_t, int64_t);
REGISTER_INDIRECT_STORE(1, int64_t, int64_t);
REGISTER_INDIRECT_STORE(1, uint8_t, int64_t);
REGISTER_INDIRECT_STORE(1, uint16_t, int64_t);
REGISTER_INDIRECT_STORE(1, uint32_t, int64_t);
REGISTER_INDIRECT_STORE(1, uint64_t, int64_t);
REGISTER_INDIRECT_STORE(1, bool, int64_t);
REGISTER_INDIRECT_STORE(1, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE(1, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE(1, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE(1, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE(1, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE(1, float4_e1m2_t, int64_t);
 
REGISTER_INDIRECT_STORE(2, float, int32_t);
REGISTER_INDIRECT_STORE(2, half, int32_t);
REGISTER_INDIRECT_STORE(2, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE(2, int8_t, int32_t);
REGISTER_INDIRECT_STORE(2, int16_t, int32_t);
REGISTER_INDIRECT_STORE(2, int32_t, int32_t);
REGISTER_INDIRECT_STORE(2, int64_t, int32_t);
REGISTER_INDIRECT_STORE(2, uint8_t, int32_t);
REGISTER_INDIRECT_STORE(2, uint16_t, int32_t);
REGISTER_INDIRECT_STORE(2, uint32_t, int32_t);
REGISTER_INDIRECT_STORE(2, uint64_t, int32_t);
REGISTER_INDIRECT_STORE(2, bool, int32_t);
REGISTER_INDIRECT_STORE(2, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE(2, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE(2, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE(2, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE(2, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE(2, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE(2, float, int64_t);
REGISTER_INDIRECT_STORE(2, half, int64_t);
REGISTER_INDIRECT_STORE(2, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE(2, int8_t, int64_t);
REGISTER_INDIRECT_STORE(2, int16_t, int64_t);
REGISTER_INDIRECT_STORE(2, int32_t, int64_t);
REGISTER_INDIRECT_STORE(2, int64_t, int64_t);
REGISTER_INDIRECT_STORE(2, uint8_t, int64_t);
REGISTER_INDIRECT_STORE(2, uint16_t, int64_t);
REGISTER_INDIRECT_STORE(2, uint32_t, int64_t);
REGISTER_INDIRECT_STORE(2, uint64_t, int64_t);
REGISTER_INDIRECT_STORE(2, bool, int64_t);
REGISTER_INDIRECT_STORE(2, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE(2, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE(2, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE(2, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE(2, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE(2, float4_e1m2_t, int64_t);
 
REGISTER_INDIRECT_STORE(3, float, int32_t);
REGISTER_INDIRECT_STORE(3, half, int32_t);
REGISTER_INDIRECT_STORE(3, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE(3, int8_t, int32_t);
REGISTER_INDIRECT_STORE(3, int16_t, int32_t);
REGISTER_INDIRECT_STORE(3, int32_t, int32_t);
REGISTER_INDIRECT_STORE(3, int64_t, int32_t);
REGISTER_INDIRECT_STORE(3, uint8_t, int32_t);
REGISTER_INDIRECT_STORE(3, uint16_t, int32_t);
REGISTER_INDIRECT_STORE(3, uint32_t, int32_t);
REGISTER_INDIRECT_STORE(3, uint64_t, int32_t);
REGISTER_INDIRECT_STORE(3, bool, int32_t);
REGISTER_INDIRECT_STORE(3, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE(3, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE(3, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE(3, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE(3, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE(3, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE(3, float, int64_t);
REGISTER_INDIRECT_STORE(3, half, int64_t);
REGISTER_INDIRECT_STORE(3, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE(3, int8_t, int64_t);
REGISTER_INDIRECT_STORE(3, int16_t, int64_t);
REGISTER_INDIRECT_STORE(3, int32_t, int64_t);
REGISTER_INDIRECT_STORE(3, int64_t, int64_t);
REGISTER_INDIRECT_STORE(3, uint8_t, int64_t);
REGISTER_INDIRECT_STORE(3, uint16_t, int64_t);
REGISTER_INDIRECT_STORE(3, uint32_t, int64_t);
REGISTER_INDIRECT_STORE(3, uint64_t, int64_t);
REGISTER_INDIRECT_STORE(3, bool, int64_t);
REGISTER_INDIRECT_STORE(3, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE(3, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE(3, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE(3, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE(3, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE(3, float4_e1m2_t, int64_t);
 
REGISTER_INDIRECT_STORE(4, float, int32_t);
REGISTER_INDIRECT_STORE(4, half, int32_t);
REGISTER_INDIRECT_STORE(4, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE(4, int8_t, int32_t);
REGISTER_INDIRECT_STORE(4, int16_t, int32_t);
REGISTER_INDIRECT_STORE(4, int32_t, int32_t);
REGISTER_INDIRECT_STORE(4, int64_t, int32_t);
REGISTER_INDIRECT_STORE(4, uint8_t, int32_t);
REGISTER_INDIRECT_STORE(4, uint16_t, int32_t);
REGISTER_INDIRECT_STORE(4, uint32_t, int32_t);
REGISTER_INDIRECT_STORE(4, uint64_t, int32_t);
REGISTER_INDIRECT_STORE(4, bool, int32_t);
REGISTER_INDIRECT_STORE(4, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE(4, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE(4, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE(4, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE(4, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE(4, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE(4, float, int64_t);
REGISTER_INDIRECT_STORE(4, half, int64_t);
REGISTER_INDIRECT_STORE(4, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE(4, int8_t, int64_t);
REGISTER_INDIRECT_STORE(4, int16_t, int64_t);
REGISTER_INDIRECT_STORE(4, int32_t, int64_t);
REGISTER_INDIRECT_STORE(4, int64_t, int64_t);
REGISTER_INDIRECT_STORE(4, uint8_t, int64_t);
REGISTER_INDIRECT_STORE(4, uint16_t, int64_t);
REGISTER_INDIRECT_STORE(4, uint32_t, int64_t);
REGISTER_INDIRECT_STORE(4, uint64_t, int64_t);
REGISTER_INDIRECT_STORE(4, bool, int64_t);
REGISTER_INDIRECT_STORE(4, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE(4, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE(4, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE(4, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE(4, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE(4, float4_e1m2_t, int64_t);
 
REGISTER_INDIRECT_STORE(5, float, int32_t);
REGISTER_INDIRECT_STORE(5, half, int32_t);
REGISTER_INDIRECT_STORE(5, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE(5, int8_t, int32_t);
REGISTER_INDIRECT_STORE(5, int16_t, int32_t);
REGISTER_INDIRECT_STORE(5, int32_t, int32_t);
REGISTER_INDIRECT_STORE(5, int64_t, int32_t);
REGISTER_INDIRECT_STORE(5, uint8_t, int32_t);
REGISTER_INDIRECT_STORE(5, uint16_t, int32_t);
REGISTER_INDIRECT_STORE(5, uint32_t, int32_t);
REGISTER_INDIRECT_STORE(5, uint64_t, int32_t);
REGISTER_INDIRECT_STORE(5, bool, int32_t);
REGISTER_INDIRECT_STORE(5, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE(5, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE(5, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE(5, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE(5, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE(5, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE(5, float, int64_t);
REGISTER_INDIRECT_STORE(5, half, int64_t);
REGISTER_INDIRECT_STORE(5, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE(5, int8_t, int64_t);
REGISTER_INDIRECT_STORE(5, int16_t, int64_t);
REGISTER_INDIRECT_STORE(5, int32_t, int64_t);
REGISTER_INDIRECT_STORE(5, int64_t, int64_t);
REGISTER_INDIRECT_STORE(5, uint8_t, int64_t);
REGISTER_INDIRECT_STORE(5, uint16_t, int64_t);
REGISTER_INDIRECT_STORE(5, uint32_t, int64_t);
REGISTER_INDIRECT_STORE(5, uint64_t, int64_t);
REGISTER_INDIRECT_STORE(5, bool, int64_t);
REGISTER_INDIRECT_STORE(5, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE(5, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE(5, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE(5, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE(5, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE(5, float4_e1m2_t, int64_t);


REGISTER_INDIRECT_STORE_NO_MASK(1, float, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, half, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, int8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, int16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, int32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, int64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, uint8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, uint16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, uint32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, uint64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, bool, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(1, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(1, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE_NO_MASK(1, float, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, half, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, int8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, int16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, int32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, int64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, uint8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, uint16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, uint32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, uint64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, bool, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(1, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(1, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(1, float4_e1m2_t, int64_t);
 
REGISTER_INDIRECT_STORE_NO_MASK(2, float, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, half, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, int8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, int16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, int32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, int64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, uint8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, uint16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, uint32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, uint64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, bool, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(2, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(2, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE_NO_MASK(2, float, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, half, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, int8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, int16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, int32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, int64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, uint8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, uint16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, uint32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, uint64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, bool, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(2, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(2, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(2, float4_e1m2_t, int64_t);
 
REGISTER_INDIRECT_STORE_NO_MASK(3, float, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, half, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, int8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, int16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, int32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, int64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, uint8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, uint16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, uint32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, uint64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, bool, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(3, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(3, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE_NO_MASK(3, float, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, half, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, int8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, int16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, int32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, int64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, uint8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, uint16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, uint32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, uint64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, bool, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(3, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(3, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(3, float4_e1m2_t, int64_t);
 
REGISTER_INDIRECT_STORE_NO_MASK(4, float, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, half, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, int8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, int16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, int32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, int64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, uint8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, uint16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, uint32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, uint64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, bool, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(4, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(4, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE_NO_MASK(4, float, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, half, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, int8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, int16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, int32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, int64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, uint8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, uint16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, uint32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, uint64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, bool, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(4, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(4, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(4, float4_e1m2_t, int64_t);
 
REGISTER_INDIRECT_STORE_NO_MASK(5, float, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, half, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, bfloat16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, int8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, int16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, int32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, int64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, uint8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, uint16_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, uint32_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, uint64_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, bool, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, hifloat8_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, hifloat4x2_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, float8_e4m3_t, int32_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, float8_e5m2_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(5, float4_e2m1_t, int32_t);
// REGISTER_INDIRECT_STORE_NO_MASK(5, float4_e1m2_t, int32_t);
 
REGISTER_INDIRECT_STORE_NO_MASK(5, float, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, half, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, bfloat16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, int8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, int16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, int32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, int64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, uint8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, uint16_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, uint32_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, uint64_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, bool, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, hifloat8_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, hifloat4x2_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, float8_e4m3_t, int64_t);
REGISTER_INDIRECT_STORE_NO_MASK(5, float8_e5m2_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(5, float4_e2m1_t, int64_t);
// REGISTER_INDIRECT_STORE_NO_MASK(5, float4_e1m2_t, int64_t);
}
#endif
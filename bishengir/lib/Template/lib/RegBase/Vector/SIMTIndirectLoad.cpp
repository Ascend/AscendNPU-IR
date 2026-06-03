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
constexpr unsigned int MAX_THREAD_NUM3D = 512;
constexpr unsigned int MAX_THREAD_NUM4_5D = 256;


template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void simtIndirectLoad1D(
    volatile __gm__ DTYPE *src,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *dst,
    __ubuf__ bool *mask,
    __ubuf__ DTYPE *other,
    const uint32_t indice_size0,
    const uint32_t indice_stride0,
    const uint32_t dst_stride0,
    const uint32_t mask_stride0,
    const uint32_t other_stride0
)
{
    for (uint32_t i0 = threadIdx.x; i0 < indice_size0; i0 += blockDim.x) {
        uint32_t maskIdx = i0 * mask_stride0;
        uint32_t dstIdx = i0 * dst_stride0;
        if (mask[maskIdx]) {
            uint32_t indiceIdx = i0 * indice_stride0;
            ITYPE srcIdx = indices[indiceIdx];
            dst[dstIdx] = src[srcIdx];
        }
        else {
            uint32_t otherIdx = i0 * other_stride0;
            dst[dstIdx] = other[otherIdx];
        }
    }
}

template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void simtIndirectLoad1DWithOutMask(
    volatile __gm__ DTYPE *src,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *dst,
    const uint32_t indice_size0,
    const uint32_t indice_stride0,
    const uint32_t dst_size0,
    const uint32_t dst_stride0
)
{
    for (uint32_t i0 = threadIdx.x; i0 < indice_size0; i0 += blockDim.x) {
        uint32_t indiceIdx = i0 * indice_stride0;
        uint32_t dstIdx = i0 * dst_stride0;
        ITYPE srcIdx = indices[indiceIdx];
        dst[dstIdx] = src[srcIdx];
    }
}

template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void simtIndirectLoad2D(
    volatile __gm__ DTYPE *src,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *dst,
    __ubuf__ bool *mask,
    __ubuf__ DTYPE *other,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t dst_stride0,
    const uint32_t dst_stride1,
    const uint32_t mask_stride0,
    const uint32_t mask_stride1,
    const uint32_t other_stride0,
    const uint32_t other_stride1
) {
    for (uint32_t ty = threadIdx.y; ty < indice_size0; ty += blockDim.y) {
        for (uint32_t tx = threadIdx.x; tx < indice_size1; tx += blockDim.x) {
            uint32_t maskIdx = ty * mask_stride0 + tx * mask_stride1;
            uint32_t dstIdx = ty * dst_stride0 + tx * dst_stride1;
            if (mask[maskIdx]) {
                uint32_t indiceIdx = ty * indice_stride0 + tx * indice_stride1;
                ITYPE srcIdx = indices[indiceIdx];
                dst[dstIdx] = src[srcIdx];
            }
            else {
                uint32_t otherIdx = ty * other_stride0 + tx * other_stride1;
                dst[dstIdx] = other[otherIdx];
            }
        }
    }
}


template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM3D)
__aiv__ __attribute__((always_inline)) static void simtIndirectLoad3D(
    volatile __gm__ DTYPE *src,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *dst,
    __ubuf__ bool *mask,
    __ubuf__ DTYPE* other,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_size2,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t indice_stride2,
    const uint32_t dst_stride0,
    const uint32_t dst_stride1,
    const uint32_t dst_stride2,
    const uint32_t mask_stride0,
    const uint32_t mask_stride1,
    const uint32_t mask_stride2,
    const uint32_t other_stride0,
    const uint32_t other_stride1,
    const uint32_t other_stride2
) {
    const uint32_t num_elems = indice_size0 * indice_size1 * indice_size2;
    for (uint32_t tx = threadIdx.x; tx < num_elems; tx += blockDim.x) {
        uint32_t elem_idx = tx;
        uint32_t i2 = elem_idx % indice_size2; elem_idx /= indice_size2;
        uint32_t i1 = elem_idx % indice_size1; elem_idx /= indice_size1;
        uint32_t i0 = elem_idx;

        uint32_t maskIdx = i0*mask_stride0 + i1*mask_stride1 + i2*mask_stride2;
        uint32_t dstIdx = i0*dst_stride0 + i1*dst_stride1 + i2*dst_stride2;
        if (mask[maskIdx]) {
            uint32_t indiceIdx = i0*indice_stride0 + i1*indice_stride1 + i2*indice_stride2;
            ITYPE srcIdx = indices[indiceIdx];
            dst[dstIdx] = src[srcIdx];
        }
        else {
            uint32_t otherIdx = i0*other_stride0 + i1*other_stride1 + i2*other_stride2;
            dst[dstIdx] = other[otherIdx];
        }
    }
}

template<typename DTYPE, typename ITYPE >
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM4_5D)
__aiv__ __attribute__((always_inline)) static void simtIndirectLoad4D(
    volatile __gm__ DTYPE *src,
    __ubuf__ ITYPE  *indices,
    __ubuf__ DTYPE  *dst,
    __ubuf__ bool *mask,
    __ubuf__ DTYPE * other,
    const uint32_t indice_size0,
    const uint32_t indice_size1,
    const uint32_t indice_size2,
    const uint32_t indice_size3,
    const uint32_t indice_stride0,
    const uint32_t indice_stride1,
    const uint32_t indice_stride2,
    const uint32_t indice_stride3,
    const uint32_t dst_stride0,
    const uint32_t dst_stride1,
    const uint32_t dst_stride2,
    const uint32_t dst_stride3,
    const uint32_t mask_stride0,
    const uint32_t mask_stride1,
    const uint32_t mask_stride2,
    const uint32_t mask_stride3,
    const uint32_t other_stride0,
    const uint32_t other_stride1,
    const uint32_t other_stride2,
    const uint32_t other_stride3
) {
    const uint32_t num_elems = indice_size0 * indice_size1 * indice_size2 * indice_size3;
    for (uint32_t tx = threadIdx.x; tx < num_elems; tx += blockDim.x) {
        uint32_t elem_idx = tx;
        uint32_t i3 = elem_idx % indice_size3; elem_idx /= indice_size3;
        uint32_t i2 = elem_idx % indice_size2; elem_idx /= indice_size2;
        uint32_t i1 = elem_idx % indice_size1; elem_idx /= indice_size1;
        uint32_t i0 = elem_idx;

        uint32_t maskIdx = i0*mask_stride0 + i1*mask_stride1 + i2*mask_stride2 + i3*mask_stride3;
        uint32_t dstIdx = i0*dst_stride0 + i1*dst_stride1 + i2*dst_stride2 + i3*dst_stride3;
        if (mask[maskIdx]) {
            uint32_t indiceIdx = i0*indice_stride0 + i1*indice_stride1 + i2*indice_stride2 + i3*indice_stride3;
            ITYPE srcIdx = indices[indiceIdx];
            dst[dstIdx] = src[srcIdx];
        }
        else {
            uint32_t otherIdx = i0*other_stride0 + i1*other_stride1 + i2*other_stride2 + i3*other_stride3;
            dst[dstIdx] = other[otherIdx];
        }
    }
}

template<typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM4_5D)
__aiv__ __attribute__((always_inline)) static void simtIndirectLoad5D(
    volatile __gm__ DTYPE *src,
    __ubuf__ ITYPE *indices,
    __ubuf__ DTYPE *dst,
    __ubuf__ bool *mask,
    __ubuf__ DTYPE* other,
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
    const uint32_t dst_stride0,
    const uint32_t dst_stride1,
    const uint32_t dst_stride2,
    const uint32_t dst_stride3,
    const uint32_t dst_stride4,
    const uint32_t mask_stride0,
    const uint32_t mask_stride1,
    const uint32_t mask_stride2,
    const uint32_t mask_stride3,
    const uint32_t mask_stride4,
    const uint32_t other_stride0,
    const uint32_t other_stride1,
    const uint32_t other_stride2,
    const uint32_t other_stride3,
    const uint32_t other_stride4
)  {
    const uint32_t num_elems = indice_size0 * indice_size1 * indice_size2 * indice_size3 * indice_size4;
    for (uint32_t tx = threadIdx.x; tx < num_elems; tx += blockDim.x) {
        uint32_t elem_idx = tx;
        uint32_t i4 = elem_idx % indice_size4; elem_idx /= indice_size4;
        uint32_t i3 = elem_idx % indice_size3; elem_idx /= indice_size3;
        uint32_t i2 = elem_idx % indice_size2; elem_idx /= indice_size2;
        uint32_t i1 = elem_idx % indice_size1; elem_idx /= indice_size1;
        uint32_t i0 = elem_idx;

        uint32_t maskIdx = i0*mask_stride0 + i1*mask_stride1 + i2*mask_stride2 + i3*mask_stride3 + i4*mask_stride4;
        uint32_t dstIdx = i0*dst_stride0 + i1*dst_stride1 + i2*dst_stride2 + i3*dst_stride3 + i4*dst_stride4;
        if (mask[maskIdx]) {
            uint32_t indiceIdx = i0*indice_stride0 + i1*indice_stride1 + i2*indice_stride2 + i3*indice_stride3 + i4*indice_stride4;
            ITYPE srcIdx = indices[indiceIdx];
            dst[dstIdx] = src[srcIdx];
        }
        else {
            uint32_t otherIdx = i0*other_stride0 + i1*other_stride1 + i2*other_stride2 + i3*other_stride3 + i4*other_stride4;
            dst[dstIdx] = other[otherIdx];
        }
    }
}

template<int DIM, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) void indirect_load(
    memref_t<__gm__ DTYPE, 1> *src,
    memref_t<__ubuf__ ITYPE, DIM> *indices,
    memref_t<__ubuf__ DTYPE, DIM> *dst,
    memref_t<__ubuf__ bool, DIM> *mask,
    memref_t<__ubuf__ DTYPE, DIM> *other
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
        cce::async_invoke<simtIndirectLoad1D<DTYPE, ITYPE>>(cce::dim3{MAX_THREAD_NUM},
            reinterpret_cast<volatile __gm__ DTYPE*> (src->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (dst->aligned + dst->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (other->aligned + other->offset),
            indices->sizes[0],
            indices->strides[0],
            dst->strides[0],
            mask->strides[0],
            other->strides[0]);
    }else if constexpr (DIM == 2) {
        const uint32_t size0 = indices->sizes[0];
        const uint32_t size1 = indices->sizes[1];
        const uint32_t stride0 = indices->strides[0];
        const uint32_t stride1 = indices->strides[1];

        if (size1 >= MAX_THREAD_NUM) {
            block_dim_x = MAX_THREAD_NUM;
        } else if (size1 * size0 >= MAX_THREAD_NUM) {
            block_dim_x = size1;
            block_dim_y = MAX_THREAD_NUM / block_dim_x;
        } else {
            block_dim_x = size1;
            block_dim_y = size0;
        }
        
        cce::async_invoke<simtIndirectLoad2D<DTYPE, ITYPE>>(cce::dim3{block_dim_x, block_dim_y},
            reinterpret_cast<volatile __gm__ DTYPE*> (src->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (dst->aligned + dst->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (other->aligned + other->offset),
            size0, size1,
            stride0, stride1,
            dst->strides[0], dst->strides[1],
            mask->strides[0], mask->strides[1],
            other->strides[0], other->strides[1]
        );
    }else if constexpr (DIM == 3) {
        const uint32_t size0 = indices->sizes[0];
        const uint32_t size1 = indices->sizes[1];
        const uint32_t size2 = indices->sizes[2];
        const uint32_t stride0 = indices->strides[0];
        const uint32_t stride1 = indices->strides[1];
        const uint32_t stride2 = indices->strides[2];

        cce::async_invoke<simtIndirectLoad3D<DTYPE, ITYPE>>(cce::dim3{MAX_THREAD_NUM3D},
            reinterpret_cast<volatile __gm__ DTYPE*> (src->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (dst->aligned + dst->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (other->aligned + other->offset),
            size0, size1, size2,
            stride0, stride1, stride2,
            dst->strides[0], dst->strides[1], dst->strides[2],
            mask->strides[0], mask->strides[1],  mask->strides[2],
            other->strides[0], other->strides[1], other->strides[2]
            );
    }else if constexpr (DIM == 4) {
        const uint32_t size0 = indices->sizes[0];
        const uint32_t size1 = indices->sizes[1];
        const uint32_t size2 = indices->sizes[2];
        const uint32_t size3 = indices->sizes[3];
        const uint32_t stride0 = indices->strides[0];
        const uint32_t stride1 = indices->strides[1];
        const uint32_t stride2 = indices->strides[2];
        const uint32_t stride3 = indices->strides[3];
        
        cce::async_invoke<simtIndirectLoad4D<DTYPE, ITYPE>>(cce::dim3{MAX_THREAD_NUM4_5D},
            reinterpret_cast<volatile __gm__ DTYPE*> (src->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (dst->aligned + dst->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (other->aligned + other->offset),
            size0, size1, size2, size3,
            stride0, stride1, stride2, stride3,
            dst->strides[0], dst->strides[1], dst->strides[2], dst->strides[3],
            mask->strides[0], mask->strides[1], mask->strides[2], mask->strides[3],
            other->strides[0], other->strides[1], other->strides[2], other->strides[3]
        );
    }else if constexpr (DIM == 5) {
        const uint32_t stride0 = indices->strides[0];
        const uint32_t stride1 = indices->strides[1];
        const uint32_t stride2 = indices->strides[2];
        const uint32_t stride3 = indices->strides[3];
        const uint32_t stride4 = indices->strides[4];
        const uint32_t size0 = indices->sizes[0];
        const uint32_t size1 = indices->sizes[1];
        const uint32_t size2 = indices->sizes[2];
        const uint32_t size3 = indices->sizes[3];
        const uint32_t size4 = indices->sizes[4];

        cce::async_invoke<simtIndirectLoad5D<DTYPE, ITYPE>>(cce::dim3{MAX_THREAD_NUM4_5D},
            reinterpret_cast<volatile __gm__ DTYPE*> (src->aligned),
            reinterpret_cast<__ubuf__ ITYPE*> (indices->aligned + indices->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (dst->aligned + dst->offset),
            reinterpret_cast<__ubuf__ bool*> (mask->aligned + mask->offset),
            reinterpret_cast<__ubuf__ DTYPE*> (other->aligned + other->offset),
            size0, size1, size2, size3, size4,
            stride0, stride1, stride2, stride3, stride4,
            dst->strides[0], dst->strides[1], dst->strides[2], dst->strides[3], dst->strides[4],
            mask->strides[0], mask->strides[1], mask->strides[2], mask->strides[3], mask->strides[4],
            other->strides[0], other->strides[1], other->strides[2], other->strides[3], other->strides[4]
        );
    }
}

extern "C" {
REGISTE_INDIRECT_LOAD(1, float, int32_t);
REGISTE_INDIRECT_LOAD(1, half, int32_t);
REGISTE_INDIRECT_LOAD(1, bfloat16_t, int32_t);
REGISTE_INDIRECT_LOAD(1, int8_t, int32_t);
REGISTE_INDIRECT_LOAD(1, int16_t, int32_t);
REGISTE_INDIRECT_LOAD(1, int32_t, int32_t);
REGISTE_INDIRECT_LOAD(1, int64_t, int32_t);
REGISTE_INDIRECT_LOAD(1, uint8_t, int32_t);
REGISTE_INDIRECT_LOAD(1, uint16_t, int32_t);
REGISTE_INDIRECT_LOAD(1, uint32_t, int32_t);
REGISTE_INDIRECT_LOAD(1, uint64_t, int32_t);
REGISTE_INDIRECT_LOAD(1, bool, int32_t);
REGISTE_INDIRECT_LOAD(1, hifloat8_t, int32_t);
REGISTE_INDIRECT_LOAD(1, hifloat4x2_t, int32_t);
REGISTE_INDIRECT_LOAD(1, float8_e4m3_t, int32_t);
REGISTE_INDIRECT_LOAD(1, float8_e5m2_t, int32_t);
// REGISTE_INDIRECT_LOAD(1, float4_e2m1_t, int32_t);
// REGISTE_INDIRECT_LOAD(1, float4_e1m2_t, int32_t);
 
REGISTE_INDIRECT_LOAD(1, float, int64_t);
REGISTE_INDIRECT_LOAD(1, half, int64_t);
REGISTE_INDIRECT_LOAD(1, bfloat16_t, int64_t);
REGISTE_INDIRECT_LOAD(1, int8_t, int64_t);
REGISTE_INDIRECT_LOAD(1, int16_t, int64_t);
REGISTE_INDIRECT_LOAD(1, int32_t, int64_t);
REGISTE_INDIRECT_LOAD(1, int64_t, int64_t);
REGISTE_INDIRECT_LOAD(1, uint8_t, int64_t);
REGISTE_INDIRECT_LOAD(1, uint16_t, int64_t);
REGISTE_INDIRECT_LOAD(1, uint32_t, int64_t);
REGISTE_INDIRECT_LOAD(1, uint64_t, int64_t);
REGISTE_INDIRECT_LOAD(1, bool, int64_t);
REGISTE_INDIRECT_LOAD(1, hifloat8_t, int64_t);
REGISTE_INDIRECT_LOAD(1, hifloat4x2_t, int64_t);
REGISTE_INDIRECT_LOAD(1, float8_e4m3_t, int64_t);
REGISTE_INDIRECT_LOAD(1, float8_e5m2_t, int64_t);
// REGISTE_INDIRECT_LOAD(1, float4_e2m1_t, int64_t);
// REGISTE_INDIRECT_LOAD(1, float4_e1m2_t, int64_t);
 
REGISTE_INDIRECT_LOAD(2, float, int32_t);
REGISTE_INDIRECT_LOAD(2, half, int32_t);
REGISTE_INDIRECT_LOAD(2, bfloat16_t, int32_t);
REGISTE_INDIRECT_LOAD(2, int8_t, int32_t);
REGISTE_INDIRECT_LOAD(2, int16_t, int32_t);
REGISTE_INDIRECT_LOAD(2, int32_t, int32_t);
REGISTE_INDIRECT_LOAD(2, int64_t, int32_t);
REGISTE_INDIRECT_LOAD(2, uint8_t, int32_t);
REGISTE_INDIRECT_LOAD(2, uint16_t, int32_t);
REGISTE_INDIRECT_LOAD(2, uint32_t, int32_t);
REGISTE_INDIRECT_LOAD(2, uint64_t, int32_t);
REGISTE_INDIRECT_LOAD(2, bool, int32_t);
REGISTE_INDIRECT_LOAD(2, hifloat8_t, int32_t);
REGISTE_INDIRECT_LOAD(2, hifloat4x2_t, int32_t);
REGISTE_INDIRECT_LOAD(2, float8_e4m3_t, int32_t);
REGISTE_INDIRECT_LOAD(2, float8_e5m2_t, int32_t);
// REGISTE_INDIRECT_LOAD(2, float4_e2m1_t, int32_t);
// REGISTE_INDIRECT_LOAD(2, float4_e1m2_t, int32_t);
 
REGISTE_INDIRECT_LOAD(2, float, int64_t);
REGISTE_INDIRECT_LOAD(2, half, int64_t);
REGISTE_INDIRECT_LOAD(2, bfloat16_t, int64_t);
REGISTE_INDIRECT_LOAD(2, int8_t, int64_t);
REGISTE_INDIRECT_LOAD(2, int16_t, int64_t);
REGISTE_INDIRECT_LOAD(2, int32_t, int64_t);
REGISTE_INDIRECT_LOAD(2, int64_t, int64_t);
REGISTE_INDIRECT_LOAD(2, uint8_t, int64_t);
REGISTE_INDIRECT_LOAD(2, uint16_t, int64_t);
REGISTE_INDIRECT_LOAD(2, uint32_t, int64_t);
REGISTE_INDIRECT_LOAD(2, uint64_t, int64_t);
REGISTE_INDIRECT_LOAD(2, bool, int64_t);
REGISTE_INDIRECT_LOAD(2, hifloat8_t, int64_t);
REGISTE_INDIRECT_LOAD(2, hifloat4x2_t, int64_t);
REGISTE_INDIRECT_LOAD(2, float8_e4m3_t, int64_t);
REGISTE_INDIRECT_LOAD(2, float8_e5m2_t, int64_t);
// REGISTE_INDIRECT_LOAD(2, float4_e2m1_t, int64_t);
// REGISTE_INDIRECT_LOAD(2, float4_e1m2_t, int64_t);
 
REGISTE_INDIRECT_LOAD(3, float, int32_t);
REGISTE_INDIRECT_LOAD(3, half, int32_t);
REGISTE_INDIRECT_LOAD(3, bfloat16_t, int32_t);
REGISTE_INDIRECT_LOAD(3, int8_t, int32_t);
REGISTE_INDIRECT_LOAD(3, int16_t, int32_t);
REGISTE_INDIRECT_LOAD(3, int32_t, int32_t);
REGISTE_INDIRECT_LOAD(3, int64_t, int32_t);
REGISTE_INDIRECT_LOAD(3, uint8_t, int32_t);
REGISTE_INDIRECT_LOAD(3, uint16_t, int32_t);
REGISTE_INDIRECT_LOAD(3, uint32_t, int32_t);
REGISTE_INDIRECT_LOAD(3, uint64_t, int32_t);
REGISTE_INDIRECT_LOAD(3, bool, int32_t);
REGISTE_INDIRECT_LOAD(3, hifloat8_t, int32_t);
REGISTE_INDIRECT_LOAD(3, hifloat4x2_t, int32_t);
REGISTE_INDIRECT_LOAD(3, float8_e4m3_t, int32_t);
REGISTE_INDIRECT_LOAD(3, float8_e5m2_t, int32_t);
// REGISTE_INDIRECT_LOAD(3, float4_e2m1_t, int32_t);
// REGISTE_INDIRECT_LOAD(3, float4_e1m2_t, int32_t);
 
REGISTE_INDIRECT_LOAD(3, float, int64_t);
REGISTE_INDIRECT_LOAD(3, half, int64_t);
REGISTE_INDIRECT_LOAD(3, bfloat16_t, int64_t);
REGISTE_INDIRECT_LOAD(3, int8_t, int64_t);
REGISTE_INDIRECT_LOAD(3, int16_t, int64_t);
REGISTE_INDIRECT_LOAD(3, int32_t, int64_t);
REGISTE_INDIRECT_LOAD(3, int64_t, int64_t);
REGISTE_INDIRECT_LOAD(3, uint8_t, int64_t);
REGISTE_INDIRECT_LOAD(3, uint16_t, int64_t);
REGISTE_INDIRECT_LOAD(3, uint32_t, int64_t);
REGISTE_INDIRECT_LOAD(3, uint64_t, int64_t);
REGISTE_INDIRECT_LOAD(3, bool, int64_t);
REGISTE_INDIRECT_LOAD(3, hifloat8_t, int64_t);
REGISTE_INDIRECT_LOAD(3, hifloat4x2_t, int64_t);
REGISTE_INDIRECT_LOAD(3, float8_e4m3_t, int64_t);
REGISTE_INDIRECT_LOAD(3, float8_e5m2_t, int64_t);
// REGISTE_INDIRECT_LOAD(3, float4_e2m1_t, int64_t);
// REGISTE_INDIRECT_LOAD(3, float4_e1m2_t, int64_t);
 
REGISTE_INDIRECT_LOAD(4, float, int32_t);
REGISTE_INDIRECT_LOAD(4, half, int32_t);
REGISTE_INDIRECT_LOAD(4, bfloat16_t, int32_t);
REGISTE_INDIRECT_LOAD(4, int8_t, int32_t);
REGISTE_INDIRECT_LOAD(4, int16_t, int32_t);
REGISTE_INDIRECT_LOAD(4, int32_t, int32_t);
REGISTE_INDIRECT_LOAD(4, int64_t, int32_t);
REGISTE_INDIRECT_LOAD(4, uint8_t, int32_t);
REGISTE_INDIRECT_LOAD(4, uint16_t, int32_t);
REGISTE_INDIRECT_LOAD(4, uint32_t, int32_t);
REGISTE_INDIRECT_LOAD(4, uint64_t, int32_t);
REGISTE_INDIRECT_LOAD(4, bool, int32_t);
REGISTE_INDIRECT_LOAD(4, hifloat8_t, int32_t);
REGISTE_INDIRECT_LOAD(4, hifloat4x2_t, int32_t);
REGISTE_INDIRECT_LOAD(4, float8_e4m3_t, int32_t);
REGISTE_INDIRECT_LOAD(4, float8_e5m2_t, int32_t);
// REGISTE_INDIRECT_LOAD(4, float4_e2m1_t, int32_t);
// REGISTE_INDIRECT_LOAD(4, float4_e1m2_t, int32_t);
 
REGISTE_INDIRECT_LOAD(4, float, int64_t);
REGISTE_INDIRECT_LOAD(4, half, int64_t);
REGISTE_INDIRECT_LOAD(4, bfloat16_t, int64_t);
REGISTE_INDIRECT_LOAD(4, int8_t, int64_t);
REGISTE_INDIRECT_LOAD(4, int16_t, int64_t);
REGISTE_INDIRECT_LOAD(4, int32_t, int64_t);
REGISTE_INDIRECT_LOAD(4, int64_t, int64_t);
REGISTE_INDIRECT_LOAD(4, uint8_t, int64_t);
REGISTE_INDIRECT_LOAD(4, uint16_t, int64_t);
REGISTE_INDIRECT_LOAD(4, uint32_t, int64_t);
REGISTE_INDIRECT_LOAD(4, uint64_t, int64_t);
REGISTE_INDIRECT_LOAD(4, bool, int64_t);
REGISTE_INDIRECT_LOAD(4, hifloat8_t, int64_t);
REGISTE_INDIRECT_LOAD(4, hifloat4x2_t, int64_t);
REGISTE_INDIRECT_LOAD(4, float8_e4m3_t, int64_t);
REGISTE_INDIRECT_LOAD(4, float8_e5m2_t, int64_t);
// REGISTE_INDIRECT_LOAD(4, float4_e2m1_t, int64_t);
// REGISTE_INDIRECT_LOAD(4, float4_e1m2_t, int64_t);
 
REGISTE_INDIRECT_LOAD(5, float, int32_t);
REGISTE_INDIRECT_LOAD(5, half, int32_t);
REGISTE_INDIRECT_LOAD(5, bfloat16_t, int32_t);
REGISTE_INDIRECT_LOAD(5, int8_t, int32_t);
REGISTE_INDIRECT_LOAD(5, int16_t, int32_t);
REGISTE_INDIRECT_LOAD(5, int32_t, int32_t);
REGISTE_INDIRECT_LOAD(5, int64_t, int32_t);
REGISTE_INDIRECT_LOAD(5, uint8_t, int32_t);
REGISTE_INDIRECT_LOAD(5, uint16_t, int32_t);
REGISTE_INDIRECT_LOAD(5, uint32_t, int32_t);
REGISTE_INDIRECT_LOAD(5, uint64_t, int32_t);
REGISTE_INDIRECT_LOAD(5, bool, int32_t);
REGISTE_INDIRECT_LOAD(5, hifloat8_t, int32_t);
REGISTE_INDIRECT_LOAD(5, hifloat4x2_t, int32_t);
REGISTE_INDIRECT_LOAD(5, float8_e4m3_t, int32_t);
REGISTE_INDIRECT_LOAD(5, float8_e5m2_t, int32_t);
// REGISTE_INDIRECT_LOAD(5, float4_e2m1_t, int32_t);
// REGISTE_INDIRECT_LOAD(5, float4_e1m2_t, int32_t);
 
REGISTE_INDIRECT_LOAD(5, float, int64_t);
REGISTE_INDIRECT_LOAD(5, half, int64_t);
REGISTE_INDIRECT_LOAD(5, bfloat16_t, int64_t);
REGISTE_INDIRECT_LOAD(5, int8_t, int64_t);
REGISTE_INDIRECT_LOAD(5, int16_t, int64_t);
REGISTE_INDIRECT_LOAD(5, int32_t, int64_t);
REGISTE_INDIRECT_LOAD(5, int64_t, int64_t);
REGISTE_INDIRECT_LOAD(5, uint8_t, int64_t);
REGISTE_INDIRECT_LOAD(5, uint16_t, int64_t);
REGISTE_INDIRECT_LOAD(5, uint32_t, int64_t);
REGISTE_INDIRECT_LOAD(5, uint64_t, int64_t);
REGISTE_INDIRECT_LOAD(5, bool, int64_t);
REGISTE_INDIRECT_LOAD(5, hifloat8_t, int64_t);
REGISTE_INDIRECT_LOAD(5, hifloat4x2_t, int64_t);
REGISTE_INDIRECT_LOAD(5, float8_e4m3_t, int64_t);
REGISTE_INDIRECT_LOAD(5, float8_e5m2_t, int64_t);
// REGISTE_INDIRECT_LOAD(5, float4_e2m1_t, int64_t);
// REGISTE_INDIRECT_LOAD(5, float4_e1m2_t, int64_t);
}
#endif

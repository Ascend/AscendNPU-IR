/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_macros.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_KERNEL_MACROS_H
#define ASCENDC_BISHENG_KERNEL_MACROS_H

#include <cstdint>
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#define ASSERT(x) assert(x)
#define DEBUG_CODE(T) T

#else

#ifndef ASSERT
#define ASSERT(x)
#endif
#define DEBUG_CODE(T)

// For ascc preprocess: __global__ should not be replaced
#ifdef __ASCC_PRE__
#ifdef __global__
#undef __global__
#endif
#else

#ifndef __aicore__
#define __aicore__ [aicore]
#endif // __aicore__

#ifndef __host_aicore__
#define __host_aicore__ [host, aicore]
#endif // __host_aicore__
#endif

#if (__CCE__)
#define _ASCENDC_HAS_BISHENG_COMPILER 1
#endif

#if (_ASCENDC_HAS_BISHENG_COMPILER)
#define ASCENDC_HOST __host__
#define ASCENDC_AICORE __aicore__
#define ASCENDC_HOST_AICORE __host_aicore__
#else
#define ASCENDC_HOST
#define ASCENDC_AICORE
#define ASCENDC_HOST_AICORE
#endif

#if __NPU_ARCH__ == 2002
#ifndef __BLOCK_LOCAL__
#define __BLOCK_LOCAL__ [[block_local]]
#endif // __BLOCK_LOCAL__
#else
#ifndef __WORKGROUP_LOCAL__
#define __WORKGROUP_LOCAL__ [[workgroup_local]]
#endif // __WORKGROUP_LOCAL__
#endif

#ifndef __BLOCK_LOCAL__
#define __BLOCK_LOCAL__ [[block_local]]
#endif // __BLOCK_LOCAL__
#endif // ASCENDC_CPU_DEBUG

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif

#ifndef QBUF_MAX_LEN
#define QBUF_MAX_LEN 64
#endif

#ifndef QBUFPOOL_MAX_LEN
#define QBUFPOOL_MAX_LEN 16
#endif

#ifndef MAX_MSG_COUNT
#define MAX_MSG_COUNT 64
#endif

#ifndef QBUF_L0A_RESERVED_LEN
#define QBUF_L0A_RESERVED_LEN 2
#endif

#ifndef QBUF_L0B_RESERVED_LEN
#define QBUF_L0B_RESERVED_LEN 2
#endif
#ifndef QBUF_TOTAL_RESERVED_LEN
#define QBUF_TOTAL_RESERVED_LEN 4
#endif

#ifndef TPIPE_MAX_TYPE
#define TPIPE_MAX_TYPE 4
#endif

#if defined(__DAV_C220_CUBE__)  || defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
#define SPLIT_CORE_CUBE
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_310R6_VEC__)
#define SPLIT_CORE_VEC
#endif
#if (defined(__NPU_ARCH__) &&                                               \
     ((__NPU_ARCH__ == 1001) || (__NPU_ARCH__ == 2002) ||                   \
      (__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3002) ||                   \
      (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)))
#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

namespace AscendCBisheng {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 2201) ||           \
    (__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3510) ||           \
    (__NPU_ARCH__ == 5102)) // Available for V200 and V210
constexpr int32_t QUE_MAX_EVENT = 8;
#else
constexpr int32_t QUE_MAX_EVENT = 4;
#endif
constexpr int32_t HF32_MODE_BIT = 46;
constexpr int32_t HF32_TRANS_MODE_BIT = 47;
constexpr int32_t MM_LAYOUT_MODE_BIT = 51;
constexpr int32_t LEAKY_RELU_MODE_BIT = 50;
constexpr int32_t CAST_MODE_BIT = 59;

constexpr int32_t MIX = 0;
constexpr int32_t AIC = 1;
constexpr int32_t AIV = 2;
} // namespace AscendC

#if defined(ASCENDC_CPU_DEBUG)
extern int32_t g_matmulCount;
extern int32_t g_coreType;
#define ASCEND_IS_AIV (g_coreType == AscendCBisheng::AIV)
#define ASCEND_IS_AIC (g_coreType == AscendCBisheng::AIC)
#define ASCEND_IS_NOT_AIV (g_coreType != AscendCBisheng::AIV)
#define ASCEND_IS_NOT_AIC (g_coreType != AscendCBisheng::AIC)
#else
#if defined(SPLIT_CORE_CUBE)
constexpr int32_t g_coreType = AscendCBisheng::AIC;
#elif defined(SPLIT_CORE_VEC)
constexpr int32_t g_coreType = AscendCBisheng::AIV;
#else
constexpr int32_t g_coreType = AscendCBisheng::MIX;
#endif
#define ASCEND_IS_AIV constexpr(g_coreType == AscendCBisheng::AIV)
#define ASCEND_IS_AIC constexpr(g_coreType == AscendCBisheng::AIC)
#define ASCEND_IS_NOT_AIV constexpr(g_coreType != AscendCBisheng::AIV)
#define ASCEND_IS_NOT_AIC constexpr(g_coreType != AscendCBisheng::AIC)
#endif

#endif // ASCENDC_BISHENG_KERNEL_MACROS_H

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
 * \file kernel_utils_macros.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_UTILS_KERNEL_MACROS_H
#define ASCENDC_BISHENG_UTILS_KERNEL_MACROS_H
#define USE_ISA_INS 1
#define GM_ADDR __gm__ uint8_t*
#define __kfc_workspace__ __attribute__((annotate("kfc_workspace")))

#define UB_ADDR __ubuf__ uint8_t*
#define SSBUF_ADDR __ssbuf__ uint32_t*


#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#include "catlass/detail/macros.hpp"
#include "../kernel_log.h"
#include "../event_utils.hpp"

// this marco is used to define new array with dim
#define ASCENDC_SHAPE(dimValue, ...) \
    dimValue, (const uint32_t[])     \
    {                                \
        __VA_ARGS__                  \
    }

namespace AscendCBisheng {
template <typename T>
__aicore__ inline constexpr static auto IsLite(int) -> typename T::LiteType;
template <typename T>
__aicore__ inline constexpr static auto IsLite(void*) -> T;

template <typename T>
using PrimT = decltype(IsLite<T>(0));

enum class CacheMode {
    CACHE_MODE_DISABLE = 0,
    CACHE_MODE_NORMAL = 1,
    CACHE_MODE_LAST = 2,
    CACHE_MODE_PERSISTENT = 4
};

enum class CacheRwMode {
    READ = 1,
    WRITE = 2,
    RW = 3
};

constexpr uint64_t L2_CACHE_OFFSET = 60;
constexpr uint64_t L2_CACHE_OFFSET_MASK = (1ul << L2_CACHE_OFFSET) - 1;
template <class T, CacheRwMode rwMode = CacheRwMode::RW>
__aicore__ __inline__ __gm__ T *L2CacheAlter(__gm__ T *addr, CacheMode mode)
{
    uint64_t value = 0;
    if (mode == CacheMode::CACHE_MODE_DISABLE) {
        value = uint64_t(0b100) << L2_CACHE_OFFSET;
    } else if (mode == CacheMode::CACHE_MODE_NORMAL) {
        value = uint64_t(0b000) << L2_CACHE_OFFSET;
    }
    return (__gm__ T *)((reinterpret_cast<uint64_t>(addr) & L2_CACHE_OFFSET_MASK) | value);
}

__aicore__ __inline__ CacheMode ToCacheModeEnum(uint8_t mode)
{
    if (mode == 0b100) {
        return CacheMode::CACHE_MODE_DISABLE;
    }
    return CacheMode::CACHE_MODE_NORMAL;
}

template <typename T>
__aicore__ inline __gm__ T* ExtractL2CacheGmAddr(__gm__ T* addr)
{
    return (__gm__ T *)((uint64_t)addr & ((1ul << L2_CACHE_OFFSET) - 1));
}

template <typename T>
__aicore__ inline uint8_t ExtractCacheMode(__gm__ T* addr) {
    return static_cast<uint8_t>(((uint64_t)addr) >> L2_CACHE_OFFSET);
}

template <typename T> class GlobalTensor;
template <typename T>
__aicore__ inline uint8_t ExtractCacheMode(const GlobalTensor<T>& cacheMode)
{
    return ExtractCacheMode(cacheMode.address_);
}
}

// In order to pass __COUNTER__ to variable name, need 3 times of MACRO to pass argument
#define TILING_STRUCT_SECTION_INIT_BASE(counter, val)                                                                \
    static const uint64_t __ascendc_tiling_struct_##counter __attribute__((used, section(".ascendc_tiling."#val))) = \
    sizeof(val)
#define TILING_STRUCT_SECTION_INIT(counter, val)       TILING_STRUCT_SECTION_INIT_BASE(counter, val)

#ifdef __CHECK_FEATURE_AT_PRECOMPILE
#define ENABLE_FEATURE_FOR_COMPILE(f, val) auto __enable_feature_for_compile_##f = val
#define ENABLE_FEATURE_FOR_TILING(expression, val) auto __enable_custom_tiling val = expression
#define REGISTER_NONE_TILING auto __enable_no_register_custom_tiling ascendc_trigger_tiling_struct = default
#else
#define ENABLE_FEATURE_FOR_COMPILE(f, val)
#define ENABLE_FEATURE_FOR_TILING(expression, val) TILING_STRUCT_SECTION_INIT(__COUNTER__, val)
#define REGISTER_NONE_TILING
#endif

#define ENABLE_DETERMINISTIC() ENABLE_FEATURE_FOR_COMPILE(deterministic, 1)

#define KERNEL_TASK_TYPE(key, value)  ENABLE_FEATURE_FOR_COMPILE(key, value)
#define KERNEL_TASK_TYPE_DEFAULT(value)  ENABLE_FEATURE_FOR_COMPILE(default, value)

#define REGISTER_TILING_DEFAULT(tiling_struct)  ENABLE_FEATURE_FOR_TILING(default, tiling_struct)
#define REGISTER_TILING_FOR_TILINGKEY(expression, tiling_struct)  ENABLE_FEATURE_FOR_TILING(expression, tiling_struct)

#define ENABLE_PRINTF() ENABLE_FEATURE_FOR_COMPILE(printf, 1)
#define ENABLE_PRINTF_DUMP_SIZE() ENABLE_FEATURE_FOR_COMPILE(printfBufSize, 1048576)
#define ENABLE_ASSERT() ENABLE_FEATURE_FOR_COMPILE(assert, 1)
#define ENABLE_ASSERT_DUMP_SIZE() ENABLE_FEATURE_FOR_COMPILE(assertBufSize, 1024)

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE (1024 * 1024)
#endif

#ifndef SIMT_ONE_CORE_DUMP_SIZE
#define SIMT_ONE_CORE_DUMP_SIZE (2048 * 2048)
#endif

using fp4x2_e2m1_t = float4_e2m1x2_t;
using fp4x2_e1m2_t = float4_e1m2x2_t;
using fp8_e5m2_t = float8_e5m2_t;
using fp8_e4m3fn_t = float8_e4m3_t;

namespace AscendCBisheng {
constexpr size_t DUMP_UINTSIZE = ONE_CORE_DUMP_SIZE;
} // namespace AscendCBisheng

#include <stdint.h>
#ifndef TILING_KEY_VAR
#define TILING_KEY_VAR g_tilingKey
#endif

#define TILING_KEY_IS(k) (TILING_KEY_VAR == (k))

#define TILING_KEY_LIST_INOUT(...) TILING_KEY_LIST_INOUT_IMPL(__VA_ARGS__)
#define TILING_KEY_LIST_INOUT_IMPL(...) TILING_KEY_ARGS_CONCAT(TILING_KEY_INDEX_INOUT_, TILING_KEY_ARG_COUNT(__VA_ARGS__)(__VA_ARGS__))

#define TILING_KEY_INDEX_INOUT_1(a) TILING_KEY_VAR == (a)
#define TILING_KEY_INDEX_INOUT_2(a, ...) TILING_KEY_INDEX_INOUT_1(a) || TILING_KEY_INDEX_INOUT_1(__VA_ARGS__)
#define TILING_KEY_INDEX_INOUT_3(a, ...) TILING_KEY_INDEX_INOUT_1(a) || TILING_KEY_INDEX_INOUT_2(__VA_ARGS__)
#define TILING_KEY_INDEX_INOUT_4(a, ...) TILING_KEY_INDEX_INOUT_1(a) || TILING_KEY_INDEX_INOUT_3(__VA_ARGS__)
#define TILING_KEY_INDEX_INOUT_5(a, ...) TILING_KEY_INDEX_INOUT_1(a) || TILING_KEY_INDEX_INOUT_4(__VA_ARGS__)
#define TILING_KEY_INDEX_INOUT_6(a, ...) TILING_KEY_INDEX_INOUT_1(a) || TILING_KEY_INDEX_INOUT_5(__VA_ARGS__)
#define TILING_KEY_INDEX_INOUT_7(a, ...) TILING_KEY_INDEX_INOUT_1(a) || TILING_KEY_INDEX_INOUT_6(__VA_ARGS__)
#define TILING_KEY_INDEX_INOUT_8(a, ...) TILING_KEY_INDEX_INOUT_1(a) || TILING_KEY_INDEX_INOUT_7(__VA_ARGS__)

#define TILING_KEY_ARG_COUNT(...) TILING_KEY_ARG_COUNT_IMPL(__VA_ARGS__,8,7,6,5,4,3,2,1,0)
#define TILING_KEY_ARG_COUNT_IMPL(_1,_2,_3,_4,_5,_6,_7,_8,N,...) N

#define TILING_KEY_ARGS_CONCAT(a,b) TILING_KEY_ARGS_CONCAT_IMPL(a,b)
#define TILING_KEY_ARGS_CONCAT_IMPL(a, b) a##b

#ifdef __CHECK_FEATURE_AT_PRECOMPILE
#define TILING_KEY_LIST(...) (TILING_KEY_LIST_INOUT(__VA_ARGS__)) "TILING_KEY_LIST"
#else
#define TILING_KEY_LIST(...) (TILING_KEY_LIST_INOUT(__VA_ARGS__))
#endif

#define IMPL_MODE_IS(x) constexpr((impl_mode::x) == 1)

#endif // ASCENDC_BISHENG_UTILS_KERNEL_MACROS_H
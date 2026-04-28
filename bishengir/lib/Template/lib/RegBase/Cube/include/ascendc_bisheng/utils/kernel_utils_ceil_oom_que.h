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
 * \file kernel_utils_ceil_oom_que.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_UTILS_KERNEL_CEIL_OOM_QUE_H
#define ASCENDC_BISHENG_UTILS_KERNEL_CEIL_OOM_QUE_H
#include "../common_types.h"
#include "./kernel_utils_macros.h"
namespace AscendCBisheng {

__aicore__ constexpr inline uint32_t DivCeil(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ constexpr inline uint32_t AlignUp(uint32_t a, uint32_t b)
{
    return DivCeil(a, b) * b;
}

__aicore__ constexpr inline uint32_t ConstCeil(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t Ceil(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ constexpr inline int32_t CeilDivision(int32_t num1, int32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
}

// only for ascend910, ascend310p
__aicore__ inline void WriteBackOverflow(GM_ADDR overflowStatus)
{
    (void)overflowStatus;
#if (__NPU_ARCH__ == 5102)
#elif defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 1001) || (__NPU_ARCH__ == 2002))
    uint64_t statusOverflow[1] = {0};
    statusOverflow[0] = get_status();
    statusOverflow[0] = (statusOverflow[0] << 0x20) >> 0x20;
    uint64_t statusMask = 0x520;
    statusOverflow[0] = statusOverflow[0] & statusMask;
    if (statusOverflow[0] != 0) {
        uint64_t *ptr = (uint64_t *)get_imm(0x43FE0);
        uint64_t buff[0x4];
        buff[0x0] = ptr[0x0];
        buff[0x1] = ptr[0x1];
        buff[0x2] = ptr[0x2] | statusOverflow[0x0];
        buff[0x3] = ptr[0x3];
        if (buff[0x0] == 0) {
            ptr[0x0] = 0xFFFFFFFFFFFFFFFF;
            ptr[0x1] = block_idx;
        }
        ptr[0x2] = buff[0x2];

        __ubuf__ uint8_t* tmpStatus = (__ubuf__ uint8_t *)get_imm(0);
        *tmpStatus = 0;
        if (buff[0x2] > 0) {
            *tmpStatus = 0x3;
        }
        pipe_barrier(PIPE_ALL);
        copy_ubuf_to_gm(((__gm__ int32_t *)overflowStatus), ((__ubuf__ int32_t *)tmpStatus), 0, 1, 1, 0, 0);
        pipe_barrier(PIPE_ALL);
    }
#endif
}

template <typename T>
__aicore__ static inline void OOMCheckTensorListRange(__gm__ T *gmInputAddr, const int inputSize)
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    uint64_t ioCount = g_oomAddrArange.count;
    if (ioCount >= g_oomAddrRangeMaxSize) {
        return;
    }
    g_oomAddrArange.addr[ioCount] = reinterpret_cast<uintptr_t>(gmInputAddr);
    g_oomAddrArange.len[ioCount] = inputSize;
    g_oomAddrArange.isLevelOnePointer[ioCount] = 0;
    g_oomAddrArange.count += 1;
#endif
}

__aicore__ static inline bool OOMCheckAddrInTensorList(uint64_t index, uintptr_t gmAddrConvert,
    uintptr_t& inputOutputAddr, uint64_t& inputOutputLen)
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    uintptr_t gmInputAddr = g_oomAddrArange.addr[index];
    uint64_t inputSize = g_oomAddrArange.len[index] & 0xffff;
    uint64_t scaleTmp = (g_oomAddrArange.len[index] >> 16) & 0xffff;  // high 16bit is scale value
    uint64_t scaleValue = (scaleTmp == 0) ? 1 : scaleTmp;

    __gm__ uint64_t *dynamicPtr = (__gm__ uint64_t *)gmInputAddr;
    uint64_t dynamicOffset = *dynamicPtr / 8;
    uint64_t offset = 1;
    __gm__ uint64_t *dynAddr = dynamicPtr + dynamicOffset;
    while (offset < dynamicOffset) {
        dynamicPtr += 1;
        offset += 1;
        inputOutputAddr = reinterpret_cast<uintptr_t>(*dynAddr);
        dynAddr = dynAddr + 1;
        uint64_t dimCnt = *dynamicPtr;
        uint64_t dims = dimCnt & 0xFFFFFFFF;
        uint64_t tensorSize = inputSize;
        for (int i = 0; i < dims; i++) {
            dynamicPtr += 1;
            offset += 1;
            tensorSize = tensorSize * (*dynamicPtr);
        }
        inputOutputLen = tensorSize / scaleValue;
        if (gmAddrConvert >= inputOutputAddr && gmAddrConvert < inputOutputAddr + inputOutputLen)
        {
            return true;
        }
    }
#endif
    (void)index;
    (void)gmAddrConvert;
    (void)inputOutputAddr;
    (void)inputOutputLen;
    return false;
}

template <typename T>
__aicore__ static inline void OOMCheckAddrRange(__gm__ T* gmAddr, const uint64_t gmSize)
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    uint64_t ioCount = g_oomAddrArange.count;
    if (ioCount >= g_oomAddrRangeMaxSize) {
        return;
    }
    g_oomAddrArange.addr[ioCount] = reinterpret_cast<uintptr_t>(gmAddr);
    g_oomAddrArange.len[ioCount] = gmSize;
    g_oomAddrArange.isLevelOnePointer[ioCount] = 1;
    g_oomAddrArange.count += 1;
#endif
}

template <typename T>
__aicore__ static inline void OOMAddAddrForL2Cache(__gm__ T* gmAddr, __gm__ T* oriAddr)
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    uint64_t ioCount = g_oomAddrArange.count;
    if (ioCount >= g_oomAddrRangeMaxSize) {
        return;
    }

    // gmAddr: addr with l2cache offset
    if (gmAddr != oriAddr) {
        for (uint32_t i = 0; i < ioCount; i++) {
            if (g_oomAddrArange.addr[i] <= reinterpret_cast<uintptr_t>(oriAddr) &&
                reinterpret_cast<uintptr_t>(oriAddr) <
                    reinterpret_cast<uintptr_t>(g_oomAddrArange.addr[i]) + g_oomAddrArange.len[i]) {
                g_oomAddrArange.addr[ioCount] = reinterpret_cast<uintptr_t>(gmAddr);
                g_oomAddrArange.len[ioCount] =
                    g_oomAddrArange.len[i] -
                    (reinterpret_cast<uintptr_t>(oriAddr) - reinterpret_cast<uintptr_t>(g_oomAddrArange.addr[i]));
                g_oomAddrArange.isLevelOnePointer[ioCount] = 1;
                g_oomAddrArange.count += 1;
                return;
            }
        }
    }
#endif
}

__aicore__ static inline void OOMInit()
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    g_oomAddrArange.count = 0;
#endif
}

struct TQueConfig {
    bool nd2nz = false;
    bool nz2nd = false;
    bool scmBlockGroup = false;
    uint32_t bufferLen = 0;
    uint32_t bufferNumber = 0;
    uint32_t consumerSize = 0;
    TPosition consumer[8] = {};
    bool enableStaticEvtId = false;
    bool enableLoopQueue = false;
};

__aicore__ constexpr TQueConfig GetTQueConfig(bool nd2nzIn, bool nz2ndIn, bool scmBlockGroupIn,
    uint32_t bufferLenIn, uint32_t bufferNumberIn, uint32_t consumerSizeIn,
    const TPosition consumerIn[], bool enableStaticEvtIdIn, bool enableLoopQueueIn)
{
    return {
        .nd2nz = nd2nzIn,
        .nz2nd = nz2ndIn,
        .scmBlockGroup = scmBlockGroupIn,
        .bufferLen = bufferLenIn,
        .bufferNumber = bufferNumberIn,
        .consumerSize = consumerSizeIn,
        .consumer = {consumerIn[0], consumerIn[1], consumerIn[2], consumerIn[3],
            consumerIn[4], consumerIn[5], consumerIn[6], consumerIn[7]},
        .enableStaticEvtId = enableStaticEvtIdIn,
        .enableLoopQueue = enableLoopQueueIn
    };
}

__aicore__ constexpr TQueConfig GetTQueConfig(const int32_t mask)
{
    return {
        .nd2nz = static_cast<bool>(static_cast<uint32_t>(mask) & 0x1u),
        .nz2nd = static_cast<bool>((static_cast<uint32_t>(mask) & 0x2u) >> 1),
        .scmBlockGroup = static_cast<bool>((static_cast<uint32_t>(mask) & 0x4u) >> 2),
        .bufferLen = 0,
        .bufferNumber = 0,
        .consumerSize = 0,
        .consumer = {TPosition::MAX, TPosition::MAX, TPosition::MAX, TPosition::MAX,
            TPosition::MAX, TPosition::MAX, TPosition::MAX, TPosition::MAX},
        .enableStaticEvtId = false,
        .enableLoopQueue = false
    };
}

__aicore__ constexpr TQueConfig GetTQueConfig(const TQueConfig* conf)
{
    return {
        .nd2nz = conf->nd2nz,
        .nz2nd = conf->nz2nd,
        .scmBlockGroup = conf->scmBlockGroup,
        .bufferLen = conf->bufferLen,
        .bufferNumber = conf->bufferNumber,
        .consumerSize = conf->consumerSize,
        .consumer = {conf->consumer[0], conf->consumer[1], conf->consumer[2], conf->consumer[3],
            conf->consumer[4], conf->consumer[5], conf->consumer[6], conf->consumer[7]},
        .enableStaticEvtId = conf->enableStaticEvtId,
        .enableLoopQueue = conf->enableLoopQueue
    };
}

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)) || defined(__ASC_NPU_HOST__)
__aicore__ constexpr bool UseAltBufId(TPosition queDstPos, TPosition dstConsumerPos, uint32_t consumerSize)
{
    if (consumerSize <= 1) {
        return false;
    }
    return queDstPos != dstConsumerPos;
}

__aicore__ constexpr bool IdentifyPos(TQueConfig config, TPosition pos)
{
    if (config.consumerSize <= 1) {
        return true;
    }
    return config.consumer[0] == pos;
}
#endif

template <bool b> struct BoolInst {
    using Type = BoolInst<b>;
    static constexpr bool value = b;
};

using TrueType = BoolInst<true>;
using FalseType = BoolInst<false>;

template <typename T, typename U> struct IsSameType : public FalseType {};

template <typename T> struct IsSameType<T, T> : public TrueType {};

template <typename... Arg>
struct Tuple {};

template <typename T, typename U, typename... Args>
__aicore__ constexpr bool SupportType()
{
    if constexpr (sizeof...(Args) > 0) {
        return IsSameType<T, U>::value || SupportType<T, Args...>();
    }
    return IsSameType<T, U>::value;
}

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)) || defined(__ASC_NPU_HOST__)
template <typename T> struct GetComplexElementType {
    using Type = T;
};

template<class T>
struct Complex {
    using EleType = typename GetComplexElementType<T>::Type;
    __simd_callee__ inline Complex() : real(0), imag(0) {}
    __simd_callee__ inline Complex(T realTmp, T imagTmp) : real(realTmp), imag(imagTmp) {}

    template<class U>
    __simd_callee__ inline Complex(const U realTmp) : real(T(realTmp)), imag(0) {}

    __simd_callee__ inline Complex(const Complex& other) : real(other.real), imag(other.imag) {}

    __simd_callee__ inline bool operator == (const Complex& val) const
    {
        return (real == val.real) && (imag == val.imag);
    }

    __simd_callee__ inline Complex& operator = (const Complex& other)
    {
        if (this == &other) {
            return *this;
        }
        real = other.real;
        imag = other.imag;
        return *this;
    }

    T real;
    T imag;
};
#endif

template <typename T, int U, int... Args> __aicore__ constexpr bool SupportBytes()
{
    if constexpr (sizeof...(Args) > 0) {
        return sizeof(T) == U || SupportBytes<T, Args...>();
    }
    return sizeof(T) == U;
}
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || \
    (__NPU_ARCH__ == 3103) || (__NPU_ARCH__ == 3113))
template <auto T, auto U, auto... Args> __aicore__ constexpr bool SupportEnum()
{
    if constexpr (sizeof...(Args) > 0) {
        return T == U || SupportEnum<T, Args...>();
    }
    return T == U;
}

#if !defined(__DAV_L310_EFF__)
template <auto funcPtr, typename... Args> __aicore__ inline void VF_CALL(Args &&... args)
{
    __VEC_SCOPE__
    {
        funcPtr(args...);
    }
}
#endif
#endif
} // namespace AscendCBisheng

namespace AscendCBisheng {
template <typename T> class LocalTensor;

template <auto T, auto U, auto... Args> __aicore__ constexpr bool SupportEnum()
{
    if constexpr (sizeof...(Args) > 0) {
        return T == U || SupportEnum<T, Args...>();
    }
    return T == U;
}

template <auto funcPtr, typename... Args> __aicore__ inline void VF_CALL(Args &&... args)
{

}
} // namespace AscendCBisheng

#endif // ASCENDC_BISHENG_UTILS_KERNEL_CEIL_OOM_QUE_H

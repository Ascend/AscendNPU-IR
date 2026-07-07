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
 * \file kernel_utils_constants.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_UTILS_KERNEL_CONSTANTS_H
#define ASCENDC_BISHENG_UTILS_KERNEL_CONSTANTS_H
#include "./kernel_utils_ceil_oom_que.h"
#include "./kernel_utils_dump_constants.h"

namespace AscendCBisheng {
const int32_t DEFAULT_BLK_NUM = 8;
const int32_t POWER_MASK_NUM = 8;
const int32_t HALF_FACTOR = 2;
const int32_t DOUBLE_FACTOR = 2;
const int32_t DEFAULT_BLK_STRIDE = 1;
const uint8_t DEFAULT_REPEAT_STRIDE = 8;
const uint8_t HALF_DEFAULT_REPEAT_STRIDE = 4;
const uint8_t ONE_FOURTH_DEFAULT_REPEAT_STRIDE = 2;
const uint64_t FULL_MASK = 0xffffffffffffffff;
const uint64_t CONST_MASK_VALUE = 0x8000000000000000;
const uint16_t MAX_HALF_MASK_LEN = 64;
const int32_t DEFAULT_C0_SIZE = 32;
const int32_t DEFAULT_BLOCK_SIZE = 256;
const int32_t MAX_REPEAT_TIMES = 255;
const int32_t MIN_REPEAT_TIMES = 0;
const bool DEFAULT_REPEAT_STRIDE_MODE = 0;
const bool STRIDE_SIZE_MODE = 0;
const int32_t ONE_BYTE_BIT_SIZE = 8;
const uint16_t MASK_ARRAY_SIZE = 4;
const int32_t B32_BIT_SIZE = 32;
const uint32_t TOTAL_L0A_SIZE = 64 * 1024;
const uint32_t TOTAL_L0B_SIZE = 64 * 1024;
const uint32_t TMP_UB_SIZE = 8 * 1024;
const uint32_t MAX_SLICE_SIZE = 6 * 256;
const uint32_t F32_INF = 0x7f800000;
const uint32_t F32_NEG_INF = 0xff800000;
const uint32_t F32_NAN = 0x7fc00000;

const uint16_t VALUE_512 = 512;         // align with 512B / value range [0, 512]
const uint16_t UINT12_MAX = 4095;       // 12 bit range is [0, 4095]
const uint16_t UINT15_MAX = 32767;      // 15 bit range is [0, 32767]

// Ctrl bit Pos
constexpr int32_t CTRL_46_BIT = 46;
constexpr int32_t CTRL_47_BIT = 47;
constexpr int32_t CTRL_48_BIT = 48;
constexpr int32_t CTRL_53_BIT = 53;

// power param
constexpr uint32_t TENSOR_TENSOR_FLOAT_POWER_FACTOR = 4;
constexpr uint32_t TENSOR_TENSOR_INT_POWER_FACTOR = 6;
constexpr uint32_t TENSOR_TENSOR_HALF_POWER_FACTOR = 7;
constexpr uint32_t TENSOR_SCALAR_FLOAT_POWER_FACTOR = 5;
constexpr uint32_t TENSOR_SCALAR_INT_POWER_FACTOR = 7;
constexpr uint32_t TENSOR_SCALAR_HALF_POWER_FACTOR = 7;
constexpr uint32_t POWER_TWO = 2;
constexpr uint32_t POWER_THREE = 3;
constexpr uint32_t POWER_INT32_BITS = 32;

// int4b_t param
constexpr uint32_t INT4_TWO = 2;
constexpr uint32_t INT4_BIT_NUM = 4;

namespace ConstantsInternal {
constexpr uint32_t ASCENDC_B4_TWO = 2;
constexpr uint32_t ASCENDC_B4_BIT_NUM = 4;
} // namespace ConstantsInternal

// AddDeqRelu param
constexpr int32_t DEQ_SHIFT_LEFT_17_BIT = 131072;
constexpr float DEQ_SHIFT_RIGHT_17_BIT = 1.0 / DEQ_SHIFT_LEFT_17_BIT;
constexpr int8_t ADDDEQRELU_MASK_MODE_ONE = 1;
constexpr int8_t ADDDEQRELU_MASK_MODE_TWO = 2;

const int32_t TOTAL_VEC_LOCAL_SIZE = 248 * 1024;
const uint32_t TOTAL_UB_SIZE = 248 * 1024;
const uint32_t TMP_UB_OFFSET = 248 * 1024;
const uint32_t TOTAL_L1_SIZE = 512 * 1024;
const uint32_t SINGLE_MSG_SIZE = 64;
const uint32_t CACHE_LINE_SIZE = 64;
const uint32_t TOTAL_L0C_SIZE = 256 * 1024;
const uint32_t VECTOR_REG_WIDTH = 256;
const uint32_t VECTOR_REG_WIDTH_2XVL = 512;
const uint32_t ONE_BLOCK_SIZE = 32;

const int32_t BLOCK_CUBE = 16;

const uint16_t ONE_BLK_SIZE = 32;

const int32_t CUBE_MAX_SIZE = 256;

const uint8_t PAD_SIZE = 4;
const uint8_t MRG_SORT_ELEMENT_LEN = 4;
const uint8_t DEFAULT_DATA_COPY_NBURST = 1;
const uint8_t DEFAULT_DATA_COPY_STRIDE = 0;
const int32_t BYTE_PER_FRACTAL = 512;
const int32_t SRC_BURST_LEN_SIZE_ELE = 16;
const int32_t SRC_GAP_SIZE_BYTE = 32;
const int32_t DST_BURST_LEN_SIZE_ELE = 256;
const int32_t VREDUCE_PER_REP_OUTPUT = 2;
const uint16_t ONE_PARAM_SIZE = 8;
const uint16_t AIV_CORE_NUM = 72;
const uint16_t DUMP_MSG_HEAD_SIZE = 24;
const int32_t ONE_REPEAT_BYTE_SIZE = 256;
const int32_t FULL_MASK_LEN = 128;
const int32_t HLAF_MASK_LEN = 64;
const int32_t DEFAULT_REDUCE_DST_REP_SRIDE = 1;
const uint8_t B64_BYTE_SIZE = 8;
const uint8_t B32_BYTE_SIZE = 4;
const uint8_t B16_BYTE_SIZE = 2;
const uint8_t B8_BYTE_SIZE = 1;
const uint8_t B32_DATA_NUM_PER_BLOCK = 8;
const uint8_t B16_DATA_NUM_PER_BLOCK = 16;
const int32_t B16_DATA_NUM_PER_REPEAT = 128;
const int32_t B32_DATA_NUM_PER_REPEAT = 64;

const uint32_t B64_DATA_NUM_PER_REPEAT = 32;
const uint32_t B4_BYTE_SIZE_PER_REPEAT = 64;
const uint32_t L1_DUMP_UB_SIZE = TOTAL_UB_SIZE - 32 * 1024;

const int32_t BLOCK_STRIDE_POS_IN_SM = 16;
const int32_t PLD_BUFFER_SIZE = 2;
const uint8_t FIXPIPE_DEQ_TENSOR_SIZE = 16;
const uint8_t SET_DATA_EXP_ZERO = 0;
const uint8_t SET_DATA_EXP_ONE = 1;
const uint8_t SET_DATA_EXP_TWO = 2;
const uint8_t SET_DATA_EXP_THREE = 3;
const uint8_t VDEQ_TENSOR_SIZE = 16;
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;

// nchwconv address list size
const int32_t NCHW_CONV_ADDR_LIST_SIZE = 16;
const int32_t VA_REG_ARRAY_LEN = 8;
const uint8_t CONV2D_IMG_SIZE = 2;
const uint8_t CONV2D_KERNEL_SIZE = 2;
const uint8_t CONV2D_STRIDE = 2;
const uint8_t CONV2D_PAD = 4;
const uint8_t CONV2D_DILATION = 2;
const int32_t K_MAX_DIM = 8;

const uint32_t TWO_OF_STACK_BUFFER = 2;
const uint32_t THREE_OF_STACK_BUFFER = 3;
const uint32_t HALF_REPEAT_SIZE = ONE_REPEAT_BYTE_SIZE / B16_BYTE_SIZE;
const uint32_t FLOAT_REPEAT_SIZE = ONE_REPEAT_BYTE_SIZE / B32_BYTE_SIZE;
const uint32_t ONE_REPEAT_FLOAT_SIZE = ONE_REPEAT_BYTE_SIZE / B32_BYTE_SIZE;
const uint32_t ONE_REPEAT_HALF_SIZE = ONE_REPEAT_BYTE_SIZE / B16_BYTE_SIZE;
const uint32_t MAX_REPEAT_FLOAT_SIZE = ONE_REPEAT_FLOAT_SIZE * MAX_REPEAT_TIMES;
const uint32_t MAX_REPEAT_HALF_SIZE = ONE_REPEAT_HALF_SIZE * MAX_REPEAT_TIMES;
const uint32_t ONE_BLK_HALF_NUM = ONE_BLK_SIZE / B16_BYTE_SIZE;
const uint32_t ONE_BLK_FLOAT_NUM = ONE_BLK_SIZE / B32_BYTE_SIZE;
// #if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
namespace ConstantsInternal {
const uint32_t ONE_BLK_FP4_NUM = 64;
const uint32_t ONE_BLK_B2_NUM = 128;
} // namespace ConstantsInternal
// #endif
const uint32_t BRCB_BROADCAST_NUMBER = 8;
const uint32_t BRCB_MAX_REPEAT_SIZE = BRCB_BROADCAST_NUMBER * MAX_REPEAT_TIMES;
const int32_t MIN_BLOCK_LEN = 1;
const uint32_t PAIR_REDUCE_REPEAT_STRIDE_LEN = 128;
const uint32_t PAIR_REDUCE_SUM_MERGES = 2;
const uint32_t TWO_HUNDRED_FIFTY_TWO_REPEAT = 252;
const uint32_t TWO_HUNDRED_FIFTY_TWO_REPEAT_BYTE_SIZE = TWO_HUNDRED_FIFTY_TWO_REPEAT * ONE_REPEAT_BYTE_SIZE;
const uint32_t REDUCEV2_MODE_SEVEN = 7;
const uint32_t DROPOUT_MODE_BYTE_MISALIGN = 1;
const uint32_t DROPOUT_MODE_BYTE_ALIGN = 2;
const uint32_t DROPOUT_MODE_BIT_ALIGN = 3;
const uint32_t DROPOUT_MODE_BIT_MISALIGN = 4;
const uint32_t REDUCEV2_MODE_ONE = 1;
const uint32_t REDUCEV2_MODE_TWO = 2;
const uint32_t REDUCEV2_MODE_THREE = 3;

// 4dTrans param size
const int32_t B8_TMP_ELE_LEN = 1024;
const int32_t B16_TMP_ELE_LEN = 256;
const int32_t B32_TMP_ELE_LEN = 128;
const int32_t B8_TRANS_LEN = 1024;
const int32_t B8_TRANS_FRACTAL = 512;
const int32_t B8_TRANS_ROW = 32;
const int32_t B8_COPY_COL = 32;

// load3dPro config
const uint64_t LOAD_M_START_POSITION = 48;
const uint64_t LOAD_K_START_POSITION = 32;
const uint64_t LOAD_M_EXTENSION = 16;
const uint64_t LOAD_DILATION_FILTER_H = 40;
const uint64_t LOAD_DILATION_FILTER_W = 32;
const uint64_t LOAD_FILTER_H = 24;
const uint64_t LOAD_FILTER_W = 16;
const uint64_t LOAD_STRIDE_H = 8;

namespace Internal {
constexpr int32_t TSCM_CROSS_SYNC_ID_MAX = 11;
}

template <typename T> struct GetPadValueType {
    using Type = T;
};

// To support FP8 datacopypad, pad type needs transfer to b8
template <> struct GetPadValueType<fp8_e5m2_t> {
    using Type = uint8_t;
};

template <> struct GetPadValueType<fp8_e4m3fn_t> {
    using Type = uint8_t;
};

template <> struct GetPadValueType<hifloat8_t> {
    using Type = uint8_t;
};

// To support FP4 datacopypad, pad type needs transfer to b8
template <> struct GetPadValueType<fp4x2_e1m2_t> {
    using Type = uint8_t;
};

template <> struct GetPadValueType<fp4x2_e2m1_t> {
    using Type = uint8_t;
};
// #endif

template <bool condition, class T1, class T2>
struct Conditional {
    using type = T1;
};

template <class T1, class T2>
struct Conditional<false, T1, T2> {
    using type = T2;
};

template <bool condition1, bool condition2, class T1, class T2, class T3, class T4> struct ConditionalMulti {
    using type = typename Conditional<condition1, typename Conditional<condition2, T1, T2>::type,
        typename Conditional<condition2, T3, T4>::type>::type;
};

template <int bitNum, bool sign = true>
struct IntegerSubType {
    static int const kBits = bitNum;
    static bool const kSigned = sign;

    using T = typename Conditional<kSigned, int8_t, uint8_t>::type;
    using Storage = uint8_t;

    static Storage const mask = Storage(((static_cast<uint64_t>(1)) << static_cast<uint32_t>(kBits)) - 1);
    Storage storage;
    __aicore__ inline IntegerSubType() = default;

    __aicore__ inline IntegerSubType(uint32_t value)
        : storage(reinterpret_cast<Storage const &>(value) & mask) {}

    __aicore__ inline IntegerSubType(int32_t value)
        : storage(reinterpret_cast<Storage const &>(value) & mask) {}

    __aicore__ inline operator T() const
    {
        if (kSigned && ((storage & Storage(static_cast<uint64_t>(1) << static_cast<uint32_t>(kBits - 1))) != 0)) {
            // Sign extend
            return T(storage) | ~T(mask);
        }
        return T(storage);
    }

    __aicore__ inline bool operator == (IntegerSubType const &rhs) const
    {
        return storage == rhs.storage;
    }

    __aicore__ inline bool operator != (IntegerSubType const &rhs) const
    {
        return storage != rhs.storage;
    }

    __aicore__ inline bool operator > (IntegerSubType const &rhs) const
    {
        bool lhsIsNeg = (this->storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(this->kBits - 1)));
        bool rhsIsNeg = (rhs.storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(rhs.kBits - 1)));
        if (kSigned && (lhsIsNeg != rhsIsNeg)) {
            return (!lhsIsNeg) && rhsIsNeg;
        }
        return this->storage > rhs.storage;
    }

    __aicore__ inline bool operator >= (IntegerSubType const &rhs) const
    {
        bool lhsIsNeg = (this->storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(this->kBits - 1)));
        bool rhsIsNeg = (rhs.storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(rhs.kBits - 1)));
        if (kSigned && (lhsIsNeg != rhsIsNeg)) {
            return (!lhsIsNeg) && rhsIsNeg;
        }
        return storage >= rhs.storage;
    }

    __aicore__ inline bool operator < (IntegerSubType const &rhs) const
    {
        return !(*this >= rhs);
    }

    __aicore__ inline bool operator <= (IntegerSubType const &rhs) const
    {
        return !(*this > rhs);
    }
};

using int4b_t = IntegerSubType<INT4_BIT_NUM, true>;

using fp8_e8m0_t = uint8_t;
using mx_fp8_e5m2_t = struct {};
using mx_fp8_e4m3_t = struct {};

template <typename T>
struct GetDstType {
    using Type = T;
};

template <> struct GetDstType<mx_fp8_e5m2_t> {
    using Type = fp8_e5m2_t;
};

template <> struct GetDstType<mx_fp8_e4m3_t> {
    using Type = fp8_e4m3fn_t;
};

struct maskStruct {
    uint64_t maskArray[MASK_ARRAY_SIZE] = { 0 };
};

template <typename T>
__aicore__ constexpr bool IsHalfByteDataType()
{
    return SupportType<T, int4b_t, fp4x2_e2m1_t, fp4x2_e1m2_t>();
}

template <typename T> struct SizeOfBits {
    static constexpr uint32_t value = sizeof(T) * ONE_BYTE_BIT_SIZE;
};

template <>
struct SizeOfBits<int4b_t> {
    static int const value = INT4_BIT_NUM;
};

#if (__NPU_ARCH__ == 5102)
template <>
struct SizeOfBits<int2b_t> {
    static constexpr uint32_t value = INT2_BIT_NUM;
};
#endif

template <> struct SizeOfBits<fp4x2_e2m1_t> {
    static constexpr uint32_t value = INT4_BIT_NUM;
};
template <> struct SizeOfBits<fp4x2_e1m2_t> {
    static constexpr uint32_t value = INT4_BIT_NUM;
};

__aicore__ inline bool CheckCastOverlappingHigh(const uint64_t dstAddr, const uint64_t srcAddr,
    const uint32_t dstTypeSize, const uint32_t srcTypeSize, const uint32_t count)
{
    uint64_t addrLow = dstAddr > srcAddr ? srcAddr : dstAddr;
    uint64_t addrHigh = dstAddr > srcAddr ? dstAddr : srcAddr;
    uint64_t needSizeLow = dstAddr > srcAddr ? count * srcTypeSize : count * dstTypeSize;

    if ((srcTypeSize < dstTypeSize) && (srcAddr >= AlignUp(dstAddr + count * srcTypeSize, ONE_BLK_SIZE))) {
        return true;
    }
    if (dstTypeSize > srcTypeSize && srcAddr == dstAddr) {
        return false;
    }
    if ((needSizeLow > static_cast<uint64_t>(ONE_REPEAT_BYTE_SIZE)) && (srcAddr != dstAddr) &&
        ((addrLow + needSizeLow > addrHigh))) {
        return false;
    }
    return true;
}

__aicore__ inline constexpr uint8_t CalculatesShiftedBit(uint32_t bufferLen)
{
    uint8_t pos = 0;
    while (bufferLen > 1) {
        bufferLen >>= 1;
        pos++;
    }
    return pos;
}

#if defined(__ASCENDC_SUPERKERNEL_EARLY_START_V1) || defined(__ASCENDC_SUPERKERNEL_EARLY_START_V2)
constexpr uint32_t ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_AIC = 0b00;
constexpr uint32_t ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_AIV = 0b01;
constexpr uint32_t ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_MIX = 0b10;
constexpr uint32_t ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_AIC = 0b0100;
constexpr uint32_t ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_AIV = 0b0101;
constexpr uint32_t ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_MIX = 0b0110;
constexpr uint32_t ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_AIC = 0b1000;
constexpr uint32_t ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_AIV = 0b1001;
constexpr uint32_t ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_MIX = 0b1010;
#endif

#if defined(__ASCENDC_SUPERKERNEL_AUTO_SYNC_ALL__)
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_superKernelAutoSyncAllConfigGmBaseAddr;
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_superKernelAutoSyncAllConfigGmAddr;
__BLOCK_LOCAL__ __inline__ uint32_t g_superKernelAutoSyncAllSyncIdx;
__BLOCK_LOCAL__ __inline__ uint32_t g_superKernelAutoSyncAllEnable;
constexpr uint8_t SK_AUTO_SYNC_ALL_VALID_MAGIC_NUM = 0x0;
// |    32bits  |   16bits  |   8bits   |   8bits   |
// |  sync idx  | sync type |   is end  | is valid  |
constexpr uint32_t SK_AUTO_SYNC_ALL_IS_VALID_CONFIG_BIT_OFFSET = 0;
constexpr uint32_t SK_AUTO_SYNC_ALL_IS_END_CONFIG_BIT_OFFSET = 8;
constexpr uint32_t SK_AUTO_SYNC_ALL_SYNC_TYPE_CONFIG_BIT_OFFSET = 16;
constexpr uint32_t SK_AUTO_SYNC_ALL_SYNC_IDX_CONFIG_BIT_OFFSET = 32;

__aicore__ inline void SuperKernelAutoSyncAllDcciBarrier()
{
    __asm__ __volatile__("");
}
#endif
} // namespace AscendCBisheng
#endif // ASCENDC_BISHENG_UTILS_KERNEL_CONSTANTS_H
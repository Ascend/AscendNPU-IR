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
 * \file kernel_utils_dump_constants.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_KERNEL_UTILS_DUMP_CONSTANTS_H
#define ASCENDC_BISHENG_KERNEL_UTILS_DUMP_CONSTANTS_H
#include "./kernel_utils_ceil_oom_que.h"
namespace AscendCBisheng {
const int32_t ONE_DUMP_BACKUP_SIZE = 1024;
const int32_t DUMP_UB_SIZE = 256;
const int32_t DUMP_EXC_FLAG = 7;

// BlockInfo Pos
const uint32_t BLOCK_INFO_LEN_POS = 0;
const uint32_t BLOCK_INFO_CORE_POS = 1;
const uint32_t BLOCK_INFO_BLOCKNUM_POS = 2;
const uint32_t BLOCK_INFO_DUMPOFFSET_POS = 3;
const uint32_t BLOCK_INFO_MAGIC_POS = 4;
const uint32_t BLOCK_INFO_RSV_POS = 5;
const uint32_t BLOCK_INFO_DUMP_ADDR = 6;
const uint32_t BLOCK_INFO_MAGIC_NUM = 0x5aa5bccd;
// DUMP_META Pos 以uint8_t为单位计算位置
const uint32_t DUMP_META_TYPE_POS = 0;
const uint32_t DUMP_META_LEN_POS = 4;
const uint16_t DUMP_META_BLOCK_DIM_POS = 8;
const uint16_t DUMP_META_SIMT_THREAD_ID_POS = 8;
const uint8_t DUMP_META_CORE_TYPE_POS = 10;
const uint8_t DUMP_META_TASK_RATION = 11;
const uint32_t DUMP_META_RSV_POS = 12;
// DumpMessageHead Pos
const uint32_t DUMP_MESSAGE_HEAD_TYPE_POS = 0;
const uint32_t DUMP_MESSAGE_HEAD_LEN_POS = 1;
const uint32_t DUMP_MESSAGE_HEAD_ADDR_POS = 2;
const uint32_t DUMP_MESSAGE_HEAD_DATA_TYPE_POS = 3;
const uint32_t DUMP_MESSAGE_HEAD_DESC_POS = 4;
const uint32_t DUMP_MESSAGE_HEAD_BUFFERID_POS = 5;
const uint32_t DUMP_MESSAGE_HEAD_POSITION_POS = 6;
const uint32_t DUMP_MESSAGE_HEAD_DUMP_SIZE_POS = 7;
const uint32_t DUMP_SCALAR_POS = 8;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
const uint32_t DUMP_CORE_COUNT = 108;
#ifdef ASCENDC_RECOGNIZE_SIMT_VF
constexpr uint32_t DUMP_SIMT_CORE_COUNT = 72;
const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * ONE_CORE_DUMP_SIZE + DUMP_SIMT_CORE_COUNT * SIMT_ONE_CORE_DUMP_SIZE;
#else
const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * ONE_CORE_DUMP_SIZE;
#endif
#else
const uint32_t DUMP_CORE_COUNT = 75;
const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * ONE_CORE_DUMP_SIZE;
#endif
// DumpShapeMessageHead Pos
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_TYPE_POS = 0;
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_LEN_POS = 1;
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_DIM_POS = 2;
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_SHAPE_START_POS = 3;
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_RSV_POS = 11;
const uint32_t DUMP_SHAPE_MESSAGE_TL_LEN = 8;

namespace Internal {
enum class DumpTensorDataType : uint32_t {
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_STRING = 13,
    ACL_COMPLEX64 = 16,
    ACL_COMPLEX128 = 17,
    ACL_BF16 = 27,
    ACL_INT4 = 29,
    ACL_UINT1 = 30,
    ACL_COMPLEX32 = 33,
    ACL_HIFLOAT8 = 34,
    ACL_FLOAT8_E5M2 = 35,
    ACL_FLOAT8_E4M3FN = 36,
    ACL_FLOAT8_E8M0 = 37,
    ACL_FLOAT6_E3M2 = 38,
    ACL_FLOAT6_E2M3 = 39,
    ACL_FLOAT4_E2M1 = 40,
    ACL_FLOAT4_E1M2 = 41,
    ACL_MAX = 42,
};
}

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
namespace ConstantsInternal {
// L12UBDumpCtrlMessage signal
const uint32_t L1_2_UB_DUMP_CTRL_SIGNAL_DUMP = 1;
const uint32_t DUMP_INTRA_BLOCK_ID = 14;

// math compute
constexpr uint8_t BIT_32_LEN = 32;
constexpr uint8_t BIT_64_LEN = 64;
constexpr uint64_t UINT_64_MAX = 0xFFFFFFFFFFFFFFFF;
constexpr uint64_t UINT_64_HIGHEST_BIT_MASK = 0x8000000000000000;
constexpr uint32_t INT_32_MAX = 0x7FFFFFFF;
constexpr uint64_t INT_64_MAX = 0x7FFFFFFFFFFFFFFF;
} // namespace ConstantsInternal
#endif

// DumpTimeStamp
const uint32_t DUMP_TIME_STAMP_LEN = 24; // desc_id(uint32_t)+rsv(uint32_t)+cycle(uint64_t)
const uint32_t DUMP_TIME_STAMP_TOTAL_LEN = 32; // 6 * 4
const uint32_t DUMP_TIME_STAMP_LEN_POS = 1;
const uint32_t DUMP_TIME_STAMP_ID_POS = 2;
const uint32_t DUMP_TIME_STAMP_CYCLE_POS = 4;
const uint32_t DUMP_TIME_STAMP_PTR_POS = 6;

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
int32_t TensorWriteFile(const std::string& fileName, const void* buffer, size_t size);
#endif
} // namespace AscendCBisheng
#endif // ASCENDC_BISHENG_KERNEL_UTILS_DUMP_CONSTANTS_H

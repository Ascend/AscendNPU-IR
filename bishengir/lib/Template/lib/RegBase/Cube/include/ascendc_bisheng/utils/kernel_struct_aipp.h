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
 * \file kernel_struct_aipp.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_KERNEL_STRUCT_AIPP_H
#define ASCENDC_BISHENG_KERNEL_STRUCT_AIPP_H

namespace AscendCBisheng {
enum class AippInputFormat : uint8_t {
    YUV420SP_U8 = 0,
    XRGB8888_U8 = 1,
    RGB888_U8 = 4,
    YUV400_U8 = 9,
};

template <typename T>
struct AippPaddingParams {
    uint32_t paddingMode{ 0 };
    T paddingValueCh0{ 0 };
    T paddingValueCh1{ 0 };
    T paddingValueCh2{ 0 };
    T paddingValueCh3{ 0 };
};

struct AippSwapParams {
    bool isSwapRB{ false };
    bool isSwapUV{ false };
    bool isSwapAX{ false };
};

struct AippSingleLineParams {
    bool isSingleLineCopy{ false };
};

struct AippDataTypeConvParams {
    uint8_t dtcMeanCh0{ 0 };
    uint8_t dtcMeanCh1{ 0 };
    uint8_t dtcMeanCh2{ 0 };
    half dtcMinCh0{ 0 };
    half dtcMinCh1{ 0 };
    half dtcMinCh2{ 0 };
    half dtcVarCh0{ 1.0 };
    half dtcVarCh1{ 1.0 };
    half dtcVarCh2{ 1.0 };
    uint32_t dtcRoundMode{ 0 };
};

template <typename T>
struct AippChannelPaddingParams {
    uint32_t cPaddingMode;
    T cPaddingValue;
};

struct AippColorSpaceConvParams {
    bool isEnableCsc{ false };
    int16_t cscMatrixR0C0{ 0 };
    int16_t cscMatrixR0C1{ 0 };
    int16_t cscMatrixR0C2{ 0 };
    int16_t cscMatrixR1C0{ 0 };
    int16_t cscMatrixR1C1{ 0 };
    int16_t cscMatrixR1C2{ 0 };
    int16_t cscMatrixR2C0{ 0 };
    int16_t cscMatrixR2C1{ 0 };
    int16_t cscMatrixR2C2{ 0 };
    uint8_t cscBiasIn0{ 0 };
    uint8_t cscBiasIn1{ 0 };
    uint8_t cscBiasIn2{ 0 };
    uint8_t cscBiasOut0{ 0 };
    uint8_t cscBiasOut1{ 0 };
    uint8_t cscBiasOut2{ 0 };
};

template <typename T> struct AippParams {
    AippPaddingParams<T> paddingParams;
    AippSwapParams swapParams;
    AippSingleLineParams singleLineParams;
    AippDataTypeConvParams dtcParams;
    AippChannelPaddingParams<T> cPaddingParams;
    AippColorSpaceConvParams cscParams;
};
} // namespace AscendCBisheng
#endif // AASCENDC_BISHENG_KERNEL_STRUCT_AIPP_H
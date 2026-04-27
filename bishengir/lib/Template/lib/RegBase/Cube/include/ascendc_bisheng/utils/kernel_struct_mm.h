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
 * \file kernel_struct_mm.h
 * \brief
 */
#ifndef ASCENDC_BISHENG_KERNEL_STRUCT_MM_H
#define ASCENDC_BISHENG_KERNEL_STRUCT_MM_H
#include "./kernel_utils_constants.h"

namespace AscendCBisheng {
// MM intr params
using LoadData2dParams = struct LoadData2DParams;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
struct LoadData2DParams {
    __aicore__ LoadData2DParams() {}

    __aicore__ LoadData2DParams(const uint16_t startIndexIn, const uint8_t repeatTimesIn, const uint16_t srcStrideIn,
        const uint8_t sidIn, const uint16_t dstGapIn, const bool ifTransposeIn, const uint8_t addrModeIn)
        : startIndex(startIndexIn),
          repeatTimes(repeatTimesIn),
          srcStride(srcStrideIn),
          sid(sidIn),
          dstGap(dstGapIn),
          ifTranspose(ifTransposeIn),
          addrMode(addrModeIn)
    {}
    __aicore__ inline void SetStartIndex(uint16_t value)
    {
        startIndex = value;
    }

    __aicore__ inline void SetDstGap(uint16_t value)
    {
        dstGap = value;
    }

    __aicore__ inline void SetSrcStride(uint16_t value)
    {
        srcStride = value;
    }

    __aicore__ inline void SetIfTranspose(bool value)
    {
        ifTranspose = value;
    }

    __aicore__ inline void SetRepeatTimes(uint8_t value)
    {
        repeatTimes = value;
    }

    __aicore__ inline void SetSid(uint8_t value)
    {
        sid = value;
    }

    __aicore__ inline void SetAddrMode(uint8_t value)
    {
        addrMode = value;
    }

    __aicore__ inline void SetMStartPosition(uint32_t mStartPosition_)
    {
        (void)mStartPosition_;
    }

    __aicore__ inline void SetKStartPosition(uint32_t kStartPosition_)
    {
        (void)kStartPosition_;
    }

    __aicore__ inline void SetMStep(uint16_t mStep_)
    {
        (void)mStep_;
    }

    __aicore__ inline void SetKStep(uint16_t kStep_)
    {
        (void)kStep_;
    }

    __aicore__ inline void SetSrcStride(int32_t srcStride_)
    {
        (void)srcStride_;
    }

    __aicore__ inline void SetDstStride(uint16_t dstStride_)
    {
        (void)dstStride_;
    }

    __aicore__ inline void SetQmode(uint8_t qmode_)
    {
        (void)qmode_;
    }

    uint16_t startIndex = 0;
    uint16_t dstGap = 0;
    uint16_t srcStride = 0;
    bool ifTranspose = 0;
    uint8_t repeatTimes = 0;

    uint8_t sid = 0;
    uint8_t addrMode = 0;
};
#else
struct LoadData2DParams {
    __aicore__ LoadData2DParams() {}

    __aicore__ LoadData2DParams(const uint16_t startIndexIn, const uint8_t repeatTimesIn, const uint16_t srcStrideIn,
        const uint8_t sidIn, const uint16_t dstGapIn, const bool ifTransposeIn, const uint8_t addrModeIn)
        : startIndex(startIndexIn),
          repeatTimes(repeatTimesIn),
          srcStride(srcStrideIn),
          sid(sidIn),
          dstGap(dstGapIn),
          ifTranspose(ifTransposeIn),
          addrMode(addrModeIn)
    {}

    uint16_t startIndex = 0;
    uint16_t dstGap = 0;
    uint16_t srcStride = 0;
    bool ifTranspose = 0;
    uint8_t repeatTimes = 0;

    uint8_t sid = 0;
    uint8_t addrMode = 0;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3113) && defined(__DAV_L311__))
using LoadData2DParamsV2 = struct LoadData2DParamsV311Gen;
using LoadData2dTransposeParams = struct LoadData2dTransposeParamsV311Gen;
#elif defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3103) && defined(__DAV_L310__))
using LoadData2DParamsV2 = struct LoadData2DParamsV311Gen;
using LoadData2dTransposeParams = struct LoadData2dTransposeParamsV311Gen;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
using LoadData2DParamsV2 = struct LoadData2DParamsV311Gen;
using LoadData2dTransposeParams = struct LoadData2dTransposeParamsV300;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2103)
using LoadData2DParamsV2 = struct LoadData2DParamsV311Gen;
using LoadData2dTransposeParams = struct LoadData2dTransposeParamsV210;
#else // Turing versions
struct LoadData2DParamsV2 {
    __aicore__ LoadData2DParamsV2() {}

    __aicore__ LoadData2DParamsV2(const uint32_t mStartPositionIn, const uint32_t kStartPositionIn,
        const uint16_t mStepIn, const uint16_t kStepIn, const int32_t srcStrideIn, const uint16_t dstStrideIn,
        const bool ifTransposeIn, const uint8_t sidIn)
        : mStartPosition(mStartPositionIn),
          kStartPosition(kStartPositionIn),
          mStep(mStepIn),
          kStep(kStepIn),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn),
          ifTranspose(ifTransposeIn),
          sid(sidIn)
    {}

    uint32_t mStartPosition = 0;
    uint32_t kStartPosition = 0;
    uint16_t mStep = 0;
    uint16_t kStep = 0;
    int32_t srcStride = 0;
    uint16_t dstStride = 0;
    bool ifTranspose = false;
    uint8_t sid = 0;
};

struct LoadData2dTransposeParams {
    __aicore__ LoadData2dTransposeParams() {}

    __aicore__ LoadData2dTransposeParams(const uint16_t startIndexIn, const uint8_t repeatTimesIn,
        const uint16_t srcStrideIn, const uint16_t dstGapIn, const uint16_t dstfracGapIn, const uint8_t addrModeIn)
        : startIndex(startIndexIn),
          repeatTimes(repeatTimesIn),
          srcStride(srcStrideIn),
          dstGap(dstGapIn),
          dstFracGap(dstfracGapIn),
          addrMode(addrModeIn)
    {}

    __aicore__ LoadData2dTransposeParams(const uint16_t startIndexIn, const uint8_t repeatTimesIn,
        const uint16_t srcStrideIn, const uint16_t dstGapIn, const uint16_t dstfracGapIn)
        : startIndex(startIndexIn),
          repeatTimes(repeatTimesIn),
          srcStride(srcStrideIn),
          dstGap(dstGapIn),
          dstFracGap(dstfracGapIn)
    {}

    uint16_t startIndex = 0;
    uint8_t repeatTimes = 0;
    uint16_t srcStride = 0;
    uint16_t dstGap = 0;
    uint16_t dstFracGap = 0;
    uint8_t addrMode = 0;
};
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
struct Nd2NzParamsV2 {
    uint64_t lookupTable0 = 0;
    uint64_t lookupTable1 = 0;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
struct LoadData2DParamsV300Gen {
    __aicore__ LoadData2DParamsV300Gen()
    {
        startIndex = 0;
        repeatTimes = 0;
        srcStride = 0;
        sid = 0;
        dstGap = 0;
        ifTranspose = false;
        addrMode = 0;
    }

    __aicore__ LoadData2DParamsV300Gen(const uint16_t startIndexIn, const uint8_t repeatTimesIn, const uint16_t srcStrideIn,
        const uint8_t sidIn, const uint16_t dstGapIn, const bool ifTransposeIn, const uint8_t addrModeIn)
    {
        startIndex = startIndexIn;
        repeatTimes = repeatTimesIn;
        srcStride = srcStrideIn;
        sid = sidIn;
        dstGap = dstGapIn;
        ifTranspose = ifTransposeIn;
        addrMode = addrModeIn;
    }

     __aicore__ inline void SetStartIndex(uint16_t value)
    {
        startIndex = value;
    }

    __aicore__ inline void SetDstGap(uint16_t value)
    {
        dstGap = value;
    }

    __aicore__ inline void SetSrcStride(uint16_t value)
    {
        srcStride = value;
    }

    __aicore__ inline void SetIfTranspose(bool value)
    {
        ifTranspose = value;
    }

    __aicore__ inline void SetRepeatTimes(uint8_t value)
    {
        repeatTimes = value;
    }

    __aicore__ inline void SetSid(uint8_t value)
    {
        sid = value;
    }

        __aicore__ inline void SetAddrMode(uint8_t value)
    {
        addrMode = value;
    }

    uint16_t startIndex = 0;
    uint16_t dstGap = 0;
    uint16_t srcStride = 0;
    bool ifTranspose = 0;
    uint8_t repeatTimes = 0;
    uint8_t sid = 0;
    uint8_t addrMode = 0;
};

struct LoadData2dTransposeParamsV210 {
    __aicore__ LoadData2dTransposeParamsV210()
    {
        startIndex = 0;
        repeatTimes = 0;
        srcStride = 0;
        dstGap = 0;
        dstFracGap = 0;
        addrMode = 0;
    }

    __aicore__ LoadData2dTransposeParamsV210(const uint16_t startIndexIn, const uint8_t repeatTimesIn,
        const uint16_t srcStrideIn, const uint16_t dstGapIn, const uint16_t dstfracGapIn, const uint8_t addrModeIn)
    {
        startIndex = startIndexIn;
        repeatTimes = repeatTimesIn;
        srcStride = srcStrideIn;
        dstGap = dstGapIn;
        dstFracGap = dstfracGapIn;
        addrMode = addrModeIn;
    }

    __aicore__ LoadData2dTransposeParamsV210(const uint16_t startIndexIn, const uint8_t repeatTimesIn,
        const uint16_t srcStrideIn, const uint16_t dstGapIn, const uint16_t dstfracGapIn)
    {
        startIndex = startIndexIn;
        repeatTimes = repeatTimesIn;
        srcStride = srcStrideIn;
        dstGap = dstGapIn;
        dstFracGap = dstfracGapIn;
    }

    __aicore__ inline void SetStartIndex(uint16_t startIndex_)
    {
        startIndex = startIndex_;
    }

    __aicore__ inline void SetRepeatTimes(uint8_t repeatTimes_)
    {
        repeatTimes = repeatTimes_;
    }

    __aicore__ inline void SetSrcStride(uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstGap(uint16_t dstGap_)
    {
        dstGap = dstGap_;
    }

    __aicore__ inline void SetDstFracGap(uint16_t dstFracGap_)
    {
        dstFracGap = dstFracGap_;
    }

    __aicore__ inline void SetAddrMode(uint8_t addrMode_)
    {
        addrMode = addrMode_;
    }

    __aicore__ inline void SetMStartPosition(uint32_t mStartPosition_)
    {
        (void)mStartPosition_;
    }

    __aicore__ inline void SetKStartPosition(uint32_t kStartPosition_)
    {
        (void)kStartPosition_;
    }

    __aicore__ inline void SetMStep(uint16_t mStep_)
    {
        (void)mStep_;
    }

    __aicore__ inline void SetKStep(uint16_t kStep_)
    {
        (void)kStep_;
    }

    __aicore__ inline void SetSrcStride(int32_t srcStride_)
    {
        (void)srcStride_;
    }

    __aicore__ inline void SetDstStride(uint16_t dstStride_)
    {
        (void)dstStride_;
    }

    __aicore__ inline void SetIfTranspose(bool ifTranspose_)
    {
        (void)ifTranspose_;
    }

    __aicore__ inline void SetSid(uint8_t sid_)
    {
        (void)sid_;
    }

    __aicore__ inline void SetQmode(uint8_t qmode_)
    {
        (void)qmode_;
    }

    uint16_t startIndex = 0;
    uint8_t repeatTimes = 0;
    uint16_t srcStride = 0;
    uint16_t dstGap = 0;
    uint16_t dstFracGap = 0;
    uint8_t addrMode = 0;
};

struct LoadData2dTransposeParamsV300 {
    __aicore__ LoadData2dTransposeParamsV300()
    {
        startIndex = 0;
        repeatTimes = 0;
        srcStride = 0;
        dstGap = 0;
        dstFracGap = 0;
        addrMode = 0;
    }

    __aicore__ LoadData2dTransposeParamsV300(const uint16_t startIndexIn, const uint8_t repeatTimesIn,
        const uint16_t srcStrideIn, const uint16_t dstGapIn, const uint16_t dstfracGapIn, const uint8_t addrModeIn)
    {
        startIndex = startIndexIn;
        repeatTimes = repeatTimesIn;
        srcStride = srcStrideIn;
        dstGap = dstGapIn;
        dstFracGap = dstfracGapIn;
        addrMode = addrModeIn;
    }

    __aicore__ LoadData2dTransposeParamsV300(const uint16_t startIndexIn, const uint8_t repeatTimesIn,
        const uint16_t srcStrideIn, const uint16_t dstGapIn, const uint16_t dstfracGapIn)
    {
        startIndex = startIndexIn;
        repeatTimes = repeatTimesIn;
        srcStride = srcStrideIn;
        dstGap = dstGapIn;
        dstFracGap = dstfracGapIn;
    }

    __aicore__ inline void SetStartIndex(uint16_t startIndex_)
    {
        startIndex = startIndex_;
    }

    __aicore__ inline void SetRepeatTimes(uint8_t repeatTimes_)
    {
        repeatTimes = repeatTimes_;
    }

    __aicore__ inline void SetSrcStride(uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstGap(uint16_t dstGap_)
    {
        dstGap = dstGap_;
    }

    __aicore__ inline void SetDstFracGap(uint16_t dstFracGap_)
    {
        dstFracGap = dstFracGap_;
    }

    __aicore__ inline void SetAddrMode(uint8_t addrMode_)
    {
        addrMode = addrMode_;
    }

    __aicore__ inline void SetMStartPosition(uint32_t mStartPosition_)
    {
        (void)mStartPosition_;
    }

    __aicore__ inline void SetKStartPosition(uint32_t kStartPosition_)
    {
        (void)kStartPosition_;
    }

    __aicore__ inline void SetMStep(uint16_t mStep_)
    {
        (void)mStep_;
    }

    __aicore__ inline void SetKStep(uint16_t kStep_)
    {
        (void)kStep_;
    }

    __aicore__ inline void SetSrcStride(int32_t srcStride_)
    {
        (void)srcStride_;
    }

    __aicore__ inline void SetDstStride(uint16_t dstStride_)
    {
        (void)dstStride_;
    }

    __aicore__ inline void SetIfTranspose(bool ifTranspose_)
    {
        (void)ifTranspose_;
    }

    __aicore__ inline void SetSid(uint8_t sid_)
    {
        (void)sid_;
    }

    __aicore__ inline void SetQmode(uint8_t qmode_)
    {
        (void)qmode_;
    }

    uint16_t startIndex = 0;
    uint8_t repeatTimes = 0;
    uint16_t srcStride = 0;
    uint16_t dstGap = 0;
    uint16_t dstFracGap = 0;
    uint8_t addrMode = 0;
};

struct LoadData2DParamsV311Gen {
    __aicore__ LoadData2DParamsV311Gen()
    {
        mStartPosition = 0;
        kStartPosition = 0;
        mStep = 0;
        kStep = 0;
        srcStride = 0;
        dstStride = 0;
        ifTranspose = false;
        sid = 0;
        qmode = 0;
    }

    __aicore__ LoadData2DParamsV311Gen(const uint32_t mStartPositionIn, const uint32_t kStartPositionIn,
        const uint16_t mStepIn, const uint16_t kStepIn, const uint32_t srcStrideIn, const uint16_t dstStrideIn,
        const bool ifTransposeIn, const uint8_t sidIn, const uint8_t qmodeIn)
    {
        mStartPosition = mStartPositionIn;
        kStartPosition = kStartPositionIn;
        mStep = mStepIn;
        kStep = kStepIn;
        srcStride = srcStrideIn;
        dstStride = dstStrideIn;
        ifTranspose = ifTransposeIn;
        sid = sidIn;
        qmode = qmodeIn;
    }

    __aicore__ inline void SetMStartPosition(uint32_t mStartPosition_)
    {
        mStartPosition = mStartPosition_;
    }

    __aicore__ inline void SetKStartPosition(uint32_t kStartPosition_)
    {
        kStartPosition = kStartPosition_;
    }

    __aicore__ inline void SetMStep(uint16_t mStep_)
    {
        mStep = mStep_;
    }

    __aicore__ inline void SetKStep(uint16_t kStep_)
    {
        kStep = kStep_;
    }

    __aicore__ inline void SetSrcStride(int32_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(uint16_t dstStride_)
    {
        dstStride = dstStride_;
    }

    __aicore__ inline void SetIfTranspose(bool ifTranspose_)
    {
        ifTranspose = ifTranspose_;
    }

    __aicore__ inline void SetSid(uint8_t sid_)
    {
        sid = sid_;
    }

    __aicore__ inline void SetQmode(uint8_t qmode_)
    {
        qmode = qmode_;
    }

    uint32_t mStartPosition = 0;
    uint32_t kStartPosition = 0;
    uint16_t mStep = 0;
    uint16_t kStep = 0;
    int32_t srcStride = 0;
    uint16_t dstStride = 0;
    bool ifTranspose = false;
    uint8_t sid = 0;
    uint8_t qmode = 0;
};

struct LoadData2dTransposeParamsV311Gen {
    __aicore__ LoadData2dTransposeParamsV311Gen()
    {
        mStartPosition = 0;
        kStartPosition = 0;
        mStep = 0;
        kStep = 0;
        srcStride = 0;
        dstStride = 0;
        ifTranspose = false;
        sid = 0;
        qmode = 0;
    }

    __aicore__ LoadData2dTransposeParamsV311Gen(const uint32_t mStartPositionIn, const uint32_t kStartPositionIn,
        const uint16_t mStepIn, const uint16_t kStepIn, const uint32_t srcStrideIn, const uint16_t dstStrideIn,
        const bool ifTransposeIn, const uint8_t sidIn, const uint8_t qmodeIn)
    {
        mStartPosition = mStartPositionIn;
        kStartPosition = kStartPositionIn;
        mStep = mStepIn;
        kStep = kStepIn;
        srcStride = srcStrideIn;
        dstStride = dstStrideIn;
        ifTranspose = ifTransposeIn;
        sid = sidIn;
        qmode = qmodeIn;
    }

    __aicore__ inline void SetMStartPosition(uint32_t mStartPosition_)
    {
        mStartPosition = mStartPosition_;
    }

    __aicore__ inline void SetKStartPosition(uint32_t kStartPosition_)
    {
        kStartPosition = kStartPosition_;
    }

    __aicore__ inline void SetMStep(uint16_t mStep_)
    {
        mStep = mStep_;
    }

    __aicore__ inline void SetKStep(uint16_t kStep_)
    {
        kStep = kStep_;
    }

    __aicore__ inline void SetSrcStride(int32_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(uint16_t dstStride_)
    {
        dstStride = dstStride_;
    }

    __aicore__ inline void SetIfTranspose(bool ifTranspose_)
    {
        ifTranspose = ifTranspose_;
    }

    __aicore__ inline void SetSid(uint8_t sid_)
    {
        sid = sid_;
    }

    __aicore__ inline void SetQmode(uint8_t qmode_)
    {
        qmode = qmode_;
    }

    __aicore__ inline void SetStartIndex(uint16_t startIndex_)
    {
        (void)startIndex_;
    }

    __aicore__ inline void SetRepeatTimes(uint8_t repeatTimes_)
    {
        (void)repeatTimes_;
    }

    __aicore__ inline void SetDstGap(uint16_t dstGap_)
    {
        (void)dstGap_;
    }

    __aicore__ inline void SetDstFracGap(uint16_t dstFracGap_)
    {
        (void)dstFracGap_;
    }

    __aicore__ inline void SetAddrMode(uint8_t addrMode_)
    {
        (void)addrMode_;
    }

    uint32_t mStartPosition = 0;
    uint32_t kStartPosition = 0;
    uint16_t mStep = 0;
    uint16_t kStep = 0;
    int32_t srcStride = 0;
    uint16_t dstStride = 0;
    bool ifTranspose = false;
    uint8_t sid = 0;
    uint8_t qmode = 0;
};


enum class QuantScheme : uint8_t {
    Perlayer = 0,
    PerChannel = 1
};

template<typename T>
struct LoadDataPaddingParam {
    __aicore__ LoadDataPaddingParam()
    {
        quantSch = QuantScheme::Perlayer;
        padValue = 0;
    }

    __aicore__ inline void SetQuantSch(QuantScheme quantSch_)
    {
        quantSch = quantSch_;
    }

    __aicore__ inline void SetPadValue(T padValue_)
    {
        padValue = padValue_;
    }
    QuantScheme quantSch;
    T padValue;
};
#endif // Kirin versions

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
struct LoadData2DMxParams {
    __aicore__ LoadData2DMxParams() {}

    __aicore__ LoadData2DMxParams(const uint16_t xStartPositionIn, const uint16_t yStartPositionIn,
        const uint8_t xStepIn, const uint8_t yStepIn, const uint16_t srcStrideIn, const uint16_t dstStrideIn)
    {
        xStartPosition = xStartPositionIn;
        yStartPosition = yStartPositionIn;
        xStep = xStepIn;
        yStep = yStepIn;
        srcStride = srcStrideIn;
        dstStride = dstStrideIn;
    }

    uint16_t xStartPosition = 0;
    uint16_t yStartPosition = 0;
    uint8_t xStep = 0;
    uint8_t yStep = 0;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
template <typename TYPE>
struct LoadData3DParamsV1 {
    using T = typename GetPadValueType<TYPE>::Type;
#else
template <typename T>
struct LoadData3DParamsV1 {
#endif
    __aicore__ LoadData3DParamsV1()
    {
        for (int32_t i = 0; i < PAD_SIZE; ++i) {
            padList[i] = 0;
        }
    }

    __aicore__ LoadData3DParamsV1(const uint8_t padListIn[PAD_SIZE], const uint16_t l1HIn, const uint16_t l1WIn,
        const uint16_t c1IndexIn, const uint8_t fetchFilterWIn, const uint8_t fetchFilterHIn, const int16_t leftTopWIn,
        const int16_t leftTopHIn, const uint8_t strideWIn, const uint8_t strideHIn, const uint8_t filterWIn,
        const uint8_t filterHIn, const uint8_t dilationFilterWIn, const uint8_t dilationFilterHIn,
        const uint8_t jumpStrideIn, const uint8_t repeatModeIn, const uint8_t repeatTimeIn, const uint8_t cSizeIn,
        const T padValueIn)
        : l1H(l1HIn),
          l1W(l1WIn),
          c1Index(c1IndexIn),
          fetchFilterW(fetchFilterWIn),
          fetchFilterH(fetchFilterHIn),
          leftTopW(leftTopWIn),
          leftTopH(leftTopHIn),
          strideW(strideWIn),
          strideH(strideHIn),
          filterW(filterWIn),
          filterH(filterHIn),
          dilationFilterW(dilationFilterWIn),
          dilationFilterH(dilationFilterHIn),
          jumpStride(jumpStrideIn),
          repeatMode(repeatModeIn),
          repeatTime(repeatTimeIn),
          cSize(cSizeIn),
          padValue(padValueIn)
    {
        for (int32_t i = 0; i < PAD_SIZE; ++i) {
            padList[i] = padListIn[i];
        }
    }

    uint8_t padList[PAD_SIZE] = {0};
    uint8_t strideW = 0;
    uint8_t strideH = 0;
    uint8_t filterW = 0;
    uint8_t filterH = 0;
    uint8_t dilationFilterW = 0;
    uint8_t dilationFilterH = 0;
    uint8_t jumpStride = 0;
    uint8_t repeatMode = 0;
    uint8_t repeatTime = 0;
    uint8_t cSize = 0;
    T padValue = 0;
    uint8_t fetchFilterW = 0;
    uint8_t fetchFilterH = 0;
    uint16_t l1H = 0;
    uint16_t l1W = 0;
    uint16_t c1Index = 0;
    int16_t leftTopW = 0;
    int16_t leftTopH = 0;
};

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
template <typename TYPE>
struct LoadData3DParamsV2 {
    using T = typename GetPadValueType<TYPE>::Type;
#else
template <typename T>
struct LoadData3DParamsV2 {
#endif
    __aicore__ LoadData3DParamsV2()
    {
        for (int32_t i = 0; i < PAD_SIZE; ++i) {
            padList[i] = 0;
        }
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
        enDualSrc = BM_DISABLE;
#endif
    }

    __aicore__ LoadData3DParamsV2(const uint8_t padListIn[PAD_SIZE], const uint16_t l1HIn, const uint16_t l1WIn,
        const uint16_t channelSizeIn, const uint16_t kExtensionIn, const uint16_t mExtensionIn,
        const uint16_t kStartPtIn, const uint16_t mStartPtIn, const uint8_t strideWIn, const uint8_t strideHIn,
        const uint8_t filterWIn, const uint8_t filterHIn, const uint8_t dilationFilterWIn,
        const uint8_t dilationFilterHIn, const bool enTransposeIn, const bool enSmallKIn, const T padValueIn)
        : l1H(l1HIn),
          l1W(l1WIn),
          channelSize(channelSizeIn),
          kExtension(kExtensionIn),
          mExtension(mExtensionIn),
          kStartPt(kStartPtIn),
          mStartPt(mStartPtIn),
          strideW(strideWIn),
          strideH(strideHIn),
          filterW(filterWIn),
          filterH(filterHIn),
          dilationFilterW(dilationFilterWIn),
          dilationFilterH(dilationFilterHIn),
          enTranspose(enTransposeIn),
          enSmallK(enSmallKIn),
          padValue(padValueIn)
    {
        for (int32_t i = 0; i < PAD_SIZE; ++i) {
            padList[i] = padListIn[i];
        }
    }

    __aicore__ LoadData3DParamsV2(const uint8_t padListIn[PAD_SIZE], const uint16_t l1HIn, const uint16_t l1WIn,
        const uint16_t channelSizeIn, const uint16_t kExtensionIn, const uint16_t mExtensionIn,
        const uint16_t kStartPtIn, const uint16_t mStartPtIn, const uint8_t strideWIn, const uint8_t strideHIn,
        const uint8_t filterWIn, const uint8_t filterHIn, const uint8_t dilationFilterWIn,
        const uint8_t dilationFilterHIn, const bool enTransposeIn, const bool enSmallKIn, const T padValueIn,
        const bool filterSizeWIn, const bool filterSizeHIn, const bool fMatrixCtrlIn)
        : l1H(l1HIn),
          l1W(l1WIn),
          channelSize(channelSizeIn),
          kExtension(kExtensionIn),
          mExtension(mExtensionIn),
          kStartPt(kStartPtIn),
          mStartPt(mStartPtIn),
          strideW(strideWIn),
          strideH(strideHIn),
          filterW(filterWIn),
          filterH(filterHIn),
          dilationFilterW(dilationFilterWIn),
          dilationFilterH(dilationFilterHIn),
          enTranspose(enTransposeIn),
          enSmallK(enSmallKIn),
          padValue(padValueIn),
          filterSizeW(filterSizeWIn),
          filterSizeH(filterSizeHIn),
          fMatrixCtrl(fMatrixCtrlIn)
    {
        for (int32_t i = 0; i < PAD_SIZE; ++i) {
            padList[i] = padListIn[i];
        }
    }

    uint8_t padList[PAD_SIZE] = {0};
    uint16_t l1H = 0;
    uint16_t l1W = 0;
    uint16_t channelSize = 0;
    uint16_t kExtension = 0;
    uint16_t mExtension = 0;
    uint16_t kStartPt = 0;
    uint16_t mStartPt = 0;

    uint8_t strideW = 1;
    uint8_t strideH = 1;
    uint8_t filterW = 1;
    uint8_t filterH = 1;
    uint8_t dilationFilterW = 1;
    uint8_t dilationFilterH = 1;
    bool enTranspose = false;
    bool enSmallK = false;
    T padValue = 0;
    bool filterSizeW = false;
    bool filterSizeH = false;
    bool fMatrixCtrl = false;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
    bm_t enDualSrc = BM_DISABLE;
#endif
};
struct LoadData3DParamsV2Pro {
    __aicore__ LoadData3DParamsV2Pro()
    {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
        enDualSrc = BM_DISABLE;
#endif
    }

    __aicore__ LoadData3DParamsV2Pro(const uint16_t channelSizeIn, const bool enTransposeIn, const bool enSmallKIn,
        const bool filterSizeWIn, const bool filterSizeHIn, const bool fMatrixCtrlIn, const uint64_t extConfigIn,
        const uint64_t filterConfigIn)
        : channelSize(channelSizeIn),
          enTranspose(enTransposeIn),
          enSmallK(enSmallKIn),
          filterSizeW(filterSizeWIn),
          filterSizeH(filterSizeHIn),
          fMatrixCtrl(fMatrixCtrlIn),
          extConfig(extConfigIn),
          filterConfig(filterConfigIn)
    {}

    uint16_t channelSize = 0;
    bool enTranspose = false;
    bool enSmallK = false;
    bool filterSizeW = false;
    bool filterSizeH = false;
    bool fMatrixCtrl = false;
    uint64_t extConfig = 0;
    uint64_t filterConfig = 0X10101010101;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
    bm_t enDualSrc = BM_DISABLE;
#endif
};

struct LoadData2dTransposeParamsV2 {
    __aicore__ LoadData2dTransposeParamsV2() {}

    __aicore__ LoadData2dTransposeParamsV2(const uint16_t startIndexIn, const uint8_t repeatTimesIn,
        const uint16_t srcStrideIn, const uint16_t dstGapIn, const uint16_t dstFracGapIn,
        const uint16_t srcFracGapIn)
        : startIndex(startIndexIn),
          repeatTimes(repeatTimesIn),
          srcStride(srcStrideIn),
          dstGap(dstGapIn),
          dstFracGap(dstFracGapIn),
          srcFracGap(srcFracGapIn)
    {}

    __aicore__ LoadData2dTransposeParamsV2(const uint16_t startIndexIn, const uint8_t repeatTimesIn,
        const uint16_t srcStrideIn, const uint16_t dstGapIn, const uint16_t dstFracGapIn,
        const uint16_t srcFracGapIn, const uint8_t addrModeIn)
        : startIndex(startIndexIn),
          repeatTimes(repeatTimesIn),
          srcStride(srcStrideIn),
          dstGap(dstGapIn),
          dstFracGap(dstFracGapIn),
          srcFracGap(srcFracGapIn),
          addrMode(addrModeIn)
    {}

    uint16_t startIndex = 0;
    uint8_t repeatTimes = 0;
    uint16_t srcStride = 0;
    uint16_t dstGap = 0;
    uint16_t dstFracGap = 0;
    uint16_t srcFracGap = 0;
    uint8_t addrMode = 0;
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2103)
struct MmadParamsV210 {
    __aicore__ MmadParamsV210()
    {
        m = 0;
        n = 0;
        k = 0;
        fmOffset = 0;
        smaskBufferAddr = 0;
        unitFlag = 0;
        hardwareSet = 0;
        enWinogradA = false;
        enWinogradB = false;
        isWeightOffset = false;
        enSsparse = false;
        cmatrixInitVal = true;
    }

    __aicore__ MmadParamsV210(const uint16_t mIn, const uint16_t nIn, const uint16_t kIn,
        const int32_t fmOffsetIn, const bool enSsparseIn, const bool enWinogradAIn, const bool enWinogradBIn)
    {
        m = mIn;
        n = nIn;
        k = kIn;
        fmOffset = fmOffsetIn;
        enSsparse = enSsparseIn;
        enWinogradA = enWinogradAIn;
        enWinogradB = enWinogradBIn;
    }

    __aicore__ MmadParamsV210(const uint16_t mIn, const uint16_t nIn, const uint16_t kIn, const uint8_t unitFlagIn,
        const bool cmatrixInitValIn)
    {
        m = mIn;
        n = nIn;
        k = kIn;
        unitFlag = unitFlagIn;
        cmatrixInitVal = cmatrixInitValIn;
    }

    __aicore__ inline void SetM(uint16_t m_)
    {
        m = m_;
    }

    __aicore__ inline void SetN(uint16_t n_)
    {
        n = n_;
    }

    __aicore__ inline void SetK(uint16_t k_)
    {
        k = k_;
    }

    __aicore__ inline void SetIsBias(bool isBias_)
    {
        (void)isBias_;
    }

    __aicore__ inline void SetFmOffset(uint8_t fmOffset_)
    {
        fmOffset = fmOffset_;
    }

    __aicore__ inline void SetSmaskBufferAddr(uint8_t smaskBufferAddr_)
    {
        smaskBufferAddr = smaskBufferAddr_;
    }

    __aicore__ inline void SetUnitFlag(uint8_t unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    __aicore__ inline void SetS16S8RightShift(bool s16s8rightShift_)
    {
        (void)s16s8rightShift_;
    }

    __aicore__ inline void SetS16S8SubDtype(bool s16s8subDtype_)
    {
        (void)s16s8subDtype_;
    }

    __aicore__ inline void SetHardwareSet(bool hardwareSet_)
    {
        hardwareSet = hardwareSet_;
    }

    __aicore__ inline void SetEnWinogradA(bool enWinogradA_)
    {
        enWinogradA = enWinogradA_;
    }

    __aicore__ inline void SetEnWinogradB(bool enWinogradB_)
    {
        enWinogradB = enWinogradB_;
    }

    __aicore__ inline void SetIsWeightOffset(bool isWeightOffset_)
    {
        isWeightOffset = isWeightOffset_;
    }

    __aicore__ inline void SetGemvCtrl(bool gemvCtrl_)
    {
        (void)gemvCtrl_;
    }

    __aicore__ inline void SetEnSsparse(bool enSsparse_)
    {
        enSsparse = enSsparse_;
    }

    __aicore__ inline void SetCmatrixSource(bool cmatrixSource_)
    {
        (void)cmatrixSource_;
    }

    __aicore__ inline void SetCmatrixInitVal(bool cmatrixInitVal_)
    {
        cmatrixInitVal = cmatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixInitVal(bool biasMatrixInitVal_)
    {
        (void)biasMatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixBroadcast(bool biasMatrixBroadcast_)
    {
        (void)biasMatrixBroadcast_;
    }

    __aicore__ inline void SetPreQuantMode(pre_quant_t preQuantMode_)
    {
        (void)preQuantMode_;
    }

    __aicore__ inline void SetPreReluMode(pre_relu_t preReluMode_)
    {
        (void)preReluMode_;
    }

    __aicore__ inline void SetPreClipReluMode(ClipReluMode_t preClipReluMode_)
    {
        (void)preClipReluMode_;
    }

    __aicore__ inline void SetPostQuantMode(post_quant_t postQuantMode_)
    {
        (void)postQuantMode_;
    }

    __aicore__ inline void SetEltwiseOp(eltwise_op_t eltwiseOp_)
    {
        (void)eltwiseOp_;
    }

    __aicore__ inline void SetEltwiseAntiqEnable(bool eltwiseAntiqEnable_)
    {
        (void)eltwiseAntiqEnable_;
    }

    __aicore__ inline void SetEltwiseBroadcastEnable(bool eltwiseBroadcastEnable_)
    {
        (void)eltwiseBroadcastEnable_;
    }

    __aicore__ inline void SetLsbMask(lsb_mask_t lsbMask_)
    {
        (void)lsbMask_;
    }

    __aicore__ inline void SetDependEnable(bool dependEnable_)
    {
        (void)dependEnable_;
    }

    __aicore__ inline void SetInstrId(instr_id_t instrId_)
    {
        (void)instrId_;
    }

    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    uint8_t fmOffset = 0;
    uint8_t smaskBufferAddr = 0;
    uint8_t unitFlag = 0;
    bool hardwareSet = false;
    bool enWinogradA = false;
    bool enWinogradB = false;
    bool isWeightOffset = false;
    bool enSsparse = false;
    bool cmatrixInitVal = true;
};
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003)
struct MmadParamsV300 {
    __aicore__ MmadParamsV300()
    {
        m = 0;
        n = 0;
        k = 0;
        fmOffset = 0;
        smaskBufferAddr = 0;
        unitFlag = 0;
        s16s8rightShift = true;
        s16s8subDtype = false;
        isWeightOffset = false;
        cmatrixSource = false;
        cmatrixInitVal = true;
    }

    __aicore__ MmadParamsV300(const uint16_t mIn, const uint16_t nIn, const uint16_t kIn, const uint8_t unitFlagIn,
        const bool cmatrixSourceIn, const bool cmatrixInitValIn)
    {
        m = mIn;
        n = nIn;
        k = kIn;
        unitFlag = unitFlagIn;
        cmatrixSource = cmatrixSourceIn;
        cmatrixInitVal = cmatrixInitValIn;
    }

    __aicore__ inline void SetM(uint16_t m_)
    {
        m = m_;
    }

    __aicore__ inline void SetN(uint16_t n_)
    {
        n = n_;
    }

    __aicore__ inline void SetK(uint16_t k_)
    {
        k = k_;
    }

    __aicore__ inline void SetIsBias(bool isBias_)
    {
        isBias = isBias_;
    }

    __aicore__ inline void SetEnSsparse(bool enSsparse_)
    {
        (void)enSsparse_;
    }

    __aicore__ inline void SetEnWinogradA(bool enWinogradA_)
    {
        (void)enWinogradA_;
    }

    __aicore__ inline void SetEnWinogradB(bool enWinogradB_)
    {
        (void)enWinogradB_;
    }

    __aicore__ inline void SetKDirectionAlign(bool kDirectionAlign_)
    {
        kDirectionAlign = kDirectionAlign_;
    }

    __aicore__ inline void SetFmOffset(int32_t fmOffset_)
    {
        fmOffset = fmOffset_;
    }

    __aicore__ inline void SetSmaskBufferAddr(uint8_t smaskBufferAddr_)
    {
        smaskBufferAddr = smaskBufferAddr_;
    }

    __aicore__ inline void SetUnitFlag(uint8_t unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    __aicore__ inline void SetS16S8RightShift(bool s16s8rightShift_)
    {
        s16s8rightShift = s16s8rightShift_;
    }

    __aicore__ inline void SetS16S8SubDtype(bool s16s8subDtype_)
    {
        s16s8subDtype = s16s8subDtype_;
    }

    __aicore__ inline void SetIsWeightOffset(bool isWeightOffset_)
    {
        isWeightOffset = isWeightOffset_;
    }

    __aicore__ inline void SetGemvCtrl(bool gemvCtrl_) {
    }

    __aicore__ inline void SetCmatrixSource(bool cmatrixSource_)
    {
        cmatrixSource = cmatrixSource_;
    }

    __aicore__ inline void SetCmatrixInitVal(bool cmatrixInitVal_)
    {
        cmatrixInitVal = cmatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixInitVal(bool biasMatrixInitVal_)
    {
        (void)biasMatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixBroadcast(bool biasMatrixBroadcast_)
    {
        (void)biasMatrixBroadcast_;
    }

    __aicore__ inline void SetPreQuantMode(pre_quant_t preQuantMode_)
    {
        (void)preQuantMode_;
    }

    __aicore__ inline void SetPreReluMode(pre_relu_t preReluMode_)
    {
        (void)preReluMode_;
    }

    __aicore__ inline void SetPostQuantMode(post_quant_t postQuantMode_)
    {
        (void)postQuantMode_;
    }

    __aicore__ inline void SetPreClipReluMode(ClipReluMode_t preClipReluMode_)
    {
        (void)preClipReluMode_;
    }

    __aicore__ inline void SetEltwiseOp(eltwise_op_t eltwiseOp_)
    {
        (void)eltwiseOp_;
    }

    __aicore__ inline void SetEltwiseAntiqEnable(bool eltwiseAntiqEnable_)
    {
        (void)eltwiseAntiqEnable_;
    }

    __aicore__ inline void SetEltwiseBroadcastEnable(bool eltwiseBroadcastEnable_)
    {
        (void)eltwiseBroadcastEnable_;
    }

    __aicore__ inline void SetLsbMask(lsb_mask_t lsbMask_)
    {
        (void)lsbMask_;
    }

    __aicore__ inline void SetDependEnable(bool dependEnable_)
    {
        (void)dependEnable_;
    }

    __aicore__ inline void SetInstrId(instr_id_t instrId_)
    {
        (void)instrId_;
    }

    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    int32_t fmOffset = 0;
    uint8_t smaskBufferAddr = 0;
    uint8_t unitFlag = 0;
    bool s16s8rightShift = true;
    bool s16s8subDtype = false;
    bool isWeightOffset = false;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;
    bool kDirectionAlign = false;
    bool isBias = false;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3103) && defined(__DAV_L310__))
struct MmadParamsV310Gen {
    __aicore__ MmadParamsV310Gen()
    {
        m = 0;
        n = 0;
        k = 0;
        fmOffset = 0;
        smaskBufferAddr = 0;
        unitFlag = 0;
        s16s8rightShift = true;
        s16s8subDtype = false;
        isWeightOffset = false;
        gemvCtrl = false;
        cmatrixSource = false;
        cmatrixInitVal = true;
    }

    __aicore__ MmadParamsV310Gen(const uint16_t mIn, const uint16_t nIn, const uint16_t kIn, const uint8_t unitFlagIn,
        const bool cmatrixSourceIn, const bool cmatrixInitValIn)
    {
        m = mIn;
        n = nIn;
        k = kIn;
        unitFlag = unitFlagIn;
        cmatrixSource = cmatrixSourceIn;
        cmatrixInitVal = cmatrixInitValIn;
    }

    __aicore__ inline void SetM(uint16_t m_)
    {
        m = m_;
    }

    __aicore__ inline void SetN(uint16_t n_)
    {
        n = n_;
    }

    __aicore__ inline void SetK(uint16_t k_)
    {
        k = k_;
    }

    __aicore__ inline void SetIsBias(bool isBias_)
    {
        (void)isBias_;
    }

    __aicore__ inline void SetEnSsparse(bool enSsparse_)
    {
        (void)enSsparse_;
    }

    __aicore__ inline void SetEnWinogradA(bool enWinogradA_)
    {
        (void)enWinogradA_;
    }

    __aicore__ inline void SetEnWinogradB(bool enWinogradB_)
    {
        (void)enWinogradB_;
    }

    __aicore__ inline void SetKDirectionAlign(bool kDirectionAlign_)
    {
        (void)kDirectionAlign_;
    }

    __aicore__ inline void SetFmOffset(int32_t fmOffset_)
    {
        fmOffset = fmOffset_;
    }

    __aicore__ inline void SetSmaskBufferAddr(uint8_t smaskBufferAddr_)
    {
        smaskBufferAddr = smaskBufferAddr_;
    }

    __aicore__ inline void SetUnitFlag(uint8_t unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    __aicore__ inline void SetS16S8RightShift(bool s16s8rightShift_)
    {
        s16s8rightShift = s16s8rightShift_;
    }

    __aicore__ inline void SetS16S8SubDtype(bool s16s8subDtype_)
    {
        s16s8subDtype = s16s8subDtype_;
    }

    __aicore__ inline void SetIsWeightOffset(bool isWeightOffset_)
    {
        isWeightOffset = isWeightOffset_;
    }

    __aicore__ inline void SetGemvCtrl(bool gemvCtrl_)
    {
        gemvCtrl = gemvCtrl_;
    }

    __aicore__ inline void SetCmatrixSource(bool cmatrixSource_)
    {
        cmatrixSource = cmatrixSource_;
    }

    __aicore__ inline void SetCmatrixInitVal(bool cmatrixInitVal_)
    {
        cmatrixInitVal = cmatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixInitVal(bool biasMatrixInitVal_)
    {
        (void)biasMatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixBroadcast(bool biasMatrixBroadcast_)
    {
        (void)biasMatrixBroadcast_;
    }

    __aicore__ inline void SetPreQuantMode(pre_quant_t preQuantMode_)
    {
        (void)preQuantMode_;
    }

    __aicore__ inline void SetPreReluMode(pre_relu_t preReluMode_)
    {
        (void)preReluMode_;
    }

    __aicore__ inline void SetPostQuantMode(post_quant_t postQuantMode_)
    {
        (void)postQuantMode_;
    }

    __aicore__ inline void SetPreClipReluMode(ClipReluMode_t preClipReluMode_)
    {
        (void)preClipReluMode_;
    }

    __aicore__ inline void SetEltwiseOp(eltwise_op_t eltwiseOp_)
    {
        (void)eltwiseOp_;
    }

    __aicore__ inline void SetEltwiseAntiqEnable(bool eltwiseAntiqEnable_)
    {
        (void)eltwiseAntiqEnable_;
    }

    __aicore__ inline void SetEltwiseBroadcastEnable(bool eltwiseBroadcastEnable_)
    {
        (void)eltwiseBroadcastEnable_;
    }

    __aicore__ inline void SetLsbMask(lsb_mask_t lsbMask_)
    {
        (void)lsbMask_;
    }

    __aicore__ inline void SetDependEnable(bool dependEnable_)
    {
        (void)dependEnable_;
    }

    __aicore__ inline void SetInstrId(instr_id_t instrId_)
    {
        (void)instrId_;
    }

    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    int32_t fmOffset = 0;
    uint8_t smaskBufferAddr = 0;
    uint8_t unitFlag = 0;
    bool s16s8rightShift = true;
    bool s16s8subDtype = false;
    bool isWeightOffset = false;
    bool gemvCtrl = false;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3113) && defined(__DAV_L311__))
struct MmadParamsV311Gen {
    __aicore__ inline MmadParamsV311Gen()
    {
        m = 0;
        n = 0;
        k = 0;
        fmOffset = 0;
        smaskBufferAddr = 0;
        unitFlag = 0;
        s16s8rightShift = true;
        s16s8subDtype = false;
        isWeightOffset = false;
        gemvCtrl = false;
        cmatrixSource = false;
        cmatrixInitVal = true;
    }

    __aicore__ inline MmadParamsV311Gen(const uint16_t mIn, const uint16_t nIn, const uint16_t kIn, const uint8_t unitFlagIn,
        const bool cmatrixSourceIn, const bool cmatrixInitValIn)
    {
        m = mIn;
        n = nIn;
        k = kIn;
        unitFlag = unitFlagIn;
        cmatrixSource = cmatrixSourceIn;
        cmatrixInitVal = cmatrixInitValIn;
    }

    __aicore__ inline void SetM(uint16_t m_)
    {
        m = m_;
    }

    __aicore__ inline void SetN(uint16_t n_)
    {
        n = n_;
    }

    __aicore__ inline void SetK(uint16_t k_)
    {
        k = k_;
    }

    __aicore__ inline void SetIsBias(bool isBias_)
    {
        isBias = isBias_;
    }

    __aicore__ inline void SetEnSsparse(bool enSsparse_)
    {
        (void)enSsparse_;
    }

    __aicore__ inline void SetEnWinogradA(bool enWinogradA_)
    {
        (void)enWinogradA_;
    }

    __aicore__ inline void SetEnWinogradB(bool enWinogradB_)
    {
        (void)enWinogradB_;
    }

    __aicore__ inline void SetKDirectionAlign(bool kDirectionAlign_)
    {
        kDirectionAlign = kDirectionAlign_;
    }

    __aicore__ inline void SetFmOffset(int32_t fmOffset_)
    {
        fmOffset = fmOffset_;
    }

    __aicore__ inline void SetSmaskBufferAddr(uint8_t smaskBufferAddr_)
    {
        smaskBufferAddr = smaskBufferAddr_;
    }

    __aicore__ inline void SetUnitFlag(uint8_t unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    __aicore__ inline void SetS16S8RightShift(bool s16s8rightShift_)
    {
        s16s8rightShift = s16s8rightShift_;
    }

    __aicore__ inline void SetS16S8SubDtype(bool s16s8subDtype_)
    {
        s16s8subDtype = s16s8subDtype_;
    }

    __aicore__ inline void SetIsWeightOffset(bool isWeightOffset_)
    {
        isWeightOffset = isWeightOffset_;
    }

    __aicore__ inline void SetGemvCtrl(bool gemvCtrl_)
    {
        gemvCtrl = gemvCtrl_;
    }

    __aicore__ inline void SetCmatrixSource(bool cmatrixSource_)
    {
        cmatrixSource = cmatrixSource_;
    }

    __aicore__ inline void SetCmatrixInitVal(bool cmatrixInitVal_)
    {
        cmatrixInitVal = cmatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixInitVal(bool biasMatrixInitVal_)
    {
        (void)biasMatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixBroadcast(bool biasMatrixBroadcast_)
    {
        (void)biasMatrixBroadcast_;
    }

    __aicore__ inline void SetPreQuantMode(pre_quant_t preQuantMode_)
    {
        (void)preQuantMode_;
    }

    __aicore__ inline void SetPreReluMode(pre_relu_t preReluMode_)
    {
        (void)preReluMode_;
    }

    __aicore__ inline void SetPostQuantMode(post_quant_t postQuantMode_)
    {
        (void)postQuantMode_;
    }

    __aicore__ inline void SetPreClipReluMode(ClipReluMode_t preClipReluMode_)
    {
        (void)preClipReluMode_;
    }

    __aicore__ inline void SetEltwiseOp(eltwise_op_t eltwiseOp_)
    {
        (void)eltwiseOp_;
    }

    __aicore__ inline void SetEltwiseAntiqEnable(bool eltwiseAntiqEnable_)
    {
        (void)eltwiseAntiqEnable_;
    }

    __aicore__ inline void SetEltwiseBroadcastEnable(bool eltwiseBroadcastEnable_)
    {
        (void)eltwiseBroadcastEnable_;
    }

    __aicore__ inline void SetLsbMask(lsb_mask_t lsbMask_)
    {
        (void)lsbMask_;
    }

    __aicore__ inline void SetDependEnable(bool dependEnable_)
    {
        (void)dependEnable_;
    }

    __aicore__ inline void SetInstrId(instr_id_t instrId_)
    {
        (void)instrId_;
    }

    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    int32_t fmOffset = 0;
    uint8_t smaskBufferAddr = 0;
    uint8_t unitFlag = 0;
    bool s16s8rightShift = true;
    bool s16s8subDtype = false;
    bool isWeightOffset = false;
    bool gemvCtrl = false;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;
    bool kDirectionAlign = false;
    bool isBias = false;
};
#endif

struct MmadParams {
    __aicore__ inline MmadParams() {}

    __aicore__ inline MmadParams(const uint16_t mIn, const uint16_t nIn, const uint16_t kIn, const bool isBiasIn,
        const int32_t fmOffsetIn, const bool enSsparseIn, const bool enWinogradAIn, const bool enWinogradBIn)
        : m(mIn),
          n(nIn),
          k(kIn),
          isBias(isBiasIn),
          fmOffset(fmOffsetIn),
          enSsparse(enSsparseIn),
          enWinogradA(enWinogradAIn),
          enWinogradB(enWinogradBIn)
    {}

    __aicore__ inline MmadParams(const uint16_t mIn, const uint16_t nIn, const uint16_t kIn, const uint8_t unitFlagIn,
        const bool cmatrixSourceIn, const bool cmatrixInitValIn)
        : m(mIn),
          n(nIn),
          k(kIn),
          unitFlag(unitFlagIn),
          cmatrixSource(cmatrixSourceIn),
          cmatrixInitVal(cmatrixInitValIn)
    {}

    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    // Indicates whether to accumulate the initial matrix, 0: matrix multiplication, 1: matrix multiplication and
    // addition
    bool isBias = false;
    // Left matrix offset
    int32_t fmOffset = 0;
    // Enable the structured sparse feature, default value is false
    bool enSsparse = false;
    // Indicates whether matrix a is generated by winograd_feature_map_transform, default value is false;
    bool enWinogradA = false;
    // Indicates whether matrix b is generated by winograd_feature_map_transform, default value is false;
    bool enWinogradB = false;
    uint8_t unitFlag = 0;
    // also mean gemvCtrl in 3510 and 5102
    bool kDirectionAlign = false;
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    bool cmatrixSource = false;
    // Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    bool cmatrixInitVal = true;
    bool disableGemv = false;
};

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
struct ConvFixParams {
    __aicore__ ConvFixParams()
    {
        n = 0;
        biasMatrixInitVal = true;
        biasMatrixBroadcast = false;
        preQuantMode = QuantMode_t::NoQuant;
        preReluMode = ReluMode_t::NoRelu;
        postQuantMode = QuantMode_post::NoConv;
        actPostEnable = false;
        preClipReluMode = false;
        eltwiseOp = eltwise_op_t::No_Eltwise;
        eltwiseAntiqEnable = false;
        eltwiseBroadcastEnable = false;
        lsbMask = lsb_mask_t::Disable;
        gemvCtrl = false;
        dependEnable = false;
        postProcessBandwithCtrl = false;
        instrId = instr_id_t::ID_0;
    }

    __aicore__ inline void SetM(uint16_t m_)
    {
        (void)m_;
    }

    __aicore__ inline void SetN(uint16_t n_)
    {
        n = n_;
    }

    __aicore__ inline void SetK(uint16_t k_)
    {
        (void)k_;
    }

    __aicore__ inline void SetIsBias(bool isBias_)
    {
        (void)isBias_;
    }

    __aicore__ inline void SetEnSsparse(bool enSsparse_)
    {
        (void)enSsparse_;
    }

    __aicore__ inline void SetEnWinogradA(bool enWinogradA_)
    {
        (void)enWinogradA_;
    }

    __aicore__ inline void SetEnWinogradB(bool enWinogradB_)
    {
        (void)enWinogradB_;
    }

    __aicore__ inline void SetKDirectionAlign(bool kDirectionAlign_)
    {
        (void)kDirectionAlign_;
    }

    __aicore__ inline void SetFmOffset(int32_t fmOffset_)
    {
        (void)fmOffset_;
    }

    __aicore__ inline void SetSmaskBufferAddr(uint8_t smaskBufferAddr_)
    {
        (void)smaskBufferAddr_;
    }

    __aicore__ inline void SetUnitFlag(uint8_t unitFlag_)
    {
        (void)unitFlag_;
    }

    __aicore__ inline void SetS16S8RightShift(bool s16s8rightShift_)
    {
        (void)s16s8rightShift_;
    }

    __aicore__ inline void SetS16S8SubDtype(bool s16s8subDtype_)
    {
        (void)s16s8subDtype_;
    }

    __aicore__ inline void SetIsWeightOffset(bool isWeightOffset_)
    {
        (void)isWeightOffset_;
    }

    __aicore__ inline void SetGemvCtrl(bool gemvCtrl_)
    {
        gemvCtrl = gemvCtrl_;
    }

    __aicore__ inline void SetCmatrixSource(bool cmatrixSource_)
    {
        (void)cmatrixSource_;
    }

    __aicore__ inline void SetCmatrixInitVal(bool cmatrixInitVal_)
    {
        (void)cmatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixInitVal(bool biasMatrixInitVal_)
    {
        biasMatrixInitVal = biasMatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixBroadcast(bool biasMatrixBroadcast_)
    {
        biasMatrixBroadcast = biasMatrixBroadcast_;
    }

    __aicore__ inline void SetPreQuantMode(QuantMode_t preQuantMode_)
    {
        preQuantMode = preQuantMode_;
    }

    __aicore__ inline void SetPreReluMode(ReluMode_t preReluMode_)
    {
        preReluMode = preReluMode_;
    }

    __aicore__ inline void SetPostQuantMode(QuantMode_post postQuantMode_)
    {
        postQuantMode = postQuantMode_;
    }

    __aicore__ inline void SetActPostEnable(bool actPostEnable_) // new
    {
        actPostEnable = actPostEnable_;
    }

    __aicore__ inline void SetPreClipReluMode(bool preClipReluMode_)
    {
        preClipReluMode = preClipReluMode_;
    }

    __aicore__ inline void SetEltwiseOp(eltwise_op_t eltwiseOp_)
    {
        eltwiseOp = eltwiseOp_;
    }

    __aicore__ inline void SetEltwiseAntiqEnable(bool eltwiseAntiqEnable_)
    {
        eltwiseAntiqEnable = eltwiseAntiqEnable_;
    }

    __aicore__ inline void SetEltwiseBroadcastEnable(bool eltwiseBroadcastEnable_)
    {
        eltwiseBroadcastEnable = eltwiseBroadcastEnable_;
    }

    __aicore__ inline void SetLsbMask(lsb_mask_t lsbMask_)
    {
        lsbMask = lsbMask_;
    }

    __aicore__ inline void SetDependEnable(bool dependEnable_)
    {
        dependEnable = dependEnable_;
    }

    __aicore__ inline void SetPostProcessBandwithCtrl (bool postProcessBandwithCtrl_)
    {
        postProcessBandwithCtrl = postProcessBandwithCtrl_;
    }

    __aicore__ inline void SetInstrId(instr_id_t instrId_)
    {
        instrId = instrId_;
    }

    uint16_t n = 0;
    bool biasMatrixInitVal = true;
    bool biasMatrixBroadcast = false;
    QuantMode_t preQuantMode = QuantMode_t::NoQuant;
    ReluMode_t preReluMode = ReluMode_t::NoRelu;
    QuantMode_post postQuantMode = QuantMode_post::NoConv;
    bool actPostEnable = false;
    bool preClipReluMode = false;
    eltwise_op_t eltwiseOp = eltwise_op_t::No_Eltwise;
    bool eltwiseAntiqEnable = false;
    bool eltwiseBroadcastEnable = false;
    lsb_mask_t lsbMask = lsb_mask_t::Disable;
    bool gemvCtrl = false;
    bool dependEnable = false;
    bool postProcessBandwithCtrl = false;
    instr_id_t instrId = instr_id_t::ID_0;
};

#endif // Kirin versions

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
struct MmadFixParams {
    __aicore__ MmadFixParams()
    {
        n = 0;
        biasMatrixInitVal = true;
        biasMatrixBroadcast = false;
        preQuantMode = QuantMode_t::NoQuant;
        preReluMode = ReluMode_t::NoRelu;
        postQuantMode = QuantMode_post::NoConv;
        preClipReluMode = false;
        eltwiseOp = eltwise_op_t::No_Eltwise;
        eltwiseAntiqEnable = false;
        eltwiseBroadcastEnable = false;
        lsbMask = lsb_mask_t::Disable;
        gemvCtrl = false;
        dependEnable = false;
        instrId = instr_id_t::ID_0;
    }

    __aicore__ inline void SetM(uint16_t m_)
    {
        (void)m_;
    }

    __aicore__ inline void SetN(uint16_t n_)
    {
        n = n_;
    }

    __aicore__ inline void SetK(uint16_t k_)
    {
        (void)k_;
    }

    __aicore__ inline void SetIsBias(bool isBias_)
    {
        (void)isBias_;
    }

    __aicore__ inline void SetEnSsparse(bool enSsparse_)
    {
        (void)enSsparse_;
    }

    __aicore__ inline void SetEnWinogradA(bool enWinogradA_)
    {
        (void)enWinogradA_;
    }

    __aicore__ inline void SetEnWinogradB(bool enWinogradB_)
    {
        (void)enWinogradB_;
    }

    __aicore__ inline void SetKDirectionAlign(bool kDirectionAlign_)
    {
        (void)kDirectionAlign_;
    }

    __aicore__ inline void SetFmOffset(int32_t fmOffset_)
    {
        (void)fmOffset_;
    }

    __aicore__ inline void SetSmaskBufferAddr(uint8_t smaskBufferAddr_)
    {
        (void)smaskBufferAddr_;
    }

    __aicore__ inline void SetUnitFlag(uint8_t unitFlag_)
    {
        (void)unitFlag_;
    }

    __aicore__ inline void SetS16S8RightShift(bool s16s8rightShift_)
    {
        (void)s16s8rightShift_;
    }

    __aicore__ inline void SetS16S8SubDtype(bool s16s8subDtype_)
    {
        (void)s16s8subDtype_;
    }

    __aicore__ inline void SetIsWeightOffset(bool isWeightOffset_)
    {
        (void)isWeightOffset_;
    }

    __aicore__ inline void SetGemvCtrl(bool gemvCtrl_)
    {
        gemvCtrl = gemvCtrl_;
    }

    __aicore__ inline void SetCmatrixSource(bool cmatrixSource_)
    {
        (void)cmatrixSource_;
    }

    __aicore__ inline void SetCmatrixInitVal(bool cmatrixInitVal_)
    {
        (void)cmatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixInitVal(bool biasMatrixInitVal_)
    {
        biasMatrixInitVal = biasMatrixInitVal_;
    }

    __aicore__ inline void SetBiasMatrixBroadcast(bool biasMatrixBroadcast_)
    {
        biasMatrixBroadcast = biasMatrixBroadcast_;
    }

    __aicore__ inline void SetPreQuantMode(QuantMode_t preQuantMode_)
    {
        preQuantMode = preQuantMode_;
    }

    __aicore__ inline void SetPreReluMode(ReluMode_t preReluMode_)
    {
        preReluMode = preReluMode_;
    }

    __aicore__ inline void SetPostQuantMode(QuantMode_post postQuantMode_)
    {
        postQuantMode = postQuantMode_;
    }

    __aicore__ inline void SetPreClipReluMode(bool preClipReluMode_)
    {
        preClipReluMode = preClipReluMode_;
    }

    __aicore__ inline void SetEltwiseOp(eltwise_op_t eltwiseOp_)
    {
        eltwiseOp = eltwiseOp_;
    }

    __aicore__ inline void SetEltwiseAntiqEnable(bool eltwiseAntiqEnable_)
    {
        eltwiseAntiqEnable = eltwiseAntiqEnable_;
    }

    __aicore__ inline void SetEltwiseBroadcastEnable(bool eltwiseBroadcastEnable_)
    {
        eltwiseBroadcastEnable = eltwiseBroadcastEnable_;
    }

    __aicore__ inline void SetLsbMask(lsb_mask_t lsbMask_)
    {
        lsbMask = lsbMask_;
    }

    __aicore__ inline void SetDependEnable(bool dependEnable_)
    {
        dependEnable = dependEnable_;
    }

    __aicore__ inline void SetInstrId(instr_id_t instrId_)
    {
        instrId = instrId_;
    }

    uint16_t n = 0;
    bool biasMatrixInitVal = true;
    bool biasMatrixBroadcast = false;
    QuantMode_t preQuantMode = QuantMode_t::NoQuant;
    ReluMode_t preReluMode = ReluMode_t::NoRelu;
    QuantMode_post postQuantMode = QuantMode_post::NoConv;
    bool preClipReluMode = false;
    eltwise_op_t eltwiseOp = eltwise_op_t::No_Eltwise;
    bool eltwiseAntiqEnable = false;
    bool eltwiseBroadcastEnable = false;
    lsb_mask_t lsbMask = lsb_mask_t::Disable;
    bool gemvCtrl = false;
    bool dependEnable = false;
    instr_id_t instrId = instr_id_t::ID_0;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))

struct MatrixParams {
    __aicore__ MatrixParams()
    {
        m = 0;
        k = 0;
    }

    __aicore__ MatrixParams(uint16_t mIn, uint16_t nIn)
    {
        m = mIn;
        k = nIn;
    }

    __aicore__ inline void SetM(uint16_t m_)
    {
        m = m_;
    }

    __aicore__ inline void SetK(uint16_t k_)
    {
        k = k_;
    }

    uint16_t m = 0;
    uint16_t k = 0;
};
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2103)
template <typename T>
struct InitConstValueParams {
    __aicore__ InitConstValueParams()
    {
        repeatTimes = 0;
        initValue = 0;
    }

    __aicore__ InitConstValueParams(const uint8_t repeatTimesIn, const T initValueIn)
    {
        repeatTimes = repeatTimesIn;
        initValue = initValueIn;
    }

    __aicore__ inline void SetRepeatTimes(uint8_t repeatTimes_)
    {
        repeatTimes = repeatTimes_;
    }

    __aicore__ inline void SetBlockNum(uint16_t blockNum_)
    {
        (void)blockNum_;
    }

    __aicore__ inline void SetDstGap(uint16_t dstGap_)
    {
        (void)dstGap_;
    }

    __aicore__ inline void SetInitValue(T initValue_)
    {
        initValue = initValue_;
    }

    uint8_t repeatTimes = 0;
    T initValue = 0;
};
#elif defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || (__NPU_ARCH__ == 3113))
template <typename T>
struct InitConstValueParams {
    __aicore__ InitConstValueParams() {}

    __aicore__ InitConstValueParams(const uint16_t repeatTimesIn,
        const uint16_t blockNumIn, const uint16_t dstGapIn, const T initValueIn)
        : repeatTimes(repeatTimesIn),
          blockNum(blockNumIn),
          dstGap(dstGapIn),
          initValue(initValueIn)
    {}

    __aicore__ InitConstValueParams(const uint16_t repeatTimesIn, const T initValueIn)
        : repeatTimes(repeatTimesIn),
          initValue(initValueIn)
    {}
    __aicore__ inline void SetRepeatTimes(uint16_t repeatTimes_)
    {
        repeatTimes = repeatTimes_;
    }

    __aicore__ inline void SetBlockNum(uint16_t blockNum_)
    {
        blockNum = blockNum_;
    }

    __aicore__ inline void SetDstGap(uint16_t dstGap_)
    {
        dstGap = dstGap_;
    }

    __aicore__ inline void SetInitValue(T initValue_)
    {
        initValue = initValue_;
    }

    uint16_t repeatTimes = 0;
    uint16_t blockNum = 0;
    uint16_t dstGap = 0;
    T initValue = 0;
};
#else
template <typename T>
struct InitConstValueParams {
    __aicore__ InitConstValueParams() {}

    __aicore__ InitConstValueParams(const uint16_t repeatTimesIn,
        const uint16_t blockNumIn, const uint16_t dstGapIn, const T initValueIn)
        : repeatTimes(repeatTimesIn),
          blockNum(blockNumIn),
          dstGap(dstGapIn),
          initValue(initValueIn)
    {}

    __aicore__ InitConstValueParams(const uint16_t repeatTimesIn, const T initValueIn)
        : repeatTimes(repeatTimesIn),
          initValue(initValueIn)
    {}

    uint16_t repeatTimes = 0;
    uint16_t blockNum = 0;
    uint16_t dstGap = 0;
    T initValue = 0;
};
#endif

enum class FmatrixMode : uint8_t {
    FMATRIX_LEFT = 0,
    FMATRIX_RIGHT = 1,
};

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103))
struct LoadDataRepeatParam {
    __aicore__ LoadDataRepeatParam() {}

    __aicore__ LoadDataRepeatParam(const uint16_t repeatStrideIn, const uint8_t repeatTimeIn,
        const uint8_t repeatModeIn)
        : repeatStride(repeatStrideIn),
          repeatTime(repeatTimeIn),
          repeatMode(repeatModeIn)
    {}
    __aicore__ inline void SetRepeatStride(uint16_t repeatStride_)
    {
        repeatStride = repeatStride_;
    }

    __aicore__ inline void SetRepeatTime(uint8_t repeatTime_)
    {
        repeatTime = repeatTime_;
    }

    __aicore__ inline void SetRepeatMode(uint8_t repeatMode_)
    {
        repeatMode = repeatMode_;
    }

    __aicore__ inline void SetDstStride(uint16_t dstStride_)
    {
        (void)dstStride_;
    }

    __aicore__ inline void SetDstMposition(uint16_t dstMposition_)
    {
        (void)dstMposition_;
    }

    uint16_t repeatStride = 0;
    uint8_t repeatTime = 1;
    uint8_t repeatMode = 0;
    uint8_t reserved = 0;
};
#endif
#else
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
struct LoadDataRepeatParam {
    __aicore__ LoadDataRepeatParam() {}

    __aicore__ LoadDataRepeatParam(const uint16_t repeatStrideIn, const uint8_t repeatTimeIn,
        const uint8_t repeatModeIn,  const uint16_t dstStrideIn)
        : repeatStride(repeatStrideIn),
          repeatTime(repeatTimeIn),
          repeatMode(repeatModeIn),
          dstStride(dstStrideIn)
    {}

    uint16_t repeatStride = 0;
    uint8_t repeatTime = 1;
    uint8_t repeatMode = 0;
    uint16_t dstStride = 0;
};
#else
struct LoadDataRepeatParam {
    __aicore__ LoadDataRepeatParam() {}

    __aicore__ LoadDataRepeatParam(const uint16_t repeatStrideIn, const uint8_t repeatTimeIn,
        const uint8_t repeatModeIn)
        : repeatStride(repeatStrideIn),
          repeatTime(repeatTimeIn),
          repeatMode(repeatModeIn)
    {}

    uint16_t repeatStride = 0;
    uint8_t repeatTime = 1;
    uint8_t repeatMode = 0;
    uint8_t reserved = 0;
};
#endif // Turing versions
#endif // Kirin versions

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113)
struct LoadDataRepeatParamV311Gen {
    __aicore__ LoadDataRepeatParamV311Gen()
    {
        repeatStride = 0;
        repeatTime = 1;
        repeatMode = 0;
        reserved = 0;
        dstStride = 0;
        dstMposition = 0;
    }

    __aicore__ LoadDataRepeatParamV311Gen(const uint16_t repeatStrideIn, const uint8_t repeatTimeIn,
        const uint8_t repeatModeIn, const uint16_t dstStrideIn, const uint16_t dstMpositionIn)
    {
        repeatStride = repeatStrideIn;
        repeatTime = repeatTimeIn;
        repeatMode = repeatModeIn;
        dstStride = dstStrideIn;
        dstMposition = dstMpositionIn;
    }

    __aicore__ inline void SetRepeatStride(uint16_t repeatStride_)
    {
        repeatStride = repeatStride_;
    }

    __aicore__ inline void SetRepeatTime(uint8_t repeatTime_)
    {
        repeatTime = repeatTime_;
    }

    __aicore__ inline void SetRepeatMode(uint8_t repeatMode_)
    {
        repeatMode = repeatMode_;
    }

    __aicore__ inline void SetDstStride(uint16_t dstStride_)
    {
        dstStride = dstStride_;
    }

    __aicore__ inline void SetDstMposition(uint16_t dstMposition_)
    {
        dstMposition = dstMposition_;
    }

    uint16_t repeatStride = 0;
    uint8_t repeatTime = 1;
    uint8_t repeatMode = 0;
    uint8_t reserved = 0;
    uint16_t dstStride = 0;
    uint16_t dstMposition = 0;
};

#if ((__NPU_ARCH__ == 3113))
using LoadDataRepeatParam = struct LoadDataRepeatParamV311Gen;
#endif

#endif
struct LoadImageToLocalParams {
    __aicore__ LoadImageToLocalParams() {}

    __aicore__ LoadImageToLocalParams(const uint16_t horizSizeIn, const uint16_t vertSizeIn,
        const uint16_t horizStartPosIn, const uint16_t vertStartPosIn, const uint16_t srcHorizSizeIn,
        const uint8_t topPadSizeIn, const uint8_t botPadSizeIn, const uint16_t leftPadSizeIn,
        const uint16_t rightPadSizeIn)
        : horizSize(horizSizeIn),
          vertSize(vertSizeIn),
          horizStartPos(horizStartPosIn),
          vertStartPos(vertStartPosIn),
          srcHorizSize(srcHorizSizeIn),
          topPadSize(topPadSizeIn),
          botPadSize(botPadSizeIn),
          leftPadSize(leftPadSizeIn),
          rightPadSize(rightPadSizeIn)
    {}

    uint16_t horizSize = 0;
    uint16_t vertSize = 0;
    uint16_t horizStartPos = 0;
    uint16_t vertStartPos = 0;
    uint16_t srcHorizSize = 0;
    uint8_t topPadSize = 0;
    uint8_t botPadSize = 0;
    uint16_t leftPadSize = 0;
    uint16_t rightPadSize = 0;
    uint8_t sid = 0;
};

struct CheckLocalMemoryIAParam {
    __aicore__ CheckLocalMemoryIAParam() {}

    __aicore__ CheckLocalMemoryIAParam(const uint8_t enableBitIn, const uint32_t startAddrIn, const uint32_t endAddrIn,
        const bool isScalarReadIn, const bool isScalarWriteIn, const bool isVectorReadIn, const bool isVectorWriteIn,
        const bool isMteReadIn, const bool isMteWriteIn, const bool isEnableIn)
        : enableBit(enableBitIn),
          startAddr(startAddrIn),
          endAddr(endAddrIn),
          isScalarRead(isScalarReadIn),
          isScalarWrite(isScalarWriteIn),
          isVectorRead(isVectorReadIn),
          isVectorWrite(isVectorWriteIn),
          isMteRead(isMteReadIn),
          isMteWrite(isMteWriteIn),
          isEnable(isEnableIn)
    {}

    uint8_t enableBit = 0;
    uint32_t startAddr = 0;
    uint32_t endAddr = 0;
    bool isScalarRead = false;
    bool isScalarWrite = false;
    bool isVectorRead = false;
    bool isVectorWrite = false;
    bool isMteRead = false;
    bool isMteWrite = false;
    bool isEnable = false;
    uint32_t reserved = 0;
};
} // namespace AscendC

/* **************************************************************************************************
 * LoadData(Layout) API Level2                                                                      *
 * **************************************************************************************************/
namespace AscendCBisheng {

struct LoadDataTrait {
    __aicore__ constexpr LoadDataTrait() {}

    __aicore__ constexpr LoadDataTrait(const bool transposedIn) : transposed(transposedIn) {}

    bool transposed = false;
};
constexpr LoadDataTrait DEFAULT_LOAD_DATA_TRAIT{};

}

#endif // ASCENDC_BISHENG_KERNEL_STRUCT_MM_H
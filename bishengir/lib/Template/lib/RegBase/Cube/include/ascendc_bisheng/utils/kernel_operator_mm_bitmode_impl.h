/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 *\file kernel_operator_mm_bitmode_impl.h
 *\brief
*/

#ifndef ASCENDC_BISHENG_KERNEL_OPERATOR_MM_BITMODE_IMPL_H
#define ASCENDC_BISHENG_KERNEL_OPERATOR_MM_BITMODE_IMPL_H
#include "./kernel_struct_mm.h"

namespace AscendCBisheng {
constexpr uint8_t PADLIST2 = 2;
constexpr uint8_t PADLIST3 = 3;

struct Load2DBitModeConfig0 {
    uint64_t mStartPosition:16;
    uint64_t kStartPosition:16;
    uint64_t mStep:8;
    uint64_t kStep:8;

    __aicore__ inline void Init(const LoadData2DParamsV2& loadData2DParams_) {
        mStartPosition = loadData2DParams_.mStartPosition;
        kStartPosition = loadData2DParams_.kStartPosition;
        mStep = loadData2DParams_.mStep;
        kStep = loadData2DParams_.kStep;
    }
    __aicore__ inline void SetMStartPosition(uint32_t mStartPosition_) {
        mStartPosition = mStartPosition_;
    };
    __aicore__ inline void SetKStartPosition(uint32_t kStartPosition_) {
        kStartPosition = kStartPosition_;
    };
    __aicore__ inline void SetMStep(uint16_t mStep_) {
        mStep = mStep_;
    };
    __aicore__ inline void SetKStep(uint16_t kStep_) {
        kStep = kStep_;
    };
    __aicore__ inline uint32_t GetMStartPosition() const {
        return mStartPosition;
    };
    __aicore__ inline uint32_t GetKStartPosition() const {
        return kStartPosition;
    };
    __aicore__ inline uint16_t GetMStep() const {
        return mStep;
    };
    __aicore__ inline uint16_t GetKStep() const {
        return kStep;
    };
};

struct Load2DBitModeConfig1 {
    uint64_t srcStride:16;
    uint64_t dstStride:16;
    __aicore__ inline void Init(const LoadData2DParamsV2& loadData2DParams_) {
        srcStride = loadData2DParams_.srcStride;
        dstStride = loadData2DParams_.dstStride;
    }
    __aicore__ inline void SetSrcStride(int32_t srcStride_) {
        srcStride = srcStride_;
    };
    __aicore__ inline void SetDstStride(uint16_t dstStride_) {
        dstStride = dstStride_;
    };
    __aicore__ inline int32_t GetSrcStride() const {
        return srcStride;
    };
    __aicore__ inline uint16_t GetDstStride() const {
        return dstStride;
    };
};

struct Load3DBitModeConfig0 {
    uint64_t kStep:16;
    uint64_t mStep:16;
    uint64_t kPos:16;
    uint64_t mPos:16;
    
    template <typename T>
    __aicore__ inline void Init(const LoadData3DParamsV2<T>& loadData3DParams_) {
        kStep = loadData3DParams_.kExtension;
        mStep = loadData3DParams_.mExtension;
        kPos = loadData3DParams_.kStartPt;
        mPos = loadData3DParams_.mStartPt;
    }
    __aicore__ inline void SetKExtension(uint16_t kExtension) {
        kStep = kExtension;
    };
    __aicore__ inline void SetMExtension(uint16_t mExtension) {
        mStep = mExtension;
    };
    __aicore__ inline void SetKStartPt(uint16_t kStartPt) {
        kPos = kStartPt;
    };
    __aicore__ inline void SetMStartPt(uint16_t mStartPt) {
        mPos = mStartPt;
    };
    __aicore__ inline uint16_t GetKExtension() const {
        return kStep;
    };
    __aicore__ inline uint16_t GetMExtension() const {
        return mStep;
    };
    __aicore__ inline uint16_t GetKStartPt() const {
        return kPos;
    };
    __aicore__ inline uint16_t GetMStartPt() const {
        return mPos;
    };
};

struct Load3DBitModeConfig1 {
    uint64_t strideW:6;
    uint64_t strideH:6;
    uint64_t Wk:8;
    uint64_t Hk:8;
    uint64_t dilationW:8;
    uint64_t dilationH:8;
    uint64_t filterW:1;
    uint64_t filterH:1;
    uint64_t transpose:1;
    uint64_t fMatrixCtrl:1;
    uint64_t sizeChannel:16;

    template <typename T>
    __aicore__ inline void Init(const LoadData3DParamsV2<T>& loadData3DParams_) {
        strideW = loadData3DParams_.strideW;
        strideH = loadData3DParams_.strideH;
        Wk = loadData3DParams_.filterW;
        Hk = loadData3DParams_.filterH;
        dilationW = loadData3DParams_.dilationFilterW;
        dilationH = loadData3DParams_.dilationFilterH;
        filterW = loadData3DParams_.filterSizeW;
        filterH = loadData3DParams_.filterSizeH;
        transpose = loadData3DParams_.enTranspose;
        fMatrixCtrl = loadData3DParams_.fMatrixCtrl;
        sizeChannel = loadData3DParams_.channelSize;
    }

    __aicore__ inline void SetStrideW(uint8_t strideW_) {
        strideW = strideW_;
    };
    __aicore__ inline void SetStrideH(uint8_t strideH_) {
        strideH = strideH_;
    };
    __aicore__ inline void SetFilterW(uint8_t filterW_) {
        Wk = filterW_;
    };
    __aicore__ inline void SetFilterH(uint8_t filterH_) {
        Hk = filterH_;
    };
    __aicore__ inline void SetDilationFilterW(uint8_t dilationFilterW_) {
        dilationW = dilationFilterW_;
    };
    __aicore__ inline void SetDilationFilterH(uint8_t dilationFilterH_) {
        dilationH = dilationFilterH_;
    };
    __aicore__ inline void SetFilterSizeW(bool filterSizeW_) {
        filterW = filterSizeW_;
    };
    __aicore__ inline void SetFilterSizeH(bool filterSizeH_) {
        filterH = filterSizeH_;
    };
    __aicore__ inline void SetTranspose(bool transpose_) {
        transpose = transpose_;
    };
    __aicore__ inline void SetFMatrixCtrl(bool fMatrixCtrl_) {
        fMatrixCtrl = fMatrixCtrl_;
    };
    __aicore__ inline void SetChannelSize(uint16_t channelSize_) {
        sizeChannel = channelSize_;
    };
    __aicore__ inline uint8_t GetStrideW() const {
        return strideW;
    };
    __aicore__ inline uint8_t GetStrideH() const {
        return strideH;
    };
    __aicore__ inline uint8_t GetFilterW() const {
        return Wk;
    };
    __aicore__ inline uint8_t GetFilterH() const {
        return Hk;
    };
    __aicore__ inline uint8_t GetDilationFilterW() const {
        return dilationW;
    };
    __aicore__ inline uint8_t GetDilationFilterH() const {
        return dilationH;
    };
    __aicore__ inline bool GetFilterSizeW() const {
        return filterW;
    };
    __aicore__ inline bool GetFilterSizeH() const {
        return filterH;
    };
    __aicore__ inline bool GetTranspose() const {
        return transpose;
    };
    __aicore__ inline bool GetFMatrixCtrl() const {
        return fMatrixCtrl;
    };
    __aicore__ inline uint16_t GetChannelSize() const {
        return sizeChannel;
    };
};

struct SetFMatrixBitModeConfig0 {
    uint64_t l1H:16;
    uint64_t l1W:16;
    uint64_t padList0:8;
    uint64_t padList1:8;
    uint64_t padList2:8;
    uint64_t padList3:8;

    template <typename T>
    __aicore__ inline void Init(const LoadData3DParamsV2<T>& loadData3DParams_) {
        l1H = loadData3DParams_.l1H;
        l1W = loadData3DParams_.l1W;
    }
    __aicore__ inline void SetL1H(uint16_t l1H_) {
        l1H = l1H_;
    };
    __aicore__ inline void SetL1W(uint16_t l1W_) {
        l1W = l1W_;
    };
    __aicore__ inline void SetPadList(const uint8_t padList_[4]) {
        padList0 = padList_[0];
        padList1 = padList_[1];
        padList2 = padList_[PADLIST2];
        padList3 = padList_[PADLIST3];
    };
};

struct MmadBitModeConfig0 {
    uint64_t m:12;
    uint64_t k:12;
    uint64_t n:12;
    uint64_t :19;
    uint64_t unitFlag:2;
    uint64_t :4;
    uint64_t disableGemv:1;
    uint64_t cmatrixSource:1;
    uint64_t cmatrixInitVal:1;

    __aicore__ inline void Init(const MmadParams& mmadParams_) {
        m = mmadParams_.m;
        k = mmadParams_.k;
        n = mmadParams_.n;
        unitFlag = mmadParams_.unitFlag;
        disableGemv = mmadParams_.disableGemv;
        cmatrixSource = mmadParams_.cmatrixSource;
        cmatrixInitVal = mmadParams_.cmatrixInitVal;
    }
    __aicore__ inline void SetM(uint16_t m_) {
        m = m_;
    };
    __aicore__ inline void SetK(uint16_t k_) {
        k = k_;
    };
    __aicore__ inline void SetN(uint16_t n_) {
        n = n_;
    };
    __aicore__ inline void SetUnitFlag(uint8_t unitFlag_) {
        unitFlag = unitFlag_;
    };
    __aicore__ inline void SetDisableGemv(bool disableGemv_) {
        disableGemv = disableGemv_;
    };
    __aicore__ inline void SetCmatrixSource(bool cmatrixSource_) {
        cmatrixSource = cmatrixSource_;
    };
    __aicore__ inline void SetCmatrixInitVal(bool cmatrixInitVal_) {
        cmatrixInitVal = cmatrixInitVal_;
    };
    __aicore__ inline uint16_t GetM() const {
        return m;
    };
    __aicore__ inline uint16_t GetK() const {
        return k;
    };
    __aicore__ inline uint16_t GetN() const {
        return n;
    };
    __aicore__ inline uint8_t GetUnitFlag() const {
        return unitFlag;
    };
    __aicore__ inline bool GetDisableGemv() const {
        return disableGemv;
    };
    __aicore__ inline bool GetCmatrixSource() const {
        return cmatrixSource;
    };
    __aicore__ inline bool GetCmatrixInitVal() const {
        return cmatrixInitVal;
    };
};
}  // namespace AscendCBisheng

#endif  // ASCENDC_BISHENG_KERNEL_OPERATOR_MM_BITMODE_IMPL_H
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
 *\file kernel_operator_mm_bitmode_intf.h
 *\brief
*/

#ifndef ASCENDC_BISHENG_KERNEL_OPERATOR_MM_BITMODE_INTERFACE_H
#define ASCENDC_BISHENG_KERNEL_OPERATOR_MM_BITMODE_INTERFACE_H
#include "kernel_struct_mm.h"
// #include "../../impl/basic_api/kernel_operator_mm_bitmode_struct.h"
#include "./kernel_operator_mm_bitmode_impl.h"

namespace AscendCBisheng {
class Load2DBitModeParam {
public:
    __aicore__ inline Load2DBitModeParam();
    __aicore__ inline Load2DBitModeParam(const LoadData2DParamsV2 &loadData2DParams_);
    __aicore__ inline void SetMStartPosition(uint32_t mStartPosition_);
    __aicore__ inline void SetKStartPosition(uint32_t kStartPosition_);
    __aicore__ inline void SetMStep(uint16_t mStep_);
    __aicore__ inline void SetKStep(uint16_t kStep_);
    __aicore__ inline void SetSrcStride(int32_t srcStride_);
    __aicore__ inline void SetDstStride(uint16_t dstStride_);
    __aicore__ inline void SetIfTranspose(bool ifTranspose_);
    __aicore__ inline void SetConfig0(uint64_t config0_) {
        config0 = config0_;
    };
    __aicore__ inline void SetConfig1(uint64_t config1_) {
        config1 = config1_;
    };

    __aicore__ inline uint32_t GetMStartPosition() const;
    __aicore__ inline uint32_t GetKStartPosition() const;
    __aicore__ inline uint16_t GetMStep() const;
    __aicore__ inline uint16_t GetKStep() const;
    __aicore__ inline int32_t GetSrcStride() const;
    __aicore__ inline uint16_t GetDstStride() const;
    __aicore__ inline uint64_t GetIfTranspose() const {
        return ifTranspose;
    };
    __aicore__ inline uint64_t GetConfig0() const {
        return config0;
    };
    __aicore__ inline uint64_t GetConfig1() const {
        return config1;
    };

private:
    union {
        uint64_t config0;
        struct Load2DBitModeConfig0 config0BitMode;
    };
    union {
        uint64_t config1;
        struct Load2DBitModeConfig1 config1BitMode;
    };
    bool ifTranspose = false;
};

class Load3DBitModeParam {
public:
    __aicore__ inline Load3DBitModeParam();
    template <typename T>
    __aicore__ inline Load3DBitModeParam(const LoadData3DParamsV2<T> &loadData3DParams_);
    __aicore__ inline void SetKExtension(uint16_t kStep_);
    __aicore__ inline void SetMExtension(uint16_t mStep_);
    __aicore__ inline void SetKStartPt(uint16_t kPos_);
    __aicore__ inline void SetMStartPt(uint16_t mPos_);
    __aicore__ inline void SetStrideW(uint8_t strideW_);
    __aicore__ inline void SetStrideH(uint8_t strideH_);
    __aicore__ inline void SetFilterW(uint8_t Wk_);
    __aicore__ inline void SetFilterH(uint8_t Hk_);
    __aicore__ inline void SetDilationFilterW(uint8_t dilationW_);
    __aicore__ inline void SetDilationFilterH(uint8_t dilationH_);
    __aicore__ inline void SetFilterSizeW(bool filterSizeW_);
    __aicore__ inline void SetFilterSizeH(bool filterSizeH_);
    __aicore__ inline void SetTranspose(bool transpose_);
    __aicore__ inline void SetFMatrixCtrl(bool fmatrixCtrl_);
    __aicore__ inline void SetChannelSize(uint16_t sizeChannel_);
    __aicore__ inline void SetConfig0(uint64_t config0_) {
        config0 = config0_;
    };
    __aicore__ inline void SetConfig1(uint64_t config1_) {
        config1 = config1_;
    };

    __aicore__ inline uint16_t GetKExtension() const;
    __aicore__ inline uint16_t GetMExtension() const;
    __aicore__ inline uint16_t GetKStartPt() const;
    __aicore__ inline uint16_t GetMStartPt() const;
    __aicore__ inline uint8_t GetStrideW() const;
    __aicore__ inline uint8_t GetStrideH() const;
    __aicore__ inline uint8_t GetFilterW() const;
    __aicore__ inline uint8_t GetFilterH() const;
    __aicore__ inline uint8_t GetDilationFilterW() const;
    __aicore__ inline uint8_t GetDilationFilterH() const;
    __aicore__ inline bool GetFilterSizeW() const;
    __aicore__ inline bool GetFilterSizeH() const;
    __aicore__ inline bool GetTranspose() const;
    __aicore__ inline bool GetFMatrixCtrl() const;
    __aicore__ inline uint16_t GetChannelSize() const;
    __aicore__ inline uint64_t GetConfig0() const {
        return config0;
    };
    __aicore__ inline uint64_t GetConfig1() const {
        return config1;
    };

private:
    union {
        uint64_t config0;
        struct Load3DBitModeConfig0 config0BitMode;
    };
    union {
        uint64_t config1;
        struct Load3DBitModeConfig1 config1BitMode;
    };
};

class SetFMatrixBitModeParams {
public:
    __aicore__ inline SetFMatrixBitModeParams();
    template <typename T>
    __aicore__ inline SetFMatrixBitModeParams(const LoadData3DParamsV2<T> &loadData3DParams_);
    __aicore__ inline void SetL1H(uint16_t l1H_);
    __aicore__ inline void SetL1W(uint16_t l1W_);
    __aicore__ inline void SetPadList(const uint8_t padList_[4]);
    __aicore__ inline void SetConfig0(uint64_t config0_) {
        config0 = config0_;
    };
    __aicore__ inline uint64_t GetConfig0() const {
        return config0;
    };

private:
    union {
        uint64_t config0;
        struct SetFMatrixBitModeConfig0 config0BitMode;
    };
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
class MmadBitModeParams {
public:
    __aicore__ inline MmadBitModeParams();
    __aicore__ inline MmadBitModeParams(const MmadParams &mmadParams_);
    __aicore__ inline void SetM(uint16_t m_);
    __aicore__ inline void SetK(uint16_t k_);
    __aicore__ inline void SetN(uint16_t n_);
    __aicore__ inline void SetUnitFlag(uint8_t unitFlag_);
    __aicore__ inline void SetDisableGemv(bool disableGemv_);
    __aicore__ inline void SetCmatrixSource(bool cmatrixSource_);
    __aicore__ inline void SetCmatrixInitVal(bool cmatrixInitVal_);
    __aicore__ inline void SetConfig0(uint64_t config0_) {
        config0 = config0_;
    };
    
    __aicore__ inline uint16_t GetM() const;
    __aicore__ inline uint16_t GetK() const;
    __aicore__ inline uint16_t GetN() const;
    __aicore__ inline uint8_t GetUnitFlag() const;
    __aicore__ inline bool GetDisableGemv() const;
    __aicore__ inline bool GetCmatrixSource() const;
    __aicore__ inline bool GetCmatrixInitVal() const;
    __aicore__ inline uint64_t GetConfig0() const {
        return config0;
    };

private:
    union {
        uint64_t config0;
        struct MmadBitModeConfig0 config0BitMode;
    };
};
#endif
}  // namespace AscendCBisheng

#endif // ASCENDC_BISHENG_KERNEL_OPERATOR_MM_BITMODE_INTERFACE_H
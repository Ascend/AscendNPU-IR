/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCENDC_BISHENG_KERNEL_OPERATOR_MM_IMPL_H
#define ASCENDC_BISHENG_KERNEL_OPERATOR_MM_IMPL_H

#include "./kernel_utils.h"
#include "./kernel_struct_aipp.h"
#include "./kernel_struct_mm.h"
#include "./kernel_macros.h"

namespace AscendCBisheng {
/* **************************************************************************************************
 * LoadData 2dv1                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData2DGM2L0ACal(__ca__ T *dst, __gm__ T *src, const LoadData2DParams &loadDataParam,
    const uint8_t cacheMode = 0)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
        "unsupported loaddata_2d from gm to A2 on current device"); });
}
 
template <typename T>
__aicore__ inline void LoadData2DGM2L0BCal(__cb__ T *dst, __gm__ T *src, const LoadData2DParams &loadDataParam,
    const uint8_t cacheMode = 0)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
        "unsupported loaddata_2d from gm to B2 on current device"); });
}
 
template <typename T>
__aicore__ inline void LoadData2DGM2L1Cal(__cbuf__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam,
    const uint8_t cacheMode = 0)
{
    static_assert(SupportType<T, uint8_t, int8_t, uint16_t, int16_t, half, bfloat16_t, float, int32_t, uint32_t>(),
        "LoadData 2dv2 only support uint8_t, int8_t, uint16_t, int16_t, half, bfloat16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
        uint16_t mStartPosition = 0;
        uint16_t kStartPosition = loadDataParam.startIndex;
        uint8_t mStep = loadDataParam.srcStride;
        uint8_t kStep = loadDataParam.repeatTimes;
        int16_t srcStride = static_cast<int16_t>(loadDataParam.srcStride);
        uint16_t dstStride = loadDataParam.dstGap + 1;
        set_mte2_src_para(uint64_t(srcStride));
        load_gm_to_cbuf_2dv2(dst, src, mStartPosition, kStartPosition, dstStride, mStep, kStep, loadDataParam.sid, 0, cacheMode);
    }
}

template <typename T>
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParam)
{
    static_assert(SupportType<T, uint8_t, int8_t, uint16_t, int16_t, half, bfloat16_t, float, int32_t, uint32_t>(),
        "LoadData 2dv2 only support uint8_t, int8_t, uint16_t, int16_t, half, bfloat16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
        uint16_t mStartPosition = 0;
        uint16_t kStartPosition = loadDataParam.startIndex;
        uint8_t mStep = loadDataParam.srcStride;
        uint8_t kStep = loadDataParam.repeatTimes;
        int16_t srcStride = static_cast<int16_t>(loadDataParam.srcStride);
        uint16_t dstStride = loadDataParam.dstGap + 1;
 
        if (loadDataParam.ifTranspose) {
            load_cbuf_to_ca(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        } else {
            load_cbuf_to_ca(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
        }
    }
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParam)
{
    static_assert(SupportType<T, uint8_t, int8_t, uint16_t, int16_t, half, bfloat16_t, float, int32_t, uint32_t>(),
        "LoadData 2dv2 only support uint8_t, int8_t, uint16_t, int16_t, half, bfloat16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
        uint16_t mStartPosition = 0;
        uint16_t kStartPosition = loadDataParam.startIndex;
        uint8_t mStep = loadDataParam.srcStride;
        uint8_t kStep = loadDataParam.repeatTimes;
        int16_t srcStride = static_cast<int16_t>(loadDataParam.srcStride);
        uint16_t dstStride = loadDataParam.dstGap + 1;
 
        if (loadDataParam.ifTranspose) {
            load_cbuf_to_cb(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        } else {
            load_cbuf_to_cb(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
        }
    }
}
/* **************************************************************************************************
 * LoadData 2dv2                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T *dst, __cbuf__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    static_assert(SupportType<T, uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, half, bfloat16_t,
                float, int32_t, uint32_t>(),
        "LoadData 2dv2 only support uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
         half, bfloat16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
        if (loadDataParam.ifTranspose) {
            load_cbuf_to_ca(dst,
                src,
                loadDataParam.mStartPosition,
                loadDataParam.kStartPosition,
                loadDataParam.mStep,
                loadDataParam.kStep,
                loadDataParam.srcStride,
                loadDataParam.dstStride,
                1);
        } else {
            load_cbuf_to_ca(dst,
                src,
                loadDataParam.mStartPosition,
                loadDataParam.kStartPosition,
                loadDataParam.mStep,
                loadDataParam.kStep,
                loadDataParam.srcStride,
                loadDataParam.dstStride,
                0);
        }
    }
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ T *dst, __cbuf__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    static_assert(SupportType<T, uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, half, bfloat16_t,
                float, int32_t, uint32_t>(),
        "LoadData 2dv2 only support uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
         half, bfloat16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
        if (loadDataParam.ifTranspose) {
            load_cbuf_to_cb(dst,
                src,
                loadDataParam.mStartPosition,
                loadDataParam.kStartPosition,
                loadDataParam.mStep,
                loadDataParam.kStep,
                loadDataParam.srcStride,
                loadDataParam.dstStride,
                1);
        } else {
            load_cbuf_to_cb(dst,
                src,
                loadDataParam.mStartPosition,
                loadDataParam.kStartPosition,
                loadDataParam.mStep,
                loadDataParam.kStep,
                loadDataParam.srcStride,
                loadDataParam.dstStride,
                0);
        }
    }
}

/* **************************************************************************************************
 * LoadDataWithTranspose                                        *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData2DL12L0BTransposeCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData2dTransposeParams &loadDataParam)
{
    static_assert(SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int32_t, uint32_t>(),
        "LoadDataWithTranspose only support uint8_t, int8_t, half, bfloat16_t, float, int32_t, uint32_t \
         on current device!");
    uint32_t unitMultiples = 1;
    if (IsSameType<T, int8_t>::value || IsSameType<T, uint8_t>::value || IsSameType<T, int32_t>::value ||
        IsSameType<T, uint32_t>::value || IsSameType<T, float>::value) {
        unitMultiples = 2;
    }
    // Value multiplication due to unit mismatch
    LoadData2dTransposeParamsV2 loadDataParamsV2(loadDataParam.startIndex * unitMultiples,
        loadDataParam.repeatTimes,
        loadDataParam.srcStride  * unitMultiples,
        loadDataParam.dstGap,
        loadDataParam.dstFracGap,
        0,
        loadDataParam.addrMode);
    LoadData2DL12L0BTransposeCal(dst, src, loadDataParamsV2);
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BTransposeCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData2dTransposeParamsV2 &loadDataParam)
{
    static_assert(SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int32_t, uint32_t>(),
        "LoadDataWithTranspose only support uint8_t, int8_t, half, bfloat16_t, float, int32_t, uint32_t \
         on current device!");
    if ASCEND_IS_AIC {
        load_cbuf_to_cb_transpose(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes,
            loadDataParam.srcStride, loadDataParam.dstGap, inc, loadDataParam.dstFracGap, loadDataParam.srcFracGap);
    }
}

template <typename T, typename U = T>
__aicore__ inline void LoadData2DL12L0ACal(__ca__ U *dst, __cbuf__ T *src0, __cbuf__ fp8_e8m0_t *src1,
    const LoadData2DParamsV2 &loadDataParam, const LoadData2DMxParams &loadMxDataParams)
{
    static_assert(SupportType<T, fp4x2_e2m1_t, fp4x2_e1m2_t>() ||
        SupportType<Tuple<U, T>,
                    Tuple<mx_fp8_e4m3_t, fp8_e4m3fn_t>,
                    Tuple<mx_fp8_e5m2_t, fp8_e5m2_t>>(),
        "LoadData 2dv2 with scale matrix only support fp4/fp8 dtype on current device!");
    if ASCEND_IS_AIC {
        if constexpr (SupportType<U, mx_fp8_e4m3_t, mx_fp8_e5m2_t>()) {
            if (loadDataParam.ifTranspose) {
                load_cbuf_to_ca(reinterpret_cast<__ca__ T *>(dst),
                    (__cbuf__ T *)src0,
                    loadDataParam.mStartPosition,
                    loadDataParam.kStartPosition,
                    loadDataParam.mStep,
                    loadDataParam.kStep,
                    loadDataParam.srcStride,
                    loadDataParam.dstStride,
                    1);
            } else {
                load_cbuf_to_ca(reinterpret_cast<__ca__ T *>(dst),
                    (__cbuf__ T *)src0,
                    loadDataParam.mStartPosition,
                    loadDataParam.kStartPosition,
                    loadDataParam.mStep,
                    loadDataParam.kStep,
                    loadDataParam.srcStride,
                    loadDataParam.dstStride,
                    0);
            }
        } else {
            if (loadDataParam.ifTranspose) {
                load_cbuf_to_ca_s4((__ca__ T *)dst,
                    (__cbuf__ T *)src0,
                    loadDataParam.mStartPosition,
                    loadDataParam.kStartPosition,
                    loadDataParam.mStep,
                    loadDataParam.kStep,
                    loadDataParam.srcStride,
                    loadDataParam.dstStride,
                    1);
            } else {
                load_cbuf_to_ca_s4((__ca__ T *)dst,
                    (__cbuf__ T *)src0,
                    loadDataParam.mStartPosition,
                    loadDataParam.kStartPosition,
                    loadDataParam.mStep,
                    loadDataParam.kStep,
                    loadDataParam.srcStride,
                    loadDataParam.dstStride,
                    0);
            }
        }
#if ASCENDC_CPU_DEBUG
        uint64_t l0ABaseAddr = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(ConstDefiner::Instance().GetHardwareBaseAddr(Hardware::L0A)));
        uint64_t bufferOffset = (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst)) - l0ABaseAddr) / 16;
        uint64_t mxDstAddr =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ConstDefiner::Instance().cpuL0AMx)) + bufferOffset;
#else
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst)) / 16;
#endif
        load_cbuf_to_ca_mx(mxDstAddr,
            static_cast<__cbuf__ void *>(src1),
            loadMxDataParams.xStartPosition,
            loadMxDataParams.yStartPosition,
            loadMxDataParams.xStep,
            loadMxDataParams.yStep,
            loadMxDataParams.srcStride,
            loadMxDataParams.dstStride);
    }
}

template <typename T, typename U = T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ U *dst, __cbuf__ T *src0, __cbuf__ fp8_e8m0_t *src1,
    const LoadData2DParamsV2 &loadDataParam, const LoadData2DMxParams &loadMxDataParams)
{
    static_assert(SupportType<T, fp4x2_e2m1_t, fp4x2_e1m2_t>() ||
        SupportType<Tuple<U, T>, 
                    Tuple<mx_fp8_e4m3_t, fp8_e4m3fn_t>,
                    Tuple<mx_fp8_e5m2_t, fp8_e5m2_t>>(),
        "LoadData 2dv2 with scale matrix only support fp4/fp8 dtype on current device!");
    if ASCEND_IS_AIC {
        if constexpr (SupportType<U, mx_fp8_e4m3_t, mx_fp8_e5m2_t>()) {
            if (loadDataParam.ifTranspose) {
                load_cbuf_to_cb(reinterpret_cast<__cb__ T *>(dst),
                    (__cbuf__ T *)(src0),
                    loadDataParam.mStartPosition,
                    loadDataParam.kStartPosition,
                    loadDataParam.mStep,
                    loadDataParam.kStep,
                    loadDataParam.srcStride,
                    loadDataParam.dstStride,
                    1);
            } else {
                load_cbuf_to_cb(reinterpret_cast<__cb__ T *>(dst),
                    (__cbuf__ T *)(src0),
                    loadDataParam.mStartPosition,
                    loadDataParam.kStartPosition,
                    loadDataParam.mStep,
                    loadDataParam.kStep,
                    loadDataParam.srcStride,
                    loadDataParam.dstStride,
                    0);
            }
        } else {
            if (loadDataParam.ifTranspose) {
                load_cbuf_to_cb_s4((__cb__ T *)dst,
                    (__cbuf__ T *)src0,
                    loadDataParam.mStartPosition,
                    loadDataParam.kStartPosition,
                    loadDataParam.mStep,
                    loadDataParam.kStep,
                    loadDataParam.srcStride,
                    loadDataParam.dstStride,
                    1);
            } else {
                load_cbuf_to_cb_s4((__cb__ T *)dst,
                    (__cbuf__ T *)src0,
                    loadDataParam.mStartPosition,
                    loadDataParam.kStartPosition,
                    loadDataParam.mStep,
                    loadDataParam.kStep,
                    loadDataParam.srcStride,
                    loadDataParam.dstStride,
                    0);
            }
        }

#if ASCENDC_CPU_DEBUG
        uint64_t l0BBaseAddr = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(ConstDefiner::Instance().GetHardwareBaseAddr(Hardware::L0B)));
        uint64_t bufferOffset = (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst)) - l0BBaseAddr) / 16;
        uint64_t mxDstAddr =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ConstDefiner::Instance().cpuL0BMx)) + bufferOffset;
#else
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst)) / 16;
#endif
        load_cbuf_to_cb_mx(mxDstAddr,
            static_cast<__cbuf__ void *>(src1),
            loadMxDataParams.xStartPosition,
            loadMxDataParams.yStartPosition,
            loadMxDataParams.xStep,
            loadMxDataParams.yStep,
            loadMxDataParams.srcStride,
            loadMxDataParams.dstStride);
    }
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0ACal(__ca__ T *dst, __gm__ T *src, const LoadData2DParamsV2 &loadDataParam,
    const uint8_t cacheMode = 0)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
        "unsupported loaddata_2d_v2 from gm to A2 on current device"); });
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0BCal(__cb__ T *dst, __gm__ T *src, const LoadData2DParamsV2 &loadDataParam,
    const uint8_t cacheMode = 0)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
        "unsupported loaddata_2d_v2 from gm to B2 on current device"); });
}

template <typename T>
__aicore__ inline void LoadData2DGM2L1Cal(__cbuf__ T *dst, __gm__ T *src, const LoadData2DParamsV2 &loadDataParam,
    const uint8_t cacheMode = 0)
{
    static_assert(SupportType<T, fp4x2_e2m1_t, fp4x2_e1m2_t, uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t,
                half, bfloat16_t, float, int32_t, uint32_t>(),
        "LoadData 2dv2 only support fp4x2_e2m1_t, fp4x2_e1m2_t, uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
         half, bfloat16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
        set_mte2_src_para(uint64_t(loadDataParam.srcStride));
        if constexpr (SupportType<T, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
            load_gm_to_cbuf_2dv2_s4(dst, src, loadDataParam.mStartPosition, loadDataParam.kStartPosition,
                loadDataParam.dstStride, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.sid, 0, cacheMode);
        } else {
            load_gm_to_cbuf_2dv2(dst, src, loadDataParam.mStartPosition, loadDataParam.kStartPosition,
                loadDataParam.dstStride, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.sid, 0, cacheMode);
        }
    }
}

template <typename T>
__aicore__ inline void LoadData2DL12L0ATransposeCal(__ca__ T *dst, __cbuf__ T *src,
    const LoadData2dTransposeParams &loadDataParam)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR,
        "LoadDataWithTranspose from A1 to A2 is not supported on current device"); });
}

/* **************************************************************************************************
 * Mmad                                             *
 * ************************************************************************************************* */

#if defined(__DAV_310R6__) 
template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void MmadCal(__cc__ DstT* c, __ca__ Src0T* a, __cb__ Src1T* b, const MmadParams& mmadParams)
{
    if ASCEND_IS_AIC {
        bool cmatrixInitVal = mmadParams.cmatrixInitVal && (!mmadParams.isBias);
        constexpr bool isMx = SupportType<Tuple<DstT, Src0T, Src1T>,
            Tuple<float, fp4x2_e1m2_t, fp4x2_e1m2_t>,
            Tuple<float, fp4x2_e1m2_t, fp4x2_e2m1_t>,
            Tuple<float, fp4x2_e2m1_t, fp4x2_e1m2_t>,
            Tuple<float, fp4x2_e2m1_t, fp4x2_e2m1_t>,
            Tuple<float, mx_fp8_e4m3_t, mx_fp8_e4m3_t>,
            Tuple<float, mx_fp8_e4m3_t, mx_fp8_e5m2_t>,
            Tuple<float, mx_fp8_e5m2_t, mx_fp8_e4m3_t>,
            Tuple<float, mx_fp8_e5m2_t, mx_fp8_e5m2_t>,
            Tuple<float, mx_fp8_e4m3_t, fp4x2_e2m1_t>,
            Tuple<float, mx_fp8_e4m3_t, fp4x2_e1m2_t>,
            Tuple<float, mx_fp8_e5m2_t, fp4x2_e2m1_t>,
            Tuple<float, mx_fp8_e5m2_t, fp4x2_e1m2_t>>();
        if constexpr (isMx) {
            constexpr uint32_t MX_BUFFER_SCALE = 16;
            using AType = typename GetDstType<Src0T>::Type;
            using BType = typename GetDstType<Src1T>::Type;
            uint64_t l0aMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(a)) / MX_BUFFER_SCALE;
            uint64_t l0bMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(b)) / MX_BUFFER_SCALE;
            mad_mx(c, 0, (__ca__ AType*)a, l0aMxBufferStartAddr, (__cb__ BType*)b, l0bMxBufferStartAddr, mmadParams.m, mmadParams.k,
                mmadParams.n, mmadParams.unitFlag, mmadParams.disableGemv, mmadParams.cmatrixSource, cmatrixInitVal);
        } else {
            mad(c, a, b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.disableGemv,
                mmadParams.cmatrixSource, cmatrixInitVal);
        }
    }
}
 
template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void MmadCal(
    __cc__ DstT* c, __ca__ Src0T* a, __cb__ Src1T* b, uint64_t bias, const MmadParams& mmadParams, bool cmatrixSource)
{
    if ASCEND_IS_AIC {
        bool cmatrixInitVal = mmadParams.cmatrixInitVal && (!mmadParams.isBias);
        constexpr bool isMx = SupportType<Tuple<DstT, Src0T, Src1T>,
            Tuple<float, fp4x2_e1m2_t, fp4x2_e1m2_t>,
            Tuple<float, fp4x2_e1m2_t, fp4x2_e2m1_t>,
            Tuple<float, fp4x2_e2m1_t, fp4x2_e1m2_t>,
            Tuple<float, fp4x2_e2m1_t, fp4x2_e2m1_t>,
            Tuple<float, mx_fp8_e4m3_t, mx_fp8_e4m3_t>,
            Tuple<float, mx_fp8_e4m3_t, mx_fp8_e5m2_t>,
            Tuple<float, mx_fp8_e5m2_t, mx_fp8_e4m3_t>,
            Tuple<float, mx_fp8_e5m2_t, mx_fp8_e5m2_t>,
            Tuple<float, mx_fp8_e4m3_t, fp4x2_e2m1_t>,
            Tuple<float, mx_fp8_e4m3_t, fp4x2_e1m2_t>,
            Tuple<float, mx_fp8_e5m2_t, fp4x2_e2m1_t>,
            Tuple<float, mx_fp8_e5m2_t, fp4x2_e1m2_t>>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        if constexpr (isMx) {
            using AType = typename GetDstType<Src0T>::Type;
            using BType = typename GetDstType<Src1T>::Type;
            mad_mx((__cc__ DstT*)c, (__ca__ AType*)a, (__cb__ BType*)b, bias, mmadParams.m, mmadParams.k, mmadParams.n,
                mmadParams.unitFlag, mmadParams.disableGemv, cmatrixSource, cmatrixInitVal);
        } else {
            mad(c, a, b, bias, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.disableGemv,
                cmatrixSource, cmatrixInitVal);
        }
#else
        if constexpr (isMx) {
            constexpr uint32_t MX_BUFFER_SCALE = 16;
            using AType = typename GetDstType<Src0T>::Type;
            using BType = typename GetDstType<Src1T>::Type;
            uint64_t l0aMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(a)) / MX_BUFFER_SCALE;
            uint64_t l0bMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(b)) / MX_BUFFER_SCALE;
            mad_mx(c, bias, (__ca__ AType*)a, l0aMxBufferStartAddr, (__cb__ BType*)b, l0bMxBufferStartAddr, mmadParams.m, mmadParams.k,
                mmadParams.n,mmadParams.unitFlag, mmadParams.disableGemv, cmatrixSource, cmatrixInitVal);
        } else {
            // Xd[31:0]: matrix c addr in L0C, Xd[63:32]: bias addr in bias table buffer;
            uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
            mad((__cc__ DstT*)xd, a, b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.disableGemv,
                cmatrixSource, cmatrixInitVal);
        }
#endif
    }
}
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void MmadCal(__cc__ DstT* c, __ca__ Src0T* a, __cb__ Src1T* b, const MmadParams& mmadParams)
{
    if ASCEND_IS_AIC {
        bool cmatrixInitVal = mmadParams.cmatrixInitVal && (!mmadParams.isBias);
        constexpr bool isMx = SupportType<Tuple<DstT, Src0T, Src1T>,
            Tuple<float, fp4x2_e1m2_t, fp4x2_e1m2_t>,
            Tuple<float, fp4x2_e1m2_t, fp4x2_e2m1_t>,
            Tuple<float, fp4x2_e2m1_t, fp4x2_e1m2_t>,
            Tuple<float, fp4x2_e2m1_t, fp4x2_e2m1_t>,
            Tuple<float, mx_fp8_e4m3_t, mx_fp8_e4m3_t>,
            Tuple<float, mx_fp8_e4m3_t, mx_fp8_e5m2_t>,
            Tuple<float, mx_fp8_e5m2_t, mx_fp8_e4m3_t>,
            Tuple<float, mx_fp8_e5m2_t, mx_fp8_e5m2_t>>();
        if constexpr (isMx) {
            using AType = typename GetDstType<Src0T>::Type;
            using BType = typename GetDstType<Src1T>::Type;
            mad_mx(c, (__ca__ AType*)a, (__cb__ BType*)b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.disableGemv,
                mmadParams.cmatrixSource, cmatrixInitVal);
        } else {
            mad(c, a, b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.disableGemv,
                mmadParams.cmatrixSource, cmatrixInitVal);
        }
    }
}

template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void MmadCal(
    __cc__ DstT* c, __ca__ Src0T* a, __cb__ Src1T* b, uint64_t bias, const MmadParams& mmadParams, bool cmatrixSource)
{
    if ASCEND_IS_AIC {
        bool cmatrixInitVal = mmadParams.cmatrixInitVal && (!mmadParams.isBias);
        constexpr bool isMx = SupportType<Tuple<DstT, Src0T, Src1T>,
            Tuple<float, fp4x2_e1m2_t, fp4x2_e1m2_t>,
            Tuple<float, fp4x2_e1m2_t, fp4x2_e2m1_t>,
            Tuple<float, fp4x2_e2m1_t, fp4x2_e1m2_t>,
            Tuple<float, fp4x2_e2m1_t, fp4x2_e2m1_t>,
            Tuple<float, mx_fp8_e4m3_t, mx_fp8_e4m3_t>,
            Tuple<float, mx_fp8_e4m3_t, mx_fp8_e5m2_t>,
            Tuple<float, mx_fp8_e5m2_t, mx_fp8_e4m3_t>,
            Tuple<float, mx_fp8_e5m2_t, mx_fp8_e5m2_t>>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        if constexpr (isMx) {
            using AType = typename GetDstType<Src0T>::Type;
            using BType = typename GetDstType<Src1T>::Type;
            mad_mx((__cc__ DstT*)c, (__ca__ AType*)a, (__cb__ BType*)b, bias, mmadParams.m, mmadParams.k, mmadParams.n,
                mmadParams.unitFlag, mmadParams.disableGemv, cmatrixSource, cmatrixInitVal);
        } else {
            mad(c, a, b, bias, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.disableGemv,
                cmatrixSource, cmatrixInitVal);
        }
#else
        // Xd[31:0]: matrix c addr in L0C, Xd[63:32]: bias addr in bias table buffer;
        uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
        if constexpr (isMx) {
            using AType = typename GetDstType<Src0T>::Type;
            using BType = typename GetDstType<Src1T>::Type;
            mad_mx((__cc__ DstT*)xd, (__ca__ AType*)a, (__cb__ BType*)b, mmadParams.m, mmadParams.k, mmadParams.n,
                mmadParams.unitFlag, mmadParams.disableGemv, cmatrixSource, cmatrixInitVal);
        } else {
            mad((__cc__ DstT*)xd, a, b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.disableGemv,
                cmatrixSource, cmatrixInitVal);
        }
#endif
    }
}
#endif

/* **************************************************************************************************
 * BroadCastVecToMM                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void BroadCastVecToMMCal(__cc__ T *dstLocal, __ubuf__ T *srcLocal, const int32_t blockCount,
    const uint8_t blockLen, const uint8_t srcGap, const uint8_t dstGap)
{
    broadcast_ub_to_cc(dstLocal, srcLocal, blockCount, blockLen, srcGap, dstGap);
}

/* **************************************************************************************************
 * InitL1Buffer                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL1BufferCal(__cbuf__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    if ASCEND_IS_AIC {
        int64_t repeatBit = (static_cast<uint64_t>(initConstValueParams.blockNum) << 16) |
                            (static_cast<uint64_t>(initConstValueParams.dstGap) << 32) | initConstValueParams.repeatTimes;
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            create_cbuf_matrix_bf16(dst, repeatBit, initConstValueParams.initValue);
        } else if constexpr (IsSameType<T, uint32_t>::value || IsSameType<T, half>::value) {
            create_cbuf_matrix(dst, repeatBit, static_cast<T>(initConstValueParams.initValue));
        } else if constexpr (IsSameType<T, int16_t>::value || IsSameType<T, uint16_t>::value) {
            create_cbuf_matrix(dst, repeatBit, GetScalarBitcodeToHalf(initConstValueParams.initValue));
        } else if constexpr (IsSameType<T, float>::value || IsSameType<T, int32_t>::value) {
            create_cbuf_matrix(
                dst, repeatBit, static_cast<uint32_t>(GetScalarBitcodeValue(initConstValueParams.initValue)));
        } else {
            ASCENDC_ASSERT(false,
                { KERNEL_LOG(KERNEL_ERROR, "InitConstValue doesn't support current data type on current device"); });
        }
    }
}

/* **************************************************************************************************
 * InitL0ANzMatrix                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL0ANzMatrixCal(__ca__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "InitConstValue in A2 is not supported on current device"); });
}

/* **************************************************************************************************
 * InitL0BNzMatrix                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL0BNzMatrixCal(__cb__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "InitConstValue in B2 is not supported on current device"); });
}

/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV1L12L0ACal(__ca__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v1 from l1 to l0a"); });
}

template <typename T>
__aicore__ inline void LoadData3DV1L12L0BCal(__cb__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v1 from l1 to l0b"); });
}

template <typename T>
__aicore__ inline void LoadData3DV1L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v1 from l1 to ubuf"); });
}

/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * ************************************************************************************************* */
__aicore__ inline void Load3DSetFMatrixCal(uint16_t l1H, uint16_t l1W, const uint8_t padList[4])
{
    if ASCEND_IS_AIC {
        uint64_t regFMatrix = 0;
        regFMatrix |= uint64_t(l1W & 0xFFFF);

        constexpr uint32_t l1HShiftBit = 16;
        regFMatrix |= uint64_t(l1H & 0xFFFF) << l1HShiftBit;

        constexpr uint32_t padNumber = 4;
        constexpr uint32_t padListShiftBit = 8;
        constexpr uint32_t padListShiftBase = 32;
        for (uint32_t i = 0; i < padNumber; i++) {
            regFMatrix |= uint64_t(padList[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
        }
        set_fmatrix(regFMatrix);
    }
}

__aicore__ inline void Load3DSetFMatrixBCal(uint16_t l1H, uint16_t l1W, const uint8_t padList[4])
{
    if ASCEND_IS_AIC {
        uint64_t regFMatrix = 0;
        regFMatrix |= static_cast<uint64_t>(l1W);

        constexpr uint32_t l1HShiftBit = 16;
        regFMatrix |= static_cast<uint64_t>(l1H) << l1HShiftBit;

        constexpr uint32_t padNumber = 4;
        constexpr uint32_t padListShiftBit = 8;
        constexpr uint32_t padListShiftBase = 32;
        for (uint32_t i = 0; i < padNumber; i++) {
            regFMatrix |= uint64_t(padList[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
        }
        set_fmatrix_b(regFMatrix);
    }
}

template <typename T>
__aicore__ inline void Load3DSetPaddingCal(const T padValue)
{
    uint16_t paddingValue = 0;
    constexpr uint16_t padValueShiftBit = 8;

    if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        paddingValue = static_cast<uint16_t>(GetScalarBitcodeValue((T)padValue));
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        paddingValue = static_cast<uint32_t>(GetScalarBitcodeValue((T)padValue));
    } else {
        // type = b8, PADDING[15:8] and PADDING[7:0] should be the same, while PADDING[31:16] is neglected.
        uint16_t u16Value = static_cast<uint16_t>(padValue) & 0xFFu;
        paddingValue = (u16Value << padValueShiftBit) | u16Value;
    }
    set_padding(paddingValue);
}

__aicore__ inline void SetLoadDataRepeatCal(const LoadDataRepeatParam repeatParams)
{
    uint64_t rptConfig = 0;
    constexpr uint32_t repeatTimeShiftBit = 16;
    rptConfig |= uint64_t(repeatParams.repeatStride);
    rptConfig |= uint64_t(repeatParams.repeatTime) << repeatTimeShiftBit;
    constexpr uint32_t repeatModeShiftBit = 24;
    rptConfig |= uint64_t(repeatParams.repeatMode) << repeatModeShiftBit;

    constexpr uint32_t dstStrideShiftBit = 32;
    rptConfig |= uint64_t(repeatParams.dstStride) << dstStrideShiftBit;
    set_l3d_rpt(rptConfig);
}

/* **************************************************************************************************
 * SetLoadDataBoundary                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataBoundaryCal(uint32_t boundaryValue)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetLoadDataBoundary is not supported on current device!"); });
}

/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
    static_assert(SupportType<T, uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, half, bfloat16_t, uint16_t,
                int16_t, float, int32_t, uint32_t>(),
        "LoadData 3dv2 only support uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
         half, bfloat16_t, uint16_t, int16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_ca(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
            loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
            loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
            loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
            loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
    }
}

template <typename T>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
    static_assert(SupportType<T, uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, half, bfloat16_t, uint16_t,
                int16_t, float, int32_t, uint32_t>(),
        "LoadData 3dv2 only support uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
         half, bfloat16_t, uint16_t, int16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_cb(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
            loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
            loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
            loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
            loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
    }
}

__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ half* dst, __cbuf__ half* src,
    const LoadData3DParamsV2<bfloat16_t>& loadDataParams)
{
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_ca(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
            loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
            loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
            loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
            loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
    }
}

__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ half* dst, __cbuf__ half* src,
    const LoadData3DParamsV2<bfloat16_t>& loadDataParams)
{
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_cb(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
            loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
            loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
            loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
            loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
    }
}

template <typename T>
__aicore__ inline void LoadData3DV2L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LoadData3DV2L12UB is not supported on current device"); });
}
/* **************************************************************************************************
 * LoadData 3dv2Pro                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ T *dst, __cbuf__ T *src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    static_assert(SupportType<T, uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, half, bfloat16_t,
                float, int32_t, uint32_t>(),
        "LoadData 3dv2Pro only support uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
         half, bfloat16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_ca(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
            loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
            loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
            loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.filterSizeW,
            loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
            loadDataParams.channelSize);
    }
}

template <>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ bfloat16_t* dst, __cbuf__ bfloat16_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_ca((__ca__ half*)dst, (__cbuf__ half*)src,
            loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
            loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
            loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
            loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.filterSizeW,
            loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
            loadDataParams.channelSize);
    }
}

template <typename T>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    static_assert(SupportType<T, uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, half, bfloat16_t,
            float, int32_t, uint32_t>(),
        "LoadData 3dv2Pro only support uint8_t, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
         half, bfloat16_t, float, int32_t, uint32_t on current device!");
    if ASCEND_IS_AIC {
            img2colv2_cbuf_to_cb(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
                loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
                loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
                loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
                loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
                loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.filterSizeW,
                loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
                loadDataParams.channelSize);
    }
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_cb((__cb__ half*)dst, (__cbuf__ half*)src,
            loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
            loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
            loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
            loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.filterSizeW,
            loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
            loadDataParams.channelSize);
    }
}

template <typename T>
__aicore__ inline void LoadData3DV2L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR,
        "LoadData3DV2L12UB is not supported 3DParamsV2Pro on current device"); });
}
}  // namespace AscendCBisheng
#endif  // ASCENDC_BISHENG_KERNEL_OPERATOR_MM_IMPL_H

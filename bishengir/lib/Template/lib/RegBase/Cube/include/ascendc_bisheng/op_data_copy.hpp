#ifndef ASCENDC_BISHENG_OP_DATA_COPY_HPP
#define ASCENDC_BISHENG_OP_DATA_COPY_HPP

#include "__clang_cce_types.h"
#include "catlass/catlass.hpp"
#include "local_tensor.hpp"
#include "kernel_log.h"
#include "utils/kernel_utils_constants.h"
#include "utils/kernel_macros.h"
#include "event_utils.hpp"
#include "utils/std/is_same.h"

namespace AscendCBisheng {

struct DataCopyParams {
    __aicore__ inline DataCopyParams() {}

    __aicore__ inline DataCopyParams(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn,
        const uint16_t dstStrideIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn)
    {}

    uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;
    uint16_t blockLen = 0;
    // srcStride and dstStride will be deprecated, use srcGap and dstGap instead.
    union {
        uint16_t srcGap = 0;
        // srcStride will be deprecated, use srcGap instead
        uint16_t srcStride;
    };

    union {
        uint16_t dstGap = 0;
        // dstStride will be deprecated, use dstGap instead
        uint16_t dstStride;
    };
};


// Only A1(L1) To C2(Bias) is requred for DataCopy

template <typename T>
__aicore__ inline void CopyCbufToBt(uint64_t dst, __cbuf__ T* src, const uint16_t convControl,
                                    const uint16_t blockCount, const uint16_t blockLen, const uint16_t srcStride,
                                    const uint16_t dstStride)
{
    if ASCEND_IS_AIC {
        if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, float>::value
                      || std::is_same<T, int32_t>::value || std::is_same<T, half>::value) {
            copy_cbuf_to_bt(dst, src, static_cast<bool>(convControl), blockCount, blockLen, srcStride, dstStride);
        }
    }
}

template <typename T>
__aicore__ inline __in_pipe__(MTE1)
    __out_pipe__(MTE1) void DataCopyL12BTImpl(const uint64_t dst, __cbuf__ T* src, const uint16_t isEnableConv,
                                              const DataCopyParams& intriParams)
{
    CopyCbufToBt(dst, src, isEnableConv, intriParams.blockCount, intriParams.blockLen, intriParams.srcStride,
                 intriParams.dstStride);
}

/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] intriParams.blockCount number of blocks
 * @param [in] intriParams.blockLen Length of blocks
 * @param [in] intriParams.srcGap src block gap
 * @param [in] intriParams.dstGap dst block gap
 */
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const DataCopyParams &repeatParams)
{
    // only A1 to C2 is required
    using PrimType = PrimT<T>;

    const Hardware dstHWPos = GetPhyType((TPosition)dst.GetPosition());
    const Hardware srcHWPos = GetPhyType((TPosition)src.GetPosition());

    if (srcHWPos == Hardware::L1) {
        if (dstHWPos == Hardware::BIAS) {
            DataCopyL12BTImpl((uint64_t)dst.GetPhyAddr(), (__cbuf__ PrimType*)src.GetPhyAddr(), static_cast<uint16_t>(0),
                            repeatParams);
        }
    }
}

/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from L1 to bt, applicable to vector data
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] intriParams.blockCount number of blocks
 * @param [in] intriParams.blockLen Length of blocks
 * @param [in] intriParams.srcGap src block gap
 * @param [in] intriParams.dstGap dst block gap
 */
template <typename T, typename U>
__aicore__ inline void DataCopy(const LocalTensor<T> &dst, const LocalTensor<U> &src,
    const DataCopyParams &repeatParams)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
    const Hardware dstHWPos = GetPhyType((TPosition)dst.GetPosition());
    const Hardware srcHWPos = GetPhyType((TPosition)src.GetPosition());

    if (srcHWPos == Hardware::L1) {
        if (dstHWPos == Hardware::BIAS) {
            // l1 -> bt
            if constexpr (Std::is_same<PrimSrcType, bfloat16_t>::value && (!Std::is_same<PrimDstType, float>::value)) {
                ASCENDC_ASSERT((false), {
                    KERNEL_LOG(KERNEL_ERROR,
                        "unsupport case where src dtype is bfloat16, dst dtype is not float on current device");
                });
            } else if constexpr (Std::is_same<PrimDstType, PrimSrcType>::value ||
                (Std::is_same<PrimSrcType, bfloat16_t>::value && Std::is_same<PrimDstType, float>::value)) {
                DataCopyL12BTImpl((uint64_t)dst.GetPhyAddr(), (__cbuf__ PrimSrcType*)src.GetPhyAddr(),
                    static_cast<uint16_t>(0), repeatParams);
            } else if constexpr (Std::is_same<PrimDstType, float>::value && Std::is_same<PrimSrcType, half>::value) {
                DataCopyL12BTImpl((uint64_t)dst.GetPhyAddr(), (__cbuf__ half *)src.GetPhyAddr(), static_cast<uint16_t>(1),
                    repeatParams);
            } else {
                ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in DataCopy from C1 to C2, "
                    "current api support dtype combination is U = T or src: half, dst: float.");});
            }
        }
    }
}

}

#endif // ASCENDC_BISHENG_OP_DATA_COPY_HPP

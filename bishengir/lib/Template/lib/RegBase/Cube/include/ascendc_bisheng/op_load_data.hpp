#ifndef ASCENDC_BISHENG_OP_LOAD_DATA_HPP
#define ASCENDC_BISHENG_OP_LOAD_DATA_HPP

#include "__clang_cce_types.h"
#include "event_utils.hpp"
#include "utils/kernel_utils_constants.h"

namespace AscendCBisheng {

/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor/GlobalTensor
 * @param [in] loadDataParams.mStartPosition m start position
 * @param [in] loadDataParams.kStartPosition k start position
 * @param [in] loadDataParams.srcStride src block stride
 * @param [in] loadDataParams.dstStride dst block stride
 * @param [in] loadDataParams.mStep m step
 * @param [in] loadDataParams.kStep k step
 * @param [in] loadDataParams.sid SMMU SID
 * @param [in] loadDataParams.ifTranspose enable parameters of transpose function
 */
template <typename T>
__aicore__ inline void LoadDataImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LoadData2DParamsV2& loadDataParams)
{
    const Hardware dstScope = GetPhyType((TPosition)dst.GetPosition());
    if (dstScope == Hardware::L0A) {
        LoadData2DL12L0ACal((__ca__ PrimT<T>*)dst.GetPhyAddr(),
                            (__cbuf__ PrimT<T>*)src.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L0B) {
        LoadData2DL12L0BCal((__cb__ PrimT<T>*)dst.GetPhyAddr(),
                            (__cbuf__ PrimT<T>*)src.GetPhyAddr(), loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dst", "A2 / B2",
            "LoadData with LoadData2DParamsV2",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dst.GetPosition())));
    }
}

// check LoadData2D datatype
template <typename T>
__aicore__ static inline void CheckLoadData2dDatatype()
{
  // no related code for __NPU_ARCH = 3510 in original implementation, kept for compatibility
}

/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor/GlobalTensor
 * @param [in] loadDataParams.mStartPosition m start position
 * @param [in] loadDataParams.kStartPosition k start position
 * @param [in] loadDataParams.srcStride src block stride
 * @param [in] loadDataParams.dstStride dst block stride
 * @param [in] loadDataParams.mStep m step
 * @param [in] loadDataParams.kStep k step
 * @param [in] loadDataParams.sid SMMU SID
 * @param [in] loadDataParams.ifTranspose enable parameters of transpose function
 */
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LoadData2DParamsV2& loadDataParams)
{
    CheckLoadData2dDatatype<T>();
    LoadDataImpl(dst, src, loadDataParams);
}

} // namespace AscendCBisheng

#endif // ASCENDC_BISHENG_OP_LOAD_DATA_HPP

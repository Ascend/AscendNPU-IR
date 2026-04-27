#ifndef ASCENDC_BISHENG_LOCAL_TENSOR_HPP
#define ASCENDC_BISHENG_LOCAL_TENSOR_HPP

#include "utils/kernel_utils_constants.h"
#include "common_types.h"
#include "catlass/catlass.hpp"

namespace AscendCBisheng {

using TBufHandle = uint8_t*;

struct TBuffAddr {
    uint32_t dataLen;
    uint32_t bufferAddr;
    TBufHandle bufferHandle;
    uint8_t logicPos;

    __aicore__ inline TBuffAddr() {}
};

template<typename T>
class LocalTensor {
public:
    using PrimType = PrimT<T>;
    TBuffAddr address_;

    template<typename U>
    __aicore__ inline void CreateTensor(TPosition pos, uint32_t addr, uint32_t tileSize);
    __aicore__ inline LocalTensor<T>(TPosition pos, uint32_t addr, uint32_t tileSize);
    __aicore__ inline LocalTensor operator[](uint32_t offset) const;
    __aicore__ inline int32_t GetPosition() const;
    __aicore__ inline uint64_t GetPhyAddr() const;
    __aicore__ inline uint64_t GetPhyAddr(const uint32_t offset) const;
};

template <typename T>
template <typename U>
__aicore__ inline void LocalTensor<T>::CreateTensor(TPosition pos, uint32_t addr, uint32_t tileSize)
{
    this->address_.dataLen = SizeOfBits<U>::value * tileSize / SizeOfBits<uint8_t>::value;
    this->address_.bufferAddr = addr;
    this->address_.logicPos = static_cast<uint8_t>(pos);
}

template <typename T>
__aicore__ inline LocalTensor<T>::LocalTensor(TPosition pos, uint32_t addr, uint32_t tileSize)
{
    CreateTensor<T>(pos, addr, tileSize);
}

template <typename T> __aicore__ inline LocalTensor<T> LocalTensor<T>::operator[](const uint32_t offset) const
{
    LocalTensor result = *this;
    if constexpr (IsHalfByteDataType<PrimType>()) {
        result.address_.dataLen -= (offset / INT4_TWO);
        result.address_.bufferAddr = result.address_.bufferAddr + offset / INT4_TWO;
    } else {
        result.address_.dataLen -= (offset * sizeof(PrimType));
        result.address_.bufferAddr = result.address_.bufferAddr + offset * sizeof(PrimType);
    }
    return result;
}

template <typename T> __aicore__ inline int32_t LocalTensor<T>::GetPosition() const
{
    return this->address_.logicPos;
}

template <typename T> __aicore__ inline uint64_t LocalTensor<T>::GetPhyAddr() const
{
    return GetPhyAddr(0);
}
template <typename T> __aicore__ inline uint64_t LocalTensor<T>::GetPhyAddr(const uint32_t offset) const
{
    if constexpr (IsSameType<PrimType, int4b_t>::value) {
        return this->address_.bufferAddr + offset / INT4_TWO;
    } else {
        return this->address_.bufferAddr + offset * sizeof(PrimType);
    }
}

} //namespace AscendCBisheng

#endif // ASCENDC_BISHENG_LOCAL_TENSOR_HPP

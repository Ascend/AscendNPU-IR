/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INCLUDE_CATLASS_ARCH_MEMORY_H
#define INCLUDE_CATLASS_ARCH_MEMORY_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Arch {

struct LocalTensorBufferBase {
public:
    template <class Element = half>
    CATLASS_DEVICE
    AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    CATLASS_DEVICE
    LocalTensorBufferBase() = default;

    AscendC::LocalTensor<uint8_t> tensor;
};

template <
    class ArchTag,
    AscendCBisheng::TPosition Position
>
struct LocalTensorBuffer {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported local tensor buffer, can not find the specialization.");
};

/// Partial specialization for TPosition::A1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::A1> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::A1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::A1, 0, ArchTag::L1_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::A2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::A2> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::A2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::A2, 0, ArchTag::L0A_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::B1> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::B1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::B1, 0, ArchTag::L1_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::B2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::B2> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::B2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::B2, 0, ArchTag::L0B_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::C1> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::C1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::C1, 0, ArchTag::L1_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C2
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::C2> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::C2;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::C2, 0, ArchTag::BIAS_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::CO1
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::CO1> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::CO1;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::CO1, 0, ArchTag::L0C_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::C2PIPE2GM
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::C2PIPE2GM> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::C2PIPE2GM;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::C2PIPE2GM, 0, ArchTag::FIXBUF_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECIN
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::VECIN> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::VECIN;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::VECIN, 0, ArchTag::UB_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECOUT
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::VECOUT> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::VECOUT;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::VECOUT, 0, ArchTag::UB_SIZE);
    }
};

///////////////////////////////////////////////////////////

/// Partial specialization for TPosition::VECCALC
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendCBisheng::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    static constexpr AscendCBisheng::TPosition Position = AscendCBisheng::TPosition::VECCALC;

    CATLASS_DEVICE
    LocalTensorBuffer()
    {
        tensor = AscendC::LocalTensor<uint8_t>(AscendCBisheng::TPosition::VECCALC, 0, ArchTag::UB_SIZE);
    }
};

}  // namespace Catlass::Arch

#endif  // INCLUDE_CATLASS_ARCH_MEMORY_H
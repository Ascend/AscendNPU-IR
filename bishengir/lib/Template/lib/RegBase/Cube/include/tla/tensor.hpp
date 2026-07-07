/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TENSOR_HPP
#define TLA_TENSOR_HPP

#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"                     // tla::Shape
#include "tla/numeric/integral_constant.hpp"  // tla::is_integral
#include "tla/int_tuple.hpp"
#include "catlass/catlass.hpp"

namespace tla {
//
// Tensor
//

template <typename T, class Layout_, class Coord_, AscendCBisheng::TPosition Position>
struct Tensor {
    using Element = typename AscendCBisheng::LocalTensor<T>::PrimType;
    using Layout = Layout_;
    using Coord = Coord_;
    static constexpr AscendCBisheng::TPosition position = Position;

    CATLASS_HOST_DEVICE constexpr
    Tensor() {}

    CATLASS_HOST_DEVICE constexpr
    Tensor(AscendCBisheng::LocalTensor<T> const& builtinTensor, Layout const& layout, Coord const& coord = {})
        : rep_(builtinTensor, layout, coord) {}

    //
    // Accessors
    //

    static constexpr int rank  = Layout::rank;

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) tensor() const
    {
        return *this;
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) data() const
    {
        return get<0>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) data()
    {
        return get<0>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) layout() const
    {
        return get<1>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) coord() const
    {
        return get<2>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) shape() const
    {
        return layout().shape();
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) stride() const
    {
        return layout().stride();
    }

    tla::tuple<AscendCBisheng::LocalTensor<T>, Layout, Coord> rep_;
};

template <typename T, class Layout, class PositionType>
CATLASS_HOST_DEVICE constexpr
auto MakeTensor(AscendCBisheng::LocalTensor<T> const& builtinTensor, Layout const& layout, PositionType)
{
    using Coord = detail::MakeZeroTuple<Layout::rank>;
    return Tensor<T, Layout, Coord, PositionType::value>(builtinTensor, layout);
}

template <typename T, class Layout, class Coord, class PositionType>
CATLASS_HOST_DEVICE constexpr
auto MakeTensor(AscendCBisheng::LocalTensor<T> const& builtinTensor, Layout const& layout, Coord const& coord, PositionType)
{
    return Tensor<T, Layout, Coord, PositionType::value>(builtinTensor, layout, coord);
}

template <class Tensor, class Coord, class Shape>
CATLASS_DEVICE constexpr
auto GetTile(Tensor const& tensor, Coord const& coord, Shape const& shape)
{
    auto layout = tensor.layout();
    auto builtinTensor = tensor.data();
    auto layoutNew = MakeLayoutTile(layout, shape);
    auto coordNew = Add(tensor.coord(), coord);
    return MakeTensor(builtinTensor, layoutNew, coordNew, Catlass::Arch::PositionType<Tensor::position>{});
}

} // end namespace tla

#endif // TLA_TENSOR_HPP

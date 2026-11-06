/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Utils.h"
#include "ascendc_bisheng/ascendc_bisheng.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a_a5.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b_a5.hpp"
#include "tla/tensor.hpp"

namespace {

// Besides the conversion-time checks, validate the effective address here:
// pointer_cast/subview can provide a runtime base or offset. Do not silently
// truncate a pointer when constructing LocalTensor's 32-bit local address.
template <typename T, typename QualifiedT>
__aicore__ __attribute__((always_inline)) uint32_t
checkedAddress(memref_t<QualifiedT, 4> *ref, uint32_t capacityBytes) {
  constexpr int64_t c0 = 32 / sizeof(T);
  constexpr int64_t fractal = 16 * c0;
  const int64_t capacity = capacityBytes / sizeof(T);
  if (ref->sizes[0] <= 0 || ref->sizes[0] > capacity || ref->sizes[1] <= 0 ||
      ref->sizes[1] > capacity || ref->sizes[2] != 16 || ref->sizes[3] != c0)
    trap();
  if (ref->strides[3] != 1 || ref->strides[2] != c0 ||
      ref->strides[1] != fractal || ref->strides[0] < ref->sizes[1] * fractal ||
      ref->strides[0] > capacity || ref->strides[0] % fractal != 0)
    trap();
  if (ref->sizes[0] - 1 >
      (capacity - ref->sizes[1] * fractal) / ref->strides[0])
    trap();
  if (ref->offset < 0 || ref->offset > capacity)
    trap();
  if (ref->offset % fractal != 0)
    trap();
  uint64_t span =
      (ref->sizes[0] - 1) * ref->strides[0] + ref->sizes[1] * fractal;
  uint64_t base = reinterpret_cast<uint64_t>(ref->aligned);
  uint64_t offsetBytes = ref->offset * sizeof(T);
  if (base > capacityBytes || offsetBytes > capacityBytes - base)
    trap();
  uint64_t address = base + offsetBytes;
  if (address % 512 != 0 || span * sizeof(T) > capacityBytes - address)
    trap();
  return static_cast<uint32_t>(address);
}

// One explicit copy, with no K loop, implicit L0 allocation or synchronization.
// Both physical descriptors are [outer, inner, 16, C0]. L1 is zN when not
// transposed and nZ otherwise; A5 L0A is zN and L0B is nZ.
template <typename T, typename DstT, bool ToA, bool Transposed>
__aicore__ __attribute__((always_inline)) void
copyL12L0(memref_t<__cbuf__ T, 4> *src, memref_t<DstT, 4> *dst,
          int64_t kPartIdx, int64_t kPart, int64_t kPartCeil, int64_t kPartLoop,
          int64_t kPartActual, int64_t mn) {
  using namespace Catlass;
  using ArchTag = Arch::AtlasA5;
  constexpr uint32_t c0 = 32 / sizeof(T);
  uint32_t srcAddr = checkedAddress<T>(src, ArchTag::L1_SIZE);
  uint32_t dstAddr =
      checkedAddress<T>(dst, ToA ? ArchTag::L0A_SIZE : ArchTag::L0B_SIZE);
  uint32_t rows = Transposed ? src->sizes[0] * c0 : src->sizes[1] * 16;
  uint32_t cols = Transposed ? src->sizes[1] * 16 : src->sizes[0] * c0;
  int64_t srcK = ToA ? cols : rows;
  int64_t srcMN = ToA ? rows : cols;
  if (kPartIdx < 0 || kPartLoop <= kPartIdx || kPart <= 0 ||
      kPart > ArchTag::L1_SIZE / sizeof(T))
    trap();
  if (kPartLoop > (srcK + kPart - 1) / kPart || kPart % 16 != 0)
    trap();
  if (kPartActual <= 0 || kPartActual > kPart ||
      kPartCeil != CEIL_FACTOR(kPartActual, 16) ||
      (kPartIdx != kPartLoop - 1 && kPartActual != kPart))
    trap();
  if (mn <= 0 || mn % 16 != 0 || mn > srcMN)
    trap();
  if (kPartIdx > srcK / kPart || kPartCeil > srcK - kPartIdx * kPart)
    trap();
  if (kPartCeil > dst->sizes[0] * c0 || mn > dst->sizes[1] * 16)
    trap();

  using SrcTag = std::conditional_t<Transposed, layout::nZ, layout::zN>;
  using DstTag = std::conditional_t<ToA, layout::zN, layout::nZ>;
  using DstPosition =
      std::conditional_t<ToA, Arch::PositionL0A, Arch::PositionL0B>;
  auto srcLayout = tla::MakeLayout<T, SrcTag>(rows, cols);
  if constexpr (Transposed)
    srcLayout.template stride<0, 1>() = src->strides[0];
  else
    srcLayout.template stride<1, 1>() = src->strides[0];

  // Size the copy from this tile, but retain the destination parent's outer
  // stride. This is essential for a tail copied into a larger reusable slot.
  uint32_t tileRows = ToA ? mn : kPartCeil;
  uint32_t tileCols = ToA ? kPartCeil : mn;
  auto dstLayout = tla::MakeLayout<T, DstTag>(tileRows, tileCols);
  if constexpr (ToA)
    dstLayout.template stride<1, 1>() = dst->strides[0];
  else
    dstLayout.template stride<0, 1>() = dst->strides[0];

  // LocalTensor describes the backing view, including gaps between outer
  // fractals. The layouts above, rather than the buffer span, select the tile.
  uint32_t srcSpan =
      (src->sizes[0] - 1) * src->strides[0] + src->sizes[1] * 16 * c0;
  uint32_t dstSpan =
      (dst->sizes[0] - 1) * dst->strides[0] + dst->sizes[1] * 16 * c0;
  AscendCBisheng::LocalTensor<T> srcData{AscendCBisheng::TPosition::A1, srcAddr,
                                         srcSpan};
  AscendCBisheng::LocalTensor<T> dstData{DstPosition::value, dstAddr, dstSpan};
  auto srcTensor = tla::MakeTensor(srcData, srcLayout, Arch::PositionL1{});
  uint32_t kOffset = kPartIdx * kPart;
  auto srcTile = tla::GetTile(
      srcTensor, tla::MakeCoord(ToA ? 0u : kOffset, ToA ? kOffset : 0u),
      tla::MakeShape(tileRows, tileCols));
  auto dstTensor = tla::MakeTensor(dstData, dstLayout, DstPosition{});

  // Match the dispatch types used by LocalMmad.cpp. TileCopyTla's call
  // operator accepts the actual MakeTensor/GetTile objects independently.
  using SrcTensor =
      tla::Tensor<AscendCBisheng::LocalTensor<T>,
                  detail::TagToLayout_t<T, SrcTag>,
                  tla::Coord<tla::_0, tla::_0>, AscendCBisheng::TPosition::A1>;
  using DstTensor =
      tla::Tensor<AscendCBisheng::LocalTensor<T>,
                  detail::TagToLayout_t<T, DstTag>,
                  tla::Coord<tla::_0, tla::_0>, DstPosition::value>;
  Gemm::Tile::TileCopyTla<ArchTag, SrcTensor, DstTensor> copy;
  copy(dstTensor, srcTile);
}

} // namespace

// Same C-interface ABI and names as membase: the target selects c310 bitcode.
// Rank-4 descriptors precede the six i64 fields; transpose is in the symbol.
#define REGISTER_L12L0(type, scope, toA, transposed, suffix)                   \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_l12l0_cbuf_to_##scope##_##type##suffix(                     \
          memref_t<__cbuf__ type, 4> *src,                                     \
          memref_t<__##scope##__ type, 4> *dst, int64_t kPartIdx,              \
          int64_t kPart, int64_t kPartCeil, int64_t kPartLoop,                 \
          int64_t kPartActual, int64_t mn) {                                   \
    copyL12L0<type, __##scope##__ type, toA, transposed>(                      \
        src, dst, kPartIdx, kPart, kPartCeil, kPartLoop, kPartActual, mn);     \
  }

#define REGISTER_L12L0_TYPE(type)                                              \
  REGISTER_L12L0(type, ca, true, false, )                                      \
  REGISTER_L12L0(type, ca, true, true, _transpose)                             \
  REGISTER_L12L0(type, cb, false, false, )                                     \
  REGISTER_L12L0(type, cb, false, true, _transpose)

extern "C" {
REGISTER_L12L0_TYPE(half)
REGISTER_L12L0_TYPE(bfloat16_t)
REGISTER_L12L0_TYPE(float)
}

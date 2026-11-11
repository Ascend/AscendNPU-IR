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
#include "catlass/gemm/tile/copy_l1_to_bt_a5.hpp"
#include "tla/tensor.hpp"

namespace {

// This is the A5 ABI, not the membase descriptor/descriptor/index ABI:
// src is a rank-1 cbuf descriptor; dstByteAddress is the effective BT byte
// address (including dst's element offset); n is an explicit signed i64.
template <typename SrcT, typename DstT>
__aicore__ __attribute__((always_inline)) void
copyL12BT(memref_t<__cbuf__ SrcT, 1> *src, uint64_t dstByteAddress, int64_t n) {
  using namespace Catlass;
  using ArchTag = Arch::AtlasA5;
  static_assert(
      (std::is_same_v<DstT, float> &&
       (std::is_same_v<SrcT, half> || std::is_same_v<SrcT, bfloat16_t> ||
        std::is_same_v<SrcT, float>)) ||
      (std::is_same_v<SrcT, int32_t> && std::is_same_v<DstT, int32_t>));
  constexpr int64_t maxN = ArchTag::BIAS_SIZE / sizeof(DstT);
  constexpr int64_t srcCapacity = ArchTag::L1_SIZE / sizeof(SrcT);
  if (n <= 0 || n > maxN)
    trap();
  // B16: ceil(n / 16) source blocks, converted to B32 in BT.
  // B32: ceil(n / 8) rounded to an even number of source blocks.
  // Both transfer ceil(n / 16) * 16 elements, not merely n elements.
  const uint32_t paddedN = CEIL_FACTOR(n, 16);
  // At maxN=1024, blockLen is at most 128 (B32) or 64 (B16),
  // safely below DataCopyParams' uint16_t limit. blockCount=1, both gaps=0.
  if (src->strides[0] != 1 || src->sizes[0] < paddedN ||
      src->sizes[0] > srcCapacity || src->offset < 0 ||
      src->offset > srcCapacity - src->sizes[0] ||
      src->offset % (32 / sizeof(SrcT)) != 0)
    trap();
  const uint64_t srcBase = reinterpret_cast<uint64_t>(src->aligned);
  const uint64_t srcOffsetBytes = src->offset * sizeof(SrcT);
  if (srcBase > ArchTag::L1_SIZE || srcOffsetBytes > ArchTag::L1_SIZE - srcBase)
    trap();
  const uint64_t srcByteAddress = srcBase + srcOffsetBytes;
  if (srcByteAddress % 32 != 0 ||
      src->sizes[0] * sizeof(SrcT) > ArchTag::L1_SIZE - srcByteAddress ||
      dstByteAddress > ArchTag::BIAS_SIZE || dstByteAddress % 64 != 0 ||
      paddedN * sizeof(DstT) > ArchTag::BIAS_SIZE - dstByteAddress)
    trap();

  // Check the full address before LocalTensor narrows to uint32_t. No BT
  // allocation, base selection, event handling or ping-pong occurs here.
  AscendCBisheng::LocalTensor<SrcT> srcData{
      AscendCBisheng::TPosition::A1, static_cast<uint32_t>(srcByteAddress),
      paddedN};
  AscendCBisheng::LocalTensor<DstT> dstData{
      AscendCBisheng::TPosition::C2, static_cast<uint32_t>(dstByteAddress),
      paddedN};
  auto vectorLayout = tla::MakeLayout(static_cast<uint32_t>(n));
  auto srcTensor = tla::MakeTensor(srcData, vectorLayout, Arch::PositionL1{});
  auto dstTensor = tla::MakeTensor(dstData, vectorLayout, Arch::PositionBias{});
  // Use the same CATLASS dispatch tags as LocalMmad; the call operator takes
  // the actual MakeTensor objects. VectorLayout encodes unit element stride.
  using SrcTensor =
      tla::Tensor<AscendCBisheng::LocalTensor<SrcT>,
                  detail::TagToLayout_t<SrcT, layout::VectorLayout>,
                  tla::Coord<tla::_0>, AscendCBisheng::TPosition::A1>;
  using DstTensor =
      tla::Tensor<AscendCBisheng::LocalTensor<DstT>,
                  detail::TagToLayout_t<DstT, layout::VectorLayout>,
                  tla::Coord<tla::_0>, AscendCBisheng::TPosition::C2>;
  Gemm::Tile::TileCopyTla<ArchTag, SrcTensor, DstTensor> copy;
  copy(dstTensor, srcTensor);
}

} // namespace

#define REGISTER_L12BT(srcType, dstType)                                       \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_l12bt_a5_##srcType##_to_##dstType(                          \
          memref_t<__cbuf__ srcType, 1> *src, uint64_t dstByteAddress,         \
          int64_t n) {                                                         \
    copyL12BT<srcType, dstType>(src, dstByteAddress, n);                       \
  }

extern "C" {
REGISTER_L12BT(half, float)
REGISTER_L12BT(bfloat16_t, float)
REGISTER_L12BT(float, float)
REGISTER_L12BT(int32_t, int32_t)
}

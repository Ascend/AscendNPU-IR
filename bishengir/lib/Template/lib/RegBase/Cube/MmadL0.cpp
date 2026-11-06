//===- MmadL0.cpp - Standalone A5 CATLASS MMAD
//-----------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// This translation unit is included only in the A5/c310 AIC library.
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 3510 && __NPU_ARCH__ != 5102)
#error "Standalone MmadL0 requires an A5 compiler target"
#endif
#ifndef CATLASS_ARCH_A5_ENABLED
#define CATLASS_ARCH_A5_ENABLED
#endif

#include "Utils.h"
#include "ascendc_bisheng/ascendc_bisheng.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/tensor.hpp"

namespace {

template <typename T, typename QualifiedT>
__aicore__ __attribute__((always_inline)) uint32_t
checkedTileAddress(memref_t<QualifiedT, 4> *ref, int64_t outer, int64_t inner,
                   uint32_t capacityBytes, uint32_t alignmentBytes) {
  // The caller has bounded outer/inner before reaching this helper. Use
  // aligned, not allocated, and apply the element offset exactly once.
  const int64_t elements = outer * inner * 256;
  const int64_t capacity = capacityBytes / sizeof(T);
  if (ref->sizes[0] != outer || ref->sizes[1] != inner || ref->sizes[2] != 16 ||
      ref->sizes[3] != 16)
    trap();
  if (ref->strides[0] != inner * 256 || ref->strides[1] != 256 ||
      ref->strides[2] != 16 || ref->strides[3] != 1)
    trap();
  if (elements > capacity)
    trap();
  if (ref->offset < 0 || ref->offset > capacity - elements)
    trap();
  if (ref->offset % (alignmentBytes / sizeof(T)) != 0)
    trap();

  const uint64_t base = reinterpret_cast<uint64_t>(ref->aligned);
  const uint64_t offsetBytes = ref->offset * sizeof(T);
  if (base > capacityBytes || offsetBytes > capacityBytes - base)
    trap();
  const uint64_t address = base + offsetBytes;
  if (address % alignmentBytes != 0 ||
      elements * sizeof(T) > capacityBytes - address)
    trap();
  return static_cast<uint32_t>(address);
}

template <typename T, bool WithBias>
__aicore__ __attribute__((always_inline)) void
mmadL0(memref_t<__ca__ T, 4> *a, memref_t<__cb__ T, 4> *b,
       memref_t<__cc__ float, 4> *c, uint64_t biasByteAddress, int64_t m,
       int64_t k, int64_t n, int64_t clear) {
  using namespace Catlass;
  using ArchTag = Arch::AtlasA5;
  // The independent API deliberately supports only full 16-element blocks.
  // Bound before multiplication and narrowing to MmadParams/LocalTensor.
  if (m <= 0 || m > 2048 || k <= 0 || k > 2048 || n <= 0 || n > 2048 ||
      m % 16 != 0 || k % 16 != 0 || n % 16 != 0 || (clear != 0 && clear != 1))
    trap();
  const uint32_t aAddress =
      checkedTileAddress<T>(a, k / 16, m / 16, ArchTag::L0A_SIZE, 512);
  const uint32_t bAddress =
      checkedTileAddress<T>(b, k / 16, n / 16, ArchTag::L0B_SIZE, 512);
  const uint32_t cAddress =
      checkedTileAddress<float>(c, n / 16, m / 16, ArchTag::L0C_SIZE, 1024);
  const uint32_t rows = static_cast<uint32_t>(m);
  const uint32_t cols = static_cast<uint32_t>(n);
  const uint32_t reduction = static_cast<uint32_t>(k);
  AscendCBisheng::LocalTensor<T> aData{AscendCBisheng::TPosition::A2, aAddress,
                                       rows * reduction};
  AscendCBisheng::LocalTensor<T> bData{AscendCBisheng::TPosition::B2, bAddress,
                                       reduction * cols};
  AscendCBisheng::LocalTensor<float> cData{AscendCBisheng::TPosition::CO1,
                                           cAddress, rows * cols};
  auto aTensor =
      tla::MakeTensor(aData, tla::MakeLayout<T, layout::zN>(rows, reduction),
                      Arch::PositionL0A{});
  auto bTensor =
      tla::MakeTensor(bData, tla::MakeLayout<T, layout::nZ>(reduction, cols),
                      Arch::PositionL0B{});
  auto cTensor = tla::MakeTensor(cData, tla::MakeLayoutL0C(rows, cols),
                                 Arch::PositionL0C{});
  Gemm::Tile::TileMmadTla<ArchTag, T, layout::zN> tileMmad;
  if constexpr (WithBias) {
    if (biasByteAddress > ArchTag::BIAS_SIZE || biasByteAddress % 64 != 0 ||
        n * sizeof(float) > ArchTag::BIAS_SIZE - biasByteAddress)
      trap();
    AscendCBisheng::LocalTensor<float> biasData{
        AscendCBisheng::TPosition::C2, static_cast<uint32_t>(biasByteAddress),
        cols};
    auto biasTensor =
        tla::MakeTensor(biasData, tla::MakeLayout(cols), Arch::PositionBias{});
    tileMmad(cTensor, aTensor, bTensor, biasTensor, rows, cols, reduction,
             /*initC=*/false, /*unitFlag=*/uint8_t(0));
  } else {
    tileMmad(cTensor, aTensor, bTensor, rows, cols, reduction,
             /*initC=*/clear != 0, /*unitFlag=*/uint8_t(0));
  }
  // TileMmadTla owns the small-tile PIPE_M barrier. No additional barriers,
  // K loops, copies, allocations or slot selection belong in this wrapper.
}

} // namespace

// Rank-4 descriptors with i64 metadata, followed by explicit i64 scalars.
// BiasBUF has no CCE pointer qualifier; its ABI value is an effective byte
// address. Distinct names prevent accidental resolution to another ABI.
extern "C" {
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_mmadl0_a5_half_to_float(memref_t<__ca__ half, 4> *a,
                                     memref_t<__cb__ half, 4> *b,
                                     memref_t<__cc__ float, 4> *c, int64_t m,
                                     int64_t k, int64_t n, int64_t clear) {
  mmadL0<half, false>(a, b, c, 0, m, k, n, clear);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_mmadl0_a5_half_to_float_bias(memref_t<__ca__ half, 4> *a,
                                          memref_t<__cb__ half, 4> *b,
                                          memref_t<__cc__ float, 4> *c,
                                          uint64_t biasByteAddress, int64_t m,
                                          int64_t k, int64_t n) {
  mmadL0<half, true>(a, b, c, biasByteAddress, m, k, n, 0);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_mmadl0_a5_bfloat16_t_to_float(memref_t<__ca__ bfloat16_t, 4> *a,
                                           memref_t<__cb__ bfloat16_t, 4> *b,
                                           memref_t<__cc__ float, 4> *c,
                                           int64_t m, int64_t k, int64_t n,
                                           int64_t clear) {
  mmadL0<bfloat16_t, false>(a, b, c, 0, m, k, n, clear);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_mmadl0_a5_bfloat16_t_to_float_bias(
    memref_t<__ca__ bfloat16_t, 4> *a, memref_t<__cb__ bfloat16_t, 4> *b,
    memref_t<__cc__ float, 4> *c, uint64_t biasByteAddress, int64_t m,
    int64_t k, int64_t n) {
  mmadL0<bfloat16_t, true>(a, b, c, biasByteAddress, m, k, n, 0);
}
}

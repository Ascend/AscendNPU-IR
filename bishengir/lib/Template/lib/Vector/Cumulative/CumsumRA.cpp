/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "DMA/DMAUtils.h"
#include "Utils.h"
#include "Vector/Cumulative/CumsumUtils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_cumsum_ra(memref_t<__ubuf__ T, 2> *src,
                          memref_t<__ubuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_ptr = src->aligned + src->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert(isSizeAlignedToBlock<T>(src->strides[0]) &&
         "The src strides[0] must be aligned to block.");
  assert((src->strides[1] == 1 && dst->strides[1] == 1) &&
         "The src/dst last dim must be continuous.");
#endif
}

/// cumsum ra op description:
/// Returns the cumulative sum of elements of input in the first axis.
///
/// \param src (type: memref<a x b x T>)
/// \param dst (type: memref<a x b x T>)
///
/// Constraints:
/// 1. cumsum ra op supports int16_t, int32_t, float16 and float32 types.
/// 2. cumsum ra op only accepts 2d type memrefs as src and dst.
/// 3. r axis should be aligned to ub_block_unit.
/// 4. a axis should be continuous.
/// 5. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  // check type T, need to be int16_t, int32_t, float16 or float32
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value ||
          std::is_same<T, float>::value || std::is_same<T, half>::value,
      "cumsum ra op only support int16_t, int32_t, float16 and float32 type "
      "operands in template!");

  // Input parameter constraints assert.
  check_inputs_of_cumsum_ra(src, dst);

  // dst[0] = src[0]
  // for i = 1 to src->sizes[0]
  //     dst[i] = dst[i - 1] + src[i]
  // step1: vcopy(dst, src, src->sizes[1])
  memref_t<__ubuf__ T, 1> src_1d{src->allocated,
                                 src->aligned,
                                 src->offset,
                                 {src->sizes[1]},
                                 {src->strides[1]}};
  memref_t<__ubuf__ T, 1> dst_1d{dst->allocated,
                                 dst->aligned,
                                 dst->offset,
                                 {dst->sizes[1]},
                                 {dst->strides[1]}};
  copy_ubuf_to_ubuf_1d_core(&src_1d, &dst_1d);
  INTRINSIC(pipe_barrier, PIPE_V);

  // step2: for i = 1 to src->sizes[0]
  //            vadd(dst + i * dst->strides[0], src + i * src->strides[0], dst +
  //            (i - 1) * dst->strides[0], src->sizes[1])
  memref_t<__ubuf__ T, 1> last_dst_1d = dst_1d;
  for (int64_t i = 1; i < src->sizes[0]; ++i) {
    dst_1d.offset += dst->strides[0];
    src_1d.offset += src->strides[0];
    vector_eltwise_vv_1d<VectorOpTy::VADD, T>(&last_dst_1d, &src_1d, &dst_1d);
    last_dst_1d.offset += dst->strides[0];
    INTRINSIC(pipe_barrier, PIPE_V);
  }
  INTRINSIC_NO_ARGS(set_mask_norm);
}

extern "C" {
//===-------------------------------------------------------------------===//
// cumsum ra, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_CUMSUM(2, int16_t)
REGISTE_CUMSUM(2, int32_t)
REGISTE_CUMSUM(2, half)
REGISTE_CUMSUM(2, float)
}
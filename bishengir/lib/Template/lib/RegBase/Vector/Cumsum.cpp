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
#include "RegBase/Cumulative/Cumulative.h"
#include "Utils.h"
#include "RegBase/VecUtils.h"
#include "Vector/VecUtils.h"
 
#if defined(__DAV_C310__)
 
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
  assert((isSizeAlignedToBlock<T>(src->strides[0]) &&
          isSizeAlignedToBlock<T>(dst->strides[0])) &&
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
/// 1. cumsum ra op supports uint8/16/32/64_t, int8/16/32/64_t, float16/32 and
/// bfloat16 type.
/// 2. cumsum ra op only accepts 2d type memrefs as src and dst.
/// 3. r axis should be aligned to ub_block_unit.
/// 4. a axis should be continuous.
/// 5. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
template <typename T>
__aiv__ __attribute__((always_inline)) void
vector_cumsum_ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst, bool reverse) {
  // check type T, need to be int16_t, int32_t, float16 or float32
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value ||
          std::is_same<T, uint16_t>::value ||
          std::is_same<T, uint32_t>::value || std::is_same<T, float>::value ||
          std::is_same<T, half>::value || std::is_same<T, int8_t>::value ||
          std::is_same<T, bfloat16_t>::value || std::is_same<T, uint8_t>::value,
      "cumprod ra op only uint8/16/32/64_t, int8/16/32/64_t, float16/32 and "
      "bfloat16 type "
      "operands in template!");
 
  // Input parameter constraints assert.
  check_inputs_of_cumsum_ra(src, dst);
 
  if (src->sizes[0] <= 0 || dst->sizes[0] <= 0) {
    return;
  }
 
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
  if(!reverse) {
    copy_ubuf_to_ubuf_1d_core(&src_1d, &dst_1d);
  } else {
    memref_t<__ubuf__ T, 1> src_1d_tmp = src_1d;
    src_1d_tmp.offset += (src->sizes[0] - 1) * src->strides[0];
    copy_ubuf_to_ubuf_1d_core(&src_1d_tmp, &dst_1d);
  }
  
  // step2: for i = 1 to src->sizes[0]
  //            vadd(dst + i * dst->strides[0], src + i * src->strides[0], dst +
  //            (i - 1) * dst->strides[0], src->sizes[1])
  memref_t<__ubuf__ T, 1> last_dst_1d = dst_1d;
  cumsum_2d<T, T>(
    cumulative_args<T, T>{&dst_1d, {&src_1d, &last_dst_1d}, 
                          dst->strides[0], {src->strides[0], dst->strides[0]},
                          src->sizes[0], reverse}
  );
}
 
template <typename SRC_TYPE, typename DST_TYPE>
__aiv__ __attribute__((always_inline)) void
cumsum_2d(cumulative_args<SRC_TYPE, DST_TYPE> args) {
  auto dst_offset = args.dst->offset;
  auto src0_offset = args.src[0]->offset;
  auto src1_offset = args.src[1]->offset;
  auto dst_aligned = args.dst->aligned;
  auto src0_aligned = args.src[0]->aligned;
  auto src1_aligned = args.src[1]->aligned;
  uint32_t size0 = args.src[0]->sizes[0];
  auto dst_stride = args.dst_stride;
  auto src0_stride = args.src_stride[0];
  auto src1_stride = args.src_stride[1];
  uint16_t src_size = (uint16_t)args.src_size;
  constexpr int num_per_reg = REG_REGISTER_SIZE / sizeof(SRC_TYPE);
  uint16_t repeat_times = CEIL_DIV(size0, num_per_reg);
  __VEC_SCOPE__ {
    for (uint16_t i = 0; i < repeat_times; i++) {
      VectorReg<SRC_TYPE> src0_reg, src1_reg;
      VectorReg<DST_TYPE> dst_reg;
      vector_bool full_mask;
      CREATE_MASK_BY_SIZE(full_mask, SRC_TYPE, size0);
      vlds(src1_reg, src1_aligned + src1_offset, i * num_per_reg, NORM);
      for (uint16_t j = 1; j < src_size; j++) {
        auto dst_offset_tmp = dst_offset + j * dst_stride;
        int src1_steps = args.reverse ? (src_size -1 - j) : j;
        auto src0_offset_tmp = src0_offset + src1_steps * src0_stride;
        vlds(src0_reg, src0_aligned + src0_offset_tmp, i * num_per_reg, NORM);
        vadd(dst_reg, src0_reg, src1_reg, full_mask, MODE_ZEROING);
        vsts(dst_reg, dst_aligned + dst_offset_tmp, i * num_per_reg, NORM_B32, full_mask);
        src1_reg = dst_reg;
      }
    }
  }
}
 
extern "C" {
//===-------------------------------------------------------------------===//
// cumsum ra, 2 dim
//===-------------------------------------------------------------------===//
REGISTER_CUMSUM(int8_t) { vector_cumsum_ra<int8_t>(src, dst, reverse); }
REGISTER_CUMSUM(uint8_t) { vector_cumsum_ra<uint8_t>(src, dst, reverse); }
REGISTER_CUMSUM(int16_t) { vector_cumsum_ra<int16_t>(src, dst, reverse); }
REGISTER_CUMSUM(uint16_t) { vector_cumsum_ra<uint16_t>(src, dst, reverse); }
REGISTER_CUMSUM(int32_t) { vector_cumsum_ra<int32_t>(src, dst, reverse); }
REGISTER_CUMSUM(uint32_t) { vector_cumsum_ra<uint32_t>(src, dst, reverse); }
REGISTER_CUMSUM(half) { vector_cumsum_ra<half>(src, dst, reverse); }
REGISTER_CUMSUM(float) { vector_cumsum_ra<float>(src, dst, reverse); }
REGISTER_CUMSUM(bfloat16_t) { vector_cumsum_ra<bfloat16_t>(src, dst, reverse); }
}
#endif
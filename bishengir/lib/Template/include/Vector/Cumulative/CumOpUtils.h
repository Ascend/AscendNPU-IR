#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_CUM_OP_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_CUM_OP_UTILS_H

#pragma once
#include "DMA/DMAUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void static_assert_supported_type() {
  static_assert(
      std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value ||
          std::is_same<T, float>::value || std::is_same<T, half>::value,
      "cum* op only supports int16_t, int32_t, float16 and float32 types");
}

template <typename T, size_t Rank>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_cumop(memref_t<__ubuf__ T, Rank> *src,
                      memref_t<__ubuf__ T, Rank> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_ptr = src->aligned + src->offset;

  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");

  if constexpr (Rank >= 2) {
    assert(isSizeAlignedToBlock<T>(src->strides[0]) &&
           "The src strides[0] must be aligned to block.");
  }
  if constexpr (Rank >= 2) {
    assert((src->strides[Rank - 1] == 1 && dst->strides[Rank - 1] == 1) &&
           "The src/dst last dim must be continuous.");
  }
#endif
}

template <VectorOpTy Op, typename T, bool reverse = false>
__aiv__ __attribute__((always_inline)) void
vector_cum_op_ra(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  // -----------------------------------------------------------------
  //   Copy the first row (dst = src)
  // -----------------------------------------------------------------
  if constexpr (!reverse) {
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

    // -----------------------------------------------------------------
    //   Row‑wise cumulative operation
    // -----------------------------------------------------------------
    memref_t<__ubuf__ T, 1> last_dst_1d = dst_1d;
    for (int64_t i = 1; i < src->sizes[0]; ++i) {
      dst_1d.offset += dst->strides[0];
      src_1d.offset += src->strides[0];
      vector_eltwise_vv_1d<Op, T>(&last_dst_1d, &src_1d, &dst_1d);
      last_dst_1d.offset += dst->strides[0];
      INTRINSIC(pipe_barrier, PIPE_V);
    }
  } else { // reverse = true
    memref_t<__ubuf__ T, 1> src_1d{src->allocated,
                                   src->aligned,
                                   src->offset +
                                       src->strides[0] * (src->sizes[0] - 1),
                                   {src->sizes[1]},
                                   {src->strides[1]}};
    memref_t<__ubuf__ T, 1> dst_1d{dst->allocated,
                                   dst->aligned,
                                   dst->offset +
                                       dst->strides[0] * (src->sizes[0] - 1),
                                   {dst->sizes[1]},
                                   {dst->strides[1]}};
    copy_ubuf_to_ubuf_1d_core(&src_1d, &dst_1d);
    INTRINSIC(pipe_barrier, PIPE_V);

    memref_t<__ubuf__ T, 1> last_dst_1d = dst_1d;

    for (int64_t i = 1; i < src->sizes[0]; ++i) {
      dst_1d.offset -= dst->strides[0];
      src_1d.offset -= src->strides[0];
      vector_eltwise_vv_1d<Op, T>(&last_dst_1d, &src_1d, &dst_1d);
      last_dst_1d.offset -= dst->strides[0];
      INTRINSIC(pipe_barrier, PIPE_V);
    }
  }
}

template <VectorOpTy Op, typename T, bool reverse = false>
__aiv__ __attribute__((always_inline)) void
vector_cum_op_ara(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst) {

  constexpr int element_size = sizeof(T);
  constexpr int elements_per_block = INTR_BYTES_PER_BLOCK / element_size;
  const uint32_t copy_block_count = CEIL_DIV(src->sizes[2], elements_per_block);

  if constexpr (!reverse) {
    // -----------------------------------------------------------------
    //   Copy the first row (dst = src)
    // -----------------------------------------------------------------
    __ubuf__ T *src_base = src->aligned + src->offset;
    __ubuf__ T *dst_base = dst->aligned + dst->offset;

    copy_ubuf_to_ubuf_intrin_core<T>(dst_base, src_base, src->sizes[0],
                                     copy_block_count, src->strides[0],
                                     dst->strides[0]);
    INTRINSIC(pipe_barrier, PIPE_V);

    // -----------------------------------------------------------------
    //   Row‑wise cumulative operation
    // -----------------------------------------------------------------
    memref_t<__ubuf__ T, 2> cur_src{src->allocated,
                                    src->aligned,
                                    src->offset,
                                    {src->sizes[0], src->sizes[2]},
                                    {src->strides[0], src->strides[2]}};

    memref_t<__ubuf__ T, 2> cur_dst{dst->allocated,
                                    dst->aligned,
                                    dst->offset,
                                    {dst->sizes[0], dst->sizes[2]},
                                    {dst->strides[0], dst->strides[2]}};

    memref_t<__ubuf__ T, 2> last_dst = cur_dst;

    for (int64_t i = 1; i < src->sizes[1]; ++i) {
      cur_dst.offset += dst->strides[1];
      cur_src.offset += src->strides[1];

      vector_eltwise_2d<Op, T, 2>(
          /* src0 */ &last_dst,
          /* src1 */ &cur_src,
          /* scalar */ 0,
          /* dst */ &cur_dst,
          /* mode */ VectorLastAxisMode::VV);

      last_dst.offset += dst->strides[1];
      INTRINSIC(pipe_barrier, PIPE_V);
    }
  } else { // reverse = true
    __ubuf__ T *src_base =
        src->aligned + src->offset + src->strides[1] * (src->sizes[1] - 1);
    __ubuf__ T *dst_base =
        dst->aligned + dst->offset + src->strides[1] * (src->sizes[1] - 1);

    copy_ubuf_to_ubuf_intrin_core<T>(dst_base, src_base, src->sizes[0],
                                     copy_block_count, src->strides[0],
                                     dst->strides[0]);
    INTRINSIC(pipe_barrier, PIPE_V);

    memref_t<__ubuf__ T, 2> cur_src{src->allocated,
                                    src->aligned,
                                    src->offset +
                                        dst->strides[1] * (src->sizes[1] - 1),
                                    {src->sizes[0], src->sizes[2]},
                                    {src->strides[0], src->strides[2]}};

    memref_t<__ubuf__ T, 2> cur_dst{dst->allocated,
                                    dst->aligned,
                                    dst->offset +
                                        dst->strides[1] * (src->sizes[1] - 1),
                                    {dst->sizes[0], dst->sizes[2]},
                                    {dst->strides[0], dst->strides[2]}};

    memref_t<__ubuf__ T, 2> last_dst = cur_dst;

    for (int64_t i = 1; i < src->sizes[1]; ++i) {
      cur_dst.offset -= dst->strides[1];
      cur_src.offset -= src->strides[1];

      vector_eltwise_2d<Op, T, 2>(
          /* src0 */ &last_dst,
          /* src1 */ &cur_src,
          /* scalar */ 0,
          /* dst */ &cur_dst,
          /* mode */ VectorLastAxisMode::VV);

      last_dst.offset -= dst->strides[1];
      INTRINSIC(pipe_barrier, PIPE_V);
    }
  }
}

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_CUM_OP_UTILS_H

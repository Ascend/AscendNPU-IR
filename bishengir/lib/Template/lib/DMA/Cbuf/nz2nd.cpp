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

#include "DMA/NZ2ND.h"

template <typename T>
__aicore__ __attribute__((always_inline)) void
check_inputs_of_copy_cbuf_to_gm_4d_to_2d_core(memref_t<__cbuf__ T, 4> *src,
                                              memref_t<__gm__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride3_src = src->strides[3];
  assert((stride3_src == 1) && "Last dimension of src must be contiguous.");
#endif
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
check_inputs_of_copy_cbuf_to_gm_5d_to_3d_core(memref_t<__cbuf__ T, 5> *src,
                                              memref_t<__gm__ T, 3> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  const int64_t stride4_src = src->strides[4];
  assert((stride4_src == 1) && "Last dimension of src must be contiguous.");
#endif
}

template <typename T>
__aicore__ __attribute__((always_inline)) __gm__ T *
convert_cbuf_array_to_gm(__cbuf__ T *ptr, __gm__ T *tmp, size_t count) {
  __cbuf__ uint8_t *aligned_ptr = (__cbuf__ uint8_t *)((uint64_t)ptr & ~0xF);
  const uint32_t offset = (uint64_t)ptr - (uint64_t)aligned_ptr;
  const int64_t unit_size = 32;
  const int64_t total_bytes = (int64_t)(sizeof(T) * count) + offset;
  const int64_t chunks = total_bytes / unit_size;

  if (chunks != 0) {
    INTRINSIC(pipe_barrier, PIPE_ALL);
    INTRINSIC(dcci, tmp, 0);
    // Copy by aligned address without tail: data out of last 16-byte chunk are
    // lost.
    INTRINSIC(copy_cbuf_to_gm, tmp, aligned_ptr, 0, 1, chunks, 0, 0);
    INTRINSIC(pipe_barrier, PIPE_ALL);
  }
  return (__gm__ T *)((__gm__ uint8_t *)tmp +
                      offset); // pointer to actual copied data
};

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_cbuf_chunks_to_gm(__gm__ T *dst, __cbuf__ T *src, const uint16_t chunks,
                       const uint16_t count = 1, const uint16_t src_gap = 0,
                       const uint16_t dst_gap = 1) {
  INTRINSIC(pipe_barrier, PIPE_ALL);
  INTRINSIC(dcci, dst, 0);
  INTRINSIC(copy_cbuf_to_gm, dst, src, 0, count, chunks, src_gap, dst_gap);
  INTRINSIC(pipe_barrier, PIPE_ALL);
};

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_cbuf_to_gm_4d_to_2d_core_unaligned(memref_t<__cbuf__ T, 4> *src,
                                        memref_t<__gm__ T, 2> *dst) {
  check_inputs_of_copy_cbuf_to_gm_4d_to_2d_core(src, dst);

  __cbuf__ T *src_ptr = src->aligned + src->offset;
  __gm__ T *dst_ptr = dst->aligned + dst->offset;

  const int64_t max_chunk_size = 16;

  // Destination dimensions: [N, Z, H, W]
  if (src->strides[3] == 1 && src->sizes[3] <= max_chunk_size)
      [[likely]] {         // keep this because check is only on when
                           // ENABLE_CPU_TRACE_INTRINSIC
    T tmp[max_chunk_size]; // temporary array for first items
    int64_t tmp_size = 0;
    for (int64_t x = (dst->sizes[1] / src->sizes[3]) * src->sizes[3]; x >= 0;
         x = x - src->sizes[3]) {                        // destination column
      for (int64_t y = dst->sizes[0] - 1; y >= 0; y--) { // destination row
        // Search for destination's sub-row in source tensor
        int64_t dst_offset = y * dst->sizes[1] + x; // destination flat offset
        int64_t a = y / src->sizes[2];              // fractal row
        int64_t b = y - a * src->sizes[2];          // row in fractal
        int64_t c = x / src->sizes[3];              // fractal column
        int64_t chunk_size = // number of elements in sub-row
            c == src->sizes[0] - 1 ? dst->sizes[1] - x : src->sizes[3];
        int64_t src_offset =
            a * src->sizes[2] *
                src->sizes[3] + // Elements in fractals of the last column
            b * src->sizes[3] + // Elements in fractal's rows
            c * src->sizes[1] * src->sizes[2] *
                src->sizes[3]; // Elements in fractals columns
        // Use first bytes of dst->aligned as a buffer for copy
        __gm__ T *converted = convert_cbuf_array_to_gm(
            src_ptr + src_offset, dst->aligned, src->sizes[3]);
        for (int64_t i = chunk_size - 1; i >= 0; i--) {
          int64_t dst_address = dst_offset + i;
          if (dst_address < max_chunk_size) {
            // TODO: Use % to prevent out-of-bounds runtime error (direct
            // index sporadically exceeds max_chunk_size). The % can be deleted
            // after the Bisheng compiler resolves this issue.
            tmp[dst_address % max_chunk_size] = converted[i];
            if (tmp_size < dst_address + 1) {
              tmp_size = dst_address + 1;
            }
          } else {
            dst_ptr[dst_address] = converted[i];
          }
        }
      }
    }
    for (int64_t i = 0; i < tmp_size; i++) {
      dst_ptr[i] = tmp[i];
    }
  }
  // in other cases, the destination will not be updated and will contain
  // undefined contents
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_cbuf_to_gm_4d_to_2d_core_aligned(memref_t<__cbuf__ T, 4> *src,
                                      memref_t<__gm__ T, 2> *dst) {
  check_inputs_of_copy_cbuf_to_gm_4d_to_2d_core(src, dst);
  __cbuf__ T *src_ptr = src->aligned + src->offset;
  __gm__ T *dst_ptr = dst->aligned + dst->offset;

  const int64_t unit_size = 32; // 32 bytes minimum block size
  const int64_t fractal_columns = src->sizes[0];
  const int64_t fractal_rows = src->sizes[1];
  const int64_t fractal_height = src->sizes[2];
  const int64_t fractal_width = src->sizes[3];
  const int64_t dst_height = dst->sizes[0];
  const int64_t dst_width = dst->sizes[1];
  const int64_t chunk_size_bytes = fractal_width * sizeof(T);
  const uint16_t len = chunk_size_bytes / unit_size;

  // Cannot copy data if destination row is less then copy unit size
  if (dst_width * sizeof(T) < unit_size)
    return;

  // Check if the destination row length is 32-byte aligned
  const bool is_row_length_aligned = (dst_width * sizeof(T)) % unit_size == 0;

  if (is_row_length_aligned) {
    // CASE 1: Optimized code using stride-collapsed multi-row copies

    // Gap from the end of a row at (y) to the same row in the next fractal
    // block position (y + fractal_height) in 32 bytes
    const uint16_t dst_gap =
        (fractal_height * dst_width * sizeof(T)) / unit_size - len;

    // Loop over fractal columns
    for (int64_t x = 0; x < dst_width; x += fractal_width) {
      const int64_t c = x / fractal_width;
      if (c >= fractal_columns)
        continue; // Guard against out of bounds fractal columns

      // Loop over the inner row indices of the fractal (Dimension 2: 'b')
      // Rows of fractal in one fractals column that have the same index can be
      // copied by one intrinsic call
      for (int64_t b = 0; b < fractal_height; b++) {
        // Limit the outer fractal rows (Dimension 1: 'a') based on destination
        // matrix limits Each step of 'a' advances the global matrix row by
        // fractal_height
        const int64_t max_a_count = fractal_rows;
        if (b >= dst_height) {
          continue; // This particular row index is already out of destination
                    // bounds
        }

        // Calculate how many 'a' blocks can fit into the destination row
        // dimension
        const int64_t available_a_blocks =
            (dst_height - b + fractal_height - 1) / fractal_height;
        const int64_t a_count =
            available_a_blocks < max_a_count ? available_a_blocks : max_a_count;

        if (a_count <= 0)
          continue;

        // Calculate starting offsets for this specific fractal column 'c' and
        // row in fractal 'b'
        const int64_t src_offset =
          b * fractal_width +
          c * fractal_rows * fractal_height * fractal_width;
        const int64_t dst_offset = b * dst_width + x;

        __cbuf__ T *current_src = src_ptr + src_offset;
        __gm__ T *current_dst = dst_ptr + dst_offset;

        const bool is_last_fractals_column = c == (fractal_columns - 1);
        const uint16_t actual_len =
            is_last_fractals_column ? (dst_width - x) / unit_size : len;

        // Source gap calculations in units of 32 bytes
        // Gap from the end of a row in block 'a' to the start of the same row
        // in block 'a + 1' One full 'a' block span is (src->sizes[2] *
        // src->sizes[3] * sizeof(T)) bytes
        const uint16_t src_gap =
            (fractal_height * fractal_width * sizeof(T)) / unit_size -
            actual_len;

        // Direct transfer from CBUF to GM
        copy_cbuf_chunks_to_gm(current_dst, current_src, actual_len, a_count,
                               src_gap, dst_gap);
      }
    }
  } else {
    // CASE 2: Unaligned row length / partial last column handling

    // Process last chunk of data of the last fractal row in source fractal.
    // Copying is done through first 32 bytes memory of destination to avoid
    // overflow of allocated destination memory partition.
    {
      const int64_t last_dst_row = dst_height - 1;
      const int64_t last_dst_column = dst_width - 1;
      const int64_t a = last_dst_row / fractal_height; // last fractal row
      const int64_t b =
          last_dst_row - a * fractal_height;             // row in last fractal
      const int64_t c = last_dst_column / fractal_width; // last fractal column
      const int64_t actual_chunk_size =
          dst_width -
          c * fractal_width; // actual number of elements in last row
      const uint16_t offset =
        (((actual_chunk_size * sizeof(T)) / unit_size) * unit_size) / sizeof(T);

      const int64_t src_offset =
          c * fractal_rows * fractal_height * fractal_width +
          a * fractal_height * fractal_width +
          b * fractal_width +
          offset;
      __cbuf__ T *last_chunk_src = src_ptr + src_offset;

      // Copy last data not aligned to 32 bytes to dst_ptr as a cache
      copy_cbuf_chunks_to_gm(dst_ptr, last_chunk_src, 1);

      const int64_t dst_offset =
          last_dst_row * dst_width + c * fractal_width + offset;
      __gm__ T *last_chunk_dst = dst_ptr + dst_offset;

      // Copy data from cache to final location in destination
      const int64_t last_chunk_size = actual_chunk_size - offset;
      for (int64_t i = 0; i < last_chunk_size; ++i) {
        last_chunk_dst[i] = dst_ptr[i];
      }
    }

    // Iterate from the last fractal's column position to the first
    for (int64_t x = ((dst_width - 1) / fractal_width) * fractal_width; x >= 0;
         x -= fractal_width) {
      const int64_t c = x / fractal_width; // fractal column
      if (c >= fractal_columns)
        continue; // Guard against out of bounds fractal columns

      const bool is_last_fractals_column = c == (fractal_columns - 1);
      // Actual number of elements to be copied
      const int64_t actual_chunk_size =
        is_last_fractals_column ? dst_width - x : fractal_width;
      // Number of chunks to be copied
      const uint16_t actual_len =
        is_last_fractals_column
          ? (actual_chunk_size * sizeof(T) + unit_size - 1) / unit_size
          : len;

      // Iterate through destination rows
      for (int64_t y = 0; y < dst_height; ++y) {
        const int64_t a = y / fractal_height;     // fractal row
        const int64_t b = y - a * fractal_height; // row in fractal

        if (a >= fractal_rows)
          continue; // Guard against out if bounds fractal rows

        const int64_t src_offset =
          a * fractal_height * fractal_width +
          b * fractal_width +
          c * fractal_rows * fractal_height * fractal_width;
        const int64_t dst_offset = y * dst_width + x;

        __cbuf__ T *current_src = src_ptr + src_offset;
        __gm__ T *current_dst = dst_ptr + dst_offset;

        // Last chunk of overall data is already transfered
        const uint16_t aligned_len =
          is_last_fractals_column && (y == dst_height - 1) ? actual_len - 1
                                                           : actual_len;
        copy_cbuf_chunks_to_gm(current_dst, current_src, aligned_len);
      }
    }
  }

  INTRINSIC(pipe_barrier, PIPE_ALL);
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_cbuf_to_gm_4d_to_2d_core(memref_t<__cbuf__ T, 4> *src,
                              memref_t<__gm__ T, 2> *dst) {
  __cbuf__ T *src_ptr = src->aligned + src->offset;

  uint64_t src_addr = reinterpret_cast<uint64_t>(src_ptr);
  size_t row_size_bytes = src->sizes[3] * sizeof(T);
  size_t dst_stride_bytes = dst->sizes[1] * sizeof(T);

  // Alignment verification:
  // 1. CBUF source pointer must be 32-byte aligned.
  // 2. Fractal block row size must be a multiple of 32 bytes.
  bool is_aligned = (src_addr % 32 == 0) && (row_size_bytes % 32 == 0);

  if (is_aligned) {
    copy_cbuf_to_gm_4d_to_2d_core_aligned(src, dst);
  } else {
    copy_cbuf_to_gm_4d_to_2d_core_unaligned(src, dst);
  }
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
copy_cbuf_to_gm_5d_to_3d_core(memref_t<__cbuf__ T, 5> *src,
                              memref_t<__gm__ T, 3> *dst) {
  check_inputs_of_copy_cbuf_to_gm_5d_to_3d_core(src, dst);

  int64_t size0 = src->sizes[0];
  int64_t size1 = src->sizes[1];
  int64_t size2 = src->sizes[2];
  int64_t size3 = src->sizes[3];
  int64_t size4 = src->sizes[4];
  int64_t stride0 = src->strides[0];
  int64_t stride1 = src->strides[1];
  int64_t stride2 = src->strides[2];
  int64_t stride3 = src->strides[3];
  int64_t stride4 = src->strides[4];
  for (int64_t i = 0; i < size0; i++) {
    memref_t<__cbuf__ T, 4> src_4d = {src->allocated,
                                      src->aligned,
                                      src->offset + i * stride0,
                                      {size1, size2, size3, size4},
                                      {stride1, stride2, stride3, stride4}};
    memref_t<__gm__ T, 2> dst_2d = {dst->allocated,
                                    dst->aligned,
                                    dst->offset + i * dst->strides[0],
                                    {dst->sizes[1], dst->sizes[2]},
                                    {dst->strides[1], dst->strides[2]}};
    copy_cbuf_to_gm_4d_to_2d_core(&src_4d, &dst_2d);
  }
}

extern "C" {
REGISTER_NZ2ND(float, 4, 2)
REGISTER_NZ2ND(bfloat16_t, 4, 2)
REGISTER_NZ2ND(half, 4, 2)
REGISTER_NZ2ND(int8_t, 4, 2)
REGISTER_NZ2ND(uint8_t, 4, 2)
REGISTER_NZ2ND(int16_t, 4, 2)
REGISTER_NZ2ND(uint16_t, 4, 2)
REGISTER_NZ2ND(int32_t, 4, 2)
REGISTER_NZ2ND(uint32_t, 4, 2)
REGISTER_NZ2ND(int64_t, 4, 2)
REGISTER_NZ2ND(uint64_t, 4, 2)

REGISTER_NZ2ND(float, 5, 3)
REGISTER_NZ2ND(bfloat16_t, 5, 3)
REGISTER_NZ2ND(half, 5, 3)
REGISTER_NZ2ND(int8_t, 5, 3)
REGISTER_NZ2ND(uint8_t, 5, 3)
REGISTER_NZ2ND(int16_t, 5, 3)
REGISTER_NZ2ND(uint16_t, 5, 3)
REGISTER_NZ2ND(int32_t, 5, 3)
REGISTER_NZ2ND(uint32_t, 5, 3)
REGISTER_NZ2ND(int64_t, 5, 3)
REGISTER_NZ2ND(uint64_t, 5, 3)
}

#include "Vector/Transpose/TransposeUtils.h"

/// transpose 2d with last axis op description:
/// transpose src (a, b) to dst (b, a),
///
/// \param src (type: memref<axbxT, strided<[c, 1]>>)
/// \param dst (type: memref<bxaxT, strided<[d, 1]>>)
/// 'a' and 'b' are size of src and dst, and 'c' is stride of src, 'd' is stride
/// of dst
///
/// Constraints:
/// 1. if data type is b8, b16 and b64, 'c' and 'd' should be aligned to
/// multiple of ub_block_unit
/// 2. if data type is b32, stride should satisfy the one of rules below,
///    1) 'c' is multiple of 16, and 'd' is multiple of 8, or
///    2) 'c' is multiple of 8, and 'd' is multiple of 16.
/// 3. if data type is b64, tmp_buf is required, and the size of it should be
/// 512xi64 which includes 2 tmp_buf, and single tmp_buf size is
/// VNCHWCONV_INTR_BYTES_PER_REPEAT / sizeof(int64_t) * 4

template <typename T>
__aiv__ __attribute__((always_inline)) void
set_va_register_args(MODE mode, int64_t src_buf_stride, int64_t *src_va_args,
                     int64_t dst_buf_stride, int64_t *dst_va_args) {
  static_assert((sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4) &&
                "Transpose can not support this type");
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  constexpr int64_t HALFNUM = TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT / 2;
  if constexpr (sizeof(T) == 1 || sizeof(T) == 2) {
    // calc b16 va args
    for (int64_t i = 0; i < TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT; ++i) {
      src_va_args[i] = i * src_buf_stride;
      dst_va_args[i] = i * dst_buf_stride;
    }
  } else if constexpr (sizeof(T) == 4) {
    // calc b32 va args
    if (mode == TRANSPOSE_MODE1) {
      // b32 TRANSPOSE_MODE1
      for (int64_t i = 0; i < TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT; ++i) {
        src_va_args[i] = i * src_buf_stride;
      }
      for (int64_t i = 0; i < HALFNUM; ++i) {
        dst_va_args[2 * i] = i * dst_buf_stride;
        dst_va_args[2 * i + 1] = i * dst_buf_stride + num_per_block;
      }
    } else if (mode == TRANSPOSE_MODE2) {
      // b32 TRANSPOSE_MODE2
      for (int64_t i = 0; i < HALFNUM; ++i) {
        src_va_args[i] = i * src_buf_stride;
        src_va_args[i + HALFNUM] = i * src_buf_stride + num_per_block;
      }
      for (int64_t i = 0; i < HALFNUM; ++i) {
        dst_va_args[2 * i] = i * dst_buf_stride;
        dst_va_args[2 * i + 1] = (i + HALFNUM) * dst_buf_stride;
      }
    }
  }
  return;
}

template <typename T, size_t GROUP = 0>
__aiv__ __attribute__((always_inline)) void
set_va_registers(__ubuf__ T *src_buf_ptr, int64_t *src_va_args,
                 __ubuf__ T *dst_buf_ptr, int64_t *dst_va_args) {
  static_assert(
      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8) &&
      "Transpose can not support this type");
  constexpr int64_t HALFNUM = TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT / 2;
  // set dst va registers
  uint64_t dst_va_reg_array_0[HALFNUM] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint64_t dst_va_reg_array_1[HALFNUM] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int64_t i = 0; i < HALFNUM; ++i) {
    dst_va_reg_array_0[i] =
        ((uint64_t)((__ubuf__ T *)dst_buf_ptr + dst_va_args[i]));
  }
  for (int64_t i = 0; i < HALFNUM; ++i) {
    dst_va_reg_array_1[i] =
        ((uint64_t)((__ubuf__ T *)dst_buf_ptr + dst_va_args[i + HALFNUM]));
  }

  // set src va registers
  uint64_t src_va_reg_array_0[HALFNUM] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint64_t src_va_reg_array_1[HALFNUM] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int64_t i = 0; i < HALFNUM; ++i) {
    src_va_reg_array_0[i] =
        ((uint64_t)((__ubuf__ T *)src_buf_ptr + src_va_args[i]));
  }
  for (int64_t i = 0; i < HALFNUM; ++i) {
    src_va_reg_array_1[i] =
        ((uint64_t)((__ubuf__ T *)src_buf_ptr + src_va_args[i + HALFNUM]));
  }

  if constexpr (GROUP == 0) {
    INTRINSIC(set_va_reg_sb, VA0, dst_va_reg_array_0);
    INTRINSIC(set_va_reg_sb, VA1, dst_va_reg_array_1);
    INTRINSIC(set_va_reg_sb, VA2, src_va_reg_array_0);
    INTRINSIC(set_va_reg_sb, VA3, src_va_reg_array_1);
  } else if constexpr (GROUP == 1) {
    INTRINSIC(set_va_reg_sb, VA4, dst_va_reg_array_0);
    INTRINSIC(set_va_reg_sb, VA5, dst_va_reg_array_1);
    INTRINSIC(set_va_reg_sb, VA6, src_va_reg_array_0);
    INTRINSIC(set_va_reg_sb, VA7, src_va_reg_array_1);
  } else {
    static_assert("unsupport");
  }

  return;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
scatter_vnchwconv_intr(int64_t intr_repeat_num, uint16_t dst_repeat_stride,
                       uint16_t src_repeat_stride) {
  static_assert((sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4) &&
                "Transpose can not support this type");
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  if constexpr (sizeof(T) == 1) {
    // for b8, transpose 4 times
    INTRINSIC(scatter_vnchwconv_b8, VA0, VA2, intr_repeat_num,
              dst_repeat_stride, src_repeat_stride, 0, 0);
    INTRINSIC(scatter_vnchwconv_b8, VA4, VA2, intr_repeat_num,
              dst_repeat_stride, src_repeat_stride, 0, 1);
    INTRINSIC(scatter_vnchwconv_b8, VA0, VA6, intr_repeat_num,
              dst_repeat_stride, src_repeat_stride, 1, 0);
    INTRINSIC(scatter_vnchwconv_b8, VA4, VA6, intr_repeat_num,
              dst_repeat_stride, src_repeat_stride, 1, 1);
  } else if constexpr (sizeof(T) == 2) {
    INTRINSIC(scatter_vnchwconv_b16, VA0, VA2, intr_repeat_num,
              dst_repeat_stride, src_repeat_stride);
  } else if constexpr (sizeof(T) == 4) {
    INTRINSIC(scatter_vnchwconv_b32, VA0, VA2, intr_repeat_num,
              dst_repeat_stride, src_repeat_stride);
  }

  return;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
gen_vnchwconv_core(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
                   int64_t tile_m, int64_t tile_n, MODE mode) {
  static_assert((sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4) &&
                "Transpose can not support this type");
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  int64_t src_size0 = src->sizes[0];
  int64_t src_size1 = src->sizes[1];
  int64_t src_stride0 = src->strides[0];
  int64_t dst_stride0 = dst->strides[0];

  bool is_loop_m = (src_size0 / tile_m) <= (src_size1 / tile_n);
  int64_t loop_num = is_loop_m ? src_size0 / tile_m : src_size1 / tile_n;
  int64_t intr_repeat_num = is_loop_m ? src_size1 / tile_n : src_size0 / tile_m;

  uint16_t dst_repeat_stride =
      is_loop_m ? tile_n * dst_stride0 / num_per_block : tile_m / num_per_block;
  // set dst_repeat_stride to 0 if intr_repeat_num == 1, otherwise there will be
  // error.
  dst_repeat_stride = (intr_repeat_num == 1) ? 0 : dst_repeat_stride;
  uint16_t src_repeat_stride =
      is_loop_m ? tile_n / num_per_block : tile_m * src_stride0 / num_per_block;
  // set src_repeat_stride to 0 if intr_repeat_num == 1, otherwise there will be
  // error.
  src_repeat_stride = (intr_repeat_num == 1) ? 0 : src_repeat_stride;

  int64_t dst_loop_offset = 0;
  int64_t src_loop_offset = 0;

  int64_t dst_va_args[TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int64_t src_va_args[TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  set_va_register_args<T>(mode, src_stride0, src_va_args, dst_stride0,
                          dst_va_args);
  for (int64_t loop_var = 0; loop_var < loop_num; ++loop_var) {
    dst_loop_offset =
        is_loop_m ? loop_var * tile_m : loop_var * tile_n * dst_stride0;
    src_loop_offset =
        is_loop_m ? loop_var * tile_m * src_stride0 : loop_var * tile_n;

    // set va0, va1, va2, va3 args
    set_va_registers<T>(src_ptr + src_loop_offset, src_va_args,
                        dst_ptr + dst_loop_offset, dst_va_args);

    if (sizeof(T) == 1) {
      // for b8, it uses two group va registers, so set for another group
      constexpr int64_t va_args_num = 16;
      int64_t src_va_gap = src_stride0 * va_args_num;
      int64_t dst_va_gap = dst_stride0 * va_args_num;

      // set va4, va5, va6, va7 args
      set_va_registers<T, 1>(
          src_ptr + src_loop_offset + src_va_gap, src_va_args,
          dst_ptr + dst_loop_offset + dst_va_gap, dst_va_args);
    }

    scatter_vnchwconv_intr<T>(intr_repeat_num, dst_repeat_stride,
                              src_repeat_stride);
  }
  return;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
check_inputs_of_vnchwconv_2d(memref_t<__ubuf__ T, 2> *src,
                             memref_t<__ubuf__ T, 2> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  assert(isAddress32ByteAligned(src_ptr) &&
         "The starting address of src must be 32byte aligned.");
  assert(isAddress32ByteAligned(dst_ptr) &&
         "The starting address of dst must be 32byte aligned.");
  assert((isSizeAlignedToBlock<T>(src->strides[0]) &&
          isSizeAlignedToBlock<T>(dst->strides[0])) &&
         "The src/dst strides[0] must be aligned to block.");
  assert((src->strides[1] == 1 && dst->strides[1] == 1) &&
         "The src/dst last dim must be continuous.");
  if (sizeof(T) == 4) {
    assert(((dst->strides[0] % 8 == 0 && src->strides[0] % 16 == 0) ||
            (dst->strides[0] % 16 == 0 && src->strides[0] % 8 == 0)) &&
           "In the b32 scenario, dst stride0 and src stride0 must have a"
           "multiple of 16 and a multiple of 8.");
  }
#endif
}

/// Transpose src (m, n) with stride [u, 1] to dst (n, m) with stride [v, 1].
/// If n < aligned(n, num_per_block) or m < aligned(m, num_per_block), the
/// intrin will transpose src(aligned(m, num_per_block), aligned(n,
/// num_per_block)) to dst(aligned(n, num_per_block), aligned(m, num_per_block))
/// \param src (type: memref<m x n xT, stride[u, 1]>)
/// \param dst (type: memref<n x m xT, stride[v, 1]>)
///
/// Constraints:
/// 1. 'u''v' must be ub_block_unit aligned(u%num_per_block = 0 &&
/// v%num_per_block = 0), and num_per_block means element number of UB blocks
/// unit.
///
/// Example:
/// memref<20x10xf16, strided<[16,1]>> => memref<10x20xf16, strided<[32,1]>>
template <typename T>
__aiv__ __attribute__((always_inline)) void
vnchwconv_2d(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }
  static_assert((sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4) &&
                "Transpose can not support this type");

  // Input parameter constraints assert.
  check_inputs_of_vnchwconv_2d(src, dst);
  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);

  if (src->sizes[0] == 1 && src->sizes[1] == 1) {
    // src: memref<1x1xT, stride<[1, 1]>>
    // dst: memref<1x1xT, stride<[1, 1]>>
    __ubuf__ T *src_ptr = src->aligned + src->offset;
    __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    dst_ptr[0] = src_ptr[0];
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  // convert size to be aligned. e.g.
  // scenario1:
  // src: memref<30x10xf16, stride<[16, 1]>> to
  //      memref<32x16xf16>
  // dst: memref<10x30xf16, stride<[32, 1]>> to
  //      memref<16x32xf16>
  // scenario2:
  // src: memref<1x1xi8, stride<[1024, 32]>> to
  //      memref<32x32xi8>
  // dst: memref<1x1xi8, stride<[1024, 32]>> to
  //      memref<32x32xi8>
  int64_t new_src_size0 = CEIL_FACTOR(src->sizes[0], num_per_block);
  int64_t new_src_size1 = CEIL_FACTOR(src->sizes[1], num_per_block);
  int64_t new_dst_size0 = CEIL_FACTOR(dst->sizes[0], num_per_block);
  int64_t new_dst_size1 = CEIL_FACTOR(dst->sizes[1], num_per_block);
  int64_t new_src_stride0 =
      src->sizes[0] == 1
          ? CEIL_FACTOR(src->sizes[1], num_per_block)
          : src->strides[0];
  int64_t new_src_stride1 = src->sizes[1] == 1 ? 1 : src->strides[1];
  int64_t new_dst_stride0 =
      dst->sizes[0] == 1
          ? CEIL_FACTOR(dst->sizes[1], num_per_block)
          : dst->strides[0];
  int64_t new_dst_stride1 = dst->sizes[1] == 1 ? 1 : dst->strides[1];
  memref_t<__ubuf__ T, 2> new_src{src->allocated,
                                  src->aligned,
                                  src->offset,
                                  {new_src_size0, new_src_size1},
                                  {new_src_stride0, new_src_stride1}};
  memref_t<__ubuf__ T, 2> new_dst{dst->allocated,
                                  dst->aligned,
                                  dst->offset,
                                  {new_dst_size0, new_dst_size1},
                                  {new_dst_stride0, new_dst_stride1}};

  MODE mode;
  gen_transpose_mode(&new_src, &new_dst, mode);

  // choose m axis or n axis as loop_number
  int64_t tile_m = 0;
  int64_t tile_n = 0;
  if constexpr (sizeof(T) == 1) {
    // b8 vnchwconv
    tile_m = num_per_block;
    tile_n = num_per_block;
  } else if constexpr (sizeof(T) == 2) {
    // b16 vnchwconv
    tile_m = TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT;
    tile_n = num_per_block;
  } else if constexpr (sizeof(T) == 4) {
    // b32 vnchwconv
    if (mode == TRANSPOSE_MODE1) {
      tile_m = TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT;
      tile_n = num_per_block;
    } else {
      // TRANSPOSE_MODE2
      tile_m = num_per_block;
      tile_n = TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT;
    }
  }

  gen_vnchwconv_core<T>(&new_src, &new_dst, tile_m, tile_n, mode);
  INTRINSIC(pipe_barrier, PIPE_V);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
gen_transpose_4x4_b64_core(__ubuf__ T *src_ptr, __ubuf__ T *dst_ptr) {
  constexpr int row_num_per_transpose = 4;
  constexpr int col_num_per_transpose = 4;

  int64_t dst_va_args[TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int64_t src_va_args[TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // step1: use scatter_vnchwconv_b16 to transpose b64
  // src -> dst
  for (int64_t i = 0; i < TRANSPOSE_INSTR_ROW_OR_COL_NUM_PER_REPEAT; ++i) {
    src_va_args[i] = i * row_num_per_transpose;
    dst_va_args[i] = i * row_num_per_transpose;
  }
  set_va_registers<T>(src_ptr, src_va_args, dst_ptr, dst_va_args);
  INTRINSIC(scatter_vnchwconv_b16, VA0, VA2,
            4,  // repeat_times
            16, // dst_repeat_stride
            1   // src_repeat_stride
  );
  INTRINSIC(pipe_barrier, PIPE_V);

  // step2: use copyUB2UB to rearrange data
  // dst -> src
  for (int64_t k = 0; k < row_num_per_transpose; ++k) {
    INTRINSIC(copy_ubuf_to_ubuf,
              src_ptr + 4 * col_num_per_transpose * k,  // dst
              dst_ptr + 16 * col_num_per_transpose * k, // src
              0,                                        // sid
              4,                                        // nburst
              4,                                        // lenBurst
              0,                                        // srcGap
              12                                        // dstGap
    );
  }
  INTRINSIC(pipe_barrier, PIPE_V);

  // step3: use scatter_vnchwconv_b16 to transpose rearrange data
  // src -> dst
  set_va_registers<T>(src_ptr, src_va_args, dst_ptr, dst_va_args);
  INTRINSIC(scatter_vnchwconv_b16, VA0, VA2,
            4, // repeat_times
            1, // dst_repeat_stride
            16 // src_repeat_stride
  );
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
vnchwconv_2d_with_tmp(memref_t<__ubuf__ T, 2> *src,
                      memref_t<__ubuf__ T, 2> *dst,
                      memref_t<__ubuf__ T, 1> *tmp) {
  if (is_no_op<2>(src->sizes)) {
    return;
  }

  static_assert(sizeof(T) == 8 && "Transpose b64 can not support this type");
  // Input parameter constraints assert.
  check_inputs_of_vnchwconv_2d(src, dst);

  if (src->sizes[0] == 1 && src->sizes[1] == 1) {
    // src: memref<1x1xT, stride<[1, 1]>>
    // dst: memref<1x1xT, stride<[1, 1]>>
    __ubuf__ T *src_ptr = src->aligned + src->offset;
    __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
    INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
    dst_ptr[0] = src_ptr[0];
    INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
    return;
  }

  constexpr int num_per_block = INTR_BYTES_PER_BLOCK / sizeof(T);
  int64_t src_size0_aligned = CEIL_FACTOR(src->sizes[0], num_per_block);
  int64_t src_size1_aligned = CEIL_FACTOR(src->sizes[1], num_per_block);
  int64_t dst_size0_aligned = CEIL_FACTOR(dst->sizes[0], num_per_block);
  int64_t dst_size1_aligned = CEIL_FACTOR(dst->sizes[1], num_per_block);

  __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
  __ubuf__ T *src_ptr = src->aligned + src->offset;
  constexpr int row_num_per_transpose = 4;
  constexpr int col_num_per_transpose = 4;
  constexpr int tmp_buf_size =
      VNCHWCONV_INTR_BYTES_PER_REPEAT / sizeof(T) * row_num_per_transpose;

  memref_t<__ubuf__ T, 1> tmp_buf0{
      tmp->allocated, tmp->aligned, tmp->offset, {tmp_buf_size}, {1}};

  memref_t<__ubuf__ T, 1> tmp_buf1{tmp->allocated,
                                   tmp->aligned,
                                   tmp->offset + tmp_buf_size,
                                   {tmp_buf_size},
                                   {1}};

  // src: memref<1x23xi64, strided<[23, 1]> => memref<1x23xi64, strided<[24, 1]>
  int64_t new_src_stride0 = src->sizes[0] == 1
                                ? CEIL_FACTOR(src->sizes[1], num_per_block)
                                : src->strides[0];
  // dst: memref<1x23xi64, strided<[23, 1]> => memref<1x23xi64, strided<[24, 1]>
  int64_t new_dst_stride0 = dst->sizes[0] == 1
                                ? CEIL_FACTOR(dst->sizes[1], num_per_block)
                                : dst->strides[0];
  // src: memref<23x1xi64, strided<[4, 4]> => memref<23x1xi64, strided<[4, 1]>
  int64_t new_src_stride1 = src->sizes[1] == 1 ? 1 : src->strides[1];
  // dst: memref<23x1xi64, strided<[4, 4]> => memref<23x1xi64, strided<[4, 1]>
  int64_t new_dst_stride1 = dst->sizes[1] == 1 ? 1 : dst->strides[1];

  /// scenario: when stride is more than size aligned
  /// src: memref<6x17xf16, stride<[48, 1]>>
  /// dst: memref<17x6xf16, stride<[32, 1]>>
  int64_t src_stride0 = new_src_stride0 / new_src_stride1;
  int64_t dst_stride0 = new_dst_stride0 / new_dst_stride1;

  int64_t loop_row = src_size0_aligned / row_num_per_transpose;
  int64_t loop_col = src_size1_aligned / col_num_per_transpose;

  __ubuf__ T *tmp_buf0_ptr = tmp_buf0.aligned + tmp_buf0.offset;
  __ubuf__ T *tmp_buf1_ptr = tmp_buf1.aligned + tmp_buf1.offset;
  for (int64_t i = 0; i < loop_row; ++i) {
    for (int64_t j = 0; j < loop_col; ++j) {
      int64_t src_offset =
          row_num_per_transpose * src_stride0 * i + col_num_per_transpose * j;
      int64_t dst_offset =
          col_num_per_transpose * dst_stride0 * j + row_num_per_transpose * i;

      /// step1: copy 4x4xi64 block of src to tmp_buf0
      /// b64 transpose will be implemented in the units of 4x4xi64,
      /// so each loop will transpose 4x4xi64 to 4x4xi64
      int64_t src_gap = (src_stride0 - num_per_block) / num_per_block;
      INTRINSIC(copy_ubuf_to_ubuf,
                tmp_buf0_ptr,          // dst
                src_ptr + src_offset,  // src
                0,                     // sid
                row_num_per_transpose, // nburst
                1,                     // lenBurst
                src_gap,
                0 // dstGap
      );
      INTRINSIC(pipe_barrier, PIPE_V);

      /// step2: use vnchwconv + copyUB2UB + vnchwconv to transpose 4x4xi64
      /// block tmp_buf0 -> tmp_buf1
      gen_transpose_4x4_b64_core<T>(tmp_buf0_ptr, tmp_buf1_ptr);
      INTRINSIC(pipe_barrier, PIPE_V);

      /// step3: copy tmp_buf1 to dst
      int64_t dst_gap = (dst_stride0 - num_per_block) / num_per_block;
      INTRINSIC(copy_ubuf_to_ubuf,
                dst_ptr + dst_offset,  // dst
                tmp_buf1_ptr,          // src
                0,                     // sid
                row_num_per_transpose, // nburst
                1,                     // lenBurst
                0,                     // srcGap
                dst_gap);
      INTRINSIC(pipe_barrier, PIPE_V);
    }
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// Transpose 2d with last axis
//===-------------------------------------------------------------------===//
REGISTE_TRANSPOSE_WITH_LAST_AXIS(2, uint8_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(2, int8_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(2, uint16_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(2, int16_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(2, half)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(2, bfloat16_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(2, uint32_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(2, int32_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS(2, float)
REGISTE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(2, uint64_t)
REGISTE_TRANSPOSE_WITH_LAST_AXIS_WITH_TMP(2, int64_t)

//===-------------------------------------------------------------------===//
// vnchwconv_2d func
//===-------------------------------------------------------------------===//
REGISTE_VNCHWCONV_2D(uint8_t);
REGISTE_VNCHWCONV_2D(int8_t);
REGISTE_VNCHWCONV_2D(uint16_t);
REGISTE_VNCHWCONV_2D(int16_t);
REGISTE_VNCHWCONV_2D(half);
REGISTE_VNCHWCONV_2D(bfloat16_t);
REGISTE_VNCHWCONV_2D(uint32_t);
REGISTE_VNCHWCONV_2D(int32_t);
REGISTE_VNCHWCONV_2D(float);
}
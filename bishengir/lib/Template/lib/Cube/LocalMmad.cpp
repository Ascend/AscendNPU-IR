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

#include "Cube/LocalMmad/LocalMmadUtils.h"

template <typename TYPE> struct b16_converter {
  using type = TYPE;
};
template <> struct b16_converter<bfloat16_t> {
  using type = half;
};

__aicore__ __attribute__((always_inline)) void set_hf32_ctrl() {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(set_ctrl,
            (get_ctrl() | (uint64_t)((uint64_t)1 << HF32_CTRL_REGISTER_BIT)));
#endif
}

__aicore__ __attribute__((always_inline)) void set_hf32_ctrl_none() {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
  INTRINSIC(set_ctrl, (get_ctrl() & (uint64_t)(0xFFFDFFFFFFFFFFFF)));
#endif
}

/// transpose L1: n1, k1, k0, n0, where k0 = 16, n0 = 32
/// to L0: k1',n1',n0', k0', where n0' = 16, k0' = 32
template <typename DST_QUALIFER>
__aicore__ __attribute__((always_inline)) void
load2d_transpose_b8(DST_QUALIFER *l0_buf, __cbuf__ int8_t *l1_ptr, int64_t k,
                    int64_t n, int64_t k_total) {
  // gm: k, n
  // L1: n1, k1, k0, n0 = [ceilDiv(n, 32), ceilDiv(k, 16), 16, 32]
  // L1 reshape as n1, k1/2(k1'), 2*k0(k0'), 2, n0/2(n0')
  // L0 transpose into k1/2(k1'), n1, 2, n0/2(n0'), 2*k0(k0')
  // L0 reshape: k1'(k1/2), n1'(n1*2), n0'(n0/2), k0'(2*k0)
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(int8_t);
  uint64_t l0_k1 = CEIL_DIV(k, elem_num_per_block);
  uint64_t l1_n1 = CEIL_DIV(n, elem_num_per_block);
  for (uint64_t i = 0; i < l0_k1; i++) {
    for (uint64_t j = 0; j < l1_n1; j++) {
      if (std::is_same<DST_QUALIFER, __cb__ int8_t>::value) {
        auto l0_loc = (l1_n1 * elem_num_per_block - n >= 16)
                          ? (l0_buf + i * (l1_n1 * 1024 - 512) + j * 1024)
                          : (l0_buf + i * l1_n1 * 1024 + j * 1024);
        load2d_transpose_cbuf_to_cb_intrin_core(
            load2d_transpose_intrin_args<int8_t, DST_QUALIFER>{
                l0_loc, l1_ptr + i * 1024 + j * k_total * 32, 0, 1, 1, 1,
                Load2DTransposeMode::ADDADDR, 0});
      } else if (std::is_same<DST_QUALIFER, __ca__ int8_t>::value) {
        load2d_transpose_cbuf_to_ca_intrin_core(
            load2d_transpose_intrin_args<int8_t, DST_QUALIFER>{
                l0_buf + j * l0_k1 * 1024 + i * 512,
                l1_ptr + i * 1024 + j * k_total * 32, 0, 1, 1, 1,
                Load2DTransposeMode::ADDADDR,
                static_cast<uint16_t>(l0_k1 - 1)});
      }
    }
  }
}

/// transpose L1: n1, k1, k0, n0, where k0 = 16, n0 = 64
/// to L0: k1',n1',n0', k0', where n0' = 16, k0' = 64
template <typename DST_QUALIFER>
__aicore__ __attribute__((always_inline)) void
load2d_transpose_b4(DST_QUALIFER *l0_buf, __cbuf__ int8_t *l1_ptr, int64_t k,
                    int64_t n, int64_t k_total) {
  // gm: k, n
  // L1: n1, k1, k0, n0 = [ceilDiv(n, 64), ceilDiv(k, 16), 16, 64]
  // L1 reshape as n1, k1/4(k1'), 4*k0(k0'), 4, n0/4(n0')
  // L0 transpose into k1/4(k1'), n1, 4, n0/4(n0'), 4*k0(k0')
  // L0 reshape: k1'(k1/4), n1'(n1*4), n0'(n0/4), k0'(4*k0)
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(int8_t);
  uint64_t l0_k1 = CEIL_DIV(k, 2 * elem_num_per_block);
  uint64_t l1_n1 = CEIL_DIV(n, elem_num_per_block);
  for (uint64_t i = 0; i < l0_k1; i++) {
    for (uint64_t j = 0; j < l1_n1; j++) {
      if (std::is_same<DST_QUALIFER, __cb__ int8_t>::value) {
        auto l0_loc = l0_buf;
        if ((l1_n1 * elem_num_per_block - n) >= 24) {
          l0_loc = l0_buf + i * (l1_n1 * 2048 - 1536) + j * 2048;
        } else if ((l1_n1 * elem_num_per_block - n) >= 16) {
          l0_loc = l0_buf + i * (l1_n1 * 2048 - 1024) + j * 2048;
        } else if ((l1_n1 * elem_num_per_block - n) >= 8) {
          l0_loc = l0_buf + i * (l1_n1 * 2048 - 512) + j * 2048;
        } else {
          l0_loc = l0_buf + i * l1_n1 * 2048 + j * 2048;
        }
        load2d_transpose_cbuf_to_cb_intrin_core_s4(
            load2d_transpose_intrin_args<void, DST_QUALIFER>{
                l0_loc, l1_ptr + i * 2048 + j * k_total * 32, 0, 1, 1, 3,
                Load2DTransposeMode::ADDADDR, 0});
      }
    }
  }
}


/// move the L1 data to L0A data, where only tile k axis and no m axis
/// k_part means that how much k can be put into L0, it is main block aligned
/// tile size k_part_ceil means the aligned k tile size in L0 for k_part_idx
/// iteration k_part_actual means that unaligned k tile size in L0 for
/// k_part_idx iteration
///
/// case 1: (No Transpose)
///   L1 zN data : k1, m1, m0, k0
///   L0 zZ data : am1, ak1, m0, k0
/// case 2: (TA)
///   L1 nZ data :  m1, k1, k0, m0
///   L0 zZ data :  am1, ak1, m0, k0
template <typename SRC_TYPE, bool TA, bool I4>
__aicore__ __attribute__((always_inline)) void
load_l1_to_l0a(__ca__ SRC_TYPE *l0a_buf, memref_t<__cbuf__ SRC_TYPE, 4> *ma,
               int64_t k_part_idx, int64_t k_part, int64_t k_part_ceil,
               int64_t k_part_loop, int64_t k_part_actual, int64_t m) {
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t fractal_num = FRACTAL_BLOCK_NUM * elem_num_per_block;
  auto am = TA ? (ma->sizes[0] * ma->sizes[3]) : (ma->sizes[1] * ma->sizes[2]);
  auto ak = TA ? (ma->sizes[1] * ma->sizes[2]) : (ma->sizes[0] * ma->sizes[3]);
  __cbuf__ SRC_TYPE *ma_ptr = ma->aligned + ma->offset;
  auto src_part_offset_a =
      TA ? k_part_idx * k_part * ma->sizes[3] : k_part_idx * k_part * am;
  if constexpr (I4) {
    for (uint64_t i = 0; i < ma->sizes[1]; i++) {
      INTRINSIC(load_cbuf_to_ca_s4, l0a_buf + i * k_part_ceil * ma->sizes[2],
                ma_ptr + src_part_offset_a + i * ma->strides[1], 0,
                k_part_ceil / ma->strides[2], ma->strides[0] / fractal_num, 0,
                0, false, inc);
    }
  } else if (sizeof(SRC_TYPE) == 1) {
    if constexpr (TA) {
      load2d_transpose_b8<__ca__ int8_t>(
          reinterpret_cast<__ca__ int8_t *>(l0a_buf),
          reinterpret_cast<__cbuf__ int8_t *>(src_part_offset_a + ma_ptr),
          k_part_actual, m, ak);
    } else {
      for (uint64_t i = 0; i < ma->sizes[1]; i++) {
        INTRINSIC(load_cbuf_to_ca, l0a_buf + i * k_part_ceil * ma->sizes[2],
                  ma_ptr + src_part_offset_a + i * ma->strides[1], 0,
                  k_part_ceil / ma->strides[2], ma->strides[0] / fractal_num, 0,
                  0, false, inc);
      }
    }
  } else if (sizeof(SRC_TYPE) == 2 || sizeof(SRC_TYPE) == 4) {
    using cast_src_type = typename b16_converter<SRC_TYPE>::type;
    if constexpr (TA) {
      //   L1 nZ data :  m1, k1, k0, m0
      //   L0 zZ data :  am1, ak1, m0, k0
      uint64_t setFmatrix = (ak | (1 << 16));
      uint64_t channelSize = CEIL_FACTOR(m, elem_num_per_block);
      uint64_t kFactor = CEIL_FACTOR(k_part_actual, 16);
      INTRINSIC(set_fmatrix, setFmatrix);
      img2colv2_cbuf_to_ca_intrin_core(
          img2colv2_intrin_args<cast_src_type, __ca__ cast_src_type>{
              reinterpret_cast<__ca__ cast_src_type *>(l0a_buf),
              reinterpret_cast<__cbuf__ cast_src_type *>(ma_ptr +
                                                         src_part_offset_a),
              static_cast<uint16_t>(channelSize),
              static_cast<uint16_t>(kFactor), static_cast<uint16_t>(0),
              static_cast<uint16_t>(0), 1, 1, 1, 1, 1, 1, 0, 0, 1 /*transpose*/,
              0 /*fmatrixCtrl*/, static_cast<uint16_t>(channelSize)});
    } else {
      //   L1 zN data : k1, m1, m0, k0
      //   L0 zZ data : am1, ak1, m0, k0
      uint64_t setFmatrix = (am | (1 << 16));
      uint64_t channelSize = CEIL_FACTOR(k_part_actual, elem_num_per_block);
      uint64_t mFactor = CEIL_FACTOR(m, 16);
      INTRINSIC(set_fmatrix, setFmatrix);
      img2colv2_cbuf_to_ca_intrin_core(
          img2colv2_intrin_args<cast_src_type, __ca__ cast_src_type>{
              reinterpret_cast<__ca__ cast_src_type *>(l0a_buf),
              reinterpret_cast<__cbuf__ cast_src_type *>(ma_ptr +
                                                         src_part_offset_a),
              static_cast<uint16_t>(channelSize),
              static_cast<uint16_t>(mFactor), static_cast<uint16_t>(0),
              static_cast<uint16_t>(0), 1, 1, 1, 1, 1, 1, 0, 0, 0 /*transpose*/,
              0 /*fmatrixCtrl*/, static_cast<uint16_t>(channelSize)});
    }
  }
}

template <typename SRC_TYPE>
__aicore__ __attribute__((always_inline)) void load_l1_to_l0b_with_trans(
    __cb__ SRC_TYPE *l0b_buf, memref_t<__cbuf__ SRC_TYPE, 4> *mb,
    int64_t src_part_offset_b, int64_t k_part_ceil, int64_t n) {
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t fractal_num = FRACTAL_BLOCK_NUM * elem_num_per_block;
  auto bn = mb->sizes[1] * mb->sizes[2];
  __cbuf__ SRC_TYPE *mb_ptr = mb->aligned + mb->offset;

  // L1: k1, n1, n0, k0, where k0 = 32, n0 = 16, l1 n1 is tiling of n
  // L0: k1, n1', n0', k0', where k0 = 32, n0 = 16, l0 n1 is ceil of actual n
  auto n_ceil = CEIL_FACTOR(n, FRACTAL_BLOCK_NUM);
  bool isk1n1Contiguous = n_ceil < bn;
  if (isk1n1Contiguous) {
    // for case that n_ceil is smaller than bn,  only valid data will be
    // loaded to l0
    auto k1 = mb->sizes[0];
    auto l0_n1 = n_ceil / FRACTAL_BLOCK_NUM;
    auto l0_k1 = k_part_ceil / elem_num_per_block;
    bool is_repeat_n1 = k1 <= l0_n1;
    int64_t repeat_time = is_repeat_n1 ? l0_n1 : l0_k1;
    int64_t loop_num = is_repeat_n1 ? k1 : l0_n1;
    auto n1 = mb->sizes[1];
    int64_t src_repeat_stride = is_repeat_n1 ? 1 : n1;
    int64_t dst_repeat_gap = is_repeat_n1 ? 0 : l0_n1 - 1;
    int64_t src_loop_stride = (is_repeat_n1 ? n1 : 1) * fractal_num;
    int64_t dst_loop_stride = (is_repeat_n1 ? l0_n1 : 1) * fractal_num;

    for (auto i = 0; i < loop_num; i++) {
      auto l0b_dst_offset = dst_loop_stride * i;
      auto l1b_src_offset = src_loop_stride * i;
      INTRINSIC(load_cbuf_to_cb, l0b_buf + l0b_dst_offset,
                mb_ptr + src_part_offset_b + l1b_src_offset, 0, repeat_time,
                src_repeat_stride, dst_repeat_gap, 0, false, inc);
    }
  } else {
    INTRINSIC(load_cbuf_to_cb, l0b_buf, mb_ptr + src_part_offset_b, 0,
              k_part_ceil * bn / fractal_num, 1, 0, 0, false, inc);
  }
}

/// move the L1 data to L0B data, where only tile k axis and no n axis
/// k_part means that how much k can be put into L0, it is main block aligned
/// tile size k_part_ceil means the aligned k tile size in L0 for k_part_idx
/// iteration k_part_actual means that unaligned k tile size in L0 for
/// k_part_idx iteration
///
/// case 1: (No Transpose)
///   L1 zN data : n1, k1, k0, n0
///   L0 nZ data : bk1, bn1, n0, k0
/// case 2: (TB)
///   L1 nZ data :  k1, n1, n0, k0
///   L0 nZ data :  bk1, bn1, n0, k0
template <typename SRC_TYPE, bool TB, bool I4>
__aicore__ __attribute__((always_inline)) void
load_l1_to_l0b(__cb__ SRC_TYPE *l0b_buf, memref_t<__cbuf__ SRC_TYPE, 4> *mb,
               int64_t k_part_idx, int64_t k_part, int64_t k_part_ceil,
               int64_t k_part_loop, int64_t k_part_actual, int64_t n) {
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t fractal_num = FRACTAL_BLOCK_NUM * elem_num_per_block;
  auto bn = TB ? (mb->sizes[1] * mb->sizes[2]) : (mb->sizes[0] * mb->sizes[3]);
  auto bk = TB ? (mb->sizes[0] * mb->sizes[3]) : (mb->sizes[1] * mb->sizes[2]);
  __cbuf__ SRC_TYPE *mb_ptr = mb->aligned + mb->offset;
  auto src_part_offset_b =
      TB ? k_part_idx * k_part * bn : k_part_idx * k_part * mb->sizes[3];

  if constexpr (I4) {
    uint16_t n_ceil = static_cast<uint16_t>(n);
    if constexpr (TB) {
      // TODO: IMPLEMENT
    } else {
      // L1: n1, k1, k0, n0, where k0 = 16, n0 = 32
      // L0B: k1',n1',n0', k0', where n0' = 16, k0' = 32
      load2d_transpose_b4<__cb__ int8_t>(l0b_buf, mb_ptr + src_part_offset_b,
                                         k_part_actual, n_ceil, bk);
    }
  } else if (sizeof(SRC_TYPE) == 1) {
    uint16_t n_ceil =
        (sizeof(SRC_TYPE) == 1 && n < 32) ? 32 : static_cast<uint16_t>(n);
    if constexpr (TB) {
      // The function internally handles n_ceil.
      load_l1_to_l0b_with_trans<SRC_TYPE>(l0b_buf, mb, src_part_offset_b,
                                          k_part_ceil, n);
    } else {
      // L1: n1, k1, k0, n0, where k0 = 16, n0 = 32
      uint16_t n_ceil = CEIL_FACTOR(n, /*32*/ elem_num_per_block);
      load2d_transpose_b8<__cb__ int8_t>(
          reinterpret_cast<__cb__ int8_t *>(l0b_buf),
          reinterpret_cast<__cbuf__ int8_t *>(mb_ptr + src_part_offset_b),
          k_part_actual, n_ceil, bk);
    }
  } else if (sizeof(SRC_TYPE) == 2 || sizeof(SRC_TYPE) == 4) {
    if constexpr (TB) {
      load_l1_to_l0b_with_trans<SRC_TYPE>(l0b_buf, mb, src_part_offset_b,
                                          k_part_ceil, n);
    } else {
      // L1 zN data : n1, k1, k0, n0
      // L0 nZ data : bk1, bn1, n0, k0
      uint64_t setFmatrix = (bk | (1 << 16));
      uint64_t channelSize = CEIL_FACTOR(n, elem_num_per_block);
      uint64_t kFactor = CEIL_FACTOR(k_part_actual, 16);
      INTRINSIC(set_fmatrix_b, setFmatrix);
      using cast_src_type = typename b16_converter<SRC_TYPE>::type;
      img2colv2_cbuf_to_cb_intrin_core(
          img2colv2_intrin_args<cast_src_type, __cb__ cast_src_type>{
              reinterpret_cast<__cb__ cast_src_type *>(l0b_buf),
              reinterpret_cast<__cbuf__ cast_src_type *>(mb_ptr +
                                                         src_part_offset_b),
              static_cast<uint16_t>(channelSize),
              static_cast<uint16_t>(kFactor), static_cast<uint16_t>(0),
              static_cast<uint16_t>(0), 1, 1, 1, 1, 1, 1, 0, 0, 1 /*transpose*/,
              1 /*fmatrixCtrl*/, static_cast<uint16_t>(channelSize)});
    }
  }
}

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE, bool TA,
          bool TB, bool HF32, bool I4>
__aicore__ __attribute__((always_inline)) void
mma_tile(memref_t<__cbuf__ SRC_TYPE, 4> *ma, memref_t<__cbuf__ SRC_TYPE, 4> *mb,
         bool init, int64_t m, int64_t k, int64_t n,
         memref_t<__cbuf__ BIAS_TYPE, 4> *bias,
         memref_t<__cc__ DST_TYPE, 4> *mc, int64_t mmad_l1_wait_l1a_event,
         int64_t mmad_l1_wait_l1b_event, int64_t l1a_wait_mmad_l1_event,
         int64_t l1b_wait_mmad_l1_event, int64_t kloop_db_cond,
         int64_t back_pipe_m_pipe_mte1_db_event0,
         int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag) {
  if constexpr (I4) {
      static_assert(TA == 0 && "i4 mmad doesn't support transpose A");
  }

  __cc__ DST_TYPE *mc_ptr = mc->aligned + mc->offset;
  __ca__ SRC_TYPE *l0a_base = reinterpret_cast<__ca__ SRC_TYPE *>((uintptr_t)0);
  __cb__ SRC_TYPE *l0b_base = reinterpret_cast<__cb__ SRC_TYPE *>((uintptr_t)0);

  const int64_t k_actual = k;
  const int64_t m0 = TA ? (ma->sizes[0] * ma->sizes[3]) : (ma->sizes[1] * ma->sizes[2]);
  const int64_t n0 = TB ? (mb->sizes[1] * mb->sizes[2]) : (mb->sizes[0] * mb->sizes[3]);
  const int64_t mn_max = m0 > n0 ? m0 : n0;
  const int64_t l0c_m_size = mc->sizes[1] * mc->sizes[2];

  bool k_direction_align = (sizeof(SRC_TYPE) == 4) && TA; // k alignment mode
  int64_t k_ceil = CEIL_FACTOR(k, L1_ALIGN_BYTES / sizeof(SRC_TYPE));
  k_ceil = k_direction_align ? CEIL_FACTOR(k_ceil, FRACTAL_BLOCK_NUM) : k_ceil;

  const int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  const int64_t max_align_value = elem_num_per_block > 16 ? elem_num_per_block : 16;

  int64_t n_ceil = n;
  if constexpr ((sizeof(SRC_TYPE) == 1)  && !I4) {
    // i8L1(TB):  k1, n1, n0, k0, where n0 = 16, k0 = 32
    // i8L1    :  n1, k1, k0, n0, where k0 = 16, n0 = 32
    n_ceil = TB ? CEIL_FACTOR(n_ceil, /*16*/ FRACTAL_BLOCK_NUM)
                : CEIL_FACTOR(n_ceil, /*32*/ elem_num_per_block);
  }
  n_ceil = k_direction_align ? CEIL_FACTOR(n, 16) : n_ceil;

  // compute k_part, namely how much k can be put into L0, it is main block
  // aligned tile size
  bool enable_double_buffer = true;
  int64_t l0ab_pingpong_buffer_len =
      L0AB_BUFFER_BYTES / 2 / sizeof(SRC_TYPE);
  int64_t k_part = FLOOR_FACTOR(l0ab_pingpong_buffer_len /
                                    CEIL_FACTOR(mn_max, max_align_value),
                                max_align_value);
  if (k_part <= 0) {
    enable_double_buffer = false;
    l0ab_pingpong_buffer_len = L0AB_BUFFER_BYTES / sizeof(SRC_TYPE);
    k_part = FLOOR_FACTOR(l0ab_pingpong_buffer_len /
      CEIL_FACTOR(mn_max, max_align_value),
      max_align_value);
  }

  if (k_part <= 0 || k_actual <= 0) {
    if (mmad_l1_wait_l1a_event != -1) {
      INTRINSIC(wait_flag, PIPE_MTE2, PIPE_MTE1, mmad_l1_wait_l1a_event);
    }
    if (mmad_l1_wait_l1b_event != -1) {
      INTRINSIC(wait_flag, PIPE_MTE2, PIPE_MTE1, mmad_l1_wait_l1b_event);
    }
    if (l1a_wait_mmad_l1_event != -1) {
      INTRINSIC(set_flag, PIPE_MTE1, PIPE_MTE2, l1a_wait_mmad_l1_event);
    }
    if (l1b_wait_mmad_l1_event != -1) {
      INTRINSIC(set_flag, PIPE_MTE1, PIPE_MTE2, l1b_wait_mmad_l1_event);
    }
    return;
  }

  if (back_pipe_m_pipe_mte1_db_event0 == -1) {
    INTRINSIC(set_flag, PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  if (back_pipe_m_pipe_mte1_db_event1 == -1) {
    INTRINSIC(set_flag, PIPE_M, PIPE_MTE1, EVENT_ID1);
  }

  const int64_t k_part_loop = (k_actual + k_part - 1) / k_part;
  for (int64_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
    const bool is_outer_k_start = k_part_idx == 0;
    const bool is_outer_k_end = k_part_idx + 1 == k_part_loop;
    const bool init_c = init && is_outer_k_start;

    // compute k_part_actual and k_part_ceil
    // k_part_ceil means the aligned k tile size in L0 for k_part_idx iteration
    // k_part_actual means that unaligned k tile size in L0 for k_part_idx
    // iteration
    int64_t k_part_ceil =
        !is_outer_k_end ? k_part : (k_ceil - k_part_idx * k_part);
    int64_t k_part_actual =
        !is_outer_k_end ? k_part : (k_actual - k_part_idx * k_part);

    int64_t ping_pong_id = 0;
    if (enable_double_buffer) {
      ping_pong_id = (kloop_db_cond != -1)
                         ? (kloop_db_cond * k_part_loop + k_part_idx) % 2
                         : k_part_idx % 2;
    }
    int64_t m_mte1_event_id = (ping_pong_id == 0)
                                  ? (back_pipe_m_pipe_mte1_db_event0 != -1
                                         ? back_pipe_m_pipe_mte1_db_event0
                                         : EVENT_ID0)
                                  : (back_pipe_m_pipe_mte1_db_event1 != -1
                                         ? back_pipe_m_pipe_mte1_db_event1
                                         : EVENT_ID1);
    __ca__ SRC_TYPE *l0a_buf =
        l0a_base + ping_pong_id * l0ab_pingpong_buffer_len;
    __cb__ SRC_TYPE *l0b_buf =
        l0b_base + ping_pong_id * l0ab_pingpong_buffer_len;

    // pipe_mte1 wait for previous iteration pipe_m to finish
    INTRINSIC(wait_flag, PIPE_M, PIPE_MTE1, m_mte1_event_id);

    // load matrix A from L1 to L0A
    if (is_outer_k_start && mmad_l1_wait_l1a_event != -1) {
      INTRINSIC(wait_flag, PIPE_MTE2, PIPE_MTE1, mmad_l1_wait_l1a_event);
    }

    load_l1_to_l0a<SRC_TYPE, TA, I4>(l0a_buf, ma, k_part_idx, k_part, k_part_ceil,
                                     k_part_loop, k_part_actual, l0c_m_size);

    if (is_outer_k_end && l1a_wait_mmad_l1_event != -1) {
      INTRINSIC(set_flag, PIPE_MTE1, PIPE_MTE2, l1a_wait_mmad_l1_event);
    }

    // load matrix B from L1 to L0B
    if (is_outer_k_start && mmad_l1_wait_l1b_event != -1) {
      INTRINSIC(wait_flag, PIPE_MTE2, PIPE_MTE1, mmad_l1_wait_l1b_event);
    }
    load_l1_to_l0b<SRC_TYPE, TB, I4>(l0b_buf, mb, k_part_idx, k_part, k_part_ceil,
                                     k_part_loop, k_part_actual, n);

    if (is_outer_k_end && l1b_wait_mmad_l1_event != -1) {
      INTRINSIC(set_flag, PIPE_MTE1, PIPE_MTE2, l1b_wait_mmad_l1_event);
    }

    // load matrix bias from L1 to BT
    if (bias) {
      const uint64_t btAddr = 0;
      __cbuf__ BIAS_TYPE *bias_ptr = bias->aligned + bias->offset;
      int64_t bias_burst_len = CEIL_DIV(n0 * sizeof(BIAS_TYPE), 64);
      int16_t conv_control = (sizeof(BIAS_TYPE) == 2 && sizeof(DST_TYPE) == 4);
      INTRINSIC(copy_cbuf_to_bt, btAddr, bias_ptr,
                conv_control,   // convControl
                1,              // nBurst
                bias_burst_len, // lenBurst
                0,              // sourceGap
                0);             // dstGap
    }

    // mmad
    if constexpr (HF32) {
      set_hf32_ctrl();
    }

    // pipe_m wait for pipe_mte1 instructions to finish
    INTRINSIC(set_flag, PIPE_MTE1, PIPE_M, m_mte1_event_id);
    INTRINSIC(wait_flag, PIPE_MTE1, PIPE_M, m_mte1_event_id);

    uint8_t unit_flag_mode =
        unit_flag ? (is_outer_k_end ? unit_flag : (uint8_t)0b10) : (uint8_t)0;

    if constexpr (I4) {
      if (bias) {
        bool isSourceFromBT = init_c;
        mad_intrin_core_s4(mmad_intrin_args<void, int32_t>{
                           mc_ptr, l0a_buf, l0b_buf, static_cast<uint16_t>(l0c_m_size),
                           static_cast<uint16_t>(2 * k_part_actual), static_cast<uint16_t>(2 * n_ceil),
                           unit_flag_mode, 0, isSourceFromBT, 0});
      } else {
        mad_intrin_core_s4(mmad_intrin_args<void, int32_t>{
                           mc_ptr, l0a_buf, l0b_buf, static_cast<uint16_t>(l0c_m_size),
                           static_cast<uint16_t>(2 * k_part_actual), static_cast<uint16_t>(2 * n_ceil), unit_flag_mode,
                           0, 0, init_c});
      }
    } else {
        if (bias) {
        bool isSourceFromBT = init_c;
        mad_intrin_core(mmad_intrin_args<SRC_TYPE, DST_TYPE>{
            mc_ptr, l0a_buf, l0b_buf, static_cast<uint16_t>(l0c_m_size),
            static_cast<uint16_t>(k_part_actual), static_cast<uint16_t>(n_ceil),
            unit_flag_mode, k_direction_align, /*cmatrixSource=*/isSourceFromBT,
            /*cmatrixInitVal=*/0});
      } else {
        mad_intrin_core(mmad_intrin_args<SRC_TYPE, DST_TYPE>{
            mc_ptr, l0a_buf, l0b_buf, static_cast<uint16_t>(l0c_m_size),
            static_cast<uint16_t>(k_part_actual), static_cast<uint16_t>(n_ceil),
            unit_flag_mode, k_direction_align, /*cmatrixSource=*/0,
            /*cmatrixInitVal=*/init_c});
      }
    }

    // The same l0c address is used. When M/16 * N/16>=10,
    // it is not necessary to insert BAR. M is applicable
    // to V300 and C200 versions.
    if (l0c_m_size / FRACTAL_BLOCK_NUM * n / FRACTAL_BLOCK_NUM < 10) {
      INTRINSIC(pipe_barrier, PIPE_M);
    }

    // pipe_m set next iteration pipe_mte1
    INTRINSIC(set_flag, PIPE_M, PIPE_MTE1, m_mte1_event_id);

    if constexpr (HF32) {
      set_hf32_ctrl_none();
    }
  }
  if (back_pipe_m_pipe_mte1_db_event0 == -1) {
    INTRINSIC(wait_flag, PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  if (back_pipe_m_pipe_mte1_db_event1 == -1) {
    INTRINSIC(wait_flag, PIPE_M, PIPE_MTE1, EVENT_ID1);
  }
}

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE, bool TA,
          bool TB, bool HF32, bool I4>
__aicore__ __attribute__((always_inline)) void
mma_tile_core(memref_t<__cbuf__ SRC_TYPE, 4> *ma,
              memref_t<__cbuf__ SRC_TYPE, 4> *mb, bool init, int64_t m,
              int64_t k, int64_t n, memref_t<__cc__ DST_TYPE, 4> *mc,
              int64_t mmad_l1_wait_l1a_event, int64_t mmad_l1_wait_l1b_event,
              int64_t l1a_wait_mmad_l1_event, int64_t l1b_wait_mmad_l1_event,
              int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,
              int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag) {
    mma_tile<SRC_TYPE, DST_TYPE, BIAS_TYPE, TA, TB, HF32, I4>(
        ma, mb, init, m, k, n, nullptr, mc, mmad_l1_wait_l1a_event,
        mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event, l1b_wait_mmad_l1_event,
        kloop_db_cond, back_pipe_m_pipe_mte1_db_event0,
        back_pipe_m_pipe_mte1_db_event1, unit_flag);
}

template <typename SRC_TYPE, typename DST_TYPE, typename BIAS_TYPE, bool TA,
          bool TB, bool HF32, bool I4>
__aicore__ __attribute__((always_inline)) void
mma_tile_bias(memref_t<__cbuf__ SRC_TYPE, 4> *ma,
              memref_t<__cbuf__ SRC_TYPE, 4> *mb, bool init, int64_t m,
              int64_t k, int64_t n, memref_t<__cbuf__ BIAS_TYPE, 4> *bias,
              memref_t<__cc__ DST_TYPE, 4> *mc, int64_t mmad_l1_wait_l1a_event,
              int64_t mmad_l1_wait_l1b_event, int64_t l1a_wait_mmad_l1_event,
              int64_t l1b_wait_mmad_l1_event, int64_t kloop_db_cond,
              int64_t back_pipe_m_pipe_mte1_db_event0,
              int64_t back_pipe_m_pipe_mte1_db_event1, uint8_t unit_flag) {
  mma_tile<SRC_TYPE, DST_TYPE, BIAS_TYPE, TA, TB, HF32, I4>(
      ma, mb, init, m, k, n, bias, mc, mmad_l1_wait_l1a_event,
      mmad_l1_wait_l1b_event, l1a_wait_mmad_l1_event, l1b_wait_mmad_l1_event,
      kloop_db_cond, back_pipe_m_pipe_mte1_db_event0,
      back_pipe_m_pipe_mte1_db_event1, unit_flag);
}

extern "C" {
REGISTER_MMA_TILE(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TA_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TA(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TB(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_TB_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, half, float, float);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, bfloat16_t, float, float);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TA_TB_HF32(cbuf, cc, 4, float, float, float);
REGISTER_MMA_TILE_TA_TB(cbuf, cc, 4, int8_t, int32_t, int32_t);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, half, float, half);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, float, float, half);
REGISTER_MMA_TILE_BIAS(cbuf, cc, 4, bfloat16_t, float, half);
REGISTER_MMA_TILE_I4(cbuf, cc, 4, int8_t, int32_t, float);
REGISTER_MMA_TILE_TB_I4(cbuf, cc, 4, int8_t, int32_t, float);
REGISTER_MMA_TILE_HF32_I4(cbuf, cc, 4, int8_t, int32_t, float);
REGISTER_MMA_TILE_TB_HF32_I4(cbuf, cc, 4, int8_t, int32_t, float);
}

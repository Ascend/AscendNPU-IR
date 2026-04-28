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

#include "Cube/LocalConv/LocalConvUtils.h"

template <typename TYPE> struct b16_converter {
  using type = TYPE;
};
template <> struct b16_converter<bfloat16_t> {
  using type = half;
};

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void
check_input_alignment(memref_t<__cbuf__ SRC_TYPE, 5> *input,
                      memref_t<__cbuf__ SRC_TYPE, 5> *weight,
                      memref_t<__cc__ DST_TYPE, 4> *output, int64_t G) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t iC0 = input->sizes[4];
  int64_t oC = weight->sizes[3];
  int64_t oC1 = output->sizes[0];
  int64_t oC0 = output->sizes[3];
  int64_t oC_per_g = oC / G;
  int64_t oC1_per_g = oC1 / G;
  int64_t oC_ceil = CEIL_FACTOR(oC_per_g, FRACTAL_BLOCK_NUM);

  assert(oC_ceil == oC1_per_g * oC0 &&
         "ceil_factor(oC_weight) must equal oC_output ");
  assert(iC0 == elem_num_per_block && "iC0 must equal elem_num_per_block");
#endif
}

template <typename SRC_TYPE>
__aicore__ __attribute__((always_inline)) void
load_l1_to_l0a(__ca__ SRC_TYPE *l0a_buf, memref_t<__cbuf__ SRC_TYPE, 5> *input,
               int64_t k_part_idx, int64_t k_part, int64_t k_part_ceil,
               int64_t batch_idx, int64_t group_idx, int64_t G, int64_t wH,
               int64_t wW, int64_t oHW1, int64_t oHW0, int64_t padL,
               int64_t padR, int64_t padT, int64_t padB, int64_t strideH,
               int64_t strideW, int64_t dilationH, int64_t dilationW) {

  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t fractal_num = FRACTAL_BLOCK_NUM * elem_num_per_block;
  __cbuf__ SRC_TYPE *input_ptr = input->aligned + input->offset;

  int64_t B = input->sizes[0];
  int64_t iC1 = input->sizes[1];
  int64_t iH = input->sizes[2];
  int64_t iW = input->sizes[3];
  int64_t iC0 = input->sizes[4];
  int64_t iC1_per_g = iC1 / G;
  int64_t stepK = k_part_ceil;
  int64_t stepM = oHW1 * oHW0;
  int64_t channelSize = k_part_ceil / wH / wW;
  int64_t align_block = wH * wW * elem_num_per_block;

  auto src_part_offset_a = batch_idx * G * iC1_per_g * iH * iW * iC0 +
                           group_idx * iC1_per_g * iH * iW * iC0 +
                           k_part_idx * (k_part / align_block) * iH * iW * iC0;

  if (sizeof(SRC_TYPE) == 2 || sizeof(SRC_TYPE) == 4) {
    using cast_src_type = typename b16_converter<SRC_TYPE>::type;
    uint64_t fmatrix_input = (uint64_t)(static_cast<uint16_t>(iW)) |
                             ((uint64_t)(static_cast<uint16_t>(iH)) << 16) |
                             ((uint64_t)(static_cast<uint8_t>(padL)) << 32) |
                             ((uint64_t)(static_cast<uint8_t>(padR)) << 40) |
                             ((uint64_t)(static_cast<uint8_t>(padT)) << 48) |
                             ((uint64_t)(static_cast<uint8_t>(padB)) << 56);
    INTRINSIC(set_fmatrix, fmatrix_input);

    img2colv2_cbuf_to_ca_intrin_core(
        img2colv2_intrin_args<cast_src_type, __ca__ cast_src_type>{
            reinterpret_cast<__ca__ cast_src_type *>(l0a_buf),
            reinterpret_cast<__cbuf__ cast_src_type *>(input_ptr +
                                                       src_part_offset_a),
            static_cast<uint16_t>(stepK), static_cast<uint16_t>(stepM),
            static_cast<uint16_t>(0) /*posK*/,
            static_cast<uint16_t>(0) /*posM*/, static_cast<uint8_t>(strideW),
            static_cast<uint8_t>(strideH), static_cast<uint8_t>(wW),
            static_cast<uint8_t>(wH), static_cast<uint8_t>(dilationW),
            static_cast<uint8_t>(dilationH), 0 /*filterW*/, 0 /*filterH*/,
            0 /*transpose*/, 0 /*fmatrixCtrl*/,
            static_cast<uint16_t>(channelSize)});
  }
}

// load k-tile of weight matrix into L0B
// no loop_k, because src_repeat_stride cannot be defined under this case
// TODO: Refactor this function together with the implemantation in LocalMmad
template <typename SRC_TYPE>
__aicore__ __attribute__((always_inline)) void load_l1_to_l0b_unaligned(
    __cb__ SRC_TYPE *l0b_buf, memref_t<__cbuf__ SRC_TYPE, 5> *weight,
    int64_t src_part_offset_b, int64_t k_part_ceil, int64_t n_group_ceil) {
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t fractal_num = FRACTAL_BLOCK_NUM * elem_num_per_block;
  __cbuf__ SRC_TYPE *weight_ptr = weight->aligned + weight->offset;

  // 1 loop option: along n axis
  auto l0_n1 = n_group_ceil / FRACTAL_BLOCK_NUM;
  auto l0_k1 = k_part_ceil / elem_num_per_block;
  int64_t repeat_time = l0_n1;
  int64_t loop_num = l0_k1;
  auto n = weight->sizes[3];
  int64_t src_repeat_stride = 1;
  int64_t dst_repeat_gap = 0;

  int64_t src_loop_stride = n * elem_num_per_block;
  int64_t dst_loop_stride = l0_n1 * fractal_num;

  for (auto i = 0; i < loop_num; i++) {
    auto l0b_dst_offset = dst_loop_stride * i;
    auto l1b_src_offset = src_loop_stride * i;
    INTRINSIC(load_cbuf_to_cb, l0b_buf + l0b_dst_offset,
              weight_ptr + src_part_offset_b + l1b_src_offset, 0, repeat_time,
              src_repeat_stride, dst_repeat_gap, 0, false, inc);
  }
}

// load k-tile and n-tile (group tiling) of weight matrix into L0B
// TODO: Refactor this function together with the implemantation in LocalMmad
template <typename SRC_TYPE>
__aicore__ __attribute__((always_inline)) void load_l1_to_l0b_group(
    __cb__ SRC_TYPE *l0b_buf, memref_t<__cbuf__ SRC_TYPE, 5> *weight,
    int64_t src_part_offset_b, int64_t k_part_ceil, int64_t n_group) {
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t fractal_num = FRACTAL_BLOCK_NUM * elem_num_per_block;
  __cbuf__ SRC_TYPE *weight_ptr = weight->aligned + weight->offset;

  // just for safe, not necessary, required from input check
  auto n_ceil = CEIL_FACTOR(n_group, FRACTAL_BLOCK_NUM);

  // 2 loop options: along k axis or n axis
  auto l0_n1 = n_ceil / FRACTAL_BLOCK_NUM;
  auto l0_k1 = k_part_ceil / elem_num_per_block;
  bool is_repeat_n1 = l0_k1 <= l0_n1;
  int64_t repeat_time = is_repeat_n1 ? l0_n1 : l0_k1;
  int64_t loop_num = is_repeat_n1 ? l0_k1 : l0_n1;
  auto n1 = weight->sizes[3] / FRACTAL_BLOCK_NUM;
  int64_t src_repeat_stride = is_repeat_n1 ? 1 : n1;
  int64_t dst_repeat_gap = is_repeat_n1 ? 0 : l0_n1 - 1;

  int64_t src_loop_stride = (is_repeat_n1 ? n1 : 1) * fractal_num;
  int64_t dst_loop_stride = (is_repeat_n1 ? l0_n1 : 1) * fractal_num;

  for (auto i = 0; i < loop_num; i++) {
    auto l0b_dst_offset = dst_loop_stride * i;
    auto l1b_src_offset = src_loop_stride * i;
    INTRINSIC(load_cbuf_to_cb, l0b_buf + l0b_dst_offset,
              weight_ptr + src_part_offset_b + l1b_src_offset, 0, repeat_time,
              src_repeat_stride, dst_repeat_gap, 0, false, inc);
  }
}

/// move the L1 data to L0B data, where tile both k axis and n axis
/// k_part means that how much k can be put into L0, it is main block aligned
/// tile size k_part_ceil means the aligned k tile size in L0 for k_part_idx
/// iteration k_part_actual means that unaligned k tile size in L0 for
/// k_part_idx iteration
/// due to previous padding in NPU IR, k_part_ceil should be equal to
/// k_part_actual
template <typename SRC_TYPE>
__aicore__ __attribute__((always_inline)) void
load_l1_to_l0b(__cb__ SRC_TYPE *l0b_buf, memref_t<__cbuf__ SRC_TYPE, 5> *weight,
               int64_t k_part_idx, int64_t k_part, int64_t k_part_ceil,
               int64_t group_idx, int64_t n_group, int64_t G) {
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t fractal_num = FRACTAL_BLOCK_NUM * elem_num_per_block;
  int64_t oC = weight->sizes[3]; // if G = 1, bn = n_group
  int64_t wH = weight->sizes[1];
  int64_t wW = weight->sizes[2];
  int64_t iC0 = weight->sizes[4];
  __cbuf__ SRC_TYPE *weight_ptr = weight->aligned + weight->offset;
  auto src_part_offset_b =
      k_part_idx * k_part * oC + group_idx * n_group * elem_num_per_block;
  bool is_one_group = G == 1;
  bool is_n_aligned = n_group % FRACTAL_BLOCK_NUM == 0;
  int64_t n_group_ceil = CEIL_FACTOR(n_group, FRACTAL_BLOCK_NUM);

  if (sizeof(SRC_TYPE) == 2 || sizeof(SRC_TYPE) == 4) {
    if (is_one_group) {
      if (is_n_aligned) {
        uint8_t repeat = k_part_ceil * oC / fractal_num;
        INTRINSIC(load_cbuf_to_cb, l0b_buf, weight_ptr + src_part_offset_b,
                  static_cast<uint16_t>(0), repeat, static_cast<uint16_t>(1),
                  static_cast<uint16_t>(0), static_cast<uint8_t>(0), false,
                  inc);
      } else {
        load_l1_to_l0b_unaligned<SRC_TYPE>(l0b_buf, weight, src_part_offset_b,
                                           k_part_ceil, n_group_ceil);
      }
    } else {
      if (is_n_aligned) {
        load_l1_to_l0b_group<SRC_TYPE>(l0b_buf, weight, src_part_offset_b,
                                       k_part_ceil, n_group);
      } else {
        load_l1_to_l0b_unaligned<SRC_TYPE>(l0b_buf, weight, src_part_offset_b,
                                           k_part_ceil, n_group_ceil);
      }
    }
  }
}

template <typename SRC_TYPE, typename DST_TYPE>
__aicore__ __attribute__((always_inline)) void conv2d_group(
    memref_t<__cbuf__ SRC_TYPE, 5> *input,           // [B, iC1, iH, iW, iC0]
    memref_t<__cbuf__ SRC_TYPE, 5> *weight,          // [iC1/G, wH, wW, oC, iC0]
    bool init, memref_t<__cc__ DST_TYPE, 4> *output, // [oC1, oHW1, oHW0, oC0]
    int64_t G, int64_t padT, int64_t padB, int64_t padL, int64_t padR,
    int64_t strideH, int64_t strideW, int64_t dilationH, int64_t dilationW,
    int64_t conv_l1_wait_l1a_event, int64_t conv_l1_wait_l1b_event,
    int64_t l1a_wait_conv_l1_event, int64_t l1b_wait_conv_l1_event,
    int64_t back_pipe_m_pipe_mte1_db_event0,
    int64_t back_pipe_m_pipe_mte1_db_event1) {
  // -----------------------------
  // Derived dimensions
  // -----------------------------
  check_input_alignment<SRC_TYPE, DST_TYPE>(input, weight, output, G);
  int64_t B = input->sizes[0];
  int64_t iC1 = input->sizes[1];
  int64_t iH = input->sizes[2];
  int64_t iW = input->sizes[3];
  int64_t iC0 = input->sizes[4];    // should be 32 Byte aligned
  int64_t wH = weight->sizes[1];
  int64_t wW = weight->sizes[2];
  int64_t oC = weight->sizes[3];    // if aligned, oC should be equal to oC1*oC0
  int64_t oC1 = output->sizes[0];
  int64_t oHW1 = output->sizes[1];
  int64_t oHW0 = output->sizes[2];  // should be 16
  int64_t oC0 = output->sizes[3];   // should be 16

  int64_t iC1_per_g = iC1 / G;      // input channels per group
  int64_t oC_per_g = oC / G;        // weight channels per group
  int64_t oC1_per_g = oC1 / B / G;  // output-channel blocks per group

  int64_t m = oHW1 * oHW0;
  int64_t n = oC1_per_g * oC0;
  int64_t k = iC1_per_g * wH * wW * iC0;

  // k_actual should be equal to k_ceil under our case
  auto k_actual = k;
  auto k_ceil = CEIL_FACTOR(k, L1_ALIGN_BYTES / sizeof(SRC_TYPE));

  if (back_pipe_m_pipe_mte1_db_event0 == -1) {
    INTRINSIC(set_flag, PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  if (back_pipe_m_pipe_mte1_db_event1 == -1) {
    INTRINSIC(set_flag, PIPE_M, PIPE_MTE1, EVENT_ID1);
  }

  // L0 buffers
  __cc__ DST_TYPE *output_base = output->aligned + output->offset;
  __ca__ SRC_TYPE *l0a_base = reinterpret_cast<__ca__ SRC_TYPE *>((uintptr_t)0);
  __cb__ SRC_TYPE *l0b_base = reinterpret_cast<__cb__ SRC_TYPE *>((uintptr_t)0);

  int64_t mn_max = m > n ? m : n;
  int64_t L0AB_PINGPONG_BUFFER_LEN = L0AB_BUFFER_BYTES / 2 / sizeof(SRC_TYPE);
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);

  int64_t align_block = wH * wW * elem_num_per_block;
  int64_t k_part = L0AB_PINGPONG_BUFFER_LEN /
                   CEIL_FACTOR(mn_max, FRACTAL_BLOCK_NUM) / align_block *
                   align_block;
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert(k_part != 0 && "k_part should not be zero");
#endif
  int64_t k_part_loop = (k_actual + k_part - 1) / k_part;

  // -----------------------------
  // Loop over batches
  // -----------------------------
  for (int64_t batch_idx = 0; batch_idx < B; ++batch_idx) {
    // -----------------------------
    // Loop over groups
    // -----------------------------
    for (int64_t group_idx = 0; group_idx < G; ++group_idx) {
      // -----------------------------
      // Loop over k tiles
      // -----------------------------
      for (int64_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
        int64_t part_idx =
            batch_idx * G * k_part_loop + group_idx * k_part_loop + k_part_idx;
        int64_t part_loop = B * G * k_part_loop;

        // k_part_ceil should equal k_part_actual under our case, and they might
        // differ from k_part for tailing part
        int64_t k_part_ceil = (k_part_idx < k_part_loop - 1)
                                  ? k_part
                                  : k_ceil - k_part_idx * k_part;
        int64_t k_part_actual = (k_part_idx < k_part_loop - 1)
                                    ? k_part
                                    : k_actual - k_part_idx * k_part;

        auto mte1_conv_ping_flag = 1 - part_idx % 2;
        auto mte1_conv_event_id = mte1_conv_ping_flag
                                      ? (back_pipe_m_pipe_mte1_db_event0 != -1
                                             ? back_pipe_m_pipe_mte1_db_event0
                                             : EVENT_ID0)
                                      : (back_pipe_m_pipe_mte1_db_event1 != -1
                                             ? back_pipe_m_pipe_mte1_db_event1
                                             : EVENT_ID1);

        __ca__ SRC_TYPE *l0a_buf =
            l0a_base + (1 - mte1_conv_ping_flag) * L0AB_PINGPONG_BUFFER_LEN;
        __cb__ SRC_TYPE *l0b_buf =
            l0b_base + (1 - mte1_conv_ping_flag) * L0AB_PINGPONG_BUFFER_LEN;

        INTRINSIC(wait_flag, PIPE_M, PIPE_MTE1, mte1_conv_event_id);
        // load input from L1 to L0A
        if (part_idx == 0 && conv_l1_wait_l1a_event != -1) {
          INTRINSIC(wait_flag, PIPE_MTE2, PIPE_MTE1, conv_l1_wait_l1a_event);
        }

        load_l1_to_l0a<SRC_TYPE>(l0a_buf, input, k_part_idx, k_part,
                                 k_part_ceil, batch_idx, group_idx, G, wH, wW,
                                 oHW1, oHW0, padL, padR, padT, padB, strideH,
                                 strideW, dilationH, dilationW);

        if (part_idx == part_loop - 1 && l1a_wait_conv_l1_event != -1) {
          INTRINSIC(set_flag, PIPE_MTE1, PIPE_MTE2, l1a_wait_conv_l1_event);
        }

        // load weight from L1 to L0B
        if (part_idx == 0 && conv_l1_wait_l1b_event != -1) {
          INTRINSIC(wait_flag, PIPE_MTE2, PIPE_MTE1, conv_l1_wait_l1b_event);
        }

        load_l1_to_l0b<SRC_TYPE>(l0b_buf, weight, k_part_idx, k_part,
                                 k_part_ceil, group_idx, oC_per_g, G);

        if (part_idx == part_loop - 1 && l1b_wait_conv_l1_event != -1) {
          INTRINSIC(set_flag, PIPE_MTE1, PIPE_MTE2, l1b_wait_conv_l1_event);
        }

        INTRINSIC(set_flag, PIPE_MTE1, PIPE_M, mte1_conv_event_id);

        // ---------------------------------
        // MMA accumulate
        // ---------------------------------
        INTRINSIC(wait_flag, PIPE_MTE1, PIPE_M, mte1_conv_event_id);
        __cc__ DST_TYPE *output_ptr =
            output_base + batch_idx * G * oC1_per_g * oHW1 * oHW0 * oC0 +
            group_idx * oC1_per_g * oHW1 * oHW0 * oC0;

        bool init_c = init && !k_part_idx;

        mad_intrin_core(mmad_intrin_args<SRC_TYPE, DST_TYPE>{
            output_ptr, l0a_buf, l0b_buf, static_cast<uint16_t>(m),
            static_cast<uint16_t>(k_part_ceil), static_cast<uint16_t>(n),
            static_cast<uint8_t>(0) /*unitFlag*/,
            static_cast<bool>(0) /*kDirectionAlign*/,
            static_cast<bool>(0) /*cmatrixSource*/, init_c});

        if (m / FRACTAL_BLOCK_NUM * n / FRACTAL_BLOCK_NUM < 10) {
          INTRINSIC(pipe_barrier, PIPE_M);
        }
        INTRINSIC(set_flag, PIPE_M, PIPE_MTE1, mte1_conv_event_id);
      }
    }
  }

  if (back_pipe_m_pipe_mte1_db_event0 == -1) {
    INTRINSIC(wait_flag, PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  if (back_pipe_m_pipe_mte1_db_event1 == -1) {
    INTRINSIC(wait_flag, PIPE_M, PIPE_MTE1, EVENT_ID1);
  }
}

extern "C" {
REGISTER_CONV2D_GROUP(cbuf, cc, 5, float, float);
REGISTER_CONV2D_GROUP(cbuf, cc, 5, half, float);
REGISTER_CONV2D_GROUP(cbuf, cc, 5, bfloat16_t, float);
}
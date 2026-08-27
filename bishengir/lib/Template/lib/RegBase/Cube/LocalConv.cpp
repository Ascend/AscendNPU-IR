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

  assert(oC_per_g == oC1_per_g * oC0 && "oC_weight must equal oC_output ");
  assert(iC0 == elem_num_per_block && "iC0 must equal elem_num_per_block");
#endif
}

/// Load one input K tile from L1 (cbuf) to L0A and perform img2col.
///
/// The input in cbuf must use the [N, iC1, H, W, iC0] layout. Each invocation
/// loads [1, a k_part slice of iC1 / G, H, W, iC0] for one batch and one
/// group, then converts it to the logical L0A matrix [M, K], where
/// M = oHW and K = k_part. The K tile represents the selected iC1 / G slice
/// expanded over wH * wW * iC0.
template <typename SRC_TYPE>
__aicore__ __attribute__((always_inline)) void
conv_load_l1_to_l0a(__ca__ SRC_TYPE *l0a_buf, memref_t<__cbuf__ SRC_TYPE, 5> *input,
               int64_t k_part_idx, int64_t k_part, int64_t k_part_ceil,
               int64_t batch_idx, int64_t group_idx, int64_t G, int64_t wH,
               int64_t wW, int64_t oHW1, int64_t oHW0, int64_t padL,
               int64_t padR, int64_t padT, int64_t padB, int64_t strideH,
               int64_t strideW, int64_t dilationH, int64_t dilationW) {

  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  __cbuf__ SRC_TYPE *input_ptr = input->aligned + input->offset;

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
    uint64_t fmatrix_input = (uint64_t)(static_cast<uint16_t>(iW)) |
                             ((uint64_t)(static_cast<uint16_t>(iH)) << 16) |
                             ((uint64_t)(static_cast<uint8_t>(padL)) << 32) |
                             ((uint64_t)(static_cast<uint8_t>(padR)) << 40) |
                             ((uint64_t)(static_cast<uint8_t>(padT)) << 48) |
                             ((uint64_t)(static_cast<uint8_t>(padB)) << 56);
    set_fmatrix(fmatrix_input);
    set_padding(static_cast<uint64_t>(0));

    // A5 Load3Dv2 requires the destination stride in L3D_RPT. This path does
    // not transpose the output, so the stride is the number of 16-element
    // fractals covered by mExtension (stepM). Keep repeat disabled by using a
    // single iteration with zero repeat stride.
    uint16_t dstStride = static_cast<uint16_t>(
        (stepM + FRACTAL_BLOCK_NUM - 1) / FRACTAL_BLOCK_NUM);
    uint64_t l3dRptConfig =
        (static_cast<uint64_t>(dstStride) << 32) |
        (static_cast<uint64_t>(1) << 16);
    set_l3d_rpt(l3dRptConfig);

    img2colv2_cbuf_to_ca_intrin_core(
        img2colv2_intrin_args<SRC_TYPE, __ca__ SRC_TYPE>{
            l0a_buf, input_ptr + src_part_offset_a,
            static_cast<uint16_t>(stepK), static_cast<uint16_t>(stepM),
            static_cast<uint16_t>(0) /*posK*/,
            static_cast<uint16_t>(0) /*posM*/, static_cast<uint8_t>(strideW),
            static_cast<uint8_t>(strideH), static_cast<uint8_t>(wW),
            static_cast<uint8_t>(wH), static_cast<uint8_t>(dilationW),
            static_cast<uint8_t>(dilationH), false /*fetchFilterW*/,
            false /*fetchFilterH*/, false /*transpose*/,
            false /*fmatrixCtrl*/, static_cast<uint16_t>(channelSize)});
  }
}

/// Load one weight tile from L1 (cbuf) to L0B.
///
/// The weight in cbuf must use the [iC1 / G, wH, wW, oC, iC0] layout. Viewed
/// as a logical K-by-N matrix, K = iC1 / G * wH * wW * iC0 and N = oC. The oC
/// dimension can be viewed as [oC1, oC0], where oC = oC1 * oC0 and oC0 is the
/// N-axis fractal size. After splitting the blocked dimensions, its physical
/// order is [K1, N1, N0, K0], i.e. zN; here N1/N0 correspond to oC1/oC0.
///
/// Each invocation selects one group's N range and one K tile. Its source
/// region is [a k_part slice of iC1 / G, wH, wW, oC / G, iC0], corresponding
/// logically to a [k_part, oC / G] matrix. The logical values are loaded
/// unchanged; load_cbuf_to_cb converts the L1 zN layout to the L0B nZ layout
/// required by MMAD. Both K and the per-group N dimension have already been
/// padded to fractal boundaries by the preceding IR normalization.
template <typename SRC_TYPE>
__aicore__ __attribute__((always_inline)) void
conv_load_l1_to_l0b(__cb__ SRC_TYPE *l0b_buf, memref_t<__cbuf__ SRC_TYPE, 5> *weight,
               int64_t k_part_idx, int64_t k_part, int64_t k_part_ceil,
               int64_t group_idx, int64_t n_group) {
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t oC = weight->sizes[3];
  int64_t iC0 = weight->sizes[4];
  __cbuf__ SRC_TYPE *weight_ptr = weight->aligned + weight->offset;
  auto src_part_offset_b =
      k_part_idx * k_part * oC + group_idx * n_group * elem_num_per_block;

  if (n_group % FRACTAL_BLOCK_NUM != 0 ||
      oC % FRACTAL_BLOCK_NUM != 0 || k_part_ceil % iC0 != 0) {
    trap();
  }

  uint16_t mStep = static_cast<uint16_t>(n_group / FRACTAL_BLOCK_NUM);
  uint16_t kStep = static_cast<uint16_t>(k_part_ceil / iC0);
  int32_t srcStride = static_cast<int32_t>(oC / FRACTAL_BLOCK_NUM);
  uint16_t dstStride = mStep;
  load_cbuf_to_cb_intrin_core<SRC_TYPE, false>(
      l0b_buf, weight_ptr + src_part_offset_b, static_cast<uint32_t>(0),
      static_cast<uint32_t>(0), mStep, kStep, srcStride, dstStride);
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

  if (m == 0 || k == 0 || n == 0) {
    if (conv_l1_wait_l1a_event != -1) {
      wait_flag(PIPE_MTE2, PIPE_MTE1, conv_l1_wait_l1a_event);
    }
    if (conv_l1_wait_l1b_event != -1) {
      wait_flag(PIPE_MTE2, PIPE_MTE1, conv_l1_wait_l1b_event);
    }
    if (l1a_wait_conv_l1_event != -1) {
      set_flag(PIPE_MTE1, PIPE_MTE2, l1a_wait_conv_l1_event);
    }
    if (l1b_wait_conv_l1_event != -1) {
      set_flag(PIPE_MTE1, PIPE_MTE2, l1b_wait_conv_l1_event);
    }
    return;
  }

  // k_actual should be equal to k_ceil under our case
  auto k_actual = k;
  auto k_ceil = CEIL_FACTOR(k, L1_ALIGN_BYTES / sizeof(SRC_TYPE));

  // L0 buffers
  __cc__ DST_TYPE *output_base = output->aligned + output->offset;
  __ca__ SRC_TYPE *l0a_base = reinterpret_cast<__ca__ SRC_TYPE *>((uintptr_t)0);
  __cb__ SRC_TYPE *l0b_base = reinterpret_cast<__cb__ SRC_TYPE *>((uintptr_t)0);

  int64_t mn_max = m > n ? m : n;
  int64_t elem_num_per_block = L1_ALIGN_BYTES / sizeof(SRC_TYPE);
  int64_t align_block = wH * wW * elem_num_per_block;
  bool enable_double_buffer = true;
  int64_t l0ab_pingpong_buffer_len =
      L0AB_BUFFER_BYTES / 2 / sizeof(SRC_TYPE);
  int64_t k_part = l0ab_pingpong_buffer_len /
                   CEIL_FACTOR(mn_max, FRACTAL_BLOCK_NUM) / align_block *
                   align_block;
  if (k_part == 0) {
    enable_double_buffer = false;
    l0ab_pingpong_buffer_len = L0AB_BUFFER_BYTES / sizeof(SRC_TYPE);
    k_part = l0ab_pingpong_buffer_len /
             CEIL_FACTOR(mn_max, FRACTAL_BLOCK_NUM) / align_block *
             align_block;
  }

  if (k_part == 0) {
    trap();
  }

  if (back_pipe_m_pipe_mte1_db_event0 == -1) {
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  if (back_pipe_m_pipe_mte1_db_event1 == -1) {
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  }

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

        int64_t ping_pong_id = enable_double_buffer ? part_idx % 2 : 0;
        auto mte1_conv_event_id =
            (ping_pong_id == 0)
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

        wait_flag(PIPE_M, PIPE_MTE1, mte1_conv_event_id);
        // load input from L1 to L0A
        if (part_idx == 0 && conv_l1_wait_l1a_event != -1) {
          wait_flag(PIPE_MTE2, PIPE_MTE1, conv_l1_wait_l1a_event);
        }

        conv_load_l1_to_l0a<SRC_TYPE>(l0a_buf, input, k_part_idx, k_part,
                                 k_part_ceil, batch_idx, group_idx, G, wH, wW,
                                 oHW1, oHW0, padL, padR, padT, padB, strideH,
                                 strideW, dilationH, dilationW);

        if (part_idx == part_loop - 1 && l1a_wait_conv_l1_event != -1) {
          set_flag(PIPE_MTE1, PIPE_MTE2, l1a_wait_conv_l1_event);
        }

        // load weight from L1 to L0B
        if (part_idx == 0 && conv_l1_wait_l1b_event != -1) {
          wait_flag(PIPE_MTE2, PIPE_MTE1, conv_l1_wait_l1b_event);
        }

        conv_load_l1_to_l0b<SRC_TYPE>(l0b_buf, weight, k_part_idx, k_part,
                                 k_part_ceil, group_idx, oC_per_g);

        if (part_idx == part_loop - 1 && l1b_wait_conv_l1_event != -1) {
          set_flag(PIPE_MTE1, PIPE_MTE2, l1b_wait_conv_l1_event);
        }

        set_flag(PIPE_MTE1, PIPE_M, mte1_conv_event_id);

        // ---------------------------------
        // MMA accumulate
        // ---------------------------------
        wait_flag(PIPE_MTE1, PIPE_M, mte1_conv_event_id);
        __cc__ DST_TYPE *output_ptr =
            output_base + batch_idx * G * oC1_per_g * oHW1 * oHW0 * oC0 +
            group_idx * oC1_per_g * oHW1 * oHW0 * oC0;

        bool init_c = init && !k_part_idx;

        mad_intrin_core(mmad_intrin_args<SRC_TYPE, DST_TYPE>{
            output_ptr, l0a_buf, l0b_buf, static_cast<uint16_t>(m),
            static_cast<uint16_t>(k_part_ceil), static_cast<uint16_t>(n),
            static_cast<uint8_t>(0) /*unitFlag*/, true /*disableGemv*/,
            false /*cmatrixSource*/, init_c});

        if (m / FRACTAL_BLOCK_NUM * n / FRACTAL_BLOCK_NUM < 10) {
          pipe_barrier(PIPE_M);
        }
        set_flag(PIPE_M, PIPE_MTE1, mte1_conv_event_id);
      }
    }
  }

  if (back_pipe_m_pipe_mte1_db_event0 == -1) {
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  if (back_pipe_m_pipe_mte1_db_event1 == -1) {
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  }
}

extern "C" {
REGISTER_CONV2D_GROUP(cbuf, cc, 5, float, float);
REGISTER_CONV2D_GROUP(cbuf, cc, 5, half, float);
REGISTER_CONV2D_GROUP(cbuf, cc, 5, bfloat16_t, float);
}

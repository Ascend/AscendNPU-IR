/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
#if defined(__DAV_M300__) || defined(__DAV_C310__)
#include "Macro/MacroUtils.h"
#include "Macro/common.h"
#include "Macro/core/gemm_core.h"
#include "Macro/tiling.h"

#define ALIGN_ROUNDUP(base, align)                                             \
  (((base) + (align) - 1) - ((base) + (align) - 1) % (align))

template <uint16_t M, uint16_t N, uint16_t K>
[aicore] void matmul_impl_tiled(__gm__ half *__restrict__ a,
                                __gm__ half *__restrict__ b,
                                __gm__ half *__restrict__ result) {
  constexpr uint16_t FRACDIM = 16;
  constexpr uint16_t M_F = ALIGN_ROUNDUP(M, FRACDIM);
  constexpr uint16_t N_F = ALIGN_ROUNDUP(N, FRACDIM);
  constexpr uint16_t K_F = ALIGN_ROUNDUP(K, FRACDIM);

  // A will have tiles with size M_F / 64 by K_F
  constexpr uint16_t TILE_SIZE_A = (K == 1024) ? M_F / 64 : M_F;

  // B will have tiles with size K_F by N_F / 64
  constexpr uint16_t TILE_SIZE_B = (K == 1024) ? N_F / 64 : N_F;
  constexpr uint16_t DST_L1A_C0_STRIDE = (K == 1024) ? TILE_SIZE_A : 16;
  constexpr uint16_t DST_L1B_C0_STRIDE = (K == 1024) ? N : 16;

  constexpr uint32_t MATRIX_SIZE_A = K_F * sizeof(half) * TILE_SIZE_A;
  constexpr uint32_t MATRIX_SIZE_B = K_F * sizeof(half) * TILE_SIZE_B;

  constexpr uint32_t CBUF_PAD_SIZE = 64;
  constexpr uint32_t CBUF_ALIGN = 32;
  constexpr uint32_t ADDR_L1A = 0;
  constexpr uint32_t ADDR_L1B = ALIGN_ROUNDUP(MATRIX_SIZE_A, CBUF_ALIGN);

  __cbuf__ half *l1_a = (__cbuf__ half *)get_imm(ADDR_L1A);
  __cbuf__ half *l1_b = (__cbuf__ half *)get_imm(ADDR_L1B);
  __ca__ half *l0a_a = (__ca__ half *)get_imm(0);
  __cb__ half *l0b_b = (__cb__ half *)get_imm(0);
  __cc__ float *l0c_o_float = (__cc__ float *)get_imm(0);

  // Parameters for L0c to GM copy out
  uint16_t MSize = TILE_SIZE_A;
  uint16_t NSize = TILE_SIZE_B;
  uint32_t dstStride_dst_D = N_F;
  uint16_t srcStride = TILE_SIZE_B;
  uint64_t ndNum = 1;
  uint64_t src_nd_stride = 1;
  uint64_t dst_nd_stride = 1;

  uint8_t UnitFlagMode = 0;
  uint64_t QuantPRE = 1; // NoQuant;
  uint8_t ReLUPRE = 0;
  bool channelSplit = false;
  bool NZ2ND_EN = true;

  uint64_t config = 0, nd_para = 0;
  nd_para = nd_para | (ndNum & 0xffff);
  nd_para = nd_para | ((src_nd_stride & 0xffff) << 16);
  nd_para = nd_para | ((dst_nd_stride & 0xffff) << 32);
  set_nd_para(nd_para);

  // Parameters for mad
  config |= (TILE_SIZE_A & 0x0fff) << 0;
  config |= (K & 0x0fff) << 12;
  config |= (TILE_SIZE_B & 0x0fff) << 24;
  config |= 1ull << 63;

  // For each A tile, multiply all B tiles once to get a TILE_SIZE_A by N_F row
  // in C
  for (uint16_t a_row = 0; a_row < M_F / TILE_SIZE_A; a_row++) {
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    for (uint16_t b_col = 0; b_col < N_F / TILE_SIZE_B; b_col++) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0); // backward 1

      // Only load A tile once per outer loop iteration
      if (!b_col) {
#if defined(__DAV_C310__)
        uint64_t config =
            ((uint64_t)(TILE_SIZE_A / 16)) | ((uint64_t)1) << 16 |
            ((uint64_t)((K == 1024) ? DST_L1A_C0_STRIDE : K_F)) << 32 |
            ((uint64_t)((K == 1024) ? K_F : DST_L1A_C0_STRIDE)) << 48;
        set_mte2_nz_para(config);
        copy_gm_to_cbuf_multi_nd2nz(l1_a, a + a_row * K * TILE_SIZE_A, 0,
                                    K * sizeof(half), 0, 16, K,
                                    K * 16 * sizeof(half), false);
#else
        copy_gm_to_cbuf_multi_nd2nz_b16(l1_a, a + a_row * K * TILE_SIZE_A, 0,
                                        TILE_SIZE_A / 16, 16, K, K * 16, K,
                                        DST_L1A_C0_STRIDE, 1, K_F * 16);
#endif
      }

// Copy tile for B to L1, K_F by TILE_SIZE_B
#if defined(__DAV_C310__)
      uint64_t config =
          ((uint64_t)(K_F / 16)) | ((uint64_t)1) << 16 |
          ((uint64_t)((K == 1024) ? DST_L1B_C0_STRIDE : TILE_SIZE_B)) << 32 |
          ((uint64_t)((K == 1024) ? TILE_SIZE_B : DST_L1B_C0_STRIDE)) << 48;
      set_mte2_nz_para(config);
      copy_gm_to_cbuf_multi_nd2nz(l1_b, b + b_col * TILE_SIZE_B, 0,
                                  N * sizeof(half), 0, 16, TILE_SIZE_B,
                                  N * 16 * sizeof(half), false);
#else
      copy_gm_to_cbuf_multi_nd2nz_b16(l1_b, b + b_col * TILE_SIZE_B, 0,
                                      K_F / 16, 16, TILE_SIZE_B, N * 16, N,
                                      DST_L1B_C0_STRIDE, 1, TILE_SIZE_B * 16);
#endif
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0); // backward 2

      if (!b_col) {
#if defined(__DAV_C310__)
        load_cbuf_to_ca(l0a_a, l1_a, 0, 0, 1, TILE_SIZE_A * K_F / (16 * 16), 1,
                        1, false);
#else
        load_cbuf_to_ca(l0a_a, l1_a, 0, TILE_SIZE_A * K_F / (16 * 16), 1, 0,
                        false, addr_cal_mode_t::inc);
#endif
      }
#if defined(__DAV_C310__)
      load_cbuf_to_cb(l0b_b, l1_b, 0, 0, K_F / 16, TILE_SIZE_B / 16, K_F / 16,
                      TILE_SIZE_B / 16, true);
#else
      load_cbuf_to_cb_transpose(l0b_b, l1_b, 0, K_F * TILE_SIZE_B / (16 * 16),
                                1, 0, false, 0);
#endif

      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0); // backward 1

      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0); // backward 3
#if defined(__DAV_C310__)
      mad(l0c_o_float, l0a_a, l0b_b, TILE_SIZE_A, K, TILE_SIZE_B, 0, false,
          false, true);
#else
      mad(l0c_o_float, l0a_a, l0b_b, config);
#endif

      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0); // backward 2

#if defined(__DAV_C310__)
      uint64_t nd_para = 0;
      nd_para = nd_para | (1 & 0xffff);
      nd_para = nd_para | ((((uint64_t)(1)) & 0xffff) << 16);
      nd_para = nd_para | ((((uint64_t)(1)) & 0xffff) << 32);
      set_nd_para(nd_para);
      copy_matrix_cc_to_gm((__gm__ half *)(result + a_row * N * TILE_SIZE_A +
                                           b_col * TILE_SIZE_B),
                           (__cc__ float *)l0c_o_float, 0, NSize, MSize,
                           (uint32_t)N_F, (uint16_t)MSize, 0, 0, 0,
                           QuantMode_t::F322F16, 0, false, true, 0, 0, false,
                           false, 0, false, false, true, false, false, false);
#else
      copy_matrix_cc_to_gm((__gm__ half *)(result + a_row * N * TILE_SIZE_A +
                                           b_col * TILE_SIZE_B),
                           (__cc__ float *)l0c_o_float,
                           /*uint8_t sid*/ 0, NSize, MSize, dstStride_dst_D,
                           srcStride,
                           /*uint8_t clip_relu_pre*/ 0, UnitFlagMode, QuantPRE,
                           ReLUPRE, channelSplit, NZ2ND_EN,
                           /*bool C0_Pad_EN*/ false);
#endif

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0); // backward 3
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
}

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__
    __attribute__((always_inline)) void matmul_Xbias_Xdescale_XtransposeA_XtransposeB(
        memref_t<__gm__ ElementA, 2> *A, memref_t<__gm__ ElementB, 2> *B,
        memref_t<__gm__ ElementC, 2> *C) {
  auto a_ptr = A->aligned + A->offset;
  auto b_ptr = B->aligned + B->offset;
  auto c_ptr = C->aligned + C->offset;
  int64_t m = A->sizes[0];
  int64_t k = A->sizes[1];
  int64_t n = B->sizes[1];

  if constexpr (sizeof(ElementA) == 2 && sizeof(ElementB) == 2 &&
                sizeof(ElementC) == 2) {
    if (m == 64 && k == 64 && n == 64) {
      matmul_impl_tiled<64, 64, 64>(reinterpret_cast<__gm__ ElementA *>(a_ptr),
                                    reinterpret_cast<__gm__ ElementB *>(b_ptr),
                                    reinterpret_cast<__gm__ ElementC *>(c_ptr));
    } else if (m == 1024 && k == 1024 && n == 1024) {
      matmul_impl_tiled<1024, 1024, 1024>(
          reinterpret_cast<__gm__ ElementA *>(a_ptr),
          reinterpret_cast<__gm__ ElementB *>(b_ptr),
          reinterpret_cast<__gm__ ElementC *>(c_ptr));
    }
  }
}

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_transposeA_XtransposeB(memref_t<__gm__ ElementA, 2> *A,
                                             memref_t<__gm__ ElementB, 2> *B,
                                             memref_t<__gm__ ElementC, 2> *C) {
  auto a_ptr = A->aligned + A->offset;
  auto b_ptr = B->aligned + B->offset;
  auto c_ptr = C->aligned + C->offset;
  int64_t m = A->sizes[1];
  int64_t k = A->sizes[0];
  int64_t n = B->sizes[1];
  // TODO
}

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_XtransposeA_transposeB(memref_t<__gm__ ElementA, 2> *A,
                                             memref_t<__gm__ ElementB, 2> *B,
                                             memref_t<__gm__ ElementC, 2> *C) {
  auto a_ptr = A->aligned + A->offset;
  auto b_ptr = B->aligned + B->offset;
  auto c_ptr = C->aligned + C->offset;
  int64_t m = A->sizes[0];
  int64_t k = A->sizes[1];
  int64_t n = B->sizes[0];
  // TODO
}

template <typename ElementA, typename ElementB, typename ElementC>
__aicore__ __attribute__((always_inline)) void
matmul_Xbias_Xdescale_transposeA_transposeB(memref_t<__gm__ ElementA, 2> *A,
                                            memref_t<__gm__ ElementB, 2> *B,
                                            memref_t<__gm__ ElementC, 2> *C) {
  auto a_ptr = A->aligned + A->offset;
  auto b_ptr = B->aligned + B->offset;
  auto c_ptr = C->aligned + C->offset;
  int64_t m = A->sizes[1];
  int64_t k = A->sizes[0];
  int64_t n = B->sizes[0];
  // TODO
}

extern "C" {
//===-------------------------------------------------------------------===//
// matmul, Xbias, Xdescale
//===-------------------------------------------------------------------===//
REGISTE_MATMUL_XB_XD(XtransposeA, XtransposeB, half, half, half);
REGISTE_MATMUL_XB_XD(transposeA, XtransposeB, half, half, half);
REGISTE_MATMUL_XB_XD(XtransposeA, transposeB, half, half, half);
REGISTE_MATMUL_XB_XD(transposeA, transposeB, half, half, half);
REGISTE_MATMUL_XB_XD(XtransposeA, XtransposeB, int8_t, int8_t, half);
REGISTE_MATMUL_XB_XD(transposeA, XtransposeB, int8_t, int8_t, half);
REGISTE_MATMUL_XB_XD(XtransposeA, transposeB, int8_t, int8_t, half);
REGISTE_MATMUL_XB_XD(transposeA, transposeB, int8_t, int8_t, half);
}
#endif
#endif
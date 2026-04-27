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

#include "RegBase/ReductionUtils.h"
#include "Utils.h"
#include <type_traits>

// Helper type trait
template <typename T>
inline constexpr bool isB64Type =
    std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>;

// Generic Reduction Template (Aligned)
template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1,
          typename AlignedHint = __cce_simd::AlignedHint,
          typename Enable = void>
struct ReduceImpl {
  static inline __aiv__ __attribute__((always_inline)) void
  reduce_leading_dim(__ubuf__ T0 *data_UB, __ubuf__ T0 *reduce_val_UB,
                     __ubuf__ int32_t *reduce_idx_UB, uint16_t loop_a,
                     uint16_t dim_a, uint16_t loop_r, uint16_t src_r_stride) {
    unsigned int plt_size = dim_a;
    const uint16_t ele_per_VL = VL_IN_BYTE / sizeof(T0);
    const uint32_t ele_per_VL_B32 = VL_IN_BYTE / sizeof(int32_t);
    VectorReg<T0> cur_val, acc_val;
    VectorReg<T1> cur_idx, acc_idx;

    vector_bool cmp_mask;

    for (uint16_t a = 0; a < loop_a; ++a) {

      vbr(acc_idx, 0);
      vector_bool mask;
      CREATE_MASK_BY_SIZE(mask, T0, plt_size);
      vlds(acc_val, (__ubuf__ T0 *)data_UB, a * ele_per_VL, NORM);
      for (uint16_t i = 0; i < loop_r; ++i) {

        vbr(cur_idx, i + 1);
        vlds(cur_val, (__ubuf__ T0 *)data_UB,
             (i + 1) * src_r_stride + a * ele_per_VL, NORM);
        if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT) 
            vcmp_gt(cmp_mask, cur_val, acc_val, mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_ge(cmp_mask, cur_val, acc_val, mask);
        } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT)
            vcmp_lt(cmp_mask, cur_val, acc_val, mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_le(cmp_mask, cur_val, acc_val, mask);
        }
        vsel(acc_val, cur_val, acc_val, cmp_mask);
        vsel(acc_idx, cur_idx, acc_idx, cmp_mask);
      }

      if constexpr (std::is_same_v<T0, int16_t> || std::is_same_v<T0, half>)
        vsts(acc_val, (__ubuf__ T0 *)reduce_val_UB, a * ele_per_VL, NORM_B16,
             mask);
      else
        vsts(acc_val, (__ubuf__ T0 *)reduce_val_UB, a * ele_per_VL, NORM_B32,
             mask);

      if constexpr (std::is_same_v<T1, int16_t>) {
        unsigned int plt_size_b32 = dim_a;
        vector_s32 lower_half_idx, higher_half_idx;
        vector_bool lower_mask, higher_mask;
        vunpack(lower_half_idx, acc_idx, LOWER);
        lower_mask = plt_b32(plt_size_b32, POST_UPDATE);
        vsts(lower_half_idx, (__ubuf__ int32_t *)reduce_idx_UB, a * ele_per_VL,
             NORM_B32, lower_mask);
        vunpack(higher_half_idx, acc_idx, HIGHER);
        higher_mask = plt_b32(plt_size_b32, POST_UPDATE);
        vsts(higher_half_idx, (__ubuf__ int32_t *)reduce_idx_UB,
             a * ele_per_VL + ele_per_VL_B32, NORM_B32, higher_mask);
      } else
        vsts(acc_idx, (__ubuf__ int32_t *)reduce_idx_UB, a * ele_per_VL,
             NORM_B32, mask);
    }
  }

  static inline __aiv__ __attribute__((always_inline)) void
  reduce_last_axis_contiguous_vec(
      __ubuf__ T0 *data_UB, __ubuf__ T0 *reduce_val_UB,
      __ubuf__ int32_t *reduce_idx_UB, uint16_t dim_a, uint16_t dim_r,
      uint16_t src_a_stride, uint16_t dst_val_stride, uint16_t dst_idx_stride,
      T0 initvalue, uint16_t main_loop, uint16_t ele_per_VL) {
    constexpr int dsize = sizeof(T0);

    VectorReg<T0> cur_val, acc_val, reduced_val, brc_reduced_val;
    VectorReg<T1> cur_idx, acc_idx, helper_idx, reduced_idx, tmp_idx;
    VectorReg<int32_t> final_idx;

    vector_bool full_mask, two_mask, one_mask, cmp_mask;
    CREATE_MASK_BY_PAT(full_mask, T0, PAT_ALL);
    CREATE_MASK_BY_PAT(two_mask, T1, PAT_VL2);
    CREATE_MASK_BY_PAT(one_mask, T1, PAT_VL1);

    vci(tmp_idx, 0);
    vbr(helper_idx, 1);
    vdup(helper_idx, ele_per_VL, full_mask, MODE_MERGING);

    for (uint16_t a = 0; a < dim_a; ++a) {
      unsigned int plt_size = dim_r;
      uint16_t data_UB_offset = a * src_a_stride;

      // TODO: preload to get rid of vbr(initvalue)
      vbr(acc_val, initvalue);
      vbr(acc_idx, 0);
      for (uint16_t i = 0; i < main_loop; ++i) {
        vector_bool cur_mask;
        CREATE_MASK_BY_SIZE(cur_mask, T0, plt_size);
        vlds(cur_val, data_UB, data_UB_offset + i * ele_per_VL, NORM);
        vbr(cur_idx, i);

        if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT) 
            vcmp_gt(cmp_mask, cur_val, acc_val, cur_mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_ge(cmp_mask, cur_val, acc_val, cur_mask);
        } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT)
            vcmp_lt(cmp_mask, cur_val, acc_val, cur_mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_le(cmp_mask, cur_val, acc_val, cur_mask);
        }
        vsel(acc_val, cur_val, acc_val, cmp_mask);
        vsel(acc_idx, cur_idx, acc_idx, cmp_mask);
      }

      if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX)
        vcmax(reduced_val, acc_val, full_mask);
      else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX)
        vcmin(reduced_val, acc_val, full_mask);

      if constexpr (std::is_same_v<T0, int16_t> || std::is_same_v<T0, half>)
        vsts(reduced_val, reduce_val_UB, a * dst_val_stride, ONEPT_B16,
             full_mask);
      else if constexpr (dsize == BYTES_B32)
        vsts(reduced_val, reduce_val_UB, a * dst_val_stride, ONEPT_B32,
             full_mask);

      vdup(brc_reduced_val, reduced_val, full_mask, POS_LOWEST, MODE_ZEROING);
      vcmp_eq(cmp_mask, brc_reduced_val, acc_val, full_mask);

      vmul(acc_idx, acc_idx, helper_idx, cmp_mask);
      vadd(acc_idx, acc_idx, tmp_idx, cmp_mask);

      if constexpr (TIE_BREAK == TieBreak::LEFT) 
        vcmin(reduced_idx, acc_idx, cmp_mask);
      else if constexpr (TIE_BREAK == TieBreak::RIGHT)
        vcmax(reduced_idx, acc_idx, cmp_mask);
        
      vcadd(final_idx, reduced_idx, one_mask);
      vsts(final_idx, reduce_idx_UB, a * dst_idx_stride, ONEPT_B32, full_mask);
    }
  }
};

// Generic Reduction Template (Unaligned)
template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
struct ReduceImpl<OP, TIE_BREAK, T0, T1, __cce_simd::UnAlignedHint,
                  typename std::enable_if<!isB64Type<T0>>::type> {
  static inline __aiv__ __attribute__((always_inline)) void
  reduce_leading_dim(__ubuf__ T0 *data_UB, __ubuf__ T0 *reduce_val_UB,
                     __ubuf__ int32_t *reduce_idx_UB, uint16_t loop_a,
                     uint16_t dim_a, uint16_t loop_r, uint16_t src_r_stride) {
#if defined(__DAV_C310__)
    unsigned int plt_size = dim_a;
    const uint16_t ele_per_VL = VL_IN_BYTE / sizeof(T0);
    const uint32_t ele_per_VL_B32 = VL_IN_BYTE / sizeof(int32_t);
    VectorReg<T0> cur_val, acc_val;
    VectorReg<T1> cur_idx, acc_idx;

    vector_bool cmp_mask;

    for (uint16_t a = 0; a < loop_a; ++a) {
      vbr(acc_idx, 0);
      vector_bool mask;
      CREATE_MASK_BY_SIZE(mask, T0, plt_size);

      vload<__cce_simd::UnAlignedHint>(
          acc_val, (__ubuf__ T0 *)data_UB + a * ele_per_VL, ADDRESS_UNALIGNED);

      for (uint16_t i = 0; i < loop_r; ++i) {
        vbr(cur_idx, i + 1);
        vload<__cce_simd::UnAlignedHint>(
            cur_val,
            (__ubuf__ T0 *)data_UB + (i + 1) * src_r_stride + a * ele_per_VL,
            ADDRESS_UNALIGNED);

        if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT) 
            vcmp_gt(cmp_mask, cur_val, acc_val, mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_ge(cmp_mask, cur_val, acc_val, mask);
        } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT)
            vcmp_lt(cmp_mask, cur_val, acc_val, mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_le(cmp_mask, cur_val, acc_val, mask);
        }
        vsel(acc_val, cur_val, acc_val, cmp_mask);
        vsel(acc_idx, cur_idx, acc_idx, cmp_mask);
      }

      // TODO: use aligned vsts for now until cur_len calc for vstore fixed
      if constexpr (std::is_same_v<T0, int16_t> || std::is_same_v<T0, half>)
        vsts(acc_val, (__ubuf__ T0 *)reduce_val_UB, a * ele_per_VL, NORM_B16,
             mask);
      else
        vsts(acc_val, (__ubuf__ T0 *)reduce_val_UB, a * ele_per_VL, NORM_B32,
             mask);

      // TODO: use aligned vsts for now until cur_len calc for vstore fixed
      if constexpr (std::is_same_v<T1, int16_t>) {
        unsigned int plt_size_b32 = dim_a;
        vector_s32 lower_half_idx, higher_half_idx;
        vector_bool lower_mask, higher_mask;
        vunpack(lower_half_idx, acc_idx, LOWER);
        lower_mask = plt_b32(plt_size_b32, POST_UPDATE);
        vsts(lower_half_idx, (__ubuf__ int32_t *)reduce_idx_UB, a * ele_per_VL,
             NORM_B32, lower_mask);
        vunpack(higher_half_idx, acc_idx, HIGHER);
        higher_mask = plt_b32(plt_size_b32, POST_UPDATE);
        vsts(higher_half_idx, (__ubuf__ int32_t *)reduce_idx_UB,
             a * ele_per_VL + ele_per_VL_B32, NORM_B32, higher_mask);
      } else
        vsts(acc_idx, (__ubuf__ int32_t *)reduce_idx_UB, a * ele_per_VL,
             NORM_B32, mask);
    }
#endif
  }

  static inline __aiv__ __attribute__((always_inline)) void
  reduce_last_axis_contiguous_vec(
      __ubuf__ T0 *data_UB, __ubuf__ T0 *reduce_val_UB,
      __ubuf__ int32_t *reduce_idx_UB, uint16_t dim_a, uint16_t dim_r,
      uint16_t src_a_stride, uint16_t dst_val_stride, uint16_t dst_idx_stride,
      T0 initvalue, uint16_t main_loop, uint16_t ele_per_VL) {
#if defined(__DAV_C310__)
    constexpr int dsize = sizeof(T0);

    VectorReg<T0> cur_val, acc_val, reduced_val, brc_reduced_val;
    VectorReg<T1> cur_idx, acc_idx, helper_idx, reduced_idx, tmp_idx;
    VectorReg<int32_t> final_idx;

    vector_bool full_mask, two_mask, one_mask, cmp_mask;
    CREATE_MASK_BY_PAT(full_mask, T0, PAT_ALL);
    CREATE_MASK_BY_PAT(two_mask, T1, PAT_VL2);
    CREATE_MASK_BY_PAT(one_mask, T1, PAT_VL1);

    vci(tmp_idx, 0);
    vbr(helper_idx, 1);
    vdup(helper_idx, ele_per_VL, full_mask, MODE_MERGING);

    for (uint16_t a = 0; a < dim_a; ++a) {
      unsigned int plt_size = dim_r;
      uint16_t data_UB_offset = a * src_a_stride;

      // TODO: preload to get rid of vbr(initvalue)
      vbr(acc_val, initvalue);
      vbr(acc_idx, 0);
      for (uint16_t i = 0; i < main_loop; ++i) {
        vector_bool cur_mask;
        CREATE_MASK_BY_SIZE(cur_mask, T0, plt_size);

        vload<__cce_simd::UnAlignedHint>(
            cur_val, (__ubuf__ T0 *)data_UB + data_UB_offset + i * ele_per_VL,
            ADDRESS_UNALIGNED);

        vbr(cur_idx, i);

        if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT) 
            vcmp_gt(cmp_mask, cur_val, acc_val, cur_mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_ge(cmp_mask, cur_val, acc_val, cur_mask);
        } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT)
            vcmp_lt(cmp_mask, cur_val, acc_val, cur_mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_le(cmp_mask, cur_val, acc_val, cur_mask);
        }
        vsel(acc_val, cur_val, acc_val, cmp_mask);
        vsel(acc_idx, cur_idx, acc_idx, cmp_mask);
      }

      if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX)
        vcmax(reduced_val, acc_val, full_mask);
      else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX)
        vcmin(reduced_val, acc_val, full_mask);

      // ONEPT mode vsts for scalar store
      if constexpr (std::is_same_v<T0, int16_t> || std::is_same_v<T0, half>)
        vsts(reduced_val, reduce_val_UB, a * dst_val_stride, ONEPT_B16,
             full_mask);
      else if constexpr (dsize == BYTES_B32)
        vsts(reduced_val, reduce_val_UB, a * dst_val_stride, ONEPT_B32,
             full_mask);

      vdup(brc_reduced_val, reduced_val, full_mask, POS_LOWEST, MODE_ZEROING);
      vcmp_eq(cmp_mask, brc_reduced_val, acc_val, full_mask);

      vmul(acc_idx, acc_idx, helper_idx, cmp_mask);
      vadd(acc_idx, acc_idx, tmp_idx, cmp_mask);

      if constexpr (TIE_BREAK == TieBreak::LEFT) 
        vcmin(reduced_idx, acc_idx, cmp_mask);
      else if constexpr (TIE_BREAK == TieBreak::RIGHT)
        vcmax(reduced_idx, acc_idx, cmp_mask);
        
      vcadd(final_idx, reduced_idx, one_mask);

      // ONEPT mode vsts for scalar store
      vsts(final_idx, reduce_idx_UB, a * dst_idx_stride, ONEPT_B32, full_mask);
    }
#endif
  }
};

// i64 Specialized Reduction Template (Aligned/Unaligned)
template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1, typename AlignedHint>
struct ReduceImpl<OP, TIE_BREAK, T0, T1, AlignedHint,
                  typename std::enable_if<isB64Type<T0>>::type> {
  static inline __aiv__ __attribute__((always_inline)) void
  reduce_leading_dim(__ubuf__ T0 *data_UB, __ubuf__ T0 *reduce_val_UB,
                     __ubuf__ int32_t *reduce_idx_UB, uint16_t loop_a,
                     uint16_t dim_a, uint16_t loop_r, uint16_t src_r_stride) {
#if defined(__DAV_C310__)
    unsigned int plt_size = dim_a;
    const uint16_t ele_per_VL =
        2 * VL_IN_BYTE / sizeof(T0); // using 2x32bit vec registers for 64 bit
    const uint32_t ele_per_VL_B32 = VL_IN_BYTE / sizeof(int32_t);

    VectorReg<T0> cur_val, acc_val;
    VectorReg<T1> cur_idx, acc_idx;

    vector_bool cmp_mask;

    for (uint16_t a = 0; a < loop_a; ++a) {
      vbr(acc_idx, 0);
      vector_bool mask;
      mask = plt_2xvl_b64(plt_size, POST_UPDATE);

      if constexpr (AlignedHint::value == ADDRESS_ALIGNED)
        vlds(acc_val, (__ubuf__ T0 *)data_UB, a * ele_per_VL);
      else
        vload<AlignedHint>(acc_val, (__ubuf__ T0 *)data_UB + a * ele_per_VL,
                           AlignedHint{});

      for (uint16_t i = 0; i < loop_r; ++i) {
        vbr(cur_idx, i + 1);
        if constexpr (AlignedHint::value == ADDRESS_ALIGNED)
          vlds(cur_val, (__ubuf__ T0 *)data_UB,
               (i + 1) * src_r_stride + a * ele_per_VL);
        else
          vload<AlignedHint>(cur_val,
                             (__ubuf__ T0 *)data_UB + (i + 1) * src_r_stride +
                                 a * ele_per_VL,
                             AlignedHint{});

        if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT) 
            vcmp_gt(cmp_mask, cur_val, acc_val, mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_ge(cmp_mask, cur_val, acc_val, mask);
        } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT)
            vcmp_lt(cmp_mask, cur_val, acc_val, mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_le(cmp_mask, cur_val, acc_val, mask);
        }

        vsel(acc_val, cur_val, acc_val, cmp_mask);
        vsel(acc_idx, cur_idx, acc_idx, cmp_mask);
      }

      // TODO: use aligned vsts for now until cur_len calc for vstore fixed
      vsts(acc_val, (__ubuf__ T0 *)reduce_val_UB,
           static_cast<int32_t>(a) * ele_per_VL, mask);

      vsts(acc_idx, (__ubuf__ int32_t *)reduce_idx_UB,
           static_cast<int32_t>(a) * ele_per_VL, NORM_B32, mask);
    }
#endif
  }

  static inline __aiv__ __attribute__((always_inline)) void
  reduce_last_axis_contiguous_vec(
      __ubuf__ T0 *data_UB, __ubuf__ T0 *reduce_val_UB,
      __ubuf__ int32_t *reduce_idx_UB, uint16_t dim_a, uint16_t dim_r,
      uint16_t src_a_stride, uint16_t dst_val_stride, uint16_t dst_idx_stride,
      T0 initvalue, uint16_t main_loop, uint16_t ele_per_VL) {
#if defined(__DAV_C310__)
    constexpr int dsize = sizeof(T0);

    VectorReg<T0> cur_val, acc_val, reduced_val, brc_reduced_val;
    VectorReg<T1> cur_idx, acc_idx, helper_idx, reduced_idx, tmp_idx, 
                  tmp_reduced_idx, brc_tmp_reduced_idx;
    VectorReg<int32_t> final_idx;

    vector_bool full_mask, two_mask, one_mask, cmp_mask;
    CREATE_MASK_BY_PAT(full_mask, T0, PAT_ALL);
    CREATE_MASK_BY_PAT(two_mask, T1, PAT_VL2);
    CREATE_MASK_BY_PAT(one_mask, T1, PAT_VL1);

    ele_per_VL *= 2; // using 2x32bit vec registers for 64 bit

    vci(tmp_idx, 0);
    vbr(helper_idx, 1);
    vdup(helper_idx, ele_per_VL, full_mask, MODE_MERGING);

    for (uint16_t a = 0; a < dim_a; ++a) {
      unsigned int plt_size = dim_r;
      uint16_t data_UB_offset = a * src_a_stride;

      // Preload to get rid of vbr(acc_val, initvalue)
      vbr(acc_val, initvalue);
      vbr(acc_idx, 0);

      for (uint16_t i = 0; i < main_loop; ++i) {
        vector_bool cur_mask;
        cur_mask = plt_2xvl_b64(plt_size, POST_UPDATE);

        if constexpr (AlignedHint::value == ADDRESS_ALIGNED)
          vlds(cur_val, data_UB, data_UB_offset + i * ele_per_VL);
        else
          vload<AlignedHint>(
              cur_val, (__ubuf__ T0 *)data_UB + data_UB_offset + i * ele_per_VL,
              AlignedHint{});

        vbr(cur_idx, i);

        if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT) 
            vcmp_gt(cmp_mask, cur_val, acc_val, cur_mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_ge(cmp_mask, cur_val, acc_val, cur_mask);
        } else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX) {
          if constexpr (TIE_BREAK == TieBreak::LEFT)
            vcmp_lt(cmp_mask, cur_val, acc_val, cur_mask);
          else if constexpr (TIE_BREAK == TieBreak::RIGHT)
            vcmp_le(cmp_mask, cur_val, acc_val, cur_mask);
        }
        vsel(acc_val, cur_val, acc_val, cmp_mask);
        vsel(acc_idx, cur_idx, acc_idx, cmp_mask);
      }

      if constexpr (OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX)
        vcmax(reduced_val, acc_val, full_mask);
      else if constexpr (OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX)
        vcmin(reduced_val, acc_val, full_mask);

      // Emulate ONEPT mode with vstore, somehow only works for VSTORE_B64
      vstore<__cce_simd::UnAlignedHint>((__ubuf__ T0 *)reduce_val_UB +
                                            a * dst_val_stride,
                                        reduced_val, 1, ADDRESS_UNALIGNED);

      vdup(brc_reduced_val, reduced_val, full_mask, POS_LOWEST, MODE_ZEROING);
      vcmp_eq(cmp_mask, brc_reduced_val, acc_val, full_mask);

      vmul(acc_idx, acc_idx, helper_idx, cmp_mask);
      vadd(acc_idx, acc_idx, tmp_idx, cmp_mask);

      if constexpr (TIE_BREAK == TieBreak::LEFT) 
        vcmin(reduced_idx, acc_idx, cmp_mask);
      else if constexpr (TIE_BREAK == TieBreak::RIGHT)
        vcmax(reduced_idx, acc_idx, cmp_mask);
      
      vcadd(final_idx, reduced_idx, one_mask);
      vsts(final_idx, reduce_idx_UB, a * dst_idx_stride, ONEPT_B32, full_mask);
    }
#endif
  }
};

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
inline __aiv__ __attribute__((always_inline)) void
vf_wrapper_reduce_ar_last_axis_contiguous(
    __ubuf__ T0 *data_UB, __ubuf__ T0 *reduce_val_UB,
    __ubuf__ int32_t *reduce_idx_UB, uint16_t dim_a, uint16_t dim_r,
    uint16_t src_a_stride, uint16_t dst_val_stride, uint16_t dst_idx_stride,
    T0 initvalue) {
  constexpr int dsize = sizeof(T0);
  uint16_t main_loop = CEIL_DIV(dim_r * dsize, VL_IN_BYTE);
  uint16_t ele_per_VL = VL_IN_BYTE / dsize;

  __VEC_SCOPE__ {
// calls helper vector function to reduce (a,r) -> (a,1)
// last dimension must be contiguous

// TODO: in current implementation, the index register is forced to have
// same length as the data(b16 / b32). for b16 it might cause overflow when
// input size gets larger

// TODO: now it selects the rightmost index (right break tie), but we need
// to consider left break tie is this info can be passed into the template
// function in certain way(extra input,distinct func name,etc)

// TODO: if the OP is tiled along the reduction axis, then a start index
// maybe needed
#if defined(__DAV_C310__)
    if (isAddress32ByteAligned(data_UB + src_a_stride) &&
        isAddress32ByteAligned(data_UB)) {
      ReduceImpl<OP, TIE_BREAK, T0, T1>::reduce_last_axis_contiguous_vec(
          data_UB, reduce_val_UB, reduce_idx_UB, dim_a, dim_r, src_a_stride,
          dst_val_stride, dst_idx_stride, initvalue, main_loop, ele_per_VL);
    } else {
      ReduceImpl<OP, TIE_BREAK, T0, T1, __cce_simd::UnAlignedHint>::
          reduce_last_axis_contiguous_vec(
              data_UB, reduce_val_UB, reduce_idx_UB, dim_a, dim_r, src_a_stride,
              dst_val_stride, dst_idx_stride, initvalue, main_loop, ele_per_VL);
    }
#else
    // TODO: if the OP is tiled along the reduction axis, then a start index
    // maybe needed
    ReduceImpl<OP, TIE_BREAK, T0, T1>::reduce_last_axis_contiguous_vec(
        data_UB, reduce_val_UB, reduce_idx_UB, dim_a, dim_r, src_a_stride,
        dst_val_stride, dst_idx_stride, initvalue, main_loop, ele_per_VL);
#endif
  }
}

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
inline __aiv__ __attribute__((always_inline)) void
vf_wrapper_reduce_ra_last_axis_contiguous(__ubuf__ T0 *data_UB,
                                          __ubuf__ T0 *reduce_val_UB,
                                          __ubuf__ int32_t *reduce_idx_UB,
                                          uint16_t dim_a, uint16_t dim_r,
                                          uint16_t src_r_stride) {
  int64_t size_a_in_byte = dim_a * sizeof(T0);
  uint16_t loop_a = CEIL_DIV(size_a_in_byte, VL_IN_BYTE);
  uint16_t loop_r = dim_r > 1 ? (dim_r - 1) : 0;

  __VEC_SCOPE__ {
#if defined(__DAV_C310__)
    if (isAddress32ByteAligned(data_UB + src_r_stride) &&
        isAddress32ByteAligned(data_UB)) {
      ReduceImpl<OP, TIE_BREAK, T0, T1>::reduce_leading_dim(
          data_UB, reduce_val_UB, reduce_idx_UB, loop_a, dim_a,
          loop_r, src_r_stride);
    } else {
      ReduceImpl<OP, TIE_BREAK, T0, T1, __cce_simd::UnAlignedHint>::reduce_leading_dim(
          data_UB, reduce_val_UB, reduce_idx_UB, loop_a, dim_a, loop_r,
          src_r_stride);
    }
#else
    ReduceImpl<OP, TIE_BREAK, T0, T1>::reduce_leading_dim(
        data_UB, reduce_val_UB, reduce_idx_UB, loop_a, dim_a,
        loop_r, src_r_stride);
#endif
  }
}

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
inline __aiv__ __attribute__((always_inline)) void
vf_wrapper_reduce_ra0a1_last_axis_contiguous(
    __ubuf__ T0 *data_UB, __ubuf__ T0 *reduce_val_UB,
    __ubuf__ int32_t *reduce_idx_UB, uint16_t dim_a0, uint16_t dim_a1,
    uint16_t dim_r, uint16_t src_a0_stride, uint16_t src_r_stride,
    uint16_t dst_val_a0_stride, uint16_t dst_idx_a0_stride) {
  int64_t size_a_in_byte = dim_a1 * sizeof(T0);
  uint16_t loop_a1 = CEIL_DIV(size_a_in_byte, VL_IN_BYTE);
  uint16_t loop_r = dim_r > 1 ? (dim_r - 1) : 0;

  __VEC_SCOPE__ {

    for (uint16_t a0 = 0; a0 < dim_a0; ++a0) {
      // usually dst_a0 and src_a0 offset are the same
      __ubuf__ T0 *cur_data_UB = a0 * src_a0_stride + data_UB;
      __ubuf__ T0 *cur_reduce_val_UB = a0 * dst_val_a0_stride + reduce_val_UB;
      __ubuf__ int32_t *cur_reduce_idx_UB =
          a0 * dst_idx_a0_stride + reduce_idx_UB;
#if defined(__DAV_C310__)
      if (isAddress32ByteAligned(cur_data_UB + src_r_stride) &&
          isAddress32ByteAligned(cur_data_UB)) {
        ReduceImpl<OP, TIE_BREAK, T0, T1>::reduce_leading_dim(
            cur_data_UB, cur_reduce_val_UB, cur_reduce_idx_UB, loop_a1, dim_a1,
            loop_r, src_r_stride);
      } else {
        ReduceImpl<OP, TIE_BREAK, T0, T1, __cce_simd::UnAlignedHint>::reduce_leading_dim(
            cur_data_UB, cur_reduce_val_UB, cur_reduce_idx_UB, loop_a1, dim_a1,
            loop_r, src_r_stride);
      }
#else
      ReduceImpl<OP, TIE_BREAK, T0, T1>::reduce_leading_dim(
          cur_data_UB, cur_reduce_val_UB, cur_reduce_idx_UB, loop_a1,
          dim_a1, loop_r, src_r_stride);
#endif
    }
  }
}

/// reduce src (r, ) to dst (1, ) and return the reduction value and index
/// separately.
///
/// constraint:
/// 1. dim of src/dst must be 1.
/// 2. the start pointer address, namely aligned + offset, should be aligned
/// to ub_block_unit.
///
template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
__aiv__ __attribute__((always_inline)) void
reduce_r_with_index(memref_t<__ubuf__ T0, 1> *src0,
                    memref_t<__ubuf__ T0, 1> *dst_value,
                    memref_t<__ubuf__ int32_t, 1> *dst_index, T0 initvalue) {

  static_assert((OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX ||
                 OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) &&
                "reduce_r_with_index do not support this reduce op type");
  static_assert((TIE_BREAK == TieBreak::LEFT ||
                 TIE_BREAK == TieBreak::RIGHT) &&
                "reduce_r_with_index do not support this tie break type");

  const int64_t size0 = src0->sizes[0];
  const int64_t src_stride = src0->strides[0];
  __ubuf__ T0 *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T0 *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;
  bool is_last_dim_contiguous = (size0 == 1) || (src_stride == 1);
  // last dim contiguous
  if (is_last_dim_contiguous) {
    vf_wrapper_reduce_ar_last_axis_contiguous<OP, TIE_BREAK, T0, T1>(
        src_ptr, dst_value_ptr, dst_index_ptr, 1, size0, 1, 1, 1, initvalue);
  } else {
    // TODO: vgather last axis
    return;
  }
}

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
__aiv__ __attribute__((always_inline)) void
reduce_ar_with_index(memref_t<__ubuf__ T0, 2> *src0,
                     memref_t<__ubuf__ T0, 2> *dst_value,
                     memref_t<__ubuf__ int32_t, 2> *dst_index, T0 initvalue) {

  static_assert((OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX ||
                 OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) &&
                "reduce_r_with_index do not support this reduce op type");
  static_assert((TIE_BREAK == TieBreak::LEFT ||
                 TIE_BREAK == TieBreak::RIGHT) &&
                "reduce_r_with_index do not support this tie break type");

  const int64_t size0 = src0->sizes[0];
  const int64_t size1 = src0->sizes[1];
  const int64_t src_stride0 = src0->strides[0];
  const int64_t src_stride1 = src0->strides[1];
  // e.g. in mlir memref<3x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  // the stride for dst is 8 in unit of element
  const int64_t dst_value_stride = dst_value->strides[0];
  const int64_t dst_idx_stride = dst_index->strides[0];
  const int64_t dst_val_size1 = dst_value->sizes[1];
  const int64_t dst_idx_size1 = dst_index->sizes[1];
  const int64_t dst_val_stride1 = dst_value->strides[1];
  const int64_t dst_idx_stride1 = dst_index->strides[1];
  __ubuf__ T0 *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T0 *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;
  bool is_last_dim_contiguous =
      ((size1 == 1) || (src_stride1 == 1)) &&
      ((dst_val_size1 == 1) || (dst_val_stride1 == 1)) &&
      ((dst_idx_size1 == 1) || (dst_idx_stride1 == 1));
  // last dim contiguous
  if (is_last_dim_contiguous) {
    vf_wrapper_reduce_ar_last_axis_contiguous<OP, TIE_BREAK, T0, T1>(
        src_ptr, dst_value_ptr, dst_index_ptr, size0, size1, src_stride0,
        dst_value_stride, dst_idx_stride, initvalue);
  } else {
    // TODO: vgather last axis
    return;
  }
}

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
__aiv__ __attribute__((always_inline)) void
reduce_ra_with_index(memref_t<__ubuf__ T0, 2> *src0,
                     memref_t<__ubuf__ T0, 2> *dst_value,
                     memref_t<__ubuf__ int32_t, 2> *dst_index, T0 initvalue) {
  static_assert((OP == ReduceOpTy::REDUCE_MIN_WITH_INDEX ||
                 OP == ReduceOpTy::REDUCE_MAX_WITH_INDEX) &&
                "reduce_r_with_index do not support this reduce op type");
  static_assert((TIE_BREAK == TieBreak::LEFT ||
                 TIE_BREAK == TieBreak::RIGHT) &&
                "reduce_r_with_index do not support this tie break type");

  const int64_t size0 = src0->sizes[0]; // dimR
  const int64_t size1 = src0->sizes[1]; // dimA
  const int64_t src_stride0 = src0->strides[0];
  const int64_t src_stride1 = src0->strides[1];
  const int64_t dst_val_size1 = dst_value->sizes[1];
  const int64_t dst_idx_size1 = dst_value->sizes[1];
  const int64_t dst_val_stride1 = dst_value->strides[1];
  const int64_t dst_idx_stride1 = dst_index->strides[1];
  __ubuf__ T0 *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T0 *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;

  bool is_last_dim_contiguous =
      ((size1 == 1) || (src_stride1 == 1)) &&
      ((dst_val_size1 == 1) || (dst_val_stride1 == 1)) &&
      ((dst_idx_size1 == 1) || (dst_idx_stride1 == 1));

  if (is_last_dim_contiguous) {
    vf_wrapper_reduce_ra_last_axis_contiguous<OP, TIE_BREAK, T0, T1>(
        src_ptr, dst_value_ptr, dst_index_ptr, size1, size0, src_stride0);
    return;
  } else {
    // TODO: data non-contiguous last dim & non-contiguous store
  }
}

template <ReduceOpTy OP, TieBreak TIE_BREAK, typename T0, typename T1>
__aiv__ __attribute__((always_inline)) void reduce_ra0a1_with_index(
    memref_t<__ubuf__ T0, 3> *src0, memref_t<__ubuf__ T0, 3> *dst_value,
    memref_t<__ubuf__ int32_t, 3> *dst_index, T0 initvalue) {
  const int64_t src_size0 = src0->sizes[0];              // src dimR
  const int64_t src_size1 = src0->sizes[1];              // src dimA0
  const int64_t src_size2 = src0->sizes[2];              // src dimA1
  const int64_t src_stride0 = src0->strides[0];          // src strideR
  const int64_t src_stride1 = src0->strides[1];          // src strideA0
  const int64_t src_stride2 = src0->strides[2];          // src strideA1
  const int64_t dst_val_size1 = dst_value->sizes[1];     // dst dimA0
  const int64_t dst_val_size2 = dst_value->sizes[2];     // dst dimA1
  const int64_t dst_val_stride1 = dst_value->strides[1]; // dst strideA0
  const int64_t dst_val_stride2 = dst_value->strides[2]; // dst strideA1
  const int64_t dst_idx_size1 = dst_index->sizes[1];     // dst dimA0
  const int64_t dst_idx_size2 = dst_index->sizes[2];     // dst dimA1
  const int64_t dst_idx_stride1 = dst_index->strides[1]; // dst strideA0
  const int64_t dst_idx_stride2 = dst_index->strides[2]; // dst strideA1
  __ubuf__ T0 *src_ptr = src0->aligned + src0->offset;
  __ubuf__ T0 *dst_value_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *dst_index_ptr = dst_index->aligned + dst_index->offset;

  bool is_last_dim_contiguous =
      ((src_size2 == 1) || (src_stride2 == 1)) &&
      ((dst_val_size2 == 1) || (dst_val_stride2 == 1)) &&
      ((dst_idx_size2 == 1) || (dst_idx_stride2 == 1));

  // TODO: last axis non-contiguous
  if (!is_last_dim_contiguous) {
    return;
  }

  // case 1 where last two dims are contiguous, equivalent to
  // reduce_ra_with_index
  if ((src_size1 == 1 || src_stride1 == src_size2) &&
      (dst_val_size1 == 1 || dst_val_size2 == dst_val_stride1) &&
      (dst_idx_size1 == 1 || dst_idx_size2 == dst_idx_stride1)) {
    int64_t dimA = src_size1 * src_size2;
    vf_wrapper_reduce_ra_last_axis_contiguous<OP, TIE_BREAK, T0, T1>(
        src_ptr, dst_value_ptr, dst_index_ptr, dimA, src_size0, src_stride0);
    return;
  } else {
    vf_wrapper_reduce_ra0a1_last_axis_contiguous<OP, TIE_BREAK, T0, T1>(
        src_ptr, dst_value_ptr, dst_index_ptr, src_size1, src_size2, src_size0,
        src_stride1, src_stride0, dst_val_stride1, dst_idx_stride1);
    return;
  }
}

extern "C" {
//===-------------------------------------------------------------------===//
// reduce r with index, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, half,
                            int16_t);
REGISTE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, float,
                            int32_t);
REGISTE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, int16_t,
                            int16_t);
REGISTE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, int32_t,
                            int32_t);
REGISTE_REDUCE_R_WITH_INDEX(reduce_max_with_index,
                            ReduceOpTy::REDUCE_MAX_WITH_INDEX, 1, int64_t,
                            int32_t);

REGISTE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, half,
                            int16_t);
REGISTE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, float,
                            int32_t);
REGISTE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, int16_t,
                            int16_t);
REGISTE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, int32_t,
                            int32_t);
REGISTE_REDUCE_R_WITH_INDEX(reduce_min_with_index,
                            ReduceOpTy::REDUCE_MIN_WITH_INDEX, 1, int64_t,
                            int32_t);

//===-------------------------------------------------------------------===//
// reduce ar with index, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, half,
                             int16_t);
REGISTE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, float,
                             int32_t);
REGISTE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int16_t,
                             int16_t);
REGISTE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int32_t,
                             int32_t);
REGISTE_REDUCE_AR_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int64_t,
                             int32_t);

REGISTE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, half,
                             int16_t);
REGISTE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, float,
                             int32_t);
REGISTE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int16_t,
                             int16_t);
REGISTE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int32_t,
                             int32_t);
REGISTE_REDUCE_AR_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int64_t,
                             int32_t);

//===-------------------------------------------------------------------===//
// reduce ra with index, 2 dim
//===-------------------------------------------------------------------===//
REGISTE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, half,
                             int16_t);
REGISTE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, float,
                             int32_t);
REGISTE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int16_t,
                             int16_t);
REGISTE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int32_t,
                             int32_t);
REGISTE_REDUCE_RA_WITH_INDEX(reduce_max_with_index,
                             ReduceOpTy::REDUCE_MAX_WITH_INDEX, 2, int64_t,
                             int32_t);

REGISTE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, half,
                             int16_t);
REGISTE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, float,
                             int32_t);
REGISTE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int16_t,
                             int16_t);
REGISTE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int32_t,
                             int32_t);
REGISTE_REDUCE_RA_WITH_INDEX(reduce_min_with_index,
                             ReduceOpTy::REDUCE_MIN_WITH_INDEX, 2, int64_t,
                             int32_t);

//===-------------------------------------------------------------------===//
// reduce ra0a1 with index, 3 dim
//===-------------------------------------------------------------------===//
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, half,
                                int16_t);
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, float,
                                int32_t);
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, int16_t,
                                int16_t);
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, int32_t,
                                int32_t);
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_max_with_index,
                                ReduceOpTy::REDUCE_MAX_WITH_INDEX, 3, int64_t,
                                int32_t);
                                
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, half,
                                int16_t);
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, float,
                                int32_t);
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, int16_t,
                                int16_t);
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, int32_t,
                                int32_t);
REGISTE_REDUCE_RA0A1_WITH_INDEX(reduce_min_with_index,
                                ReduceOpTy::REDUCE_MIN_WITH_INDEX, 3, int64_t,
                                int32_t);
}
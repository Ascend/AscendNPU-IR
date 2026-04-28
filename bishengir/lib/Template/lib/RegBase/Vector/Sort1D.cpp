/**
  copy_ubuf_to_ubuf_1d_core(src, dst_value);
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
#include "RegBase/ReductionUtils.h"
#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/Broadcast/BrcUtils.h"
#include "Vector/Sort/SortUtils.h"
#include "Vector/VecUtils.h"

#if defined(__DAV_C310__)
template <typename T>
__aiv__ __attribute__((always_inline)) void
lower_sort_for_last_axis_is_one(memref_t<__ubuf__ T, 1> *src,
                                memref_t<__ubuf__ T, 1> *dst_value) {
  copy_ubuf_to_ubuf_1d_core(src, dst_value);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
prepare_src_value(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                  int64_t real_num, int64_t sort_num, bool descending) {
  auto src_ptr = src->aligned + src->offset;
  int64_t fill_num = sort_num - real_num;
  // In descending order, the minimum value needs to be filled to ensure that
  // it does not affect the sorting result.
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < fill_num; ++i) {
  *(src_ptr + real_num + i) = static_cast<T>(FLOAT_NEG_INF);
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

__aiv__ __attribute__((always_inline)) void
prepare_src_index(memref_t<__ubuf__ int32_t, 1> *src_index, int64_t real_num,
                  int64_t sort_num) {
  src_index->sizes[0] = sort_num;
  auto dst_ptr = src_index->aligned + src_index->offset;

  // Use arange to fill the index
  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  for (int64_t i = 0; i < sort_num; ++i) {
    dst_ptr[i] = static_cast<int32_t>(i);
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  src_index->sizes[0] = real_num;
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
block_sort(memref_t<__ubuf__ T, 1> *src_value,
           memref_t<__ubuf__ int32_t, 1> *src_index,
           memref_t<__ubuf__ T, 1> *dst, int64_t real_num, int64_t sort_num) {
  auto repeat = sort_num / BIT_SORT_NUM_PER_REPEAT;
  auto sort_num_per_intrinsic = INTR_MAX_REPEAT_CNTS * BIT_SORT_NUM_PER_REPEAT;
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);
  auto dst_ptr = dst->aligned + dst->offset;
  auto src_value_ptr = src_value->aligned + src_value->offset;
  auto src_index_ptr = src_index->aligned + src_index->offset;
  if (repeat >= INTR_MAX_REPEAT_CNTS) [[unlikely]] {
    for (int64_t i = 0; i < repeat / INTR_MAX_REPEAT_CNTS; ++i) {
      INTRINSIC(
          vbitsort, dst_ptr + i * sort_num_per_intrinsic * num_per_proposal,
          src_value_ptr + i * sort_num_per_intrinsic,
          (__ubuf__ uint32_t *)(src_index_ptr + i * sort_num_per_intrinsic),
          INTR_MAX_REPEAT_CNTS); // repeat
    }
  }

  if (repeat % INTR_MAX_REPEAT_CNTS != 0) [[likely]] {
    auto loop_num = repeat / INTR_MAX_REPEAT_CNTS;
    INTRINSIC(vbitsort,
              dst_ptr + loop_num * sort_num_per_intrinsic * num_per_proposal,
              src_value_ptr + loop_num * sort_num_per_intrinsic,
              (__ubuf__ uint32_t *)(src_index_ptr +
                                    loop_num * sort_num_per_intrinsic),
              repeat % INTR_MAX_REPEAT_CNTS); // repeat
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
lower_vms_quotient(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                   int64_t &merge_times, int64_t factor, int64_t sort_num) {
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);
  int64_t list_interval_offset = factor * num_per_proposal;
  int64_t quotient_repeat = sort_num / factor / MERGE_SORT_MAX_PROPOSALS_LIST;
  uint64_t list_num = factor & (int64_t)MAX_UINT16;
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;

  // Xn[15:0] is list 0 address
  // Xn[31:16] is list 1 address
  // Xn[47:32] is list 2 address
  // Xn[63:48] is list 3 address
  __ubuf__ T *xn[4] = {
      merge_times % 2 ? src_ptr : dst_ptr,
      merge_times % 2 ? src_ptr + list_interval_offset
                      : dst_ptr + list_interval_offset,
      merge_times % 2 ? src_ptr + list_interval_offset * 2
                      : dst_ptr + list_interval_offset * 2,
      merge_times % 2 ? src_ptr + list_interval_offset * 3
                      : dst_ptr + list_interval_offset * 3,
  };

  // xm[15:0]: number of structures in input list 0
  // xm[31:16]: number of structures in input list 1
  // xm[47:32]: number of structures in input list 2
  // xm[63:48]: number of structures in input list 3
  uint64_t xm =
      ((list_num | (list_num << 16)) | (list_num << 32)) | (list_num << 48);

  // Enabled repeat mode and disable exhaustion mode when running vmrgsort4
  // (config[12] = 0 and config[7:0] > 1 and config[11:8] means whether the
  // four lists valid?)
  uint64_t config =
      (WAY4_CONFIG_MODE | (quotient_repeat & INTR_MAX_REPEAT_CNTS));
  INTRINSIC(vmrgsort4, merge_times % 2 ? dst_ptr : src_ptr,
            xn,      // the address of each list
            xm,      // the number of each list
            config); // config
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
lower_vms_remainder(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                    int64_t &merge_times, int64_t factor, int64_t sort_num,
                    merge_remainder_info *merge_remainder) {
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);
  int64_t quotient_repeat = sort_num / factor / MERGE_SORT_MAX_PROPOSALS_LIST;
  int64_t cur_remainder = sort_num / factor % MERGE_SORT_MAX_PROPOSALS_LIST;
  uint64_t list_num = factor & (int64_t)MAX_UINT16;
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  // Calculate the offset of cur_remainder
  auto base_offset = quotient_repeat * factor * MERGE_SORT_MAX_PROPOSALS_LIST *
                     num_per_proposal;

  if (cur_remainder == 0 && merge_remainder->merge_remainder_flag) {
    memref_t<__ubuf__ T, 1> src_tmp{
        src->aligned,
        src->allocated,
        static_cast<int64_t>(src->offset + base_offset),
        {static_cast<int64_t>(merge_remainder->merge_remainder_num *
                              num_per_proposal)},
        {1}};
    memref_t<__ubuf__ T, 1> dst_tmp{
        dst->aligned,
        dst->allocated,
        static_cast<int64_t>(dst->offset + base_offset),
        {static_cast<int64_t>(merge_remainder->merge_remainder_num *
                              num_per_proposal)},
        {1}};
    // At this point, there is only one list, no need to merge
    copy_ubuf_to_ubuf_1d_core(merge_times % 2 ? &src_tmp : &dst_tmp,
                              merge_times % 2 ? &dst_tmp : &src_tmp);
  } else if (cur_remainder == 1) {
    if (merge_remainder->merge_remainder_num > 0) {
      // Perform a 2-way sort, list1 is the tail block left over from the
      // previous round(merge_remainder_num)
      __ubuf__ T *xn[2] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
      };
      // 0011(proposal list) + 000000001(repeat)
      uint64_t config = WAY2_CONFIG_MODE | 1;
      uint64_t xm = list_num |
                    ((merge_remainder->merge_remainder_num & MAX_UINT16) << 16);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    } else {
      memref_t<__ubuf__ T, 1> src_tmp{
          src->aligned,
          src->allocated,
          static_cast<int64_t>(src->offset + base_offset),
          {static_cast<int64_t>(list_num * num_per_proposal)},
          {1}};
      memref_t<__ubuf__ T, 1> dst_tmp{
          dst->aligned,
          dst->allocated,
          static_cast<int64_t>(dst->offset + base_offset),
          {static_cast<int64_t>(list_num * num_per_proposal)},
          {1}};
      // At this point, there is only one list, no need to merge
      copy_ubuf_to_ubuf_1d_core(merge_times % 2 ? &src_tmp : &dst_tmp,
                                merge_times % 2 ? &dst_tmp : &src_tmp);
    }

    // After sorting, a tail block is formed for the next round of merge
    merge_remainder->merge_remainder_flag = true;
    merge_remainder->merge_remainder_num += factor;
  } else if (cur_remainder == 2) {
    if (merge_remainder->merge_remainder_num > 0) {
      // Perform a 3-way sort, list2 is the tail block left over from the
      // previous round(merge_remainder_num)
      __ubuf__ T *xn[3] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
          merge_times % 2
              ? src_ptr + base_offset + factor * num_per_proposal * 2
              : dst_ptr + base_offset + factor * num_per_proposal * 2,
      };
      // 0111(proposal list) + 000000001(repeat)
      uint64_t config = WAY3_CONFIG_MODE | 1;
      uint64_t xm = (list_num | (list_num << 16)) |
                    ((merge_remainder->merge_remainder_num & MAX_UINT16) << 32);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    } else {
      // Perform a 2-way sort
      __ubuf__ T *xn[2] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
      };
      // 0011(proposal list) + 000000001(repeat)
      uint64_t config = WAY2_CONFIG_MODE | 1;
      uint64_t xm = list_num | (list_num << 16);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    }

    // After sorting, a tail block is formed for the next round of merge
    merge_remainder->merge_remainder_flag = true;
    merge_remainder->merge_remainder_num += factor * 2;
  } else if (cur_remainder == 3) {
    if (merge_remainder->merge_remainder_num > 0) {
      // Perform a 4-way sort, list3 is the tail block left over from the
      // previous round(merge_remainder_num)
      __ubuf__ T *xn[4] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
          merge_times % 2
              ? src_ptr + base_offset + factor * num_per_proposal * 2
              : dst_ptr + base_offset + factor * num_per_proposal * 2,
          merge_times % 2
              ? src_ptr + base_offset + factor * num_per_proposal * 3
              : dst_ptr + base_offset + factor * num_per_proposal * 3,
      };
      // 1111(proposal list) + 000000001(repeat)
      uint64_t config = WAY4_CONFIG_MODE | 1;
      uint64_t xm = ((list_num | (list_num << 16)) | (list_num << 32)) |
                    ((merge_remainder->merge_remainder_num & MAX_UINT16) << 48);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    } else {
      // Perform a 3-way sort
      __ubuf__ T *xn[3] = {
          merge_times % 2 ? src_ptr + base_offset : dst_ptr + base_offset,
          merge_times % 2 ? src_ptr + base_offset + factor * num_per_proposal
                          : dst_ptr + base_offset + factor * num_per_proposal,
          merge_times % 2
              ? src_ptr + base_offset + factor * num_per_proposal * 2
              : dst_ptr + base_offset + factor * num_per_proposal * 2,
      };
      // 0111(proposal list) + 000000001(repeat)
      uint64_t config = WAY3_CONFIG_MODE | 1;
      uint64_t xm = (list_num | (list_num << 16)) | (list_num << 32);
      INTRINSIC(vmrgsort4,
                merge_times % 2 ? dst_ptr + base_offset : src_ptr + base_offset,
                xn,      // the address of each list
                xm,      // the number of each list
                config); // config
    }

    // After sorting, a tail block is formed for the next round of merge
    merge_remainder->merge_remainder_flag = true;
    merge_remainder->merge_remainder_num += factor * 3;
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
merge_sort(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
           int64_t real_num, int64_t sort_num, int64_t &merge_times) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);

  merge_remainder_info merge_remainder;
  // Whether there is a tail block left in the previous round of merge.
  merge_remainder.merge_remainder_flag = false;
  // The number of proposals to be sorted in the tail block left over from the
  // previous merge round
  merge_remainder.merge_remainder_num = 0;

  // current merge round number
  while (1) {
    int64_t factor = 32 << (merge_times * 2);
    int64_t quotient_repeat = sort_num / factor / MERGE_SORT_MAX_PROPOSALS_LIST;
    merge_times++;

    // Processing 4-way merge of main block
    // The main block requires 4 equal lengths, and the length of each path is
    // factor(32 << (i * 2))
    if (quotient_repeat > 0) {
      lower_vms_quotient(src, dst, merge_times, factor, sort_num);
    }

    // Processing 2/3/4-way merge of tail block
    lower_vms_remainder(src, dst, merge_times, factor, sort_num,
                        &merge_remainder);

    // merge completion conditions
    if (quotient_repeat == 0 ||
        (quotient_repeat == 1 && merge_remainder.merge_remainder_num == 0)) {
      break;
    }
  }
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
move_out_result(memref_t<__ubuf__ T, 1> *src,
                memref_t<__ubuf__ T, 1> *dst_value,
                memref_t<__ubuf__ int32_t, 1> *dst_index, bool descending,
                bool need_index) {
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];

  auto src_ptr = src->aligned + src->offset;
  auto dst_value_ptr = dst_value->aligned + dst_value->offset;
  auto dst_index_ptr = dst_index->aligned + dst_index->offset;
  memref_t<__ubuf__ int32_t, 1> src_int32;
  view_as<T, int32_t, 1>(src, &src_int32);
  auto src_int32_ptr = src_int32.aligned + src_int32.offset;

  // separate dst_value/dst_index from the sorted proposal.
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, real_num);

  __VEC_SCOPE__ {
    vector_bool full_mask;
    CREATE_MASK_BY_PAT(full_mask, T, PAT_ALL);

    VectorReg<T> result;
    VectorReg<int32_t> index_result;
    constexpr int dsize = sizeof(T);
    vector_u32 indices;
    if (descending) {
      if (need_index) {
        for (uint16_t i = 0; i < (uint16_t)dst_index->sizes[0]; ++i) {
          vbr(indices, i * 2 + 1);
          vgather2_bc(index_result, src_int32_ptr, indices, full_mask);
          vsts(index_result, dst_index_ptr, i, ONEPT_B32, full_mask);
        }
      }
      for (uint16_t i = 0; i < (uint16_t)dst_value->sizes[0]; ++i) {
        if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, half>) {
          vbr(indices, i * 4);
          vgather2_bc(result, src_ptr, indices, full_mask);
          vsts(result, dst_value_ptr, i, ONEPT_B16, full_mask);
        } else if constexpr (dsize == BYTES_B32) {
          vbr(indices, i * 2);
          vgather2_bc(result, src_ptr, indices, full_mask);
          vsts(result, dst_value_ptr, i, ONEPT_B32, full_mask);
        }
      }
    } else {
      uint16_t end = (uint16_t)dst_value->sizes[0];
      if (need_index) {
        for (uint16_t i = 0; i < end; ++i) {
          vbr(indices, (end - i - 1) * 2 + 1);
          vgather2_bc(index_result, src_int32_ptr, indices, full_mask);
          vsts(index_result, dst_index_ptr, i, ONEPT_B32, full_mask);
        }
      }
      for (uint16_t i = 0; i < end; ++i) {
        if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, half>) {
          vbr(indices, (end - i - 1) * 4);
          vgather2_bc(result, src_ptr, indices, full_mask);
          vsts(result, dst_value_ptr, i, ONEPT_B16, full_mask);
        } else if constexpr (dsize == BYTES_B32) {
          vbr(indices, (end - i - 1) * 2);
          vgather2_bc(result, src_ptr, indices, full_mask);
          vsts(result, dst_value_ptr, i, ONEPT_B32, full_mask);
        }
      }
    }
  } // end of __VEC_SCOPE__
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
move_out_result_with_index(memref_t<__ubuf__ T, 1> *src,
                           memref_t<__ubuf__ T, 1> *dst_value,
                           memref_t<__ubuf__ int32_t, 1> *dst_index) {
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];

  __ubuf__ T *src_ptr = src->aligned + src->offset;
  __ubuf__ T *dst_ptr = dst_value->aligned + dst_value->offset;
  __ubuf__ int32_t *index_ptr = dst_index->aligned + dst_index->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, EVENT_ID0);
  for (int i = 0; i < real_num; i++) {
    dst_ptr[i] = src_ptr[index_ptr[i]];
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, EVENT_ID0);
}

__aiv__ __attribute__((always_inline)) void
cast_i32_to_f32_for_sort(memref_t<__ubuf__ int32_t, 1> *src,
                         memref_t<__ubuf__ float, 1> *dst, int64_t sort_num) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  constexpr int dsize = sizeof(int32_t);
  uint16_t loop_times = CEIL_DIV(sort_num * dsize, VL_IN_BYTE);
  uint16_t ele_per_VL = VL_IN_BYTE / dsize;
  __VEC_SCOPE__ {
    VectorReg<int32_t> src_reg;
    VectorReg<float> dst_reg;
    unsigned int plt_size = sort_num;
    for (uint16_t i = 0; i < (uint16_t)loop_times; ++i) {
      vector_bool full_mask;
      CREATE_MASK_BY_SIZE(full_mask, int32_t, plt_size);
      vlds(src_reg, src_ptr, i * ele_per_VL, NORM);
      vcvt(dst_reg, src_reg, full_mask, ROUND_R, MODE_ZEROING);
      vsts(dst_reg, dst_ptr, i * ele_per_VL, NORM_B32, full_mask);
    }
  } // end of __VEC_SCOPE__
}

__aiv__ __attribute__((always_inline)) void
cast_low_to_f32_for_sort(memref_t<__ubuf__ int32_t, 1> *src,
                         memref_t<__ubuf__ float, 1> *dst, int64_t sort_num) {
  auto src_ptr = src->aligned + src->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  constexpr int dsize = sizeof(int32_t);
  uint16_t loop_times = CEIL_DIV(sort_num * dsize, VL_IN_BYTE);
  uint16_t ele_per_VL = VL_IN_BYTE / dsize;
  __VEC_SCOPE__ {
    VectorReg<int32_t> src_reg;
    VectorReg<float> dst_reg;
    unsigned int plt_size = sort_num;
    for (uint16_t i = 0; i < (uint16_t)loop_times; ++i) {
      vector_bool full_mask;
      CREATE_MASK_BY_SIZE(full_mask, int32_t, plt_size);
      vector_s32 tmp, lowPart;
      vlds(src_reg, src_ptr, i * ele_per_VL, NORM);
      vshls(tmp, src_reg, 16, full_mask);
      vshrs(lowPart, tmp, 16, full_mask);
      vcvt(dst_reg, lowPart, full_mask, ROUND_R, MODE_ZEROING);
      vsts(dst_reg, dst_ptr, i * ele_per_VL, NORM_B32, full_mask);
    }
  } // end of __VEC_SCOPE__
}

template <typename DST>
__aiv__ __attribute__((always_inline)) void
extract_i64_to_i32(memref_t<__ubuf__ int64_t, 1> *src,
                   memref_t<__ubuf__ DST, 1> *dst, bool isHigh) {
  static_assert(sizeof(DST) == 4,
                "Destination type must be 32-bit for i64 extraction");
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];

  __ubuf__ int64_t *src_ptr = src->aligned + src->offset;
  __ubuf__ DST *dst_ptr = dst->aligned + dst->offset;

  INTRINSIC(set_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_V, PIPE_S, LIB_EVENT_ID0);
  if (isHigh) {
    for (int64_t i = 0; i < real_num; ++i) {
      dst_ptr[i] = static_cast<DST>(src_ptr[i] >> 32);
    }
  } else {
    for (int64_t i = 0; i < real_num; ++i) {
      dst_ptr[i] = static_cast<DST>(src_ptr[i]);
    }
  }
  INTRINSIC(set_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
  INTRINSIC(wait_flag, PIPE_S, PIPE_V, LIB_EVENT_ID0);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void lower_sort_operation(
    memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst_value,
    memref_t<__ubuf__ int32_t, 1> *dst_index, memref_t<__ubuf__ T, 1> *tmp_buf,
    bool descending, bool need_index) {
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];
  // Calculate the number of data involved in the sort, (vbitsort requires 32
  // elements to be aligned)
  int64_t sort_num = CEIL_FACTOR(src->sizes[0], BIT_SORT_NUM_PER_REPEAT);

  if (real_num == 1) {
    // When the sort axis is 1, no need to sort, move out directly
    lower_sort_for_last_axis_is_one(src, dst_value);
    return;
  }

  // Step1: Prepare src value for vbitsort:
  // 1. Fill the data to be sorted to ensure that the number is a multiple of 32
  // 2. If it is ascending order, need to reverse the data
  memref_t<__ubuf__ T, 1> src_inverse{
      tmp_buf->aligned, tmp_buf->allocated, tmp_buf->offset, {sort_num}, {1}};
  prepare_src_value(src, &src_inverse, real_num, sort_num, descending);

  // Step2: Vector Bitonic Sorter
  // VBS32 will combine index with corresponding score after sorting and
  // output a proposal structure 8B
  auto num_per_proposal = PROPOSALS_BYTES / sizeof(T);
  memref_t<__ubuf__ T, 1> tmp_buf_for_block_sort{
      tmp_buf->aligned,
      tmp_buf->allocated,
      tmp_buf->offset,
      {static_cast<int64_t>(sort_num * num_per_proposal)},
      {1}};
  block_sort(src, dst_index, &tmp_buf_for_block_sort, real_num, sort_num);

  // Step3: Vector Merge Sorter
  memref_t<__ubuf__ T, 1> tmp_buf_for_merge_sort{
      tmp_buf->aligned,
      tmp_buf->allocated,
      tmp_buf->offset + static_cast<int64_t>(sort_num * num_per_proposal),
      {static_cast<int64_t>(sort_num * num_per_proposal)},
      {1}};
  int64_t merge_times = 0;
  merge_sort(&tmp_buf_for_block_sort, &tmp_buf_for_merge_sort, real_num,
             sort_num, merge_times);

  // Step4: Process proposal data and move it to dst_value/dst_index
  move_out_result(merge_times % 2 ? &tmp_buf_for_merge_sort
                                  : &tmp_buf_for_block_sort,
                  dst_value, dst_index, descending, need_index);
}

__aiv__ __attribute__((always_inline)) void
lower_sort_operation_for_i32(memref_t<__ubuf__ int32_t, 1> *src,
                             memref_t<__ubuf__ int32_t, 1> *dst_value,
                             memref_t<__ubuf__ int32_t, 1> *dst_index,
                             memref_t<__ubuf__ int32_t, 1> *tmp_buf,
                             bool descending, bool need_index) {
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];
  // Calculate the number of data involved in the sort, (vbitsort requires 32
  // elements to be aligned)
  int64_t sort_num = CEIL_FACTOR(src->sizes[0], BIT_SORT_NUM_PER_REPEAT);

  if (real_num == 1) {
    // When the sort axis is 1, no need to sort, move out directly
    lower_sort_for_last_axis_is_one(src, dst_value);
    return;
  }

  // Step1: Sort by the lower 16 bits
  memref_t<__ubuf__ int32_t, 1> tmp_dst_value{
      tmp_buf->aligned, tmp_buf->allocated, tmp_buf->offset, {real_num}, {1}};
  tmp_buf->offset += sort_num;
  memref_t<__ubuf__ float, 1> tmp_buf_float;
  view_as<int32_t, float, 1>(tmp_buf, &tmp_buf_float);
  memref_t<__ubuf__ float, 1> tmp_buf_for_low_sort{tmp_buf_float.aligned,
                                                   tmp_buf_float.allocated,
                                                   tmp_buf_float.offset,
                                                   {real_num},
                                                   {1}};
  tmp_buf_float.offset += sort_num;
  cast_low_to_f32_for_sort(src, &tmp_buf_for_low_sort, sort_num);

  // Get the index of the sorted results.
  lower_sort_operation<float>(&tmp_buf_for_low_sort, &tmp_buf_for_low_sort,
                              dst_index, &tmp_buf_float, true, need_index);
  move_out_result_with_index(src, &tmp_dst_value, dst_index);

  // Step2: Secondary sorting, ensuring overall order by first sorting lower
  // positions
  memref_t<__ubuf__ float, 1> tmp_buf_for_sort{tmp_buf_float.aligned,
                                               tmp_buf_float.allocated,
                                               tmp_buf_float.offset,
                                               {real_num},
                                               {1}};
  tmp_buf_float.offset += sort_num;
  cast_i32_to_f32_for_sort(&tmp_dst_value, &tmp_buf_for_sort, sort_num);

  // Get the index of the sorted results.
  lower_sort_operation<float>(&tmp_buf_for_sort, &tmp_buf_for_sort, dst_index,
                              &tmp_buf_float, descending, need_index);
  move_out_result_with_index(src, dst_value, dst_index);
}

/// 64-bit integer sorting using two-pass radix sort (low32→high32).
///
/// Implements 64-bit integer sorting by splitting into two 32-bit passes.
/// This workaround is necessary because the vbitsort instruction only supports
/// f16 and f32 types. Direct i64→f32→i64 conversion causes precision loss
/// (f32 has only 24-bit mantissa). The solution performs radix sort:
///   1. Extract lower 32 bits → sort as i32 → reorder original array
///   2. Extract higher 32 bits → sort as i32 → reorder again
///
/// Example of the two-pass radix sort (ascending order):
///
///   Original i64 array (hex representation):
///     [0x12345678'AABBCCDD, 0x87654321'11223344, 0x00000000'FFFFFFFF]
///
///   Step 1 - Sort by lower 32 bits:
///     Extract low32 bits: [0xAABBCCDD, 0x11223344, 0xFFFFFFFF]
///     Sort low32 bits (ascending): [0x11223344, 0xAABBCCDD, 0xFFFFFFFF]
///     Get indices from sort: [1, 0, 2]
///     Intermediate result: [0x87654321'11223344, 0x12345678'AABBCCDD, 0x00000000'FFFFFFFF]
///
///   Step 2 - Sort by higher 32 bits:
///     Extract high32 bits from intermediate: [0x87654321, 0x12345678, 0x00000000]
///     Sort high32 bits (ascending): [0x00000000, 0x12345678, 0x87654321]
///     Get indices from sort: [2, 1, 0]
///     Final sorted array: [0x00000000'FFFFFFFF, 0x12345678'AABBCCDD, 0x87654321'11223344]
///
/// \param descending: true for descending order, false for ascending
/// \param need_index: true if indices are needed, false otherwise
__aiv__ __attribute__((always_inline)) void
lower_sort_operation_for_i64(memref_t<__ubuf__ int64_t, 1> *src,
                             memref_t<__ubuf__ int64_t, 1> *dst_value,
                             memref_t<__ubuf__ int32_t, 1> *dst_index,
                             memref_t<__ubuf__ int32_t, 1> *tmp_buf,
                             bool descending, bool need_index) {
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];
  // Calculate the number of data involved in the sort, (vbitsort requires 32
  // elements to be aligned)
  int64_t sort_num = CEIL_FACTOR(src->sizes[0], BIT_SORT_NUM_PER_REPEAT);

  if (real_num == 1) {
    // When the sort axis is 1, no need to sort, move out directly
    lower_sort_for_last_axis_is_one(src, dst_value);
    return;
  }

  memref_t<__ubuf__ int32_t, 1> tmp_dst_i32{
      tmp_buf->aligned, tmp_buf->allocated, tmp_buf->offset, {real_num}, {1}};
  tmp_buf->offset += sort_num;
  memref_t<__ubuf__ int64_t, 1> tmp_buf_int64;
  view_as<int32_t, int64_t, 1>(tmp_buf, &tmp_buf_int64);
  memref_t<__ubuf__ int64_t, 1> tmp_dst_i64{tmp_buf_int64.aligned,
                                            tmp_buf_int64.allocated,
                                            tmp_buf_int64.offset,
                                            {real_num},
                                            {1}};
  tmp_buf->offset += sort_num * 2;
  extract_i64_to_i32(src, &tmp_dst_i32, false);
  lower_sort_operation_for_i32(&tmp_dst_i32, &tmp_dst_i32, dst_index, tmp_buf,
                               true, need_index);
  move_out_result_with_index(src, &tmp_dst_i64, dst_index);

  memref_t<__ubuf__ float, 1> tmp_buf_float;
  view_as<int32_t, float, 1>(tmp_buf, &tmp_buf_float);
  memref_t<__ubuf__ float, 1> tmp_buf_for_sort{tmp_buf_float.aligned,
                                               tmp_buf_float.allocated,
                                               tmp_buf_float.offset,
                                               {real_num},
                                               {1}};
  tmp_buf_float.offset += sort_num;
  extract_i64_to_i32(&tmp_dst_i64, &tmp_buf_for_sort, true);
  lower_sort_operation<float>(&tmp_buf_for_sort, &tmp_buf_for_sort, dst_index,
                              &tmp_buf_float, descending, need_index);
  move_out_result_with_index(src, dst_value, dst_index);
}

/// Sort src=(a,) with stride[1,], a is the sorting axis returns the sorted
/// value of dst_value.
///
/// constraint:
/// 1. Make sure the sum of all bufs is less than UB_MAX.
/// 2. The data type to be sorted only supports half/float.
/// 3. src id Continuous 1D.
/// 4. The size of tmp_buf is as follows:
///    a == 1:
///        tmp_buf = 0
///    a != 1:
///        descending = false:
///            dtype = half:
///                tmp_buf > aligned(a, 32) * 11
///            dtype = float:
///                tmp_buf > aligned(a, 32) * 6
///        descending = true:
///            dtype = half:
///                tmp_buf > aligned(a, 32) * 10
///            dtype = float:
///                tmp_buf > aligned(a, 32) * 5
///
/// \param:
/// descending: descending = true to sort in descending order, descending =
/// false to sort in ascending order.
template <typename T>
__aiv__ __attribute__((always_inline)) void
sort_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
        memref_t<__ubuf__ T, 1> *tmp_buf, bool descending) {

  static_assert(
      (std::is_same<T, half>::value || std::is_same<T, float>::value) &&
      "Sort unsupport this type");

  // Calculate the number of data involved in the sort, (vbitsort requires 32
  // elements to be aligned)
  int64_t sort_num = CEIL_FACTOR(src->sizes[0], BIT_SORT_NUM_PER_REPEAT);

  // When sorting, need to sort with index, so need to allocate a space in
  // tmp_buf to store the index corresponding to src_value
  memref_t<__ubuf__ int32_t, 1> tmp_buf_int32;
  view_as<T, int32_t, 1>(tmp_buf, &tmp_buf_int32);
  memref_t<__ubuf__ int32_t, 1> dst_index{tmp_buf_int32.aligned,
                                          tmp_buf_int32.allocated,
                                          tmp_buf_int32.offset,
                                          {sort_num},
                                          {1}};
  tmp_buf->offset += sort_num * (sizeof(int32_t) / sizeof(T));
  lower_sort_operation(src, dst, &dst_index, tmp_buf, descending, false);
}

template <>
__aiv__ __attribute__((always_inline)) void
sort_1d<int32_t>(memref_t<__ubuf__ int32_t, 1> *src,
                 memref_t<__ubuf__ int32_t, 1> *dst,
                 memref_t<__ubuf__ int32_t, 1> *tmp_buf, bool descending) {
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];
  // Calculate the number of data involved in the sort, (vbitsort requires 32
  // elements to be aligned)
  int64_t sort_num = CEIL_FACTOR(src->sizes[0], BIT_SORT_NUM_PER_REPEAT);

  // When sorting, need to sort with index, so need to allocate a space in
  // tmp_buf to store the index corresponding to src_value
  memref_t<__ubuf__ int32_t, 1> dst_index{
      tmp_buf->aligned, tmp_buf->allocated, tmp_buf->offset, {sort_num}, {1}};
  tmp_buf->offset += sort_num;

  // Prepare src index for vbitsort
  prepare_src_index(&dst_index, real_num, sort_num);

  // When handling i32, convert to f32 for sorting, and then use index to sort
  // avoid precision loss from i32 to f32
  lower_sort_operation_for_i32(src, dst, &dst_index, tmp_buf, descending, true);
}

template <>
__aiv__ __attribute__((always_inline)) void
sort_1d<int64_t>(memref_t<__ubuf__ int64_t, 1> *src,
                 memref_t<__ubuf__ int64_t, 1> *dst,
                 memref_t<__ubuf__ int64_t, 1> *tmp_buf, bool descending) {
  // The actual number of data to be sorted
  int64_t real_num = src->sizes[0];
  // Calculate the number of data involved in the sort, (vbitsort requires 32
  // elements to be aligned)
  int64_t sort_num = CEIL_FACTOR(src->sizes[0], BIT_SORT_NUM_PER_REPEAT);

  // When sorting, need to sort with index, so need to allocate a space in
  // tmp_buf to store the index corresponding to src_value
  memref_t<__ubuf__ int32_t, 1> tmp_buf_int32;
  view_as<int64_t, int32_t, 1>(tmp_buf, &tmp_buf_int32);
  memref_t<__ubuf__ int32_t, 1> dst_index{tmp_buf_int32.aligned,
                                          tmp_buf_int32.allocated,
                                          tmp_buf_int32.offset,
                                          {sort_num},
                                          {1}};
  tmp_buf_int32.offset += sort_num;

  // Prepare src index for vbitsort
  prepare_src_index(&dst_index, real_num, sort_num);

  lower_sort_operation_for_i64(src, dst, &dst_index, &tmp_buf_int32, descending,
                               true);
}

extern "C" {
//===----------------------------------------------------------------------===//
// sort, 1 dim
//===----------------------------------------------------------------------===//
REGISTE_SORT(1, half);
REGISTE_SORT(1, float);
REGISTE_SORT(1, int32_t);
REGISTE_SORT(1, int64_t);
}
#endif
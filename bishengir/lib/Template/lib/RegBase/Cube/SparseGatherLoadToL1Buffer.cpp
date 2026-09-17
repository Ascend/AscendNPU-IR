/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

// sparse_gather_load_to_L1buffer
//
// Gathers the KV regions named by a packed region list from GM straight into an
// L1 zN tile. A region's tokens are contiguous in the physical token space, so
// a run of consecutive region ids is one contiguous GM range and goes out as a
// single MTE2 descriptor; coalescing those runs is the reason this exists.
//
// Cube core only: on A5 the vector core has no GM->L1 instruction (every
// copy_gm_to_cbuf form is rejected for dav-c310-vec) and there is no GM->L0
// path (load_gm_to_ca/cb are rejected for dav-c310-cube), so GM->L1->L0 is the
// only route to the mmad.
//
// A single descriptor moves ~32B per token row, nowhere near saturating MTE2,
// so the descriptor-emitting loop binds rather than the transfers. The loop is
// written to keep per-slot cube scalar work at its floor: each metadata word is
// read once and carried across the iteration boundary, the destination walks by
// a running pointer, and the nz parameter word is written once per call.

#if defined(__DAV_C310__)

#include "DMA/ND2NZ.h"
#include "DMA/SparseGatherNd2nz.h"
#include "Utils.h"

namespace {

// Geometry of the destination zN tile, which stores element (n, d) at
//   (d / c0) * nTileCeil * c0 + n * c0 + (d % c0),   c0 * sizeof(T) == 32B
// One token row is therefore exactly one 32B block, and a contiguous range of
// token rows is a contiguous block run inside every C0 column. Both the gather
// and the zero fill rely on that.
struct NzTile {
  int64_t c0Size;
  int64_t nTileCeil;
  int64_t d1;
};

template <typename T>
__aicore__ __attribute__((always_inline)) NzTile
makeNzTile(memref_t<__cbuf__ T, 4> *dst, int32_t dimSize, int64_t totalRows) {
  NzTile tile;
  tile.c0Size = L1_ALIGN_BYTES / sizeof(T);
  // Computed from the tile's own row count, not dst->strides: createLibCall
  // hands the callee a descriptor whose shape and strides were made dynamic,
  // and this buffer also goes through hivm.hir.pointer_cast, so the strides are
  // not dependable here.
  tile.nTileCeil = CEIL_FACTOR(totalRows, tile.c0Size);
  tile.d1 = CEIL_DIV(static_cast<int64_t>(dimSize), tile.c0Size);
  return tile;
}

// Per-call invariants of a gather descriptor.
template <typename T> struct RunEmitter {
  NzTile tile;
  int32_t dimSize;
  int64_t strideToken;
  uint8_t l2Ctl;
  // srcDValue and dstNzC0Stride reach the hardware through 16-bit fields. When
  // either overflows, a run has to go out one token row at a time, and those
  // descriptors need a different nz parameter word (dstNzNStride 0, not 1), so
  // the choice is made once and the whole call sticks to one of the two forms.
  bool packed;
};

// Same packing as copy_gm_to_cbuf_intrin_core: ndNum, dstNzNStride,
// dstNzC0Stride, dstNzMatrixStride. ndNum is 1, so the matrix-stride field is
// never consumed and is left zero.
template <typename T>
__aicore__ __attribute__((always_inline)) uint64_t
gatherNzParaWord(const NzTile &tile) {
  return (uint64_t)1 | ((uint64_t)1) << 16 | ((uint64_t)tile.nTileCeil) << 32 |
         ((uint64_t)0) << 48;
}

// The copy half of copy_gm_to_cbuf_intrin_core. The caller has already written
// the sticky nz parameter register for this tile.
template <typename T>
__aicore__ __attribute__((always_inline)) void
copyGmToCbuf(const RunEmitter<T> &emitter, __gm__ T *src, __cbuf__ T *dst,
             int32_t rows) {
  INTRINSIC(copy_gm_to_cbuf_multi_nd2nz, dst, src,
            /*uint8_t sid*/ 0,
            /*uint64_t loop1_src_stride*/
            emitter.strideToken * sizeof(T),
            /*uint8_t l2_cache_ctrl_mode*/ emitter.l2Ctl,
            static_cast<uint16_t>(rows), static_cast<uint16_t>(emitter.dimSize),
            /*srcNdMatrixStride bytes*/ 0,
            /*bool smallc0_en*/ false);
}

template <typename T>
__aicore__ __attribute__((always_inline)) RunEmitter<T>
makeRunEmitter(const NzTile &tile, int32_t dimSize, int64_t strideToken,
               uint8_t l2Ctl) {
  RunEmitter<T> emitter;
  emitter.tile = tile;
  emitter.dimSize = dimSize;
  emitter.strideToken = strideToken;
  emitter.l2Ctl = l2Ctl;
  emitter.packed =
      strideToken < MAX_LEN_UNIT16 && tile.nTileCeil < MAX_LEN_UNIT16;
  if (emitter.packed)
    INTRINSIC(set_mte2_nz_para, gatherNzParaWord<T>(tile));
  return emitter;
}

// Move `rows` GM token rows starting at `src` into the zN tile at `dst`.
template <typename T>
__aicore__ __attribute__((always_inline)) void
gatherRun(const RunEmitter<T> &emitter, __gm__ T *src, __cbuf__ T *dst,
          int32_t rows) {
  const NzTile &tile = emitter.tile;
  if (emitter.packed) {
    copyGmToCbuf<T>(emitter, src, dst, rows);
    return;
  }

  for (int32_t r = 0; r < rows; ++r) {
    copy_gm_to_cbuf_intrin_core<T>(nd2nz_intrin_args<T>{
        dst + r * tile.c0Size, src + r * emitter.strideToken,
        /*sid*/ 0, /*ndNum*/ 1, /*nValue*/ 1,
        static_cast<uint16_t>(emitter.dimSize),
        /*srcNdMatrixStride*/ 0, /*srcDValue*/ 0,
        static_cast<uint16_t>(tile.nTileCeil), /*dstNzNStride*/ 0,
        /*dstNzMatrixStride*/ 1, static_cast<uint16_t>(tile.c0Size),
        emitter.l2Ctl});
  }
}

// Zero `rows` token rows starting at zN token row `dst`: one block run per C0
// column, all reached from a single instruction through the repeat gap.
// create_cbuf_matrix is a cube instruction, the other reason this template
// belongs on the cube.
template <typename T>
__aicore__ __attribute__((always_inline)) void
zeroNzRows(__cbuf__ T *dst, const NzTile &tile, int64_t rows) {
  if (rows <= 0)
    return;
  // nTileCeil rounds the tile's row count up and `rows` never exceeds it, so
  // the gap is non-negative and both counts fit the instruction's 16-bit
  // fields. The guard states those invariants; it is not a fallback path.
  const int64_t dstGap = tile.nTileCeil - rows;
  if (dstGap < 0 || rows > 0xFFFF || tile.d1 > 0xFFFF)
    return;

  // repeatTimes in bits 0..15, blockNum in 16..31, dstGap in 32..47, all in 32B
  // units -- matching InitConstValueParams{repeatTimes, blockNum, dstGap}.
  const int64_t repeatBit =
      (rows << 16) | (dstGap << 32) | static_cast<uint64_t>(tile.d1);

  if constexpr (std::is_same<T, bfloat16_t>::value) {
    INTRINSIC(create_cbuf_matrix_bf16, dst, repeatBit, static_cast<T>(0));
  } else if constexpr (sizeof(T) == 2) {
    INTRINSIC(create_cbuf_matrix, dst, repeatBit, static_cast<half>(0));
  } else {
    static_assert(sizeof(T) == 4, "sparse_gather_load_to_L1buffer zero fill "
                                  "supports 2- and 4-byte types");
    INTRINSIC(create_cbuf_matrix, dst, repeatBit, static_cast<uint32_t>(0));
  }
}

} // namespace

/// gm token rows selected by a packed region list -> l1 zN tile.
template <typename T, typename ITYPE>
__aicore__ __attribute__((always_inline)) void
sparse_gather_load_to_L1buffer_core(
    memref_t<__cbuf__ T, 4> *dst, memref_t<__gm__ T, 1> *base,
    memref_t<__gm__ ITYPE, 1> *index, int32_t blockR, int32_t validRegionCount,
    int32_t regionSize, int32_t dimSize, int64_t strideToken,
    int32_t regionValidShift, int32_t regionIdMask, int32_t l2CacheCtrlMode) {
  if (blockR <= 0 || regionSize <= 0 || dimSize <= 0)
    return;

  __gm__ T *gm = base->aligned + base->offset;
  __cbuf__ T *l1 = dst->aligned + dst->offset;
  __gm__ ITYPE *meta = index->aligned + index->offset;
  const int64_t metaStride = index->strides[0];

  const NzTile tile =
      makeNzTile<T>(dst, dimSize, static_cast<int64_t>(blockR) * regionSize);
  const RunEmitter<T> emitter = makeRunEmitter<T>(
      tile, dimSize, strideToken, static_cast<uint8_t>(l2CacheCtrlMode));
  const int32_t liveSlots = MIN(blockR, MAX(validRegionCount, 0));

  // Hoisted strides: one slot advances the destination by slotElems, one region
  // id advances the source by regionElems.
  const int64_t slotElems = static_cast<int64_t>(regionSize) * tile.c0Size;
  const int64_t regionElems = static_cast<int64_t>(regionSize) * strideToken;

  // Walk the live slots, coalescing maximal runs of consecutive region ids into
  // one transfer. `word` holds the metadata for `slot` on entry to each
  // iteration; the run scan leaves the first word it rejected there, which is
  // the next slot's word, so a non-mergeable neighbour costs one GM read
  // rather than two.
  //
  // A padding slot is the negative sentinel and an invisible region has
  // valid == 0, so `valid <= 0` covers both: the valid field is the word's top
  // bits and the arithmetic shift keeps the sign.
  __cbuf__ T *slotDst = l1;
  __gm__ ITYPE *wordPtr = meta;
  int32_t slot = 0;
  int32_t word = liveSlots > 0 ? static_cast<int32_t>(*wordPtr) : 0;
  while (slot < liveSlots) {
    const int32_t valid = word >> regionValidShift;
    if (valid <= 0) {
      // Dead slots inside the live prefix; rare, since the selector pads at
      // the tail.
      int32_t dead = 1;
      wordPtr += metaStride;
      while (slot + dead < liveSlots) {
        word = static_cast<int32_t>(*wordPtr);
        if ((word >> regionValidShift) > 0)
          break;
        ++dead;
        wordPtr += metaStride;
      }
      zeroNzRows<T>(slotDst, tile, static_cast<int64_t>(dead) * regionSize);
      slot += dead;
      slotDst += dead * slotElems;
      continue;
    }

    const int32_t region = word & regionIdMask;
    int32_t runLen = 1;
    int32_t rows = valid;
    int32_t lastValid = valid;
    // `nextPtr` tracks meta[slot + runLen], so it is the next iteration's
    // `wordPtr` however this run ends; `haveNext` says whether `word` already
    // holds that slot because the scan read a neighbour and rejected it.
    __gm__ ITYPE *nextPtr = wordPtr + metaStride;
    bool haveNext = false;
    // A partial region can only be the last member of a run: its unused tail
    // would otherwise sit in the middle of the merged range, and a single
    // descriptor cannot skip it.
    while (lastValid == regionSize && slot + runLen < liveSlots) {
      word = static_cast<int32_t>(*nextPtr);
      haveNext = true;
      const int32_t nextValid = word >> regionValidShift;
      if (nextValid <= 0 || (word & regionIdMask) != region + runLen)
        break;
      lastValid = nextValid;
      rows += nextValid;
      ++runLen;
      nextPtr += metaStride;
      haveNext = false;
    }

    gatherRun<T>(emitter, gm + region * regionElems, slotDst, rows);
    // Tail of the run's last region: every earlier member was full, so the
    // whole gap is regionSize - lastValid.
    if (lastValid < regionSize)
      zeroNzRows<T>(slotDst + rows * tile.c0Size, tile, regionSize - lastValid);

    slot += runLen;
    slotDst += runLen * slotElems;
    wordPtr = nextPtr;
    if (!haveNext && slot < liveSlots)
      word = static_cast<int32_t>(*wordPtr);
  }

  // Slots past validRegionCount hold no metadata. Every row of `dst` is written
  // on every call, so zero them -- as one fill, not one per slot.
  zeroNzRows<T>(slotDst, tile,
                static_cast<int64_t>(blockR - liveSlots) * regionSize);
}

extern "C" {
REGISTER_SPARSE_GATHER_LOAD_TO_L1BUFFER(half, int32_t);
REGISTER_SPARSE_GATHER_LOAD_TO_L1BUFFER(bfloat16_t, int32_t);
REGISTER_SPARSE_GATHER_LOAD_TO_L1BUFFER(float, int32_t);
REGISTER_SPARSE_GATHER_LOAD_TO_L1BUFFER(half, int64_t);
REGISTER_SPARSE_GATHER_LOAD_TO_L1BUFFER(bfloat16_t, int64_t);
REGISTER_SPARSE_GATHER_LOAD_TO_L1BUFFER(float, int64_t);
}

#endif // defined(__DAV_C310__)
